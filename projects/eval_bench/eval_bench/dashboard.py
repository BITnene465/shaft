from __future__ import annotations

from contextlib import asynccontextmanager
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import socket
import threading
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from .artifacts import DEFAULT_STORE_ROOT, RunArtifacts, atomic_write_json, read_json
from .benchmark import create_benchmark_from_raw_data
from .comparison import compare_runs, filter_comparison_reports, list_comparison_reports
from .database import EvalBenchDatabase
from .evaluator import evaluate_run
from .job_spec import job_templates, preflight_job_payload, resolve_job_payload
from .job_lifecycle import job_holds_scheduler_resources
from .orchestrator import EvalBenchOrchestrator
from .prediction_import import import_predictions_for_benchmark
from .schema import utc_now_iso
from .services import EvalBenchServiceManager
from .store import EvalBenchStore
from .worker import EvalBenchWorker, terminate_runtime_process_group

IMAGE_PREVIEW_MAX_SIDE = 1800
IMAGE_PREVIEW_QUALITY = 82
IMAGE_TILE_SIZE = 512
IMAGE_TILE_QUALITY = 86


def _run_sample_detail_payload(run_id: str, detail: Any) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "sample": {
            **detail.sample.__dict__,
            **_sample_image_urls("runs", run_id, detail.sample.index),
        },
        "gt_instances": detail.gt_instances,
        "pred_instances": detail.pred_instances,
        "raw_payload": detail.raw_payload,
        "prediction_payload": detail.prediction_payload,
        "diagnostics": detail.diagnostics,
    }


def _sample_image_urls(scope: str, owner_id: str, sample_index: int) -> dict[str, Any]:
    image_url = f"/api/{scope}/{owner_id}/samples/{sample_index}/image"
    return {
        "image_url": image_url,
        "image_preview_url": f"{image_url}/preview?max_side={IMAGE_PREVIEW_MAX_SIDE}",
        "image_tile_url_template": f"{image_url}/tiles/{{level}}/{{x}}/{{y}}",
        "image_tile_size": IMAGE_TILE_SIZE,
    }


def project_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def frontend_dist_dir() -> Path:
    return project_dir() / "frontend" / "dist"


def static_dir() -> Path:
    return project_dir() / "static"


def _find_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _index_html(frontend_dist: Path) -> Path:
    return frontend_dist / "index.html"


def _spa_index_response(index: Path) -> FileResponse:
    return FileResponse(index, headers={"Cache-Control": "no-store"})


def _frontend_not_built_html() -> str:
    return """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Shaft Eval Bench</title>
    <style>
      body {
        margin: 0;
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        color: #171717;
        background: #f4f6f8;
      }
      main {
        max-width: 760px;
        margin: 12vh auto;
        padding: 0 24px;
      }
      code {
        background: #e9edf2;
        border: 1px solid #d7dee8;
        border-radius: 6px;
        padding: 2px 6px;
      }
    </style>
  </head>
  <body>
    <main>
      <h1>Shaft Eval Bench</h1>
      <p>The dashboard frontend has not been built yet.</p>
      <p>Run <code>npm install</code> and <code>npm run build</code> in
      <code>projects/eval_bench/frontend</code>, then restart this server.</p>
      <p>The API is available at <code>/api/state</code>.</p>
    </main>
  </body>
</html>
""".strip()


def _clamped_int(value: int, *, minimum: int, maximum: int) -> int:
    return min(maximum, max(minimum, int(value)))


def _filter_value(value: str | None) -> str:
    return str(value).strip() if value is not None else ""


def _cache_key(image_path: Path, *parts: object) -> str:
    stat = image_path.stat()
    payload = "|".join(
        [
            str(image_path.resolve()),
            str(stat.st_mtime_ns),
            str(stat.st_size),
            *(str(part) for part in parts),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _image_cache_path(store_root: Path, image_path: Path, *parts: object) -> Path:
    digest = _cache_key(image_path, *parts)
    return store_root / "cache" / "image_proxy" / digest[:2] / f"{digest}.jpg"


def _save_rgb_jpeg(image: Any, path: Path, *, quality: int) -> None:
    from PIL import Image

    if image.mode in {"RGBA", "LA"} or image.info.get("transparency") is not None:
        background = Image.new("RGB", image.size, (255, 255, 255))
        alpha = image.getchannel("A") if image.mode in {"RGBA", "LA"} else None
        background.paste(image.convert("RGBA"), mask=alpha)
        image = background
    elif image.mode != "RGB":
        image = image.convert("RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp.jpg")
    image.save(tmp_path, format="JPEG", quality=quality, optimize=True, progressive=True)
    tmp_path.replace(path)


def _image_preview_response(
    *,
    store_root: Path,
    image_path: Path,
    max_side: int,
    quality: int,
) -> FileResponse:
    from PIL import Image

    max_side = _clamped_int(max_side, minimum=256, maximum=4096)
    quality = _clamped_int(quality, minimum=50, maximum=95)
    cache_path = _image_cache_path(store_root, image_path, "preview", max_side, quality)
    if not cache_path.exists():
        with Image.open(image_path) as image:
            image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            _save_rgb_jpeg(image, cache_path, quality=quality)
    return FileResponse(
        cache_path,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=86400"},
    )


def _image_tile_response(
    *,
    store_root: Path,
    image_path: Path,
    level: int,
    tile_x: int,
    tile_y: int,
) -> FileResponse:
    from PIL import Image

    if level < 0 or tile_x < 0 or tile_y < 0:
        raise HTTPException(status_code=400, detail="tile level and coordinates must be non-negative.")
    quality = IMAGE_TILE_QUALITY
    cache_path = _image_cache_path(
        store_root,
        image_path,
        "tile",
        level,
        tile_x,
        tile_y,
        IMAGE_TILE_SIZE,
        quality,
    )
    if not cache_path.exists():
        scale = 2**level
        with Image.open(image_path) as image:
            width, height = image.size
            level_width = math.ceil(width / scale)
            level_height = math.ceil(height / scale)
            max_tile_x = max(0, math.ceil(level_width / IMAGE_TILE_SIZE) - 1)
            max_tile_y = max(0, math.ceil(level_height / IMAGE_TILE_SIZE) - 1)
            if tile_x > max_tile_x or tile_y > max_tile_y:
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"tile {level}/{tile_x}/{tile_y} outside pyramid bounds "
                        f"{max_tile_x + 1}x{max_tile_y + 1}."
                    ),
                )
            left = tile_x * IMAGE_TILE_SIZE * scale
            top = tile_y * IMAGE_TILE_SIZE * scale
            right = min(width, (tile_x + 1) * IMAGE_TILE_SIZE * scale)
            bottom = min(height, (tile_y + 1) * IMAGE_TILE_SIZE * scale)
            tile = image.crop((left, top, right, bottom))
            if scale > 1:
                tile_width = max(1, math.ceil((right - left) / scale))
                tile_height = max(1, math.ceil((bottom - top) / scale))
                tile = tile.resize((tile_width, tile_height), Image.Resampling.LANCZOS)
            _save_rgb_jpeg(tile, cache_path, quality=quality)
    return FileResponse(
        cache_path,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=86400"},
    )


def _prompt_template_map(database: EvalBenchDatabase) -> dict[str, dict[str, Any]]:
    return {
        record.prompt_id: record.to_dict()
        for record in database.list_prompt_templates(limit=1000)
    }


def _configure_backend_logging(store: EvalBenchStore) -> logging.Logger:
    log_path = store.layout.logs_dir / "backend.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("eval_bench")
    logger.setLevel(logging.INFO)
    resolved_path = str(log_path.resolve())
    stale_handlers: list[logging.Handler] = []
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == resolved_path:
            return logger
        if getattr(handler, "_eval_bench_backend_log", False):
            stale_handlers.append(handler)
    for handler in stale_handlers:
        logger.removeHandler(handler)
        handler.close()
    handler = logging.FileHandler(log_path, encoding="utf-8")
    setattr(handler, "_eval_bench_backend_log", True)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.propagate = True
    return logger


def _tail_text_lines(path: Path, *, max_lines: int) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    if max_lines <= 0:
        return lines
    return lines[-max_lines:]


def _process_job_in_background(store_root: Path, job_id: str) -> None:
    EvalBenchWorker(store_root).process_job(job_id)


def _job_runtime_log_path(store_root: Path, job: Any) -> Path:
    runtime_log_path = job.metadata.get("runtime_log_path") if isinstance(job.metadata, dict) else None
    if isinstance(runtime_log_path, str) and runtime_log_path.strip():
        return Path(runtime_log_path)
    run_id = str(job.payload.get("run_id") or job.job_id)
    return RunArtifacts(store_root, run_id).logs_dir / "runtime.log"


def _pid_exists(pid: Any) -> bool:
    try:
        parsed = int(pid)
    except (TypeError, ValueError):
        return False
    if parsed <= 0:
        return False
    try:
        os.kill(parsed, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _metadata_int(metadata: dict[str, Any], key: str) -> int | None:
    try:
        parsed = int(metadata.get(key))
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _is_live_running_job(job: Any) -> bool:
    metadata = job.metadata if isinstance(job.metadata, dict) else {}
    return any(
        _pid_exists(metadata.get(key))
        for key in ("dashboard_worker_pid", "runtime_pid")
    )


def create_app(
    *,
    store_root: str | Path = DEFAULT_STORE_ROOT,
    frontend_dist: str | Path | None = None,
    enable_orchestrator: bool = False,
) -> FastAPI:
    store = EvalBenchStore(store_root)
    database = EvalBenchDatabase(store_root)
    service_manager = EvalBenchServiceManager(store_root)
    worker = EvalBenchWorker(store_root)
    orchestrator = EvalBenchOrchestrator.from_env(store_root) if enable_orchestrator else None
    dist = Path(frontend_dist) if frontend_dist is not None else frontend_dist_dir()
    logger = _configure_backend_logging(store)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.eval_bench_orchestrator is not None:
            app.state.eval_bench_orchestrator.start()
        try:
            yield
        finally:
            if app.state.eval_bench_orchestrator is not None:
                app.state.eval_bench_orchestrator.stop()

    app = FastAPI(title="Shaft Eval Bench", lifespan=lifespan)
    app.state.eval_bench_store = store
    app.state.eval_bench_database = database
    app.state.eval_bench_services = service_manager
    app.state.eval_bench_worker = worker
    app.state.eval_bench_orchestrator = orchestrator
    app.state.frontend_dist = dist

    app.mount("/static", StaticFiles(directory=str(static_dir()), check_dir=False), name="static")
    app.mount("/assets", StaticFiles(directory=str(dist / "assets"), check_dir=False), name="assets")
    app.mount("/icons", StaticFiles(directory=str(dist / "icons"), check_dir=False), name="icons")

    @app.middleware("http")
    async def request_logging_middleware(request: Request, call_next):
        request_id = uuid4().hex[:10]
        request.state.request_id = request_id
        try:
            response = await call_next(request)
        except Exception:
            logger.exception(
                "request failed request_id=%s method=%s path=%s",
                request_id,
                request.method,
                request.url.path,
            )
            raise
        if response.status_code >= 400:
            logger.warning(
                "request returned error request_id=%s method=%s path=%s status=%s",
                request_id,
                request.method,
                request.url.path,
                response.status_code,
            )
        response.headers["X-Eval-Bench-Request-Id"] = request_id
        return response

    @app.exception_handler(HTTPException)
    async def logged_http_exception_handler(request: Request, exc: HTTPException):
        request_id = getattr(request.state, "request_id", uuid4().hex[:10])
        if exc.status_code >= 500:
            logger.error(
                "http exception request_id=%s method=%s path=%s status=%s detail=%s",
                request_id,
                request.method,
                request.url.path,
                exc.status_code,
                exc.detail,
            )
        elif exc.status_code >= 400:
            logger.warning(
                "http exception request_id=%s method=%s path=%s status=%s detail=%s",
                request_id,
                request.method,
                request.url.path,
                exc.status_code,
                exc.detail,
            )
        response = await http_exception_handler(request, exc)
        response.headers["X-Eval-Bench-Request-Id"] = request_id
        return response

    @app.get("/api/health")
    async def health(request: Request):
        return JSONResponse(
            {
                "ok": True,
                "store_root": str(request.app.state.eval_bench_store.layout.root),
                "frontend_built": _index_html(request.app.state.frontend_dist).exists(),
                "scheduler_enabled": request.app.state.eval_bench_orchestrator is not None,
            }
        )

    @app.get("/api/scheduler/status")
    async def scheduler_status(request: Request):
        orchestrator = request.app.state.eval_bench_orchestrator
        if orchestrator is None:
            return JSONResponse({"enabled": False})
        return JSONResponse(orchestrator.status())

    @app.get("/api/logs/backend")
    async def backend_logs(request: Request, max_lines: int = 200):
        log_path = request.app.state.eval_bench_store.layout.logs_dir / "backend.log"
        line_limit = 0 if max_lines <= 0 else min(max_lines, 2000)
        lines = _tail_text_lines(log_path, max_lines=line_limit)
        return JSONResponse(
            {
                "log_path": str(log_path),
                "lines": lines,
                "text": "".join(lines),
            }
        )

    @app.get("/api/benchmarks")
    async def benchmarks(
        request: Request,
        offset: int = 0,
        limit: int = 100,
        task: str | None = None,
        layer: str | None = None,
        split: str | None = None,
        query: str | None = None,
    ):
        page = request.app.state.eval_bench_store.benchmark_page(
            offset=max(0, offset),
            limit=_clamped_int(limit, minimum=1, maximum=500),
            task=task,
            layer=layer,
            split=split,
            query=query,
        )
        return JSONResponse(page.to_dict())

    @app.post("/api/benchmarks")
    async def create_benchmark(request: Request):
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="benchmark payload must be a JSON object")
        required_fields = ("benchmark_id", "source_root", "source_manifest", "split")
        missing_fields = [field for field in required_fields if not str(payload.get(field) or "").strip()]
        if missing_fields:
            raise HTTPException(
                status_code=400,
                detail=f"missing required benchmark fields: {', '.join(missing_fields)}",
            )
        tasks = payload.get("tasks")
        if not isinstance(tasks, list) or not tasks:
            raise HTTPException(status_code=400, detail="tasks must be a non-empty list")
        layers = payload.get("layers") or []
        if not isinstance(layers, list):
            raise HTTPException(status_code=400, detail="layers must be a list when provided")
        try:
            manifest = create_benchmark_from_raw_data(
                store_root=request.app.state.eval_bench_store.layout.root,
                benchmark_id=str(payload["benchmark_id"]).strip(),
                tasks=[str(item) for item in tasks],  # type: ignore[list-item]
                source_root=str(payload["source_root"]).strip(),
                source_manifest=str(payload["source_manifest"]).strip(),
                split=str(payload["split"]).strip(),
                layers=[str(item) for item in layers],
                overwrite=bool(payload.get("overwrite", False)),
            )
        except FileExistsError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JSONResponse(manifest.to_dict(), status_code=201)

    @app.get("/api/benchmarks/{benchmark_id}/samples")
    async def benchmark_samples(
        benchmark_id: str,
        request: Request,
        offset: int = 0,
        limit: int = 80,
        label: str | None = None,
    ):
        try:
            page = request.app.state.eval_bench_store.benchmark_sample_page(
                benchmark_id,
                offset=offset,
                limit=min(max(1, limit), 500),
                label=label,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return JSONResponse(
            {
                "benchmark_id": benchmark_id,
                "offset": page.offset,
                "limit": page.limit,
                "total": page.total,
                "labels": page.labels,
                "samples": [
                    {
                        **sample.__dict__,
                        **_sample_image_urls("benchmarks", benchmark_id, sample.index),
                    }
                    for sample in page.samples
                ],
            }
        )

    @app.get("/api/benchmarks/{benchmark_id}/samples/{sample_index}")
    async def benchmark_sample_detail(benchmark_id: str, sample_index: int, request: Request):
        try:
            detail = request.app.state.eval_bench_store.benchmark_sample_detail(
                benchmark_id,
                sample_index=sample_index,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except IndexError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return JSONResponse(
            {
                "benchmark_id": benchmark_id,
                "sample": {
                    **detail.sample.__dict__,
                    **_sample_image_urls("benchmarks", benchmark_id, detail.sample.index),
                },
                "gt_instances": detail.gt_instances,
                "raw_payload": detail.raw_payload,
            }
        )

    @app.get("/api/benchmarks/{benchmark_id}/samples/{sample_index}/image")
    async def benchmark_sample_image(benchmark_id: str, sample_index: int, request: Request):
        try:
            image_path = request.app.state.eval_bench_store.benchmark_sample_image_path(
                benchmark_id,
                sample_index=sample_index,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except IndexError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if not image_path.exists():
            raise HTTPException(status_code=404, detail=f"sample image does not exist: {image_path}")
        return FileResponse(image_path)

    @app.get("/api/benchmarks/{benchmark_id}/samples/{sample_index}/image/preview")
    async def benchmark_sample_image_preview(
        benchmark_id: str,
        sample_index: int,
        request: Request,
        max_side: int = IMAGE_PREVIEW_MAX_SIDE,
        quality: int = IMAGE_PREVIEW_QUALITY,
    ):
        try:
            image_path = request.app.state.eval_bench_store.benchmark_sample_image_path(
                benchmark_id,
                sample_index=sample_index,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except IndexError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if not image_path.exists():
            raise HTTPException(status_code=404, detail=f"sample image does not exist: {image_path}")
        return _image_preview_response(
            store_root=request.app.state.eval_bench_store.layout.root,
            image_path=image_path,
            max_side=max_side,
            quality=quality,
        )

    @app.get("/api/benchmarks/{benchmark_id}/samples/{sample_index}/image/tiles/{level}/{tile_x}/{tile_y}")
    async def benchmark_sample_image_tile(
        benchmark_id: str,
        sample_index: int,
        level: int,
        tile_x: int,
        tile_y: int,
        request: Request,
    ):
        try:
            image_path = request.app.state.eval_bench_store.benchmark_sample_image_path(
                benchmark_id,
                sample_index=sample_index,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except IndexError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if not image_path.exists():
            raise HTTPException(status_code=404, detail=f"sample image does not exist: {image_path}")
        return _image_tile_response(
            store_root=request.app.state.eval_bench_store.layout.root,
            image_path=image_path,
            level=level,
            tile_x=tile_x,
            tile_y=tile_y,
        )

    @app.get("/api/settings/preview-sample")
    async def settings_preview_sample(request: Request, benchmark_id: str | None = None):
        try:
            resolved_benchmark_id, detail = request.app.state.eval_bench_store.benchmark_preview_sample(
                benchmark_id=benchmark_id,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return JSONResponse(
            {
                "benchmark_id": resolved_benchmark_id,
                "sample": {
                    **detail.sample.__dict__,
                    **_sample_image_urls("benchmarks", resolved_benchmark_id, detail.sample.index),
                },
                "gt_instances": detail.gt_instances,
                "raw_payload": detail.raw_payload,
            }
        )

    @app.get("/api/runs")
    async def runs(
        request: Request,
        offset: int = 0,
        limit: int = 100,
        task: str | None = None,
        benchmark_id: str | None = None,
        status: str | None = None,
        label: str | None = None,
        model_id: str | None = None,
        prompt_id: str | None = None,
        metric_profile: str | None = None,
        query: str | None = None,
    ):
        page = request.app.state.eval_bench_store.run_page(
            offset=max(0, offset),
            limit=_clamped_int(limit, minimum=1, maximum=500),
            task=task,
            benchmark_id=benchmark_id,
            status=status,
            label=label,
            model_id=model_id,
            prompt_id=prompt_id,
            metric_profile=metric_profile,
            query=query,
        )
        return JSONResponse(page.to_dict())

    @app.get("/api/rank-board")
    async def rank_board(
        request: Request,
        offset: int = 0,
        limit: int = 100,
        task: str | None = None,
        benchmark_id: str | None = None,
        status: str | None = None,
        label: str | None = None,
        model_id: str | None = None,
        prompt_id: str | None = None,
        metric_profile: str | None = None,
        min_score: float | None = None,
        sort_by: str = "f1_iou50",
        sort_order: str = "desc",
        query: str | None = None,
        rank_scheme: str | None = None,
    ):
        try:
            board = request.app.state.eval_bench_store.rank_board(
                offset=max(0, offset),
                limit=min(max(1, limit), 500),
                task=task,
                benchmark_id=benchmark_id,
                status=status,
                label=label,
                model_id=model_id,
                prompt_id=prompt_id,
                metric_profile=metric_profile,
                min_score=min_score,
                sort_by=sort_by,
                sort_order=sort_order,
                query=query,
                rank_scheme=rank_scheme,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JSONResponse(board.to_dict())

    @app.get("/api/runs/{run_id}/note")
    async def run_note(run_id: str, request: Request):
        try:
            note = request.app.state.eval_bench_store.run_note(run_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return JSONResponse(note.to_dict())

    @app.patch("/api/runs/{run_id}/note")
    async def update_run_note(run_id: str, request: Request):
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="run note payload must be a JSON object")
        note = payload.get("note")
        if not isinstance(note, str):
            raise HTTPException(status_code=400, detail="note must be a string")
        try:
            updated = request.app.state.eval_bench_store.update_run_note(run_id, note)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JSONResponse(updated.to_dict())

    @app.post("/api/runs/import-predictions")
    async def import_predictions(request: Request):
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="import payload must be a JSON object")
        required_fields = ("run_id", "benchmark_id", "prediction_root", "task", "model_id")
        missing_fields = [field for field in required_fields if not str(payload.get(field) or "").strip()]
        if missing_fields:
            raise HTTPException(
                status_code=400,
                detail=f"missing required import fields: {', '.join(missing_fields)}",
            )
        try:
            result = import_predictions_for_benchmark(
                store_root=request.app.state.eval_bench_store.layout.root,
                run_id=str(payload["run_id"]).strip(),
                benchmark_id=str(payload["benchmark_id"]).strip(),
                prediction_root=str(payload["prediction_root"]).strip(),
                task=str(payload["task"]),  # type: ignore[arg-type]
                model_id=str(payload["model_id"]).strip(),
                model_path=str(payload.get("model_path") or "imported"),
                prompt_id=str(payload.get("prompt_id") or "imported"),
                spec_id=str(payload.get("spec_id") or "").strip() or None,
                target_labels=payload.get("target_labels"),
                strict=bool(payload.get("strict", False)),
                overwrite=bool(payload.get("overwrite", False)),
                evaluate=bool(payload.get("evaluate", True)),
            )
        except FileExistsError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JSONResponse(result.to_dict(), status_code=201)

    @app.post("/api/runs/{run_id}/evaluate")
    async def evaluate_run_endpoint(run_id: str, request: Request):
        try:
            path = evaluate_run(
                store_root=request.app.state.eval_bench_store.layout.root,
                run_id=run_id,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return JSONResponse({"run_id": run_id, "report_path": str(path)})

    @app.post("/api/runs/{run_id}/archive")
    async def archive_run_endpoint(run_id: str, request: Request):
        run_dir = request.app.state.eval_bench_store.layout.runs_dir / run_id
        manifest_path = run_dir / "run.json"
        if not manifest_path.exists():
            raise HTTPException(status_code=404, detail=f"run does not exist: {run_id}")
        payload = read_json(manifest_path)
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail=f"invalid run manifest: {manifest_path}")
        payload["status"] = "archived"
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        metadata["archived_at"] = utc_now_iso()
        payload["metadata"] = metadata
        atomic_write_json(manifest_path, payload)
        return JSONResponse({"run_id": run_id, "status": "archived"})

    @app.delete("/api/runs/{run_id}")
    async def delete_run_endpoint(run_id: str, request: Request):
        run_dir = request.app.state.eval_bench_store.layout.runs_dir / run_id
        if not run_dir.exists():
            raise HTTPException(status_code=404, detail=f"run does not exist: {run_id}")
        trash_path = request.app.state.eval_bench_store.layout.move_to_trash(
            run_dir,
            category="runs",
        )
        return JSONResponse(
            {
                "run_id": run_id,
                "deleted": True,
                "trash_path": str(trash_path) if trash_path is not None else None,
            }
        )

    @app.get("/api/runs/{run_id}/report")
    async def run_report(run_id: str, request: Request):
        report_path = (
            request.app.state.eval_bench_store.layout.runs_dir
            / run_id
            / "reports"
            / "metrics.json"
        )
        if not report_path.exists():
            raise HTTPException(status_code=404, detail="run report does not exist")
        return JSONResponse(json.loads(report_path.read_text(encoding="utf-8")))

    @app.get("/api/runs/{run_id}/samples")
    async def run_samples(
        run_id: str,
        request: Request,
        offset: int = 0,
        limit: int = 80,
        label: str | None = None,
        error_filter: str = "all",
    ):
        try:
            page = request.app.state.eval_bench_store.run_sample_page(
                run_id,
                offset=offset,
                limit=min(max(1, limit), 500),
                label=label,
                error_filter=error_filter,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return JSONResponse(
            {
                "run_id": run_id,
                "offset": page.offset,
                "limit": page.limit,
                "total": page.total,
                "labels": page.labels,
                "samples": [
                    {
                        **sample.__dict__,
                        **_sample_image_urls("runs", run_id, sample.index),
                    }
                    for sample in page.samples
                ],
            }
        )

    @app.get("/api/runs/{run_id}/samples/{sample_index}")
    async def run_sample_detail(run_id: str, sample_index: int, request: Request):
        try:
            detail = request.app.state.eval_bench_store.run_sample_detail(
                run_id,
                sample_index=sample_index,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except IndexError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return JSONResponse(_run_sample_detail_payload(run_id, detail))

    @app.get("/api/runs/{run_id}/samples/{sample_index}/image")
    async def run_sample_image(run_id: str, sample_index: int, request: Request):
        try:
            image_path = request.app.state.eval_bench_store.run_sample_image_path(
                run_id,
                sample_index=sample_index,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except IndexError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if not image_path.exists():
            raise HTTPException(status_code=404, detail=f"sample image does not exist: {image_path}")
        return FileResponse(image_path)

    @app.get("/api/runs/{run_id}/samples/{sample_index}/image/preview")
    async def run_sample_image_preview(
        run_id: str,
        sample_index: int,
        request: Request,
        max_side: int = IMAGE_PREVIEW_MAX_SIDE,
        quality: int = IMAGE_PREVIEW_QUALITY,
    ):
        try:
            image_path = request.app.state.eval_bench_store.run_sample_image_path(
                run_id,
                sample_index=sample_index,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except IndexError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if not image_path.exists():
            raise HTTPException(status_code=404, detail=f"sample image does not exist: {image_path}")
        return _image_preview_response(
            store_root=request.app.state.eval_bench_store.layout.root,
            image_path=image_path,
            max_side=max_side,
            quality=quality,
        )

    @app.get("/api/runs/{run_id}/samples/{sample_index}/image/tiles/{level}/{tile_x}/{tile_y}")
    async def run_sample_image_tile(
        run_id: str,
        sample_index: int,
        level: int,
        tile_x: int,
        tile_y: int,
        request: Request,
    ):
        try:
            image_path = request.app.state.eval_bench_store.run_sample_image_path(
                run_id,
                sample_index=sample_index,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except IndexError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if not image_path.exists():
            raise HTTPException(status_code=404, detail=f"sample image does not exist: {image_path}")
        return _image_tile_response(
            store_root=request.app.state.eval_bench_store.layout.root,
            image_path=image_path,
            level=level,
            tile_x=tile_x,
            tile_y=tile_y,
        )

    @app.get("/api/jobs")
    async def jobs(
        request: Request,
        offset: int = 0,
        limit: int = 100,
        kind: str | None = None,
        status: str | None = None,
        query: str | None = None,
    ):
        page = request.app.state.eval_bench_database.job_page(
            offset=max(0, offset),
            limit=_clamped_int(limit, minimum=1, maximum=500),
            kind=kind,
            status=status,
            query=query,
        )
        return JSONResponse(page.to_dict())

    @app.get("/api/jobs/{job_id}/logs")
    async def job_logs(job_id: str, request: Request, max_lines: int = 200):
        record = request.app.state.eval_bench_database.get_job(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"job does not exist: {job_id}")
        log_path = _job_runtime_log_path(
            request.app.state.eval_bench_store.layout.root,
            record,
        )
        line_limit = 0 if max_lines <= 0 else min(max_lines, 2000)
        lines = _tail_text_lines(log_path, max_lines=line_limit)
        return JSONResponse(
            {
                "job_id": job_id,
                "log_path": str(log_path) if log_path.exists() else None,
                "lines": lines,
                "text": "".join(lines),
            }
        )

    @app.get("/api/job-templates")
    async def templates(request: Request):
        return JSONResponse({"templates": job_templates()})

    @app.get("/api/prompt-templates")
    async def prompt_templates(request: Request, task: str | None = None):
        records = request.app.state.eval_bench_database.list_prompt_templates(task=task)
        return JSONResponse(
            {
                "templates": [record.to_dict() for record in records],
                "by_id": {record.prompt_id: record.to_dict() for record in records},
            }
        )

    @app.post("/api/prompt-templates")
    async def upsert_prompt_template(request: Request):
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="prompt template payload must be a JSON object")
        try:
            record = request.app.state.eval_bench_database.upsert_prompt_template(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JSONResponse(record.to_dict(), status_code=201)

    @app.delete("/api/prompt-templates/{prompt_id}")
    async def delete_prompt_template(prompt_id: str, request: Request):
        try:
            record = request.app.state.eval_bench_database.delete_prompt_template(prompt_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return JSONResponse({"prompt_id": record.prompt_id, "deleted": True})

    @app.post("/api/jobs/preflight")
    async def preflight_job(request: Request):
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="job payload must be a JSON object")
        result = preflight_job_payload(
            payload,
            store_root=request.app.state.eval_bench_store.layout.root,
            prompt_templates=_prompt_template_map(request.app.state.eval_bench_database),
        )
        return JSONResponse(result)

    @app.post("/api/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str, request: Request):
        database = request.app.state.eval_bench_database
        try:
            record = database.cancel_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        runtime_pid = _metadata_int(record.metadata, "runtime_pid")
        if runtime_pid is not None:
            terminated = terminate_runtime_process_group(runtime_pid)
            record = database.update_job(
                job_id,
                status="cancelled",
                metadata_update={
                    "runtime_termination_requested_at": utc_now_iso(),
                    "runtime_terminated": terminated,
                    "runtime_terminated_pid": runtime_pid,
                },
            )
        return JSONResponse(record.to_dict())

    @app.delete("/api/jobs/{job_id}")
    async def delete_job(job_id: str, request: Request):
        try:
            record = request.app.state.eval_bench_database.delete_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        trash_path = (
            request.app.state.eval_bench_store.layout.trash_dir
            / "jobs"
            / f"{record.job_id}.json"
        )
        atomic_write_json(trash_path, record.to_dict())
        return JSONResponse(
            {
                "job_id": job_id,
                "deleted": True,
                "trash_path": str(trash_path),
            }
        )

    @app.get("/api/services")
    async def services(
        request: Request,
        offset: int = 0,
        limit: int = 100,
        kind: str | None = None,
        status: str | None = None,
        query: str | None = None,
    ):
        page = request.app.state.eval_bench_services.service_page(
            offset=max(0, offset),
            limit=_clamped_int(limit, minimum=1, maximum=500),
            kind=kind,
            status=status,
            query=query,
        )
        return JSONResponse(page.to_dict())

    @app.post("/api/services")
    async def create_service(request: Request):
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="service payload must be a JSON object")
        try:
            record = request.app.state.eval_bench_services.register_service(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JSONResponse(record.to_dict(), status_code=201)

    @app.get("/api/services/{service_id}/command")
    async def service_command(service_id: str, request: Request):
        try:
            command = request.app.state.eval_bench_services.launch_command(service_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JSONResponse({"service_id": service_id, "command": command})

    @app.post("/api/services/{service_id}/start")
    async def start_service(service_id: str, request: Request):
        try:
            record = request.app.state.eval_bench_services.start_service(service_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JSONResponse(record.to_dict())

    @app.post("/api/services/{service_id}/health")
    async def service_health(service_id: str, request: Request, timeout_s: float = 2.0):
        try:
            record = request.app.state.eval_bench_services.check_service_health(
                service_id,
                timeout_s=timeout_s,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return JSONResponse(record.to_dict())

    @app.get("/api/services/{service_id}/logs")
    async def service_logs(service_id: str, request: Request, max_lines: int = 200):
        try:
            payload = request.app.state.eval_bench_services.service_log(
                service_id,
                max_lines=min(max(1, max_lines), 2000),
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return JSONResponse(payload)

    @app.post("/api/services/{service_id}/stop")
    async def stop_service(service_id: str, request: Request):
        try:
            record = request.app.state.eval_bench_services.stop_service(service_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return JSONResponse(record.to_dict())

    @app.delete("/api/services/{service_id}")
    async def delete_service(service_id: str, request: Request):
        try:
            payload = request.app.state.eval_bench_services.delete_service(service_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return JSONResponse(payload)

    @app.get("/api/comparisons")
    async def comparison_report(
        request: Request,
        baseline_run_id: str | None = None,
        candidate_run_id: str | None = None,
        task: str | None = None,
        label: str | None = None,
        query: str | None = None,
        offset: int = 0,
        limit: int = 100,
    ):
        if baseline_run_id is None and candidate_run_id is None:
            filters = {
                "task": _filter_value(task),
                "label": _filter_value(label),
                "query": (query or "").strip(),
            }
            reports = filter_comparison_reports(
                list_comparison_reports(
                    store_root=request.app.state.eval_bench_store.layout.root,
                ),
                task=filters["task"],
                label=filters["label"],
                query=filters["query"],
            )
            start = max(0, int(offset))
            page_limit = _clamped_int(limit, minimum=1, maximum=500)
            return JSONResponse(
                {
                    "comparisons": reports[start : start + page_limit],
                    "total": len(reports),
                    "offset": start,
                    "limit": page_limit,
                    "filters": filters,
                }
            )
        if baseline_run_id is None or candidate_run_id is None:
            raise HTTPException(
                status_code=400,
                detail="baseline_run_id and candidate_run_id must be provided together",
            )
        try:
            path = compare_runs(
                store_root=request.app.state.eval_bench_store.layout.root,
                baseline_run_id=baseline_run_id,
                candidate_run_id=candidate_run_id,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return JSONResponse(json.loads(path.read_text(encoding="utf-8")))

    @app.get("/api/comparisons/sample")
    async def comparison_sample_detail(
        request: Request,
        baseline_run_id: str,
        candidate_run_id: str,
        sample_index: int,
    ):
        try:
            baseline = request.app.state.eval_bench_store.run_sample_detail(
                baseline_run_id,
                sample_index=sample_index,
            )
            candidate = request.app.state.eval_bench_store.run_sample_detail(
                candidate_run_id,
                sample_index=sample_index,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except IndexError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return JSONResponse(
            {
                "baseline_run_id": baseline_run_id,
                "candidate_run_id": candidate_run_id,
                "sample_index": sample_index,
                "baseline": _run_sample_detail_payload(baseline_run_id, baseline),
                "candidate": _run_sample_detail_payload(candidate_run_id, candidate),
            }
        )

    @app.post("/api/jobs")
    async def create_job(request: Request):
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="job payload must be a JSON object")
        prompt_templates = _prompt_template_map(request.app.state.eval_bench_database)
        try:
            resolved = resolve_job_payload(payload, prompt_templates=prompt_templates)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        preflight = preflight_job_payload(
            payload,
            store_root=request.app.state.eval_bench_store.layout.root,
            prompt_templates=prompt_templates,
        )
        if not preflight.get("ok"):
            raise HTTPException(status_code=400, detail=preflight)
        kind = "eval" if resolved.kind == "eval_job" else "preannotate"
        record = request.app.state.eval_bench_database.create_job(
            kind=kind,
            payload={
                **dict(preflight.get("resolved_payload") or {}),
                "manifest": preflight.get("resolved_manifest"),
            },
        )
        return JSONResponse(record.to_dict(), status_code=201)

    @app.post("/api/jobs/process-next")
    async def process_next_job(request: Request):
        orchestrator = request.app.state.eval_bench_orchestrator
        if orchestrator is not None:
            launched = orchestrator.schedule_once()
            return JSONResponse(
                {
                    "jobs": [record.to_dict() for record in launched],
                    "job": launched[0].to_dict() if launched else None,
                    "processed": bool(launched),
                    "background": True,
                    "scheduler": orchestrator.status(),
                    "message": (
                        f"scheduled {len(launched)} job(s)"
                        if launched
                        else "no schedulable queued eval job"
                    ),
                }
            )
        database = request.app.state.eval_bench_database
        running = next(
            (
                record
                for record in database.list_jobs(limit=200)
                if job_holds_scheduler_resources(record) and _is_live_running_job(record)
            ),
            None,
        )
        if running is not None:
            return JSONResponse(
                {
                    "job": running.to_dict(),
                    "processed": False,
                    "background": True,
                    "message": "a job is already running",
                }
            )
        record = request.app.state.eval_bench_worker.claim_next(kind="eval")
        if record is None:
            return JSONResponse({"job": None, "processed": False})
        record = database.update_job(
            record.job_id,
            status="running",
            metadata_update={
                "dashboard_worker_pid": os.getpid(),
                "dashboard_worker_started_at": utc_now_iso(),
                "progress_phase": "worker_starting",
                "progress_message": "Background worker thread is starting.",
                "progress_updated_at": utc_now_iso(),
            },
        )
        thread = threading.Thread(
            target=_process_job_in_background,
            args=(request.app.state.eval_bench_store.layout.root, record.job_id),
            name=f"eval-bench-job-{record.job_id}",
            daemon=True,
        )
        thread.start()
        return JSONResponse({"job": record.to_dict(), "processed": True, "background": True})

    @app.get("/api/state")
    async def state(request: Request):
        return JSONResponse(request.app.state.eval_bench_store.state().to_dict())

    @app.get("/", include_in_schema=False)
    async def dashboard(request: Request):
        index = _index_html(request.app.state.frontend_dist)
        if index.exists():
            return _spa_index_response(index)
        return HTMLResponse(_frontend_not_built_html(), status_code=200)

    @app.get("/logo.png", include_in_schema=False)
    async def frontend_logo(request: Request):
        logo = request.app.state.frontend_dist / "logo.png"
        if logo.exists():
            return FileResponse(
                logo,
                media_type="image/png",
                headers={"Cache-Control": "public, max-age=86400"},
            )
        raise HTTPException(status_code=404)

    @app.get("/{path:path}", include_in_schema=False)
    async def spa_fallback(path: str, request: Request):
        if path.startswith("api/"):
            raise HTTPException(status_code=404)
        index = _index_html(request.app.state.frontend_dist)
        if index.exists():
            return _spa_index_response(index)
        return HTMLResponse(_frontend_not_built_html(), status_code=200)

    return app


def main(
    *,
    host: str = "127.0.0.1",
    port: int | None = None,
    store_root: str | Path = DEFAULT_STORE_ROOT,
    frontend_dist: str | Path | None = None,
) -> None:
    resolved_port = int(port) if port is not None else _find_free_port(host)
    app = create_app(
        store_root=store_root,
        frontend_dist=frontend_dist,
        enable_orchestrator=True,
    )
    uvicorn.run(app, host=host, port=resolved_port, log_level="info")
