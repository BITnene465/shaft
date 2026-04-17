from __future__ import annotations

from pathlib import Path
import socket
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

from shaft.webui.controller import ShaftSFTWebUIController, render_status_html
from shaft.webui.services import ShaftRunStore, ShaftSFTTrainService, ShaftWebUIConfigService
from shaft.webui.theme import static_dir, templates_dir


DEFAULT_SFT_CONFIG = "configs/train/train_sft_4b.yaml"


def _find_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _nav_items(active_key: str) -> list[dict[str, str | bool]]:
    entries = [
        ("sft", "SFT", "/sft"),
        ("dpo", "DPO", "/rlhf/dpo"),
        ("ppo", "PPO", "/rlhf/ppo"),
        ("grpo", "GRPO", "/rlhf/grpo"),
    ]
    return [
        {
            "key": key,
            "label": label,
            "href": href,
            "active": key == active_key,
        }
        for key, label, href in entries
    ]


def create_app(
    *,
    default_config_path: str = DEFAULT_SFT_CONFIG,
    config_service: ShaftWebUIConfigService | None = None,
    train_service: ShaftSFTTrainService | None = None,
) -> FastAPI:
    config_service = config_service or ShaftWebUIConfigService()
    train_service = train_service or ShaftSFTTrainService(run_store=ShaftRunStore())
    controller = ShaftSFTWebUIController(config_service=config_service, train_service=train_service)
    templates = Jinja2Templates(directory=str(templates_dir()))

    try:
        default_yaml_text = config_service.read_config_text(default_config_path)
        default_status = render_status_html(
            None,
            message="Loaded default SFT config. Keep overrides minimal and use YAML for uncommon fields.",
        )
    except Exception as exc:  # noqa: BLE001
        default_yaml_text = ""
        default_status = render_status_html(None, error=str(exc))

    app = FastAPI(title="Shaft Web UI")
    app.mount("/static", StaticFiles(directory=str(static_dir())), name="static")
    app.state.controller = controller
    app.state.default_config_path = default_config_path
    app.state.default_yaml_text = default_yaml_text
    app.state.default_status = default_status
    app.state.templates = templates

    @app.get("/")
    async def root():
        return RedirectResponse(url="/sft", status_code=307)

    @app.get("/sft")
    async def sft_page(request: Request):
        controller_: ShaftSFTWebUIController = request.app.state.controller
        initial_state = controller_.build_initial_view(
            request.app.state.default_config_path,
            request.app.state.default_yaml_text,
            request.app.state.default_status,
        )
        return request.app.state.templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "state": initial_state,
                "nav_items": _nav_items("sft"),
                "page_title": "SFT Research Console",
                "page_subtitle": "HF-first Multimodal Training Workspace",
            },
        )

    @app.get("/rlhf/dpo")
    async def dpo_page(request: Request):
        return request.app.state.templates.TemplateResponse(
            request=request,
            name="placeholder.html",
            context={
                "nav_items": _nav_items("dpo"),
                "page_title": "DPO Console",
                "page_subtitle": "RLHF workspace under staged rollout",
                "placeholder_title": "DPO configuration surface is not implemented yet.",
                "placeholder_body": (
                    "The navigation shell is now stable. DPO will get its own controller, "
                    "validation flow, and launch surface instead of being crammed into the SFT page."
                ),
            },
        )

    @app.get("/rlhf/ppo")
    async def ppo_page(request: Request):
        return request.app.state.templates.TemplateResponse(
            request=request,
            name="placeholder.html",
            context={
                "nav_items": _nav_items("ppo"),
                "page_title": "PPO Console",
                "page_subtitle": "RLHF workspace under staged rollout",
                "placeholder_title": "PPO configuration surface is not implemented yet.",
                "placeholder_body": (
                    "PPO needs a different runtime shape from SFT. The new navigation shell keeps "
                    "that complexity isolated instead of bloating a single page."
                ),
            },
        )

    @app.get("/rlhf/grpo")
    async def grpo_page(request: Request):
        return request.app.state.templates.TemplateResponse(
            request=request,
            name="placeholder.html",
            context={
                "nav_items": _nav_items("grpo"),
                "page_title": "GRPO Console",
                "page_subtitle": "RLHF workspace under staged rollout",
                "placeholder_title": "GRPO configuration surface is not implemented yet.",
                "placeholder_body": (
                    "GRPO will land as a dedicated page with task-specific controls. "
                    "This route exists now so the top-level Web UI architecture is no longer SFT-only."
                ),
            },
        )

    @app.get("/favicon.ico")
    async def favicon():
        return RedirectResponse(url="/static/favicon.svg", status_code=307)

    @app.post("/api/load-config")
    async def load_config(request: Request):
        payload = await request.json()
        return JSONResponse(controller.load_config(str(payload.get("config_path", ""))))

    @app.post("/api/validate")
    async def validate(request: Request):
        payload = await request.json()
        return JSONResponse(
            controller.validate(
                config_path=str(payload.get("config_path", "")),
                yaml_text=str(payload.get("yaml_text", "")),
                form_payload=dict(payload.get("form", {})),
            )
        )

    @app.post("/api/start")
    async def start(request: Request):
        payload = await request.json()
        return JSONResponse(
            controller.start(
                config_path=str(payload.get("config_path", "")),
                yaml_text=str(payload.get("yaml_text", "")),
                form_payload=dict(payload.get("form", {})),
            )
        )

    @app.post("/api/refresh")
    async def refresh(request: Request):
        payload = await request.json()
        return JSONResponse(controller.refresh(str(payload.get("current_run_id", ""))))

    @app.post("/api/stop")
    async def stop(request: Request):
        payload = await request.json()
        return JSONResponse(controller.stop(str(payload.get("current_run_id", ""))))

    @app.post("/api/load-run")
    async def load_run(request: Request):
        payload = await request.json()
        return JSONResponse(controller.load_run(str(payload.get("run_id", ""))))

    @app.post("/api/delete-run")
    async def delete_run(request: Request):
        payload = await request.json()
        return JSONResponse(
            controller.delete_run(
                str(payload.get("run_id", "")),
                str(payload.get("current_run_id", "")),
            )
        )

    return app


def main(
    *,
    host: str = "127.0.0.1",
    port: int | None = None,
    base_config_path: str = DEFAULT_SFT_CONFIG,
    share: bool = False,
) -> None:
    _ = share
    resolved_port = int(port) if port is not None else _find_free_port(host)
    app = create_app(default_config_path=base_config_path)
    uvicorn.run(app, host=host, port=resolved_port, log_level="info")
