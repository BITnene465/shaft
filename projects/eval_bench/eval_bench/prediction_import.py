from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import (
    DEFAULT_STORE_ROOT,
    BenchmarkArtifacts,
    RunArtifacts,
    atomic_write_json,
    read_json,
)
from .evaluator import evaluate_run
from .label_policy import resolve_target_label_policy
from .sample_paths import sample_image_path
from .schema import (
    BenchmarkRef,
    EvalRunManifest,
    EvalSpec,
    InferenceParams,
    ModelRef,
    PredictionDocument,
    PredictionInstance,
    PromptRef,
    TaskKind,
)


@dataclass(frozen=True)
class ImportedPredictionRun:
    run_id: str
    run_manifest_path: Path
    report_path: Path | None
    imported_predictions: int
    missing_predictions: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_manifest_path": str(self.run_manifest_path),
            "report_path": str(self.report_path) if self.report_path else None,
            "imported_predictions": self.imported_predictions,
            "missing_predictions": self.missing_predictions,
            "missing_prediction_count": len(self.missing_predictions),
        }


def import_predictions_for_benchmark(
    *,
    store_root: str | Path = DEFAULT_STORE_ROOT,
    run_id: str,
    benchmark_id: str,
    prediction_root: str | Path,
    task: TaskKind,
    model_id: str,
    model_path: str = "imported",
    prompt_id: str = "imported",
    spec_id: str | None = None,
    target_labels: list[str] | str | None = None,
    strict: bool = False,
    overwrite: bool = False,
    evaluate: bool = True,
) -> ImportedPredictionRun:
    benchmark_artifacts = BenchmarkArtifacts(store_root, benchmark_id)
    benchmark_manifest = read_json(benchmark_artifacts.manifest_path)
    if not isinstance(benchmark_manifest, dict):
        raise ValueError(f"benchmark manifest must be a JSON object: {benchmark_artifacts.manifest_path}")
    run_artifacts = RunArtifacts(store_root, run_id)
    if run_artifacts.run_dir.exists():
        if not overwrite:
            raise FileExistsError(f"run already exists, pass --overwrite to replace it: {run_id}")
        shutil.rmtree(run_artifacts.run_dir)

    prediction_root_path = Path(prediction_root)
    source_index = _PredictionSourceIndex(prediction_root_path)
    split_path = Path(str(benchmark_manifest.get("manifest_path") or ""))
    benchmark_root = Path(str(benchmark_manifest.get("root") or benchmark_artifacts.data_dir))
    split_entries = _read_split(split_path)
    tasks = [str(item) for item in benchmark_manifest.get("tasks") or [task]]
    target_policy = resolve_target_label_policy(
        explicit=target_labels,
        prompt_id=prompt_id,
        task=task,
    )

    manifest = EvalRunManifest(
        run_id=run_id,
        model=ModelRef(model_id=model_id, path=model_path),
        benchmark=BenchmarkRef(
            benchmark_id=benchmark_id,
            root=str(benchmark_root),
            split=str(benchmark_manifest.get("split") or "test"),
            tasks=tasks,  # type: ignore[arg-type]
            manifest_path=str(split_path),
        ),
        spec=EvalSpec(
            spec_id=spec_id or f"{task}.imported",
            task=task,
            prompt=PromptRef(
                prompt_id=prompt_id,
                metadata={"source": "imported_prediction_snapshot"},
            ),
            inference=InferenceParams(backend="imported"),
            target_labels=target_policy.labels,
            metadata={
                "source": "imported_prediction_snapshot",
                "target_labels_source": target_policy.source,
            },
        ),
        status="succeeded",
        artifact_root=str(run_artifacts.run_dir),
        metadata={
            "source": "imported_prediction_snapshot",
            "prediction_root": str(prediction_root_path),
        },
    )
    manifest_path = run_artifacts.write_manifest(manifest)

    imported_count = 0
    missing: list[str] = []
    for json_relative in split_entries:
        gt_payload = read_json(benchmark_root / json_relative)
        if not isinstance(gt_payload, dict):
            raise ValueError(f"GT JSON must be an object: {benchmark_root / json_relative}")
        image = sample_image_path(json_relative, gt_payload, root=benchmark_root)
        source_path = source_index.find(json_relative=json_relative, image=image)
        if source_path is None:
            missing.append(str(json_relative))
            continue
        prediction = _prediction_from_payload(
            read_json(source_path),
            fallback_image=str(image),
            run_id=run_id,
            task=task,
            model_id=model_id,
            source_path=source_path,
        )
        run_artifacts.write_prediction(prediction, task=task)
        imported_count += 1

    if strict and missing:
        raise FileNotFoundError(f"missing {len(missing)} prediction files: {missing[:5]}")

    atomic_write_json(
        run_artifacts.reports_dir / "import_summary.json",
        {
            "run_id": run_id,
            "benchmark_id": benchmark_id,
            "prediction_root": str(prediction_root_path),
            "imported_predictions": imported_count,
            "missing_predictions": missing,
        },
    )
    report_path = evaluate_run(store_root=store_root, run_id=run_id) if evaluate else None
    return ImportedPredictionRun(
        run_id=run_id,
        run_manifest_path=manifest_path,
        report_path=report_path,
        imported_predictions=imported_count,
        missing_predictions=missing,
    )


class _PredictionSourceIndex:
    def __init__(self, root: Path) -> None:
        if not root.exists():
            raise FileNotFoundError(f"prediction root does not exist: {root}")
        self.root = root
        self._basename_index = self._build_basename_index(root)

    def find(self, *, json_relative: Path, image: Path) -> Path | None:
        direct_candidates = [
            self.root / json_relative,
            self.root / image.with_suffix(".json"),
            self.root / Path("json") / json_relative.name,
        ]
        for candidate in direct_candidates:
            if candidate.exists():
                return candidate
        matches = self._basename_index.get(json_relative.name, [])
        if len(matches) > 1:
            raise ValueError(
                f"ambiguous prediction basename={json_relative.name!r}: "
                + ", ".join(str(item) for item in matches[:5])
            )
        return matches[0] if matches else None

    @staticmethod
    def _build_basename_index(root: Path) -> dict[str, list[Path]]:
        index: dict[str, list[Path]] = {}
        for path in root.rglob("*.json"):
            index.setdefault(path.name, []).append(path)
        return index


def _read_split(path: Path) -> list[Path]:
    if not path.exists():
        raise FileNotFoundError(f"benchmark split manifest does not exist: {path}")
    return [Path(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _prediction_from_payload(
    payload: Any,
    *,
    fallback_image: str,
    run_id: str,
    task: TaskKind,
    model_id: str,
    source_path: Path,
) -> PredictionDocument:
    if not isinstance(payload, dict):
        raise ValueError(f"prediction JSON must be an object: {source_path}")
    normalized = dict(payload)
    normalized["image"] = str(
        normalized.get("image")
        or normalized.get("image_path")
        or normalized.get("imagePath")
        or fallback_image
    )
    normalized["metadata"] = {
        **dict(normalized.get("metadata") or {}),
        "producer": dict(normalized.get("metadata") or {}).get("producer", "imported"),
        "run_id": run_id,
        "model_id": model_id,
        "task": task,
        "source_path": str(source_path),
    }
    instances = []
    for item in normalized.get("instances") or []:
        if not isinstance(item, dict):
            raise ValueError(f"prediction instance must be an object: {source_path}")
        bbox = item.get("bbox") or item.get("bbox_2d")
        instances.append(
            PredictionInstance(
                label=str(item.get("label") or ""),
                bbox=list(bbox or []),
                keypoints=item.get("keypoints"),
                linestrip=item.get("linestrip"),
                score=item.get("score"),
                extra=dict(item.get("extra") or {}),
            )
        )
    document = PredictionDocument(
        image=str(normalized["image"]),
        status=str(normalized.get("status") or "predicted"),  # type: ignore[arg-type]
        instances=instances,
        image_id=normalized.get("image_id"),
        metadata=dict(normalized["metadata"]),
    )
    document.validate(task=task)
    return document
