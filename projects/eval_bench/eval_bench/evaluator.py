from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

from .artifacts import DEFAULT_STORE_ROOT, RunArtifacts, atomic_write_json
from .benchmark import resolve_benchmark_split_path
from .eval_semantics import resolve_eval_semantics
from .metrics import LabelMetric, MetricSample, evaluate_metric_samples
from .sample_paths import prediction_json_path, sample_image_path


@dataclass(frozen=True)
class EvalReport:
    run_id: str
    task: str
    metric_profile: str
    sample_count: int
    prediction_file_count: int
    gt_instance_count: int
    pred_instance_count: int
    matched_count: int
    precision_iou50: float
    recall_iou50: float
    mean_iou: float
    keypoint_pair_count: int = 0
    mean_keypoint_distance: float | None = None
    target_labels: list[str] = field(default_factory=list)
    target_labels_source: str = "unscoped"
    labels: list[LabelMetric] = field(default_factory=list)
    missing_predictions: list[str] = field(default_factory=list)
    samples: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "labels": [asdict(item) for item in self.labels],
        }


def evaluate_run(
    *,
    store_root: str | Path = DEFAULT_STORE_ROOT,
    run_id: str,
    iou_threshold: float = 0.5,
) -> Path:
    artifacts = RunArtifacts(store_root, run_id)
    manifest = _load_json(artifacts.manifest_path)
    report = evaluate_manifest(
        run_id=run_id,
        run_manifest=manifest,
        predictions_dir=artifacts.predictions_dir,
        iou_threshold=iou_threshold,
    )
    path = artifacts.reports_dir / "metrics.json"
    atomic_write_json(path, report.to_dict())
    atomic_write_json(artifacts.reports_dir / "summary.json", _report_summary(report))
    return path


def evaluate_manifest(
    *,
    run_id: str,
    run_manifest: dict[str, Any],
    predictions_dir: Path,
    iou_threshold: float = 0.5,
) -> EvalReport:
    benchmark = dict(run_manifest.get("benchmark") or {})
    spec = dict(run_manifest.get("spec") or {})
    semantics = resolve_eval_semantics(spec)
    task = semantics.task
    target_labels = semantics.target_labels
    target_label_set = set(target_labels)
    benchmark_root = Path(str(benchmark.get("root") or ""))
    split_path = resolve_benchmark_split_path(benchmark, split=benchmark.get("split"))
    if not split_path.exists():
        raise FileNotFoundError(f"benchmark split manifest does not exist: {split_path}")
    if not benchmark_root.exists():
        raise FileNotFoundError(f"benchmark root does not exist: {benchmark_root}")

    sample_entries = _read_split(split_path)
    missing_predictions: list[str] = []
    metric_samples: list[MetricSample] = []
    prediction_file_count = 0

    for sample_index, json_relative in enumerate(sample_entries):
        gt_doc = _load_json(benchmark_root / json_relative)
        image = sample_image_path(json_relative, gt_doc, root=benchmark_root)
        prediction_path = prediction_json_path(predictions_dir, image)
        if prediction_path.exists():
            pred_doc = _load_json(prediction_path)
            prediction_file_count += 1
        else:
            pred_doc = {"instances": []}
            missing_predictions.append(str(json_relative))
        gt_instances = [
            item
            for item in (_normalize_instance(instance) for instance in gt_doc.get("instances") or [])
            if not target_label_set or item["label"] in target_label_set
        ]
        pred_instances = [
            item
            for item in (_normalize_instance(instance) for instance in pred_doc.get("instances") or [])
            if not target_label_set or item["label"] in target_label_set
        ]
        metric_samples.append(
            MetricSample(
                sample_index=sample_index,
                json_relative=json_relative,
                image=image,
                gt_instances=gt_instances,
                pred_instances=pred_instances,
                has_prediction=prediction_path.exists(),
            )
        )
    metrics = evaluate_metric_samples(
        metric_samples,
        profile=semantics.metric_profile,
        iou_threshold=iou_threshold,
    )
    return EvalReport(
        run_id=run_id,
        task=task,
        metric_profile=semantics.metric_profile.profile_id,
        sample_count=len(sample_entries),
        prediction_file_count=prediction_file_count,
        gt_instance_count=metrics.gt_instance_count,
        pred_instance_count=metrics.pred_instance_count,
        matched_count=metrics.matched_count,
        precision_iou50=metrics.precision_iou50,
        recall_iou50=metrics.recall_iou50,
        mean_iou=metrics.mean_iou,
        keypoint_pair_count=metrics.keypoint_pair_count,
        mean_keypoint_distance=metrics.mean_keypoint_distance,
        target_labels=target_labels,
        target_labels_source=semantics.target_labels_source,
        labels=metrics.labels,
        missing_predictions=missing_predictions,
        samples=metrics.samples,
    )


def _report_summary(report: EvalReport) -> dict[str, Any]:
    return {
        "run_id": report.run_id,
        "task": report.task,
        "metric_profile": report.metric_profile,
        "sample_count": report.sample_count,
        "prediction_file_count": report.prediction_file_count,
        "gt_instance_count": report.gt_instance_count,
        "pred_instance_count": report.pred_instance_count,
        "matched_count": report.matched_count,
        "precision_iou50": report.precision_iou50,
        "recall_iou50": report.recall_iou50,
        "mean_iou": report.mean_iou,
        "keypoint_pair_count": report.keypoint_pair_count,
        "mean_keypoint_distance": report.mean_keypoint_distance,
        "label_count": len(report.labels),
        "labels": [item.label for item in report.labels],
        "target_labels": report.target_labels,
        "target_labels_source": report.target_labels_source,
        "missing_prediction_count": len(report.missing_predictions),
    }


def _normalize_instance(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("instance must be a JSON object.")
    label = str(payload.get("label") or "").strip()
    if not label:
        raise ValueError("instance label must be non-empty.")
    return {
        "label": label,
        "bbox": [float(item) for item in payload.get("bbox") or []],
        "points": _points(payload),
    }


def _points(payload: dict[str, Any]) -> list[list[float]]:
    raw_points = payload.get("keypoints") or payload.get("linestrip") or []
    if not isinstance(raw_points, list):
        return []
    points: list[list[float]] = []
    for item in raw_points:
        if isinstance(item, list) and len(item) == 2:
            points.append([float(item[0]), float(item[1])])
    return points


def _read_split(path: Path) -> list[Path]:
    return [
        Path(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return payload
