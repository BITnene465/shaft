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
    benchmark_id: str
    benchmark_split: str
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
    empty_prediction_rate: float = 0.0
    pred_gt_ratio: float = 0.0
    output_char_length: dict[str, Any] = field(default_factory=dict)
    output_token_length: dict[str, Any] = field(default_factory=dict)
    dense_sample_buckets: list[dict[str, Any]] = field(default_factory=list)
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
    output_char_lengths: list[int] = []
    output_token_lengths: list[int] = []

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
        char_count, token_count = _prediction_output_lengths(
            raw_outputs_dir=predictions_dir.parent / "raw_outputs",
            prediction=pred_doc,
            image=image,
        )
        if char_count is not None:
            output_char_lengths.append(char_count)
        if token_count is not None:
            output_token_lengths.append(token_count)
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
        benchmark_id=str(benchmark.get("benchmark_id") or ""),
        benchmark_split=str(benchmark.get("split") or ""),
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
        empty_prediction_rate=_empty_prediction_rate(metrics.samples),
        pred_gt_ratio=_safe_div(metrics.pred_instance_count, metrics.gt_instance_count),
        output_char_length=_length_distribution(output_char_lengths),
        output_token_length=_length_distribution(output_token_lengths),
        dense_sample_buckets=_dense_sample_buckets(metrics.samples),
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
        "benchmark_id": report.benchmark_id,
        "benchmark_split": report.benchmark_split,
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
        "empty_prediction_rate": report.empty_prediction_rate,
        "pred_gt_ratio": report.pred_gt_ratio,
        "output_char_length": report.output_char_length,
        "output_token_length": report.output_token_length,
        "dense_sample_buckets": report.dense_sample_buckets,
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


def _prediction_output_lengths(
    *,
    raw_outputs_dir: Path,
    prediction: dict[str, Any],
    image: Path,
) -> tuple[int | None, int | None]:
    metadata = prediction.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    char_count = _optional_int(metadata.get("output_char_count"))
    token_count = _optional_int(metadata.get("output_token_count"))
    raw_output_path = _raw_output_path(raw_outputs_dir, image=image)
    if char_count is None and raw_output_path.exists():
        text = raw_output_path.read_text(encoding="utf-8", errors="replace")
        char_count = len(text)
        token_count = token_count if token_count is not None else _approx_token_count(text)
    return char_count, token_count


def _raw_output_path(raw_outputs_dir: Path, *, image: Path) -> Path:
    parts = image.parts
    if len(parts) >= 3 and parts[1] == "images":
        relative = Path(parts[0]) / "txt" / image.with_suffix(".txt").name
    else:
        relative = image.with_suffix(".txt")
    return raw_outputs_dir / relative


def _optional_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _approx_token_count(text: str) -> int:
    return len([item for item in str(text or "").replace("\n", " ").split(" ") if item])


def _empty_prediction_rate(samples: list[dict[str, Any]]) -> float:
    if not samples:
        return 0.0
    empty_count = sum(1 for sample in samples if int(sample.get("pred_instance_count") or 0) == 0)
    return _safe_div(empty_count, len(samples))


def _length_distribution(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "p50": None, "p90": None, "max": None, "mean": None}
    ordered = sorted(int(value) for value in values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": _percentile(ordered, 0.5),
        "p90": _percentile(ordered, 0.9),
        "max": ordered[-1],
        "mean": sum(ordered) / float(len(ordered)),
    }


def _percentile(values: list[int], ratio: float) -> int:
    if not values:
        return 0
    index = min(len(values) - 1, max(0, int(round((len(values) - 1) * ratio))))
    return values[index]


def _dense_sample_buckets(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bucket_rows: dict[str, dict[str, Any]] = {}
    for sample in samples:
        gt_count = int(sample.get("gt_instance_count") or 0)
        bucket = _density_bucket(gt_count)
        row = bucket_rows.setdefault(
            bucket,
            {
                "bucket": bucket,
                "sample_count": 0,
                "gt_instance_count": 0,
                "pred_instance_count": 0,
                "matched_count": 0,
                "empty_prediction_count": 0,
            },
        )
        row["sample_count"] += 1
        row["gt_instance_count"] += gt_count
        row["pred_instance_count"] += int(sample.get("pred_instance_count") or 0)
        row["matched_count"] += int(sample.get("matched_count") or 0)
        if int(sample.get("pred_instance_count") or 0) == 0:
            row["empty_prediction_count"] += 1
    rows: list[dict[str, Any]] = []
    for bucket in ("empty", "1", "2-5", "6-20", "21+"):
        row = bucket_rows.get(bucket)
        if row is None:
            continue
        row = dict(row)
        row["precision_iou50"] = _safe_div(row["matched_count"], row["pred_instance_count"])
        row["recall_iou50"] = _safe_div(row["matched_count"], row["gt_instance_count"])
        row["empty_prediction_rate"] = _safe_div(row["empty_prediction_count"], row["sample_count"])
        rows.append(row)
    return rows


def _density_bucket(gt_count: int) -> str:
    if gt_count <= 0:
        return "empty"
    if gt_count == 1:
        return "1"
    if gt_count <= 5:
        return "2-5"
    if gt_count <= 20:
        return "6-20"
    return "21+"


def _safe_div(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if float(denominator) else 0.0


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return payload
