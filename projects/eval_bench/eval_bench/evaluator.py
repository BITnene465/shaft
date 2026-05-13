from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

from .artifacts import DEFAULT_STORE_ROOT, RunArtifacts, atomic_write_json
from .label_policy import resolve_target_labels


@dataclass(frozen=True)
class LabelMetric:
    label: str
    gt_count: int = 0
    pred_count: int = 0
    matched_count: int = 0
    precision_iou50: float = 0.0
    recall_iou50: float = 0.0
    mean_iou: float = 0.0
    keypoint_pair_count: int = 0
    mean_keypoint_distance: float | None = None


@dataclass(frozen=True)
class EvalReport:
    run_id: str
    task: str
    sample_count: int
    prediction_file_count: int
    gt_instance_count: int
    pred_instance_count: int
    matched_count: int
    precision_iou50: float
    recall_iou50: float
    mean_iou: float
    target_labels: list[str] = field(default_factory=list)
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
    task = str(spec.get("task") or "")
    target_labels = _target_labels(spec)
    target_label_set = set(target_labels)
    benchmark_root = Path(str(benchmark.get("root") or ""))
    split_path = Path(str(benchmark.get("manifest_path") or ""))
    if not split_path.exists():
        raise FileNotFoundError(f"benchmark split manifest does not exist: {split_path}")
    if not benchmark_root.exists():
        raise FileNotFoundError(f"benchmark root does not exist: {benchmark_root}")

    sample_entries = _read_split(split_path)
    label_stats: dict[str, dict[str, Any]] = defaultdict(_empty_label_stats)
    missing_predictions: list[str] = []
    sample_reports: list[dict[str, Any]] = []
    prediction_file_count = 0

    for sample_index, json_relative in enumerate(sample_entries):
        gt_doc = _load_json(benchmark_root / json_relative)
        image = _image_path_from_gt(json_relative, gt_doc)
        prediction_path = _prediction_path(predictions_dir, image)
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
        sample_report = _sample_diagnostic(
            sample_index=sample_index,
            json_relative=json_relative,
            image=image,
            gt_instances=gt_instances,
            pred_instances=pred_instances,
            iou_threshold=iou_threshold,
            has_prediction=prediction_path.exists(),
        )
        sample_reports.append(sample_report)
        _accumulate_sample(
            label_stats=label_stats,
            sample_report=sample_report,
            task=task,
        )

    label_metrics = [_finalize_label(label, stats) for label, stats in sorted(label_stats.items())]
    gt_count = sum(item.gt_count for item in label_metrics)
    pred_count = sum(item.pred_count for item in label_metrics)
    matched_count = sum(item.matched_count for item in label_metrics)
    weighted_iou_sum = sum(
        float(item.mean_iou) * int(item.matched_count)
        for item in label_metrics
        if item.matched_count > 0
    )
    return EvalReport(
        run_id=run_id,
        task=task,
        sample_count=len(sample_entries),
        prediction_file_count=prediction_file_count,
        gt_instance_count=gt_count,
        pred_instance_count=pred_count,
        matched_count=matched_count,
        precision_iou50=_safe_div(matched_count, pred_count),
        recall_iou50=_safe_div(matched_count, gt_count),
        mean_iou=_safe_div(weighted_iou_sum, matched_count),
        target_labels=target_labels,
        labels=label_metrics,
        missing_predictions=missing_predictions,
        samples=sample_reports,
    )


def _sample_diagnostic(
    *,
    sample_index: int,
    json_relative: Path,
    image: Path,
    gt_instances: list[dict[str, Any]],
    pred_instances: list[dict[str, Any]],
    iou_threshold: float,
    has_prediction: bool,
) -> dict[str, Any]:
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    matches: list[dict[str, Any]] = []
    labels: dict[str, dict[str, Any]] = {}

    for label in sorted({item["label"] for item in gt_instances + pred_instances}):
        gt_indices = [index for index, item in enumerate(gt_instances) if item["label"] == label]
        pred_indices = [index for index, item in enumerate(pred_instances) if item["label"] == label]
        gt_for_label = [gt_instances[index] for index in gt_indices]
        pred_for_label = [pred_instances[index] for index in pred_indices]
        label_matches = _match_instance_indices(
            gt_for_label,
            pred_for_label,
            iou_threshold=iou_threshold,
        )
        iou_sum = 0.0
        for local_gt_index, local_pred_index, iou in label_matches:
            gt_index = gt_indices[local_gt_index]
            pred_index = pred_indices[local_pred_index]
            matched_gt.add(gt_index)
            matched_pred.add(pred_index)
            iou_sum += iou
            match = {
                "label": label,
                "gt_index": gt_index,
                "pred_index": pred_index,
                "iou": iou,
            }
            keypoint_distance = _keypoint_distance(gt_instances[gt_index], pred_instances[pred_index])
            if keypoint_distance is not None:
                match["keypoint_distance"] = keypoint_distance
            matches.append(match)
        labels[label] = {
            "gt_count": len(gt_indices),
            "pred_count": len(pred_indices),
            "matched_count": len(label_matches),
            "false_negative_count": len(gt_indices) - len(label_matches),
            "false_positive_count": len(pred_indices) - len(label_matches),
            "mean_iou": _safe_div(iou_sum, len(label_matches)),
        }

    matches.sort(key=lambda item: (str(item["label"]), int(item["gt_index"]), int(item["pred_index"])))
    false_negatives = [
        _instance_reference(index=index, instance=instance)
        for index, instance in enumerate(gt_instances)
        if index not in matched_gt
    ]
    false_positives = [
        _instance_reference(index=index, instance=instance)
        for index, instance in enumerate(pred_instances)
        if index not in matched_pred
    ]
    matched_count = len(matches)
    return {
        "index": sample_index,
        "json_path": str(json_relative),
        "image": str(image),
        "has_prediction": has_prediction,
        "gt_instance_count": len(gt_instances),
        "pred_instance_count": len(pred_instances),
        "matched_count": matched_count,
        "false_negative_count": len(false_negatives),
        "false_positive_count": len(false_positives),
        "mean_iou": _safe_div(sum(float(item["iou"]) for item in matches), matched_count),
        "matches": matches,
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "labels": labels,
    }


def _target_labels(spec: dict[str, Any]) -> list[str]:
    prompt = spec.get("prompt")
    if not isinstance(prompt, dict):
        prompt = {}
    prompt_metadata = prompt.get("metadata")
    if not isinstance(prompt_metadata, dict):
        prompt_metadata = {}
    return resolve_target_labels(
        explicit=spec.get("target_labels"),
        prompt_id=str(prompt.get("prompt_id") or ""),
        task=str(spec.get("task") or ""),
        prompt_metadata=prompt_metadata,
    )


def _report_summary(report: EvalReport) -> dict[str, Any]:
    return {
        "run_id": report.run_id,
        "task": report.task,
        "sample_count": report.sample_count,
        "prediction_file_count": report.prediction_file_count,
        "gt_instance_count": report.gt_instance_count,
        "pred_instance_count": report.pred_instance_count,
        "matched_count": report.matched_count,
        "precision_iou50": report.precision_iou50,
        "recall_iou50": report.recall_iou50,
        "mean_iou": report.mean_iou,
        "label_count": len(report.labels),
        "labels": [item.label for item in report.labels],
        "target_labels": report.target_labels,
        "missing_prediction_count": len(report.missing_predictions),
    }


def _accumulate_sample(
    *,
    label_stats: dict[str, dict[str, Any]],
    sample_report: dict[str, Any],
    task: str,
) -> None:
    labels = sample_report.get("labels") or {}
    if not isinstance(labels, dict):
        return
    for label, sample_label_stats in labels.items():
        if not isinstance(sample_label_stats, dict):
            continue
        stats = label_stats[label]
        stats["gt_count"] += int(sample_label_stats.get("gt_count") or 0)
        stats["pred_count"] += int(sample_label_stats.get("pred_count") or 0)
        stats["matched_count"] += int(sample_label_stats.get("matched_count") or 0)
    for match in sample_report.get("matches") or []:
        if not isinstance(match, dict):
            continue
        label = str(match.get("label") or "")
        if not label:
            continue
        stats = label_stats[label]
        stats["iou_sum"] += float(match.get("iou") or 0.0)
        distance = match.get("keypoint_distance") if task == "keypoint" else None
        if distance is not None:
            stats["keypoint_pair_count"] += 1
            stats["keypoint_distance_sum"] += float(distance)


def _match_instance_indices(
    gt_instances: list[dict[str, Any]],
    pred_instances: list[dict[str, Any]],
    *,
    iou_threshold: float,
) -> list[tuple[int, int, float]]:
    candidates: list[tuple[float, int, int]] = []
    for gt_index, gt_item in enumerate(gt_instances):
        for pred_index, pred_item in enumerate(pred_instances):
            iou = _bbox_iou(gt_item["bbox"], pred_item["bbox"])
            if iou >= iou_threshold:
                candidates.append((iou, gt_index, pred_index))
    candidates.sort(reverse=True)
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for iou, gt_index, pred_index in candidates:
        if gt_index in used_gt or pred_index in used_pred:
            continue
        used_gt.add(gt_index)
        used_pred.add(pred_index)
        matches.append((gt_index, pred_index, iou))
    return matches


def _instance_reference(*, index: int, instance: dict[str, Any]) -> dict[str, Any]:
    return {
        "index": index,
        "label": instance["label"],
        "bbox": instance["bbox"],
    }


def _empty_label_stats() -> dict[str, Any]:
    return {
        "gt_count": 0,
        "pred_count": 0,
        "matched_count": 0,
        "iou_sum": 0.0,
        "keypoint_pair_count": 0,
        "keypoint_distance_sum": 0.0,
    }


def _finalize_label(label: str, stats: dict[str, Any]) -> LabelMetric:
    matched_count = int(stats["matched_count"])
    keypoint_pair_count = int(stats["keypoint_pair_count"])
    mean_keypoint_distance = (
        _safe_div(float(stats["keypoint_distance_sum"]), keypoint_pair_count)
        if keypoint_pair_count > 0
        else None
    )
    return LabelMetric(
        label=label,
        gt_count=int(stats["gt_count"]),
        pred_count=int(stats["pred_count"]),
        matched_count=matched_count,
        precision_iou50=_safe_div(int(stats["matched_count"]), int(stats["pred_count"])),
        recall_iou50=_safe_div(int(stats["matched_count"]), int(stats["gt_count"])),
        mean_iou=_safe_div(float(stats["iou_sum"]), matched_count),
        keypoint_pair_count=keypoint_pair_count,
        mean_keypoint_distance=mean_keypoint_distance,
    )


def _prediction_path(predictions_dir: Path, image: Path) -> Path:
    parts = image.parts
    if len(parts) >= 3 and parts[1] == "images":
        return predictions_dir / Path(parts[0]) / "json" / image.with_suffix(".json").name
    return predictions_dir / image.with_suffix(".json")


def _image_path_from_gt(json_relative: Path, payload: dict[str, Any]) -> Path:
    image_path = payload.get("image_path")
    if isinstance(image_path, str) and image_path.strip():
        return Path(image_path)
    if len(json_relative.parts) >= 2 and json_relative.parts[1] == "json":
        return Path(json_relative.parts[0]) / "images" / json_relative.with_suffix(".png").name
    return json_relative.with_suffix(".png")


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


def _keypoint_distance(gt_item: dict[str, Any], pred_item: dict[str, Any]) -> float | None:
    gt_points = gt_item["points"]
    pred_points = pred_item["points"]
    if len(gt_points) < 2 or len(pred_points) < 2:
        return None
    return (_point_distance(gt_points[0], pred_points[0]) + _point_distance(gt_points[-1], pred_points[-1])) / 2


def _point_distance(point_a: list[float], point_b: list[float]) -> float:
    return ((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2) ** 0.5


def _bbox_iou(box_a: list[float], box_b: list[float]) -> float:
    if len(box_a) != 4 or len(box_b) != 4:
        return 0.0
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return _safe_div(intersection, union)


def _safe_div(numerator: float | int, denominator: float | int) -> float:
    return float(numerator) / float(denominator) if float(denominator) else 0.0


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
