from __future__ import annotations

from typing import Any


def run_target_labels(payload: dict[str, Any]) -> list[str]:
    spec = payload.get("spec")
    if not isinstance(spec, dict):
        return []
    labels = spec.get("target_labels") or []
    if not isinstance(labels, list):
        return []
    return [str(item).strip() for item in labels if str(item).strip()]


def run_target_label_set(payload: dict[str, Any]) -> set[str] | None:
    labels = run_target_labels(payload)
    return set(labels) if labels else None


def filter_instances_by_labels(
    instances: list[dict[str, Any]],
    labels: set[str] | None,
) -> list[dict[str, Any]]:
    if not labels:
        return instances
    return [item for item in instances if str(item.get("label") or "") in labels]


def filter_payload_instances(
    payload: dict[str, Any] | None,
    labels: set[str] | None,
) -> dict[str, Any] | None:
    if payload is None or not labels:
        return payload
    instances = payload.get("instances")
    if not isinstance(instances, list):
        return payload
    filtered = [
        item
        for item in instances
        if isinstance(item, dict) and str(item.get("label") or "") in labels
    ]
    return {**payload, "instances": filtered}


def scope_sample_diagnostics(
    diagnostics: dict[str, Any] | None,
    *,
    gt_instances: list[dict[str, Any]],
    pred_instances: list[dict[str, Any]],
    labels: set[str] | None,
) -> dict[str, Any] | None:
    if diagnostics is None or not labels:
        return diagnostics
    label_metrics = diagnostics.get("labels")
    if not isinstance(label_metrics, dict):
        return None
    if _diagnostics_already_scoped(
        label_metrics,
        gt_instances=gt_instances,
        pred_instances=pred_instances,
        labels=labels,
    ):
        return diagnostics

    selected_metrics = {
        str(label): value
        for label, value in label_metrics.items()
        if str(label) in labels and isinstance(value, dict)
    }
    gt_index_map = _scoped_index_map(gt_instances, labels)
    pred_index_map = _scoped_index_map(pred_instances, labels)
    matches = [
        _remap_match(item, gt_index_map=gt_index_map, pred_index_map=pred_index_map)
        for item in diagnostics.get("matches") or []
        if isinstance(item, dict) and str(item.get("label") or "") in labels
    ]
    matches = [item for item in matches if item is not None]
    false_negatives = [
        _remap_reference(item, gt_index_map)
        for item in diagnostics.get("false_negatives") or []
        if isinstance(item, dict) and str(item.get("label") or "") in labels
    ]
    false_negatives = [item for item in false_negatives if item is not None]
    false_positives = [
        _remap_reference(item, pred_index_map)
        for item in diagnostics.get("false_positives") or []
        if isinstance(item, dict) and str(item.get("label") or "") in labels
    ]
    false_positives = [item for item in false_positives if item is not None]
    keypoint_pair_count = sum(
        int(item.get("keypoint_pair_count") or 0) for item in selected_metrics.values()
    )
    return {
        **diagnostics,
        "gt_instance_count": len(gt_index_map),
        "pred_instance_count": len(pred_index_map),
        "matched_count": len(matches),
        "false_negative_count": len(false_negatives),
        "false_positive_count": len(false_positives),
        "mean_iou": _weighted_metric(selected_metrics, "mean_iou", "matched_count"),
        "keypoint_pair_count": keypoint_pair_count,
        "mean_keypoint_distance": _weighted_metric(
            selected_metrics,
            "mean_keypoint_distance",
            "keypoint_pair_count",
        ),
        "matches": matches,
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "labels": selected_metrics,
    }


def _scoped_index_map(
    instances: list[dict[str, Any]],
    labels: set[str],
) -> dict[int, int]:
    result: dict[int, int] = {}
    next_index = 0
    for index, instance in enumerate(instances):
        if str(instance.get("label") or "") not in labels:
            continue
        result[index] = next_index
        next_index += 1
    return result


def _diagnostics_already_scoped(
    label_metrics: dict[Any, Any],
    *,
    gt_instances: list[dict[str, Any]],
    pred_instances: list[dict[str, Any]],
    labels: set[str],
) -> bool:
    metric_labels = {str(label) for label in label_metrics}
    if not metric_labels or not metric_labels.issubset(labels):
        return False
    gt_count = sum(
        int(item.get("gt_count") or 0)
        for item in label_metrics.values()
        if isinstance(item, dict)
    )
    pred_count = sum(
        int(item.get("pred_count") or 0)
        for item in label_metrics.values()
        if isinstance(item, dict)
    )
    return (
        gt_count == len(filter_instances_by_labels(gt_instances, labels))
        and pred_count == len(filter_instances_by_labels(pred_instances, labels))
    )


def _remap_match(
    item: dict[str, Any],
    *,
    gt_index_map: dict[int, int],
    pred_index_map: dict[int, int],
) -> dict[str, Any] | None:
    gt_index = _int_or_none(item.get("gt_index"))
    pred_index = _int_or_none(item.get("pred_index"))
    if gt_index not in gt_index_map or pred_index not in pred_index_map:
        return None
    return {**item, "gt_index": gt_index_map[gt_index], "pred_index": pred_index_map[pred_index]}


def _remap_reference(item: dict[str, Any], index_map: dict[int, int]) -> dict[str, Any] | None:
    index = _int_or_none(item.get("index"))
    if index not in index_map:
        return None
    return {**item, "index": index_map[index]}


def _weighted_metric(
    label_metrics: dict[str, dict[str, Any]],
    value_key: str,
    weight_key: str,
) -> float | None:
    weighted_sum = 0.0
    weight_sum = 0
    for item in label_metrics.values():
        value = item.get(value_key)
        weight = int(item.get(weight_key) or 0)
        if value is None or weight <= 0:
            continue
        weighted_sum += float(value) * weight
        weight_sum += weight
    if weight_sum <= 0:
        return 0.0 if value_key == "mean_iou" else None
    return weighted_sum / weight_sum


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
