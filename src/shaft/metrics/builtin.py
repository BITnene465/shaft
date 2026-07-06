from __future__ import annotations

from typing import Any

from shaft.codec import ShaftCodecResult

from .base import ShaftEvalMetric
from .registry import register_eval_metric


def _normalize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(k): _normalize_value(v)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, list):
        return [_normalize_value(v) for v in value]
    if isinstance(value, str):
        return value.strip()
    return value


def _coerce_number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, list | tuple) or len(value) != 4:
        return None
    nums = [_coerce_number(item) for item in value]
    if any(item is None for item in nums):
        return None
    x1, y1, x2, y2 = nums
    if not (x2 > x1 and y2 > y1):
        return None
    return float(x1), float(y1), float(x2), float(y2)


def _coerce_keypoints(value: Any) -> list[tuple[float, float]] | None:
    if not isinstance(value, list | tuple):
        return None
    points: list[tuple[float, float]] = []
    for point in value:
        if not isinstance(point, list | tuple) or len(point) != 2:
            return None
        px = _coerce_number(point[0])
        py = _coerce_number(point[1])
        if px is None or py is None:
            return None
        points.append((float(px), float(py)))
    return points


def _coerce_field_map(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return {str(key): value[key] for key in value}


def _coerce_positive_float(value: Any) -> float | None:
    number = _coerce_number(value)
    if number is None or number <= 0:
        return None
    return float(number)


def _normalize_label(value: Any) -> str:
    return str(value).strip().lower()


def _coerce_instances(
    value: Any,
) -> list[tuple[str, tuple[float, float, float, float]]]:
    if not isinstance(value, list):
        return []
    instances: list[tuple[str, tuple[float, float, float, float]]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        bbox = _coerce_bbox(item.get("bbox_2d"))
        if bbox is None:
            continue
        label = _normalize_label(item.get("label", ""))
        instances.append((label, bbox))
    return instances


def _iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
    union_area = area_a + area_b - inter_area
    if union_area <= 0:
        return 0.0
    return float(inter_area / union_area)


def _match_instances(
    prediction_instances: list[tuple[str, tuple[float, float, float, float]]],
    target_instances: list[tuple[str, tuple[float, float, float, float]]],
    *,
    threshold: float,
) -> tuple[int, float]:
    candidates: list[tuple[float, int, int]] = []
    for pred_idx, (pred_label, pred_bbox) in enumerate(prediction_instances):
        for target_idx, (target_label, target_bbox) in enumerate(target_instances):
            if pred_label and target_label and pred_label != target_label:
                continue
            iou = _iou_xyxy(pred_bbox, target_bbox)
            if iou >= threshold:
                candidates.append((iou, pred_idx, target_idx))

    candidates.sort(key=lambda item: item[0], reverse=True)

    used_pred: set[int] = set()
    used_target: set[int] = set()
    true_positive = 0
    matched_iou = 0.0

    for iou, pred_idx, target_idx in candidates:
        if pred_idx in used_pred or target_idx in used_target:
            continue
        used_pred.add(pred_idx)
        used_target.add(target_idx)
        true_positive += 1
        matched_iou += iou

    return true_positive, matched_iou


@register_eval_metric("parse_success")
class ParseSuccessMetric(ShaftEvalMetric):
    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.values: list[float] = []

    def update(
        self,
        *,
        prediction: ShaftCodecResult,
        target: Any,
        sample_meta: dict[str, Any],
    ) -> None:
        _ = target, sample_meta
        self.values.append(1.0 if prediction.valid and not prediction.partial else 0.0)

    def compute(self) -> float:
        if not self.values:
            return 0.0
        return float(sum(self.values) / len(self.values))

    def reset(self) -> None:
        self.values = []


@register_eval_metric("parse_partial_rate")
class ParsePartialRateMetric(ShaftEvalMetric):
    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.values: list[float] = []

    def update(
        self,
        *,
        prediction: ShaftCodecResult,
        target: Any,
        sample_meta: dict[str, Any],
    ) -> None:
        _ = target, sample_meta
        self.values.append(1.0 if prediction.valid and prediction.partial else 0.0)

    def compute(self) -> float:
        if not self.values:
            return 0.0
        return float(sum(self.values) / len(self.values))

    def reset(self) -> None:
        self.values = []


@register_eval_metric("exact_match")
class ExactMatchMetric(ShaftEvalMetric):
    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.values: list[float] = []

    def update(
        self,
        *,
        prediction: ShaftCodecResult,
        target: Any,
        sample_meta: dict[str, Any],
    ) -> None:
        _ = sample_meta
        if not prediction.valid:
            self.values.append(0.0)
            return
        prediction_value = prediction.parsed if prediction.parsed is not None else prediction.raw_text
        self.values.append(1.0 if _normalize_value(prediction_value) == _normalize_value(target) else 0.0)

    def compute(self) -> float:
        if not self.values:
            return 0.0
        return float(sum(self.values) / len(self.values))

    def reset(self) -> None:
        self.values = []


@register_eval_metric("det_f1")
class DetectionF1Metric(ShaftEvalMetric):
    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.values: list[float] = []

    def update(
        self,
        *,
        prediction: ShaftCodecResult,
        target: Any,
        sample_meta: dict[str, Any],
    ) -> None:
        _ = sample_meta
        if not prediction.valid:
            self.values.append(0.0)
            return

        prediction_value = prediction.parsed
        if prediction_value is None:
            self.values.append(0.0)
            return

        pred_instances = _coerce_instances(prediction_value)
        target_instances = _coerce_instances(target)

        threshold = float(self.params.get("iou_threshold", 0.5))
        threshold = max(0.0, min(1.0, threshold))

        true_positive, _ = _match_instances(
            prediction_instances=pred_instances,
            target_instances=target_instances,
            threshold=threshold,
        )

        pred_count = len(pred_instances)
        target_count = len(target_instances)

        if pred_count == 0 and target_count == 0:
            self.values.append(1.0)
            return

        precision = true_positive / pred_count if pred_count > 0 else 0.0
        recall = true_positive / target_count if target_count > 0 else 0.0
        if precision == 0.0 and recall == 0.0:
            self.values.append(0.0)
            return
        self.values.append(float(2.0 * precision * recall / (precision + recall)))

    def compute(self) -> float:
        if not self.values:
            return 0.0
        return float(sum(self.values) / len(self.values))

    def reset(self) -> None:
        self.values = []


@register_eval_metric("det_iou")
class DetectionMeanIoUMetric(ShaftEvalMetric):
    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.values: list[float] = []

    def update(
        self,
        *,
        prediction: ShaftCodecResult,
        target: Any,
        sample_meta: dict[str, Any],
    ) -> None:
        _ = sample_meta
        if not prediction.valid:
            self.values.append(0.0)
            return

        prediction_value = prediction.parsed
        if prediction_value is None:
            self.values.append(0.0)
            return

        pred_instances = _coerce_instances(prediction_value)
        target_instances = _coerce_instances(target)

        threshold = float(self.params.get("iou_threshold", 0.5))
        threshold = max(0.0, min(1.0, threshold))

        true_positive, matched_iou = _match_instances(
            prediction_instances=pred_instances,
            target_instances=target_instances,
            threshold=threshold,
        )

        if true_positive == 0:
            if len(pred_instances) == 0 and len(target_instances) == 0:
                self.values.append(1.0)
            else:
                self.values.append(0.0)
            return
        self.values.append(matched_iou / float(true_positive))

    def compute(self) -> float:
        if not self.values:
            return 0.0
        return float(sum(self.values) / len(self.values))

    def reset(self) -> None:
        self.values = []


@register_eval_metric("subattr_fields")
class SubAttrFieldMatchMetric(ShaftEvalMetric):
    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.values: list[float] = []

    def update(
        self,
        *,
        prediction: ShaftCodecResult,
        target: Any,
        sample_meta: dict[str, Any],
    ) -> None:
        _ = sample_meta
        if not prediction.valid:
            self.values.append(0.0)
            return

        pred_fields = _coerce_field_map(prediction.parsed)
        tgt_fields = _coerce_field_map(target)
        if pred_fields is None or tgt_fields is None:
            self.values.append(0.0)
            return

        raw_fields = self.params.get("fields", "stroke_pattern,geometry_style")
        if isinstance(raw_fields, list):
            fields = [str(item).strip() for item in raw_fields if str(item).strip()]
        else:
            fields = [item.strip() for item in str(raw_fields).split(",") if item.strip()]

        if not fields:
            fields = ["stroke_pattern", "geometry_style"]

        for field in fields:
            pred_value = pred_fields.get(field)
            tgt_value = tgt_fields.get(field)
            if pred_value is None or tgt_value is None:
                self.values.append(0.0)
                return
            if _normalize_value(pred_value) != _normalize_value(tgt_value):
                self.values.append(0.0)
                return

        self.values.append(1.0)

    def compute(self) -> float:
        if not self.values:
            return 0.0
        return float(sum(self.values) / len(self.values))

    def reset(self) -> None:
        self.values = []


@register_eval_metric("keypoint_pck")
class KeypointPCKMetric(ShaftEvalMetric):
    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.values: list[float] = []

    def update(
        self,
        *,
        prediction: ShaftCodecResult,
        target: Any,
        sample_meta: dict[str, Any],
    ) -> None:
        _ = sample_meta
        if not prediction.valid:
            self.values.append(0.0)
            return

        pred_data = _coerce_field_map(prediction.parsed)
        tgt_data = _coerce_field_map(target)
        if pred_data is None or tgt_data is None:
            self.values.append(0.0)
            return

        pred_kpts = _coerce_keypoints(pred_data.get("points_2d") or pred_data.get("keypoints_2d"))
        tgt_kpts = _coerce_keypoints(tgt_data.get("points_2d") or tgt_data.get("keypoints_2d"))
        if pred_kpts is None or tgt_kpts is None or len(pred_kpts) != len(tgt_kpts):
            self.values.append(0.0)
            return
        if len(pred_kpts) == 0:
            self.values.append(1.0)
            return

        scale = self._resolve_coordinate_scale(pred_kpts=pred_kpts, tgt_kpts=tgt_kpts, sample_meta=sample_meta)

        threshold = float(self.params.get("threshold", 0.05))
        threshold = max(0.0, threshold)
        tolerance = threshold * scale
        if tolerance <= 0.0:
            self.values.append(0.0)
            return

        correct = 0
        for pred_point, target_point in zip(pred_kpts, tgt_kpts, strict=True):
            dx = pred_point[0] - target_point[0]
            dy = pred_point[1] - target_point[1]
            if (dx * dx + dy * dy) <= tolerance * tolerance:
                correct += 1

        self.values.append(correct / len(tgt_kpts))

    def _resolve_coordinate_scale(
        self,
        *,
        pred_kpts: list[tuple[float, float]],
        tgt_kpts: list[tuple[float, float]],
        sample_meta: dict[str, Any],
    ) -> float:
        coordinate_space = str(self.params.get("coordinate_space", "normalized_1000")).strip().lower()
        if coordinate_space in {
            "normalized",
            "normalized_1000",
            "bbox_2d",
            "keypoints_2d",
            "points_2d",
            "bins",
            "quantized",
        }:
            return float(self.params.get("coordinate_scale", self.params.get("num_bins", 1000)))
        if coordinate_space != "image":
            raise ValueError(f"Unsupported keypoint_pck coordinate_space={coordinate_space!r}.")

        image_width: float | None = None
        image_height: float | None = None
        extra = sample_meta.get("extra")
        if isinstance(extra, dict):
            image_width = _coerce_positive_float(extra.get("image_width"))
            image_height = _coerce_positive_float(extra.get("image_height"))

        if image_width is None or image_height is None:
            all_x = [point[0] for point in pred_kpts + tgt_kpts]
            all_y = [point[1] for point in pred_kpts + tgt_kpts]
            image_width = max(all_x) - min(all_x)
            image_height = max(all_y) - min(all_y)

        scale_x = image_width if image_width and image_width > 0 else 1.0
        scale_y = image_height if image_height and image_height > 0 else 1.0
        return max(scale_x, scale_y)

    def compute(self) -> float:
        if not self.values:
            return 0.0
        return float(sum(self.values) / len(self.values))

    def reset(self) -> None:
        self.values = []
