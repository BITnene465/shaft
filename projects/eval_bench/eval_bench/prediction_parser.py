from __future__ import annotations

from typing import Any

from shaft.codec import decode_with_codec

from .schema import PredictionDocument, PredictionInstance, TaskKind

NUM_BINS = 1000
GROUPED_DETECTION_LABEL_ORDER = ("shape", "icon", "image", "line", "arrow")


def parse_prediction_text(
    *,
    text: str,
    task: TaskKind,
    image: str,
    image_width: int,
    image_height: int,
    metadata: dict[str, Any],
    dedupe_iou_threshold: float = 0.95,
) -> PredictionDocument:
    decoded = decode_with_codec("json_any", str(text or ""))
    _record_decode_metadata(metadata, decoded=decoded, text=text)
    parsed = decoded.parsed if decoded.valid else None
    if task == "keypoint":
        instances = _parse_keypoint_instances(parsed, image_width=image_width, image_height=image_height)
    else:
        instances = _parse_detection_instances(parsed, image_width=image_width, image_height=image_height)
    document = PredictionDocument(
        image=image,
        instances=_dedupe_instances(instances, iou_threshold=dedupe_iou_threshold),
        metadata=metadata,
    )
    document.validate(task=task)
    return document


def _record_decode_metadata(metadata: dict[str, Any], *, decoded: Any, text: str) -> None:
    parser_metadata = metadata.get("parser")
    if not isinstance(parser_metadata, dict):
        parser_metadata = {}
    parser_metadata["decode_valid"] = bool(decoded.valid)
    parser_metadata["decode_partial"] = bool(decoded.partial)
    if decoded.error_type:
        parser_metadata["decode_error_type"] = decoded.error_type
    if decoded.error:
        parser_metadata["decode_error"] = decoded.error
    parser_metadata["decode_empty_repair"] = (
        bool(decoded.valid)
        and bool(decoded.partial)
        and decoded.parsed == []
        and str(text or "").lstrip().startswith("[")
        and len(str(text or "").strip()) > 2
    )
    metadata["parser"] = parser_metadata


def _parse_detection_instances(
    payload: Any,
    *,
    image_width: int,
    image_height: int,
) -> list[PredictionInstance]:
    instances: list[PredictionInstance] = []
    for item in _items_from_payload(payload):
        if not isinstance(item, dict):
            continue
        raw_label = item.get("label")
        label = _normalize_detection_label(raw_label)
        bbox = _bbox_from_item(item, image_width=image_width, image_height=image_height)
        if label is None or bbox is None:
            continue
        extra: dict[str, Any] = {}
        original_label = _label_text(raw_label)
        if original_label and original_label != label:
            extra["source_label_before_normalization"] = original_label
        instances.append(PredictionInstance(label=label, bbox=bbox, extra=extra))
    return instances


def _parse_keypoint_instances(
    payload: Any,
    *,
    image_width: int,
    image_height: int,
) -> list[PredictionInstance]:
    if isinstance(payload, dict) and "points_2d" in payload:
        label = _label_text(payload.get("label")) or "line"
        points = _points_from_value(payload.get("points_2d"), image_width, image_height)
        bbox = _bbox_from_points(points, image_width=image_width, image_height=image_height)
        return [PredictionInstance(label=label, bbox=bbox, keypoints=points)] if bbox else []

    if isinstance(payload, dict) and "keypoints_2d" in payload:
        points = _points_from_value(payload.get("keypoints_2d"), image_width, image_height)
        bbox = _bbox_from_points(points, image_width=image_width, image_height=image_height)
        return [PredictionInstance(label="arrow", bbox=bbox, keypoints=points)] if bbox else []

    instances: list[PredictionInstance] = []
    for item in _items_from_payload(payload):
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "arrow").strip().lower() or "arrow"
        bbox = _bbox_from_item(item, image_width=image_width, image_height=image_height)
        points = _points_from_value(
            item.get("points_2d")
            or item.get("keypoints_2d")
            or item.get("keypoints")
            or item.get("linestrip"),
            image_width,
            image_height,
        )
        if bbox is None:
            bbox = _bbox_from_points(points, image_width=image_width, image_height=image_height)
        if bbox is None:
            continue
        instances.append(PredictionInstance(label=label, bbox=bbox, keypoints=points or None))
    return instances


def _items_from_payload(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        grouped_items = _items_from_grouped_detection_payload(payload)
        if grouped_items:
            return grouped_items
        for key in ("detections", "instances", "objects", "items", "result"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        if "bbox_2d" in payload or "bbox" in payload:
            return [payload]
    return []


def _items_from_grouped_detection_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for label in GROUPED_DETECTION_LABEL_ORDER:
        raw_boxes = payload.get(label)
        if raw_boxes is None:
            continue
        if not isinstance(raw_boxes, list):
            continue
        for raw_box in raw_boxes:
            if isinstance(raw_box, list) and len(raw_box) >= 4:
                items.append({"label": label, "bbox_2d": raw_box})
    return items


def _normalize_detection_label(value: Any) -> str | None:
    label = _label_text(value)
    if label == "shape_combination":
        label = "icon"
    if label in {"single_arrow", "double_arrow", "arrow_instance", "arrows"}:
        label = "arrow"
    if label in {"icon", "image", "shape", "line", "arrow"}:
        return label
    return None


def _label_text(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _bbox_from_item(
    item: dict[str, Any],
    *,
    image_width: int,
    image_height: int,
) -> list[float] | None:
    for key in ("bbox_2d", "bbox", "box_2d", "box"):
        value = item.get(key)
        if isinstance(value, list) and len(value) >= 4:
            try:
                return _bbox_to_pixels(
                    [float(value[0]), float(value[1]), float(value[2]), float(value[3])],
                    image_width=image_width,
                    image_height=image_height,
                )
            except (TypeError, ValueError):
                return None
    return None


def _bbox_to_pixels(
    bbox: list[float],
    *,
    image_width: int,
    image_height: int,
) -> list[float] | None:
    x1, y1, x2, y2 = bbox
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= float(NUM_BINS):
        x1 = x1 / float(NUM_BINS) * float(image_width)
        x2 = x2 / float(NUM_BINS) * float(image_width)
        y1 = y1 / float(NUM_BINS) * float(image_height)
        y2 = y2 / float(NUM_BINS) * float(image_height)
    x1, x2 = sorted((max(0.0, x1), min(float(image_width), x2)))
    y1, y2 = sorted((max(0.0, y1), min(float(image_height), y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _points_from_value(value: Any, image_width: int, image_height: int) -> list[list[float]]:
    if not isinstance(value, list):
        return []
    points: list[list[float]] = []
    for item in value:
        if not isinstance(item, list) or len(item) != 2:
            continue
        try:
            x = float(item[0])
            y = float(item[1])
        except (TypeError, ValueError):
            continue
        if max(abs(x), abs(y)) <= float(NUM_BINS):
            x = x / float(NUM_BINS) * float(image_width)
            y = y / float(NUM_BINS) * float(image_height)
        points.append([
            min(max(0.0, x), float(image_width)),
            min(max(0.0, y), float(image_height)),
        ])
    return points


def _bbox_from_points(
    points: list[list[float]],
    *,
    image_width: int,
    image_height: int,
) -> list[float] | None:
    if not points:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    if x2 <= x1:
        x1 -= 1.0
        x2 += 1.0
    if y2 <= y1:
        y1 -= 1.0
        y2 += 1.0
    x1 = min(max(0.0, x1), float(image_width))
    y1 = min(max(0.0, y1), float(image_height))
    x2 = min(max(0.0, x2), float(image_width))
    y2 = min(max(0.0, y2), float(image_height))
    if x2 <= x1:
        x2 = min(float(image_width), x1 + 1.0)
    if y2 <= y1:
        y2 = min(float(image_height), y1 + 1.0)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _dedupe_instances(
    instances: list[PredictionInstance],
    *,
    iou_threshold: float,
) -> list[PredictionInstance]:
    kept: list[PredictionInstance] = []
    for instance in instances:
        duplicate = False
        for previous in kept:
            if previous.label == instance.label and _bbox_iou(previous.bbox, instance.bbox) >= iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(instance)
    return kept


def _bbox_iou(box_a: list[float | int], box_b: list[float | int]) -> float:
    ax1, ay1, ax2, ay2 = [float(item) for item in box_a]
    bx1, by1, bx2, by2 = [float(item) for item in box_b]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    intersection = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0
