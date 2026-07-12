from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from PIL import Image

from shaft.codec.base import ShaftCodecResult
from shaft.codec.coordinates import dequantize_qwen_point

from .visualization import (
    ShaftVisualBox,
    ShaftVisualLineStrip,
    ShaftVisualPoint,
    save_labeled_visualization,
)


def render_prediction_visualization(
    *,
    image_path: str,
    sample_id: str,
    sample_index: int,
    prediction: ShaftCodecResult,
    out_dir: Path,
) -> str | None:
    try:
        with Image.open(image_path) as image:
            image_width, image_height = image.size
    except Exception:
        return None

    payload = prediction.parsed
    boxes: list[ShaftVisualBox] = []
    points: list[ShaftVisualPoint] = []
    line_strips: list[ShaftVisualLineStrip] = []
    summary_parts: list[str] = []

    if isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                continue
            bbox = _coerce_bbox(item.get("bbox_2d"))
            if bbox is None:
                continue
            x1, y1 = _scale_qwen_point(bbox[0], bbox[1], image_width, image_height)
            x2, y2 = _scale_qwen_point(bbox[2], bbox[3], image_width, image_height)
            label = str(item.get("label", "")).strip().lower()
            boxes.append(
                ShaftVisualBox(
                    label=label,
                    bbox=(x1, y1, x2, y2),
                    index=len(boxes) + 1,
                )
            )
    elif isinstance(payload, dict):
        raw_segments = _extract_keypoint_segments(payload)
        if raw_segments is not None:
            point_index = 1
            for segment in raw_segments:
                keypoint_points: list[ShaftVisualPoint] = []
                for x, y in segment:
                    scaled_x, scaled_y = _scale_qwen_point(x, y, image_width, image_height)
                    keypoint_points.append(
                        ShaftVisualPoint(x=scaled_x, y=scaled_y, index=point_index)
                    )
                    point_index += 1
                if len(keypoint_points) >= 2:
                    line_strips.append(ShaftVisualLineStrip(points=tuple(keypoint_points)))
                else:
                    points.extend(keypoint_points)
        if payload.get("stroke_pattern") is not None:
            summary_parts.append(f"stroke={payload['stroke_pattern']}")
        if payload.get("geometry_style") is not None:
            summary_parts.append(f"geometry={payload['geometry_style']}")

    if not boxes and not points and not line_strips and not summary_parts:
        return None

    footer_lines = [f"id={sample_id} idx={sample_index:06d}"]
    if not prediction.valid:
        footer_lines.append("pred: invalid")
    elif summary_parts:
        footer_lines.append(f"pred: {' '.join(summary_parts)}")
    if prediction.error_type:
        footer_lines.append(f"error: {prediction.error_type}")
    if boxes:
        box_parts: list[str] = []
        for box in boxes:
            x1, y1, x2, y2 = [int(round(v)) for v in box.bbox]
            box_parts.append(f"{box.index}:{box.label or 'box'}[{x1},{y1},{x2},{y2}]")
        footer_lines.append("boxes: " + " ".join(box_parts))
    if points:
        point_parts = [
            f"{point.index}=({int(round(point.x))},{int(round(point.y))})" for point in points
        ]
        footer_lines.append("points: " + " ".join(point_parts))
    if line_strips:
        point_parts = []
        for strip in line_strips:
            for point in strip.points:
                point_parts.append(f"{point.index}=({int(round(point.x))},{int(round(point.y))})")
        footer_lines.append("points: " + " ".join(point_parts))

    output_dir = out_dir / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{_sanitize_filename(f'{sample_id}_{sample_index:06d}')}.jpg"
    return save_labeled_visualization(
        image_path=image_path,
        output_path=output_path,
        boxes=boxes,
        points=points,
        line_strips=line_strips,
        footer_lines=footer_lines,
    )


def _sanitize_filename(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]", "_", text.strip())
    return cleaned[:96] or "sample"


def _coerce_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, list | tuple) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if not (x2 > x1 and y2 > y1):
        return None
    return x1, y1, x2, y2


def _coerce_keypoints(value: Any) -> list[tuple[float, float]] | None:
    if not isinstance(value, list | tuple):
        return None
    points: list[tuple[float, float]] = []
    for item in value:
        if not isinstance(item, list | tuple) or len(item) != 2:
            continue
        try:
            points.append((float(item[0]), float(item[1])))
        except (TypeError, ValueError):
            continue
    return points or None


def _coerce_keypoint_segments(value: Any) -> list[list[tuple[float, float]]] | None:
    flat_points = _coerce_keypoints(value)
    if flat_points is not None:
        return [flat_points]
    if not isinstance(value, list | tuple):
        return None
    segments: list[list[tuple[float, float]]] = []
    for item in value:
        segment = _coerce_keypoints(item)
        if segment is None or len(segment) < 2:
            continue
        segments.append(segment)
    return segments or None


def _extract_keypoint_segments(payload: dict[str, Any]) -> list[list[tuple[float, float]]] | None:
    raw_points = payload.get("points_2d") or payload.get("keypoints_2d")
    if raw_points is None:
        parameters = payload.get("parameters")
        if isinstance(parameters, dict):
            raw_points = parameters.get("points")
    return _coerce_keypoint_segments(raw_points)


def _scale_qwen_point(x: float, y: float, width: int, height: int) -> tuple[float, float]:
    scaled_x, scaled_y = dequantize_qwen_point((x, y), width=width, height=height)
    return scaled_x, scaled_y
