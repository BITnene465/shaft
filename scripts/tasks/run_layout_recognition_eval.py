#!/usr/bin/env python3
"""Run the task-local layout-recognition benchmark against a vLLM endpoint.

The inference phases intentionally never read ground truth. Ground truth is only
loaded by the explicit ``evaluate`` phase after the prediction payload is frozen.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import shutil
import sys
import time
from typing import Any, Iterable
from urllib import request

from PIL import Image

from shaft.codec import decode_with_codec, dequantize_qwen_bbox, dequantize_qwen_point
from shaft.data.prompt_source import load_prompt_source_pool
from shaft.prompting import ShaftPromptTemplate
from shaft.utils.qwen_pixel_budget import image_to_data_url_with_qwen_pixel_budget


LABELS = frozenset({"shape", "icon", "image", "line"})
RECONSTRUCTION_LABELS = frozenset({"shape", "line"})
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
POINT_KEYS = frozenset(
    {
        "point",
        "start",
        "mid",
        "end",
        "top_left",
        "top_right",
        "bottom_left",
        "bottom_right",
    }
)
POINT_COLLECTION_KEYS = frozenset(
    {"points", "corners", "body_corners", "split_corners", "stops"}
)
BBOX_KEYS = frozenset({"bbox", "bbox_2d", "body_bbox"})
SHAPE_TYPES = frozenset(
    {
        "rectangle",
        "oval",
        "triangle",
        "trapezoid",
        "parallelogram",
        "diamond",
        "step",
        "regular_pentagon",
        "regular_hexagon",
        "arrow_pentagon",
        "other_polygon",
        "callout",
        "card",
        "other",
    }
)
SHAPE_CORNER_COUNTS = {
    "rectangle": 4,
    "triangle": 3,
    "trapezoid": 4,
    "parallelogram": 4,
    "diamond": 4,
    "step": 6,
    "regular_pentagon": 5,
    "regular_hexagon": 6,
    "arrow_pentagon": 5,
}
LINE_TYPES = frozenset({"straight", "curved"})
LINE_STYLES = frozenset({"path", "shape"})
ARROW_TYPES = frozenset({"none", "line", "stealth", "triangle", "pointy", "tee", "circle"})
HEX_COLOR_PATTERN = re.compile(r"^#[0-9A-Fa-f]{6}$")


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main_prompt(path: Path) -> ShaftPromptTemplate:
    pool = load_prompt_source_pool(path)
    formulation_id = "reconstruction" if pool.explicit_formulations else "default"
    formulation = next(
        (
            item
            for item in pool.formulations
            if item.formulation_id == formulation_id
        ),
        None,
    )
    if formulation is None:
        raise ValueError(
            f"Prompt pool has no {formulation_id!r} formulation: {path}"
        )
    for variant_id in ("main", "detailed"):
        for prompt in formulation.prompt_variants:
            if prompt.variant_id == variant_id:
                return prompt
    raise ValueError(
        f"Prompt formulation {formulation_id!r} has no main/detailed variant: {path}"
    )


def collect_images(image_dir: Path) -> list[Path]:
    images = [
        path
        for path in image_dir.iterdir()
        if path.is_file() and not path.name.startswith(".") and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    by_stem: dict[str, Path] = {}
    for path in images:
        if path.stem in by_stem:
            raise ValueError(f"Duplicate image stem: {path.stem}")
        by_stem[path.stem] = path
    return [by_stem[stem] for stem in sorted(by_stem)]


def select_images(image_dir: Path, include_stems: str | None = None) -> list[Path]:
    images = collect_images(image_dir)
    if not include_stems:
        return images
    requested = {stem.strip() for stem in include_stems.split(",") if stem.strip()}
    selected = [path for path in images if path.stem in requested]
    missing = sorted(requested - {path.stem for path in selected})
    if missing:
        raise ValueError(f"Requested image stems not found: {', '.join(missing)}")
    return selected


def call_vllm(
    *,
    endpoint: str,
    served_model: str,
    image_url: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    payload = {
        "model": served_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ],
        "max_tokens": int(max_tokens),
        "temperature": 0.0,
        "top_p": 1.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    url = endpoint.rstrip("/")
    if not url.endswith("/chat/completions"):
        url = f"{url}/v1/chat/completions" if not url.endswith("/v1") else f"{url}/chat/completions"
    started = time.perf_counter()
    http_request = request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with request.urlopen(http_request, timeout=float(timeout_seconds)) as response:
        response_payload = json.loads(response.read().decode("utf-8"))
    choices = response_payload.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        raise ValueError("vLLM response has no choices[0]")
    choice = choices[0]
    finish_reason = choice.get("finish_reason")
    if finish_reason != "stop":
        raise ValueError(f"vLLM response did not stop cleanly: {finish_reason!r}")
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    content = message.get("content", "")
    if isinstance(content, list):
        content = "".join(
            str(item.get("text", "")) for item in content if isinstance(item, dict)
        )
    return {
        "content": str(content).strip(),
        "finish_reason": finish_reason,
        "usage": response_payload.get("usage"),
        "latency_seconds": round(time.perf_counter() - started, 6),
        "response_id": response_payload.get("id"),
    }


def parse_detection(content: str, *, width: int, height: int) -> list[dict[str, Any]]:
    decoded = decode_with_codec("json_any", content)
    if not decoded.valid or decoded.partial:
        raise ValueError(f"Detection JSON is not complete: {decoded.error}")
    payload = decoded.parsed
    if isinstance(payload, dict):
        payload = next(
            (payload[key] for key in ("layout", "predictions", "detections") if isinstance(payload.get(key), list)),
            None,
        )
    if not isinstance(payload, list):
        raise ValueError("Detection output must be a JSON array")
    result: list[dict[str, Any]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Detection item {index} is not an object")
        label = str(item.get("label") or item.get("type") or "").strip().lower()
        if label not in LABELS:
            raise ValueError(f"Detection item {index} has invalid label {label!r}")
        bbox = item.get("bbox_2d", item.get("bbox"))
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError(f"Detection item {index} has invalid bbox")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in bbox):
            raise ValueError(f"Detection item {index} bbox is not numeric")
        if any(not math.isfinite(float(value)) for value in bbox):
            raise ValueError(f"Detection item {index} bbox is not finite")
        normalized = [min(999.0, max(0.0, float(value))) for value in bbox]
        pixel = dequantize_qwen_bbox(normalized, width=width, height=height)
        x1, y1, x2, y2 = [int(round(value)) for value in pixel]
        x1, x2 = sorted((min(width, max(0, x1)), min(width, max(0, x2))))
        y1, y2 = sorted((min(height, max(0, y1)), min(height, max(0, y2))))
        if x2 <= x1:
            x2 = min(width, x1 + 1)
            x1 = max(0, x2 - 1)
        if y2 <= y1:
            y2 = min(height, y1 + 1)
            y1 = max(0, y2 - 1)
        result.append({"type": label, "bbox": [x1, y1, x2, y2], "parameters": {}})
    return result


def bounded_interval(center: float, length: int, *, limit: int) -> tuple[int, int]:
    length = max(1, min(int(length), int(limit)))
    if length >= limit:
        return 0, int(limit)
    start = int(math.floor(center - length / 2.0))
    stop = start + length
    if start < 0:
        return 0, length
    if stop > limit:
        return limit - length, limit
    return start, stop


def context_crop_box(
    bbox: list[int],
    *,
    image_width: int,
    image_height: int,
    padding_ratio: float = 0.65,
    minimum_size: int = 256,
    max_aspect_ratio: float = 4.0,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    width, height = max(1.0, x2 - x1), max(1.0, y2 - y1)
    left = max(0, int(math.floor(x1 - width * padding_ratio)))
    top = max(0, int(math.floor(y1 - height * padding_ratio)))
    right = min(image_width, int(math.ceil(x2 + width * padding_ratio)))
    bottom = min(image_height, int(math.ceil(y2 + height * padding_ratio)))
    crop_width, crop_height = right - left, bottom - top
    if crop_width < minimum_size:
        left, right = bounded_interval((left + right) / 2.0, minimum_size, limit=image_width)
    if crop_height < minimum_size:
        top, bottom = bounded_interval((top + bottom) / 2.0, minimum_size, limit=image_height)
    crop_width, crop_height = right - left, bottom - top
    if crop_width / crop_height > max_aspect_ratio:
        target = min(image_height, math.ceil(crop_width / max_aspect_ratio))
        top, bottom = bounded_interval((top + bottom) / 2.0, target, limit=image_height)
    elif crop_height / crop_width > max_aspect_ratio:
        target = min(image_width, math.ceil(crop_height / max_aspect_ratio))
        left, right = bounded_interval((left + right) / 2.0, target, limit=image_width)
    if not (0 <= left < right <= image_width and 0 <= top < bottom <= image_height):
        raise ValueError(f"Invalid context crop: {(left, top, right, bottom)}")
    return left, top, right, bottom


def quantize_bbox_in_crop(bbox: list[int], crop_box: tuple[int, int, int, int]) -> list[int]:
    left, top, right, bottom = crop_box
    crop_width, crop_height = right - left, bottom - top
    values = [bbox[0] - left, bbox[1] - top, bbox[2] - left, bbox[3] - top]
    result = [
        round(values[0] / crop_width * 999),
        round(values[1] / crop_height * 999),
        round(values[2] / crop_width * 999),
        round(values[3] / crop_height * 999),
    ]
    return [min(999, max(0, int(value))) for value in result]


def _is_coordinate(value: Any, *, maximum: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= maximum


def _point_issues(value: Any, *, path: str, bounds: tuple[int, int]) -> list[str]:
    if not (
        isinstance(value, list)
        and len(value) == 2
        and _is_coordinate(value[0], maximum=bounds[0])
        and _is_coordinate(value[1], maximum=bounds[1])
    ):
        return [f"{path}: expected integer [x,y] within {bounds}"]
    return []


def _bbox_issues(value: Any, *, path: str, bounds: tuple[int, int]) -> list[str]:
    if not isinstance(value, list) or len(value) != 4:
        return [f"{path}: expected four-coordinate xyxy bbox"]
    if not all(
        _is_coordinate(item, maximum=bounds[index % 2]) for index, item in enumerate(value)
    ):
        return [f"{path}: expected integer xyxy bbox within {bounds}"]
    if value[2] < value[0] or value[3] < value[1]:
        return [f"{path}: bbox coordinates must be ordered"]
    return []


def _corner_issues(value: Any, *, path: str, bounds: tuple[int, int]) -> list[str]:
    if not isinstance(value, dict):
        return [f"{path}: corner must be an object"]
    corner_type = value.get("type")
    if corner_type == "sharp":
        return _point_issues(value.get("point"), path=f"{path}.point", bounds=bounds)
    if corner_type == "round":
        issues: list[str] = []
        for key in ("start", "mid", "end"):
            issues.extend(_point_issues(value.get(key), path=f"{path}.{key}", bounds=bounds))
        return issues
    return [f"{path}.type: expected sharp or round"]


def _corner_list_issues(
    value: Any,
    *,
    path: str,
    bounds: tuple[int, int],
    expected_count: int | None = None,
    minimum_count: int = 3,
) -> list[str]:
    if not isinstance(value, list):
        return [f"{path}: expected a corner array"]
    issues: list[str] = []
    if expected_count is not None and len(value) != expected_count:
        issues.append(f"{path}: expected {expected_count} corners, got {len(value)}")
    elif expected_count is None and len(value) < minimum_count:
        issues.append(f"{path}: expected at least {minimum_count} corners, got {len(value)}")
    for index, corner in enumerate(value):
        issues.extend(_corner_issues(corner, path=f"{path}[{index}]", bounds=bounds))
    return issues


def _border_issues(value: Any, *, path: str) -> list[str]:
    if not isinstance(value, dict):
        return [f"{path}: border must be an object"]
    border_type = value.get("type")
    if border_type not in {"none", "uniform", "complex"}:
        return [f"{path}.type: expected none, uniform, or complex"]
    if border_type == "uniform":
        issues = []
        if value.get("style") not in {"solid", "dash"}:
            issues.append(f"{path}.style: expected solid or dash")
        if not isinstance(value.get("color"), str) or not HEX_COLOR_PATTERN.fullmatch(
            value["color"]
        ):
            issues.append(f"{path}.color: expected #RRGGBB")
        return issues
    return []


def _fill_issues(value: Any, *, path: str) -> list[str]:
    if not isinstance(value, dict):
        return [f"{path}: fill must be an object"]
    fill_type = value.get("type")
    if fill_type not in {"uniform", "complex"}:
        return [f"{path}.type: expected uniform or complex"]
    if fill_type == "uniform" and (
        not isinstance(value.get("color"), str)
        or not HEX_COLOR_PATTERN.fullmatch(value["color"])
    ):
        return [f"{path}.color: expected #RRGGBB"]
    return []


def _shape_contract_issues(
    parameters: dict[str, Any], *, bounds: tuple[int, int]
) -> list[str]:
    shape_type = parameters.get("shape_type")
    if shape_type not in SHAPE_TYPES:
        return [f"parameters.shape_type: unsupported value {shape_type!r}"]
    if shape_type == "other":
        return (
            []
            if set(parameters) == {"shape_type"}
            else ["parameters: shape_type other must not contain additional attributes"]
        )

    issues = _border_issues(parameters.get("border"), path="parameters.border")
    effect = parameters.get("effect")
    if not isinstance(effect, dict) or effect.get("type") not in {"none", "exist"}:
        issues.append("parameters.effect.type: expected none or exist")

    if shape_type == "card":
        issues.extend(
            _corner_list_issues(
                parameters.get("corners"),
                path="parameters.corners",
                bounds=bounds,
                expected_count=4,
            )
        )
        fills = parameters.get("fill")
        splits = parameters.get("splits")
        if not isinstance(fills, list) or not fills:
            issues.append("parameters.fill: card requires a non-empty fill array")
            fills = []
        else:
            for index, fill in enumerate(fills):
                issues.extend(_fill_issues(fill, path=f"parameters.fill[{index}]"))
        if not isinstance(splits, list):
            issues.append("parameters.splits: card requires a split array")
            splits = []
        split_axes: set[str] = set()
        for index, split in enumerate(splits):
            path = f"parameters.splits[{index}]"
            issues.extend(_border_issues(split, path=path))
            corners = split.get("split_corners") if isinstance(split, dict) else None
            issues.extend(
                _corner_list_issues(
                    corners,
                    path=f"{path}.split_corners",
                    bounds=bounds,
                    expected_count=2,
                    minimum_count=2,
                )
            )
            if isinstance(corners, list) and len(corners) == 2:
                points = [corner.get("point") for corner in corners if isinstance(corner, dict)]
                if len(points) == 2 and all(isinstance(point, list) for point in points):
                    dx = abs(points[1][0] - points[0][0])
                    dy = abs(points[1][1] - points[0][1])
                    split_axes.add("horizontal" if dx >= dy else "vertical")
        if fills and len(fills) != len(splits) + 1:
            issues.append(
                "parameters.fill: card fill count must equal split count plus one"
            )
        if len(split_axes) > 1:
            issues.append("parameters.splits: card splits must share one orientation")
        return issues

    issues.extend(_fill_issues(parameters.get("fill"), path="parameters.fill"))
    if shape_type == "oval":
        if "corners" in parameters:
            issues.append("parameters.corners: oval must not contain corners")
        return issues
    if shape_type == "callout":
        body_type = parameters.get("body_type")
        if body_type == "rectangle":
            issues.extend(
                _corner_list_issues(
                    parameters.get("body_corners"),
                    path="parameters.body_corners",
                    bounds=bounds,
                    expected_count=4,
                )
            )
        elif body_type == "oval":
            issues.extend(
                _bbox_issues(
                    parameters.get("body_bbox"),
                    path="parameters.body_bbox",
                    bounds=bounds,
                )
            )
        else:
            issues.append("parameters.body_type: expected rectangle or oval")
        tail = parameters.get("tail")
        points = tail.get("points") if isinstance(tail, dict) else None
        if not isinstance(points, list) or len(points) != 3:
            issues.append("parameters.tail.points: expected three points")
        else:
            for index, point in enumerate(points):
                issues.extend(
                    _point_issues(
                        point,
                        path=f"parameters.tail.points[{index}]",
                        bounds=bounds,
                    )
                )
        return issues

    expected_count = SHAPE_CORNER_COUNTS.get(shape_type)
    issues.extend(
        _corner_list_issues(
            parameters.get("corners"),
            path="parameters.corners",
            bounds=bounds,
            expected_count=expected_count,
        )
    )
    return issues


def _line_contract_issues(
    parameters: dict[str, Any], *, bounds: tuple[int, int]
) -> list[str]:
    issues: list[str] = []
    line_type = parameters.get("line_type")
    line_style = parameters.get("line_style")
    if line_type not in LINE_TYPES:
        issues.append("parameters.line_type: expected straight or curved")
    if line_style not in LINE_STYLES:
        issues.append("parameters.line_style: expected path or shape")
    if parameters.get("dash_style") not in {"solid", "dash"}:
        issues.append("parameters.dash_style: expected solid or dash")
    for key in ("begin_arrow", "end_arrow"):
        arrow = parameters.get(key)
        if arrow not in ARROW_TYPES:
            issues.append(f"parameters.{key}: unsupported arrow type {arrow!r}")
        elif arrow == "line" and line_style != "path":
            issues.append(f"parameters.{key}: line marker is valid only for path style")
        elif arrow == "pointy" and line_style != "shape":
            issues.append(f"parameters.{key}: pointy marker is valid only for shape style")
    if not isinstance(parameters.get("is_single"), bool):
        issues.append("parameters.is_single: expected boolean")

    points = parameters.get("points")
    if not isinstance(points, list) or not points:
        issues.append("parameters.points: expected a non-empty segment array")
        points = []
    for segment_index, segment in enumerate(points):
        path = f"parameters.points[{segment_index}]"
        if not isinstance(segment, list) or len(segment) < 2:
            issues.append(f"{path}: expected at least two points")
            continue
        if line_type == "curved" and len(segment) != 4:
            issues.append(f"{path}: curved segment requires exactly four points")
        for point_index, point in enumerate(segment):
            issues.extend(
                _point_issues(
                    point,
                    path=f"{path}[{point_index}]",
                    bounds=bounds,
                )
            )
    if isinstance(parameters.get("is_single"), bool) and points:
        if parameters["is_single"] != (len(points) == 1):
            issues.append("parameters.is_single: must agree with the number of segments")
    corner_style = parameters.get("corner_style")
    if corner_style is not None:
        if corner_style not in {"sharp", "round"}:
            issues.append("parameters.corner_style: expected sharp or round")
        elif line_type != "straight" or not any(
            isinstance(segment, list) and len(segment) > 2 for segment in points
        ):
            issues.append("parameters.corner_style: valid only for a bent straight line")
    issues.extend(_fill_issues(parameters.get("fill"), path="parameters.fill"))
    border = parameters.get("border")
    issues.extend(_border_issues(border, path="parameters.border"))
    if isinstance(border, dict) and border.get("type") == "complex":
        issues.append("parameters.border.type: line border does not support complex")
    return issues


def reconstruction_contract_issues(
    parameters: dict[str, Any],
    *,
    expected_label: str,
    coordinate_bounds: tuple[int, int] = (999, 999),
) -> list[str]:
    if expected_label == "shape":
        return _shape_contract_issues(parameters, bounds=coordinate_bounds)
    if expected_label == "line":
        return _line_contract_issues(parameters, bounds=coordinate_bounds)
    return [f"unsupported reconstruction label {expected_label!r}"]


def parse_reconstruction(content: str, *, expected_label: str) -> dict[str, Any]:
    decoded = decode_with_codec("json_any", content)
    if not decoded.valid or decoded.partial:
        raise ValueError(f"Reconstruction JSON is not complete: {decoded.error}")
    payload = decoded.parsed
    if isinstance(payload, list) and len(payload) == 1:
        payload = payload[0]
    if not isinstance(payload, dict):
        raise ValueError("Reconstruction output must be one JSON object")
    label = str(payload.get("type") or payload.get("label") or "").strip().lower()
    if label != expected_label:
        raise ValueError(f"Expected reconstruction label {expected_label!r}, got {label!r}")
    parameters = payload.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("Reconstruction parameters must be an object")
    issues = reconstruction_contract_issues(parameters, expected_label=expected_label)
    if issues:
        raise ValueError("Reconstruction contract violations: " + "; ".join(issues))
    return parameters


def map_crop_point(
    value: list[Any],
    *,
    crop_box: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> list[int]:
    left, top, right, bottom = crop_box
    local = dequantize_qwen_point(value, width=right - left, height=bottom - top)
    return [
        min(image_width, max(0, int(round(left + local[0])))),
        min(image_height, max(0, int(round(top + local[1])))),
    ]


def map_crop_bbox(
    value: list[Any],
    *,
    crop_box: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> list[int]:
    left, top, right, bottom = crop_box
    local = dequantize_qwen_bbox(value, width=right - left, height=bottom - top)
    return [
        min(image_width, max(0, int(round(left + local[0])))),
        min(image_height, max(0, int(round(top + local[1])))),
        min(image_width, max(0, int(round(left + local[2])))),
        min(image_height, max(0, int(round(top + local[3])))),
    ]


def convert_geometry(
    value: Any,
    *,
    key: str,
    crop_box: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> Any:
    if isinstance(value, dict):
        return {
            child_key: convert_geometry(
                child,
                key=str(child_key),
                crop_box=crop_box,
                image_width=image_width,
                image_height=image_height,
            )
            for child_key, child in value.items()
        }
    if isinstance(value, list):
        numeric = all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value)
        if numeric and len(value) == 2 and (
            key in POINT_KEYS or key in POINT_COLLECTION_KEYS
        ):
            return map_crop_point(
                value,
                crop_box=crop_box,
                image_width=image_width,
                image_height=image_height,
            )
        if numeric and len(value) == 4 and (key in BBOX_KEYS or key.endswith("_bbox")):
            return map_crop_bbox(
                value,
                crop_box=crop_box,
                image_width=image_width,
                image_height=image_height,
            )
        return [
            convert_geometry(
                child,
                key=key,
                crop_box=crop_box,
                image_width=image_width,
                image_height=image_height,
            )
            for child in value
        ]
    return value


def flatten_line_style(parameters: dict[str, Any]) -> dict[str, Any]:
    result = dict(parameters)
    fill = result.pop("fill", None)
    border = result.pop("border", None)
    if isinstance(fill, dict):
        fill_type = str(fill.get("type") or "").strip().lower()
        result["fill_color"] = fill.get("color", "") if fill_type == "uniform" else fill_type
    if isinstance(border, dict):
        border_type = str(border.get("type") or "").strip().lower()
        if border_type == "none":
            result.update({"has_border": False, "border_style": "", "border_color": ""})
        elif border_type == "uniform":
            result.update(
                {
                    "has_border": True,
                    "border_style": border.get("style", ""),
                    "border_color": border.get("color", ""),
                }
            )
        elif border_type:
            result.update(
                {"has_border": True, "border_style": border_type, "border_color": ""}
            )
    return result


def geometry_issues(value: Any, *, width: int, height: int, path: str = "parameters") -> list[str]:
    issues: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            issues.extend(geometry_issues(child, width=width, height=height, path=f"{path}.{key}"))
    elif isinstance(value, list):
        if len(value) == 2 and all(isinstance(item, int) for item in value):
            if not 0 <= value[0] <= width or not 0 <= value[1] <= height:
                issues.append(f"{path}:point_out_of_bounds")
        else:
            for index, child in enumerate(value):
                issues.extend(
                    geometry_issues(child, width=width, height=height, path=f"{path}[{index}]")
                )
    return issues


def run_parallel(items: list[Any], *, workers: int, function: Any) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        futures = {executor.submit(function, item): item for item in items}
        for future in as_completed(futures):
            results.append(future.result())
    return results


def select_endpoint(endpoints: list[str], request_id: str) -> str:
    if not endpoints:
        raise ValueError("At least one vLLM endpoint is required")
    digest = hashlib.sha256(str(request_id).encode("utf-8")).digest()
    return endpoints[int.from_bytes(digest[:8], "big") % len(endpoints)]


def artifact_state(
    expected_ids: set[str],
    *,
    artifact_dir: Path,
    error_dir: Path,
) -> tuple[set[str], set[str], set[str]]:
    actual_ids = {path.stem for path in artifact_dir.glob("*.json")}
    complete_ids = expected_ids & actual_ids
    unexpected_ids = actual_ids - expected_ids
    error_ids = {
        artifact_id
        for artifact_id in expected_ids
        if (error_dir / f"{artifact_id}.json").is_file()
    }
    return complete_ids, unexpected_ids, error_ids


def detect(args: argparse.Namespace) -> None:
    images = select_images(args.image_dir, args.include_stems)
    prompt = main_prompt(args.detection_prompt)
    raw_dir = args.work_dir / "detection" / "raw"
    pred_dir = args.work_dir / "detection" / "pred"
    error_dir = args.work_dir / "detection" / "errors"

    def run_one(image_path: Path) -> dict[str, Any]:
        prediction_path = pred_dir / f"{image_path.stem}.json"
        if prediction_path.is_file() and not args.force:
            return {"stem": image_path.stem, "status": "cached"}
        with Image.open(image_path) as image:
            width, height = image.size
        image_url, budget = image_to_data_url_with_qwen_pixel_budget(
            image_path,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
        )
        last_error: Exception | None = None
        for attempt in range(1, args.retries + 2):
            try:
                response_payload = call_vllm(
                    endpoint=select_endpoint(args.endpoints, image_path.stem),
                    served_model=args.served_model,
                    image_url=image_url,
                    system_prompt=prompt.system_prompt,
                    user_prompt=prompt.user_prompt,
                    max_tokens=args.max_tokens,
                    timeout_seconds=args.timeout_seconds,
                )
                layout = parse_detection(response_payload["content"], width=width, height=height)
                raw_payload = {
                    "sample_id": image_path.stem,
                    "image_path": str(image_path),
                    "source_size": [width, height],
                    "pixel_budget": budget.to_dict(),
                    "prompt_id": prompt.prompt_id,
                    "prompt_sha256": hashlib.sha256(
                        (prompt.system_prompt + "\n" + prompt.user_prompt).encode("utf-8")
                    ).hexdigest(),
                    "attempt": attempt,
                    **response_payload,
                }
                atomic_write_json(raw_dir / f"{image_path.stem}.json", raw_payload)
                atomic_write_json(
                    prediction_path,
                    {"size": [width, height], "layout": layout},
                )
                stale_error = error_dir / f"{image_path.stem}.json"
                if stale_error.is_file():
                    stale_error.unlink()
                return {
                    "stem": image_path.stem,
                    "status": "ok",
                    "detections": len(layout),
                    "finish_reason": response_payload["finish_reason"],
                }
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt <= args.retries:
                    time.sleep(min(30.0, 2.0**attempt))
        atomic_write_json(
            error_dir / f"{image_path.stem}.json",
            {"sample_id": image_path.stem, "error": repr(last_error)},
        )
        return {"stem": image_path.stem, "status": "error", "error": repr(last_error)}

    run_parallel(images, workers=args.workers, function=run_one)
    expected_stems = {path.stem for path in images}
    complete, unexpected, errors = artifact_state(
        expected_stems,
        artifact_dir=pred_dir,
        error_dir=error_dir,
    )
    summary = {
        "phase": "detection",
        "expected": len(images),
        "complete": len(complete),
        "errors": len(errors),
        "unexpected_predictions": len(unexpected),
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "max_tokens": args.max_tokens,
        "prompt_id": prompt.prompt_id,
        "gt_read": False,
    }
    atomic_write_json(args.work_dir / "detection" / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))
    if (
        summary["complete"] != summary["expected"]
        or summary["errors"]
        or summary["unexpected_predictions"]
    ):
        raise SystemExit(1)


def prepare_reconstruction(args: argparse.Namespace) -> None:
    images = {path.stem: path for path in collect_images(args.image_dir)}
    pred_dir = args.work_dir / "detection" / "pred"
    crop_dir = args.work_dir / "reconstruction" / "crops"
    rows: list[dict[str, Any]] = []
    for stem, image_path in images.items():
        prediction_path = pred_dir / f"{stem}.json"
        if not prediction_path.is_file():
            raise FileNotFoundError(prediction_path)
        prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
        width, height = prediction["size"]
        with Image.open(image_path) as image:
            for index, element in enumerate(prediction.get("layout", [])):
                label = element.get("type")
                if label not in RECONSTRUCTION_LABELS:
                    continue
                crop_box = context_crop_box(
                    element["bbox"],
                    image_width=width,
                    image_height=height,
                    padding_ratio=args.padding_ratio,
                    minimum_size=args.minimum_crop_size,
                )
                request_id = f"{stem}__det_{index:04d}_{label}"
                crop_path = crop_dir / f"{request_id}.png"
                crop_path.parent.mkdir(parents=True, exist_ok=True)
                if not crop_path.is_file() or args.force:
                    crop = image.crop(crop_box)
                    try:
                        crop.save(crop_path, format="PNG")
                    finally:
                        crop.close()
                rows.append(
                    {
                        "request_id": request_id,
                        "sample_id": stem,
                        "detection_index": index,
                        "label": label,
                        "image_path": str(image_path),
                        "crop_path": str(crop_path),
                        "image_size": [width, height],
                        "detection_bbox": list(element["bbox"]),
                        "proposal_bbox_full": list(element["bbox"]),
                        "crop_box": list(crop_box),
                        "proposal_bbox_2d": quantize_bbox_in_crop(element["bbox"], crop_box),
                        "proposal_source": "detection",
                        "gt_read": False,
                    }
                )
    rows.sort(key=lambda row: row["request_id"])
    manifest_path = args.work_dir / "reconstruction" / "manifest.jsonl"
    atomic_write_jsonl(manifest_path, rows)
    summary = {
        "phase": "prepare_reconstruction",
        "requests": len(rows),
        "shape": sum(row["label"] == "shape" for row in rows),
        "line": sum(row["label"] == "line" for row in rows),
        "proposal_source": "detection",
        "gt_read": False,
    }
    atomic_write_json(args.work_dir / "reconstruction" / "prepare_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Expected object at {path}:{line_number}")
                rows.append(row)
    return rows


def reconstruct(args: argparse.Namespace) -> None:
    rows = load_jsonl(args.work_dir / "reconstruction" / "manifest.jsonl")
    prompts = {
        "shape": main_prompt(args.shape_prompt),
        "line": main_prompt(args.line_prompt),
    }
    raw_dir = args.work_dir / "reconstruction" / "raw"
    parsed_dir = args.work_dir / "reconstruction" / "parsed"
    error_dir = args.work_dir / "reconstruction" / "errors"

    def run_one(row: dict[str, Any]) -> dict[str, Any]:
        request_id = str(row["request_id"])
        parsed_path = parsed_dir / f"{request_id}.json"
        if parsed_path.is_file() and not args.force:
            return {"request_id": request_id, "status": "cached"}
        label = str(row["label"])
        prompt = prompts[label]
        user_prompt = prompt.render({"proposal_bbox_2d": row["proposal_bbox_2d"]})
        crop_path = Path(str(row["crop_path"]))
        image_url, budget = image_to_data_url_with_qwen_pixel_budget(
            crop_path,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
        )
        last_error: Exception | None = None
        for attempt in range(1, args.retries + 2):
            try:
                response_payload = call_vllm(
                    endpoint=select_endpoint(args.endpoints, request_id),
                    served_model=args.served_model,
                    image_url=image_url,
                    system_prompt=prompt.system_prompt,
                    user_prompt=user_prompt,
                    max_tokens=args.max_tokens,
                    timeout_seconds=args.timeout_seconds,
                )
                raw_parameters = parse_reconstruction(
                    response_payload["content"], expected_label=label
                )
                width, height = row["image_size"]
                crop_box = tuple(int(value) for value in row["crop_box"])
                parameters = convert_geometry(
                    raw_parameters,
                    key="parameters",
                    crop_box=crop_box,
                    image_width=width,
                    image_height=height,
                )
                if label == "line":
                    parameters = flatten_line_style(parameters)
                issues = geometry_issues(parameters, width=width, height=height)
                atomic_write_json(
                    raw_dir / f"{request_id}.json",
                    {
                        **row,
                        "pixel_budget": budget.to_dict(),
                        "prompt_id": prompt.prompt_id,
                        "prompt_args": {"proposal_bbox_2d": row["proposal_bbox_2d"]},
                        "attempt": attempt,
                        **response_payload,
                    },
                )
                atomic_write_json(
                    parsed_path,
                    {
                        "request_id": request_id,
                        "sample_id": row["sample_id"],
                        "detection_index": row["detection_index"],
                        "label": label,
                        "proposal_bbox_full": row["proposal_bbox_full"],
                        "proposal_source": "detection",
                        "gt_read": False,
                        "parameters": parameters,
                        "contract_issues": issues,
                    },
                )
                stale_error = error_dir / f"{request_id}.json"
                if stale_error.is_file():
                    stale_error.unlink()
                return {
                    "request_id": request_id,
                    "status": "ok",
                    "contract_issues": len(issues),
                }
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt <= args.retries:
                    time.sleep(min(30.0, 2.0**attempt))
        atomic_write_json(error_dir / f"{request_id}.json", {**row, "error": repr(last_error)})
        return {"request_id": request_id, "status": "error", "error": repr(last_error)}

    run_parallel(rows, workers=args.workers, function=run_one)
    expected_ids = {str(row["request_id"]) for row in rows}
    complete_ids, unexpected_ids, error_ids = artifact_state(
        expected_ids,
        artifact_dir=parsed_dir,
        error_dir=error_dir,
    )
    contract_issue_records = sum(
        bool(json.loads((parsed_dir / f"{request_id}.json").read_text(encoding="utf-8")).get("contract_issues"))
        for request_id in complete_ids
    )
    summary = {
        "phase": "reconstruction",
        "expected": len(rows),
        "complete": len(complete_ids),
        "errors": len(error_ids),
        "unexpected_predictions": len(unexpected_ids),
        "contract_issue_records": contract_issue_records,
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "max_tokens": args.max_tokens,
        "proposal_source": "detection",
        "gt_read": False,
    }
    atomic_write_json(args.work_dir / "reconstruction" / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))
    if (
        summary["complete"] != summary["expected"]
        or summary["errors"]
        or summary["unexpected_predictions"]
    ):
        raise SystemExit(1)


def merge(args: argparse.Namespace) -> None:
    images = collect_images(args.image_dir)
    detection_dir = args.work_dir / "detection" / "pred"
    parsed_dir = args.work_dir / "reconstruction" / "parsed"
    final_dir = args.work_dir / "final" / args.dataset_name / "pred"
    installed = 0
    contract_issue_records = 0
    for image_path in images:
        detection = json.loads(
            (detection_dir / f"{image_path.stem}.json").read_text(encoding="utf-8")
        )
        for index, element in enumerate(detection.get("layout", [])):
            label = element.get("type")
            if label not in RECONSTRUCTION_LABELS:
                continue
            request_id = f"{image_path.stem}__det_{index:04d}_{label}"
            parsed_path = parsed_dir / f"{request_id}.json"
            if not parsed_path.is_file():
                raise FileNotFoundError(parsed_path)
            parsed = json.loads(parsed_path.read_text(encoding="utf-8"))
            if parsed.get("proposal_source") != "detection" or parsed.get("gt_read") is not False:
                raise ValueError(f"Invalid reconstruction provenance: {request_id}")
            if parsed.get("detection_index") != index or parsed.get("label") != label:
                raise ValueError(f"Reconstruction identity mismatch: {request_id}")
            if parsed.get("proposal_bbox_full") != element.get("bbox"):
                raise ValueError(f"Reconstruction proposal mismatch: {request_id}")
            element["parameters"] = parsed["parameters"]
            installed += 1
            contract_issue_records += bool(parsed.get("contract_issues"))
        atomic_write_json(final_dir / f"{image_path.stem}.json", detection)
    summary = {
        "phase": "merge",
        "dataset_name": args.dataset_name,
        "images": len(images),
        "installed_reconstructions": installed,
        "contract_issue_records": contract_issue_records,
        "proposal_source": "detection",
        "gt_read": False,
    }
    atomic_write_json(args.work_dir / "final" / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


def load_evaluator(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("layout_recognition_evaluator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def evaluate(args: argparse.Namespace) -> None:
    final_pred_dir = args.work_dir / "final" / args.dataset_name / "pred"
    expected = {path.stem for path in collect_images(args.image_dir)}
    actual = {path.stem for path in final_pred_dir.glob("*.json")}
    if actual != expected:
        raise ValueError(
            f"Prediction stem mismatch: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}"
        )
    internal_method = args.work_dir / "internal_eval" / args.run_name
    internal_pred = internal_method / args.dataset_name / "pred"
    if internal_method.exists():
        shutil.rmtree(internal_method)
    internal_pred.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(final_pred_dir, internal_pred)
    atomic_write_json(internal_method / "method.json", {"group": "VLM"})
    evaluator = load_evaluator(args.evaluator)
    score = evaluator.eval_method(internal_method)
    score["provenance"] = {
        "run_name": args.run_name,
        "dataset_name": args.dataset_name,
        "gt_revision": args.gt_revision,
        "gt_instance_count": score["overall"]["gt_count"],
        "evaluator_sha256": sha256_file(args.evaluator),
        "checkpoint": str(args.checkpoint),
        "checkpoint_index_sha256": sha256_file(args.checkpoint / "model.safetensors.index.json"),
        "detection_min_pixels": args.detection_min_pixels,
        "detection_max_pixels": args.detection_max_pixels,
        "reconstruction_proposal_source": "detection",
        "gt_read_during_inference": False,
    }
    atomic_write_json(args.work_dir / "internal_score.json", score)
    overall = score["overall"]
    print(
        json.dumps(
            {
                key: overall[key]
                for key in (
                    "image_count",
                    "gt_count",
                    "pred_count",
                    "precision",
                    "recall",
                    "f1",
                    "mean_iou",
                    "label_accuracy",
                    "parameter_precision",
                    "parameter_recall",
                    "parameter_f1",
                    "attribute_summary",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def package(args: argparse.Namespace) -> None:
    source = args.work_dir / "final" / args.dataset_name / "pred"
    destination = args.result_root / args.run_name / args.dataset_name / "pred"
    if destination.parent.exists():
        raise FileExistsError(destination.parent)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)
    summary = {
        "run_name": args.run_name,
        "dataset_name": args.dataset_name,
        "prediction_files": len(list(destination.glob("*.json"))),
        "payload_owner": "prediction_submitter",
        "excluded_files": ["method.json", "score.json", "methods.json"],
    }
    atomic_write_json(args.work_dir / "package_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


def add_common_runtime(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--endpoint",
        dest="endpoints",
        action="append",
        default=None,
        help="Repeat to distribute requests deterministically across vLLM replicas.",
    )
    parser.add_argument("--served-model", default="banana-v5.7-retrain-27b")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=2400.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--force", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    detection = subparsers.add_parser("detect")
    add_common_runtime(detection)
    detection.add_argument("--image-dir", type=Path, required=True)
    detection.add_argument("--detection-prompt", type=Path, required=True)
    detection.add_argument("--min-pixels", type=int, default=1_000_000)
    detection.add_argument("--max-pixels", type=int, default=4_000_000)
    detection.add_argument("--max-tokens", type=int, default=8_000)
    detection.add_argument(
        "--include-stems",
        help="Optional comma-separated image stems for a canary or partial run.",
    )
    detection.set_defaults(function=detect)

    prepare = subparsers.add_parser("prepare-reconstruction")
    prepare.add_argument("--work-dir", type=Path, required=True)
    prepare.add_argument("--image-dir", type=Path, required=True)
    prepare.add_argument("--padding-ratio", type=float, default=0.65)
    prepare.add_argument("--minimum-crop-size", type=int, default=256)
    prepare.add_argument("--force", action="store_true")
    prepare.set_defaults(function=prepare_reconstruction)

    reconstruction = subparsers.add_parser("reconstruct")
    add_common_runtime(reconstruction)
    reconstruction.add_argument("--shape-prompt", type=Path, required=True)
    reconstruction.add_argument("--line-prompt", type=Path, required=True)
    reconstruction.add_argument("--min-pixels", type=int, default=500_000)
    reconstruction.add_argument("--max-pixels", type=int, default=4_000_000)
    reconstruction.add_argument("--max-tokens", type=int, default=8_000)
    reconstruction.set_defaults(function=reconstruct)

    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--work-dir", type=Path, required=True)
    merge_parser.add_argument("--image-dir", type=Path, required=True)
    merge_parser.add_argument("--dataset-name", default="real_v1")
    merge_parser.set_defaults(function=merge)

    evaluation = subparsers.add_parser("evaluate")
    evaluation.add_argument("--work-dir", type=Path, required=True)
    evaluation.add_argument("--image-dir", type=Path, required=True)
    evaluation.add_argument("--evaluator", type=Path, required=True)
    evaluation.add_argument("--run-name", required=True)
    evaluation.add_argument("--dataset-name", default="real_v1")
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--gt-revision", required=True)
    evaluation.add_argument("--detection-min-pixels", type=int, default=1_000_000)
    evaluation.add_argument("--detection-max-pixels", type=int, default=4_000_000)
    evaluation.set_defaults(function=evaluate)

    package_parser = subparsers.add_parser("package")
    package_parser.add_argument("--work-dir", type=Path, required=True)
    package_parser.add_argument("--result-root", type=Path, required=True)
    package_parser.add_argument("--run-name", required=True)
    package_parser.add_argument("--dataset-name", default="real_v1")
    package_parser.set_defaults(function=package)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if hasattr(args, "endpoints") and not args.endpoints:
        args.endpoints = ["http://127.0.0.1:18000"]
    args.function(args)


if __name__ == "__main__":
    main()
