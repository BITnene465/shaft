#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import random
import shutil
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from shaft.codec.coordinates import quantize_qwen_bbox, quantize_qwen_coordinate
from shaft.data.context_attribute_contract import validate_shape_parameters
from shaft.prompting import load_prompt_pool


PROPOSAL_NOISE_BUCKETS = (
    ("clean", 0.10, 0.00, 0.00, 0.00),
    ("accurate", 0.50, 0.04, 0.08, 0.02),
    ("moderate", 0.30, 0.10, 0.20, 0.05),
    ("hard", 0.10, 0.20, 0.35, 0.10),
)
CONTEXT_PADDING_BUCKETS = (
    ("tight", 0.20, 0.05, 0.30),
    ("medium", 0.50, 0.30, 1.00),
    ("large", 0.25, 1.00, 2.50),
    ("extreme", 0.05, 2.50, 4.00),
)
SYNTHETIC_PIXEL_PROFILE = "synthetic_realism_v1"
PIXEL_OPERATION_WEIGHTS = {
    "resample_roundtrip": 0.35,
    "gaussian_blur": 0.20,
    "gaussian_noise": 0.20,
    "jpeg_compression": 0.25,
}
PIXEL_OPERATION_ORDER = tuple(PIXEL_OPERATION_WEIGHTS)


@dataclass(frozen=True)
class TaskSourceSpec:
    selection_path: Path
    source_root: Path
    source_kind: str
    source_image_manifest: Path | None = None
    selection_limit: int | None = None


@dataclass(frozen=True)
class TaskSpec:
    name: str
    label: str
    selection_path: Path
    source_root: Path
    prompt_path: Path
    source_kind: str
    source_image_manifest: Path | None = None
    selection_limit: int | None = None
    additional_sources: tuple[TaskSourceSpec, ...] = ()


@dataclass(frozen=True)
class Selection:
    sample_id: str
    stem: str
    instance_index: int
    source_bbox: tuple[float, float, float, float]
    source_image: str
    source_json: str
    parameters: dict[str, Any] | None = None
    weak_label: dict[str, Any] | None = None
    archive_provenance: dict[str, Any] | None = None


@dataclass(frozen=True)
class ContextView:
    proposal_bbox: tuple[float, float, float, float]
    crop_box: tuple[int, int, int, int]
    proposal_noise_bucket: str
    proposal_noise: tuple[float, float, float, float, float, float]
    padding_scale_bucket: str
    requested_padding_ratios: tuple[float, float, float, float]
    proposal_iou: float
    gt_coverage: float


@dataclass(frozen=True)
class WorkerConfig:
    spec: TaskSpec
    staging_root: Path
    prompt_pool_id: str
    output_schema: str | None
    seed: int
    min_crop_size: int
    max_aspect_ratio: float
    png_compress_level: int


@dataclass(frozen=True)
class WorkerResult:
    rows: tuple[tuple[str, str], ...]
    counts: dict[str, int]


def _is_synthetic_source(spec: TaskSpec) -> bool:
    return spec.source_kind in {"synthetic", "synthetic_point_multi"}


def _is_multi_segment_line_parameters(parameters: Any) -> bool:
    return (
        isinstance(parameters, dict)
        and parameters.get("is_single") is False
        and isinstance(parameters.get("points"), list)
        and len(parameters["points"]) > 1
    )


def _expanded_task_sources(spec: TaskSpec) -> tuple[TaskSpec, ...]:
    sources = [spec]
    for source in spec.additional_sources:
        sources.append(
            TaskSpec(
                name=spec.name,
                label=spec.label,
                selection_path=source.selection_path,
                source_root=source.source_root,
                prompt_path=spec.prompt_path,
                source_kind=source.source_kind,
                source_image_manifest=source.source_image_manifest,
                selection_limit=source.selection_limit,
            )
        )
    return tuple(sources)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent) as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
        temp_path = Path(f.name)
    os.replace(temp_path, path)


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _bbox(value: Any) -> tuple[float, float, float, float]:
    if not isinstance(value, list | tuple) or len(value) != 4:
        raise ValueError(f"Invalid bbox: {value!r}")
    if not all(_is_number(item) for item in value):
        raise ValueError(f"Invalid bbox: {value!r}")
    x1, y1, x2, y2 = [float(item) for item in value]
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Degenerate bbox: {value!r}")
    return x1, y1, x2, y2


def _clip_bbox(
    value: tuple[float, float, float, float],
    *,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = value
    x1 = min(max(x1, 0.0), float(image_width))
    y1 = min(max(y1, 0.0), float(image_height))
    x2 = min(max(x2, 0.0), float(image_width))
    y2 = min(max(y2, 0.0), float(image_height))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"BBox is outside image bounds: {value!r}")
    return x1, y1, x2, y2


def _bbox_intersection_area(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    return width * height


def _bbox_area(value: tuple[float, float, float, float]) -> float:
    return max(0.0, value[2] - value[0]) * max(0.0, value[3] - value[1])


def _bbox_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    intersection = _bbox_intersection_area(left, right)
    union = _bbox_area(left) + _bbox_area(right) - intersection
    return intersection / union if union > 0 else 0.0


def _bbox_coverage(
    target: tuple[float, float, float, float],
    container: tuple[float, float, float, float],
) -> float:
    area = _bbox_area(target)
    return _bbox_intersection_area(target, container) / area if area > 0 else 0.0


def _choose_bucket(
    rng: random.Random,
    buckets: tuple[tuple[str, float, float, float], ...]
    | tuple[tuple[str, float, float, float, float], ...],
) -> tuple[Any, ...]:
    draw = rng.random()
    cumulative = 0.0
    for bucket in buckets:
        cumulative += float(bucket[1])
        if draw < cumulative:
            return bucket
    return buckets[-1]


def _weighted_sample_without_replacement(
    rng: random.Random,
    weights: dict[str, float],
    *,
    count: int,
) -> list[str]:
    available = dict(weights)
    selected: list[str] = []
    for _ in range(min(count, len(available))):
        total = sum(available.values())
        draw = rng.random() * total
        cumulative = 0.0
        choice = next(iter(available))
        for name, weight in available.items():
            cumulative += weight
            if draw < cumulative:
                choice = name
                break
        selected.append(choice)
        del available[choice]
    return selected


def _sample_synthetic_pixel_augmentation(
    *,
    task: str,
    sample_id: str,
    seed: int,
    target_short_span: int,
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    rng = random.Random(f"{seed}:synthetic-pixel:{task}:{sample_id}")
    if target_short_span < 80:
        severity = "mild"
        stack_depth = 1
    elif target_short_span < 200:
        severity = str(
            _choose_bucket(rng, (("mild", 0.70, 0.0, 0.0), ("moderate", 0.30, 0.0, 0.0)))[0]
        )
        stack_depth = int(
            _choose_bucket(rng, ((1, 0.40, 0.0, 0.0), (2, 0.50, 0.0, 0.0), (3, 0.10, 0.0, 0.0)))[0]
        )
    else:
        severity = str(
            _choose_bucket(
                rng,
                (
                    ("mild", 0.45, 0.0, 0.0),
                    ("moderate", 0.40, 0.0, 0.0),
                    ("strong", 0.15, 0.0, 0.0),
                ),
            )[0]
        )
        depth_buckets = {
            "mild": ((1, 0.15, 0.0, 0.0), (2, 0.55, 0.0, 0.0), (3, 0.30, 0.0, 0.0)),
            "moderate": ((1, 0.20, 0.0, 0.0), (2, 0.60, 0.0, 0.0), (3, 0.20, 0.0, 0.0)),
            "strong": ((1, 0.45, 0.0, 0.0), (2, 0.55, 0.0, 0.0)),
        }
        stack_depth = int(_choose_bucket(rng, depth_buckets[severity])[0])

    selected = _weighted_sample_without_replacement(
        rng,
        PIXEL_OPERATION_WEIGHTS,
        count=stack_depth,
    )
    selected.sort(key=PIXEL_OPERATION_ORDER.index)
    short_edge = min(image_width, image_height)
    operations: list[dict[str, Any]] = []
    for name in selected:
        if name == "resample_roundtrip":
            ratio_ranges = {
                "mild": (0.82, 0.96),
                "moderate": (0.62, 0.85),
                "strong": (0.42, 0.68),
            }
            kernels = {
                "mild": ("BICUBIC", "LANCZOS"),
                "moderate": ("BILINEAR", "BICUBIC", "LANCZOS"),
                "strong": ("BILINEAR", "BICUBIC"),
            }
            low, high = ratio_ranges[severity]
            operations.append(
                {
                    "name": name,
                    "scale_down_ratio": round(rng.uniform(low, high), 6),
                    "down_kernel": rng.choice(kernels[severity]),
                    "up_kernel": rng.choice(kernels[severity]),
                }
            )
        elif name == "gaussian_blur":
            ranges = {
                "mild": (max(0.25, short_edge * 0.00020), max(0.55, short_edge * 0.00045)),
                "moderate": (max(0.55, short_edge * 0.00045), max(1.10, short_edge * 0.00090)),
                "strong": (max(1.00, short_edge * 0.00085), max(1.75, short_edge * 0.00140)),
            }
            low, high = ranges[severity]
            operations.append({"name": name, "radius": round(rng.uniform(low, high), 6)})
        elif name == "gaussian_noise":
            sigma_ranges = {"mild": (1.0, 3.0), "moderate": (3.0, 7.0), "strong": (7.0, 12.0)}
            low, high = sigma_ranges[severity]
            operations.append(
                {
                    "name": name,
                    "sigma_255": round(rng.uniform(low, high), 6),
                    "seed": rng.getrandbits(63),
                }
            )
        elif name == "jpeg_compression":
            quality_ranges = {"mild": (82, 95), "moderate": (62, 84), "strong": (42, 68)}
            low, high = quality_ranges[severity]
            subsampling = rng.choice((0, 1)) if severity == "mild" else rng.choice((1, 2))
            operations.append(
                {
                    "name": name,
                    "quality": rng.randint(low, high),
                    "subsampling": subsampling,
                }
            )
    if not operations:
        raise RuntimeError("Synthetic pixel augmentation must contain at least one operation.")
    return {
        "profile": SYNTHETIC_PIXEL_PROFILE,
        "severity": severity,
        "operations": operations,
        "dimensions_unchanged": True,
        "input_size": [image_width, image_height],
        "output_size": [image_width, image_height],
    }


def _resampling(name: str) -> Any:
    namespace = getattr(Image, "Resampling", Image)
    return getattr(namespace, name)


def _apply_synthetic_pixel_augmentation(
    image: Image.Image,
    augmentation: dict[str, Any],
) -> Image.Image:
    if augmentation.get("profile") != SYNTHETIC_PIXEL_PROFILE:
        raise ValueError(f"Unsupported synthetic pixel profile: {augmentation.get('profile')!r}")
    current = image.convert("RGB")
    for operation in augmentation.get("operations") or []:
        name = operation.get("name")
        if name == "resample_roundtrip":
            width, height = current.size
            ratio = float(operation["scale_down_ratio"])
            down_size = (
                max(1, min(width, int(round(width * ratio)))),
                max(1, min(height, int(round(height * ratio)))),
            )
            updated = current.resize(down_size, _resampling(str(operation["down_kernel"]))).resize(
                (width, height),
                _resampling(str(operation["up_kernel"])),
            )
        elif name == "gaussian_blur":
            updated = current.filter(ImageFilter.GaussianBlur(radius=float(operation["radius"])))
        elif name == "gaussian_noise":
            array = np.asarray(current, dtype=np.float32)
            noise = np.random.default_rng(int(operation["seed"])).normal(
                0.0,
                float(operation["sigma_255"]),
                size=array.shape,
            )
            updated = Image.fromarray(np.clip(array + noise, 0, 255).astype(np.uint8), mode="RGB")
        elif name == "jpeg_compression":
            with io.BytesIO() as buffer:
                current.save(
                    buffer,
                    format="JPEG",
                    quality=int(operation["quality"]),
                    subsampling=int(operation["subsampling"]),
                )
                buffer.seek(0)
                with Image.open(buffer) as decoded:
                    updated = decoded.convert("RGB")
        else:
            current.close()
            raise ValueError(f"Unsupported synthetic pixel operation: {name!r}")
        current.close()
        current = updated
    if current.size != image.size:
        current.close()
        raise ValueError(
            f"Synthetic pixel augmentation changed image size: {current.size} != {image.size}"
        )
    return current


def _ensure_extent(
    start: float,
    end: float,
    *,
    limit: int,
    minimum: float = 1.0,
) -> tuple[float, float]:
    start = min(max(start, 0.0), float(limit))
    end = min(max(end, 0.0), float(limit))
    if end - start >= minimum:
        return start, end
    center = min(max((start + end) / 2.0, 0.0), float(limit))
    start = center - minimum / 2.0
    end = center + minimum / 2.0
    if start < 0:
        end -= start
        start = 0.0
    if end > limit:
        start -= end - limit
        end = float(limit)
    return max(0.0, start), min(float(limit), end)


def _bounded_interval(center: float, length: int, *, limit: int) -> tuple[int, int]:
    length = max(1, min(int(length), int(limit)))
    if length >= limit:
        return 0, limit
    start = int(math.floor(center - length / 2.0))
    end = start + length
    if start < 0:
        start = 0
        end = length
    if end > limit:
        end = limit
        start = end - length
    return start, end


def _expand_crop(
    crop_box: tuple[int, int, int, int],
    *,
    image_width: int,
    image_height: int,
    min_crop_size: int,
    max_aspect_ratio: float,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = crop_box
    width, height = right - left, bottom - top
    if width < min_crop_size:
        left, right = _bounded_interval(
            (left + right) / 2.0,
            min_crop_size,
            limit=image_width,
        )
    if height < min_crop_size:
        top, bottom = _bounded_interval(
            (top + bottom) / 2.0,
            min_crop_size,
            limit=image_height,
        )
    width, height = right - left, bottom - top
    if width / height > max_aspect_ratio:
        target_height = min(image_height, int(math.ceil(width / max_aspect_ratio)))
        top, bottom = _bounded_interval(
            (top + bottom) / 2.0,
            max(height, target_height),
            limit=image_height,
        )
    elif height / width > max_aspect_ratio:
        target_width = min(image_width, int(math.ceil(height / max_aspect_ratio)))
        left, right = _bounded_interval(
            (left + right) / 2.0,
            max(width, target_width),
            limit=image_width,
        )
    return left, top, right, bottom


def _sample_context_view(
    *,
    source_bbox: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
    task: str,
    sample_id: str,
    seed: int,
    max_aspect_ratio: float,
    geometry_bbox: tuple[float, float, float, float] | None = None,
    min_crop_size: int = 4,
) -> ContextView:
    source_bbox = _clip_bbox(
        source_bbox,
        image_width=image_width,
        image_height=image_height,
    )
    visible_geometry = _clip_bbox(
        geometry_bbox or source_bbox,
        image_width=image_width,
        image_height=image_height,
    )
    rng = random.Random(f"{seed}:context-view:{task}:{sample_id}")
    noise_bucket = _choose_bucket(rng, PROPOSAL_NOISE_BUCKETS)
    noise_name, _, center_limit, log_scale_limit, edge_limit = noise_bucket

    x1, y1, x2, y2 = source_bbox
    width, height = x2 - x1, y2 - y1
    longest = max(width, height)
    reference_x = max(width, min(32.0, longest * 0.08))
    reference_y = max(height, min(32.0, longest * 0.08))
    center_dx = rng.uniform(-center_limit, center_limit)
    center_dy = rng.uniform(-center_limit, center_limit)
    log_scale_x = rng.uniform(-log_scale_limit, log_scale_limit)
    log_scale_y = rng.uniform(-log_scale_limit, log_scale_limit)
    edge_x = rng.uniform(-edge_limit, edge_limit)
    edge_y = rng.uniform(-edge_limit, edge_limit)
    center_x = (x1 + x2) / 2.0 + center_dx * reference_x
    center_y = (y1 + y2) / 2.0 + center_dy * reference_y
    proposal_width = max(1.0, width * math.exp(log_scale_x))
    proposal_height = max(1.0, height * math.exp(log_scale_y))
    proposal_x1 = center_x - proposal_width / 2.0 - edge_x * reference_x
    proposal_x2 = center_x + proposal_width / 2.0 + edge_x * reference_x
    proposal_y1 = center_y - proposal_height / 2.0 - edge_y * reference_y
    proposal_y2 = center_y + proposal_height / 2.0 + edge_y * reference_y
    proposal_x1, proposal_x2 = _ensure_extent(
        proposal_x1,
        proposal_x2,
        limit=image_width,
    )
    proposal_y1, proposal_y2 = _ensure_extent(
        proposal_y1,
        proposal_y2,
        limit=image_height,
    )
    proposal_bbox = (proposal_x1, proposal_y1, proposal_x2, proposal_y2)

    padding_bucket = _choose_bucket(rng, CONTEXT_PADDING_BUCKETS)
    padding_name, _, padding_min, padding_max = padding_bucket
    padding_ratios = tuple(rng.uniform(padding_min, padding_max) for _ in range(4))
    proposal_width = proposal_x2 - proposal_x1
    proposal_height = proposal_y2 - proposal_y1
    proposal_longest = max(proposal_width, proposal_height)
    padding_reference_x = max(proposal_width, min(64.0, proposal_longest * 0.10))
    padding_reference_y = max(proposal_height, min(64.0, proposal_longest * 0.10))
    crop_left = min(proposal_x1 - padding_ratios[0] * padding_reference_x, visible_geometry[0])
    crop_top = min(proposal_y1 - padding_ratios[1] * padding_reference_y, visible_geometry[1])
    crop_right = max(proposal_x2 + padding_ratios[2] * padding_reference_x, visible_geometry[2])
    crop_bottom = max(proposal_y2 + padding_ratios[3] * padding_reference_y, visible_geometry[3])
    crop_box = (
        max(0, int(math.floor(crop_left))),
        max(0, int(math.floor(crop_top))),
        min(image_width, int(math.ceil(crop_right))),
        min(image_height, int(math.ceil(crop_bottom))),
    )
    crop_box = _expand_crop(
        crop_box,
        image_width=image_width,
        image_height=image_height,
        min_crop_size=min_crop_size,
        max_aspect_ratio=max_aspect_ratio,
    )
    crop_float = tuple(float(value) for value in crop_box)
    gt_coverage = _bbox_coverage(source_bbox, crop_float)
    geometry_coverage = _bbox_coverage(visible_geometry, crop_float)
    if gt_coverage < 1.0 - 1e-9 or geometry_coverage < 1.0 - 1e-9:
        raise ValueError(
            f"Context crop does not cover GT geometry: gt={gt_coverage}, "
            f"geometry={geometry_coverage}"
        )
    crop_width, crop_height = crop_box[2] - crop_box[0], crop_box[3] - crop_box[1]
    if max(crop_width / crop_height, crop_height / crop_width) > max_aspect_ratio + 1e-9:
        raise ValueError(f"Cannot enforce crop aspect ratio <= {max_aspect_ratio}")
    return ContextView(
        proposal_bbox=proposal_bbox,
        crop_box=crop_box,
        proposal_noise_bucket=str(noise_name),
        proposal_noise=(
            center_dx,
            center_dy,
            log_scale_x,
            log_scale_y,
            edge_x,
            edge_y,
        ),
        padding_scale_bucket=str(padding_name),
        requested_padding_ratios=padding_ratios,
        proposal_iou=_bbox_iou(source_bbox, proposal_bbox),
        gt_coverage=1.0,
    )


def _point(value: Any) -> tuple[float, float]:
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise ValueError(f"Invalid point: {value!r}")
    if not all(_is_number(item) for item in value):
        raise ValueError(f"Invalid point: {value!r}")
    return float(value[0]), float(value[1])


def _geometry_bbox(
    label: str,
    source_bbox: tuple[float, float, float, float],
    parameters: dict[str, Any],
) -> tuple[float, float, float, float]:
    xs = [source_bbox[0], source_bbox[2]]
    ys = [source_bbox[1], source_bbox[3]]

    def add_point(value: Any) -> None:
        x, y = _point(value)
        xs.append(x)
        ys.append(y)

    if label == "shape":
        for key in ("corners", "body_corners"):
            corners = parameters.get(key)
            if not isinstance(corners, list):
                continue
            for corner in corners:
                if not isinstance(corner, dict):
                    raise ValueError(f"Invalid corner: {corner!r}")
                for point_key in ("point", "start", "mid", "end"):
                    if point_key in corner:
                        add_point(corner[point_key])
        if isinstance(parameters.get("body_bbox"), list):
            body_bbox = _bbox(parameters["body_bbox"])
            xs.extend((body_bbox[0], body_bbox[2]))
            ys.extend((body_bbox[1], body_bbox[3]))
        tail = parameters.get("tail")
        if isinstance(tail, dict) and isinstance(tail.get("points"), list):
            for point in tail["points"]:
                add_point(point)
    elif label == "line":
        segments = parameters.get("points")
        if not isinstance(segments, list) or not segments:
            raise ValueError("Line parameters require points.")
        for segment_index, segment in enumerate(segments):
            if not isinstance(segment, list) or not segment:
                raise ValueError(
                    f"Invalid line point segment at index {segment_index}: {segment!r}"
                )
            for point in segment:
                add_point(point)
    return min(xs), min(ys), max(xs), max(ys)


def _quantize_crop_point(
    value: Any,
    *,
    left: int,
    top: int,
    crop_width: int,
    crop_height: int,
) -> list[int]:
    x, y = _point(value)
    return [
        quantize_qwen_coordinate(x - left, size=crop_width),
        quantize_qwen_coordinate(y - top, size=crop_height),
    ]


def _quantize_crop_bbox(
    value: Any,
    *,
    left: int,
    top: int,
    crop_width: int,
    crop_height: int,
) -> list[int]:
    x1, y1, x2, y2 = _bbox(value)
    return quantize_qwen_bbox(
        [x1 - left, y1 - top, x2 - left, y2 - top],
        width=crop_width,
        height=crop_height,
        minimum_extent_bins=1,
    )


def _quantize_corner(
    value: Any,
    *,
    left: int,
    top: int,
    crop_width: int,
    crop_height: int,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Invalid corner: {value!r}")
    result = json.loads(json.dumps(value))
    for key in ("point", "start", "mid", "end"):
        if key in result:
            result[key] = _quantize_crop_point(
                result[key],
                left=left,
                top=top,
                crop_width=crop_width,
                crop_height=crop_height,
            )
    return result


def _shape_parameters(
    parameters: dict[str, Any],
    *,
    left: int,
    top: int,
    crop_width: int,
    crop_height: int,
) -> dict[str, Any]:
    result = json.loads(json.dumps(parameters))
    for key in ("corners", "body_corners"):
        if isinstance(result.get(key), list):
            result[key] = [
                _quantize_corner(
                    corner,
                    left=left,
                    top=top,
                    crop_width=crop_width,
                    crop_height=crop_height,
                )
                for corner in result[key]
            ]
    if isinstance(result.get("body_bbox"), list):
        result["body_bbox"] = _quantize_crop_bbox(
            result["body_bbox"],
            left=left,
            top=top,
            crop_width=crop_width,
            crop_height=crop_height,
        )
    tail = result.get("tail")
    if isinstance(tail, dict) and isinstance(tail.get("points"), list):
        tail["points"] = [
            _quantize_crop_point(
                point,
                left=left,
                top=top,
                crop_width=crop_width,
                crop_height=crop_height,
            )
            for point in tail["points"]
        ]
    return result


def _line_parameters(
    parameters: dict[str, Any],
    *,
    left: int,
    top: int,
    crop_width: int,
    crop_height: int,
) -> dict[str, Any]:
    result = json.loads(json.dumps(parameters))
    segments = result.get("points")
    if not isinstance(segments, list) or not segments:
        raise ValueError("Line parameters require points.")
    converted: list[list[list[int]]] = []
    for segment_index, segment in enumerate(segments):
        if not isinstance(segment, list) or not segment:
            raise ValueError(f"Invalid line point segment at index {segment_index}: {segment!r}")
        converted.append(
            [
                _quantize_crop_point(
                    point,
                    left=left,
                    top=top,
                    crop_width=crop_width,
                    crop_height=crop_height,
                )
                for point in segment
            ]
        )
    result["points"] = converted
    return result


def _target_parameters(
    label: str,
    parameters: dict[str, Any],
    *,
    crop_box: tuple[int, int, int, int],
) -> dict[str, Any]:
    left, top, right, bottom = crop_box
    kwargs = {
        "left": left,
        "top": top,
        "crop_width": right - left,
        "crop_height": bottom - top,
    }
    if label == "shape":
        return _shape_parameters(parameters, **kwargs)
    if label == "line":
        return _line_parameters(parameters, **kwargs)
    if label == "image":
        return json.loads(json.dumps(parameters))
    raise ValueError(f"Unsupported label: {label}")


def _resolve_source_instance(
    selection: Selection,
    *,
    layout: list[Any],
    label: str,
    image_width: int,
    image_height: int,
) -> tuple[int, dict[str, Any], tuple[float, float, float, float], bool]:
    def label_matches(instance: dict[str, Any]) -> bool:
        return instance.get("type", instance.get("label")) == label

    if 0 <= selection.instance_index < len(layout):
        instance = layout[selection.instance_index]
        if isinstance(instance, dict) and label_matches(instance):
            source_bbox = _clip_bbox(
                _bbox(instance.get("bbox")),
                image_width=image_width,
                image_height=image_height,
            )
            if _bbox_iou(selection.source_bbox, source_bbox) >= 0.90:
                return (
                    selection.instance_index,
                    instance,
                    source_bbox,
                    source_bbox != selection.source_bbox,
                )

    candidates: list[tuple[float, int, dict[str, Any], tuple[float, float, float, float]]] = []
    for index, instance in enumerate(layout):
        if not isinstance(instance, dict) or not label_matches(instance):
            continue
        source_bbox = _clip_bbox(
            _bbox(instance.get("bbox")),
            image_width=image_width,
            image_height=image_height,
        )
        candidates.append(
            (_bbox_iou(selection.source_bbox, source_bbox), index, instance, source_bbox)
        )
    candidates.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    if not candidates or candidates[0][0] < 0.90:
        raise ValueError(
            f"Cannot match selected {label} bbox {selection.source_bbox} "
            f"at source index {selection.instance_index}"
        )
    best_iou, best_index, best_instance, best_bbox = candidates[0]
    if len(candidates) > 1 and candidates[1][0] == best_iou:
        raise ValueError(
            f"Ambiguous selected {label} bbox {selection.source_bbox}: "
            f"indices {best_index} and {candidates[1][1]}"
        )
    return best_index, best_instance, best_bbox, True


def _read_excluded_ids(path: Path | None) -> set[str]:
    if path is None:
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError(f"Invalid test manifest: {path}")
    excluded: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        sample_id = str(item.get("id") or Path(str(item.get("image_path") or "")).stem)
        if sample_id:
            excluded.add(sample_id)
    return excluded


def _normalize_archived_line_points(value: Any) -> list[list[list[float]]]:
    if not isinstance(value, list) or not value:
        raise ValueError("Archived line source_linestrip must be a non-empty list.")

    def normalize_segment(segment: Any) -> list[list[float]]:
        if not isinstance(segment, list) or len(segment) < 2:
            raise ValueError(f"Archived line segment requires at least two points: {segment!r}")
        return [[*_point(point)] for point in segment]

    if all(
        isinstance(point, list | tuple)
        and len(point) == 2
        and all(_is_number(coordinate) for coordinate in point)
        for point in value
    ):
        return [normalize_segment(value)]
    return [normalize_segment(segment) for segment in value]


def _load_archived_full_image_map(spec: TaskSpec) -> dict[str, str]:
    manifest_path = spec.source_image_manifest
    if manifest_path is None:
        raise ValueError(f"{spec.name} requires source_image_manifest")
    source_root = spec.source_root.resolve()
    full_images: dict[str, str] = {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            extra = row.get("extra")
            if not isinstance(extra, dict) or extra.get("view_type") != "full_image":
                continue
            source_json = str(extra.get("source_json") or "")
            image_path = str(row.get("image_path") or "")
            if not source_json or not image_path:
                raise ValueError(f"Invalid full-image row: {manifest_path}:{line_no}")
            resolved_image = (manifest_path.parent / image_path).resolve()
            if not resolved_image.is_file():
                raise FileNotFoundError(resolved_image)
            try:
                relative_image = resolved_image.relative_to(source_root).as_posix()
            except ValueError as exc:
                raise ValueError(
                    f"Archived full image is outside source root: {resolved_image}"
                ) from exc
            previous = full_images.setdefault(source_json, relative_image)
            if previous != relative_image:
                raise ValueError(
                    f"Duplicate full images for {source_json}: {previous!r} and {relative_image!r}"
                )
    if not full_images:
        raise ValueError(f"No clean full-image rows found in {manifest_path}")
    return full_images


def _load_selections(
    spec: TaskSpec,
    *,
    excluded_ids: set[str],
    limit: int | None,
) -> tuple[list[Selection], int]:
    selections: list[Selection] = []
    seen_sample_ids: set[str] = set()
    excluded = 0
    archived_full_images = (
        _load_archived_full_image_map(spec) if spec.source_kind == "archived_point" else {}
    )
    with spec.selection_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            extra = row.get("extra")
            instances = row.get("instances")
            if (
                not isinstance(extra, dict)
                or not isinstance(instances, list)
                or len(instances) != 1
            ):
                raise ValueError(f"Invalid selection row: {spec.selection_path}:{line_no}")
            instance = instances[0]
            expected_source_label = "arrow" if spec.source_kind == "archived_point" else spec.label
            if not isinstance(instance, dict) or instance.get("label") != expected_source_label:
                raise ValueError(f"Wrong selection label: {spec.selection_path}:{line_no}")
            sample_id = str(row.get("sample_id") or "")
            if extra.get("view_type") == "context_crop_bbox_conditioned":
                suffix = "__context_00"
                if not sample_id.endswith(suffix):
                    raise ValueError(
                        f"Context selection sample id lacks {suffix!r}: "
                        f"{spec.selection_path}:{line_no}"
                    )
                sample_id = sample_id.removesuffix(suffix)
            if not sample_id or sample_id in seen_sample_ids:
                raise ValueError(f"Missing/duplicate sample id: {spec.selection_path}:{line_no}")
            seen_sample_ids.add(sample_id)
            source_json = str(extra.get("source_json") or "")
            source_image = str(extra.get("source_image") or "")
            stem = Path(source_json).stem
            if spec.source_kind in {"real", "real_weak", "archived_point"} and stem in excluded_ids:
                excluded += 1
                continue
            parameters = None
            weak_label = None
            archive_provenance = None
            if spec.source_kind == "real_weak":
                raw_parameters = instance.get("parameters")
                if not isinstance(raw_parameters, dict) or not raw_parameters:
                    raise ValueError(
                        f"Missing real weak parameters: {spec.selection_path}:{line_no}"
                    )
                validation_errors = validate_shape_parameters(raw_parameters)
                if validation_errors:
                    raise ValueError(
                        f"Invalid real weak parameters: {spec.selection_path}:{line_no}: "
                        f"{validation_errors}"
                    )
                parameters = json.loads(json.dumps(raw_parameters))
                raw_weak_label = extra.get("weak_label")
                required_provenance = {
                    "source": "api",
                    "task": spec.name,
                }
                if not isinstance(raw_weak_label, dict):
                    raise ValueError(
                        f"Missing real weak provenance: {spec.selection_path}:{line_no}"
                    )
                for key, expected in required_provenance.items():
                    if raw_weak_label.get(key) != expected:
                        raise ValueError(
                            f"Invalid real weak provenance {key}: "
                            f"{spec.selection_path}:{line_no}"
                        )
                for key in ("schema_version", "model_id", "batch_id", "created_at_utc"):
                    if not isinstance(raw_weak_label.get(key), str) or not raw_weak_label[key]:
                        raise ValueError(
                            f"Missing real weak provenance {key}: "
                            f"{spec.selection_path}:{line_no}"
                        )
                weak_label = json.loads(json.dumps(raw_weak_label))
            elif spec.source_kind == "archived_point":
                if source_json not in archived_full_images:
                    raise ValueError(
                        f"No clean archived full image for {source_json}: "
                        f"{spec.selection_path}:{line_no}"
                    )
                parameters = {
                    "is_single": True,
                    "points": _normalize_archived_line_points(extra.get("source_linestrip")),
                }
                parameters["is_single"] = len(parameters["points"]) == 1
                archive_provenance = {
                    "source_task": str(extra.get("task") or "point_arrow"),
                    "source_point_order": "arrow_tail_to_head",
                    "archived_crop_image": str(row.get("image_path") or ""),
                    "archived_source_image": source_image,
                    "archived_crop_box": extra.get("crop_box"),
                    "source_image_width": extra.get("source_image_width"),
                    "source_image_height": extra.get("source_image_height"),
                }
                source_image = archived_full_images[source_json]
            selections.append(
                Selection(
                    sample_id=sample_id,
                    stem=stem,
                    instance_index=int(extra["source_instance_index"]),
                    source_bbox=_bbox(extra.get("source_bbox")),
                    source_image=source_image,
                    source_json=source_json,
                    parameters=parameters,
                    weak_label=weak_label,
                    archive_provenance=archive_provenance,
                )
            )
            if limit is not None and len(selections) >= limit:
                break
    return selections, excluded


def _filter_synthetic_multi_point_selections(
    spec: TaskSpec,
    selections: list[Selection],
    *,
    seed: int,
) -> tuple[list[Selection], Counter[str]]:
    if spec.source_kind != "synthetic_point_multi":
        raise ValueError(f"Wrong source kind for synthetic multi filter: {spec.source_kind}")
    by_source: dict[str, list[Selection]] = defaultdict(list)
    for selection in selections:
        by_source[selection.source_json].append(selection)

    eligible: list[tuple[Selection, int]] = []
    counts: Counter[str] = Counter({"synthetic_multi_input": len(selections)})
    for source_json, source_selections in sorted(by_source.items()):
        source_json_path = spec.source_root / source_json
        payload = json.loads(source_json_path.read_text(encoding="utf-8"))
        size = payload.get("size")
        layout = payload.get("layout")
        if not isinstance(size, list) or len(size) != 2 or not isinstance(layout, list):
            raise ValueError(f"Invalid gt_standard source: {source_json_path}")
        image_width, image_height = int(size[0]), int(size[1])
        for selection in source_selections:
            _, instance, source_bbox, _ = _resolve_source_instance(
                selection,
                layout=layout,
                label=spec.label,
                image_width=image_width,
                image_height=image_height,
            )
            parameters = instance.get("parameters")
            if not isinstance(parameters, dict):
                raise ValueError(
                    f"Missing source parameters: {source_json}:{selection.instance_index}"
                )
            points = parameters.get("points")
            is_single = parameters.get("is_single")
            if _is_multi_segment_line_parameters(parameters):
                _geometry_bbox(spec.label, source_bbox, parameters)
                segment_count = len(points)
                eligible.append((selection, segment_count))
                counts["synthetic_multi_eligible"] += 1
                counts[f"synthetic_multi_available_segments_{segment_count}"] += 1
            elif is_single is True and isinstance(points, list) and len(points) == 1:
                counts["synthetic_multi_rejected_single"] += 1
            else:
                counts["synthetic_multi_rejected_inconsistent"] += 1
    selection_limit = spec.selection_limit
    if selection_limit is None or selection_limit >= len(eligible):
        selected = [selection for selection, _ in eligible]
    else:
        by_segment_count: dict[int, list[Selection]] = defaultdict(list)
        for selection, segment_count in eligible:
            by_segment_count[segment_count].append(selection)
        quotas = _balanced_stratum_quotas(
            {segment_count: len(group) for segment_count, group in by_segment_count.items()},
            total=selection_limit,
        )
        selected_ids: set[str] = set()
        for segment_count, group in sorted(by_segment_count.items()):
            ranked = sorted(group, key=lambda item: _stable_selection_key(item, seed=seed))
            selected_ids.update(item.sample_id for item in ranked[: quotas[segment_count]])
        selected = [selection for selection, _ in eligible if selection.sample_id in selected_ids]

    final_selected_ids = {item.sample_id for item in selected}
    selected_segment_counts = Counter(
        segment_count
        for selection, segment_count in eligible
        if selection.sample_id in final_selected_ids
    )
    counts["synthetic_multi_selected"] = len(selected)
    counts["synthetic_multi_dropped_by_cap"] = len(eligible) - len(selected)
    for segment_count, count in sorted(selected_segment_counts.items()):
        counts[f"synthetic_multi_selected_segments_{segment_count}"] = count
    return selected, counts


def _balanced_stratum_quotas(capacities: dict[int, int], *, total: int) -> dict[int, int]:
    if total < 0:
        raise ValueError("stratified selection total must be non-negative")
    total = min(total, sum(capacities.values()))
    quotas = {key: 0 for key in capacities}
    active = [key for key in sorted(capacities) if capacities[key] > 0]
    remaining = total
    while remaining and active:
        base, extra = divmod(remaining, len(active))
        granted = 0
        for position, key in enumerate(active):
            requested = base + int(position < extra)
            available = capacities[key] - quotas[key]
            current = min(requested, available)
            quotas[key] += current
            granted += current
        if not granted:
            break
        remaining -= granted
        active = [key for key in active if quotas[key] < capacities[key]]
    return quotas


def _stable_selection_key(selection: Selection, *, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{selection.sample_id}".encode()).hexdigest()


def _rectangle_attribute_stratum(selection: Selection) -> tuple[str, str, str]:
    parameters = selection.parameters or {}
    border = parameters.get("border")
    fill = parameters.get("fill")
    effect = parameters.get("effect")
    return (
        str(border.get("type")) if isinstance(border, dict) else "missing",
        str(fill.get("type")) if isinstance(fill, dict) else "missing",
        str(effect.get("type")) if isinstance(effect, dict) else "missing",
    )


def _stratify_shape_attribute_selections(
    selections: list[Selection],
    *,
    max_rectangle_fraction: float,
    seed: int,
) -> tuple[list[Selection], Counter[str]]:
    if not 0.0 < max_rectangle_fraction < 1.0:
        raise ValueError("max_rectangle_fraction must be between 0 and 1")
    rectangles = [
        selection
        for selection in selections
        if selection.parameters and selection.parameters.get("shape_type") == "rectangle"
    ]
    non_rectangles = [
        selection
        for selection in selections
        if not selection.parameters or selection.parameters.get("shape_type") != "rectangle"
    ]
    if not non_rectangles:
        counts: Counter[str] = Counter(
            {
                "sampling_input_rows": len(selections),
                "sampling_input_rectangle": len(rectangles),
                "sampling_input_non_rectangle": 0,
                "sampling_selected_rows": len(selections),
                "sampling_selected_rectangle": len(rectangles),
                "sampling_selected_non_rectangle": 0,
                "sampling_dropped_rectangle": 0,
                "sampling_rectangle_fraction_ppm": 1_000_000 if rectangles else 0,
                "sampling_cap_unavailable_no_non_rectangle": int(bool(rectangles)),
            }
        )
        return list(selections), counts
    rectangle_cap = math.floor(
        max_rectangle_fraction / (1.0 - max_rectangle_fraction) * len(non_rectangles)
    )
    rectangle_cap = min(len(rectangles), rectangle_cap)

    groups: dict[tuple[str, str, str], list[Selection]] = defaultdict(list)
    for selection in rectangles:
        groups[_rectangle_attribute_stratum(selection)].append(selection)
    for values in groups.values():
        values.sort(key=lambda selection: _stable_selection_key(selection, seed=seed))

    quotas = {key: 0 for key in groups}
    if rectangle_cap >= len(groups):
        for key in groups:
            quotas[key] = 1
    remaining = rectangle_cap - sum(quotas.values())
    if remaining > 0:
        capacities = {key: len(groups[key]) - quotas[key] for key in groups}
        total_capacity = sum(capacities.values())
        ideals = {
            key: remaining * capacity / total_capacity if total_capacity else 0.0
            for key, capacity in capacities.items()
        }
        for key, ideal in ideals.items():
            quotas[key] += min(capacities[key], math.floor(ideal))
        remainder = rectangle_cap - sum(quotas.values())
        order = sorted(
            groups,
            key=lambda key: (
                -(ideals[key] - math.floor(ideals[key])),
                key,
            ),
        )
        while remainder > 0:
            progressed = False
            for key in order:
                if quotas[key] >= len(groups[key]):
                    continue
                quotas[key] += 1
                remainder -= 1
                progressed = True
                if remainder == 0:
                    break
            if not progressed:
                raise RuntimeError("Unable to allocate rectangle sampling quota")

    selected_rectangle_ids = {
        selection.sample_id for key, values in groups.items() for selection in values[: quotas[key]]
    }
    selected = [
        selection
        for selection in selections
        if (
            not selection.parameters
            or selection.parameters.get("shape_type") != "rectangle"
            or selection.sample_id in selected_rectangle_ids
        )
    ]
    counts: Counter[str] = Counter(
        {
            "sampling_input_rows": len(selections),
            "sampling_input_rectangle": len(rectangles),
            "sampling_input_non_rectangle": len(non_rectangles),
            "sampling_selected_rows": len(selected),
            "sampling_selected_rectangle": len(selected_rectangle_ids),
            "sampling_selected_non_rectangle": len(non_rectangles),
            "sampling_dropped_rectangle": len(rectangles) - len(selected_rectangle_ids),
            "sampling_rectangle_fraction_ppm": round(
                len(selected_rectangle_ids) / len(selected) * 1_000_000
            )
            if selected
            else 0,
        }
    )
    return selected, counts


def _prompt_contract(path: Path) -> tuple[str, str | None]:
    prompts = load_prompt_pool(path)
    if not prompts:
        raise ValueError(f"Empty prompt pool: {path}")
    expected_arguments = ("proposal_bbox_2d",)
    for prompt in prompts:
        if prompt.program.schema.names != expected_arguments:
            raise ValueError(
                f"Prompt must declare only proposal_bbox_2d: {prompt.prompt_id}, "
                f"got {prompt.program.schema.names}"
            )
        rendered = prompt.render({"proposal_bbox_2d": [1, 2, 300, 400]})
        if "[1,2,300,400]" not in rendered:
            raise ValueError(f"Prompt does not render proposal_bbox_2d: {prompt.prompt_id}")
    output_schema = prompts[0].metadata.get("output_schema")
    return prompts[0].prompt_id.rsplit(".", 1)[0], (
        str(output_schema) if output_schema is not None else None
    )


def _source_image_path(spec: TaskSpec, selection: Selection) -> Path:
    path = spec.source_root / selection.source_image
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _image_shard(sample_id: str) -> str:
    return hashlib.sha1(sample_id.encode("utf-8")).hexdigest()[:2]


def _local_bbox(
    value: tuple[float, float, float, float],
    crop_box: tuple[int, int, int, int],
) -> list[float]:
    left, top, _, _ = crop_box
    return [
        round(value[0] - left, 3),
        round(value[1] - top, 3),
        round(value[2] - left, 3),
        round(value[3] - top, 3),
    ]


def _distractor_count(
    layout: list[Any],
    *,
    source_instance_index: int,
    crop_box: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> int:
    crop = tuple(float(value) for value in crop_box)
    count = 0
    for index, instance in enumerate(layout):
        if index == source_instance_index or not isinstance(instance, dict):
            continue
        try:
            instance_bbox = _clip_bbox(
                _bbox(instance.get("bbox")),
                image_width=image_width,
                image_height=image_height,
            )
        except ValueError:
            continue
        count += int(_bbox_intersection_area(instance_bbox, crop) > 0)
    return count


def _build_row(
    *,
    config: WorkerConfig,
    selection: Selection,
    source_image: Image.Image,
    source_instance_index: int,
    source_bbox: tuple[float, float, float, float],
    source_parameters: dict[str, Any],
    source_layout: list[Any] | None,
    image_width: int,
    image_height: int,
) -> tuple[str, str, Counter[str]]:
    spec = config.spec
    geometry_bbox = _geometry_bbox(spec.label, source_bbox, source_parameters)
    view = _sample_context_view(
        source_bbox=source_bbox,
        image_width=image_width,
        image_height=image_height,
        task=spec.name,
        sample_id=selection.sample_id,
        seed=config.seed,
        max_aspect_ratio=config.max_aspect_ratio,
        geometry_bbox=geometry_bbox,
        min_crop_size=config.min_crop_size,
    )
    left, top, right, bottom = view.crop_box
    crop_width, crop_height = right - left, bottom - top
    target_parameters = _target_parameters(
        spec.label,
        source_parameters,
        crop_box=view.crop_box,
    )
    prompt_bbox = quantize_qwen_bbox(
        _local_bbox(view.proposal_bbox, view.crop_box),
        width=crop_width,
        height=crop_height,
        minimum_extent_bins=1,
    )
    target_bbox = quantize_qwen_bbox(
        _local_bbox(source_bbox, view.crop_box),
        width=crop_width,
        height=crop_height,
        minimum_extent_bins=1,
    )
    target_width = target_bbox[2] - target_bbox[0]
    target_height = target_bbox[3] - target_bbox[1]
    target_short_span = min(target_width, target_height)
    sample_id = f"{selection.sample_id}__context_00"
    if "/" in sample_id or "\\" in sample_id:
        raise ValueError(f"Unsafe sample id: {sample_id!r}")
    shard = _image_shard(sample_id)
    filename = f"{sample_id}.png"
    image_relative = f"../images/train/{shard}/{filename}"
    image_output = config.staging_root / "images/train" / shard / filename
    image_output.parent.mkdir(parents=True, exist_ok=True)
    crop = source_image.crop(view.crop_box)
    pixel_augmentation: dict[str, Any] = {"profile": "none", "operations": []}
    augmented: Image.Image | None = None
    try:
        output_image = crop
        if _is_synthetic_source(spec):
            pixel_augmentation = _sample_synthetic_pixel_augmentation(
                task=spec.name,
                sample_id=selection.sample_id,
                seed=config.seed,
                target_short_span=target_short_span,
                image_width=crop_width,
                image_height=crop_height,
            )
            augmented = _apply_synthetic_pixel_augmentation(crop, pixel_augmentation)
            output_image = augmented
        output_image.save(image_output, format="PNG", compress_level=config.png_compress_level)
    finally:
        if augmented is not None:
            augmented.close()
        crop.close()

    distractors = (
        _distractor_count(
            source_layout,
            source_instance_index=source_instance_index,
            crop_box=view.crop_box,
            image_width=image_width,
            image_height=image_height,
        )
        if source_layout is not None
        else None
    )
    target_area_fraction = _bbox_area(source_bbox) / float(crop_width * crop_height)
    geometry_coverage = _bbox_coverage(
        _clip_bbox(geometry_bbox, image_width=image_width, image_height=image_height),
        tuple(float(value) for value in view.crop_box),
    )
    selection_manifest = "../selection/train.jsonl"
    noise_values = view.proposal_noise
    structured_extra = {
        "task": spec.name,
        "split": "train",
        "view_type": "context_crop_bbox_conditioned",
        "source_dataset": spec.source_root.name,
        "source_json": selection.source_json,
        "source_image": selection.source_image,
        "source_instance_index": source_instance_index,
        "selection_source_instance_index": selection.instance_index,
        "selection_source_bbox": list(selection.source_bbox),
        "source_bbox": list(source_bbox),
        "geometry_bbox": list(geometry_bbox),
        "proposal_bbox": [round(value, 6) for value in view.proposal_bbox],
        "proposal_bbox_2d": prompt_bbox,
        "proposal_noise_bucket": view.proposal_noise_bucket,
        "proposal_noise": {
            "center_dx_ref": round(noise_values[0], 6),
            "center_dy_ref": round(noise_values[1], 6),
            "log_scale_x": round(noise_values[2], 6),
            "log_scale_y": round(noise_values[3], 6),
            "edge_x_ref": round(noise_values[4], 6),
            "edge_y_ref": round(noise_values[5], 6),
        },
        "proposal_iou": round(view.proposal_iou, 6),
        "crop_box": list(view.crop_box),
        "padding_scale_bucket": view.padding_scale_bucket,
        "requested_padding_ratios": [round(value, 6) for value in view.requested_padding_ratios],
        "gt_coverage": view.gt_coverage,
        "geometry_coverage": round(geometry_coverage, 6),
        "target_bbox_2d": target_bbox,
        "target_short_span_2d": target_short_span,
        "target_area_fraction": round(target_area_fraction, 8),
        "distractor_count": distractors,
        "pixel_augmentation": pixel_augmentation,
        "prompt_coordinate_space": "qwen_0_999_context_crop",
        "target_coordinate_space": "qwen_0_999_context_crop",
        "target_num_bins": 1000,
        "selection_manifest": selection_manifest,
        "augmentation": {
            "name": "detector_proposal_context_crop",
            "proposal_noise": view.proposal_noise_bucket,
            "padding_scale": view.padding_scale_bucket,
            "asymmetric_padding": True,
            "max_aspect_ratio": config.max_aspect_ratio,
        },
    }
    if selection.weak_label is not None:
        structured_extra["weak_label"] = selection.weak_label
    if selection.archive_provenance is not None:
        structured_extra["archive_provenance"] = selection.archive_provenance
    target = {"type": spec.label, "parameters": target_parameters}
    structured = {
        "sample_id": sample_id,
        "image_path": image_relative,
        "image_width": crop_width,
        "image_height": crop_height,
        "instances": [
            {
                "label": spec.label,
                "bbox": _local_bbox(source_bbox, view.crop_box),
                "parameters": target_parameters,
            }
        ],
        "extra": structured_extra,
    }
    sft_extra: dict[str, Any] = {
        "prompt_pool_id": config.prompt_pool_id,
        "source_sample_id": selection.stem,
        "source_type": (
            "synthetic_gt_standard_context_points"
            if spec.source_kind == "synthetic_point_multi"
            else (
                "synthetic_gt_standard_context"
                if spec.source_kind == "synthetic"
                else (
                    "archived_real_context_points"
                    if spec.source_kind == "archived_point"
                    else (
                        "api_weak_real_context"
                        if spec.source_kind == "real_weak"
                        else "reviewed_raw_context"
                    )
                )
            )
        ),
        "image_width": crop_width,
        "image_height": crop_height,
        "prompt_coordinate_space": "qwen_0_999_context_crop",
        "target_coordinate_space": "qwen_0_999_context_crop",
        "num_bins": 1000,
        "structured_extra": structured_extra,
    }
    if config.output_schema:
        sft_extra["output_schema"] = config.output_schema
    sft = {
        "image_path": image_relative,
        "sample_id": sample_id,
        "dataset_name": spec.name,
        "system_prompt": "",
        "user_prompt": "",
        "prompt_args": {"proposal_bbox_2d": prompt_bbox},
        "target_text": _json_dumps(target),
        "extra": sft_extra,
    }
    counts: Counter[str] = Counter()
    counts["rows"] += 1
    counts[f"source_kind_{spec.source_kind}_rows"] += 1
    counts[f"proposal_noise_{view.proposal_noise_bucket}"] += 1
    counts[f"padding_scale_{view.padding_scale_bucket}"] += 1
    if distractors is None:
        counts["distractor_count_unavailable"] += 1
    else:
        counts["distractor_nonzero"] += int(distractors > 0)
    counts["target_short_span_lt_80"] += int(target_short_span < 80)
    counts["target_short_span_80_199"] += int(80 <= target_short_span < 200)
    counts["target_short_span_200_plus"] += int(target_short_span >= 200)
    counts[f"pixel_aug_profile_{pixel_augmentation['profile']}"] += 1
    operations = pixel_augmentation["operations"]
    counts[f"pixel_aug_stack_depth_{len(operations)}"] += 1
    if pixel_augmentation.get("severity"):
        counts[f"pixel_aug_severity_{pixel_augmentation['severity']}"] += 1
    for operation in operations:
        counts[f"pixel_aug_operation_{operation['name']}"] += 1
    if spec.label == "shape":
        counts[f"shape_type_{target_parameters.get('shape_type', 'missing')}"] += 1
    elif spec.label == "line":
        counts["line_multi_segment"] += int(len(target_parameters.get("points") or []) > 1)
        counts["line_is_single_false"] += int(target_parameters.get("is_single") is False)
    elif spec.label == "image":
        counts[f"image_type_{target_parameters.get('image_type', 'missing')}"] += 1
    return _json_dumps(structured), _json_dumps(sft), counts


def _build_synthetic_source(
    item: tuple[tuple[str, str], tuple[Selection, ...], WorkerConfig],
) -> WorkerResult:
    source_key, selections, config = item
    spec = config.spec
    source_json, source_image = source_key
    source_json_path = spec.source_root / source_json
    payload = json.loads(source_json_path.read_text(encoding="utf-8"))
    size = payload.get("size")
    layout = payload.get("layout")
    if not isinstance(size, list) or len(size) != 2 or not isinstance(layout, list):
        raise ValueError(f"Invalid gt_standard source: {source_json_path}")
    image_width, image_height = int(size[0]), int(size[1])
    image_path = _source_image_path(spec, selections[0])
    with Image.open(image_path) as opened:
        if opened.size != (image_width, image_height):
            raise ValueError(
                f"Image size mismatch for {source_image}: "
                f"{opened.size} != {(image_width, image_height)}"
            )
        source = opened.convert("RGB")
    rows: list[tuple[str, str]] = []
    counts: Counter[str] = Counter()
    try:
        for selection in selections:
            source_index, instance, source_bbox, drift = _resolve_source_instance(
                selection,
                layout=layout,
                label=spec.label,
                image_width=image_width,
                image_height=image_height,
            )
            parameters = instance.get("parameters")
            if not isinstance(parameters, dict):
                raise ValueError(
                    f"Missing source parameters: {source_json}:{selection.instance_index}"
                )
            if spec.source_kind == "synthetic_point_multi":
                points = parameters.get("points")
                if not _is_multi_segment_line_parameters(parameters):
                    raise ValueError(
                        f"Synthetic point subset received a non-multi line: "
                        f"{source_json}:{selection.instance_index}"
                    )
                parameters = {
                    "is_single": False,
                    "points": json.loads(json.dumps(points)),
                }
            structured, sft, row_counts = _build_row(
                config=config,
                selection=selection,
                source_image=source,
                source_instance_index=source_index,
                source_bbox=source_bbox,
                source_parameters=parameters,
                source_layout=layout,
                image_width=image_width,
                image_height=image_height,
            )
            rows.append((structured, sft))
            counts.update(row_counts)
            counts["source_bbox_drift"] += int(drift)
            counts["source_index_remap"] += int(source_index != selection.instance_index)
    finally:
        source.close()
    return WorkerResult(tuple(rows), dict(counts))


def _build_real_source(
    item: tuple[tuple[str, str], tuple[Selection, ...], WorkerConfig],
) -> WorkerResult:
    source_key, selections, config = item
    spec = config.spec
    source_json, source_image = source_key
    source_json_path = spec.source_root / source_json
    payload = json.loads(source_json_path.read_text(encoding="utf-8"))
    layout = payload.get("instances")
    if not isinstance(layout, list):
        raise ValueError(f"Invalid raw source: {source_json_path}")
    image_path = _source_image_path(spec, selections[0])
    with Image.open(image_path) as opened:
        image_width, image_height = opened.size
        source = opened.convert("RGB")
    annotated_size = (payload.get("image_width"), payload.get("image_height"))
    if all(_is_number(item) for item in annotated_size) and tuple(map(int, annotated_size)) != (
        image_width,
        image_height,
    ):
        source.close()
        raise ValueError(
            f"Image size mismatch for {source_json_path}: "
            f"{(image_width, image_height)} != {annotated_size}"
        )
    rows: list[tuple[str, str]] = []
    counts: Counter[str] = Counter()
    try:
        for selection in selections:
            source_index, instance, source_bbox, drift = _resolve_source_instance(
                selection,
                layout=layout,
                label=spec.label,
                image_width=image_width,
                image_height=image_height,
            )
            if spec.source_kind == "real_weak":
                if not isinstance(selection.parameters, dict) or not selection.parameters:
                    raise ValueError(
                        f"Missing selected weak parameters: {source_json_path}:{source_index}"
                    )
                source_parameters = selection.parameters
            else:
                instance_extra = instance.get("extra")
                parameters = (
                    instance_extra.get("parameters") if isinstance(instance_extra, dict) else None
                )
                image_type = parameters.get("image_type") if isinstance(parameters, dict) else None
                if not isinstance(image_type, str) or not image_type:
                    raise ValueError(f"Missing raw image_type: {source_json_path}:{source_index}")
                source_parameters = {"image_type": image_type}
            structured, sft, row_counts = _build_row(
                config=config,
                selection=selection,
                source_image=source,
                source_instance_index=source_index,
                source_bbox=source_bbox,
                source_parameters=source_parameters,
                source_layout=layout,
                image_width=image_width,
                image_height=image_height,
            )
            rows.append((structured, sft))
            counts.update(row_counts)
            counts["source_bbox_drift"] += int(drift)
            counts["source_index_remap"] += int(source_index != selection.instance_index)
    finally:
        source.close()
    return WorkerResult(tuple(rows), dict(counts))


def _build_archived_point_source(
    item: tuple[tuple[str, str], tuple[Selection, ...], WorkerConfig],
) -> WorkerResult:
    _, selections, config = item
    spec = config.spec
    image_path = _source_image_path(spec, selections[0])
    with Image.open(image_path) as opened:
        image_width, image_height = opened.size
        source = opened.convert("RGB")
    rows: list[tuple[str, str]] = []
    counts: Counter[str] = Counter()
    try:
        for selection in selections:
            if not isinstance(selection.parameters, dict):
                raise ValueError(f"Missing archived line points: {selection.sample_id}")
            provenance = selection.archive_provenance or {}
            annotated_size = (
                provenance.get("source_image_width"),
                provenance.get("source_image_height"),
            )
            if all(_is_number(value) for value in annotated_size) and tuple(
                map(int, annotated_size)
            ) != (image_width, image_height):
                raise ValueError(
                    f"Archived full-image size mismatch for {selection.source_json}: "
                    f"{(image_width, image_height)} != {annotated_size}"
                )
            source_bbox = _clip_bbox(
                selection.source_bbox,
                image_width=image_width,
                image_height=image_height,
            )
            geometry_bbox = _geometry_bbox("line", source_bbox, selection.parameters)
            clipped_geometry = _clip_bbox(
                geometry_bbox,
                image_width=image_width,
                image_height=image_height,
            )
            if any(
                not math.isclose(original, clipped, abs_tol=1e-6)
                for original, clipped in zip(geometry_bbox, clipped_geometry, strict=True)
            ):
                raise ValueError(
                    f"Archived line points leave the clean full image: {selection.sample_id}"
                )
            structured, sft, row_counts = _build_row(
                config=config,
                selection=selection,
                source_image=source,
                source_instance_index=selection.instance_index,
                source_bbox=source_bbox,
                source_parameters=selection.parameters,
                source_layout=None,
                image_width=image_width,
                image_height=image_height,
            )
            rows.append((structured, sft))
            counts.update(row_counts)
            counts["source_bbox_drift"] += int(source_bbox != selection.source_bbox)
            counts["source_index_remap"] += 0
    finally:
        source.close()
    return WorkerResult(tuple(rows), dict(counts))


def _build_source(
    item: tuple[tuple[str, str], tuple[Selection, ...], WorkerConfig],
) -> WorkerResult:
    spec = item[2].spec
    if _is_synthetic_source(spec):
        return _build_synthetic_source(item)
    if spec.source_kind == "archived_point":
        return _build_archived_point_source(item)
    return _build_real_source(item)


def _prepare_staging(spec: TaskSpec, output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{spec.name}.staging.", dir=output_root))
    staging.chmod(0o755)
    for relative in ("selection", "structured", "sft", "images/train"):
        (staging / relative).mkdir(parents=True, exist_ok=True)
    _atomic_write_text(staging / "structured/val.jsonl", "")
    _atomic_write_text(staging / "sft/val.jsonl", "")
    return staging


def _write_selection_manifest(
    path: Path,
    *,
    spec: TaskSpec,
    selections: list[Selection],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent) as f:
        for selection in selections:
            instance = {
                "label": spec.label,
                "bbox": list(selection.source_bbox),
            }
            if selection.parameters is not None:
                instance["parameters"] = selection.parameters
            extra = {
                "source_json": selection.source_json,
                "source_image": selection.source_image,
                "source_instance_index": selection.instance_index,
                "source_bbox": list(selection.source_bbox),
            }
            if selection.weak_label is not None:
                extra["weak_label"] = selection.weak_label
            if selection.archive_provenance is not None:
                extra["archive_provenance"] = selection.archive_provenance
            row = {
                "sample_id": selection.sample_id,
                "instances": [instance],
                "extra": extra,
            }
            f.write(_json_dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())
        temp_path = Path(f.name)
    os.replace(temp_path, path)


def _publish(staging: Path, destination: Path, *, clean: bool) -> None:
    if destination.exists() and not clean:
        raise FileExistsError(f"Output already exists; pass --clean to replace: {destination}")
    backup: Path | None = None
    if destination.exists():
        backup = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.backup.", dir=destination.parent)
        )
        backup.rmdir()
        os.replace(destination, backup)
    try:
        os.replace(staging, destination)
    except BaseException:
        if backup is not None:
            os.replace(backup, destination)
        raise
    if backup is not None:
        shutil.rmtree(backup, ignore_errors=True)


def _write_readme(
    spec: TaskSpec,
    staging: Path,
    *,
    counts: Counter[str],
    seed: int,
    workers: int,
    max_aspect_ratio: float,
    excluded: int,
    shape_attribute_max_rectangle_fraction: float,
) -> None:
    source_specs = _expanded_task_sources(spec)
    has_synthetic_source = any(_is_synthetic_source(source) for source in source_specs)
    has_archived_point_source = any(
        source.source_kind == "archived_point" for source in source_specs
    )
    pixel_policy = (
        f"every synthetic crop uses one to three stacked, size-preserving operations from "
        f"`{SYNTHETIC_PIXEL_PROFILE}`; real image crops are not synthetically degraded"
        if has_synthetic_source
        else "none (real image crops are not synthetically degraded)"
    )
    truth_policy = (
        "ordered real source-image bbox/linestrip truth is recovered from the archived point "
        "selection and clean full-image manifest; synthetic multi-segment truth is independently "
        "reloaded from `gt_standard`"
        if has_archived_point_source and has_synthetic_source
        else (
            "ordered source-image bbox/linestrip truth is recovered from the archived point "
            "selection; the clean full bitmap is restored through the archived grounding "
            "full-image manifest"
            if has_archived_point_source
            else (
                "target parameters are API weak-label truth snapshotted in `selection/train.jsonl`"
                if spec.source_kind == "real_weak"
                else "target truth is reloaded from source"
            )
        )
    )
    target_policy = (
        "the line geometry subset `is_single + points` only; no inferred style, color, or "
        "arrow-endpoint fields; geometry uses crop-local Qwen `0..999`"
        if has_archived_point_source
        else (
            "compact shape attribute subset; no control points or geometry fields"
            if spec.source_kind == "real_weak"
            else f"compact `{spec.label}` JSON; geometry uses the same crop-local Qwen `0..999`"
        )
    )
    sampling_policy = (
        "use archived point train rows after canonical test exclusion, then add only synthetic "
        "`is_single=false` targets with more than one path segment; deterministically cap the "
        "synthetic subset with equal quotas across observed segment-count strata; no "
        "synthetic single-path targets are admitted and no resize/multi-scale view expansion "
        "is performed"
        if has_archived_point_source and has_synthetic_source
        else (
            "use archived point train rows only, after excluding every source in the canonical "
            "current test manifest; archived validation rows are not promoted to training"
            if has_archived_point_source
            else (
                "keep every non-rectangle row and deterministically stratify rectangle rows to at "
                f"most {shape_attribute_max_rectangle_fraction:.0%} of the final task dataset"
                if spec.source_kind == "real_weak"
                else "use the complete maintained selection snapshot"
            )
        )
    )
    source_lines = "\n".join(
        f"- Source ({source.source_kind}): `{source.source_root}` via `{source.selection_path}`"
        for source in source_specs
    )
    content = f"""# {spec.name}

Derived contextual-crop reconstruction training data.

{source_lines}
- Rebuild selection snapshot: `selection/train.jsonl` ({truth_policy})
- Prompt pool: `{spec.prompt_path}`
- Target label: `{spec.label}`
- Split: train only; validation is intentionally empty
- Rows: {counts["rows"]}
- Test-manifest rows excluded: {excluded}
- Seed: {seed}
- Workers: {workers}
- Input: one bounded contextual crop per selected instance
- Prompt condition: approximate `proposal_bbox_2d` in crop-local Qwen integer `0..999`
- Target: {target_policy}
- Sampling: {sampling_policy}
- Proposal noise: clean/accurate/moderate/hard = 10/50/30/10%
- Padding buckets: tight/medium/large/extreme = 20/50/25/5%; four sides sampled independently
- GT policy: the full visible source bbox and explicit geometry are contained in the crop
- Max crop aspect ratio: {max_aspect_ratio}
- Pixel augmentation: {pixel_policy}
- Image format: PNG; pixel augmentation preserves the exact crop width and height
- Message order: image-first at training collation time

## Build counts

```json
{json.dumps(dict(sorted(counts.items())), ensure_ascii=False, indent=2)}
```
"""
    _atomic_write_text(staging / "README.md", content)
    _atomic_write_text(
        staging / "build_summary.json",
        json.dumps(
            {
                "task": spec.name,
                "rows": counts["rows"],
                "excluded_test_rows": excluded,
                "seed": seed,
                "workers": workers,
                "max_aspect_ratio": max_aspect_ratio,
                "shape_attribute_max_rectangle_fraction": (
                    shape_attribute_max_rectangle_fraction
                    if spec.source_kind == "real_weak"
                    else None
                ),
                "synthetic_pixel_profile": (
                    SYNTHETIC_PIXEL_PROFILE if has_synthetic_source else "none"
                ),
                "sources": [
                    {
                        "source_kind": source.source_kind,
                        "source_root": str(source.source_root),
                        "selection_path": str(source.selection_path),
                        "source_image_manifest": (
                            str(source.source_image_manifest)
                            if source.source_image_manifest is not None
                            else None
                        ),
                        "selection_limit": source.selection_limit,
                    }
                    for source in source_specs
                ],
                "counts": dict(sorted(counts.items())),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )


def _build_task(
    spec: TaskSpec,
    *,
    output_root: Path,
    workers: int,
    chunksize: int,
    clean: bool,
    seed: int,
    min_crop_size: int,
    max_aspect_ratio: float,
    png_compress_level: int,
    excluded_ids: set[str],
    limit: int | None,
    shape_attribute_max_rectangle_fraction: float,
) -> Counter[str]:
    destination = output_root / spec.name
    if destination.exists() and not clean:
        raise FileExistsError(f"Output already exists; pass --clean to replace: {destination}")
    staging = _prepare_staging(spec, output_root)
    try:
        return _build_task_in_staging(
            spec,
            staging=staging,
            output_root=output_root,
            workers=workers,
            chunksize=chunksize,
            clean=clean,
            seed=seed,
            min_crop_size=min_crop_size,
            max_aspect_ratio=max_aspect_ratio,
            png_compress_level=png_compress_level,
            excluded_ids=excluded_ids,
            limit=limit,
            shape_attribute_max_rectangle_fraction=shape_attribute_max_rectangle_fraction,
        )
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _build_task_in_staging(
    spec: TaskSpec,
    *,
    staging: Path,
    output_root: Path,
    workers: int,
    chunksize: int,
    clean: bool,
    seed: int,
    min_crop_size: int,
    max_aspect_ratio: float,
    png_compress_level: int,
    excluded_ids: set[str],
    limit: int | None,
    shape_attribute_max_rectangle_fraction: float,
) -> Counter[str]:
    prompt_pool_id, output_schema = _prompt_contract(spec.prompt_path)
    selections: list[Selection] = []
    sampling_counts: Counter[str] = Counter()
    excluded = 0
    work_items: list[tuple[tuple[str, str], tuple[Selection, ...], WorkerConfig]] = []
    seen_sample_ids: set[str] = set()
    remaining = limit
    for source_spec in _expanded_task_sources(spec):
        if remaining == 0:
            break
        source_limit = None if source_spec.source_kind == "synthetic_point_multi" else remaining
        source_selections, source_excluded = _load_selections(
            source_spec,
            excluded_ids=excluded_ids,
            limit=source_limit,
        )
        excluded += source_excluded
        if source_spec.source_kind == "real_weak":
            source_selections, source_sampling_counts = _stratify_shape_attribute_selections(
                source_selections,
                max_rectangle_fraction=shape_attribute_max_rectangle_fraction,
                seed=seed,
            )
            sampling_counts.update(source_sampling_counts)
        elif source_spec.source_kind == "synthetic_point_multi":
            source_selections, source_sampling_counts = _filter_synthetic_multi_point_selections(
                source_spec,
                source_selections,
                seed=seed,
            )
            sampling_counts.update(source_sampling_counts)
        if remaining is not None:
            source_selections = source_selections[:remaining]
            remaining -= len(source_selections)
        duplicates = seen_sample_ids.intersection(
            selection.sample_id for selection in source_selections
        )
        if duplicates:
            raise ValueError(
                f"Duplicate sample ids across {spec.name} sources: {sorted(duplicates)[:5]}"
            )
        seen_sample_ids.update(selection.sample_id for selection in source_selections)
        selections.extend(source_selections)
        sampling_counts[f"selection_source_{source_spec.source_kind}"] += len(source_selections)
        by_source: dict[tuple[str, str], list[Selection]] = defaultdict(list)
        for selection in source_selections:
            by_source[(selection.source_json, selection.source_image)].append(selection)
        config = WorkerConfig(
            spec=source_spec,
            staging_root=staging,
            prompt_pool_id=prompt_pool_id,
            output_schema=output_schema,
            seed=seed,
            min_crop_size=min_crop_size,
            max_aspect_ratio=max_aspect_ratio,
            png_compress_level=png_compress_level,
        )
        work_items.extend(
            (source_key, tuple(items), config) for source_key, items in sorted(by_source.items())
        )
    if not selections:
        raise ValueError(f"{spec.name}: no selections remain after filtering")
    _write_selection_manifest(
        staging / "selection/train.jsonl",
        spec=spec,
        selections=selections,
    )
    counts: Counter[str] = Counter()
    try:
        with (
            (staging / "structured/train.jsonl").open("w", encoding="utf-8") as structured,
            (staging / "sft/train.jsonl").open("w", encoding="utf-8") as sft,
        ):
            if workers == 1:
                results = map(_build_source, work_items)
                executor = None
            else:
                executor = ProcessPoolExecutor(max_workers=workers)
                results = executor.map(_build_source, work_items, chunksize=chunksize)
            try:
                for result in results:
                    counts.update(result.counts)
                    for structured_line, sft_line in result.rows:
                        structured.write(structured_line + "\n")
                        sft.write(sft_line + "\n")
            finally:
                if executor is not None:
                    executor.shutdown(wait=True)
        if counts["rows"] != len(selections):
            raise RuntimeError(
                f"{spec.name}: generated {counts['rows']} != selected {len(selections)}"
            )
        counts.update(sampling_counts)
        counts["source_groups"] = len(work_items)
        counts["excluded_test_rows"] = excluded
        _write_readme(
            spec,
            staging,
            counts=counts,
            seed=seed,
            workers=workers,
            max_aspect_ratio=max_aspect_ratio,
            excluded=excluded,
            shape_attribute_max_rectangle_fraction=shape_attribute_max_rectangle_fraction,
        )
        _publish(staging, output_root / spec.name, clean=clean)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return counts


def _task_specs(args: argparse.Namespace) -> dict[str, TaskSpec]:
    synthetic_root = Path(args.synthetic_root)
    raw_root = Path(args.raw_root)
    archive_root = Path(args.archive_root)
    return {
        "shape_context_reconstruction": TaskSpec(
            name="shape_context_reconstruction",
            label="shape",
            selection_path=Path(args.shape_selection),
            source_root=synthetic_root,
            prompt_path=Path(args.shape_prompt_pool),
            source_kind="synthetic",
        ),
        "line_context_reconstruction": TaskSpec(
            name="line_context_reconstruction",
            label="line",
            selection_path=Path(args.line_selection),
            source_root=synthetic_root,
            prompt_path=Path(args.line_prompt_pool),
            source_kind="synthetic",
        ),
        "image_context_reconstruction": TaskSpec(
            name="image_context_reconstruction",
            label="image",
            selection_path=Path(args.image_selection),
            source_root=raw_root,
            prompt_path=Path(args.image_prompt_pool),
            source_kind="real",
        ),
        "shape_context_attributes": TaskSpec(
            name="shape_context_attributes",
            label="shape",
            selection_path=Path(args.shape_attribute_selection),
            source_root=raw_root,
            prompt_path=Path(args.shape_attribute_prompt_pool),
            source_kind="real_weak",
        ),
        "line_context_points": TaskSpec(
            name="line_context_points",
            label="line",
            selection_path=Path(args.line_point_selection),
            source_root=archive_root,
            prompt_path=Path(args.line_point_prompt_pool),
            source_kind="archived_point",
            source_image_manifest=Path(args.line_point_full_image_manifest),
            additional_sources=(
                TaskSourceSpec(
                    selection_path=Path(args.line_point_synthetic_selection),
                    source_root=synthetic_root,
                    source_kind="synthetic_point_multi",
                    selection_limit=args.line_point_synthetic_limit,
                ),
            ),
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build contextual-crop proposal-conditioned reconstruction SFT data."
    )
    parser.add_argument("--synthetic-root", default="data/regulated_layout_dataset_v8_20260709")
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--archive-root", default="data/archive2")
    parser.add_argument("--output-root", default="data")
    parser.add_argument(
        "--shape-selection",
        default="data/shape_context_reconstruction/selection/train.jsonl",
    )
    parser.add_argument(
        "--line-selection",
        default="data/line_context_reconstruction/selection/train.jsonl",
    )
    parser.add_argument(
        "--image-selection",
        default="data/image_context_reconstruction/selection/train.jsonl",
    )
    parser.add_argument(
        "--shape-attribute-selection",
        default="data/raw/weak_labels/shape_context_attributes_opus48_20260717.jsonl",
    )
    parser.add_argument(
        "--line-point-selection",
        default="data/archive2/point_arrow/structured/train.jsonl",
    )
    parser.add_argument(
        "--line-point-full-image-manifest",
        default="data/archive2/grounding_layout/structured/train.jsonl",
    )
    parser.add_argument(
        "--line-point-synthetic-selection",
        default="data/line_context_reconstruction/selection/train.jsonl",
        help=(
            "Selection snapshot whose gt_standard truth is filtered to synthetic "
            "is_single=false multi-segment lines."
        ),
    )
    parser.add_argument(
        "--line-point-synthetic-limit",
        type=int,
        default=15_000,
        help=(
            "Maximum synthetic multi-segment line rows, balanced across observed segment counts."
        ),
    )
    parser.add_argument(
        "--shape-prompt-pool",
        default="configs/prompts/pools/shape_context_reconstruction.v5.3.yaml",
    )
    parser.add_argument(
        "--line-prompt-pool",
        default="configs/prompts/pools/line_context_reconstruction.v5.3.yaml",
    )
    parser.add_argument(
        "--image-prompt-pool",
        default="configs/prompts/pools/image_context_reconstruction.v5.3.yaml",
    )
    parser.add_argument(
        "--shape-attribute-prompt-pool",
        default="configs/prompts/pools/shape_context_attributes.v5.3.yaml",
    )
    parser.add_argument(
        "--line-point-prompt-pool",
        default="configs/prompts/pools/line_context_points.v5.3.yaml",
    )
    parser.add_argument("--exclude-manifest", default="data/raw/splits/vlm.test.json")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=[
            "shape_context_reconstruction",
            "line_context_reconstruction",
            "image_context_reconstruction",
        ],
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunksize", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-crop-size", type=int, default=4)
    parser.add_argument("--max-aspect-ratio", type=float, default=60.0)
    parser.add_argument("--png-compress-level", type=int, default=1)
    parser.add_argument(
        "--shape-attribute-max-rectangle-fraction",
        type=float,
        default=0.5,
        help=("Keep all non-rectangle weak labels and cap rectangle rows to this final fraction."),
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    if args.workers <= 0 or args.chunksize <= 0:
        parser.error("workers and chunksize must be positive")
    if args.line_point_synthetic_limit <= 0:
        parser.error("line-point-synthetic-limit must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("limit must be positive")
    if args.min_crop_size < 1:
        parser.error("min-crop-size must be positive")
    if args.max_aspect_ratio < 1:
        parser.error("max-aspect-ratio must be >= 1")
    if not 0 <= args.png_compress_level <= 9:
        parser.error("png-compress-level must be between 0 and 9")
    if not 0 < args.shape_attribute_max_rectangle_fraction < 1:
        parser.error("shape-attribute-max-rectangle-fraction must be between 0 and 1")
    specs = _task_specs(args)
    unknown = sorted(set(args.tasks) - set(specs))
    if unknown:
        parser.error(f"unknown tasks: {unknown}")
    exclude_path = Path(args.exclude_manifest) if args.exclude_manifest else None
    excluded_ids = _read_excluded_ids(exclude_path)
    summary: dict[str, dict[str, int]] = {}
    for task in args.tasks:
        summary[task] = dict(
            sorted(
                _build_task(
                    specs[task],
                    output_root=Path(args.output_root),
                    workers=int(args.workers),
                    chunksize=int(args.chunksize),
                    clean=bool(args.clean),
                    seed=int(args.seed),
                    min_crop_size=int(args.min_crop_size),
                    max_aspect_ratio=float(args.max_aspect_ratio),
                    png_compress_level=int(args.png_compress_level),
                    excluded_ids=excluded_ids,
                    limit=args.limit,
                    shape_attribute_max_rectangle_fraction=float(
                        args.shape_attribute_max_rectangle_fraction
                    ),
                ).items()
            )
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
