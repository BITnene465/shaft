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
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")


@dataclass(frozen=True)
class GroundingTaskSpec:
    name: str
    labels: tuple[str, ...]
    required_layers: tuple[str, ...]
    max_positive_crops: int
    min_positive_instances: int
    source_label_map: tuple[tuple[str, str], ...] = ()


TASKS: tuple[GroundingTaskSpec, ...] = (
    GroundingTaskSpec(
        name="grounding_arrow",
        labels=("arrow",),
        required_layers=("arrow",),
        max_positive_crops=2,
        min_positive_instances=3,
    ),
    GroundingTaskSpec(
        name="grounding_layout",
        labels=("icon", "image", "shape", "line"),
        required_layers=(),
        max_positive_crops=2,
        min_positive_instances=4,
        source_label_map=(
            ("icon", "icon"),
            ("image", "image"),
            ("shape", "shape"),
            ("arrow", "line"),
            ("line", "line"),
        ),
    ),
    GroundingTaskSpec(
        name="grounding_shape",
        labels=("shape",),
        required_layers=("layout",),
        max_positive_crops=1,
        min_positive_instances=4,
    ),
    GroundingTaskSpec(
        name="grounding_icon_image",
        labels=("icon", "image"),
        required_layers=("layout",),
        max_positive_crops=1,
        min_positive_instances=2,
    ),
)

DEFAULT_TASK_NAMES: tuple[str, ...] = ("grounding_layout",)


@dataclass(frozen=True)
class BuildConfig:
    raw_root: Path
    task_name: str
    split: str
    output_root: Path
    image_output_dir: Path
    seed: int
    candidate_count: int
    negative_candidate_count: int
    negative_ratio: float
    density_crop_ratio: float
    blur_ratio: float
    padded_full_ratio: float
    padding_min_ratio: float
    padding_max_ratio: float
    augmentation_profile: str
    min_pixels: int
    max_pixels: int
    processor_factor: int
    clean_resize_views: float
    degraded_resize_ratio: float


@dataclass(frozen=True)
class SourceInstance:
    index: int
    label: str
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class CandidateCrop:
    crop_box: tuple[int, int, int, int]
    instance_indices: tuple[int, ...]
    score: tuple[int, float]


@dataclass(frozen=True)
class ResizePlan:
    width: int
    height: int
    target_pixels: int
    actual_pixels: int
    pixel_band: str
    kernel: str
    progressive: bool


@dataclass(frozen=True)
class DegradationPlan:
    resize_index: int
    family: str
    severity: str
    seed: int


@dataclass(frozen=True)
class SourcePlan:
    resize_plans: tuple[ResizePlan, ...] = ()
    make_padded: bool = False
    degradation_plans: tuple[DegradationPlan, ...] = ()
    resolution_quartile: int = 0
    object_count_quartile: int = 0


@dataclass(frozen=True)
class SourceMeta:
    json_rel: str
    image_width: int
    image_height: int
    target_count: int
    covered: bool


@dataclass(frozen=True)
class SourceResult:
    full_rows: list[dict[str, Any]]
    padded_rows: list[dict[str, Any]]
    resize_rows: list[dict[str, Any]]
    degraded_rows: list[dict[str, Any]]
    positive_rows: list[dict[str, Any]]
    negative_rows: list[dict[str, Any]]
    blur_rows: list[dict[str, Any]]
    covered: bool
    source_json: str
    target_count: int


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        delete=False,
        dir=path.parent,
    ) as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
        tmp_path = Path(handle.name)
    os.replace(tmp_path, path)


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    body = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    )
    _atomic_write_text(path, body)


def _read_split(path: Path) -> list[str]:
    if path.suffix.lower() == ".json":
        return _read_json_split(path)
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_json_split(path: Path) -> list[str]:
    payload = _load_json(path)
    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError(f"JSON split manifest must contain an items list: {path}")
    entries: list[str] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"JSON split manifest item {index} must be an object: {path}")
        explicit_json_path = item.get("json_path") or item.get("annotation_path")
        if isinstance(explicit_json_path, str) and explicit_json_path.strip():
            entries.append(explicit_json_path.strip())
            continue
        sample_id = str(item.get("id") or "").strip()
        if not sample_id:
            image_path = str(item.get("image_path") or "").strip()
            sample_id = Path(image_path).stem
        if not sample_id:
            raise ValueError(
                f"JSON split manifest item {index} needs json_path, annotation_path, id, "
                f"or image_path: {path}"
            )
        entries.append(str(Path("json") / f"{sample_id}.json"))
    return entries


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _source_label_map(spec: GroundingTaskSpec) -> dict[str, str]:
    if spec.source_label_map:
        return dict(spec.source_label_map)
    return {label: label for label in spec.labels}


def _rel_id(json_rel: str) -> str:
    return (
        json_rel.replace("/json/", "__")
        .replace("/", "__")
        .replace(".json", "")
    )


def _find_image_path(raw_root: Path, raw_record: dict[str, Any], json_rel: str) -> Path:
    image_path = raw_record.get("image_path")
    if image_path:
        candidate = raw_root / str(image_path)
        if candidate.exists():
            return candidate

    json_path = Path(json_rel)
    unified_image_dir = raw_root / "images"
    for suffix in IMAGE_SUFFIXES:
        candidate = unified_image_dir / f"{json_path.stem}{suffix}"
        if candidate.exists():
            return candidate

    image_dir = raw_root / json_path.parts[0] / "images"
    for suffix in IMAGE_SUFFIXES:
        candidate = image_dir / f"{json_path.stem}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot find image for {json_rel}")


def _clean_bbox(
    value: Any,
    *,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in (x1, y1, x2, y2)):
        return None
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    x1 = min(max(x1, 0.0), float(image_width))
    x2 = min(max(x2, 0.0), float(image_width))
    y1 = min(max(y1, 0.0), float(image_height))
    y2 = min(max(y2, 0.0), float(image_height))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _extract_instances(
    raw_record: dict[str, Any],
    label_map: dict[str, str],
    *,
    image_width: int,
    image_height: int,
) -> list[SourceInstance]:
    result: list[SourceInstance] = []
    for index, instance in enumerate(raw_record.get("instances") or []):
        source_label = str(instance.get("label"))
        target_label = label_map.get(source_label)
        if target_label is None:
            continue
        bbox = _clean_bbox(
            instance.get("bbox"),
            image_width=image_width,
            image_height=image_height,
        )
        if bbox is None:
            continue
        result.append(SourceInstance(index=index, label=target_label, bbox=bbox))
    return result


def _has_required_coverage(raw_record: dict[str, Any], required_layers: tuple[str, ...]) -> bool:
    annotation = raw_record.get("annotation") or {}
    if "layers" not in annotation:
        return True
    layers = set(annotation.get("layers") or [])
    return set(required_layers).issubset(layers)


def _bbox_inside(bbox: tuple[float, float, float, float], crop: tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = bbox
    left, top, right, bottom = crop
    return x1 >= left and y1 >= top and x2 <= right and y2 <= bottom


def _bbox_intersects(
    bbox: tuple[float, float, float, float],
    crop: tuple[int, int, int, int],
) -> bool:
    x1, y1, x2, y2 = bbox
    left, top, right, bottom = crop
    return min(x2, right) > max(x1, left) and min(y2, bottom) > max(y1, top)


def _expand_crop_to_avoid_partials(
    crop: tuple[int, int, int, int],
    instances: list[SourceInstance],
    *,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = crop
    for _ in range(8):
        changed = False
        for instance in instances:
            if not _bbox_intersects(instance.bbox, (left, top, right, bottom)):
                continue
            if _bbox_inside(instance.bbox, (left, top, right, bottom)):
                continue
            x1, y1, x2, y2 = instance.bbox
            new_left = max(0, min(left, int(math.floor(x1))))
            new_top = max(0, min(top, int(math.floor(y1))))
            new_right = min(image_width, max(right, int(math.ceil(x2))))
            new_bottom = min(image_height, max(bottom, int(math.ceil(y2))))
            changed = changed or (new_left, new_top, new_right, new_bottom) != (
                left,
                top,
                right,
                bottom,
            )
            left, top, right, bottom = new_left, new_top, new_right, new_bottom
        if not changed:
            break
    return left, top, right, bottom


def _crop_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    inter = max(0, right - left) * max(0, bottom - top)
    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / float(area_a + area_b - inter)


def _crop_ranges(image_width: int, image_height: int) -> tuple[float, float]:
    max_side = max(image_width, image_height)
    if max_side >= 2400:
        return 0.28, 0.72
    if max_side >= 1500:
        return 0.35, 0.80
    return 0.45, 0.90


def _make_crop_around_anchor(
    rng: random.Random,
    anchor: SourceInstance,
    *,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    min_ratio, max_ratio = _crop_ranges(image_width, image_height)
    width = max(64, int(round(image_width * rng.uniform(min_ratio, max_ratio))))
    height = max(64, int(round(image_height * rng.uniform(min_ratio, max_ratio))))
    width = min(width, image_width)
    height = min(height, image_height)
    x1, y1, x2, y2 = anchor.bbox
    center_x = (x1 + x2) / 2.0 + rng.uniform(-0.20, 0.20) * width
    center_y = (y1 + y2) / 2.0 + rng.uniform(-0.20, 0.20) * height
    left = int(round(center_x - width / 2.0))
    top = int(round(center_y - height / 2.0))
    left = min(max(left, 0), max(0, image_width - width))
    top = min(max(top, 0), max(0, image_height - height))
    return left, top, left + width, top + height


def _evaluate_positive_crop(
    crop: tuple[int, int, int, int],
    instances: list[SourceInstance],
    *,
    image_width: int,
    image_height: int,
) -> CandidateCrop | None:
    crop = _expand_crop_to_avoid_partials(
        crop,
        instances,
        image_width=image_width,
        image_height=image_height,
    )
    crop_area_ratio = ((crop[2] - crop[0]) * (crop[3] - crop[1])) / float(
        image_width * image_height
    )
    if crop_area_ratio > 0.92:
        return None
    inside: list[int] = []
    total_area = 0.0
    for instance in instances:
        intersects = _bbox_intersects(instance.bbox, crop)
        complete = _bbox_inside(instance.bbox, crop)
        if intersects and not complete:
            return None
        if complete:
            inside.append(instance.index)
            x1, y1, x2, y2 = instance.bbox
            total_area += (x2 - x1) * (y2 - y1)
    if not inside:
        return None
    return CandidateCrop(
        crop_box=crop,
        instance_indices=tuple(sorted(inside)),
        score=(len(inside), total_area),
    )


def _select_positive_crops(
    instances: list[SourceInstance],
    *,
    image_width: int,
    image_height: int,
    max_crops: int,
    min_instances: int,
    rng: random.Random,
    candidate_count: int,
) -> list[CandidateCrop]:
    if len(instances) < min_instances:
        return []
    candidates: list[CandidateCrop] = []
    for _ in range(candidate_count):
        anchor = rng.choice(instances)
        crop = _make_crop_around_anchor(
            rng,
            anchor,
            image_width=image_width,
            image_height=image_height,
        )
        candidate = _evaluate_positive_crop(
            crop,
            instances,
            image_width=image_width,
            image_height=image_height,
        )
        if candidate is not None and len(candidate.instance_indices) >= min_instances:
            candidates.append(candidate)
    candidates.sort(key=lambda item: (-item.score[0], -item.score[1], item.crop_box))
    selected: list[CandidateCrop] = []
    seen_sets: set[tuple[int, ...]] = set()
    for candidate in candidates:
        if candidate.instance_indices in seen_sets:
            continue
        if any(_crop_iou(candidate.crop_box, item.crop_box) > 0.85 for item in selected):
            continue
        selected.append(candidate)
        seen_sets.add(candidate.instance_indices)
        if len(selected) >= max_crops:
            break
    return selected


def _select_negative_crop(
    instances: list[SourceInstance],
    *,
    image_width: int,
    image_height: int,
    rng: random.Random,
    candidate_count: int,
) -> tuple[int, int, int, int] | None:
    min_ratio, max_ratio = _crop_ranges(image_width, image_height)
    for _ in range(candidate_count):
        width = max(64, int(round(image_width * rng.uniform(min_ratio * 0.75, max_ratio))))
        height = max(64, int(round(image_height * rng.uniform(min_ratio * 0.75, max_ratio))))
        width = min(width, image_width)
        height = min(height, image_height)
        if width == image_width and height == image_height:
            continue
        left = rng.randint(0, max(0, image_width - width))
        top = rng.randint(0, max(0, image_height - height))
        crop = (left, top, left + width, top + height)
        if any(_bbox_intersects(instance.bbox, crop) for instance in instances):
            continue
        return crop
    return None


def _translate_instances(
    instances: list[SourceInstance],
    instance_indices: set[int],
    crop_box: tuple[int, int, int, int],
) -> list[dict[str, Any]]:
    left, top, _, _ = crop_box
    result: list[dict[str, Any]] = []
    for instance in instances:
        if instance.index not in instance_indices:
            continue
        x1, y1, x2, y2 = instance.bbox
        result.append(
            {
                "label": instance.label,
                "bbox": [x1 - left, y1 - top, x2 - left, y2 - top],
            }
        )
    result.sort(key=lambda item: (item["bbox"][1], item["bbox"][0], item["label"]))
    return result


def _full_instances(instances: list[SourceInstance]) -> list[dict[str, Any]]:
    result = [
        {
            "label": instance.label,
            "bbox": list(instance.bbox),
        }
        for instance in instances
    ]
    result.sort(key=lambda item: (item["bbox"][1], item["bbox"][0], item["label"]))
    return result


def _sorted_source_instance_indices(
    instances: list[SourceInstance],
    selected: set[int] | None = None,
) -> list[int]:
    filtered = [
        instance
        for instance in instances
        if selected is None or instance.index in selected
    ]
    filtered.sort(
        key=lambda instance: (
            instance.bbox[1],
            instance.bbox[0],
            instance.label,
            instance.index,
        )
    )
    return [instance.index for instance in filtered]


def _offset_full_instances(
    instances: list[SourceInstance],
    *,
    offset_x: int,
    offset_y: int,
) -> list[dict[str, Any]]:
    result = [
        {
            "label": instance.label,
            "bbox": [
                instance.bbox[0] + offset_x,
                instance.bbox[1] + offset_y,
                instance.bbox[2] + offset_x,
                instance.bbox[3] + offset_y,
            ],
        }
        for instance in instances
    ]
    result.sort(key=lambda item: (item["bbox"][1], item["bbox"][0], item["label"]))
    return result


def _resampling(name: str) -> int:
    resampling = getattr(Image, "Resampling", Image)
    return getattr(resampling, name)


PIXEL_BANDS: tuple[tuple[str, int, int], ...] = (
    ("0.2-0.5M", 200_704, 500_000),
    ("0.5-1M", 500_000, 1_000_000),
    ("1-2M", 1_000_000, 2_000_000),
    ("2-4M", 2_000_000, 4_000_001),
)


def _stable_seed(*parts: object) -> int:
    body = ":".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(body).digest()[:8], "big")


def _pixel_band(pixel_count: int) -> str:
    for name, lower, upper in PIXEL_BANDS:
        if lower <= pixel_count < upper:
            return name
    return "below-0.2M" if pixel_count < PIXEL_BANDS[0][1] else "above-4M"


def _aligned_size_for_target(
    *,
    width: int,
    height: int,
    target_pixels: int,
    min_pixels: int,
    max_pixels: int,
    factor: int,
    max_upscale: float | None,
) -> tuple[int, int] | None:
    if width <= 0 or height <= 0 or factor <= 0 or min_pixels > max_pixels:
        return None
    scale = math.sqrt(target_pixels / float(width * height))
    ideal_width = width * scale
    ideal_height = height * scale
    candidates: set[tuple[int, int]] = set()
    base_width = max(factor, int(round(ideal_width / factor)) * factor)
    base_height = max(factor, int(round(ideal_height / factor)) * factor)
    for delta in range(-4, 5):
        candidate_width = max(factor, base_width + delta * factor)
        candidate_height = max(
            factor,
            int(round(candidate_width * height / width / factor)) * factor,
        )
        candidates.add((candidate_width, candidate_height))
        candidate_height = max(factor, base_height + delta * factor)
        candidate_width = max(
            factor,
            int(round(candidate_height * width / height / factor)) * factor,
        )
        candidates.add((candidate_width, candidate_height))

    valid: list[tuple[float, float, int, int]] = []
    source_ratio = width / float(height)
    for candidate_width, candidate_height in candidates:
        pixels = candidate_width * candidate_height
        if not min_pixels <= pixels <= max_pixels:
            continue
        if max_upscale is not None and (
            candidate_width / width > max_upscale + 1e-9
            or candidate_height / height > max_upscale + 1e-9
        ):
            continue
        target_error = abs(math.log(pixels / float(target_pixels)))
        ratio_error = abs(math.log((candidate_width / candidate_height) / source_ratio))
        valid.append((target_error, ratio_error, candidate_width, candidate_height))
    if not valid:
        return None
    _, _, output_width, output_height = min(
        valid,
        key=lambda item: (item[0] + item[1], item[1], item[0]),
    )
    return output_width, output_height


def _choose_resize_kernel(
    rng: random.Random,
    *,
    source_width: int,
    source_height: int,
    output_width: int,
    output_height: int,
) -> tuple[str, bool]:
    scale = math.sqrt(
        (output_width * output_height) / float(source_width * source_height)
    )
    draw = rng.random()
    if scale < 1.0:
        if draw < 0.60:
            kernel = "bicubic"
        elif draw < 0.85:
            kernel = "lanczos"
        else:
            kernel = "area"
    else:
        kernel = "bicubic" if draw < 0.75 else "lanczos"
    progressive = scale < 0.5 and kernel in {"lanczos", "area"}
    return kernel, progressive


def _resize_image(image: Image.Image, plan: ResizePlan) -> Image.Image:
    kernel_map = {
        "bicubic": _resampling("BICUBIC"),
        "lanczos": _resampling("LANCZOS"),
        "area": _resampling("BOX"),
    }
    working = image
    owned = False
    if plan.progressive:
        while working.width > plan.width * 2 or working.height > plan.height * 2:
            next_size = (
                max(plan.width, int(round(working.width / 2))),
                max(plan.height, int(round(working.height / 2))),
            )
            resized = working.resize(next_size, _resampling("BOX"))
            if owned:
                working.close()
            working = resized
            owned = True
    result = working.resize((plan.width, plan.height), kernel_map[plan.kernel])
    if owned:
        working.close()
    return result


def _scale_instances(
    instances: list[SourceInstance],
    *,
    scale_x: float,
    scale_y: float,
    max_x: int,
    max_y: int,
    offset_x: int = 0,
    offset_y: int = 0,
) -> list[dict[str, Any]]:
    result = []
    for instance in instances:
        x1, y1, x2, y2 = instance.bbox
        result.append(
            {
                "label": instance.label,
                "bbox": [
                    min(max(x1 * scale_x + offset_x, 0.0), float(max_x)),
                    min(max(y1 * scale_y + offset_y, 0.0), float(max_y)),
                    min(max(x2 * scale_x + offset_x, 0.0), float(max_x)),
                    min(max(y2 * scale_y + offset_y, 0.0), float(max_y)),
                ],
            }
        )
    result.sort(key=lambda item: (item["bbox"][1], item["bbox"][0], item["label"]))
    return result


def _edge_median_color(image: Image.Image) -> tuple[int, int, int]:
    rgb = image.convert("RGB")
    array = np.asarray(rgb)
    edges = np.concatenate(
        (array[0, :, :], array[-1, :, :], array[:, 0, :], array[:, -1, :]),
        axis=0,
    )
    median = np.median(edges, axis=0)
    return tuple(int(round(value)) for value in median)


def _make_asymmetric_padded_image(
    image: Image.Image,
    rng: random.Random,
    *,
    min_ratio: float,
    max_ratio: float,
    factor: int,
    max_pixels: int,
) -> tuple[Image.Image, dict[str, Any], tuple[int, int, int, int]]:
    width, height = image.size
    horizontal_ratio = rng.uniform(min_ratio, max_ratio)
    vertical_ratio = rng.uniform(min_ratio, max_ratio)
    canvas_width = int(math.ceil(width * (1.0 + horizontal_ratio) / factor)) * factor
    canvas_height = int(math.ceil(height * (1.0 + vertical_ratio) / factor)) * factor
    if canvas_width * canvas_height > max_pixels:
        raise ValueError(
            f"Padded canvas exceeds pixel budget: {(canvas_width, canvas_height)} > {max_pixels}"
        )
    available_x = canvas_width - width
    available_y = canvas_height - height
    offset_x = rng.randint(0, available_x)
    offset_y = rng.randint(0, available_y)
    canvas_color = _edge_median_color(image)
    padded = Image.new("RGB", (canvas_width, canvas_height), canvas_color)
    padded.paste(image, (offset_x, offset_y))
    pads = {
        "left": offset_x,
        "right": available_x - offset_x,
        "top": offset_y,
        "bottom": available_y - offset_y,
    }
    augmentation = {
        "name": "random_padded_full",
        "requested_expansion_ratio": {
            "horizontal": round(horizontal_ratio, 6),
            "vertical": round(vertical_ratio, 6),
        },
        "padding": pads,
        "canvas_color": list(canvas_color),
        "processor_factor": factor,
    }
    content_box = (offset_x, offset_y, offset_x + width, offset_y + height)
    return padded, augmentation, content_box


def _degradation_parameters(plan: DegradationPlan, short_edge: int) -> dict[str, Any]:
    if plan.family == "gaussian_blur":
        multipliers = {"L1": (0.4, 0.0004), "L2": (0.8, 0.0008), "L3": (1.2, 0.0015)}
        floor, ratio = multipliers[plan.severity]
        return {
            "name": plan.family,
            "severity": plan.severity,
            "radius": round(max(floor, short_edge * ratio), 4),
        }
    sigma = {"L1": 2.0, "L2": 5.0, "L3": 10.0}[plan.severity]
    return {
        "name": "gaussian_noise",
        "severity": plan.severity,
        "sigma_255": sigma,
        "seed": plan.seed,
    }


def _apply_degradation(image: Image.Image, parameters: dict[str, Any]) -> Image.Image:
    if parameters["name"] == "gaussian_blur":
        return image.filter(ImageFilter.GaussianBlur(radius=float(parameters["radius"])))
    array = np.asarray(image.convert("RGB"), dtype=np.float32)
    noise = np.random.default_rng(int(parameters["seed"])).normal(
        0.0,
        float(parameters["sigma_255"]),
        size=array.shape,
    )
    return Image.fromarray(np.clip(array + noise, 0, 255).astype(np.uint8), mode="RGB")


def _read_source_meta(args: tuple[str, GroundingTaskSpec, Path]) -> SourceMeta:
    json_rel, spec, raw_root = args
    raw_record = _load_json(raw_root / json_rel)
    covered = _has_required_coverage(raw_record, spec.required_layers)
    image_path = _find_image_path(raw_root, raw_record, json_rel)
    with Image.open(image_path) as image:
        width, height = image.size
    instances = _extract_instances(
        raw_record,
        _source_label_map(spec),
        image_width=width,
        image_height=height,
    )
    return SourceMeta(json_rel, width, height, len(instances), covered)


def _quartiles(metas: list[SourceMeta], value: Any) -> dict[str, int]:
    ordered = sorted(metas, key=lambda item: (value(item), item.json_rel))
    count = len(ordered)
    return {
        meta.json_rel: min(3, index * 4 // max(1, count))
        for index, meta in enumerate(ordered)
    }


def _select_stratified_sources(
    metas: list[SourceMeta],
    *,
    target_count: int,
    seed: int,
    namespace: str,
    resolution_quartiles: dict[str, int],
    object_quartiles: dict[str, int],
) -> set[str]:
    target_count = min(max(0, target_count), len(metas))
    groups: dict[tuple[int, int], list[SourceMeta]] = {}
    for meta in metas:
        key = (resolution_quartiles[meta.json_rel], object_quartiles[meta.json_rel])
        groups.setdefault(key, []).append(meta)
    allocations: dict[tuple[int, int], int] = {}
    remainders: list[tuple[float, tuple[int, int]]] = []
    for key, group in groups.items():
        exact = target_count * len(group) / max(1, len(metas))
        allocations[key] = int(math.floor(exact))
        remainders.append((exact - allocations[key], key))
    remaining = target_count - sum(allocations.values())
    for _, key in sorted(remainders, key=lambda item: (-item[0], item[1]))[:remaining]:
        allocations[key] += 1

    selected: set[str] = set()
    for key, group in sorted(groups.items()):
        rng = random.Random(_stable_seed(seed, namespace, key))
        candidates = sorted(group, key=lambda item: item.json_rel)
        rng.shuffle(candidates)
        selected.update(meta.json_rel for meta in candidates[: allocations[key]])
    return selected


def _sample_resize_plan(
    meta: SourceMeta,
    *,
    preferred_band: tuple[str, int, int],
    selected: list[ResizePlan],
    config: BuildConfig,
    rng: random.Random,
) -> ResizePlan | None:
    source_pixels = meta.image_width * meta.image_height
    upper = min(config.max_pixels, source_pixels * 4)
    lower = config.min_pixels
    band_name, band_lower, band_upper = preferred_band
    sample_lower = max(lower, band_lower)
    sample_upper = min(upper, band_upper - 1)
    if sample_upper < sample_lower:
        return None
    native_target = min(max(source_pixels, config.min_pixels), config.max_pixels)
    native_size = _aligned_size_for_target(
        width=meta.image_width,
        height=meta.image_height,
        target_pixels=native_target,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
        factor=config.processor_factor,
        max_upscale=None,
    )
    native_pixels = native_size[0] * native_size[1] if native_size else source_pixels
    candidates: list[tuple[float, ResizePlan]] = []
    for _ in range(64):
        target = int(round(math.exp(rng.uniform(math.log(sample_lower), math.log(sample_upper)))))
        size = _aligned_size_for_target(
            width=meta.image_width,
            height=meta.image_height,
            target_pixels=target,
            min_pixels=lower,
            max_pixels=upper,
            factor=config.processor_factor,
            max_upscale=2.0,
        )
        if size is None:
            continue
        output_width, output_height = size
        actual_pixels = output_width * output_height
        if abs(actual_pixels / native_pixels - 1.0) <= 0.10:
            continue
        if any(
            max(actual_pixels, plan.actual_pixels) / min(actual_pixels, plan.actual_pixels) < 1.35
            for plan in selected
        ):
            continue
        kernel, progressive = _choose_resize_kernel(
            rng,
            source_width=meta.image_width,
            source_height=meta.image_height,
            output_width=output_width,
            output_height=output_height,
        )
        plan = ResizePlan(
            width=output_width,
            height=output_height,
            target_pixels=target,
            actual_pixels=actual_pixels,
            pixel_band=_pixel_band(actual_pixels),
            kernel=kernel,
            progressive=progressive,
        )
        candidates.append((abs(math.log(actual_pixels / target)), plan))
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]


def _build_multiscale_plans(
    metas: list[SourceMeta],
    *,
    config: BuildConfig,
) -> dict[str, SourcePlan]:
    covered = [meta for meta in metas if meta.covered]
    resolution_quartiles = _quartiles(
        covered,
        lambda item: item.image_width * item.image_height,
    )
    object_quartiles = _quartiles(covered, lambda item: item.target_count)
    padded_target = int(round(len(covered) * config.padded_full_ratio))
    padded_sources = _select_stratified_sources(
        covered,
        target_count=padded_target,
        seed=config.seed,
        namespace=f"{config.task_name}:padding",
        resolution_quartiles=resolution_quartiles,
        object_quartiles=object_quartiles,
    )
    total_clean_spatial = config.clean_resize_views + config.padded_full_ratio
    clean_slots = int(round(total_clean_spatial))
    if not math.isclose(total_clean_spatial, clean_slots, abs_tol=1e-6):
        raise ValueError("clean resize plus padded ratios must form an integer per-source budget")

    ordered = sorted(
        covered,
        key=lambda item: (
            resolution_quartiles[item.json_rel],
            object_quartiles[item.json_rel],
            _stable_seed(config.seed, config.task_name, "resize-order", item.json_rel),
        ),
    )
    band_counts: Counter[str] = Counter()
    resize_plans: dict[str, list[ResizePlan]] = {}
    for meta in ordered:
        requested = clean_slots - int(meta.json_rel in padded_sources)
        selected: list[ResizePlan] = []
        rng = random.Random(_stable_seed(config.seed, config.task_name, "resize", meta.json_rel))
        for _ in range(requested):
            source_pixels = meta.image_width * meta.image_height
            upper = min(config.max_pixels, source_pixels * 4)
            feasible = [
                band
                for band in PIXEL_BANDS
                if max(config.min_pixels, band[1]) <= min(upper, band[2] - 1)
            ]
            if not feasible:
                break
            ranked = sorted(
                feasible,
                key=lambda band: (
                    band_counts[band[0]],
                    sum(plan.pixel_band == band[0] for plan in selected),
                    rng.random(),
                ),
            )
            plan = None
            for band in ranked:
                plan = _sample_resize_plan(
                    meta,
                    preferred_band=band,
                    selected=selected,
                    config=config,
                    rng=rng,
                )
                if plan is not None:
                    break
            if plan is None:
                break
            selected.append(plan)
            band_counts[plan.pixel_band] += 1
        resize_plans[meta.json_rel] = selected

    eligible = [meta for meta in covered if resize_plans[meta.json_rel]]
    degradation_target = min(
        int(round(len(covered) * config.degraded_resize_ratio)),
        len(eligible) * 2,
    )
    if degradation_target <= len(eligible):
        primary_sources = _select_stratified_sources(
            eligible,
            target_count=degradation_target,
            seed=config.seed,
            namespace=f"{config.task_name}:primary-degradation",
            resolution_quartiles=resolution_quartiles,
            object_quartiles=object_quartiles,
        )
        second_sources: set[str] = set()
    else:
        primary_sources = {meta.json_rel for meta in eligible}
        second_sources = _select_stratified_sources(
            eligible,
            target_count=degradation_target - len(eligible),
            seed=config.seed,
            namespace=f"{config.task_name}:second-degradation",
            resolution_quartiles=resolution_quartiles,
            object_quartiles=object_quartiles,
        )
    severity_cycle = ("L1",) * 8 + ("L2",) * 7 + ("L3",) * 5
    degradation_index = 0
    plans: dict[str, SourcePlan] = {}
    for meta in ordered:
        source_resize_plans = resize_plans[meta.json_rel]
        degradation_count = int(meta.json_rel in primary_sources) + int(
            meta.json_rel in second_sources
        )
        degradation_plans: list[DegradationPlan] = []
        for local_index in range(degradation_count):
            resize_index = local_index % len(source_resize_plans)
            family = "gaussian_blur" if degradation_index % 2 == 0 else "gaussian_noise"
            severity = severity_cycle[degradation_index % len(severity_cycle)]
            if severity == "L3" and source_resize_plans[resize_index].pixel_band == "0.2-0.5M":
                non_low_indices = [
                    index
                    for index, resize_plan in enumerate(source_resize_plans)
                    if resize_plan.pixel_band != "0.2-0.5M"
                ]
                if non_low_indices:
                    resize_index = non_low_indices[local_index % len(non_low_indices)]
                else:
                    severity = "L2"
            if degradation_plans:
                previous = degradation_plans[-1]
                if (
                    previous.resize_index == resize_index
                    and previous.family == family
                    and previous.severity == severity
                ):
                    family = "gaussian_noise" if family == "gaussian_blur" else "gaussian_blur"
            degradation_plans.append(
                DegradationPlan(
                    resize_index=resize_index,
                    family=family,
                    severity=severity,
                    seed=_stable_seed(
                        config.seed,
                        config.task_name,
                        meta.json_rel,
                        "degradation",
                        local_index,
                    ),
                )
            )
            degradation_index += 1
        plans[meta.json_rel] = SourcePlan(
            resize_plans=tuple(source_resize_plans),
            make_padded=meta.json_rel in padded_sources,
            degradation_plans=tuple(degradation_plans),
            resolution_quartile=resolution_quartiles[meta.json_rel],
            object_count_quartile=object_quartiles[meta.json_rel],
        )
    return plans


def _choose_blur_augmentation(
    rng: random.Random,
    *,
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    high_resolution = max(image_width, image_height) >= 1500
    choices = ["gaussian_blur", "jpeg_compression"]
    if high_resolution:
        choices.append("resize_blur")
    name = rng.choice(choices)
    if name == "gaussian_blur":
        return {
            "name": "gaussian_blur",
            "radius": round(rng.uniform(0.35, 1.25), 3),
        }
    if name == "resize_blur":
        return {
            "name": "resize_blur",
            "scale_down_ratio": round(rng.uniform(0.45, 0.8), 3),
        }
    return {
        "name": "jpeg_compression",
        "jpeg_quality": rng.randint(60, 88),
    }


def _make_padded_full_image(
    image: Image.Image,
    rng: random.Random,
    *,
    min_ratio: float,
    max_ratio: float,
) -> tuple[Image.Image, dict[str, Any], tuple[int, int, int, int]]:
    width, height = image.size
    pads = {
        "left": int(round(width * rng.uniform(min_ratio, max_ratio))),
        "right": int(round(width * rng.uniform(min_ratio, max_ratio))),
        "top": int(round(height * rng.uniform(min_ratio, max_ratio))),
        "bottom": int(round(height * rng.uniform(min_ratio, max_ratio))),
    }
    padded = Image.new(
        "RGB",
        (width + pads["left"] + pads["right"], height + pads["top"] + pads["bottom"]),
        (255, 255, 255),
    )
    padded.paste(image, (pads["left"], pads["top"]))
    augmentation = {
        "name": "random_padded_full",
        "padding_ratio_range": [min_ratio, max_ratio],
        "padding": pads,
    }
    content_box = (
        pads["left"],
        pads["top"],
        pads["left"] + width,
        pads["top"] + height,
    )
    return padded, augmentation, content_box


def _apply_pixel_augmentation(image: Image.Image, augmentation: dict[str, Any]) -> Image.Image:
    name = augmentation.get("name")
    if name in {"jpeg_blur", "jpeg_compression"}:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=int(augmentation["jpeg_quality"]))
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")
    if name == "gaussian_blur":
        return image.filter(ImageFilter.GaussianBlur(radius=float(augmentation["radius"])))
    if name == "resize_blur":
        ratio = float(augmentation["scale_down_ratio"])
        width, height = image.size
        down_size = (max(1, int(width * ratio)), max(1, int(height * ratio)))
        return image.resize(down_size, _resampling("BILINEAR")).resize(
            (width, height),
            _resampling("BILINEAR"),
        )
    return image


def _save_generated_image(
    image: Image.Image,
    output_path: Path,
    augmentation: dict[str, Any],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_image = _apply_pixel_augmentation(image, augmentation)
    final_image.save(output_path)
    if final_image is not image:
        final_image.close()


def _build_row(
    *,
    task_name: str,
    split: str,
    sample_id: str,
    image_path: str,
    image_width: int,
    image_height: int,
    instances: list[dict[str, Any]],
    raw_record: dict[str, Any],
    json_rel: str,
    target_labels: tuple[str, ...],
    view_type: str,
    crop_box: tuple[int, int, int, int],
    source_instance_indices: list[int],
    pixel_augmentation: dict[str, Any],
    spatial_augmentation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "image_path": image_path,
        "image_width": image_width,
        "image_height": image_height,
        "instances": instances,
        "extra": {
            "task": task_name,
            "split": split,
            "target_labels": list(target_labels),
            "view_type": view_type,
            "source_json": json_rel,
            "source_image": raw_record.get("image_path"),
            "source_image_width": raw_record.get("image_width"),
            "source_image_height": raw_record.get("image_height"),
            "coverage_layers": list((raw_record.get("annotation") or {}).get("layers") or []),
            "crop_box": list(crop_box),
            "source_instance_indices": source_instance_indices,
            "pixel_augmentation": pixel_augmentation,
            "spatial_augmentation": spatial_augmentation or {"name": "none"},
        },
    }


def _build_multiscale_views(
    *,
    image: Image.Image,
    instances: list[SourceInstance],
    raw_record: dict[str, Any],
    json_rel: str,
    rel_id: str,
    spec: GroundingTaskSpec,
    config: BuildConfig,
    plan: SourcePlan,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    resize_rows: list[dict[str, Any]] = []
    padded_rows: list[dict[str, Any]] = []
    degraded_rows: list[dict[str, Any]] = []
    image_width, image_height = image.size
    source_indices = _sorted_source_instance_indices(instances)

    for resize_index, resize_plan in enumerate(plan.resize_plans):
        resized = _resize_image(image, resize_plan)
        output_name = f"{rel_id}__resize_{resize_index:02d}.png"
        _save_generated_image(
            resized,
            config.image_output_dir / output_name,
            {"name": "none"},
        )
        spatial = {
            "name": "continuous_resize_full",
            "source_size": [image_width, image_height],
            "requested_target_pixels": resize_plan.target_pixels,
            "output_size": [resize_plan.width, resize_plan.height],
            "actual_pixels": resize_plan.actual_pixels,
            "pixel_band": resize_plan.pixel_band,
            "kernel": resize_plan.kernel,
            "progressive": resize_plan.progressive,
            "scale_x": resize_plan.width / image_width,
            "scale_y": resize_plan.height / image_height,
            "processor_factor": config.processor_factor,
            "resolution_quartile": plan.resolution_quartile,
            "object_count_quartile": plan.object_count_quartile,
        }
        resize_rows.append(
            _build_row(
                task_name=spec.name,
                split=config.split,
                sample_id=f"{rel_id}__resize_{resize_index:02d}",
                image_path=f"../images/{config.split}/{output_name}",
                image_width=resize_plan.width,
                image_height=resize_plan.height,
                instances=_scale_instances(
                    instances,
                    scale_x=resize_plan.width / image_width,
                    scale_y=resize_plan.height / image_height,
                    max_x=resize_plan.width,
                    max_y=resize_plan.height,
                ),
                raw_record=raw_record,
                json_rel=json_rel,
                target_labels=spec.labels,
                view_type="continuous_resize_full",
                crop_box=(0, 0, image_width, image_height),
                source_instance_indices=source_indices,
                pixel_augmentation={"name": "none"},
                spatial_augmentation=spatial,
            )
        )
        resized.close()

    if plan.make_padded:
        if plan.resize_plans:
            base_index = min(
                range(len(plan.resize_plans)),
                key=lambda index: plan.resize_plans[index].actual_pixels,
            )
            base_plan = plan.resize_plans[base_index]
            base_image = _resize_image(image, base_plan)
            base_scale_x = base_plan.width / image_width
            base_scale_y = base_plan.height / image_height
            base_view = f"continuous_resize_full:{base_index}"
            base_kernel = base_plan.kernel
        else:
            base_image = image.copy()
            base_scale_x = 1.0
            base_scale_y = 1.0
            base_view = "full_image"
            base_kernel = "none"
        padded, spatial, content_box = _make_asymmetric_padded_image(
            base_image,
            rng,
            min_ratio=config.padding_min_ratio,
            max_ratio=config.padding_max_ratio,
            factor=config.processor_factor,
            max_pixels=config.max_pixels,
        )
        spatial.update(
            {
                "base_view": base_view,
                "base_size": list(base_image.size),
                "base_kernel": base_kernel,
                "resolution_quartile": plan.resolution_quartile,
                "object_count_quartile": plan.object_count_quartile,
            }
        )
        output_name = f"{rel_id}__padded_full.png"
        _save_generated_image(
            padded,
            config.image_output_dir / output_name,
            {"name": "none"},
        )
        padded_rows.append(
            _build_row(
                task_name=spec.name,
                split=config.split,
                sample_id=f"{rel_id}__padded_full",
                image_path=f"../images/{config.split}/{output_name}",
                image_width=padded.width,
                image_height=padded.height,
                instances=_scale_instances(
                    instances,
                    scale_x=base_scale_x,
                    scale_y=base_scale_y,
                    max_x=padded.width,
                    max_y=padded.height,
                    offset_x=content_box[0],
                    offset_y=content_box[1],
                ),
                raw_record=raw_record,
                json_rel=json_rel,
                target_labels=spec.labels,
                view_type="random_padded_full",
                crop_box=content_box,
                source_instance_indices=source_indices,
                pixel_augmentation={"name": "none"},
                spatial_augmentation=spatial,
            )
        )
        padded.close()
        base_image.close()

    for degradation_index, degradation_plan in enumerate(plan.degradation_plans):
        resize_plan = plan.resize_plans[degradation_plan.resize_index]
        resized = _resize_image(image, resize_plan)
        parameters = _degradation_parameters(
            degradation_plan,
            min(resize_plan.width, resize_plan.height),
        )
        degraded = _apply_degradation(resized, parameters)
        output_name = f"{rel_id}__degraded_{degradation_index:02d}.png"
        _save_generated_image(
            degraded,
            config.image_output_dir / output_name,
            {"name": "none"},
        )
        spatial = {
            "name": "continuous_resize_full",
            "clean_counterpart_sample_id": (
                f"{rel_id}__resize_{degradation_plan.resize_index:02d}"
            ),
            "source_size": [image_width, image_height],
            "output_size": [resize_plan.width, resize_plan.height],
            "actual_pixels": resize_plan.actual_pixels,
            "pixel_band": resize_plan.pixel_band,
            "kernel": resize_plan.kernel,
            "progressive": resize_plan.progressive,
            "processor_factor": config.processor_factor,
            "resolution_quartile": plan.resolution_quartile,
            "object_count_quartile": plan.object_count_quartile,
        }
        degraded_rows.append(
            _build_row(
                task_name=spec.name,
                split=config.split,
                sample_id=f"{rel_id}__degraded_{degradation_index:02d}",
                image_path=f"../images/{config.split}/{output_name}",
                image_width=resize_plan.width,
                image_height=resize_plan.height,
                instances=_scale_instances(
                    instances,
                    scale_x=resize_plan.width / image_width,
                    scale_y=resize_plan.height / image_height,
                    max_x=resize_plan.width,
                    max_y=resize_plan.height,
                ),
                raw_record=raw_record,
                json_rel=json_rel,
                target_labels=spec.labels,
                view_type="degraded_resize_full",
                crop_box=(0, 0, image_width, image_height),
                source_instance_indices=source_indices,
                pixel_augmentation=parameters,
                spatial_augmentation=spatial,
            )
        )
        degraded.close()
        resized.close()
    return resize_rows, padded_rows, degraded_rows


def _process_source(
    args: tuple[str, GroundingTaskSpec, BuildConfig, SourcePlan | None],
) -> SourceResult:
    json_rel, spec, config, source_plan = args
    raw_record = _load_json(config.raw_root / json_rel)
    if not _has_required_coverage(raw_record, spec.required_layers):
        return SourceResult(
            full_rows=[],
            padded_rows=[],
            resize_rows=[],
            degraded_rows=[],
            positive_rows=[],
            negative_rows=[],
            blur_rows=[],
            covered=False,
            source_json=json_rel,
            target_count=0,
        )

    image_path = _find_image_path(config.raw_root, raw_record, json_rel)
    image = Image.open(image_path).convert("RGB")
    image_width, image_height = image.size
    instances = _extract_instances(
        raw_record,
        _source_label_map(spec),
        image_width=image_width,
        image_height=image_height,
    )
    rel_id = _rel_id(json_rel)
    rng = random.Random(f"{config.seed}:{config.task_name}:{config.split}:{json_rel}")

    full_rows: list[dict[str, Any]] = []
    padded_rows: list[dict[str, Any]] = []
    resize_rows: list[dict[str, Any]] = []
    degraded_rows: list[dict[str, Any]] = []
    blur_rows: list[dict[str, Any]] = []
    full_aug = {"name": "none"}
    full_output_name = f"{rel_id}__full.png"
    full_output_path = config.image_output_dir / full_output_name
    _save_generated_image(image, full_output_path, full_aug)
    full_rows.append(
        _build_row(
            task_name=spec.name,
            split=config.split,
            sample_id=f"{rel_id}__full",
            image_path=f"../images/{config.split}/{full_output_name}",
            image_width=image_width,
            image_height=image_height,
            instances=_full_instances(instances),
            raw_record=raw_record,
            json_rel=json_rel,
            target_labels=spec.labels,
            view_type="full_image",
            crop_box=(0, 0, image_width, image_height),
            source_instance_indices=_sorted_source_instance_indices(instances),
            pixel_augmentation=full_aug,
        )
    )

    if config.split == "train" and config.augmentation_profile == "layout_multiscale_v1":
        if source_plan is None:
            raise ValueError(f"Missing multi-resolution source plan for {json_rel}")
        resize_rows, padded_rows, degraded_rows = _build_multiscale_views(
            image=image,
            instances=instances,
            raw_record=raw_record,
            json_rel=json_rel,
            rel_id=rel_id,
            spec=spec,
            config=config,
            plan=source_plan,
            rng=rng,
        )
    elif config.split == "train":
        padded_image, spatial_aug, content_box = _make_padded_full_image(
            image,
            rng,
            min_ratio=config.padding_min_ratio,
            max_ratio=config.padding_max_ratio,
        )
        padded_width, padded_height = padded_image.size
        padded_output_name = f"{rel_id}__padded_full.png"
        padded_output_path = config.image_output_dir / padded_output_name
        _save_generated_image(padded_image, padded_output_path, {"name": "none"})
        padded_image.close()
        padded_rows.append(
            _build_row(
                task_name=spec.name,
                split=config.split,
                sample_id=f"{rel_id}__padded_full",
                image_path=f"../images/{config.split}/{padded_output_name}",
                image_width=padded_width,
                image_height=padded_height,
                instances=_offset_full_instances(
                    instances,
                    offset_x=content_box[0],
                    offset_y=content_box[1],
                ),
                raw_record=raw_record,
                json_rel=json_rel,
                target_labels=spec.labels,
                view_type="random_padded_full",
                crop_box=content_box,
                source_instance_indices=_sorted_source_instance_indices(instances),
                pixel_augmentation={"name": "none"},
                spatial_augmentation=spatial_aug,
            )
        )

        full_blur_aug = _choose_blur_augmentation(
            rng,
            image_width=image_width,
            image_height=image_height,
        )
        full_blur_output_name = f"{rel_id}__blur_full.png"
        full_blur_output_path = config.image_output_dir / full_blur_output_name
        _save_generated_image(image, full_blur_output_path, full_blur_aug)
        blur_rows.append(
            _build_row(
                task_name=spec.name,
                split=config.split,
                sample_id=f"{rel_id}__blur_full",
                image_path=f"../images/{config.split}/{full_blur_output_name}",
                image_width=image_width,
                image_height=image_height,
                instances=_full_instances(instances),
                raw_record=raw_record,
                json_rel=json_rel,
                target_labels=spec.labels,
                view_type="blur_full",
                crop_box=(0, 0, image_width, image_height),
                source_instance_indices=_sorted_source_instance_indices(instances),
                pixel_augmentation=full_blur_aug,
            )
        )

    positive_rows: list[dict[str, Any]] = []
    negative_rows: list[dict[str, Any]] = []
    if config.split == "train":
        positive_crops = _select_positive_crops(
            instances,
            image_width=image_width,
            image_height=image_height,
            max_crops=spec.max_positive_crops,
            min_instances=spec.min_positive_instances,
            rng=rng,
            candidate_count=config.candidate_count,
        )
        for crop_index, candidate in enumerate(positive_crops):
            left, top, right, bottom = candidate.crop_box
            crop_image = image.crop(candidate.crop_box)
            crop_width, crop_height = crop_image.size
            crop_aug = {"name": "none"}
            output_name = f"{rel_id}__density_{crop_index:02d}.png"
            output_path = config.image_output_dir / output_name
            _save_generated_image(crop_image, output_path, crop_aug)
            positive_rows.append(
                _build_row(
                    task_name=spec.name,
                    split=config.split,
                    sample_id=f"{rel_id}__density_{crop_index:02d}",
                    image_path=f"../images/{config.split}/{output_name}",
                    image_width=crop_width,
                    image_height=crop_height,
                    instances=_translate_instances(
                        instances,
                        set(candidate.instance_indices),
                        candidate.crop_box,
                    ),
                    raw_record=raw_record,
                    json_rel=json_rel,
                    target_labels=spec.labels,
                    view_type="density_crop",
                    crop_box=candidate.crop_box,
                    source_instance_indices=_sorted_source_instance_indices(
                        instances,
                        set(candidate.instance_indices),
                    ),
                    pixel_augmentation=crop_aug,
                )
            )
            if config.augmentation_profile == "legacy":
                blur_aug = _choose_blur_augmentation(
                    rng,
                    image_width=crop_width,
                    image_height=crop_height,
                )
                blur_output_name = f"{rel_id}__blur_crop_{crop_index:02d}.png"
                blur_output_path = config.image_output_dir / blur_output_name
                _save_generated_image(crop_image, blur_output_path, blur_aug)
                blur_rows.append(
                    _build_row(
                        task_name=spec.name,
                        split=config.split,
                        sample_id=f"{rel_id}__blur_crop_{crop_index:02d}",
                        image_path=f"../images/{config.split}/{blur_output_name}",
                        image_width=crop_width,
                        image_height=crop_height,
                        instances=_translate_instances(
                            instances,
                            set(candidate.instance_indices),
                            candidate.crop_box,
                        ),
                        raw_record=raw_record,
                        json_rel=json_rel,
                        target_labels=spec.labels,
                        view_type="blur_crop",
                        crop_box=candidate.crop_box,
                        source_instance_indices=_sorted_source_instance_indices(
                            instances,
                            set(candidate.instance_indices),
                        ),
                        pixel_augmentation=blur_aug,
                    )
                )
            crop_image.close()

        negative_crop = _select_negative_crop(
            instances,
            image_width=image_width,
            image_height=image_height,
            rng=rng,
            candidate_count=config.negative_candidate_count,
        )
        if negative_crop is not None:
            crop_image = image.crop(negative_crop)
            crop_width, crop_height = crop_image.size
            output_name = f"{rel_id}__negative_00.png"
            output_path = config.image_output_dir / output_name
            negative_aug = {"name": "none"}
            _save_generated_image(crop_image, output_path, negative_aug)
            crop_image.close()
            negative_rows.append(
                _build_row(
                    task_name=spec.name,
                    split=config.split,
                    sample_id=f"{rel_id}__negative_00",
                    image_path=f"../images/{config.split}/{output_name}",
                    image_width=crop_width,
                    image_height=crop_height,
                    instances=[],
                    raw_record=raw_record,
                    json_rel=json_rel,
                    target_labels=spec.labels,
                    view_type="hard_negative_crop",
                    crop_box=negative_crop,
                    source_instance_indices=[],
                    pixel_augmentation=negative_aug,
                )
            )

    image.close()
    return SourceResult(
        full_rows=full_rows,
        padded_rows=padded_rows,
        resize_rows=resize_rows,
        degraded_rows=degraded_rows,
        positive_rows=positive_rows,
        negative_rows=negative_rows,
        blur_rows=blur_rows,
        covered=True,
        source_json=json_rel,
        target_count=len(instances),
    )


def _select_negative_rows(
    rows: list[dict[str, Any]],
    *,
    base_count: int,
    ratio: float,
    seed: int,
    task_name: str,
) -> list[dict[str, Any]]:
    if not rows or ratio <= 0:
        return []
    target_count = min(len(rows), int(round(base_count * ratio)))
    rng = random.Random(f"{seed}:{task_name}:negative_selection")
    selected = list(rows)
    rng.shuffle(selected)
    return sorted(selected[:target_count], key=lambda row: str(row["sample_id"]))


def _select_positive_rows(
    rows: list[dict[str, Any]],
    *,
    base_count: int,
    ratio: float,
) -> list[dict[str, Any]]:
    if not rows or ratio <= 0:
        return []
    target_count = min(len(rows), int(round(base_count * ratio)))
    ranked = sorted(
        rows,
        key=lambda row: (
            -len((row.get("extra") or {}).get("source_instance_indices") or []),
            str((row.get("extra") or {}).get("source_json") or ""),
            str(row.get("sample_id") or ""),
        ),
    )
    return sorted(
        ranked[:target_count],
        key=lambda row: (
            str((row.get("extra") or {}).get("source_json") or ""),
            str(row.get("sample_id") or ""),
        ),
    )


def _select_ratio_rows(
    rows: list[dict[str, Any]],
    *,
    base_count: int,
    ratio: float,
    seed: int,
    namespace: str,
) -> list[dict[str, Any]]:
    if not rows or ratio <= 0:
        return []
    target_count = min(len(rows), int(round(base_count * ratio)))
    rng = random.Random(f"{seed}:{namespace}")
    selected = list(rows)
    rng.shuffle(selected)
    return sorted(
        selected[:target_count],
        key=lambda row: (
            str((row.get("extra") or {}).get("source_json") or ""),
            str(row.get("sample_id") or ""),
        ),
    )


def _cleanup_unreferenced_images(rows: list[dict[str, Any]], image_dir: Path) -> int:
    if not image_dir.exists():
        return 0
    referenced = {
        Path(str(row["image_path"])).name
        for row in rows
        if str(row.get("image_path", "")).startswith("../images/")
    }
    removed = 0
    for path in image_dir.iterdir():
        if path.is_file() and path.name not in referenced:
            path.unlink()
            removed += 1
    return removed


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    view_counts = Counter(str(row["extra"]["view_type"]) for row in rows)
    pixel_counts = Counter(str(row["extra"]["pixel_augmentation"]["name"]) for row in rows)
    pixel_bands = Counter(_pixel_band(int(row["image_width"]) * int(row["image_height"])) for row in rows)
    clean_resize_pixel_bands = Counter(
        str(row["extra"]["spatial_augmentation"]["pixel_band"])
        for row in rows
        if row["extra"]["view_type"] == "continuous_resize_full"
    )
    resize_kernels = Counter()
    degradation_severities = Counter()
    empty_rows = sum(1 for row in rows if not row["instances"])
    instance_counts = Counter()
    for row in rows:
        spatial = row["extra"].get("spatial_augmentation") or {}
        if spatial.get("kernel"):
            resize_kernels[str(spatial["kernel"])] += 1
        pixel = row["extra"].get("pixel_augmentation") or {}
        if pixel.get("severity"):
            degradation_severities[str(pixel["severity"])] += 1
        for instance in row["instances"]:
            instance_counts[str(instance["label"])] += 1
    return {
        "rows": len(rows),
        "view_counts": dict(sorted(view_counts.items())),
        "pixel_augmentation_counts": dict(sorted(pixel_counts.items())),
        "pixel_band_counts": dict(sorted(pixel_bands.items())),
        "clean_resize_pixel_band_counts": dict(sorted(clean_resize_pixel_bands.items())),
        "resize_kernel_counts": dict(sorted(resize_kernels.items())),
        "degradation_severity_counts": dict(sorted(degradation_severities.items())),
        "empty_rows": empty_rows,
        "instance_counts": dict(sorted(instance_counts.items())),
    }


def _write_readme(
    task_root: Path,
    *,
    spec: GroundingTaskSpec,
    train_summary: dict[str, Any],
    val_summary: dict[str, Any],
    covered_train: int,
    covered_val: int,
    args: argparse.Namespace,
) -> None:
    train_views = train_summary["view_counts"]
    val_views = val_summary["view_counts"]
    train_count_row = (
        f"| train | {covered_train} | {train_summary['rows']} | "
        f"{train_views.get('full_image', 0)} | "
        f"{train_views.get('continuous_resize_full', 0)} | "
        f"{train_views.get('random_padded_full', 0)} | "
        f"{train_views.get('degraded_resize_full', 0)} | "
        f"{train_views.get('density_crop', 0)} | "
        f"{train_views.get('hard_negative_crop', 0)} | "
        f"{train_summary['empty_rows']} |"
    )
    val_count_row = (
        f"| val | {covered_val} | {val_summary['rows']} | "
        f"{val_views.get('full_image', 0)} | "
        f"{val_views.get('continuous_resize_full', 0)} | "
        f"{val_views.get('random_padded_full', 0)} | "
        f"{val_views.get('degraded_resize_full', 0)} | "
        f"{val_views.get('density_crop', 0)} | "
        f"{val_views.get('hard_negative_crop', 0)} | "
        f"{val_summary['empty_rows']} |"
    )
    content = f"""# {spec.name} structured

Generated from raw annotations for grounding detection.

- Target labels: `{", ".join(spec.labels)}`
- Source label map: `{json.dumps(dict(_source_label_map(spec)), ensure_ascii=False, sort_keys=True)}`
- Required raw coverage: `{", ".join(spec.required_layers) if spec.required_layers else "none; mapped instances define coverage"}`
- Train split source: `{args.train_split}`
- Val split source: `{args.val_split}`
- Workers: `{args.workers}`
- Seed: `{args.seed}`
- Augmentation profile: `{args.augmentation_profile}`
- Pixel budget: `{args.min_pixels}` to `{args.max_pixels}`; processor factor `{args.processor_factor}`
- Continuous clean resize ratio: `{args.clean_resize_views}`
- Degraded resize ratio: `{args.degraded_resize_ratio}`
- Density crop candidates per source: `{args.candidate_count}`
- Density crop global ratio cap: `{args.density_crop_ratio}`
- Max positive crops per source: `{spec.max_positive_crops}`
- Min positive crop instances: `{spec.min_positive_instances}`
- Hard negative ratio target: `{args.negative_ratio}` of covered train sources.
- Positive density crop target: `{args.density_crop_ratio}`
- Random padded full ratio target: `{args.padded_full_ratio}`
- Random padded full total per-axis expansion ratio: `{args.padding_min_ratio}` to `{args.padding_max_ratio}`
- Hard negative selection: bbox-disjoint crop from clean, fully annotated raw sources.
- Images: all rows reference task-local images under `images/<split>/`; raw images are copied,
  not referenced directly.
- Multi-resolution train views: every covered train source contributes one native clean
  `full_image` row; continuous clean resize and bounded degraded-resize rows provide scale and
  pixel robustness; random padded rows are asymmetric; density crops and hard negatives remain
  limited. Validation stays native clean full-image only.

## Counts

| split | covered json | rows | full | clean resize | padded | degraded | density crop | hard negative | empty |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{train_count_row}
{val_count_row}

## Train Pixel Augmentation

```json
{json.dumps(train_summary["pixel_augmentation_counts"], ensure_ascii=False, indent=2)}
```

## Train All-Row Pixel Bands

```json
{json.dumps(train_summary["pixel_band_counts"], ensure_ascii=False, indent=2)}
```

## Train Clean Resize Pixel Bands

```json
{json.dumps(train_summary["clean_resize_pixel_band_counts"], ensure_ascii=False, indent=2)}
```

## Train Resize Kernels

```json
{json.dumps(train_summary["resize_kernel_counts"], ensure_ascii=False, indent=2)}
```

## Train Degradation Severity

```json
{json.dumps(train_summary["degradation_severity_counts"], ensure_ascii=False, indent=2)}
```

## Train Instance Counts

```json
{json.dumps(train_summary["instance_counts"], ensure_ascii=False, indent=2)}
```
"""
    _atomic_write_text(task_root / "README.md", content)


def _build_task_split(
    *,
    spec: GroundingTaskSpec,
    split_path: Path,
    output_path: Path,
    config: BuildConfig,
    workers: int,
) -> tuple[list[dict[str, Any]], int, list[SourceResult]]:
    entries = _read_split(split_path)
    missing_entries = [entry for entry in entries if not (config.raw_root / entry).exists()]
    if missing_entries:
        examples = ", ".join(missing_entries[:5])
        raise FileNotFoundError(
            f"Split {split_path} contains {len(missing_entries)} entries without GT JSON under "
            f"{config.raw_root}. Examples: {examples}. Image-level manifests such as "
            f"vlm.test.json may include image-only items; filter to entries with GT JSON before "
            f"building structured data or metric benchmarks."
        )
    source_plans: dict[str, SourcePlan] = {}
    if config.split == "train" and config.augmentation_profile == "layout_multiscale_v1":
        meta_items = [(entry, spec, config.raw_root) for entry in entries]
        if workers <= 1:
            metas = [_read_source_meta(item) for item in meta_items]
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                metas = list(executor.map(_read_source_meta, meta_items, chunksize=16))
        source_plans = _build_multiscale_plans(metas, config=config)
    work_items = [
        (entry, spec, config, source_plans.get(entry))
        for entry in entries
    ]
    if workers <= 1:
        results = [_process_source(item) for item in work_items]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_process_source, work_items, chunksize=8))

    covered_results = [result for result in results if result.covered]
    full_rows: list[dict[str, Any]] = []
    padded_candidates: list[dict[str, Any]] = []
    resize_rows: list[dict[str, Any]] = []
    degraded_rows: list[dict[str, Any]] = []
    positive_rows: list[dict[str, Any]] = []
    negative_candidates: list[dict[str, Any]] = []
    blur_candidates: list[dict[str, Any]] = []
    for result in covered_results:
        full_rows.extend(result.full_rows)
        padded_candidates.extend(result.padded_rows)
        resize_rows.extend(result.resize_rows)
        degraded_rows.extend(result.degraded_rows)
        positive_rows.extend(result.positive_rows)
        negative_candidates.extend(result.negative_rows)
        blur_candidates.extend(result.blur_rows)

    negative_rows = _select_negative_rows(
        negative_candidates,
        base_count=len(covered_results),
        ratio=config.negative_ratio,
        seed=config.seed,
        task_name=spec.name,
    )
    positive_rows = _select_positive_rows(
        positive_rows,
        base_count=len(covered_results),
        ratio=(
            config.density_crop_ratio
            if config.augmentation_profile == "layout_multiscale_v1"
            else max(config.density_crop_ratio - config.negative_ratio, 0.0)
        ),
    )
    if config.augmentation_profile == "layout_multiscale_v1":
        padded_rows = padded_candidates
        blur_rows: list[dict[str, Any]] = []
    else:
        padded_rows = _select_ratio_rows(
            padded_candidates,
            base_count=len(covered_results),
            ratio=config.padded_full_ratio,
            seed=config.seed,
            namespace=f"{spec.name}:padded_full",
        )
        blur_rows = _select_ratio_rows(
            blur_candidates,
            base_count=len(covered_results),
            ratio=config.blur_ratio,
            seed=config.seed,
            namespace=f"{spec.name}:blur",
        )
    rows = sorted(
        full_rows
        + resize_rows
        + padded_rows
        + degraded_rows
        + positive_rows
        + negative_rows
        + blur_rows,
        key=lambda row: (str(row["extra"]["source_json"]), str(row["sample_id"])),
    )
    _write_jsonl_atomic(output_path, rows)
    _cleanup_unreferenced_images(rows, config.image_output_dir)
    return rows, len(covered_results), covered_results


def _clean_task_output(task_root: Path) -> None:
    if task_root.exists():
        shutil.rmtree(task_root)
    (task_root / "structured").mkdir(parents=True, exist_ok=True)
    (task_root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (task_root / "images" / "val").mkdir(parents=True, exist_ok=True)


def _task_by_name(names: list[str] | None) -> list[GroundingTaskSpec]:
    task_map = {task.name: task for task in TASKS}
    if not names:
        return [task_map[name] for name in DEFAULT_TASK_NAMES]
    missing = sorted(set(names) - set(task_map))
    if missing:
        raise ValueError(f"Unknown grounding task(s): {', '.join(missing)}")
    return [task_map[name] for name in names]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build grounding structured datasets.")
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--output-root", default="data")
    parser.add_argument("--train-split", required=True)
    parser.add_argument("--val-split", required=True)
    parser.add_argument("--task", action="append", choices=[task.name for task in TASKS])
    parser.add_argument("--workers", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--augmentation-profile",
        choices=("layout_multiscale_v1", "legacy"),
        default="layout_multiscale_v1",
    )
    parser.add_argument("--candidate-count", type=int, default=32)
    parser.add_argument("--negative-candidate-count", type=int, default=48)
    parser.add_argument("--negative-ratio", type=float, default=0.03)
    parser.add_argument("--density-crop-ratio", type=float, default=0.25)
    parser.add_argument("--blur-ratio", type=float, default=0.0)
    parser.add_argument("--padded-full-ratio", type=float, default=0.1)
    parser.add_argument("--padding-min-ratio", type=float, default=0.05)
    parser.add_argument("--padding-max-ratio", type=float, default=0.25)
    parser.add_argument("--min-pixels", type=int, default=200_704)
    parser.add_argument("--max-pixels", type=int, default=4_000_000)
    parser.add_argument("--processor-factor", type=int, default=32)
    parser.add_argument("--clean-resize-views", type=float, default=2.9)
    parser.add_argument("--degraded-resize-ratio", type=float, default=1.2)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    output_root = Path(args.output_root)
    if not raw_root.exists():
        raise FileNotFoundError(raw_root)
    if not 0 <= float(args.blur_ratio):
        raise ValueError("--blur-ratio must be >= 0.")
    if not 0 <= float(args.padded_full_ratio):
        raise ValueError("--padded-full-ratio must be >= 0.")
    if not 0 <= float(args.negative_ratio):
        raise ValueError("--negative-ratio must be >= 0.")
    if float(args.density_crop_ratio) < 0:
        raise ValueError("--density-crop-ratio must be >= 0.")
    if float(args.negative_ratio) > float(args.density_crop_ratio):
        raise ValueError("--negative-ratio must be <= --density-crop-ratio.")
    if not 0 <= float(args.padding_min_ratio) <= float(args.padding_max_ratio):
        raise ValueError("--padding-min-ratio must satisfy 0 <= min <= max.")
    if int(args.min_pixels) <= 0 or int(args.max_pixels) < int(args.min_pixels):
        raise ValueError("pixel budget must satisfy 0 < min <= max")
    if int(args.processor_factor) <= 0:
        raise ValueError("--processor-factor must be > 0")
    if float(args.clean_resize_views) < 0:
        raise ValueError("--clean-resize-views must be >= 0")
    if not 0 <= float(args.degraded_resize_ratio) <= 2:
        raise ValueError("--degraded-resize-ratio must be between 0 and 2")
    if args.augmentation_profile == "layout_multiscale_v1" and not math.isclose(
        float(args.clean_resize_views) + float(args.padded_full_ratio),
        round(float(args.clean_resize_views) + float(args.padded_full_ratio)),
        abs_tol=1e-6,
    ):
        raise ValueError(
            "multi-resolution clean resize plus padded ratios must form an integer budget"
        )

    summaries: dict[str, Any] = {}
    for spec in _task_by_name(args.task):
        task_root = output_root / spec.name
        if args.clean:
            _clean_task_output(task_root)
        else:
            (task_root / "structured").mkdir(parents=True, exist_ok=True)
            (task_root / "images" / "train").mkdir(parents=True, exist_ok=True)
            (task_root / "images" / "val").mkdir(parents=True, exist_ok=True)

        train_config = BuildConfig(
            raw_root=raw_root,
            task_name=spec.name,
            split="train",
            output_root=task_root,
            image_output_dir=task_root / "images" / "train",
            seed=int(args.seed),
            candidate_count=int(args.candidate_count),
            negative_candidate_count=int(args.negative_candidate_count),
            negative_ratio=float(args.negative_ratio),
            density_crop_ratio=float(args.density_crop_ratio),
            blur_ratio=float(args.blur_ratio),
            padded_full_ratio=float(args.padded_full_ratio),
            padding_min_ratio=float(args.padding_min_ratio),
            padding_max_ratio=float(args.padding_max_ratio),
            augmentation_profile=str(args.augmentation_profile),
            min_pixels=int(args.min_pixels),
            max_pixels=int(args.max_pixels),
            processor_factor=int(args.processor_factor),
            clean_resize_views=float(args.clean_resize_views),
            degraded_resize_ratio=float(args.degraded_resize_ratio),
        )
        val_config = BuildConfig(
            raw_root=raw_root,
            task_name=spec.name,
            split="val",
            output_root=task_root,
            image_output_dir=task_root / "images" / "val",
            seed=int(args.seed),
            candidate_count=int(args.candidate_count),
            negative_candidate_count=int(args.negative_candidate_count),
            negative_ratio=0.0,
            density_crop_ratio=0.0,
            blur_ratio=0.0,
            padded_full_ratio=0.0,
            padding_min_ratio=float(args.padding_min_ratio),
            padding_max_ratio=float(args.padding_max_ratio),
            augmentation_profile=str(args.augmentation_profile),
            min_pixels=int(args.min_pixels),
            max_pixels=int(args.max_pixels),
            processor_factor=int(args.processor_factor),
            clean_resize_views=0.0,
            degraded_resize_ratio=0.0,
        )
        train_rows, covered_train, _ = _build_task_split(
            spec=spec,
            split_path=Path(args.train_split),
            output_path=task_root / "structured" / "train.jsonl",
            config=train_config,
            workers=int(args.workers),
        )
        val_rows, covered_val, _ = _build_task_split(
            spec=spec,
            split_path=Path(args.val_split),
            output_path=task_root / "structured" / "val.jsonl",
            config=val_config,
            workers=int(args.workers),
        )
        train_summary = _summarize_rows(train_rows)
        val_summary = _summarize_rows(val_rows)
        _write_readme(
            task_root,
            spec=spec,
            train_summary=train_summary,
            val_summary=val_summary,
            covered_train=covered_train,
            covered_val=covered_val,
            args=args,
        )
        summaries[spec.name] = {
            "covered_train": covered_train,
            "covered_val": covered_val,
            "train": train_summary,
            "val": val_summary,
        }

    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
