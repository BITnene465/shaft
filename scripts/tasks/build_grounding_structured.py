#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
class SourceResult:
    full_rows: list[dict[str, Any]]
    padded_rows: list[dict[str, Any]]
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


def _process_source(args: tuple[str, GroundingTaskSpec, BuildConfig]) -> SourceResult:
    json_rel, spec, config = args
    raw_record = _load_json(config.raw_root / json_rel)
    if not _has_required_coverage(raw_record, spec.required_layers):
        return SourceResult([], [], [], [], [], False, json_rel, 0)

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
            source_instance_indices=[instance.index for instance in instances],
            pixel_augmentation=full_aug,
        )
    )

    if config.split == "train":
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
                source_instance_indices=[instance.index for instance in instances],
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
                source_instance_indices=[instance.index for instance in instances],
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
                    source_instance_indices=list(candidate.instance_indices),
                    pixel_augmentation=crop_aug,
                )
            )
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
                    source_instance_indices=list(candidate.instance_indices),
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
    empty_rows = sum(1 for row in rows if not row["instances"])
    instance_counts = Counter()
    for row in rows:
        for instance in row["instances"]:
            instance_counts[str(instance["label"])] += 1
    return {
        "rows": len(rows),
        "view_counts": dict(sorted(view_counts.items())),
        "pixel_augmentation_counts": dict(sorted(pixel_counts.items())),
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
        f"{train_views.get('random_padded_full', 0)} | "
        f"{train_views.get('density_crop', 0)} | "
        f"{train_views.get('blur_full', 0) + train_views.get('blur_crop', 0)} | "
        f"{train_views.get('hard_negative_crop', 0)} | "
        f"{train_summary['empty_rows']} |"
    )
    val_count_row = (
        f"| val | {covered_val} | {val_summary['rows']} | "
        f"{val_views.get('full_image', 0)} | "
        f"{val_views.get('random_padded_full', 0)} | "
        f"{val_views.get('density_crop', 0)} | "
        f"{val_views.get('blur_full', 0) + val_views.get('blur_crop', 0)} | "
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
- Density crop candidates per source: `{args.candidate_count}`
- Density crop global ratio cap: `{args.density_crop_ratio}`
- Max positive crops per source: `{spec.max_positive_crops}`
- Min positive crop instances: `{spec.min_positive_instances}`
- Hard negative ratio target: `{args.negative_ratio}` of covered train sources; selected inside
  the density-crop budget.
- Effective positive density crop target: `{max(float(args.density_crop_ratio) - float(args.negative_ratio), 0.0)}`
- Blur full plus blur crop ratio target: `{args.blur_ratio}`
- Random padded full ratio target: `{args.padded_full_ratio}`
- Random padded full per-side padding ratio: `{args.padding_min_ratio}` to `{args.padding_max_ratio}`
- Hard negative selection: bbox-disjoint crop from clean, fully annotated raw sources.
- Images: all rows reference task-local images under `images/<split>/`; raw images are copied,
  not referenced directly.
- Grounding default train views: every covered train source contributes one clean `full_image`
  row; density crops are density-biased local views; hard negatives are a small minority; blur
  rows are sampled from full-image and density-crop candidates; random padded full rows are sampled
  only from clean full-image candidates. Validation stays clean full-image only.

## Counts

| split | covered json | rows | full | padded full | density crop | blur rows | hard negative | empty |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{train_count_row}
{val_count_row}

## Train Pixel Augmentation

```json
{json.dumps(train_summary["pixel_augmentation_counts"], ensure_ascii=False, indent=2)}
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
    work_items = [(entry, spec, config) for entry in entries]
    if workers <= 1:
        results = [_process_source(item) for item in work_items]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_process_source, work_items, chunksize=8))

    covered_results = [result for result in results if result.covered]
    full_rows: list[dict[str, Any]] = []
    padded_candidates: list[dict[str, Any]] = []
    positive_rows: list[dict[str, Any]] = []
    negative_candidates: list[dict[str, Any]] = []
    blur_candidates: list[dict[str, Any]] = []
    for result in covered_results:
        full_rows.extend(result.full_rows)
        padded_candidates.extend(result.padded_rows)
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
        ratio=max(config.density_crop_ratio - config.negative_ratio, 0.0),
    )
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
        full_rows + padded_rows + positive_rows + negative_rows + blur_rows,
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
    parser.add_argument("--candidate-count", type=int, default=32)
    parser.add_argument("--negative-candidate-count", type=int, default=48)
    parser.add_argument("--negative-ratio", type=float, default=0.05)
    parser.add_argument("--density-crop-ratio", type=float, default=0.3)
    parser.add_argument("--blur-ratio", type=float, default=1.0)
    parser.add_argument("--padded-full-ratio", type=float, default=0.2)
    parser.add_argument("--padding-min-ratio", type=float, default=0.1)
    parser.add_argument("--padding-max-ratio", type=float, default=0.2)
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
