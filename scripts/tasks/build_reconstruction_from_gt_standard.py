#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
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

from PIL import Image

from shaft.codec.coordinates import quantize_qwen_coordinate
from shaft.prompting import load_prompt_pool


TASK_CONFIGS = {
    "shape_reconstruction": {
        "label": "shape",
        "prompt": "configs/prompts/pools/shape_reconstruction.v5.0.yaml",
    },
    "line_reconstruction": {
        "label": "line",
        "prompt": "configs/prompts/pools/line_reconstruction.v5.0.yaml",
    },
}

SHAPE_RARE_TYPES = frozenset(
    {
        "arrow_pentagon",
        "diamond",
        "other_polygon",
        "oval",
        "parallelogram",
        "regular_hexagon",
        "regular_pentagon",
        "step",
        "trapezoid",
        "triangle",
    }
)
SHAPE_HEAD_WEIGHTS_V1 = {"rectangle": 0.52, "other": 0.26, "callout": 0.22}
SHAPE_HEAD_WEIGHTS_V2 = {
    "rectangle": 40000,
    "other": 20096,
    "callout": 16000,
    "icon_as_other": 10000,
    "image_as_other": 10000,
}
LINE_MACRO_WEIGHTS = {
    "curved_shape": 0.10,
    "curved_path": 0.235,
    "straight_shape": 0.235,
    "straight_path_multi": 0.167,
    "straight_path_single": 0.263,
}
MULTI_SCALE_BUCKETS = (
    ("tight", 0.70, 0.08, 0.15),
    ("medium", 0.25, 0.15, 0.25),
    ("context", 0.05, 0.25, 0.40),
)


@dataclass(frozen=True)
class PromptVariantInfo:
    prompt_id: str
    system_prompt: str
    user_prompt: str
    output_schema: str | None


@dataclass(frozen=True)
class PromptPoolInfo:
    variants: tuple[PromptVariantInfo, ...]


@dataclass(frozen=True)
class BuildConfig:
    dataset_root: Path
    output_root: Path
    split: str
    selected_tasks: frozenset[str]
    padding_min: float
    padding_max: float
    seed: int
    min_crop_size: int
    max_aspect_ratio: float
    skip_oob_bbox: bool
    multi_scale: bool
    shape_low_resolution_ratio: float
    line_low_resolution_ratio: float
    include_visual_other_negatives: bool
    prompt_info: dict[str, PromptPoolInfo]


@dataclass(frozen=True)
class WorkerResult:
    structured_rows: dict[str, list[str]]
    sft_rows: dict[str, list[str]]
    counts: dict[str, int]


@dataclass(frozen=True)
class Candidate:
    stem: str
    instance_index: int
    task: str
    macro: str
    stratum: str
    source_label: str = ""


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent) as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
        tmp_path = Path(f.name)
    os.replace(tmp_path, path)


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _stable_score(*parts: Any) -> int:
    payload = ":".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _allocate_weighted_caps(
    counts: dict[str, int],
    *,
    target: int,
    weights: dict[str, float] | None = None,
) -> dict[str, int]:
    if target < 0:
        raise ValueError("target must be non-negative")
    available = sum(max(0, int(count)) for count in counts.values())
    if target >= available:
        return {key: max(0, int(value)) for key, value in counts.items()}
    if target == 0:
        return {key: 0 for key in counts}

    allocations = {key: 0 for key in counts}
    active = {key for key, count in counts.items() if count > 0}
    remaining = target
    while remaining > 0 and active:
        raw_weights = {
            key: max(0.0, float(weights[key]))
            if weights and key in weights
            else math.sqrt(counts[key])
            for key in active
        }
        weight_sum = sum(raw_weights.values())
        if weight_sum <= 0:
            raw_weights = {key: 1.0 for key in active}
            weight_sum = float(len(active))
        proposals = {key: remaining * raw_weights[key] / weight_sum for key in active}
        progressed = 0
        for key in sorted(active):
            capacity = counts[key] - allocations[key]
            take = min(capacity, int(math.floor(proposals[key])))
            if take > 0:
                allocations[key] += take
                remaining -= take
                progressed += take
        active = {key for key in active if allocations[key] < counts[key]}
        if remaining <= 0 or not active:
            break
        if progressed == 0 or remaining < len(active):
            ranked = sorted(
                active,
                key=lambda key: (proposals.get(key, 0.0) % 1.0, raw_weights.get(key, 0.0), key),
                reverse=True,
            )
            for key in ranked:
                if remaining <= 0:
                    break
                if allocations[key] < counts[key]:
                    allocations[key] += 1
                    remaining -= 1
    if sum(allocations.values()) != target:
        raise RuntimeError("failed to allocate the requested sampling target")
    return allocations


def _select_stratified(
    candidates: list[Candidate],
    *,
    target: int,
    seed: int,
) -> list[Candidate]:
    if target >= len(candidates):
        return list(candidates)
    by_stratum: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_stratum[candidate.stratum].append(candidate)
    quotas = _allocate_weighted_caps(
        {key: len(values) for key, values in by_stratum.items()},
        target=target,
    )
    selected: list[Candidate] = []
    for stratum in sorted(by_stratum):
        quota = quotas[stratum]
        values = by_stratum[stratum]
        if quota >= len(values):
            selected.extend(values)
            continue
        values.sort(
            key=lambda candidate: _stable_score(
                seed,
                "sample",
                candidate.task,
                candidate.stem,
                candidate.instance_index,
            )
        )
        selected.extend(values[:quota])
    return selected


def _clean_bbox(
    bbox: Any,
    *,
    image_width: int,
    image_height: int,
    skip_oob_bbox: bool,
) -> tuple[float, float, float, float] | None:
    if not isinstance(bbox, list | tuple) or len(bbox) != 4:
        return None
    if not all(_is_number(value) for value in bbox):
        return None
    x1, y1, x2, y2 = [float(value) for value in bbox]
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if x2 <= x1 or y2 <= y1:
        return None
    if skip_oob_bbox and (x1 < 0 or y1 < 0 or x2 > image_width or y2 > image_height):
        return None
    x1 = min(max(x1, 0.0), float(image_width))
    y1 = min(max(y1, 0.0), float(image_height))
    x2 = min(max(x2, 0.0), float(image_width))
    y2 = min(max(y2, 0.0), float(image_height))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _bounded_interval(center: float, length: int, *, limit: int) -> tuple[int, int]:
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


def _enforce_max_aspect_ratio(
    crop_box: tuple[int, int, int, int],
    *,
    image_width: int,
    image_height: int,
    max_aspect_ratio: float,
) -> tuple[int, int, int, int]:
    if max_aspect_ratio <= 0:
        return crop_box
    left, top, right, bottom = crop_box
    width = right - left
    height = bottom - top
    if width <= 0 or height <= 0:
        return crop_box

    if width / height > max_aspect_ratio:
        target_height = min(image_height, max(height, int(math.ceil(width / max_aspect_ratio))))
        center_y = (top + bottom) / 2.0
        top, bottom = _bounded_interval(center_y, target_height, limit=image_height)
    elif height / width > max_aspect_ratio:
        target_width = min(image_width, max(width, int(math.ceil(height / max_aspect_ratio))))
        center_x = (left + right) / 2.0
        left, right = _bounded_interval(center_x, target_width, limit=image_width)
    return left, top, right, bottom


def _crop_box(
    bbox: tuple[float, float, float, float],
    *,
    image_width: int,
    image_height: int,
    padding_ratio: float,
    max_aspect_ratio: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    pad_x = width * padding_ratio
    pad_y = height * padding_ratio
    left = max(0, int(math.floor(x1 - pad_x)))
    top = max(0, int(math.floor(y1 - pad_y)))
    right = min(image_width, int(math.ceil(x2 + pad_x)))
    bottom = min(image_height, int(math.ceil(y2 + pad_y)))
    return _enforce_max_aspect_ratio(
        (left, top, right, bottom),
        image_width=image_width,
        image_height=image_height,
        max_aspect_ratio=max_aspect_ratio,
    )


def _aspect_ratio(width: int, height: int) -> float:
    if width <= 0 or height <= 0:
        return math.inf
    return max(width / height, height / width)


def _padding_policy(
    config: BuildConfig,
    *,
    task: str,
    stem: str,
    instance_index: int,
) -> tuple[str, float]:
    if not config.multi_scale:
        rng = random.Random(f"{config.seed}:{stem}:{instance_index}")
        return "default", rng.uniform(config.padding_min, config.padding_max)
    rng = random.Random(f"{config.seed}:scale:{task}:{stem}:{instance_index}")
    draw = rng.random()
    cumulative = 0.0
    for name, probability, minimum, maximum in MULTI_SCALE_BUCKETS:
        cumulative += probability
        if draw < cumulative:
            return name, rng.uniform(minimum, maximum)
    name, _, minimum, maximum = MULTI_SCALE_BUCKETS[-1]
    return name, rng.uniform(minimum, maximum)


def _low_resolution_target(
    config: BuildConfig,
    *,
    task: str,
    stem: str,
    instance_index: int,
    width: int,
    height: int,
) -> tuple[int, int] | None:
    ratio = (
        config.shape_low_resolution_ratio
        if task == "shape_reconstruction"
        else config.line_low_resolution_ratio
    )
    if ratio <= 0:
        return None
    rng = random.Random(f"{config.seed}:lowres:{task}:{stem}:{instance_index}")
    if rng.random() >= ratio:
        return None
    minimum_short, maximum_short = (24, 96) if task == "shape_reconstruction" else (16, 64)
    target_short = int(
        round(math.exp(rng.uniform(math.log(minimum_short), math.log(maximum_short))))
    )
    current_short = min(width, height)
    if current_short <= target_short:
        return None
    scale = target_short / current_short
    resized_width = max(4, int(round(width * scale)))
    resized_height = max(4, int(round(height * scale)))
    return resized_width, resized_height


def _prompt_variant(
    config: BuildConfig, *, task: str, stem: str, instance_index: int
) -> PromptVariantInfo:
    variants = config.prompt_info[task].variants
    if not variants:
        raise ValueError(f"Prompt pool for {task} is empty")
    rng = random.Random(f"{config.seed}:prompt:{task}:{stem}:{instance_index}")
    return variants[rng.randrange(len(variants))]


def _quantize_axis(value: float, *, origin: int, size: int, num_bins: int = 1000) -> int:
    local_value = float(value) - float(origin)
    return quantize_qwen_coordinate(local_value, size=size, num_bins=num_bins)


def _quantize_crop_point(
    point: Any,
    *,
    left: int,
    top: int,
    crop_width: int,
    crop_height: int,
    num_bins: int = 1000,
) -> list[int]:
    if not isinstance(point, list | tuple) or len(point) != 2:
        raise ValueError(f"Invalid point: {point!r}")
    if not all(_is_number(value) for value in point):
        raise ValueError(f"Invalid point: {point!r}")
    return [
        _quantize_axis(float(point[0]), origin=left, size=crop_width, num_bins=num_bins),
        _quantize_axis(float(point[1]), origin=top, size=crop_height, num_bins=num_bins),
    ]


def _translate_point(point: Any, *, left: int, top: int) -> list[int | float]:
    if not isinstance(point, list | tuple) or len(point) != 2:
        raise ValueError(f"Invalid point: {point!r}")
    if not all(_is_number(value) for value in point):
        raise ValueError(f"Invalid point: {point!r}")
    x = float(point[0]) - left
    y = float(point[1]) - top
    x_value = int(round(x)) if abs(x - round(x)) < 1e-6 else round(x, 3)
    y_value = int(round(y)) if abs(y - round(y)) < 1e-6 else round(y, 3)
    return [x_value, y_value]


def _quantize_crop_bbox(
    bbox: Any,
    *,
    left: int,
    top: int,
    crop_width: int,
    crop_height: int,
    num_bins: int = 1000,
) -> list[int]:
    if not isinstance(bbox, list | tuple) or len(bbox) != 4:
        raise ValueError(f"Invalid bbox: {bbox!r}")
    if not all(_is_number(value) for value in bbox):
        raise ValueError(f"Invalid bbox: {bbox!r}")
    return [
        _quantize_axis(float(bbox[0]), origin=left, size=crop_width, num_bins=num_bins),
        _quantize_axis(float(bbox[1]), origin=top, size=crop_height, num_bins=num_bins),
        _quantize_axis(float(bbox[2]), origin=left, size=crop_width, num_bins=num_bins),
        _quantize_axis(float(bbox[3]), origin=top, size=crop_height, num_bins=num_bins),
    ]


def _translate_bbox(bbox: Any, *, left: int, top: int) -> list[int | float]:
    if not isinstance(bbox, list | tuple) or len(bbox) != 4:
        raise ValueError(f"Invalid bbox: {bbox!r}")
    if not all(_is_number(value) for value in bbox):
        raise ValueError(f"Invalid bbox: {bbox!r}")
    top_left = _translate_point([bbox[0], bbox[1]], left=left, top=top)
    bottom_right = _translate_point([bbox[2], bbox[3]], left=left, top=top)
    return [
        top_left[0],
        top_left[1],
        bottom_right[0],
        bottom_right[1],
    ]


def _quantize_crop_corner(
    corner: Any,
    *,
    left: int,
    top: int,
    crop_width: int,
    crop_height: int,
) -> dict[str, Any]:
    if not isinstance(corner, dict):
        raise ValueError(f"Invalid corner: {corner!r}")
    translated = dict(corner)
    for key in ("point", "start", "mid", "end"):
        if key in translated:
            translated[key] = _quantize_crop_point(
                translated[key],
                left=left,
                top=top,
                crop_width=crop_width,
                crop_height=crop_height,
            )
    return translated


def _translate_shape_parameters(
    params: dict[str, Any],
    *,
    left: int,
    top: int,
    crop_width: int,
    crop_height: int,
) -> dict[str, Any]:
    translated = json.loads(json.dumps(params, ensure_ascii=False))
    if isinstance(translated.get("corners"), list):
        translated["corners"] = [
            _quantize_crop_corner(
                corner,
                left=left,
                top=top,
                crop_width=crop_width,
                crop_height=crop_height,
            )
            for corner in translated["corners"]
        ]
    if isinstance(translated.get("body_corners"), list):
        translated["body_corners"] = [
            _quantize_crop_corner(
                corner,
                left=left,
                top=top,
                crop_width=crop_width,
                crop_height=crop_height,
            )
            for corner in translated["body_corners"]
        ]
    if isinstance(translated.get("body_bbox"), list):
        translated["body_bbox"] = _quantize_crop_bbox(
            translated["body_bbox"],
            left=left,
            top=top,
            crop_width=crop_width,
            crop_height=crop_height,
        )
    tail = translated.get("tail")
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
    return translated


def _translate_line_parameters(
    params: dict[str, Any],
    *,
    left: int,
    top: int,
    crop_width: int,
    crop_height: int,
) -> dict[str, Any]:
    translated = json.loads(json.dumps(params, ensure_ascii=False))
    points = translated.get("points")
    if not isinstance(points, list):
        raise ValueError("line parameters require points list")
    translated["points"] = [
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
        for segment in points
        if isinstance(segment, list)
    ]
    if not translated["points"]:
        raise ValueError("line parameters contain no valid point segments")
    return translated


def _target_parameters(
    label: str,
    params: dict[str, Any],
    *,
    left: int,
    top: int,
    crop_width: int,
    crop_height: int,
) -> dict[str, Any]:
    if label == "shape":
        return _translate_shape_parameters(
            params,
            left=left,
            top=top,
            crop_width=crop_width,
            crop_height=crop_height,
        )
    if label == "line":
        return _translate_line_parameters(
            params,
            left=left,
            top=top,
            crop_width=crop_width,
            crop_height=crop_height,
        )
    raise ValueError(f"Unsupported reconstruction label: {label}")


def _shape_stratum(params: dict[str, Any]) -> str:
    border = params.get("border") if isinstance(params.get("border"), dict) else {}
    fill = params.get("fill") if isinstance(params.get("fill"), dict) else {}
    effect = params.get("effect") if isinstance(params.get("effect"), dict) else {}
    return "|".join(
        str(value)
        for value in (
            params.get("shape_type", "missing"),
            border.get("type", "missing"),
            fill.get("type", "missing"),
            effect.get("type", "missing"),
            params.get("body_type", "missing"),
        )
    )


def _line_macro(params: dict[str, Any]) -> str:
    line_type = str(params.get("line_type") or "missing")
    line_style = str(params.get("line_style") or "missing")
    points = params.get("points") if isinstance(params.get("points"), list) else []
    is_multi = params.get("is_single") is False or len(points) > 1
    if line_type == "curved" and line_style == "shape":
        return "curved_shape"
    if line_type == "curved":
        return "curved_path"
    if line_style == "shape":
        return "straight_shape"
    if is_multi:
        return "straight_path_multi"
    return "straight_path_single"


def _line_stratum(params: dict[str, Any]) -> str:
    points = params.get("points") if isinstance(params.get("points"), list) else []
    segment_count = min(len(points), 4)
    fill_color = params.get("fill_color")
    fill_kind = "gradient" if isinstance(fill_color, list) else "solid"
    return "|".join(
        str(value)
        for value in (
            params.get("begin_arrow", "missing"),
            params.get("end_arrow", "missing"),
            params.get("dash_style", "missing"),
            segment_count,
            fill_kind,
            params.get("has_border", "missing"),
            params.get("corner_style", "missing"),
        )
    )


def _candidate_for_instance(
    *,
    stem: str,
    instance_index: int,
    instance: Any,
    image_width: int,
    image_height: int,
    config: BuildConfig,
) -> Candidate | None:
    if not isinstance(instance, dict):
        return None
    source_label = str(instance.get("type") or "")
    visual_other = (
        config.include_visual_other_negatives
        and source_label in {"icon", "image"}
        and "shape_reconstruction" in config.selected_tasks
    )
    label = "shape" if visual_other else source_label
    task = f"{label}_reconstruction"
    if task not in config.selected_tasks:
        return None
    source_params = instance.get("parameters")
    if not visual_other and not isinstance(source_params, dict):
        return None
    params = {"shape_type": "other"} if visual_other else source_params
    assert isinstance(params, dict)
    bbox = _clean_bbox(
        instance.get("bbox"),
        image_width=image_width,
        image_height=image_height,
        skip_oob_bbox=config.skip_oob_bbox,
    )
    if bbox is None:
        return None
    _, padding_ratio = _padding_policy(
        config,
        task=task,
        stem=stem,
        instance_index=instance_index,
    )
    left, top, right, bottom = _crop_box(
        bbox,
        image_width=image_width,
        image_height=image_height,
        padding_ratio=padding_ratio,
        max_aspect_ratio=config.max_aspect_ratio,
    )
    crop_width = right - left
    crop_height = bottom - top
    if crop_width < config.min_crop_size or crop_height < config.min_crop_size:
        return None
    try:
        _target_parameters(
            label,
            params,
            left=left,
            top=top,
            crop_width=crop_width,
            crop_height=crop_height,
        )
    except (TypeError, ValueError):
        return None
    if task == "shape_reconstruction":
        macro = (
            f"{source_label}_as_other"
            if visual_other
            else str(params.get("shape_type") or "missing")
        )
        stratum = _shape_stratum(params)
    else:
        macro = _line_macro(params)
        stratum = _line_stratum(params)
    return Candidate(
        stem=stem,
        instance_index=instance_index,
        task=task,
        macro=macro,
        stratum=stratum,
        source_label=source_label,
    )


def _inventory_for_json(args: tuple[str, BuildConfig]) -> list[Candidate]:
    stem, config = args
    source_json = config.dataset_root / "gt_standard" / f"{stem}.json"
    try:
        obj = _load_json(source_json)
    except Exception:
        return []
    source_size = obj.get("size")
    layout = obj.get("layout")
    if not (
        isinstance(source_size, list)
        and len(source_size) == 2
        and all(_is_number(value) for value in source_size)
        and isinstance(layout, list)
    ):
        return []
    image_width, image_height = int(source_size[0]), int(source_size[1])
    candidates: list[Candidate] = []
    for instance_index, instance in enumerate(layout):
        candidate = _candidate_for_instance(
            stem=stem,
            instance_index=instance_index,
            instance=instance,
            image_width=image_width,
            image_height=image_height,
            config=config,
        )
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _select_balanced_candidates(
    candidates: list[Candidate],
    *,
    shape_target: int,
    line_target: int,
    seed: int,
) -> tuple[list[Candidate], Counter[str]]:
    by_task_macro: dict[str, dict[str, list[Candidate]]] = defaultdict(lambda: defaultdict(list))
    for candidate in candidates:
        by_task_macro[candidate.task][candidate.macro].append(candidate)
    selected: list[Candidate] = []
    counts: Counter[str] = Counter()

    shape_groups = by_task_macro.get("shape_reconstruction", {})
    for shape_type, values in shape_groups.items():
        counts[f"available_shape_type_{shape_type}"] = len(values)
    rare_shapes = [
        candidate
        for shape_type in sorted(SHAPE_RARE_TYPES)
        for candidate in shape_groups.get(shape_type, [])
    ]
    if rare_shapes and len(rare_shapes) > shape_target:
        raise ValueError(
            f"shape target {shape_target} is smaller than {len(rare_shapes)} retained rare shapes"
        )
    selected.extend(rare_shapes)
    for candidate in rare_shapes:
        counts[f"selected_shape_type_{candidate.macro}"] += 1
    head_target = max(0, shape_target - len(rare_shapes))
    head_weights = (
        SHAPE_HEAD_WEIGHTS_V2
        if any(key in shape_groups for key in ("icon_as_other", "image_as_other"))
        else SHAPE_HEAD_WEIGHTS_V1
    )
    head_counts = {key: len(shape_groups.get(key, [])) for key in head_weights}
    head_quotas = _allocate_weighted_caps(
        head_counts,
        target=min(head_target, sum(head_counts.values())),
        weights=head_weights,
    )
    for shape_type in sorted(head_quotas):
        chosen = _select_stratified(
            shape_groups.get(shape_type, []),
            target=head_quotas[shape_type],
            seed=seed,
        )
        selected.extend(chosen)
        counts[f"selected_shape_type_{shape_type}"] += len(chosen)

    line_groups = by_task_macro.get("line_reconstruction", {})
    for macro, values in line_groups.items():
        counts[f"available_line_macro_{macro}"] = len(values)
    curved_shape = list(line_groups.get("curved_shape", []))
    if len(curved_shape) > line_target:
        raise ValueError(
            f"line target {line_target} is smaller than {len(curved_shape)} retained curved shape lines"
        )
    selected.extend(curved_shape)
    counts["selected_line_macro_curved_shape"] += len(curved_shape)
    remaining_line_target = max(0, line_target - len(curved_shape))
    line_counts = {
        key: len(line_groups.get(key, [])) for key in LINE_MACRO_WEIGHTS if key != "curved_shape"
    }
    line_weights = {
        key: value for key, value in LINE_MACRO_WEIGHTS.items() if key != "curved_shape"
    }
    line_quotas = _allocate_weighted_caps(
        line_counts,
        target=min(remaining_line_target, sum(line_counts.values())),
        weights=line_weights,
    )
    for macro in sorted(line_quotas):
        chosen = _select_stratified(
            line_groups.get(macro, []),
            target=line_quotas[macro],
            seed=seed,
        )
        selected.extend(chosen)
        counts[f"selected_line_macro_{macro}"] += len(chosen)

    counts["selected_shape_reconstruction"] = sum(
        count for key, count in counts.items() if key.startswith("selected_shape_type_")
    )
    counts["selected_line_reconstruction"] = sum(
        count for key, count in counts.items() if key.startswith("selected_line_macro_")
    )
    selected.sort(key=lambda candidate: (candidate.stem, candidate.instance_index))
    return selected, counts


def _scale_local_bbox(
    bbox: list[int | float],
    *,
    scale_x: float,
    scale_y: float,
) -> list[int | float]:
    values = [
        float(bbox[0]) * scale_x,
        float(bbox[1]) * scale_y,
        float(bbox[2]) * scale_x,
        float(bbox[3]) * scale_y,
    ]
    return [
        int(round(value)) if abs(value - round(value)) < 1e-6 else round(value, 3)
        for value in values
    ]


def _shard_for_stem(stem: str) -> str:
    return stem[:2] if len(stem) >= 2 else "00"


def _image_rel(split: str, shard: str, filename: str) -> str:
    return f"../images/{split}/{shard}/{filename}"


def _make_rows_for_instance(
    *,
    task: str,
    label: str,
    source_label: str,
    stem: str,
    instance_index: int,
    image: Image.Image,
    source_json: Path,
    source_image_rel: str,
    source_size: list[int],
    source_background: str,
    source_bbox: tuple[float, float, float, float],
    params: dict[str, Any],
    config: BuildConfig,
) -> tuple[str, str, str, str, bool]:
    image_width, image_height = int(source_size[0]), int(source_size[1])
    scale_bucket, padding_ratio = _padding_policy(
        config,
        task=task,
        stem=stem,
        instance_index=instance_index,
    )
    crop_box = _crop_box(
        source_bbox,
        image_width=image_width,
        image_height=image_height,
        padding_ratio=padding_ratio,
        max_aspect_ratio=config.max_aspect_ratio,
    )
    left, top, right, bottom = crop_box
    crop_width = right - left
    crop_height = bottom - top
    if crop_width < config.min_crop_size or crop_height < config.min_crop_size:
        raise ValueError("crop too small")
    if _aspect_ratio(crop_width, crop_height) >= 200:
        raise ValueError("crop aspect ratio is incompatible with Qwen smart_resize")

    target_params = _target_parameters(
        label,
        params,
        left=left,
        top=top,
        crop_width=crop_width,
        crop_height=crop_height,
    )
    target = {"type": label, "parameters": target_params}

    crop_image = image.crop(crop_box)
    output_width, output_height = crop_width, crop_height
    low_resolution_size = _low_resolution_target(
        config,
        task=task,
        stem=stem,
        instance_index=instance_index,
        width=crop_width,
        height=crop_height,
    )
    low_resolution_applied = low_resolution_size is not None
    if low_resolution_size is not None:
        output_width, output_height = low_resolution_size
        resized_image = crop_image.resize(low_resolution_size, Image.Resampling.LANCZOS)
        crop_image.close()
        crop_image = resized_image

    sample_id = (
        f"{stem}__{label}_{instance_index:04d}"
        if source_label == label
        else f"{stem}__{source_label}_as_{label}_{instance_index:04d}"
    )
    shard = _shard_for_stem(stem)
    filename = f"{sample_id}.png"
    image_output_path = config.output_root / task / "images" / config.split / shard / filename
    image_output_path.parent.mkdir(parents=True, exist_ok=True)
    crop_image.save(image_output_path, compress_level=1)
    crop_image.close()

    crop_bbox = _translate_bbox(list(source_bbox), left=left, top=top)
    if low_resolution_applied:
        crop_bbox = _scale_local_bbox(
            crop_bbox,
            scale_x=output_width / crop_width,
            scale_y=output_height / crop_height,
        )
    augmentation: dict[str, Any] = {
        "name": "bbox_padding_crop",
        "padding_ratio": padding_ratio,
        "scale_bucket": scale_bucket,
        "max_aspect_ratio": config.max_aspect_ratio,
    }
    if low_resolution_applied:
        augmentation["low_resolution_resize"] = {
            "source_size": [crop_width, crop_height],
            "output_size": [output_width, output_height],
            "resample": "lanczos",
        }
    structured_extra = {
        "task": task,
        "split": config.split,
        "view_type": f"{label}_reconstruction_crop",
        "source_dataset": config.dataset_root.name,
        "source_json": str(source_json.relative_to(config.dataset_root)),
        "source_image": source_image_rel,
        "source_image_width": image_width,
        "source_image_height": image_height,
        "source_background": source_background,
        "source_instance_index": instance_index,
        "source_label": source_label,
        "source_bbox": list(source_bbox),
        "crop_box": list(crop_box),
        "padding_ratio": padding_ratio,
        "max_aspect_ratio": config.max_aspect_ratio,
        "target_coordinate_space": "qwen_0_999_crop",
        "target_num_bins": 1000,
        "augmentation": augmentation,
    }
    structured_row = {
        "sample_id": sample_id,
        "image_path": _image_rel(config.split, shard, filename),
        "image_width": output_width,
        "image_height": output_height,
        "instances": [
            {
                "label": label,
                "bbox": crop_bbox,
                "parameters": target_params,
            }
        ],
        "extra": structured_extra,
    }

    prompt_info = _prompt_variant(config, task=task, stem=stem, instance_index=instance_index)
    sft_extra = {
        "prompt_id": prompt_info.prompt_id,
        "source_sample_id": stem,
        "source_type": "synthetic_gt_standard",
        "image_width": output_width,
        "image_height": output_height,
        "target_coordinate_space": "qwen_0_999_crop",
        "num_bins": 1000,
        "structured_extra": structured_extra,
    }
    if prompt_info.output_schema:
        sft_extra["output_schema"] = prompt_info.output_schema
    sft_row = {
        "image_path": _image_rel(config.split, shard, filename),
        "sample_id": sample_id,
        "dataset_name": task,
        "system_prompt": prompt_info.system_prompt,
        "user_prompt": prompt_info.user_prompt,
        "target_text": _json_dumps(target),
        "extra": sft_extra,
    }
    return (
        task,
        _json_dumps(structured_row),
        _json_dumps(sft_row),
        scale_bucket,
        low_resolution_applied,
    )


def _build_for_json(args: tuple[str, frozenset[int] | None, BuildConfig]) -> WorkerResult:
    stem, selected_indices, config = args
    source_json = config.dataset_root / "gt_standard" / f"{stem}.json"
    image_path = config.dataset_root / "img" / f"{stem}.png"
    counts: Counter[str] = Counter()
    structured_rows: dict[str, list[str]] = {task: [] for task in config.selected_tasks}
    sft_rows: dict[str, list[str]] = {task: [] for task in config.selected_tasks}

    try:
        obj = _load_json(source_json)
    except Exception:
        counts["json_parse_error"] += 1
        return WorkerResult(structured_rows, sft_rows, dict(counts))
    source_size = obj.get("size")
    if not (
        isinstance(source_size, list)
        and len(source_size) == 2
        and all(_is_number(value) for value in source_size)
    ):
        counts["bad_size"] += 1
        return WorkerResult(structured_rows, sft_rows, dict(counts))
    if not image_path.exists():
        counts["missing_image"] += 1
        return WorkerResult(structured_rows, sft_rows, dict(counts))
    layout = obj.get("layout")
    if not isinstance(layout, list):
        counts["bad_layout"] += 1
        return WorkerResult(structured_rows, sft_rows, dict(counts))

    image_width, image_height = int(source_size[0]), int(source_size[1])
    source_background = str(obj.get("background") or "")
    source_image_rel = str(image_path.relative_to(config.dataset_root))
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception:
        counts["image_open_error"] += 1
        return WorkerResult(structured_rows, sft_rows, dict(counts))
    try:
        for instance_index, instance in enumerate(layout):
            if selected_indices is not None and instance_index not in selected_indices:
                continue
            if not isinstance(instance, dict):
                counts["bad_instance"] += 1
                continue
            source_label = str(instance.get("type") or "")
            visual_other = (
                config.include_visual_other_negatives
                and source_label in {"icon", "image"}
                and "shape_reconstruction" in config.selected_tasks
            )
            label = "shape" if visual_other else source_label
            task = f"{label}_reconstruction"
            if task not in config.selected_tasks:
                continue
            source_params = instance.get("parameters")
            if not visual_other and not isinstance(source_params, dict):
                counts[f"{task}_missing_parameters"] += 1
                continue
            params = {"shape_type": "other"} if visual_other else source_params
            assert isinstance(params, dict)
            bbox = _clean_bbox(
                instance.get("bbox"),
                image_width=image_width,
                image_height=image_height,
                skip_oob_bbox=config.skip_oob_bbox,
            )
            if bbox is None:
                counts[f"{task}_bad_bbox"] += 1
                continue
            try:
                row_task, structured_line, sft_line, scale_bucket, low_resolution_applied = (
                    _make_rows_for_instance(
                        task=task,
                        label=label,
                        source_label=source_label,
                        stem=stem,
                        instance_index=instance_index,
                        image=image,
                        source_json=source_json,
                        source_image_rel=source_image_rel,
                        source_size=source_size,
                        source_background=source_background,
                        source_bbox=bbox,
                        params=params,
                        config=config,
                    )
                )
            except Exception:
                counts[f"{task}_skipped"] += 1
                continue
            structured_rows[row_task].append(structured_line)
            sft_rows[row_task].append(sft_line)
            counts[f"{row_task}_rows"] += 1
            counts[f"{row_task}_scale_{scale_bucket}"] += 1
            if low_resolution_applied:
                counts[f"{row_task}_low_resolution"] += 1
    finally:
        image.close()
    return WorkerResult(structured_rows, sft_rows, dict(counts))


def _load_prompt_info() -> dict[str, PromptPoolInfo]:
    infos: dict[str, PromptPoolInfo] = {}
    for task, task_config in TASK_CONFIGS.items():
        prompts = load_prompt_pool(Path(str(task_config["prompt"])))
        infos[task] = PromptPoolInfo(
            variants=tuple(
                PromptVariantInfo(
                    prompt_id=prompt.prompt_id,
                    system_prompt=prompt.system_prompt,
                    user_prompt=prompt.user_prompt,
                    output_schema=(
                        str(prompt.metadata.get("output_schema"))
                        if prompt.metadata.get("output_schema") is not None
                        else None
                    ),
                )
                for prompt in prompts
            )
        )
    return infos


def _write_readme(
    task: str,
    *,
    output_root: Path,
    args: argparse.Namespace,
    counters: Counter[str],
) -> None:
    task_config = TASK_CONFIGS[task]
    label = str(task_config["label"])
    rows = counters.get(f"{task}_rows", 0)
    bad_bbox = counters.get(f"{task}_bad_bbox", 0)
    skipped = counters.get(f"{task}_skipped", 0)
    if task == "shape_reconstruction":
        available_prefix = "available_shape_type_"
        selected_prefix = "selected_shape_type_"
    else:
        available_prefix = "available_line_macro_"
        selected_prefix = "selected_line_macro_"
    distribution_keys = sorted(
        key.removeprefix(available_prefix) for key in counters if key.startswith(available_prefix)
    )
    distribution_lines = ["| Group | Available | Selected |", "|---|---:|---:|"]
    distribution_lines.extend(
        f"| {key} | {counters.get(f'{available_prefix}{key}', 0)} | "
        f"{counters.get(f'{selected_prefix}{key}', 0)} |"
        for key in distribution_keys
    )
    distribution_table = "\n".join(distribution_lines)
    content = f"""# {task} SFT/structured dataset

Generated from synthetic `gt_standard` annotations.

- Source dataset: `{args.dataset_root}`
- Source annotation: `gt_standard`
- Target label: `{label}`
- Split: `train` only
- Workers: `{args.workers}`
- Seed: `{args.seed}`
- Sampling profile: `{args.sampling_profile}`
- Requested task target: `{args.shape_target if task == "shape_reconstruction" else args.line_target}`
- Padding policy: `{"70% tight 0.08-0.15, 25% medium 0.15-0.25, 5% context 0.25-0.40" if args.multi_scale else f"random uniform {args.padding_min} to {args.padding_max}"}`
- Low-resolution resize ratio: `{args.shape_low_resolution_ratio if task == "shape_reconstruction" else args.line_low_resolution_ratio}`
- Max aspect ratio: `{args.max_aspect_ratio}` before Qwen processor
- Crop policy: one bbox-padded, aspect-capped crop per `{label}` instance
- Coordinate policy: target geometry coordinates use Qwen-style integer `0..999` bins in the crop
- Target policy: `target_text` is compact JSON shaped as `{{"type":"{label}","parameters":...}}`
- Prompt pool: `{task_config["prompt"]}`
- OOB bbox policy: `{"skip" if args.skip_oob_bbox else "clip"}`

## Counts

- Rows: {rows}
- Bad/OOB bbox skipped: {bad_bbox}
- Target transform skipped: {skipped}
- Tight-scale rows: {counters.get(f"{task}_scale_tight", 0)}
- Medium-scale rows: {counters.get(f"{task}_scale_medium", 0)}
- Context-scale rows: {counters.get(f"{task}_scale_context", 0)}
- Low-resolution resized rows: {counters.get(f"{task}_low_resolution", 0)}

## Sampling Distribution

{distribution_table}

Images are stored under `images/train/<shard>/`. `structured/train.jsonl` and
`sft/train.jsonl` reference those crops with relative paths.
"""
    _atomic_write_text(output_root / task / "README.md", content)


def _prepare_output_dirs(output_root: Path, *, clean: bool, selected_tasks: frozenset[str]) -> None:
    for task in selected_tasks:
        task_root = output_root / task
        if clean and task_root.exists():
            shutil.rmtree(task_root)
        for subdir in ("structured", "sft", "images/train"):
            (task_root / subdir).mkdir(parents=True, exist_ok=True)
        for split_file in (
            "structured/train.jsonl",
            "structured/val.jsonl",
            "sft/train.jsonl",
            "sft/val.jsonl",
        ):
            path = task_root / split_file
            if path.exists():
                path.unlink()
        _atomic_write_text(task_root / "structured" / "val.jsonl", "")
        _atomic_write_text(task_root / "sft" / "val.jsonl", "")


def _iter_stems(dataset_root: Path, limit: int | None) -> list[str]:
    stems = sorted(path.stem for path in (dataset_root / "gt_standard").glob("*.json"))
    if limit is not None:
        return stems[:limit]
    return stems


def build(args: argparse.Namespace) -> Counter[str]:
    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)
    if not (dataset_root / "gt_standard").is_dir():
        raise FileNotFoundError(dataset_root / "gt_standard")
    if not (dataset_root / "img").is_dir():
        raise FileNotFoundError(dataset_root / "img")
    if args.padding_min < 0 or args.padding_max < args.padding_min:
        raise ValueError("padding range must satisfy 0 <= padding_min <= padding_max")
    if args.shape_target < 0 or args.line_target < 0:
        raise ValueError("sampling targets must be non-negative")
    for name in ("shape_low_resolution_ratio", "line_low_resolution_ratio"):
        value = float(getattr(args, name))
        if value < 0 or value > 1:
            raise ValueError(f"{name} must be between 0 and 1")
    selected_tasks = frozenset(str(task) for task in args.tasks)
    unknown_tasks = sorted(selected_tasks - set(TASK_CONFIGS))
    if unknown_tasks:
        raise ValueError(f"Unknown task(s): {unknown_tasks}")
    if not selected_tasks:
        raise ValueError("At least one task must be selected")

    _prepare_output_dirs(output_root, clean=bool(args.clean), selected_tasks=selected_tasks)
    prompt_info = _load_prompt_info()
    config = BuildConfig(
        dataset_root=dataset_root,
        output_root=output_root,
        split="train",
        selected_tasks=selected_tasks,
        padding_min=float(args.padding_min),
        padding_max=float(args.padding_max),
        seed=int(args.seed),
        min_crop_size=int(args.min_crop_size),
        max_aspect_ratio=float(args.max_aspect_ratio),
        skip_oob_bbox=bool(args.skip_oob_bbox),
        multi_scale=bool(args.multi_scale),
        shape_low_resolution_ratio=float(args.shape_low_resolution_ratio),
        line_low_resolution_ratio=float(args.line_low_resolution_ratio),
        include_visual_other_negatives=args.sampling_profile == "balanced_v2",
        prompt_info=prompt_info,
    )
    stems = _iter_stems(dataset_root, args.limit)
    counters: Counter[str] = Counter()
    selected_by_stem: dict[str, frozenset[int]] | None = None
    if args.sampling_profile in {"balanced_v1", "balanced_v2"}:
        inventory: list[Candidate] = []
        inventory_args = [(stem, config) for stem in stems]
        inventory_workers = min(8, max(1, int(args.workers)))
        if inventory_workers <= 1:
            inventory_results = map(_inventory_for_json, inventory_args)
        else:
            inventory_executor = ProcessPoolExecutor(max_workers=inventory_workers)
            inventory_results = inventory_executor.map(
                _inventory_for_json,
                inventory_args,
                chunksize=max(16, int(args.chunksize)),
            )
        try:
            for values in inventory_results:
                inventory.extend(values)
        finally:
            if inventory_workers > 1:
                inventory_executor.shutdown(wait=True)
        selected, selection_counts = _select_balanced_candidates(
            inventory,
            shape_target=int(args.shape_target),
            line_target=int(args.line_target),
            seed=int(args.seed),
        )
        counters.update(selection_counts)
        mutable_by_stem: dict[str, set[int]] = defaultdict(set)
        for candidate in selected:
            mutable_by_stem[candidate.stem].add(candidate.instance_index)
        selected_by_stem = {stem: frozenset(indices) for stem, indices in mutable_by_stem.items()}
        stems = sorted(selected_by_stem)
        del inventory
        del selected
    output_handles: dict[tuple[str, str], Any] = {}
    try:
        for task in TASK_CONFIGS:
            if task not in selected_tasks:
                continue
            for kind in ("structured", "sft"):
                path = output_root / task / kind / "train.jsonl"
                output_handles[(task, kind)] = path.open("w", encoding="utf-8")
        worker_args = [
            (stem, selected_by_stem.get(stem) if selected_by_stem is not None else None, config)
            for stem in stems
        ]
        if args.workers <= 1:
            results = map(_build_for_json, worker_args)
        else:
            executor = ProcessPoolExecutor(max_workers=int(args.workers))
            results = executor.map(_build_for_json, worker_args, chunksize=int(args.chunksize))
        try:
            for result in results:
                counters.update(result.counts)
                for task, rows in result.structured_rows.items():
                    handle = output_handles[(task, "structured")]
                    for row in rows:
                        handle.write(row + "\n")
                for task, rows in result.sft_rows.items():
                    handle = output_handles[(task, "sft")]
                    for row in rows:
                        handle.write(row + "\n")
        finally:
            if args.workers > 1:
                executor.shutdown(wait=True)
    finally:
        for handle in output_handles.values():
            handle.close()
    counters["source_json"] = len(stems)
    for task in selected_tasks:
        _write_readme(task, output_root=output_root, args=args, counters=counters)
    return counters


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build shape/line reconstruction crop SFT data from synthetic gt_standard."
    )
    parser.add_argument(
        "--dataset-root",
        default="data/regulated_layout_dataset_v7_20260705",
        help="Synthetic dataset root containing img/ and gt_standard/.",
    )
    parser.add_argument("--output-root", default="data")
    parser.add_argument("--workers", type=int, default=40)
    parser.add_argument("--chunksize", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--padding-min", type=float, default=0.1)
    parser.add_argument("--padding-max", type=float, default=0.2)
    parser.add_argument(
        "--sampling-profile",
        choices=("all", "balanced_v1", "balanced_v2"),
        default="all",
        help=(
            "Use balanced_v1 for deterministic task-specific sampling; balanced_v2 also adds "
            "sampled icon/image crops as shape_type=other."
        ),
    )
    parser.add_argument("--shape-target", type=int, default=300000)
    parser.add_argument("--line-target", type=int, default=300000)
    parser.add_argument("--multi-scale", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--shape-low-resolution-ratio", type=float, default=0.20)
    parser.add_argument("--line-low-resolution-ratio", type=float, default=0.15)
    parser.add_argument("--min-crop-size", type=int, default=4)
    parser.add_argument(
        "--max-aspect-ratio",
        type=float,
        default=60.0,
        help=(
            "Expand the shorter crop side to cap crop aspect ratio. "
            "Default 60 is a conservative guard for Qwen smart_resize, which rejects >=200."
        ),
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=sorted(TASK_CONFIGS),
        choices=sorted(TASK_CONFIGS),
        help="Task datasets to rebuild.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-oob-bbox", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    counters = build(args)
    print(json.dumps(dict(sorted(counters.items())), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
