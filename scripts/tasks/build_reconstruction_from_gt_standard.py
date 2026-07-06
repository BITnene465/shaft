#!/usr/bin/env python3
from __future__ import annotations

import argparse
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

from PIL import Image

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
    prompt_info: dict[str, PromptPoolInfo]


@dataclass(frozen=True)
class WorkerResult:
    structured_rows: dict[str, list[str]]
    sft_rows: dict[str, list[str]]
    counts: dict[str, int]


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


def _padding_ratio(config: BuildConfig, *, stem: str, instance_index: int) -> float:
    rng = random.Random(f"{config.seed}:{stem}:{instance_index}")
    return rng.uniform(config.padding_min, config.padding_max)


def _prompt_variant(config: BuildConfig, *, task: str, stem: str, instance_index: int) -> PromptVariantInfo:
    variants = config.prompt_info[task].variants
    if not variants:
        raise ValueError(f"Prompt pool for {task} is empty")
    rng = random.Random(f"{config.seed}:prompt:{task}:{stem}:{instance_index}")
    return variants[rng.randrange(len(variants))]


def _quantize_axis(value: float, *, origin: int, size: int, num_bins: int = 1000) -> int:
    if size <= 1:
        return 0
    local_value = float(value) - float(origin)
    clipped = min(max(local_value, 0.0), float(size - 1))
    normalized = clipped / float(size - 1)
    return int(round(normalized * float(num_bins - 1)))


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


def _shard_for_stem(stem: str) -> str:
    return stem[:2] if len(stem) >= 2 else "00"


def _image_rel(split: str, shard: str, filename: str) -> str:
    return f"../images/{split}/{shard}/{filename}"


def _make_rows_for_instance(
    *,
    task: str,
    label: str,
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
) -> tuple[str, str, str]:
    image_width, image_height = int(source_size[0]), int(source_size[1])
    padding_ratio = _padding_ratio(config, stem=stem, instance_index=instance_index)
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

    sample_id = f"{stem}__{label}_{instance_index:04d}"
    shard = _shard_for_stem(stem)
    filename = f"{sample_id}.png"
    image_output_path = config.output_root / task / "images" / config.split / shard / filename
    image_output_path.parent.mkdir(parents=True, exist_ok=True)
    image.crop(crop_box).save(image_output_path, compress_level=1)

    crop_bbox = _translate_bbox(list(source_bbox), left=left, top=top)
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
        "source_bbox": list(source_bbox),
        "crop_box": list(crop_box),
        "padding_ratio": padding_ratio,
        "max_aspect_ratio": config.max_aspect_ratio,
        "target_coordinate_space": "qwen_0_999_crop",
        "target_num_bins": 1000,
        "augmentation": {
            "name": "bbox_padding_crop",
            "padding_ratio": padding_ratio,
            "max_aspect_ratio": config.max_aspect_ratio,
        },
    }
    structured_row = {
        "sample_id": sample_id,
        "image_path": _image_rel(config.split, shard, filename),
        "image_width": crop_width,
        "image_height": crop_height,
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
        "image_width": crop_width,
        "image_height": crop_height,
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
    return task, _json_dumps(structured_row), _json_dumps(sft_row)


def _build_for_json(args: tuple[str, BuildConfig]) -> WorkerResult:
    stem, config = args
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
            if not isinstance(instance, dict):
                counts["bad_instance"] += 1
                continue
            label = str(instance.get("type") or "")
            task = f"{label}_reconstruction"
            if task not in config.selected_tasks:
                continue
            params = instance.get("parameters")
            if not isinstance(params, dict):
                counts[f"{task}_missing_parameters"] += 1
                continue
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
                row_task, structured_line, sft_line = _make_rows_for_instance(
                    task=task,
                    label=label,
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
            except Exception:
                counts[f"{task}_skipped"] += 1
                continue
            structured_rows[row_task].append(structured_line)
            sft_rows[row_task].append(sft_line)
            counts[f"{row_task}_rows"] += 1
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
    content = f"""# {task} SFT/structured dataset

Generated from synthetic `gt_standard` annotations.

- Source dataset: `{args.dataset_root}`
- Source annotation: `gt_standard`
- Target label: `{label}`
- Split: `train` only
- Workers: `{args.workers}`
- Seed: `{args.seed}`
- Padding ratio: random uniform `{args.padding_min}` to `{args.padding_max}` per instance
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
        for split_file in ("structured/train.jsonl", "structured/val.jsonl", "sft/train.jsonl", "sft/val.jsonl"):
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
        prompt_info=prompt_info,
    )
    stems = _iter_stems(dataset_root, args.limit)
    counters: Counter[str] = Counter()
    output_handles: dict[tuple[str, str], Any] = {}
    try:
        for task in TASK_CONFIGS:
            if task not in selected_tasks:
                continue
            for kind in ("structured", "sft"):
                path = output_root / task / kind / "train.jsonl"
                output_handles[(task, kind)] = path.open("w", encoding="utf-8")
        worker_args = [(stem, config) for stem in stems]
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
