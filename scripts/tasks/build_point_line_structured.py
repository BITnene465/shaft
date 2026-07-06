#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
SOURCE_LABELS = frozenset(("arrow", "line"))


@dataclass(frozen=True)
class BuildConfig:
    raw_root: Path
    split: str
    image_output_dir: Path
    padding_min: float
    padding_max: float
    val_padding: float
    seed: int
    min_crop_size: int


@dataclass(frozen=True)
class SplitBuildResult:
    rows: list[dict[str, Any]]
    source_count: int
    skipped_count: int
    line_count: int


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent) as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
        tmp_path = Path(f.name)
    os.replace(tmp_path, path)


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    body = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows
    )
    _atomic_write_text(path, body)


def _rel_id(json_rel: str) -> str:
    return json_rel.replace("/json/", "__").replace("/", "__").replace(".json", "")


def _find_image_path(raw_root: Path, raw_record: dict[str, Any], json_rel: str) -> Path:
    image_path = raw_record.get("image_path")
    if image_path:
        candidate = raw_root / str(image_path)
        if candidate.exists():
            return candidate

    stem = Path(json_rel).stem
    for image_dir in (raw_root / "images", raw_root / Path(json_rel).parts[0] / "images"):
        for suffix in IMAGE_SUFFIXES:
            candidate = image_dir / f"{stem}{suffix}"
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"Cannot find image for {json_rel}")


def _clean_bbox(
    bbox: Any,
    *,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
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


def _clean_linestrip(
    linestrip: Any,
    *,
    image_width: int,
    image_height: int,
) -> list[list[float]]:
    points: list[list[float]] = []
    if not isinstance(linestrip, list):
        return points
    for point in linestrip:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        try:
            x = float(point[0])
            y = float(point[1])
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        if 0 <= x <= image_width and 0 <= y <= image_height:
            points.append([x, y])
    return points


def _line_extent(
    bbox: tuple[float, float, float, float],
    linestrip: list[list[float]],
) -> tuple[float, float, float, float]:
    xs = [bbox[0], bbox[2], *(point[0] for point in linestrip)]
    ys = [bbox[1], bbox[3], *(point[1] for point in linestrip)]
    return min(xs), min(ys), max(xs), max(ys)


def _crop_box(
    extent: tuple[float, float, float, float],
    *,
    image_width: int,
    image_height: int,
    padding_ratio: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = extent
    width = x2 - x1
    height = y2 - y1
    pad_x = width * padding_ratio
    pad_y = height * padding_ratio
    left = max(0, int(math.floor(x1 - pad_x)))
    top = max(0, int(math.floor(y1 - pad_y)))
    right = min(image_width, int(math.ceil(x2 + pad_x)))
    bottom = min(image_height, int(math.ceil(y2 + pad_y)))
    return left, top, right, bottom


def _padding_ratio(
    *,
    split: str,
    seed: int,
    json_rel: str,
    instance_index: int,
    padding_min: float,
    padding_max: float,
    val_padding: float,
) -> float:
    if split == "val":
        return val_padding
    rng = random.Random(f"{seed}:{json_rel}:{instance_index}")
    return rng.uniform(padding_min, padding_max)


def _translate_bbox(
    bbox: tuple[float, float, float, float],
    crop_box: tuple[int, int, int, int],
) -> list[float]:
    left, top, _, _ = crop_box
    x1, y1, x2, y2 = bbox
    return [x1 - left, y1 - top, x2 - left, y2 - top]


def _translate_linestrip(
    linestrip: list[list[float]],
    crop_box: tuple[int, int, int, int],
) -> list[list[float]]:
    left, top, _, _ = crop_box
    return [[x - left, y - top] for x, y in linestrip]


def _points_inside(points: list[list[float]], *, width: int, height: int) -> bool:
    return all(0 <= x <= width and 0 <= y <= height for x, y in points)


def _build_rows_for_json(args: tuple[str, BuildConfig]) -> tuple[list[dict[str, Any]], int, int]:
    json_rel, config = args
    json_path = config.raw_root / json_rel
    raw_record = _load_json(json_path)
    image_path = _find_image_path(config.raw_root, raw_record, json_rel)
    image = Image.open(image_path).convert("RGB")
    image_width, image_height = image.size

    rows: list[dict[str, Any]] = []
    skipped = 0
    line_count = 0
    raw_instances = raw_record.get("instances") or []
    for instance_index, instance in enumerate(raw_instances):
        source_label = str(instance.get("label") or "")
        if source_label not in SOURCE_LABELS:
            continue
        linestrip = _clean_linestrip(
            instance.get("linestrip"),
            image_width=image_width,
            image_height=image_height,
        )
        if len(linestrip) < 2:
            skipped += 1
            continue
        bbox = _clean_bbox(
            instance.get("bbox"),
            image_width=image_width,
            image_height=image_height,
        )
        if bbox is None:
            skipped += 1
            continue

        extent = _line_extent(bbox, linestrip)
        ratio = _padding_ratio(
            split=config.split,
            seed=config.seed,
            json_rel=json_rel,
            instance_index=instance_index,
            padding_min=config.padding_min,
            padding_max=config.padding_max,
            val_padding=config.val_padding,
        )
        crop_box = _crop_box(
            extent,
            image_width=image_width,
            image_height=image_height,
            padding_ratio=ratio,
        )
        left, top, right, bottom = crop_box
        crop_width = right - left
        crop_height = bottom - top
        if crop_width < config.min_crop_size or crop_height < config.min_crop_size:
            skipped += 1
            continue
        crop_linestrip = _translate_linestrip(linestrip, crop_box)
        if not _points_inside(crop_linestrip, width=crop_width, height=crop_height):
            skipped += 1
            continue

        line_count += 1
        rel_id = _rel_id(json_rel)
        sample_id = f"{rel_id}__line_{instance_index:04d}"
        output_name = f"{sample_id}.png"
        output_path = config.image_output_dir / output_name
        image.crop(crop_box).save(output_path)

        crop_bbox = _translate_bbox(bbox, crop_box)
        rows.append(
            {
                "sample_id": sample_id,
                "image_path": f"../images/{config.split}/{output_name}",
                "image_width": crop_width,
                "image_height": crop_height,
                "instances": [
                    {
                        "label": "line",
                        "bbox": crop_bbox,
                        "linestrip": crop_linestrip,
                    }
                ],
                "extra": {
                    "task": "point_line",
                    "split": config.split,
                    "view_type": "line_crop",
                    "source_json": json_rel,
                    "source_image": str(
                        raw_record.get("image_path") or image_path.relative_to(config.raw_root)
                    ),
                    "source_image_width": image_width,
                    "source_image_height": image_height,
                    "source_instance_index": instance_index,
                    "source_label": source_label,
                    "source_bbox": list(bbox),
                    "source_linestrip": linestrip,
                    "crop_box": list(crop_box),
                    "padding_ratio": ratio,
                    "augmentation": {
                        "name": "bbox_padding_crop",
                        "padding_ratio": ratio,
                    },
                },
            }
        )
    image.close()
    return rows, skipped, line_count


def _write_readme(
    output_root: Path,
    *,
    train_result: SplitBuildResult,
    val_result: SplitBuildResult,
    args: argparse.Namespace,
) -> None:
    content = f"""# point_line structured

Generated from `data/raw` arrow/line instances with valid `linestrip`.

- Train split source: `{args.train_split}`
- Val split source: `{args.val_split}`
- Workers: `{args.workers}`
- Seed: `{args.seed}`
- Train padding ratio: `{args.padding_min}` to `{args.padding_max}`
- Val padding ratio: `{args.val_padding}`
- Augmentation: deterministic bbox padding crop only; no duplicated jitter rows or pixel
  augmentation
- Row policy: one crop row per valid raw `arrow` or raw `line` instance; output label is always
  model-facing `line`.

## Counts

| split | source json | rows | skipped line instances |
| --- | ---: | ---: | ---: |
| train | {train_result.source_count} | {len(train_result.rows)} | {train_result.skipped_count} |
| val | {val_result.source_count} | {len(val_result.rows)} | {val_result.skipped_count} |

Each JSONL row references a generated crop image under `images/<split>/` and stores crop-local
`bbox` plus crop-local ordered `linestrip` in `instances[0]`.
"""
    _atomic_write_text(output_root / "README.md", content)


def build_split(
    *,
    split_path: Path,
    output_path: Path,
    config: BuildConfig,
    workers: int,
) -> SplitBuildResult:
    entries = _read_split(split_path)
    missing_entries = [entry for entry in entries if not (config.raw_root / entry).exists()]
    if missing_entries:
        examples = ", ".join(missing_entries[:5])
        raise FileNotFoundError(
            f"Split {split_path} contains {len(missing_entries)} entries without GT JSON under "
            f"{config.raw_root}. Examples: {examples}."
        )
    config.image_output_dir.mkdir(parents=True, exist_ok=True)
    worker_args = [(entry, config) for entry in entries]
    rows: list[dict[str, Any]] = []
    skipped_count = 0
    line_count = 0
    if workers <= 1:
        results = [_build_rows_for_json(item) for item in worker_args]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_build_rows_for_json, worker_args, chunksize=8))
    for split_rows, skipped, lines in results:
        rows.extend(split_rows)
        skipped_count += skipped
        line_count += lines
    rows.sort(key=lambda row: str(row["sample_id"]))
    _write_jsonl_atomic(output_path, rows)
    return SplitBuildResult(
        rows=rows,
        source_count=len(entries),
        skipped_count=skipped_count,
        line_count=line_count,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build point_line structured crop dataset.")
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--output-root", default="data/point_line")
    parser.add_argument("--train-split", required=True)
    parser.add_argument("--val-split", required=True)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--padding-min", type=float, default=0.2)
    parser.add_argument("--padding-max", type=float, default=0.5)
    parser.add_argument("--val-padding", type=float, default=0.35)
    parser.add_argument("--min-crop-size", type=int, default=4)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    output_root = Path(args.output_root)
    if not raw_root.exists():
        raise FileNotFoundError(raw_root)
    if float(args.padding_min) < 0 or float(args.padding_max) < float(args.padding_min):
        raise ValueError("padding range must satisfy 0 <= padding_min <= padding_max")
    if float(args.val_padding) < 0:
        raise ValueError("--val-padding must be >= 0")

    if args.clean and output_root.exists():
        shutil.rmtree(output_root)

    structured_dir = output_root / "structured"
    image_root = output_root / "images"
    structured_dir.mkdir(parents=True, exist_ok=True)
    image_root.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val"):
        split_image_dir = image_root / split
        if split_image_dir.exists():
            shutil.rmtree(split_image_dir)
        split_image_dir.mkdir(parents=True, exist_ok=True)

    train_config = BuildConfig(
        raw_root=raw_root,
        split="train",
        image_output_dir=image_root / "train",
        padding_min=float(args.padding_min),
        padding_max=float(args.padding_max),
        val_padding=float(args.val_padding),
        seed=int(args.seed),
        min_crop_size=int(args.min_crop_size),
    )
    val_config = BuildConfig(
        raw_root=raw_root,
        split="val",
        image_output_dir=image_root / "val",
        padding_min=float(args.padding_min),
        padding_max=float(args.padding_max),
        val_padding=float(args.val_padding),
        seed=int(args.seed),
        min_crop_size=int(args.min_crop_size),
    )
    train_result = build_split(
        split_path=Path(args.train_split),
        output_path=structured_dir / "train.jsonl",
        config=train_config,
        workers=int(args.workers),
    )
    val_result = build_split(
        split_path=Path(args.val_split),
        output_path=structured_dir / "val.jsonl",
        config=val_config,
        workers=int(args.workers),
    )
    _write_readme(output_root, train_result=train_result, val_result=val_result, args=args)
    print(
        json.dumps(
            {
                "train_rows": len(train_result.rows),
                "train_source_json": train_result.source_count,
                "train_skipped": train_result.skipped_count,
                "val_rows": len(val_result.rows),
                "val_source_json": val_result.source_count,
                "val_skipped": val_result.skipped_count,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
