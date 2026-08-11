#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from shaft.data.synthetic_realism import (
    SYNTHETIC_REALISM_PROFILE,
    apply_synthetic_realism_augmentation,
    sample_synthetic_realism_augmentation,
)


ALLOWED_LABELS = ("shape", "icon", "image", "line")
BACKGROUND_SHAPE_AREA_RATIO = 0.9


@dataclass(frozen=True)
class BuildConfig:
    dataset_root: Path
    output_root: Path
    source_name: str
    seed: int
    png_compress_level: int


@dataclass(frozen=True)
class BuildResult:
    row: dict[str, Any]
    label_counts: dict[str, int]
    dropped_counts: dict[str, int]
    augmentation_counts: dict[str, int]


def _atomic_text_writer(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        delete=False,
        dir=path.parent,
    )


def _normalize_stem(value: str) -> str:
    return Path(value.strip()).stem


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _image_output_path(output_root: Path, sample_id: str) -> Path:
    shard = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:2]
    return output_root / "images" / "train" / shard / f"{sample_id}.png"


def _load_split(path: Path) -> list[str]:
    stems = [_normalize_stem(line) for line in path.read_text(encoding="utf-8").splitlines()]
    stems = [stem for stem in stems if stem]
    if len(stems) != len(set(stems)):
        duplicate_counts = Counter(stems)
        duplicates = sorted(stem for stem, count in duplicate_counts.items() if count > 1)
        raise ValueError(f"Duplicate source ids in {path}: {duplicates[:10]}")
    return stems


def _normalize_bbox(
    item: dict[str, Any],
    *,
    image_width: int,
    image_height: int,
) -> list[float] | None:
    raw_bbox = item.get("bbox")
    if not isinstance(raw_bbox, list | tuple) or len(raw_bbox) != 4:
        location = item.get("location")
        if not isinstance(location, list | tuple) or len(location) != 4:
            return None
        x, y, width, height = [float(value) for value in location]
        raw_bbox = [x, y, x + width, y + height]

    x1, y1, x2, y2 = [float(value) for value in raw_bbox]
    x1 = min(max(x1, 0.0), float(image_width))
    y1 = min(max(y1, 0.0), float(image_height))
    x2 = min(max(x2, 0.0), float(image_width))
    y2 = min(max(y2, 0.0), float(image_height))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _is_background_shape(
    label: str,
    bbox: list[float],
    *,
    image_width: int,
    image_height: int,
) -> bool:
    if label != "shape":
        return False
    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    image_area = float(image_width * image_height)
    return image_area > 0 and area / image_area >= BACKGROUND_SHAPE_AREA_RATIO


def _build_one(item: tuple[str, BuildConfig]) -> BuildResult:
    stem, config = item
    annotation_path = config.dataset_root / "gt_standard" / f"{stem}.json"
    image_path = config.dataset_root / "img" / f"{stem}.png"
    if not annotation_path.is_file():
        raise FileNotFoundError(annotation_path)
    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {annotation_path}")
    size = payload.get("size")
    if not isinstance(size, list | tuple) or len(size) != 2:
        raise ValueError(f"Missing size [width, height]: {annotation_path}")
    image_width, image_height = [int(value) for value in size]
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"Invalid image size in {annotation_path}: {size!r}")
    sample_id = f"grounding_layout_sync__{stem}__full"
    augmentation = sample_synthetic_realism_augmentation(
        task="grounding_layout_sync",
        sample_id=sample_id,
        seed=config.seed,
        target_short_span=min(image_width, image_height),
        image_width=image_width,
        image_height=image_height,
        required_operations=("gaussian_noise",),
    )
    if not augmentation["operations"]:
        raise RuntimeError(f"Synthetic grounding augmentation is empty: {sample_id}")
    generated_image_path = _image_output_path(config.output_root, sample_id)
    generated_image_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as image:
        if image.size != (image_width, image_height):
            raise ValueError(
                f"Image/annotation size mismatch for {stem}: {image.size} != "
                f"{(image_width, image_height)}"
            )
        generated = apply_synthetic_realism_augmentation(image, augmentation)
        try:
            generated.save(
                generated_image_path,
                format="PNG",
                compress_level=config.png_compress_level,
            )
        finally:
            generated.close()

    label_counts: Counter[str] = Counter()
    dropped_counts: Counter[str] = Counter()
    instances: list[dict[str, Any]] = []
    layout = payload.get("layout", [])
    if not isinstance(layout, list):
        raise TypeError(f"Expected layout list: {annotation_path}")
    for raw_instance in layout:
        if not isinstance(raw_instance, dict):
            dropped_counts["non_object_instance"] += 1
            continue
        label = str(raw_instance.get("type", "")).strip().lower()
        if label == "arrow":
            label = "line"
        if label not in ALLOWED_LABELS:
            dropped_counts[f"unsupported_label:{label or '<empty>'}"] += 1
            continue
        bbox = _normalize_bbox(
            raw_instance,
            image_width=image_width,
            image_height=image_height,
        )
        if bbox is None:
            dropped_counts["invalid_bbox"] += 1
            continue
        if _is_background_shape(
            label,
            bbox,
            image_width=image_width,
            image_height=image_height,
        ):
            dropped_counts["background_shape"] += 1
            continue
        instances.append({"label": label, "bbox": bbox})
        label_counts[label] += 1

    structured_dir = config.output_root / "structured"
    image_reference = os.path.relpath(
        generated_image_path.resolve(),
        start=structured_dir.resolve(),
    )
    row = {
        "sample_id": sample_id,
        "image_path": image_reference,
        "image_width": image_width,
        "image_height": image_height,
        "instances": instances,
        "source_type": "synthetic_gt_standard",
        "source_sample_id": stem,
        "extra": {
            "task": "grounding_layout_sync",
            "split": "train",
            "target_labels": list(ALLOWED_LABELS),
            "view_type": "full_image",
            "source_json": os.path.relpath(annotation_path.resolve(), start=Path.cwd()),
            "source_image": os.path.relpath(image_path.resolve(), start=Path.cwd()),
            "source_dataset": config.source_name,
            "coordinate_source": "gt_standard",
            "filtered_background_shape_policy": {
                "drop_shape_area_gte": BACKGROUND_SHAPE_AREA_RATIO,
            },
            "pixel_augmentation": augmentation,
        },
    }
    augmentation_counts: Counter[str] = Counter(
        {
            f"profile:{augmentation['profile']}": 1,
            f"severity:{augmentation['severity']}": 1,
            f"stack_depth:{len(augmentation['operations'])}": 1,
        }
    )
    augmentation_counts.update(
        f"operation:{operation['name']}" for operation in augmentation["operations"]
    )
    return BuildResult(
        row,
        dict(label_counts),
        dict(dropped_counts),
        dict(augmentation_counts),
    )


def _write_readme(
    path: Path,
    *,
    config: BuildConfig,
    row_count: int,
    empty_count: int,
    label_counts: Counter[str],
    dropped_counts: Counter[str],
    augmentation_counts: Counter[str],
    val_source_count: int,
    split_file: Path,
) -> None:
    dataset_path = _display_path(config.dataset_root)
    split_path = _display_path(split_file)
    content = f"""# grounding_layout_sync

- Source dataset: `{dataset_path}`
- Coordinate source: `gt_standard`; `size` is verified against each rendered PNG.
- Split source: `{split_path}`
- Split: train only; `{val_source_count}` ids from `val.txt` are excluded.
- View policy: every train row is a full-image `{SYNTHETIC_REALISM_PROFILE}` view with one to three
  non-empty pixel operations; no clean source view is retained.
- Pixel operations: deterministic resample round-trip, Gaussian blur, Gaussian noise, and JPEG
  compression. Image dimensions and bbox coordinates remain unchanged.
- Image policy: augmented RGB pixels are materialized as task-local PNG files; structured/SFT rows
  never reference the clean source PNG directly.
- Prompt policy: selected explicitly during SFT conversion/training; structured rows do not embed
  prompt text or silently choose a runtime pool.
- Labels: `shape`, `icon`, `image`, and `line`; source `arrow` is normalized to `line`.
- Background shape policy: drop synthetic shape instances covering at least 90% of the canvas.

## Counts

- Rows: `{row_count}`
- Empty rows: `{empty_count}`
- Instances: `{sum(label_counts.values())}`
- Label counts: `{dict(sorted(label_counts.items()))}`
- Dropped counts: `{dict(sorted(dropped_counts.items()))}`
- Pixel augmentation profiles: `{dict(sorted((key.split(":", 1)[1], value) for key, value in augmentation_counts.items() if key.startswith("profile:")))}`
- Pixel augmentation severities: `{dict(sorted((key.split(":", 1)[1], value) for key, value in augmentation_counts.items() if key.startswith("severity:")))}`
- Pixel augmentation stack depths: `{dict(sorted((key.split(":", 1)[1], value) for key, value in augmentation_counts.items() if key.startswith("stack_depth:")))}`
- Pixel augmentation operations: `{dict(sorted((key.split(":", 1)[1], value) for key, value in augmentation_counts.items() if key.startswith("operation:")))}`
"""
    path.write_text(content, encoding="utf-8")


def build_dataset(
    *,
    dataset_root: Path,
    output_root: Path,
    split_file: Path | None = None,
    workers: int = 8,
    chunksize: int = 8,
    seed: int = 42,
    png_compress_level: int = 1,
    max_samples: int | None = None,
    clean: bool = False,
) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    output_root = output_root.resolve()
    if (
        dataset_root == output_root
        or dataset_root in output_root.parents
        or output_root in dataset_root.parents
    ):
        raise ValueError(
            f"Input and output roots must be disjoint: {dataset_root} vs {output_root}"
        )
    if workers <= 0 or chunksize <= 0:
        raise ValueError("workers and chunksize must be positive")
    if not 0 <= png_compress_level <= 9:
        raise ValueError("png_compress_level must be between 0 and 9")
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive")
    split_file = (split_file or dataset_root / "train.txt").resolve()
    stems = _load_split(split_file)
    if max_samples is not None:
        stems = stems[: max(0, int(max_samples))]
    val_path = dataset_root / "val.txt"
    val_stems = set(_load_split(val_path)) if val_path.exists() else set()
    overlap = sorted(set(stems) & val_stems)
    if overlap:
        raise ValueError(f"Train/val source overlap: {overlap[:10]}")

    if clean and output_root.exists():
        shutil.rmtree(output_root)
    structured_dir = output_root / "structured"
    structured_dir.mkdir(parents=True, exist_ok=True)
    config = BuildConfig(
        dataset_root=dataset_root,
        output_root=output_root,
        source_name=dataset_root.name,
        seed=seed,
        png_compress_level=png_compress_level,
    )
    work_items = ((stem, config) for stem in stems)
    train_path = structured_dir / "train.jsonl"
    label_counts: Counter[str] = Counter()
    dropped_counts: Counter[str] = Counter()
    augmentation_counts: Counter[str] = Counter()
    empty_count = 0
    row_count = 0
    handle = _atomic_text_writer(train_path)
    temporary_path = Path(handle.name)
    try:
        if workers <= 1:
            results = map(_build_one, work_items)
        else:
            executor = ProcessPoolExecutor(max_workers=workers)
            results = executor.map(_build_one, work_items, chunksize=chunksize)
        try:
            for result in results:
                handle.write(
                    json.dumps(result.row, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
                row_count += 1
                empty_count += int(not result.row["instances"])
                label_counts.update(result.label_counts)
                dropped_counts.update(result.dropped_counts)
                augmentation_counts.update(result.augmentation_counts)
        finally:
            if workers > 1:
                executor.shutdown()
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        os.replace(temporary_path, train_path)
    except BaseException:
        handle.close()
        temporary_path.unlink(missing_ok=True)
        raise

    (structured_dir / "val.jsonl").write_text("", encoding="utf-8")
    _write_readme(
        output_root / "README.md",
        config=config,
        row_count=row_count,
        empty_count=empty_count,
        label_counts=label_counts,
        dropped_counts=dropped_counts,
        augmentation_counts=augmentation_counts,
        val_source_count=len(val_stems),
        split_file=split_file,
    )
    summary = {
        "rows": row_count,
        "empty_rows": empty_count,
        "label_counts": dict(label_counts),
        "dropped_counts": dict(dropped_counts),
        "excluded_val_sources": len(val_stems),
        "seed": seed,
        "pixel_profile": SYNTHETIC_REALISM_PROFILE,
        "pixel_augmentation_profiles": {
            key.split(":", 1)[1]: value
            for key, value in sorted(augmentation_counts.items())
            if key.startswith("profile:")
        },
        "pixel_augmentation_severities": {
            key.split(":", 1)[1]: value
            for key, value in sorted(augmentation_counts.items())
            if key.startswith("severity:")
        },
        "pixel_augmentation_stack_depths": {
            key.split(":", 1)[1]: value
            for key, value in sorted(augmentation_counts.items())
            if key.startswith("stack_depth:")
        },
        "pixel_augmentation_operations": {
            key.split(":", 1)[1]: value
            for key, value in sorted(augmentation_counts.items())
            if key.startswith("operation:")
        },
    }
    (output_root / "build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build fully augmented full-image synthetic grounding data from gt_standard."
    )
    parser.add_argument(
        "--dataset-root",
        default="data/regulated_layout_dataset_v9_20260802",
    )
    parser.add_argument("--output-root", default="data/grounding_layout_sync")
    parser.add_argument("--split-file")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunksize", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--png-compress-level", type=int, default=1)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    summary = build_dataset(
        dataset_root=Path(args.dataset_root),
        output_root=Path(args.output_root),
        split_file=Path(args.split_file) if args.split_file else None,
        workers=int(args.workers),
        chunksize=int(args.chunksize),
        seed=int(args.seed),
        png_compress_level=int(args.png_compress_level),
        max_samples=args.max_samples,
        clean=bool(args.clean),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
