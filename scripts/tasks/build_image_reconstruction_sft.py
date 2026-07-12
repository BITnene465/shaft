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

from shaft.prompting import load_prompt_pool


ALLOWED_IMAGE_TYPES = frozenset(
    {
        "chart",
        "diagram",
        "document",
        "illustration",
        "infographic",
        "map",
        "medical",
        "microscopy",
        "other",
        "photo",
        "rendering",
        "screenshot",
        "table",
    }
)
MULTI_SCALE_BUCKETS = (
    ("tight", 0.70, 0.08, 0.15),
    ("medium", 0.25, 0.15, 0.25),
    ("context", 0.05, 0.25, 0.40),
)


@dataclass(frozen=True)
class PromptInfo:
    prompt_id: str
    system_prompt: str
    user_prompt: str
    output_schema: str | None


@dataclass(frozen=True)
class Candidate:
    stem: str
    instance_index: int
    image_path: Path
    image_width: int
    image_height: int
    bbox: tuple[float, float, float, float]
    image_type: str


@dataclass(frozen=True)
class SelectedCandidate:
    candidate: Candidate
    view_index: int


@dataclass(frozen=True)
class BuildConfig:
    raw_root: Path
    output_root: Path
    prompt_info: tuple[PromptInfo, ...]
    seed: int
    min_crop_size: int
    max_aspect_ratio: float
    max_image_edge: int


@dataclass(frozen=True)
class WorkerResult:
    structured_rows: tuple[str, ...]
    sft_rows: tuple[str, ...]
    counts: dict[str, int]


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent) as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
        temp_path = Path(f.name)
    os.replace(temp_path, path)


def _stable_score(*parts: Any) -> int:
    payload = ":".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _read_excluded_ids(paths: list[Path]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        items = payload.get("items")
        if not isinstance(items, list):
            raise ValueError(f"Invalid test manifest: {path}")
        for item in items:
            if not isinstance(item, dict):
                continue
            sample_id = str(item.get("id") or Path(str(item.get("image_path") or "")).stem)
            if sample_id:
                excluded.add(sample_id)
    return excluded


def _image_index(images_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in images_dir.iterdir():
        if not path.is_file():
            continue
        if path.stem in index:
            raise ValueError(f"Duplicate image stem: {path.stem}")
        index[path.stem] = path
    return index


def _clean_bbox(
    value: Any,
    *,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float] | None:
    if not isinstance(value, list | tuple) or len(value) != 4:
        return None
    if not all(
        isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(item)
        for item in value
    ):
        return None
    x1, y1, x2, y2 = [float(item) for item in value]
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    x1 = min(max(x1, 0.0), float(image_width))
    y1 = min(max(y1, 0.0), float(image_height))
    x2 = min(max(x2, 0.0), float(image_width))
    y2 = min(max(y2, 0.0), float(image_height))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _collect_candidates(
    json_dir: Path,
    *,
    image_index: dict[str, Path],
    excluded_ids: set[str],
) -> tuple[list[Candidate], Counter[str]]:
    candidates: list[Candidate] = []
    counters: Counter[str] = Counter()
    for json_path in sorted(json_dir.glob("*.json")):
        if json_path.stem in excluded_ids:
            counters["excluded_test_json"] += 1
            continue
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        image_width = int(payload.get("image_width") or 0)
        image_height = int(payload.get("image_height") or 0)
        image_path = image_index.get(json_path.stem)
        if image_width <= 0 or image_height <= 0 or image_path is None:
            counters["invalid_source"] += 1
            continue
        for instance_index, instance in enumerate(payload.get("instances", [])):
            if not isinstance(instance, dict) or instance.get("label") != "image":
                continue
            parameters = (instance.get("extra") or {}).get("parameters")
            image_type = parameters.get("image_type") if isinstance(parameters, dict) else None
            if image_type not in ALLOWED_IMAGE_TYPES:
                counters["invalid_image_type"] += 1
                continue
            bbox = _clean_bbox(
                instance.get("bbox"),
                image_width=image_width,
                image_height=image_height,
            )
            if bbox is None:
                counters["invalid_bbox"] += 1
                continue
            candidates.append(
                Candidate(
                    stem=json_path.stem,
                    instance_index=instance_index,
                    image_path=image_path,
                    image_width=image_width,
                    image_height=image_height,
                    bbox=bbox,
                    image_type=image_type,
                )
            )
            counters[f"available_{image_type}"] += 1
    return candidates, counters


def _select_candidates(
    candidates: list[Candidate],
    *,
    minimum_per_class: int,
    maximum_per_class: int,
    seed: int,
) -> tuple[list[SelectedCandidate], Counter[str]]:
    by_type: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_type[candidate.image_type].append(candidate)
    selected: list[SelectedCandidate] = []
    counters: Counter[str] = Counter()
    for image_type in sorted(ALLOWED_IMAGE_TYPES):
        values = by_type.get(image_type, [])
        if not values:
            raise ValueError(f"No candidates for image_type={image_type}")
        values.sort(
            key=lambda candidate: _stable_score(
                seed,
                "sample",
                image_type,
                candidate.stem,
                candidate.instance_index,
            )
        )
        target = min(max(len(values), minimum_per_class), maximum_per_class)
        if len(values) >= target:
            chosen = [SelectedCandidate(candidate, 0) for candidate in values[:target]]
        else:
            chosen = [SelectedCandidate(candidate, 0) for candidate in values]
            view_counts: Counter[tuple[str, int]] = Counter()
            for offset in range(target - len(values)):
                candidate = values[offset % len(values)]
                key = (candidate.stem, candidate.instance_index)
                view_counts[key] += 1
                chosen.append(SelectedCandidate(candidate, view_counts[key]))
        selected.extend(chosen)
        counters[f"selected_{image_type}"] = len(chosen)
    selected.sort(
        key=lambda item: (
            item.candidate.stem,
            item.candidate.instance_index,
            item.view_index,
        )
    )
    counters["selected_total"] = len(selected)
    return selected, counters


def _load_prompts(path: Path) -> tuple[PromptInfo, ...]:
    return tuple(
        PromptInfo(
            prompt_id=prompt.prompt_id,
            system_prompt=prompt.system_prompt,
            user_prompt=prompt.user_prompt,
            output_schema=(
                str(prompt.metadata["output_schema"])
                if prompt.metadata.get("output_schema") is not None
                else None
            ),
        )
        for prompt in load_prompt_pool(path)
    )


def _padding_policy(
    config: BuildConfig,
    *,
    stem: str,
    instance_index: int,
    view_index: int,
) -> tuple[str, float]:
    rng = random.Random(
        f"{config.seed}:scale:image_reconstruction:{stem}:{instance_index}:{view_index}"
    )
    draw = rng.random()
    cumulative = 0.0
    for name, probability, minimum, maximum in MULTI_SCALE_BUCKETS:
        cumulative += probability
        if draw < cumulative:
            return name, rng.uniform(minimum, maximum)
    name, _, minimum, maximum = MULTI_SCALE_BUCKETS[-1]
    return name, rng.uniform(minimum, maximum)


def _bounded_interval(center: float, length: int, *, limit: int) -> tuple[int, int]:
    if length >= limit:
        return 0, limit
    start = int(math.floor(center - length / 2.0))
    end = start + length
    if start < 0:
        return 0, length
    if end > limit:
        return limit - length, limit
    return start, end


def _crop_box(
    bbox: tuple[float, float, float, float],
    *,
    image_width: int,
    image_height: int,
    padding_ratio: float,
    max_aspect_ratio: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    width, height = x2 - x1, y2 - y1
    left = max(0, int(math.floor(x1 - width * padding_ratio)))
    top = max(0, int(math.floor(y1 - height * padding_ratio)))
    right = min(image_width, int(math.ceil(x2 + width * padding_ratio)))
    bottom = min(image_height, int(math.ceil(y2 + height * padding_ratio)))
    crop_width, crop_height = right - left, bottom - top
    if crop_width > 0 and crop_height > 0:
        if crop_width / crop_height > max_aspect_ratio:
            target_height = min(
                image_height,
                max(crop_height, int(math.ceil(crop_width / max_aspect_ratio))),
            )
            top, bottom = _bounded_interval(
                (top + bottom) / 2.0,
                target_height,
                limit=image_height,
            )
        elif crop_height / crop_width > max_aspect_ratio:
            target_width = min(
                image_width,
                max(crop_width, int(math.ceil(crop_height / max_aspect_ratio))),
            )
            left, right = _bounded_interval(
                (left + right) / 2.0,
                target_width,
                limit=image_width,
            )
    return left, top, right, bottom


def _prompt_for_sample(
    prompts: tuple[PromptInfo, ...],
    *,
    sample_id: str,
    seed: int,
) -> PromptInfo:
    if not prompts:
        raise ValueError("Image reconstruction prompt pool is empty")
    index = _stable_score(seed, "prompt", sample_id) % len(prompts)
    return prompts[index]


def _sample_id(item: SelectedCandidate) -> str:
    base = f"{item.candidate.stem}__image_{item.candidate.instance_index:04d}"
    return base if item.view_index == 0 else f"{base}__view_{item.view_index:02d}"


def _shard(stem: str) -> str:
    return hashlib.blake2b(stem.encode(), digest_size=1).hexdigest()


def _scale_bbox(
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


def _build_for_stem(
    args: tuple[str, tuple[SelectedCandidate, ...], BuildConfig],
) -> WorkerResult:
    stem, items, config = args
    counts: Counter[str] = Counter()
    structured_rows: list[str] = []
    sft_rows: list[str] = []
    source_path = items[0].candidate.image_path
    try:
        source_image = Image.open(source_path).convert("RGB")
    except Exception:
        counts["image_open_error"] += len(items)
        return WorkerResult(tuple(), tuple(), dict(counts))
    try:
        if source_image.size != (
            items[0].candidate.image_width,
            items[0].candidate.image_height,
        ):
            counts["image_size_mismatch"] += len(items)
            return WorkerResult(tuple(), tuple(), dict(counts))
        for item in items:
            candidate = item.candidate
            scale_bucket, padding_ratio = _padding_policy(
                config,
                stem=stem,
                instance_index=candidate.instance_index,
                view_index=item.view_index,
            )
            crop_box = _crop_box(
                candidate.bbox,
                image_width=candidate.image_width,
                image_height=candidate.image_height,
                padding_ratio=padding_ratio,
                max_aspect_ratio=config.max_aspect_ratio,
            )
            left, top, right, bottom = crop_box
            crop_width, crop_height = right - left, bottom - top
            if crop_width < config.min_crop_size or crop_height < config.min_crop_size:
                counts["crop_too_small"] += 1
                continue
            crop_image = source_image.crop(crop_box)
            output_width, output_height = crop_width, crop_height
            resized = False
            if max(crop_width, crop_height) > config.max_image_edge:
                crop_image.thumbnail(
                    (config.max_image_edge, config.max_image_edge),
                    Image.Resampling.LANCZOS,
                )
                output_width, output_height = crop_image.size
                resized = True
            sample_id = _sample_id(item)
            shard = _shard(stem)
            filename = f"{sample_id}.png"
            image_relative = f"../images/train/{shard}/{filename}"
            output_path = (
                config.output_root / "image_reconstruction" / "images/train" / shard / filename
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            crop_image.save(output_path, compress_level=1)
            crop_image.close()
            local_bbox: list[int | float] = [
                round(candidate.bbox[0] - left, 3),
                round(candidate.bbox[1] - top, 3),
                round(candidate.bbox[2] - left, 3),
                round(candidate.bbox[3] - top, 3),
            ]
            if resized:
                local_bbox = _scale_bbox(
                    local_bbox,
                    scale_x=output_width / crop_width,
                    scale_y=output_height / crop_height,
                )
            parameters = {"image_type": candidate.image_type}
            augmentation: dict[str, Any] = {
                "name": "bbox_padding_crop",
                "padding_ratio": padding_ratio,
                "scale_bucket": scale_bucket,
                "max_aspect_ratio": config.max_aspect_ratio,
            }
            if resized:
                augmentation["max_edge_resize"] = {
                    "source_size": [crop_width, crop_height],
                    "output_size": [output_width, output_height],
                    "max_image_edge": config.max_image_edge,
                }
            structured_extra = {
                "task": "image_reconstruction",
                "split": "train",
                "view_type": "image_reconstruction_crop",
                "source_dataset": "raw",
                "source_json": f"json/{stem}.json",
                "source_image": f"images/{source_path.name}",
                "source_instance_index": candidate.instance_index,
                "source_bbox": list(candidate.bbox),
                "crop_box": list(crop_box),
                "view_index": item.view_index,
                "augmentation": augmentation,
            }
            structured = {
                "sample_id": sample_id,
                "image_path": image_relative,
                "image_width": output_width,
                "image_height": output_height,
                "instances": [
                    {
                        "label": "image",
                        "bbox": local_bbox,
                        "parameters": parameters,
                    }
                ],
                "extra": structured_extra,
            }
            prompt = _prompt_for_sample(
                config.prompt_info,
                sample_id=sample_id,
                seed=config.seed,
            )
            target = {"type": "image", "parameters": parameters}
            sft_extra: dict[str, Any] = {
                "prompt_id": prompt.prompt_id,
                "source_sample_id": stem,
                "source_type": "reviewed_raw_image_type",
                "image_width": output_width,
                "image_height": output_height,
                "structured_extra": structured_extra,
            }
            if prompt.output_schema:
                sft_extra["output_schema"] = prompt.output_schema
            sft = {
                "image_path": image_relative,
                "sample_id": sample_id,
                "dataset_name": "image_reconstruction",
                "system_prompt": prompt.system_prompt,
                "user_prompt": prompt.user_prompt,
                "target_text": _json_dumps(target),
                "extra": sft_extra,
            }
            structured_rows.append(_json_dumps(structured))
            sft_rows.append(_json_dumps(sft))
            counts["rows"] += 1
            counts[f"image_type_{candidate.image_type}"] += 1
            counts[f"scale_{scale_bucket}"] += 1
            counts["max_edge_resized"] += int(resized)
    finally:
        source_image.close()
    return WorkerResult(tuple(structured_rows), tuple(sft_rows), dict(counts))


def _prepare_output(output_root: Path, *, clean: bool) -> None:
    task_root = output_root / "image_reconstruction"
    if clean and task_root.exists():
        shutil.rmtree(task_root)
    for relative in ("structured", "sft", "images/train"):
        (task_root / relative).mkdir(parents=True, exist_ok=True)
    _atomic_write_text(task_root / "structured/val.jsonl", "")
    _atomic_write_text(task_root / "sft/val.jsonl", "")


def build(args: argparse.Namespace) -> Counter[str]:
    raw_root = Path(args.raw_root)
    output_root = Path(args.output_root)
    prompt_path = Path(args.prompt_pool)
    excluded_ids = _read_excluded_ids([Path(path) for path in args.exclude_manifests])
    candidates, inventory_counts = _collect_candidates(
        raw_root / "json",
        image_index=_image_index(raw_root / "images"),
        excluded_ids=excluded_ids,
    )
    selected, selection_counts = _select_candidates(
        candidates,
        minimum_per_class=int(args.minimum_per_class),
        maximum_per_class=int(args.maximum_per_class),
        seed=int(args.seed),
    )
    _prepare_output(output_root, clean=bool(args.clean))
    config = BuildConfig(
        raw_root=raw_root,
        output_root=output_root,
        prompt_info=_load_prompts(prompt_path),
        seed=int(args.seed),
        min_crop_size=int(args.min_crop_size),
        max_aspect_ratio=float(args.max_aspect_ratio),
        max_image_edge=int(args.max_image_edge),
    )
    by_stem: dict[str, list[SelectedCandidate]] = defaultdict(list)
    for item in selected:
        by_stem[item.candidate.stem].append(item)
    worker_args = [(stem, tuple(items), config) for stem, items in sorted(by_stem.items())]
    task_root = output_root / "image_reconstruction"
    counters = Counter(inventory_counts)
    counters.update(selection_counts)
    with (
        (task_root / "structured/train.jsonl").open("w", encoding="utf-8") as structured_file,
        (task_root / "sft/train.jsonl").open("w", encoding="utf-8") as sft_file,
        ProcessPoolExecutor(max_workers=int(args.workers)) as executor,
    ):
        for result in executor.map(_build_for_stem, worker_args, chunksize=int(args.chunksize)):
            counters.update(result.counts)
            for row in result.structured_rows:
                structured_file.write(row + "\n")
            for row in result.sft_rows:
                sft_file.write(row + "\n")

    distribution_lines = ["| image_type | Available | Selected |", "|---|---:|---:|"]
    distribution_lines.extend(
        f"| {image_type} | {counters[f'available_{image_type}']} | "
        f"{counters[f'image_type_{image_type}']} |"
        for image_type in sorted(ALLOWED_IMAGE_TYPES)
    )
    distribution_table = "\n".join(distribution_lines)
    readme = f"""# image_reconstruction SFT/structured dataset

Generated from reviewed real-image `image_type` annotations.

- Source root: `{raw_root}`
- Prompt pool: `{prompt_path}`
- Target: compact JSON `{{"type":"image","parameters":{{"image_type":"..."}}}}`
- Sampling band: `{args.minimum_per_class}` to `{args.maximum_per_class}` rows per class
- Scale policy: 70% tight, 25% medium, 5% context padding
- Test-manifest images: excluded from train
- Train rows: {counters["rows"]}
- Validation rows: 0
- Max derived crop edge: {args.max_image_edge}

## Distribution

{distribution_table}

Only `parameters.image_type` is model-facing. Raw annotation and review metadata remain in the raw
source and are not copied into `target_text`.
"""
    _atomic_write_text(task_root / "README.md", readme)
    return counters


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build stratified real image reconstruction SFT data."
    )
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--output-root", default="data")
    parser.add_argument(
        "--prompt-pool",
        default="configs/prompts/pools/image_reconstruction.v5.0.yaml",
    )
    parser.add_argument(
        "--exclude-manifests",
        nargs="+",
        default=[
            "data/raw/splits/main.test.json",
            "data/raw/splits/inpainting.test.json",
            "data/raw/splits/vlm.test.json",
        ],
    )
    parser.add_argument("--minimum-per-class", type=int, default=1000)
    parser.add_argument("--maximum-per-class", type=int, default=4000)
    parser.add_argument("--min-crop-size", type=int, default=4)
    parser.add_argument("--max-aspect-ratio", type=float, default=60.0)
    parser.add_argument("--max-image-edge", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunksize", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    if args.workers <= 0 or args.chunksize <= 0:
        parser.error("workers and chunksize must be positive")
    if args.minimum_per_class <= 0 or args.maximum_per_class < args.minimum_per_class:
        parser.error("class sampling band is invalid")
    counters = build(args)
    print(json.dumps(dict(sorted(counters.items())), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
