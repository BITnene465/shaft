#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from shaft.prompting import load_prompt_pool


@dataclass(frozen=True)
class PromptInfo:
    prompt_id: str
    system_prompt: str
    user_prompt: str
    output_schema: str | None


@dataclass(frozen=True)
class SourceRecord:
    sample_id: str
    image_name: str
    background: bool


@dataclass(frozen=True)
class BuiltRow:
    structured: str
    sft: str
    background: bool
    link_method: str


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


def _load_source_records(path: Path) -> list[SourceRecord]:
    records: list[SourceRecord] = []
    sample_ids: set[str] = set()
    image_names: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        sample_id = str(payload.get("id") or "").strip()
        image_name = Path(str(payload.get("image_path") or "")).name
        background = payload.get("background")
        level = payload.get("background_level")
        if not sample_id or not image_name:
            raise ValueError(f"Missing id/image_path at {path}:{line_number}")
        if not isinstance(background, bool):
            raise ValueError(f"Invalid background at {path}:{line_number}")
        if level not in {0, 1, 4} or background != (level == 4):
            raise ValueError(f"Inconsistent reviewed level at {path}:{line_number}")
        if sample_id in sample_ids or image_name in image_names:
            raise ValueError(f"Duplicate id/image_path at {path}:{line_number}")
        sample_ids.add(sample_id)
        image_names.add(image_name)
        records.append(SourceRecord(sample_id, image_name, background))
    return records


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


def _load_prompts(path: Path) -> tuple[PromptInfo, ...]:
    prompts = load_prompt_pool(path)
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
        for prompt in prompts
    )


def _prompt_for_sample(prompts: tuple[PromptInfo, ...], *, sample_id: str, seed: int) -> PromptInfo:
    if not prompts:
        raise ValueError("Background prompt pool is empty")
    digest = hashlib.blake2b(f"{seed}:prompt:{sample_id}".encode(), digest_size=8).digest()
    return prompts[int.from_bytes(digest, "big") % len(prompts)]


def _shard(sample_id: str) -> str:
    return hashlib.blake2b(sample_id.encode(), digest_size=1).hexdigest()


def _link_or_copy(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def _materialize_image(source: Path, target: Path, *, max_image_edge: int) -> tuple[str, list[int]]:
    with Image.open(source) as image:
        source_width, source_height = image.size
        if max(source_width, source_height) <= max_image_edge:
            return _link_or_copy(source, target), [source_width, source_height]
        image_format = image.format or "PNG"
        image.load()
        image.thumbnail((max_image_edge, max_image_edge), Image.Resampling.LANCZOS)
        if image_format.upper() in {"JPEG", "JPG"} and image.mode != "RGB":
            image = image.convert("RGB")
        target.parent.mkdir(parents=True, exist_ok=True)
        save_kwargs: dict[str, Any] = {"format": image_format}
        if image_format.upper() in {"JPEG", "JPG", "WEBP"}:
            save_kwargs["quality"] = 95
        elif image_format.upper() == "PNG":
            save_kwargs["compress_level"] = 1
        image.save(target, **save_kwargs)
        return "resized_copy", [source_width, source_height]


def _build_one(
    args: tuple[SourceRecord, Path, Path, Path, tuple[PromptInfo, ...], int, int],
) -> BuiltRow:
    record, images_dir, output_root, source_annotations, prompts, seed, max_image_edge = args
    source_image = images_dir / record.image_name
    if not source_image.is_file():
        raise FileNotFoundError(source_image)
    shard = _shard(record.sample_id)
    image_relative = f"../images/train/{shard}/{record.image_name}"
    output_image = output_root / "background" / "images" / "train" / shard / record.image_name
    link_method, source_size = _materialize_image(
        source_image,
        output_image,
        max_image_edge=max_image_edge,
    )
    with Image.open(output_image) as image:
        image_width, image_height = image.size
    prompt = _prompt_for_sample(prompts, sample_id=record.sample_id, seed=seed)
    structured_extra = {
        "task": "background",
        "split": "train",
        "view_type": "full_image",
        "source_dataset": "raw",
        "source_annotation": str(source_annotations),
        "source_image": f"images/{record.image_name}",
    }
    if link_method == "resized_copy":
        structured_extra["augmentation"] = {
            "name": "max_edge_resize",
            "source_size": source_size,
            "output_size": [image_width, image_height],
            "max_image_edge": max_image_edge,
        }
    structured = {
        "sample_id": record.sample_id,
        "image_path": image_relative,
        "image_width": image_width,
        "image_height": image_height,
        "background": record.background,
        "extra": structured_extra,
    }
    sft_extra: dict[str, Any] = {
        "prompt_id": prompt.prompt_id,
        "source_sample_id": record.sample_id,
        "source_type": "reviewed_background_annotation",
        "image_width": image_width,
        "image_height": image_height,
        "structured_extra": structured_extra,
    }
    if prompt.output_schema:
        sft_extra["output_schema"] = prompt.output_schema
    sft = {
        "image_path": image_relative,
        "sample_id": record.sample_id,
        "dataset_name": "background",
        "system_prompt": prompt.system_prompt,
        "user_prompt": prompt.user_prompt,
        "target_text": _json_dumps({"background": record.background}),
        "extra": sft_extra,
    }
    return BuiltRow(_json_dumps(structured), _json_dumps(sft), record.background, link_method)


def _prepare_output(output_root: Path, *, clean: bool) -> None:
    task_root = output_root / "background"
    if clean and task_root.exists():
        shutil.rmtree(task_root)
    for relative in ("structured", "sft", "images/train"):
        (task_root / relative).mkdir(parents=True, exist_ok=True)
    _atomic_write_text(task_root / "structured" / "val.jsonl", "")
    _atomic_write_text(task_root / "sft" / "val.jsonl", "")


def build(args: argparse.Namespace) -> Counter[str]:
    raw_root = Path(args.raw_root)
    output_root = Path(args.output_root)
    source_annotations = Path(args.annotations)
    prompt_path = Path(args.prompt_pool)
    records = _load_source_records(source_annotations)
    excluded_paths = [Path(path) for path in args.exclude_manifests]
    excluded_ids = _read_excluded_ids(excluded_paths)
    train_records = [record for record in records if record.sample_id not in excluded_ids]
    prompts = _load_prompts(prompt_path)
    _prepare_output(output_root, clean=bool(args.clean))

    task_root = output_root / "background"
    structured_path = task_root / "structured" / "train.jsonl"
    sft_path = task_root / "sft" / "train.jsonl"
    worker_args = [
        (
            record,
            raw_root / "images",
            output_root,
            source_annotations,
            prompts,
            int(args.seed),
            int(args.max_image_edge),
        )
        for record in train_records
    ]
    counters: Counter[str] = Counter(
        source_rows=len(records),
        excluded_test_rows=sum(record.sample_id in excluded_ids for record in records),
    )
    with (
        structured_path.open("w", encoding="utf-8") as structured_file,
        sft_path.open("w", encoding="utf-8") as sft_file,
    ):
        with ThreadPoolExecutor(max_workers=int(args.workers)) as executor:
            for row in executor.map(_build_one, worker_args):
                structured_file.write(row.structured + "\n")
                sft_file.write(row.sft + "\n")
                counters["train_rows"] += 1
                counters[f"background_{str(row.background).lower()}"] += 1
                counters[f"image_{row.link_method}"] += 1

    readme = f"""# background SFT/structured dataset

Generated from the reviewed full-image background annotation.

- Source annotations: `{source_annotations}`
- Source images: `{raw_root / "images"}`
- Prompt pool: `{prompt_path}`
- Task: full-image `background` boolean classification
- Target: compact JSON `{{"background":true|false}}`
- Train rows: {counters["train_rows"]}
- Background true: {counters["background_true"]}
- Background false: {counters["background_false"]}
- Excluded test rows: {counters["excluded_test_rows"]}
- Validation rows: 0
- Maximum derived image edge: {args.max_image_edge}
- Image materialization: hardlink when supported, copy fallback
- Oversized images resized: {counters["image_resized_copy"]}

`background_level`, reasons, source-model fields, and review audit fields remain only in the raw
annotation source and are not copied into model-facing targets.
"""
    _atomic_write_text(task_root / "README.md", readme)
    return counters


def main() -> None:
    parser = argparse.ArgumentParser(description="Build reviewed full-image background SFT data.")
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument(
        "--annotations",
        default="data/raw/background_annotations_opus48_reviewed_20260710.jsonl",
    )
    parser.add_argument("--output-root", default="data")
    parser.add_argument("--prompt-pool", default="configs/prompts/pools/background.v5.0.yaml")
    parser.add_argument(
        "--exclude-manifests",
        nargs="+",
        default=[
            "data/raw/splits/main.test.json",
            "data/raw/splits/inpainting.test.json",
            "data/raw/splits/vlm.test.json",
        ],
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-image-edge", type=int, default=4096)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.max_image_edge <= 0:
        parser.error("--max-image-edge must be positive")
    counters = build(args)
    print(json.dumps(dict(sorted(counters.items())), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
