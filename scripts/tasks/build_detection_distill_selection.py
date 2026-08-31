#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

from PIL import Image

from shaft.offline_kd.media_plan import (
    MEDIA_PLAN_FIELD,
    deterministic_detection_media_plan,
)
from shaft.prompting import load_prompt_template


SELECTION_VERSION = "banana-detection-distill-selection-v1"
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp"})


def _image_paths(root: Path) -> list[Path]:
    return sorted(
        path.resolve()
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _test_ids(split_root: Path) -> set[str]:
    output: set[str] = set()
    for path in sorted(split_root.glob("*.test.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        items = payload.get("items")
        if not isinstance(items, list):
            raise ValueError(f"Test split has no items list: {path}")
        for item in items:
            if not isinstance(item, dict) or not str(item.get("id", "")).strip():
                raise ValueError(f"Invalid test item in {path}")
            output.add(str(item["id"]).strip())
    return output


def discover_candidates(data_root: Path) -> tuple[list[tuple[str, Path]], list[Path]]:
    raw_root = data_root / "raw_data"
    raw_images = _image_paths(raw_root / "images")
    labeled_ids = {path.stem for path in (raw_root / "json").glob("*.json")}
    excluded_ids = labeled_ids | _test_ids(raw_root / "splits")
    candidates: list[tuple[str, Path]] = [
        ("raw_unlabeled", path) for path in raw_images if path.stem not in excluded_ids
    ]
    protected = [path for path in raw_images if path.stem in excluded_ids]

    paper_root = data_root / "paper"
    candidates.extend(("paper", path) for path in _image_paths(paper_root))

    chai_root = data_root / "ppt" / "chai_group"
    manifest_path = chai_root / "_source_manifest.tsv"
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    train_names = {
        str(row.get("filename", "")).strip()
        for row in rows
        if str(row.get("split", "")).strip().lower() == "train"
    }
    if "" in train_names:
        raise ValueError(f"Chai manifest contains an empty train filename: {manifest_path}")
    for name in sorted(train_names):
        path = (chai_root / name).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Chai train image is missing: {path}")
        candidates.append(("ppt_chai_train", path))

    candidates.extend(("ppt_wyk", path) for path in _image_paths(data_root / "ppt" / "wyk"))
    return candidates, protected


def _inspect(path: Path, *, content_hash: bool) -> tuple[int, int, str | None]:
    with Image.open(path) as image:
        width, height = image.size
        image.verify()
    digest: str | None = None
    if content_hash:
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                hasher.update(chunk)
        digest = hasher.hexdigest()
    return int(width), int(height), digest


def _parallel_inspect(
    paths: Iterable[Path], *, workers: int, content_hash: bool
) -> dict[Path, tuple[int, int, str | None]]:
    ordered = list(dict.fromkeys(paths))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        values = executor.map(
            lambda path: _inspect(path, content_hash=content_hash), ordered, chunksize=64
        )
        return dict(zip(ordered, values, strict=True))


def build_selection(
    *,
    data_root: Path,
    output_path: Path,
    prompt_path: Path,
    prompt_variant: str,
    seed: int,
    workers: int,
    content_dedupe: bool,
    max_rows: int | None = None,
) -> dict[str, Any]:
    data_root = data_root.resolve()
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(f"Selection output already exists: {output_path}")
    prompt = load_prompt_template(prompt_path, variant_id=prompt_variant)
    candidates, protected = discover_candidates(data_root)
    if max_rows is not None:
        candidates = candidates[: int(max_rows)]
    inspected = _parallel_inspect(
        [path for _, path in candidates] + (protected if content_dedupe else []),
        workers=workers,
        content_hash=content_dedupe,
    )
    protected_hashes = {
        inspected[path][2] for path in protected if inspected[path][2] is not None
    }
    seen_hashes: set[str] = set()
    exclusions: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    for source_pool, path in candidates:
        width, height, digest = inspected[path]
        if digest is not None and digest in protected_hashes:
            exclusions["protected_content_match"] += 1
            continue
        if digest is not None and digest in seen_hashes:
            exclusions["candidate_content_duplicate"] += 1
            continue
        if digest is not None:
            seen_hashes.add(digest)
        relative_path = path.relative_to(data_root).as_posix()
        sample_id = f"{source_pool}:{relative_path}"
        media_plan = deterministic_detection_media_plan(
            sample_id=sample_id,
            width=width,
            height=height,
            seed=seed,
        )
        rows.append(
            {
                "image_path": str(path),
                "sample_id": sample_id,
                "system_prompt": prompt.system_prompt,
                "user_prompt": prompt.user_prompt,
                "target_text": "",
                "task": "detection",
                "source_pool": source_pool,
                "source_relative_path": relative_path,
                "prompt_pool_id": prompt.metadata.get("prompt_pool_id"),
                "prompt_variant_id": prompt.variant_id,
                "prompt_version": prompt.version,
                MEDIA_PLAN_FIELD: media_plan.to_dict(),
            }
        )
    counts = Counter(row["source_pool"] for row in rows)
    bucket_counts = Counter(row[MEDIA_PLAN_FIELD]["bucket"] for row in rows)
    manifest = {
        "version": SELECTION_VERSION,
        "data_root": str(data_root),
        "selection_path": str(output_path),
        "seed": int(seed),
        "task": "detection",
        "prompt": {
            "path": str(prompt_path.resolve()),
            "pool_id": prompt.metadata.get("prompt_pool_id"),
            "variant_id": prompt.variant_id,
            "version": prompt.version,
        },
        "content_dedupe": bool(content_dedupe),
        "candidate_count_before_dedupe": len(candidates),
        "selected_count": len(rows),
        "source_counts": dict(sorted(counts.items())),
        "pixel_bucket_counts": dict(sorted(bucket_counts.items())),
        "exclusions": dict(sorted(exclusions.items())),
        "images_copied": 0,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output_path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze prompt-only unlabeled images for Banana detection distillation."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--prompt-pool", default="configs/prompts/pools/grounding_layout.v5.8.yaml"
    )
    parser.add_argument("--prompt-variant", default="detailed")
    parser.add_argument("--seed", type=int, default=465)
    parser.add_argument("--workers", type=int, default=50)
    parser.add_argument("--content-dedupe", action="store_true")
    parser.add_argument("--max-rows", type=int)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.workers <= 0:
        raise ValueError("--workers must be > 0")
    if args.max_rows is not None and args.max_rows <= 0:
        raise ValueError("--max-rows must be > 0")
    manifest = build_selection(
        data_root=Path(args.data_root),
        output_path=Path(args.output),
        prompt_path=Path(args.prompt_pool),
        prompt_variant=args.prompt_variant,
        seed=args.seed,
        workers=args.workers,
        content_dedupe=args.content_dedupe,
        max_rows=args.max_rows,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
