#!/usr/bin/env python3
"""Freeze the Banana v5.9 grounding-only source increment.

The snapshot reuses the frozen v5.8 grounding train identities, adds quality-gated
v5.9 annotations, and excludes real_v1/real_v2 strictly by image stem.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable
import uuid

from PIL import Image, ImageOps


IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp"})
TARGET_TYPES = frozenset({"shape", "icon", "image", "line", "arrow"})


@dataclass(frozen=True)
class SourceRecord:
    sample_id: str
    json_path: Path
    image_path: Path
    width: int
    height: int
    origin: str


@dataclass(frozen=True)
class MediaJob:
    source: Path
    target: Path
    width: int
    height: int


def _json_dump(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_split(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    values: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        values.append(Path(value).stem)
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate identities in split: {path}")
    return values


def _image_index(root: Path, *, required: bool = True) -> dict[str, Path]:
    if not root.is_dir():
        if required:
            raise FileNotFoundError(root)
        return {}
    index: dict[str, Path] = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        previous = index.get(path.stem)
        if previous is not None:
            raise ValueError(f"Ambiguous image id {path.stem!r}: {previous}, {path}")
        index[path.stem] = path
    return index


def _declared_size(payload: dict[str, Any], *, source: Path) -> tuple[int, int]:
    size = payload.get("size")
    if (
        not isinstance(size, list)
        or len(size) != 2
        or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in size)
    ):
        raise ValueError(f"Invalid compact size: {source}: {size!r}")
    width, height = int(size[0]), int(size[1])
    if width <= 0 or height <= 0 or width != size[0] or height != size[1]:
        raise ValueError(f"Invalid compact size: {source}: {size!r}")
    return width, height


def _load_record(
    sample_id: str,
    *,
    json_path: Path,
    image_path: Path,
    origin: str,
) -> tuple[SourceRecord, dict[str, Any]]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Annotation must be an object: {json_path}")
    width, height = _declared_size(payload, source=json_path)
    layout = payload.get("layout")
    if not isinstance(layout, list):
        raise ValueError(f"Annotation layout must be a list: {json_path}")
    return SourceRecord(sample_id, json_path, image_path, width, height, origin), payload


def _quality_reason(payload: dict[str, Any], *, width: int, height: int) -> str | None:
    layout = payload["layout"]
    if len(layout) <= 2:
        return "low_annotation_count"
    serialized: set[str] = set()
    for item in layout:
        if not isinstance(item, dict):
            return "invalid_layout_instance"
        token = json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if token in serialized:
            return "duplicate_full_instance"
        serialized.add(token)
        item_type = str(item.get("type") or item.get("label") or "").strip().lower()
        if item_type not in TARGET_TYPES:
            continue
        bbox = item.get("bbox")
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in bbox)
        ):
            return "invalid_target_bbox"
        x1, y1, x2, y2 = (float(value) for value in bbox)
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height or x2 <= x1 or y2 <= y1:
            return "invalid_target_bbox"
    return None


def _link_or_copy(job: MediaJob) -> str:
    job.target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(job.source, job.target)
        return "hardlink"
    except OSError:
        shutil.copy2(job.source, job.target)
        return "copy"


def _copy_annotation(pair: tuple[Path, Path]) -> None:
    source, target = pair
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _verify_media(job: MediaJob) -> str | None:
    try:
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(job.target) as opened:
            opened.load()
            if opened.size == (job.width, job.height):
                return None
            transposed = ImageOps.exif_transpose(opened)
            try:
                if transposed.size == (job.width, job.height):
                    return None
            finally:
                if transposed is not opened:
                    transposed.close()
            return f"size_mismatch:{job.target}:{opened.size}!={(job.width, job.height)}"
    except Exception as exc:  # noqa: BLE001 - aggregate every corrupt source
        return f"decode_error:{job.target}:{type(exc).__name__}:{exc}"


def _materialize(records: Iterable[SourceRecord], target: Path, *, workers: int) -> dict[str, int]:
    ordered = sorted(records, key=lambda item: item.sample_id)
    json_jobs = [(item.json_path, target / "json" / f"{item.sample_id}.json") for item in ordered]
    media_jobs = [
        MediaJob(
            item.image_path,
            target / "images" / f"{item.sample_id}{item.image_path.suffix.lower()}",
            item.width,
            item.height,
        )
        for item in ordered
    ]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(_copy_annotation, json_jobs, chunksize=64))
    materialization: Counter[str] = Counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        materialization.update(executor.map(_link_or_copy, media_jobs, chunksize=64))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        errors = [
            error for error in executor.map(_verify_media, media_jobs, chunksize=32) if error
        ]
    if errors:
        raise ValueError(f"Media verification failed ({len(errors)}): {errors[:5]}")
    materialization["verified"] = len(media_jobs)
    return dict(sorted(materialization.items()))


def _publish(staging: Path, output: Path, *, clean: bool) -> None:
    if output.exists() and not clean:
        raise FileExistsError(f"Output already exists; pass --clean to replace it: {output}")
    backup: Path | None = None
    try:
        if output.exists():
            backup = output.with_name(f".{output.name}.backup.{uuid.uuid4().hex}")
            os.replace(output, backup)
        os.replace(staging, output)
    except Exception:
        if output.exists() and output != staging:
            shutil.rmtree(output, ignore_errors=True)
        if backup is not None and backup.exists():
            os.replace(backup, output)
        raise
    if backup is not None:
        shutil.rmtree(backup)


def build(args: argparse.Namespace) -> dict[str, Any]:
    base_root = Path(args.base_raw_root).resolve()
    incoming_json = Path(args.incoming_json_dir).resolve()
    incoming_images = Path(args.incoming_image_root).resolve()
    output = Path(args.output_root).resolve()
    workers = max(1, int(args.workers))
    if output.exists() and not bool(args.clean):
        raise FileExistsError(f"Output already exists; pass --clean to replace it: {output}")
    if not base_root.is_dir() or not incoming_json.is_dir() or not incoming_images.is_dir():
        missing = next(
            path for path in (base_root, incoming_json, incoming_images) if not path.is_dir()
        )
        raise FileNotFoundError(missing)

    base_ids = _read_split(Path(args.base_train_split).resolve())
    base_images = _image_index(base_root / "images")
    incoming_image_index = _image_index(incoming_images)
    real_v1_ids = set(_image_index(Path(args.real_v1_image_dir).resolve(), required=False))
    real_v2_ids = set(_image_index(Path(args.real_v2_image_dir).resolve(), required=False))
    test_ids = real_v1_ids | real_v2_ids

    base_records: list[SourceRecord] = []
    for sample_id in base_ids:
        json_path = base_root / "json" / f"{sample_id}.json"
        image_path = base_images.get(sample_id)
        if not json_path.is_file() or image_path is None:
            raise FileNotFoundError(f"Missing base source for {sample_id}")
        record, _ = _load_record(
            sample_id,
            json_path=json_path,
            image_path=image_path,
            origin="v5.8_grounding_train",
        )
        base_records.append(record)

    incoming_paths = sorted(incoming_json.glob("*.json"))
    if not incoming_paths:
        raise ValueError(f"Incoming annotation directory is empty: {incoming_json}")
    incoming_ids = [path.stem for path in incoming_paths]
    if len(incoming_ids) != len(set(incoming_ids)):
        raise ValueError("Incoming annotation identities must be unique.")
    collisions = sorted(set(base_ids) & set(incoming_ids))
    if collisions:
        raise ValueError(f"Incoming ids collide with v5.8 train ids: {collisions[:5]}")

    exclusions: dict[str, str] = {}
    candidates: dict[str, tuple[SourceRecord, dict[str, Any]]] = {}
    for json_path in incoming_paths:
        sample_id = json_path.stem
        image_path = incoming_image_index.get(sample_id)
        if image_path is None:
            raise FileNotFoundError(f"Missing incoming image for {sample_id}")
        record, payload = _load_record(
            sample_id,
            json_path=json_path,
            image_path=image_path,
            origin="v5.9_increment",
        )
        if sample_id in test_ids:
            exclusions[sample_id] = "test_id"
            continue
        reason = _quality_reason(payload, width=record.width, height=record.height)
        if reason is not None:
            exclusions[sample_id] = reason
            continue
        candidates[sample_id] = (record, payload)

    hash_groups: dict[str, list[str]] = defaultdict(list)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        hashes = executor.map(
            _sha256,
            (candidates[sample_id][0].image_path for sample_id in sorted(candidates)),
            chunksize=32,
        )
        for sample_id, digest in zip(sorted(candidates), hashes, strict=True):
            hash_groups[digest].append(sample_id)
    for group in hash_groups.values():
        if len(group) > 1:
            for sample_id in group:
                exclusions[sample_id] = "duplicate_image_quarantine"

    selected_new = [
        record
        for sample_id, (record, _) in sorted(candidates.items())
        if sample_id not in exclusions
    ]
    all_records = base_records + selected_new
    train_ids = sorted(record.sample_id for record in all_records)
    overlap = sorted(set(train_ids) & test_ids)
    if overlap:
        raise ValueError(f"Train/test ID overlap after selection: {overlap[:5]}")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging.", dir=output.parent))
    try:
        materialization = _materialize(all_records, staging, workers=workers)
        splits = staging / "splits"
        splits.mkdir(parents=True, exist_ok=True)
        (splits / "grounding_layout.train.txt").write_text(
            "".join(f"json/{sample_id}.json\n" for sample_id in train_ids), encoding="utf-8"
        )
        (splits / "grounding_layout.val.txt").write_text("", encoding="utf-8")
        (splits / "real_v1.ids.txt").write_text(
            "".join(f"{sample_id}\n" for sample_id in sorted(real_v1_ids)), encoding="utf-8"
        )
        (splits / "real_v2.ids.txt").write_text(
            "".join(f"{sample_id}\n" for sample_id in sorted(real_v2_ids)), encoding="utf-8"
        )
        exclusion_rows = [
            {"sample_id": sample_id, "reason": reason}
            for sample_id, reason in sorted(exclusions.items())
        ]
        (splits / "exclusions.jsonl").write_text(
            "".join(_json_dump(row) + "\n" for row in exclusion_rows), encoding="utf-8"
        )
        summary = {
            "status": "passed",
            "snapshot": "banana-v5.9-grounding-source-v1",
            "base_train": len(base_records),
            "incoming_total": len(incoming_paths),
            "selected_new": len(selected_new),
            "total_train": len(all_records),
            "exclusions": dict(sorted(Counter(exclusions.values()).items())),
            "test_gate": {
                "mode": "id_only",
                "real_v1_ids": len(real_v1_ids),
                "real_v2_ids": len(real_v2_ids),
                "union_ids": len(test_ids),
                "overlap": len(overlap),
            },
            "materialization": materialization,
        }
        _write_json(staging / "splits/split_summary.json", summary)
        _write_json(staging / "reports/final_validation.json", summary)
        (staging / "README.md").write_text(
            "# Banana v5.9 grounding source snapshot\n\n"
            "This train-only snapshot combines the frozen v5.8 `grounding_layout` train IDs "
            "with the quality-gated v5.9 annotation increment. `real_v1` and `real_v2` are "
            "excluded strictly by image ID. Exact-image conflict groups inside the incoming "
            "batch are quarantined. Original image pixels and compact annotations are preserved.\n\n"
            "Only `grounding_layout` may consume this snapshot. Line point attributes remain in "
            "the raw annotation but are not activated as a v5.9 line task.\n",
            encoding="utf-8",
        )
        _publish(staging, output, clean=bool(args.clean))
        return summary
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-raw-root", required=True)
    parser.add_argument("--base-train-split", required=True)
    parser.add_argument("--incoming-json-dir", required=True)
    parser.add_argument("--incoming-image-root", required=True)
    parser.add_argument("--real-v1-image-dir", required=True)
    parser.add_argument("--real-v2-image-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--workers", type=int, default=40)
    parser.add_argument("--clean", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    print(json.dumps(build(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
