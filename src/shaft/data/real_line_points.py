from __future__ import annotations

import json
import math
import os
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _bbox(value: Any, *, width: int, height: int) -> list[float]:
    if not isinstance(value, list | tuple) or len(value) != 4:
        raise ValueError(f"Invalid line bbox: {value!r}")
    if not all(_is_number(item) for item in value):
        raise ValueError(f"Invalid line bbox: {value!r}")
    x1, y1, x2, y2 = [float(item) for item in value]
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"Line bbox outside declared image size: {value!r}")
    return [x1, y1, x2, y2]


def _validate_points(value: Any, *, width: int, height: int) -> tuple[int, int]:
    if not isinstance(value, list) or not value:
        raise ValueError("Line points must be a non-empty segment list")
    duplicate_count = 0
    for segment in value:
        if not isinstance(segment, list) or len(segment) < 2:
            raise ValueError(f"Line segment requires at least two points: {segment!r}")
        deduplicated: list[tuple[float, float]] = []
        for point in segment:
            if (
                not isinstance(point, list | tuple)
                or len(point) != 2
                or not all(_is_number(coordinate) for coordinate in point)
            ):
                raise ValueError(f"Invalid line point: {point!r}")
            x, y = float(point[0]), float(point[1])
            if not (0 <= x <= width and 0 <= y <= height):
                raise ValueError(f"Line point outside declared image size: {point!r}")
            current = (x, y)
            if deduplicated and current == deduplicated[-1]:
                duplicate_count += 1
                continue
            deduplicated.append(current)
        if len(set(deduplicated)) < 2:
            raise ValueError(f"Line segment collapses after deduplication: {segment!r}")
    return len(value), duplicate_count


def _image_path(raw_root: Path, source_json: str, payload: dict[str, Any]) -> Path:
    explicit = payload.get("image_path")
    if isinstance(explicit, str) and explicit:
        candidate = raw_root / explicit
        if candidate.is_file():
            return candidate
    stem = Path(source_json).stem
    candidates = [raw_root / "images" / f"{stem}{suffix}" for suffix in IMAGE_SUFFIXES]
    existing = [path for path in candidates if path.is_file()]
    if len(existing) != 1:
        raise FileNotFoundError(
            f"Expected exactly one source image for {source_json}, found {existing}"
        )
    return existing[0]


def _source_key(source_json: str) -> str:
    path = Path(source_json).with_suffix("")
    parts = path.parts[1:] if path.parts and path.parts[0] == "json" else path.parts
    return "__".join(parts)


def _prepare_source(arguments: tuple[str, str]) -> tuple[list[dict[str, Any]], Counter[str]]:
    raw_root_text, source_json = arguments
    raw_root = Path(raw_root_text)
    path = raw_root / source_json
    payload = json.loads(path.read_text(encoding="utf-8"))
    size = payload.get("size")
    layout = payload.get("layout")
    if (
        not isinstance(size, list)
        or len(size) != 2
        or not all(_is_number(value) for value in size)
        or not isinstance(layout, list)
    ):
        raise ValueError(f"Invalid compact raw document: {path}")
    width, height = int(size[0]), int(size[1])
    if width <= 0 or height <= 0 or [width, height] != size:
        raise ValueError(f"Invalid compact image size: {path}: {size!r}")
    image = _image_path(raw_root, source_json, payload)
    source_image = image.relative_to(raw_root).as_posix()
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter({"source_json": 1})
    for instance_index, instance in enumerate(layout):
        if not isinstance(instance, dict) or instance.get("type") != "line":
            continue
        counts["line_instances"] += 1
        parameters = instance.get("parameters")
        points = parameters.get("points") if isinstance(parameters, dict) else None
        if not isinstance(points, list) or not points:
            counts["skipped_empty_points"] += 1
            continue
        segment_count, duplicate_count = _validate_points(
            points,
            width=width,
            height=height,
        )
        source_bbox = _bbox(instance.get("bbox"), width=width, height=height)
        sample_id = f"real__{_source_key(source_json)}__line_{instance_index:05d}"
        rows.append(
            {
                "sample_id": sample_id,
                "instances": [{"label": "line", "bbox": source_bbox}],
                "extra": {
                    "source_json": source_json,
                    "source_image": source_image,
                    "source_instance_index": instance_index,
                    "source_bbox": source_bbox,
                },
            }
        )
        counts["selected_nonempty_line"] += 1
        counts["selected_single_segment" if segment_count == 1 else "selected_multi_segment"] += 1
        counts["consecutive_duplicate_points"] += duplicate_count
    return rows, counts


def _split_entries(path: Path) -> list[str]:
    entries = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(entries) != len(set(entries)):
        raise ValueError(f"Duplicate source JSON entries in train split: {path}")
    return entries


def _excluded_ids(path: Path | None) -> set[str]:
    if path is None or not path.is_file():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError(f"Invalid exclusion manifest: {path}")
    return {
        str(item.get("id") or Path(str(item.get("image_path") or "")).stem)
        for item in items
        if isinstance(item, dict)
    } - {""}


def _atomic_write(path: Path, content: str) -> None:
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
        temporary = Path(handle.name)
    os.replace(temporary, path)


def prepare_real_line_point_selection(
    *,
    raw_root: Path,
    train_split: Path,
    exclude_manifest: Path | None,
    output: Path,
    workers: int,
    clean: bool,
) -> dict[str, Any]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    if output.exists() and not clean:
        raise FileExistsError(f"Output already exists; set clean=True to replace: {output}")
    entries = _split_entries(train_split)
    excluded = _excluded_ids(exclude_manifest)
    selected_entries = [entry for entry in entries if Path(entry).stem not in excluded]
    missing = [entry for entry in selected_entries if not (raw_root / entry).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing source JSON files: {missing[:5]}")
    work = [(str(raw_root), entry) for entry in selected_entries]
    if workers == 1:
        results = map(_prepare_source, work)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        results = executor.map(_prepare_source, work, chunksize=16)
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    try:
        for source_rows, source_counts in results:
            rows.extend(source_rows)
            counts.update(source_counts)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Generated duplicate real line point sample ids")
    counts["split_source_json"] = len(entries)
    counts["excluded_source_json"] = len(entries) - len(selected_entries)
    _atomic_write(output, "".join(_json_dumps(row) + "\n" for row in rows))
    summary = {
        "schema_version": 1,
        "source": "active_compact_human_raw",
        "raw_root": str(raw_root),
        "train_split": str(train_split),
        "exclude_manifest": str(exclude_manifest) if exclude_manifest is not None else None,
        "output": str(output),
        "workers": workers,
        "sampling": "all_nonempty_line_points_without_replacement",
        "counts": dict(sorted(counts.items())),
    }
    _atomic_write(
        output.with_name("build_summary.json"),
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _atomic_write(
        output.with_name("README.md"),
        "\n".join(
            (
                "# Real line context point selection",
                "",
                "Source-identity-only selection of every active human `line` instance with",
                "non-empty `parameters.points`. Empty point annotations are not synthesized.",
                "The builder reloads ordered points from compact raw truth and removes only",
                "consecutive duplicate coordinates in the derived target.",
                "",
                "Validation data is intentionally not generated.",
                "",
                "```json",
                json.dumps(dict(sorted(counts.items())), ensure_ascii=False, indent=2),
                "```",
                "",
            )
        ),
    )
    return summary


__all__ = ["prepare_real_line_point_selection"]
