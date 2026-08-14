#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


SHAPE_TYPES = {
    "rectangle",
    "oval",
    "triangle",
    "trapezoid",
    "parallelogram",
    "diamond",
    "step",
    "regular_pentagon",
    "regular_hexagon",
    "arrow_pentagon",
    "other_polygon",
    "callout",
    "card",
    "other",
}
LINE_TYPES = {"straight", "curved"}
LINE_STYLES = {"path", "shape"}
ARROW_TYPES = {"none", "line", "stealth", "triangle", "pointy", "tee", "circle"}
ALLOWED_INSTANCE_TYPES = {"shape", "icon", "image", "line"}
HEX_DIGITS = frozenset("0123456789abcdefABCDEF")


@dataclass(frozen=True)
class AuditConfig:
    dataset_root: Path
    verify_images: bool


@dataclass(frozen=True)
class AuditResult:
    stem: str
    fatal_issues: tuple[str, ...]
    issue_counts: dict[str, int]
    issue_examples: dict[str, tuple[str, ...]]
    distributions: dict[str, int]


@dataclass(frozen=True)
class Candidate:
    stem: str
    instance_index: int
    label: str
    bbox: tuple[float, float, float, float]
    stratum: str
    segment_count: int = 0


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent) as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
        temporary_path = Path(f.name)
    os.replace(temporary_path, path)


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _is_hex_color(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 7
        and value.startswith("#")
        and all(character in HEX_DIGITS for character in value[1:])
    )


def _load_split(path: Path) -> list[str]:
    stems = [Path(line.strip()).stem for line in path.read_text(encoding="utf-8").splitlines()]
    stems = [stem for stem in stems if stem]
    if len(stems) != len(set(stems)):
        duplicates = sorted(stem for stem, count in Counter(stems).items() if count > 1)
        raise ValueError(f"Duplicate ids in {path}: {duplicates[:10]}")
    return stems


def _bbox(value: Any, *, width: int, height: int) -> tuple[float, float, float, float] | None:
    if not isinstance(value, list | tuple) or len(value) != 4:
        return None
    if not all(_is_number(item) for item in value):
        return None
    x1, y1, x2, y2 = [float(item) for item in value]
    if x2 <= x1 or y2 <= y1:
        return None
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        return None
    return x1, y1, x2, y2


def _point(value: Any, *, width: int, height: int) -> bool:
    return (
        isinstance(value, list | tuple)
        and len(value) == 2
        and all(_is_number(item) for item in value)
        and 0 <= float(value[0]) <= width
        and 0 <= float(value[1]) <= height
    )


def _validate_exact_keys(value: dict[str, Any], expected: set[str], prefix: str) -> list[str]:
    issues: list[str] = []
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing:
        issues.append(f"{prefix}.missing:{','.join(missing)}")
    if extra:
        issues.append(f"{prefix}.extra:{','.join(extra)}")
    return issues


def _validate_border(
    value: Any,
    *,
    prefix: str,
    allow_complex: bool,
) -> list[str]:
    if not isinstance(value, dict):
        return [f"{prefix}.not_object"]
    allowed_types = {"none", "uniform"} | ({"complex"} if allow_complex else set())
    border_type = value.get("type")
    if border_type not in allowed_types:
        return [f"{prefix}.type:{border_type}"]
    if border_type == "uniform":
        issues = _validate_exact_keys(value, {"type", "style", "color"}, prefix)
        if value.get("style") not in {"solid", "dash"}:
            issues.append(f"{prefix}.style:{value.get('style')}")
        if not _is_hex_color(value.get("color")):
            issues.append(f"{prefix}.color")
        return issues
    return _validate_exact_keys(value, {"type"}, prefix)


def _validate_fill(value: Any, *, prefix: str) -> list[str]:
    if not isinstance(value, dict):
        return [f"{prefix}.not_object"]
    fill_type = value.get("type")
    if fill_type == "uniform":
        issues = _validate_exact_keys(value, {"type", "color"}, prefix)
        if not _is_hex_color(value.get("color")):
            issues.append(f"{prefix}.color")
        return issues
    if fill_type == "complex":
        return _validate_exact_keys(value, {"type"}, prefix)
    return [f"{prefix}.type:{fill_type}"]


def _validate_effect(value: Any, *, prefix: str = "effect") -> list[str]:
    if not isinstance(value, dict):
        return [f"{prefix}.not_object"]
    issues = _validate_exact_keys(value, {"type"}, prefix)
    if value.get("type") not in {"none", "exist"}:
        issues.append(f"{prefix}.type:{value.get('type')}")
    return issues


def _validate_corner(value: Any, *, width: int, height: int, prefix: str) -> list[str]:
    if not isinstance(value, dict):
        return [f"{prefix}.not_object"]
    corner_type = value.get("type")
    if corner_type == "sharp":
        issues = _validate_exact_keys(value, {"type", "point"}, prefix)
        if not _point(value.get("point"), width=width, height=height):
            issues.append(f"{prefix}.point")
        return issues
    if corner_type == "round":
        issues = _validate_exact_keys(value, {"type", "start", "mid", "end"}, prefix)
        for key in ("start", "mid", "end"):
            if not _point(value.get(key), width=width, height=height):
                issues.append(f"{prefix}.{key}")
        return issues
    return [f"{prefix}.type:{corner_type}"]


def _validate_corners(value: Any, *, width: int, height: int, prefix: str) -> list[str]:
    if not isinstance(value, list) or not value:
        return [f"{prefix}.empty_or_not_list"]
    issues: list[str] = []
    for index, corner in enumerate(value):
        issues.extend(
            _validate_corner(corner, width=width, height=height, prefix=f"{prefix}[{index}]")
        )
    return issues


def _fill_signature(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("type", "missing"))
    if isinstance(value, list):
        return f"regions{len(value)}:" + "+".join(
            str(item.get("type", "missing")) if isinstance(item, dict) else "invalid"
            for item in value
        )
    return "missing"


def _border_signature(value: Any) -> str:
    if not isinstance(value, dict):
        return "missing"
    return f"{value.get('type', 'missing')}:{value.get('style', '-')}"


def _shape_stratum(parameters: dict[str, Any]) -> str:
    shape_type = str(parameters.get("shape_type", "missing"))
    corners = parameters.get("corners") or parameters.get("body_corners")
    round_count = 0
    if isinstance(corners, list):
        round_count = sum(
            1 for corner in corners if isinstance(corner, dict) and corner.get("type") == "round"
        )
    splits = parameters.get("splits")
    split_count = len(splits) if isinstance(splits, list) else 0
    return "|".join(
        (
            shape_type,
            _border_signature(parameters.get("border")),
            _fill_signature(parameters.get("fill")),
            str((parameters.get("effect") or {}).get("type", "missing")),
            f"round{round_count}",
            f"splits{split_count}",
        )
    )


def _line_stratum(parameters: dict[str, Any]) -> str:
    points = parameters.get("points")
    segment_count = len(points) if isinstance(points, list) else 0
    return "|".join(
        (
            str(parameters.get("line_type", "missing")),
            str(parameters.get("line_style", "missing")),
            f"segments{segment_count}",
            str(parameters.get("dash_style", "missing")),
            str(parameters.get("begin_arrow", "missing")),
            str(parameters.get("end_arrow", "missing")),
            _fill_signature(parameters.get("fill")),
            _border_signature(parameters.get("border")),
            str(parameters.get("corner_style", "-")),
        )
    )


def _validate_shape(parameters: Any, *, width: int, height: int) -> list[str]:
    if not isinstance(parameters, dict):
        return ["shape.parameters.not_object"]
    shape_type = parameters.get("shape_type")
    if shape_type not in SHAPE_TYPES:
        return [f"shape.shape_type:{shape_type}"]
    if shape_type == "other":
        return _validate_exact_keys(parameters, {"shape_type"}, "shape.parameters")

    issues: list[str] = []
    if shape_type == "card":
        issues.extend(
            _validate_exact_keys(
                parameters,
                {"shape_type", "border", "fill", "corners", "splits", "effect"},
                "shape.parameters",
            )
        )
        fills = parameters.get("fill")
        splits = parameters.get("splits")
        if not isinstance(fills, list) or not fills:
            issues.append("shape.card.fill")
        else:
            for index, fill in enumerate(fills):
                issues.extend(_validate_fill(fill, prefix=f"shape.card.fill[{index}]"))
        if not isinstance(splits, list) or not splits:
            issues.append("shape.card.splits")
        else:
            for index, split in enumerate(splits):
                if not isinstance(split, dict):
                    issues.append(f"shape.card.splits[{index}].not_object")
                    continue
                split_border = {key: value for key, value in split.items() if key != "split_corners"}
                issues.extend(
                    _validate_border(
                        split_border,
                        prefix=f"shape.card.splits[{index}]",
                        allow_complex=True,
                    )
                )
                split_corners = split.get("split_corners")
                if not isinstance(split_corners, list) or len(split_corners) != 2:
                    issues.append(f"shape.card.splits[{index}].split_corners")
                else:
                    for corner_index, corner in enumerate(split_corners):
                        issues.extend(
                            _validate_corner(
                                corner,
                                width=width,
                                height=height,
                                prefix=(
                                    f"shape.card.splits[{index}].split_corners[{corner_index}]"
                                ),
                            )
                        )
        if isinstance(fills, list) and isinstance(splits, list) and len(fills) != len(splits) + 1:
            issues.append("shape.card.fill_split_count")
        issues.extend(
            _validate_corners(
                parameters.get("corners"), width=width, height=height, prefix="shape.corners"
            )
        )
    elif shape_type == "callout":
        body_type = parameters.get("body_type")
        expected = {"shape_type", "border", "fill", "body_type", "tail", "effect"}
        if body_type == "rectangle":
            expected.add("body_corners")
        elif body_type == "oval":
            expected.add("body_bbox")
        else:
            issues.append(f"shape.callout.body_type:{body_type}")
        issues.extend(_validate_exact_keys(parameters, expected, "shape.parameters"))
        if body_type == "rectangle":
            issues.extend(
                _validate_corners(
                    parameters.get("body_corners"),
                    width=width,
                    height=height,
                    prefix="shape.body_corners",
                )
            )
        elif body_type == "oval" and _bbox(
            parameters.get("body_bbox"), width=width, height=height
        ) is None:
            issues.append("shape.callout.body_bbox")
        tail = parameters.get("tail")
        if not isinstance(tail, dict) or set(tail) != {"points"}:
            issues.append("shape.callout.tail")
        else:
            points = tail.get("points")
            if not isinstance(points, list) or len(points) != 3:
                issues.append("shape.callout.tail.points")
            else:
                for index, point in enumerate(points):
                    if not _point(point, width=width, height=height):
                        issues.append(f"shape.callout.tail.points[{index}]")
    else:
        expected = {"shape_type", "border", "fill", "effect"}
        if shape_type != "oval":
            expected.add("corners")
        issues.extend(_validate_exact_keys(parameters, expected, "shape.parameters"))
        if shape_type != "oval":
            issues.extend(
                _validate_corners(
                    parameters.get("corners"),
                    width=width,
                    height=height,
                    prefix="shape.corners",
                )
            )

    issues.extend(_validate_border(parameters.get("border"), prefix="shape.border", allow_complex=True))
    if shape_type != "card":
        issues.extend(_validate_fill(parameters.get("fill"), prefix="shape.fill"))
    issues.extend(_validate_effect(parameters.get("effect")))
    return issues


def _validate_line(parameters: Any, *, width: int, height: int) -> list[str]:
    if not isinstance(parameters, dict):
        return ["line.parameters.not_object"]
    required = {
        "line_type",
        "line_style",
        "is_single",
        "points",
        "dash_style",
        "begin_arrow",
        "end_arrow",
        "fill",
        "border",
    }
    allowed = required | {"corner_style"}
    issues: list[str] = []
    missing = sorted(required - set(parameters))
    extra = sorted(set(parameters) - allowed)
    if missing:
        issues.append(f"line.parameters.missing:{','.join(missing)}")
    if extra:
        issues.append(f"line.parameters.extra:{','.join(extra)}")
    line_type = parameters.get("line_type")
    line_style = parameters.get("line_style")
    if not isinstance(line_type, str) or line_type not in LINE_TYPES:
        issues.append(f"line.line_type:{line_type}")
    if not isinstance(line_style, str) or line_style not in LINE_STYLES:
        issues.append(f"line.line_style:{line_style}")
    dash_style = parameters.get("dash_style")
    if not isinstance(dash_style, str) or dash_style not in {"solid", "dash"}:
        issues.append(f"line.dash_style:{dash_style}")
    for field in ("begin_arrow", "end_arrow"):
        marker = parameters.get(field)
        if not isinstance(marker, str) or marker not in ARROW_TYPES:
            issues.append(f"line.{field}:{marker}")
        if marker == "line" and line_style != "path":
            issues.append(f"line.{field}.line_requires_path")
        if marker == "pointy" and line_style != "shape":
            issues.append(f"line.{field}.pointy_requires_shape")
    points = parameters.get("points")
    if not isinstance(points, list) or not points:
        issues.append("line.points.empty_or_not_list")
        segment_count = 0
    else:
        segment_count = len(points)
        for segment_index, segment in enumerate(points):
            if not isinstance(segment, list) or len(segment) < 2:
                issues.append(f"line.points[{segment_index}].too_short")
                continue
            if line_type == "curved" and len(segment) != 4:
                issues.append(f"line.points[{segment_index}].curved_requires_four")
            for point_index, point in enumerate(segment):
                if not _point(point, width=width, height=height):
                    issues.append(f"line.points[{segment_index}][{point_index}]")
    if not isinstance(parameters.get("is_single"), bool):
        issues.append("line.is_single.not_bool")
    elif parameters["is_single"] != (segment_count == 1):
        issues.append("line.is_single.segment_mismatch")
    corner_style = parameters.get("corner_style")
    if corner_style is not None:
        if not isinstance(corner_style, str) or corner_style not in {"sharp", "round"}:
            issues.append(f"line.corner_style:{corner_style}")
        if line_type != "straight":
            issues.append("line.corner_style.invalid_owner")
    issues.extend(_validate_fill(parameters.get("fill"), prefix="line.fill"))
    issues.extend(_validate_border(parameters.get("border"), prefix="line.border", allow_complex=False))
    return issues


def _validate_image(parameters: Any, *, width: int, height: int) -> list[str]:
    if not isinstance(parameters, dict):
        return ["image.parameters.not_object"]
    expected = {"image_type", "clip_shape", "border", "effect"}
    if "corners" in parameters:
        expected.add("corners")
    issues = _validate_exact_keys(parameters, expected, "image.parameters")
    if parameters.get("image_type") != "N/A":
        issues.append(f"image.image_type:{parameters.get('image_type')}")
    clip_shape = parameters.get("clip_shape")
    polygon_clip_shapes = {
        "rectangle",
        "triangle",
        "trapezoid",
        "parallelogram",
        "diamond",
        "regular_pentagon",
        "regular_hexagon",
        "arrow_pentagon",
        "other_polygon",
    }
    if clip_shape not in {"none", "oval", *polygon_clip_shapes}:
        issues.append(f"image.clip_shape:{clip_shape}")
    if clip_shape in polygon_clip_shapes and "corners" not in parameters:
        issues.append("image.polygon.missing_corners")
    if clip_shape in {"none", "oval"} and "corners" in parameters:
        if parameters["corners"] != []:
            issues.append("image.non_polygon.corners_must_be_empty")
    if clip_shape in polygon_clip_shapes and "corners" in parameters:
        issues.extend(
            _validate_corners(
                parameters["corners"], width=width, height=height, prefix="image.corners"
            )
        )
    issues.extend(_validate_border(parameters.get("border"), prefix="image.border", allow_complex=True))
    issues.extend(_validate_effect(parameters.get("effect"), prefix="image.effect"))
    return issues


def _record_issue(
    issue_counts: Counter[str],
    issue_examples: dict[str, list[str]],
    code: str,
    example: str,
) -> None:
    issue_counts[code] += 1
    if len(issue_examples[code]) < 8:
        issue_examples[code].append(example)


def _audit_one(item: tuple[str, AuditConfig]) -> AuditResult:
    stem, config = item
    annotation_path = config.dataset_root / "gt_standard" / f"{stem}.json"
    image_path = config.dataset_root / "img" / f"{stem}.png"
    fatal: list[str] = []
    issue_counts: Counter[str] = Counter()
    issue_examples: dict[str, list[str]] = defaultdict(list)
    distributions: Counter[str] = Counter()
    if not annotation_path.is_file():
        return AuditResult(stem, ("missing_json",), {}, {}, {})
    if not image_path.is_file():
        return AuditResult(stem, ("missing_image",), {}, {}, {})
    try:
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return AuditResult(stem, (f"json_parse:{type(exc).__name__}",), {}, {}, {})
    if not isinstance(payload, dict):
        return AuditResult(stem, ("root_not_object",), {}, {}, {})
    if set(payload) != {"size", "background", "layout"}:
        _record_issue(
            issue_counts,
            issue_examples,
            "root.keys",
            f"{stem}:{sorted(payload)}",
        )
    size = payload.get("size")
    if (
        not isinstance(size, list | tuple)
        or len(size) != 2
        or not all(isinstance(item, int) and item > 0 for item in size)
    ):
        return AuditResult(stem, ("invalid_size",), dict(issue_counts), {}, {})
    width, height = int(size[0]), int(size[1])
    if config.verify_images:
        try:
            with Image.open(image_path) as image:
                if image.size != (width, height):
                    fatal.append(f"image_size_mismatch:{image.size[0]}x{image.size[1]}")
        except Exception as exc:
            fatal.append(f"image_open:{type(exc).__name__}")
    background = payload.get("background")
    distributions[f"background:{background}"] += 1
    layout = payload.get("layout")
    if not isinstance(layout, list):
        fatal.append("layout_not_list")
        return AuditResult(
            stem,
            tuple(fatal),
            dict(issue_counts),
            {key: tuple(values) for key, values in issue_examples.items()},
            dict(distributions),
        )
    distributions["files"] += 1
    distributions["instances"] += len(layout)
    for index, instance in enumerate(layout):
        example = f"{stem}:{index}"
        if not isinstance(instance, dict):
            _record_issue(issue_counts, issue_examples, "instance.not_object", example)
            continue
        label = instance.get("type")
        distributions[f"instance_type:{label}"] += 1
        if label not in ALLOWED_INSTANCE_TYPES:
            _record_issue(issue_counts, issue_examples, f"instance.type:{label}", example)
            continue
        expected_keys = {"type", "bbox"} if label == "icon" else {"type", "bbox", "parameters"}
        for code in _validate_exact_keys(instance, expected_keys, f"{label}.instance"):
            _record_issue(issue_counts, issue_examples, code, example)
        valid_bbox = _bbox(instance.get("bbox"), width=width, height=height)
        if valid_bbox is None:
            _record_issue(issue_counts, issue_examples, f"{label}.bbox", example)
            continue
        parameters = instance.get("parameters")
        if label == "shape":
            errors = _validate_shape(parameters, width=width, height=height)
            shape_type = parameters.get("shape_type") if isinstance(parameters, dict) else None
            distributions[f"shape_type:{shape_type}"] += 1
            if not errors and isinstance(parameters, dict):
                distributions[f"shape_stratum:{_shape_stratum(parameters)}"] += 1
        elif label == "line":
            errors = _validate_line(parameters, width=width, height=height)
            if isinstance(parameters, dict):
                points = parameters.get("points")
                segment_count = len(points) if isinstance(points, list) else 0
                distributions[f"line_segments:{segment_count}"] += 1
            if not errors and isinstance(parameters, dict):
                distributions[f"line_stratum:{_line_stratum(parameters)}"] += 1
        elif label == "image":
            errors = _validate_image(parameters, width=width, height=height)
            if isinstance(parameters, dict):
                distributions[f"image_type:{parameters.get('image_type')}"] += 1
                distributions[f"image_clip_shape:{parameters.get('clip_shape')}"] += 1
        else:
            errors = []
        if errors:
            for code in errors:
                _record_issue(issue_counts, issue_examples, code, example)
        else:
            distributions[f"valid_instance:{label}"] += 1
    return AuditResult(
        stem=stem,
        fatal_issues=tuple(fatal),
        issue_counts=dict(issue_counts),
        issue_examples={key: tuple(values) for key, values in issue_examples.items()},
        distributions=dict(distributions),
    )


def _candidate_one(item: tuple[str, AuditConfig]) -> tuple[Candidate, ...]:
    stem, config = item
    payload = json.loads(
        (config.dataset_root / "gt_standard" / f"{stem}.json").read_text(encoding="utf-8")
    )
    width, height = [int(value) for value in payload["size"]]
    candidates: list[Candidate] = []
    for index, instance in enumerate(payload["layout"]):
        if not isinstance(instance, dict):
            continue
        label = instance.get("type")
        if label not in {"shape", "line"}:
            continue
        bbox = _bbox(instance.get("bbox"), width=width, height=height)
        parameters = instance.get("parameters")
        if bbox is None:
            continue
        if label == "shape":
            errors = _validate_shape(parameters, width=width, height=height)
            if errors or not isinstance(parameters, dict):
                continue
            candidates.append(
                Candidate(stem, index, label, bbox, _shape_stratum(parameters))
            )
        else:
            errors = _validate_line(parameters, width=width, height=height)
            if errors or not isinstance(parameters, dict):
                continue
            points = parameters["points"]
            candidates.append(
                Candidate(
                    stem,
                    index,
                    label,
                    bbox,
                    _line_stratum(parameters),
                    len(points),
                )
            )
    return tuple(candidates)


def _split_distributions(counter: Counter[str]) -> dict[str, dict[str, int] | int]:
    result: dict[str, dict[str, int] | int] = {}
    grouped: dict[str, dict[str, int]] = defaultdict(dict)
    for key, count in sorted(counter.items()):
        if ":" not in key:
            result[key] = count
            continue
        prefix, value = key.split(":", 1)
        grouped[prefix][value] = count
    result.update({key: dict(sorted(values.items())) for key, values in grouped.items()})
    return result


def audit_dataset(
    *,
    dataset_root: Path,
    split_path: Path,
    workers: int,
    verify_images: bool,
) -> tuple[dict[str, Any], list[str]]:
    dataset_root = dataset_root.resolve()
    stems = _load_split(split_path)
    config = AuditConfig(dataset_root=dataset_root, verify_images=verify_images)
    distributions: Counter[str] = Counter()
    issue_counts: Counter[str] = Counter()
    issue_examples: dict[str, list[str]] = defaultdict(list)
    fatal_counts: Counter[str] = Counter()
    fatal_examples: dict[str, list[str]] = defaultdict(list)
    valid_stems: list[str] = []
    work_items = ((stem, config) for stem in stems)
    if workers <= 1:
        results: Iterable[AuditResult] = map(_audit_one, work_items)
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        results = executor.map(_audit_one, work_items, chunksize=32)
    try:
        for result in results:
            distributions.update(result.distributions)
            issue_counts.update(result.issue_counts)
            for code, examples in result.issue_examples.items():
                room = max(0, 20 - len(issue_examples[code]))
                issue_examples[code].extend(examples[:room])
            if result.fatal_issues:
                for code in result.fatal_issues:
                    fatal_counts[code] += 1
                    if len(fatal_examples[code]) < 20:
                        fatal_examples[code].append(result.stem)
            else:
                valid_stems.append(result.stem)
    finally:
        if workers > 1:
            executor.shutdown()
    summary = {
        "dataset_root": str(dataset_root),
        "split_path": str(split_path.resolve()),
        "source_rows": len(stems),
        "structurally_valid_rows": len(valid_stems),
        "fatal_rows": len(stems) - len(valid_stems),
        "verify_images": verify_images,
        "fatal_counts": dict(sorted(fatal_counts.items())),
        "fatal_examples": dict(sorted(fatal_examples.items())),
        "instance_issue_counts": dict(sorted(issue_counts.items())),
        "instance_issue_examples": dict(sorted(issue_examples.items())),
        "distributions": _split_distributions(distributions),
    }
    return summary, valid_stems


def _allocate_caps(counts: dict[str, int], *, target: int) -> dict[str, int]:
    available = sum(counts.values())
    if target >= available:
        return dict(counts)
    if target <= 0:
        return {key: 0 for key in counts}
    result = {key: 0 for key in counts}
    active = {key for key, count in counts.items() if count > 0}
    remaining = target
    while remaining and active:
        weights = {key: math.sqrt(counts[key]) for key in active}
        total_weight = sum(weights.values())
        proposals = {key: remaining * weights[key] / total_weight for key in active}
        progressed = 0
        for key in sorted(active):
            capacity = counts[key] - result[key]
            take = min(capacity, int(proposals[key]))
            if take:
                result[key] += take
                remaining -= take
                progressed += take
        active = {key for key in active if result[key] < counts[key]}
        if remaining and active and progressed == 0:
            for key in sorted(active, key=lambda item: (proposals[item], item), reverse=True):
                if not remaining:
                    break
                result[key] += 1
                remaining -= 1
    if sum(result.values()) != target:
        raise RuntimeError("Failed to allocate sampling quotas")
    return result


def _shape_quotas(
    shape_strata: dict[str, int],
    *,
    target: int,
    keep_all_threshold: int,
) -> tuple[dict[str, int], dict[str, Any]]:
    by_type: dict[str, dict[str, int]] = defaultdict(dict)
    for stratum, count in shape_strata.items():
        by_type[stratum.split("|", 1)[0]][stratum] = count
    type_counts = {shape_type: sum(values.values()) for shape_type, values in by_type.items()}
    protected = {
        shape_type for shape_type, count in type_counts.items() if count <= keep_all_threshold
    }
    protected_count = sum(type_counts[shape_type] for shape_type in protected)
    head_types = {key: value for key, value in type_counts.items() if key not in protected}
    effective_target = min(sum(type_counts.values()), max(target, protected_count + len(head_types)))
    head_caps = _allocate_caps(head_types, target=effective_target - protected_count)
    quotas: dict[str, int] = {}
    for shape_type, strata in by_type.items():
        type_target = type_counts[shape_type] if shape_type in protected else head_caps[shape_type]
        quotas.update(_allocate_caps(strata, target=type_target))
    policy = {
        "requested_target": target,
        "effective_target": effective_target,
        "keep_all_threshold": keep_all_threshold,
        "keep_all_shape_types": sorted(protected),
        "available_by_shape_type": dict(sorted(type_counts.items())),
        "selected_by_shape_type": dict(
            sorted(
                {
                    shape_type: sum(quotas[stratum] for stratum in strata)
                    for shape_type, strata in by_type.items()
                }.items()
            )
        ),
    }
    return quotas, policy


def _line_quotas(
    line_strata: dict[str, int],
    *,
    target: int,
    keep_all_stratum_threshold: int,
) -> tuple[dict[str, int], dict[str, Any]]:
    protected = {
        stratum: count
        for stratum, count in line_strata.items()
        if count <= keep_all_stratum_threshold
    }
    head = {key: value for key, value in line_strata.items() if key not in protected}
    protected_count = sum(protected.values())
    effective_target = min(sum(line_strata.values()), max(target, protected_count + len(head)))
    quotas = dict(protected)
    quotas.update(_allocate_caps(head, target=effective_target - protected_count))
    return quotas, {
        "requested_target": target,
        "effective_target": effective_target,
        "keep_all_stratum_threshold": keep_all_stratum_threshold,
        "protected_strata": len(protected),
        "protected_rows": protected_count,
        "available_strata": len(line_strata),
    }


def _stable_score(candidate: Candidate, *, seed: int, task: str) -> int:
    raw = f"{seed}:{task}:{candidate.stem}:{candidate.instance_index}".encode()
    return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "big")


def _select_candidates(
    candidates: Iterable[tuple[Candidate, ...]],
    *,
    quotas: dict[str, int],
    seed: int,
    label: str,
) -> list[Candidate]:
    heaps: dict[str, list[tuple[int, str, int, Candidate]]] = defaultdict(list)
    for batch in candidates:
        for candidate in batch:
            if candidate.label != label:
                continue
            quota = quotas.get(candidate.stratum, 0)
            if quota <= 0:
                continue
            score = _stable_score(candidate, seed=seed, task=label)
            entry = (-score, candidate.stem, candidate.instance_index, candidate)
            heap = heaps[candidate.stratum]
            if len(heap) < quota:
                heapq.heappush(heap, entry)
            elif entry > heap[0]:
                heapq.heapreplace(heap, entry)
    selected = [entry[3] for heap in heaps.values() for entry in heap]
    selected.sort(key=lambda item: _stable_score(item, seed=seed, task=f"{label}:manifest"))
    expected = sum(quotas.values())
    if len(selected) != expected:
        raise RuntimeError(f"Selected {len(selected)} {label} rows, expected {expected}")
    return selected


def _select_line_points(
    selected_lines: list[Candidate], *, target: int, seed: int
) -> tuple[list[Candidate], dict[str, Any]]:
    by_segments: dict[int, list[Candidate]] = defaultdict(list)
    for candidate in selected_lines:
        if candidate.segment_count > 1:
            by_segments[candidate.segment_count].append(candidate)
    capacities = {str(key): len(values) for key, values in by_segments.items()}
    quotas = _allocate_caps(capacities, target=min(target, sum(capacities.values())))
    selected: list[Candidate] = []
    for segment_count, values in sorted(by_segments.items()):
        values.sort(key=lambda item: _stable_score(item, seed=seed, task="line_points"))
        selected.extend(values[: quotas[str(segment_count)]])
    selected.sort(key=lambda item: _stable_score(item, seed=seed, task="line_points:manifest"))
    return selected, {
        "requested_target": target,
        "effective_target": len(selected),
        "available_by_segment_count": dict(sorted(capacities.items())),
        "selected_by_segment_count": dict(sorted(quotas.items())),
    }


def _selection_row(candidate: Candidate) -> dict[str, Any]:
    label = candidate.label
    bbox = [float(value) for value in candidate.bbox]
    return {
        "sample_id": f"{candidate.stem}__{label}_{candidate.instance_index:04d}",
        "instances": [{"label": label, "bbox": bbox}],
        "extra": {
            "source_json": f"gt_standard/{candidate.stem}.json",
            "source_image": f"img/{candidate.stem}.png",
            "source_instance_index": candidate.instance_index,
            "source_bbox": bbox,
        },
    }


def _write_selection(path: Path, selected: list[Candidate]) -> None:
    body = "".join(_json_dumps(_selection_row(candidate)) + "\n" for candidate in selected)
    _atomic_write_text(path, body)


def _write_preparation_readme(
    path: Path,
    *,
    audit_summary: dict[str, Any],
    selection_summary: dict[str, Any],
) -> None:
    shape = selection_summary["shape"]
    line = selection_summary["line"]
    line_points = selection_summary["line_points"]
    content = f"""# V5.7 Reconstruction Selection

- Source dataset: `{selection_summary['source_dataset']}`
- Split source: `{audit_summary['split_path']}`
- Structurally valid source rows: `{audit_summary['structurally_valid_rows']}` / `{audit_summary['source_rows']}`
- Selection seed: `{selection_summary['seed']}`
- Selection rows: `{selection_summary['rows']}`
- Selection manifests contain source identity and bbox only; reconstruction truth is reloaded from
  `gt_standard` by the context builder.
- Validation files are intentionally empty. The source `val.txt` remains outside train.

## Shape

- Requested/effective rows: `{shape['requested_target']}` / `{shape['effective_target']}`
- Keep-all threshold: `{shape['keep_all_threshold']}` instances per shape type
- Fully retained types: `{shape['keep_all_shape_types']}`
- Selected distribution: `{shape['selected_by_shape_type']}`
- Head types are sampled without replacement, then stratified by border, fill, effect, rounded
  corners, and card split count.

## Line

- Requested/effective rows: `{line['requested_target']}` / `{line['effective_target']}`
- Attribute strata: `{line['available_strata']}`
- Rare strata kept in full: `{line['protected_strata']}` strata / `{line['protected_rows']}` rows
- Strata combine line type/style, segment count, dash, endpoints, fill, border, and corner style.

## Line Points

- Requested/effective rows: `{line_points['requested_target']}` / `{line_points['effective_target']}`
- Only multi-segment synthetic lines are selected.
- Selected segment-count distribution: `{line_points['selected_by_segment_count']}`

## Cleaning

- Fatal source rows: `{audit_summary['fatal_rows']}`
- Reconstruction-invalid instance counts: `{audit_summary['instance_issue_counts']}`
- Invalid reconstruction instances stay in source truth for audit, but are excluded from these
  selection manifests. Grounding applies its own bbox clipping/drop policy.
"""
    _atomic_write_text(path, content)


def prepare_selections(
    *,
    dataset_root: Path,
    output_root: Path,
    valid_stems: list[str],
    audit_summary: dict[str, Any],
    workers: int,
    seed: int,
    shape_target: int,
    line_target: int,
    line_points_target: int,
    shape_keep_all_threshold: int,
    line_keep_all_stratum_threshold: int,
) -> dict[str, Any]:
    distributions = audit_summary["distributions"]
    shape_strata = dict(distributions.get("shape_stratum", {}))
    line_strata = dict(distributions.get("line_stratum", {}))
    shape_quotas, shape_policy = _shape_quotas(
        shape_strata,
        target=shape_target,
        keep_all_threshold=shape_keep_all_threshold,
    )
    line_quotas, line_policy = _line_quotas(
        line_strata,
        target=line_target,
        keep_all_stratum_threshold=line_keep_all_stratum_threshold,
    )
    config = AuditConfig(dataset_root=dataset_root.resolve(), verify_images=False)

    def batches() -> Iterable[tuple[Candidate, ...]]:
        work_items = ((stem, config) for stem in valid_stems)
        if workers <= 1:
            yield from map(_candidate_one, work_items)
            return
        with ProcessPoolExecutor(max_workers=workers) as executor:
            yield from executor.map(_candidate_one, work_items, chunksize=32)

    selected_shapes = _select_candidates(
        batches(), quotas=shape_quotas, seed=seed, label="shape"
    )
    selected_lines = _select_candidates(
        batches(), quotas=line_quotas, seed=seed, label="line"
    )
    selected_line_points, line_points_policy = _select_line_points(
        selected_lines, target=line_points_target, seed=seed
    )
    _write_selection(output_root / "shape/train.jsonl", selected_shapes)
    _write_selection(output_root / "line/train.jsonl", selected_lines)
    _write_selection(output_root / "line_points/train.jsonl", selected_line_points)
    for name in ("shape", "line", "line_points"):
        _atomic_write_text(output_root / name / "val.jsonl", "")
    summary = {
        "source_dataset": str(dataset_root.resolve()),
        "seed": seed,
        "shape": shape_policy,
        "line": line_policy,
        "line_points": line_points_policy,
        "rows": {
            "shape": len(selected_shapes),
            "line": len(selected_lines),
            "line_points": len(selected_line_points),
        },
    }
    _atomic_write_text(
        output_root / "selection_summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _write_preparation_readme(
        output_root / "README.md",
        audit_summary=audit_summary,
        selection_summary=summary,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit v5.7 gt_standard data and build stratified reconstruction selections."
    )
    parser.add_argument(
        "--dataset-root",
        default="data/regulated_layout_dataset_v9_20260802",
    )
    parser.add_argument(
        "--output-root",
        default="data/reconstruction_v5_7_selection",
    )
    parser.add_argument("--split-file")
    parser.add_argument("--workers", type=int, default=40)
    parser.add_argument("--seed", type=int, default=57)
    parser.add_argument("--shape-target", type=int, default=300_000)
    parser.add_argument("--line-target", type=int, default=300_000)
    parser.add_argument("--line-points-target", type=int, default=15_000)
    parser.add_argument("--shape-keep-all-threshold", type=int, default=60_000)
    parser.add_argument("--line-keep-all-stratum-threshold", type=int, default=32)
    parser.add_argument("--skip-image-verification", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("workers must be positive")
    for name in (
        "shape_target",
        "line_target",
        "line_points_target",
        "shape_keep_all_threshold",
        "line_keep_all_stratum_threshold",
    ):
        if getattr(args, name) < 0:
            parser.error(f"{name.replace('_', '-')} must be non-negative")

    dataset_root = Path(args.dataset_root).resolve()
    output_root = Path(args.output_root).resolve()
    split_path = Path(args.split_file).resolve() if args.split_file else dataset_root / "train.txt"
    val_path = dataset_root / "val.txt"
    if val_path.is_file() and split_path != val_path.resolve():
        overlap = sorted(set(_load_split(split_path)) & set(_load_split(val_path)))
        if overlap:
            parser.error(f"train/val source overlap: {overlap[:10]}")
    summary, valid_stems = audit_dataset(
        dataset_root=dataset_root,
        split_path=split_path,
        workers=args.workers,
        verify_images=not args.skip_image_verification,
    )
    _atomic_write_text(
        output_root / "audit_summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _atomic_write_text(output_root / "clean_train.txt", "\n".join(valid_stems) + "\n")
    result: dict[str, Any] = {"audit": summary}
    if not args.audit_only:
        result["selection"] = prepare_selections(
            dataset_root=dataset_root,
            output_root=output_root,
            valid_stems=valid_stems,
            audit_summary=summary,
            workers=args.workers,
            seed=args.seed,
            shape_target=args.shape_target,
            line_target=args.line_target,
            line_points_target=args.line_points_target,
            shape_keep_all_threshold=args.shape_keep_all_threshold,
            line_keep_all_stratum_threshold=args.line_keep_all_stratum_threshold,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
