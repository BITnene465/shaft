"""Conservative, reproducible compact-raw quality audit and reversible cleaning.

Only exact instance duplicates and <=3px bbox overflow are repaired. Entire files with
unambiguous invalid geometry/media are quarantined. Partial labels and source point order
are preserved. This is not a visual semantic or test-leakage certification.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image


def vector(value, length):
    return (
        isinstance(value, list)
        and len(value) == length
        and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
            for v in value
        )
    )


def unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def inspect_annotation(original):
    data = copy.deepcopy(original)
    issues = []

    def add(level, code, index=None, detail=None):
        issues.append(dict(level=level, code=code, index=index, detail=detail))

    if not isinstance(data, dict) or not vector(data.get("size"), 2):
        add("reject", "invalid_size")
        return data, issues
    w, h = data["size"]
    if min(w, h) <= 0 or int(w) != w or int(h) != h:
        add("reject", "invalid_size")
        return data, issues
    if not isinstance(data.get("layout"), list):
        add("reject", "invalid_layout")
        return data, issues

    def geometry(value, index, field="parameters"):
        if isinstance(value, dict):
            for key, child in value.items():
                name = f"{field}.{key}"
                if key == "subbbox":
                    continue  # Unused source field; never a quality gate.
                if key in ("point", "start", "mid", "end"):
                    if not vector(child, 2):
                        add("reject", "invalid_attribute_point", index, name)
                    elif not (0 <= child[0] <= w and 0 <= child[1] <= h):
                        add("reject", "attribute_point_outside_image", index, [name, child])
                elif key == "body_bbox" and child:
                    check_box(child, index, name)
                elif key == "points":
                    check_points(child, index, name)
                else:
                    geometry(child, index, name)
        elif isinstance(value, list):
            for child in value:
                geometry(child, index, field)

    def check_points(value, index, field):
        if not isinstance(value, list):
            add("reject", "invalid_points_container", index, field)
        elif vector(value, 2):
            if not (0 <= value[0] <= w and 0 <= value[1] <= h):
                add("reject", "point_outside_image", index, [field, value])
        else:
            for child in value:
                if not isinstance(child, list):
                    add("reject", "invalid_point", index, [field, child])
                else:
                    check_points(child, index, field)

    def check_box(box, index, field):
        if not vector(box, 4) or box[0] >= box[2] or box[1] >= box[3]:
            add("reject", "invalid_attribute_bbox", index, [field, box])
            return False
        if box[0] < 0 or box[1] < 0 or box[2] > w or box[3] > h:
            add("reject", "attribute_bbox_outside_image", index, [field, box])
        return True

    retained = []
    seen = set()
    same_boxes = {}
    for index, obj in enumerate(data["layout"]):
        if not isinstance(obj, dict):
            add("reject", "invalid_instance", index)
            continue
        kind = obj.get("type")
        if kind not in ("shape", "line", "image", "icon", "full_text"):
            add("reject", "unknown_type", index, kind)
        box = obj.get("bbox")
        if not vector(box, 4) or box[0] >= box[2] or box[1] >= box[3]:
            add("reject", "invalid_bbox", index, box)
            retained.append(obj)
            continue
        overflow = max(0, -box[0], -box[1], box[2] - w, box[3] - h)
        if overflow > 3:
            add("reject", "large_bbox_overflow", index, [box, overflow])
        elif overflow:
            clipped = [max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3])]
            if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
                add("reject", "bbox_outside_image", index, box)
            else:
                obj["bbox"] = clipped
                add("fix", "clip_bbox_le3px", index, [box, clipped])
        params = obj.get("parameters")
        if params is not None and not isinstance(params, dict):
            add("reject", "invalid_parameters", index)
        elif params:
            geometry(params, index)
            if kind == "line" and params.get("points"):
                paths = params["points"]
                if not isinstance(paths, list):
                    add("reject", "invalid_line_paths", index)
                else:
                    path_seen = set()
                    for path in paths:
                        if (
                            not isinstance(path, list)
                            or len(path) < 2
                            or not all(vector(point, 2) for point in path)
                        ):
                            add("reject", "invalid_line_path", index)
                            continue
                        if len(set(map(tuple, path))) < 2:
                            add("reject", "degenerate_line_path", index)
                        if any(a == b for a, b in zip(path, path[1:])):
                            add("warning", "adjacent_points_preserved", index)
                        signature = tuple(map(tuple, path))
                        if signature in path_seen:
                            add("warning", "repeated_line_segment", index)
                        path_seen.add(signature)
                        distance = max(
                            max(0, box[0] - x, box[1] - y, x - box[2], y - box[3]) for x, y in path
                        )
                        if distance > 3:
                            add("warning", "line_points_outside_bbox", index, distance)
                        # Curves may have external control points, but their endpoints must
                        # still identify the same object as its bbox. Allow modest label noise.
                        endpoint_overflow = max(
                            max(0, box[0] - x, box[1] - y, x - box[2], y - box[3])
                            for x, y in (path[0], path[-1])
                        )
                        if endpoint_overflow > max(20, 0.01 * max(w, h)):
                            add("reject", "line_endpoint_bbox_conflict", index, endpoint_overflow)
        signature = json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=False)
        key = (kind, tuple(obj["bbox"]))
        if signature in seen:
            add("fix", "exact_duplicate_removed", index)
            continue
        if key in same_boxes and same_boxes[key] != signature:
            add("warning", "same_bbox_different_content", index)
        seen.add(signature)
        same_boxes[key] = signature
        retained.append(obj)
    # A bare non-line bbox adds no truth beside its attributed counterpart. Distinct line
    # paths never participate in this rule. Conflicting shape attributes are review quarantine.
    groups = defaultdict(list)
    for index, obj in enumerate(retained):
        if obj.get("type") == "shape" and vector(obj.get("bbox"), 4):
            groups[tuple(obj["bbox"])].append((index, obj))
    drop = set()
    for group in groups.values():
        rich = [(i, obj) for i, obj in group if obj.get("parameters")]
        if not rich:
            continue
        for index, obj in group:
            if not obj.get("parameters") and set(obj) <= {"type", "bbox", "parameters"}:
                drop.add(index)
                add("fix", "bare_shape_duplicate_removed", None, obj["bbox"])
        semantics = {
            json.dumps(
                {k: v for k, v in obj["parameters"].items() if k != "subbbox"},
                sort_keys=True,
                ensure_ascii=False,
            )
            for _, obj in rich
        }
        if len(semantics) > 1:
            add("reject", "conflicting_same_bbox_shape_attributes", None, group[0][1]["bbox"])
    data["layout"] = [obj for i, obj in enumerate(retained) if i not in drop]
    return data, issues


def audit_file(args):
    path, images = args
    raw = path.read_bytes()
    record = {"file": path.name, "sha256": hashlib.sha256(raw).hexdigest(), "issues": []}
    try:
        data = json.loads(raw, object_pairs_hook=unique_keys)
        # Reject non-finite numbers anywhere, not just in geometry fields.
        json.dumps(data, allow_nan=False)
    except (ValueError, UnicodeError) as exc:
        record["issues"].append(dict(level="reject", code="invalid_json", detail=str(exc)))
        return record
    _, record["issues"] = inspect_annotation(data)
    if not isinstance(data, dict) or not isinstance(data.get("layout"), list):
        return record
    record["labels"] = (
        dict(Counter(o.get("type") for o in data.get("layout", []) if isinstance(o, dict)))
        if isinstance(data, dict)
        else {}
    )
    if len(images) != 1:
        record["issues"].append(dict(level="reject", code="missing_or_ambiguous_image"))
    else:
        try:
            with Image.open(images[0]) as image:
                image.load()  # Full decode, not merely a header check.
                size = tuple(data.get("size", []))
                orientation = image.getexif().get(274, 1)
                if image.size != size:
                    if orientation in (5, 6, 7, 8) and image.size[::-1] == size:
                        record["issues"].append(
                            dict(level="warning", code="requires_exif_transpose")
                        )
                    else:
                        record["issues"].append(dict(level="reject", code="image_size_mismatch"))
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            record["issues"].append(
                dict(level="reject", code="image_decode_error", detail=str(exc))
            )
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--snapshot-tag", default="quality_cleaning_v5_10")
    parser.add_argument("--apply", action="store_true", help="Apply a previously generated report")
    args = parser.parse_args()
    root = args.raw_root.resolve()
    source = root / "json"
    if args.apply:
        report = json.loads(args.report.read_text())
        if report["script_sha256"] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
            raise ValueError("Audit script changed; rerun audit")
        records = report["records"]
        if {p.name for p in source.iterdir()} != {r["file"] for r in records}:
            raise ValueError("Input file inventory changed")
        for record in records:
            if (
                hashlib.sha256((source / record["file"]).read_bytes()).hexdigest()
                != record["sha256"]
            ):
                raise ValueError(f"Input changed: {record['file']}")
        if not args.snapshot_tag or any(
            c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in args.snapshot_tag
        ):
            raise ValueError("Invalid snapshot tag")
        stage = root / ("json.stage_" + args.snapshot_tag)
        backup = root / ("json.before_" + args.snapshot_tag)
        quarantine = root / ("json.quarantine_" + args.snapshot_tag)
        if any(p.exists() for p in (stage, backup, quarantine)):
            raise FileExistsError("Cleaning output/backup already exists")
        stage.mkdir()
        quarantine.mkdir()
        for record in records:
            src = source / record["file"]
            issues = record["issues"]
            if any(i["level"] == "reject" for i in issues):
                shutil.copy2(src, quarantine / src.name)
            elif any(i["level"] == "fix" for i in issues):
                cleaned, _ = inspect_annotation(
                    json.loads(src.read_bytes(), object_pairs_hook=unique_keys)
                )
                _, after = inspect_annotation(cleaned)
                if any(i["level"] in ("fix", "reject") for i in after):
                    raise ValueError(f"Non-idempotent cleaning: {src.name}")
                (stage / src.name).write_text(json.dumps(cleaned, ensure_ascii=False) + "\n")
            else:
                shutil.copy2(src, stage / src.name)
        source.rename(backup)
        try:
            stage.rename(source)
        except BaseException:
            backup.rename(source)
            raise
        print(
            json.dumps(
                {
                    "backup": str(backup),
                    "quarantine": str(quarantine),
                    "kept": len(list(source.iterdir())),
                    "quarantined": len(list(quarantine.iterdir())),
                }
            )
        )
        return
    if args.report.exists():
        raise FileExistsError(args.report)
    images = defaultdict(list)
    for path in sorted((root / "images").iterdir()):
        if path.is_file():
            images[path.stem].append(path)
    paths = sorted(source.glob("*.json"))
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for record in pool.map(audit_file, ((p, images[p.stem]) for p in paths), chunksize=10):
            records.append(record)
            if len(records) % 1000 == 0:
                print(f"decoded/audited {len(records)}/{len(paths)}", flush=True)
    counts = Counter(i["code"] for r in records for i in r["issues"])
    summary = {
        "files": len(records),
        "counts": dict(counts),
        "rejected_files": sum(any(i["level"] == "reject" for i in r["issues"]) for r in records),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            {
                "policy": "banana-v5.10-real-quality-v1",
                "summary": summary,
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "records": records,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
