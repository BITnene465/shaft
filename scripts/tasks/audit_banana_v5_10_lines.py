#!/usr/bin/env python3
"""Read-only V10 line inventory, before selection and augmentation calibration."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image

import prepare_gt_standard_v5_7 as contract


def classify(parameters: dict[str, Any]) -> str:
    paths = parameters.get("points")
    if (
        not isinstance(paths, list)
        or not paths
        or any(not isinstance(path, list) or len(path) < 2 for path in paths)
    ):
        return "invalid_structure"
    if len(paths) > 1:
        return "multi_path"
    complex_path = (
        len(paths[0]) > 2
        or parameters.get("line_type") == "curved"
        or parameters.get("line_style") == "shape"
        or parameters.get("begin_arrow") != "none"
        or parameters.get("end_arrow") not in {"none", "triangle"}
        or parameters.get("dash_style") != "solid"
    )
    return "single_complex" if complex_path else "single_simple"


def scan(item: tuple[Path, list[str]]) -> dict[str, Any]:
    root, stems = item
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[Any]] = {}
    hashes = {}
    for stem in stems:
        raw = (root / "gt_standard" / f"{stem}.json").read_bytes()
        hashes[stem] = hashlib.sha256(raw).hexdigest()
        doc = json.loads(raw)
        width, height = doc["size"]
        with Image.open(root / "img" / f"{stem}.png") as image:
            if image.size != (width, height):
                raise ValueError(f"GT/image dimension mismatch: {stem}")
        counts["documents"]["count"] += 1
        for index, obj in enumerate(doc["layout"]):
            if obj.get("type") != "line":
                continue
            p = obj.get("parameters")
            errors = contract._validate_line(p, width=width, height=height)
            p = p if isinstance(p, dict) else {}
            group = classify(p)
            counts["all_groups"][group] += 1
            bbox = obj.get("bbox")
            box_valid = (
                isinstance(bbox, list)
                and len(bbox) == 4
                and all(
                    isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                    for v in bbox
                )
            )
            if not box_valid:
                errors.append("bbox_invalid")
            else:
                x1, y1, x2, y2 = bbox
                if x1 > x2 or y1 > y2:
                    errors.append("bbox_inverted")
                if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                    errors.append("bbox_outside")
                if x1 == x2 or y1 == y2:
                    counts["flags"]["zero_extent_bbox_requires_crop_adapter"] += 1
            if errors:
                counts["invalid_groups"][group] += 1
                for error in errors:
                    counts["issues"][error] += 1
                    examples.setdefault("issue:" + error, [stem, index])
                continue
            counts["valid_groups"][group] += 1
            stratum = contract._line_stratum(p)
            counts["valid_strata"][stratum] += 1
            counts["valid_path_count"][str(len(p["points"]))] += 1
            for key in ("line_type", "line_style", "dash_style", "begin_arrow", "end_arrow"):
                counts[key][str(p[key])] += 1
            examples.setdefault("stratum:" + stratum, [stem, index])
    return {"counts": dict(counts), "examples": examples, "gt_sha256": hashes}


def read_ids(path: Path) -> list[str]:
    if path.suffix == ".json":
        payload = json.loads(path.read_text())
        return sorted(
            {str(item.get("id") or Path(item["image_path"]).stem) for item in payload["items"]}
        )
    values = [Path(line.strip()).stem for line in path.read_text().splitlines() if line.strip()]
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate split identities: {path}")
    return sorted(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-root", type=Path, required=True)
    parser.add_argument("--exclude-manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=50)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.workers < 1:
        raise ValueError("workers must be positive")
    splits = {s: read_ids(args.synthetic_root / f"{s}.txt") for s in ("train", "val")}
    if set(splits["train"]) & set(splits["val"]):
        raise ValueError("Train/val overlap")
    excluded = set().union(*(set(read_ids(p)) for p in args.exclude_manifests))
    result: dict[str, Any] = {
        "version": "banana-v5.10-line-inventory-v1",
        "workers": args.workers,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "validator_sha256": hashlib.sha256(Path(contract.__file__).read_bytes()).hexdigest(),
        "excluded_ids": sorted(excluded),
        "splits": {},
    }
    for split, source_ids in splits.items():
        ids = [s for s in source_ids if split != "train" or s not in excluded]
        totals: dict[str, Counter[str]] = defaultdict(Counter)
        examples, hashes = {}, {}
        batches = [(args.synthetic_root, ids[i : i + 250]) for i in range(0, len(ids), 250)]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for i, partial in enumerate(pool.map(scan, batches), 1):
                for key, count in partial["counts"].items():
                    totals[key].update(count)
                for key, value in partial["examples"].items():
                    examples.setdefault(key, value)
                hashes.update(partial["gt_sha256"])
                if i % 40 == 0:
                    print(f"{split}: {min(i * 250, len(ids))}/{len(ids)}", flush=True)
        result["splits"][split] = {
            "excluded_sources": len(source_ids) - len(ids),
            "counts": dict(totals),
            "examples": examples,
            "gt_sha256": hashes,
            "split_sha256": hashlib.sha256(
                (args.synthetic_root / f"{split}.txt").read_bytes()
            ).hexdigest(),
        }
        print(
            json.dumps(
                {
                    split: {
                        k: dict(v) for k, v in totals.items() if k not in {"valid_strata", "issues"}
                    }
                }
            ),
            flush=True,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    contract._atomic_write_text(args.output, json.dumps(result, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
