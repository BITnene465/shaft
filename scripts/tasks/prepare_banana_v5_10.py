#!/usr/bin/env python3
"""Portable v5.10 preparation. Currently materializes the agreed shape stage only."""

from __future__ import annotations

import argparse
import colorsys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import shutil
import tempfile

from PIL import Image, features

import build_context_reconstruction_sft as builder
import prepare_gt_standard_v5_7 as source_contract


REPO = Path(__file__).resolve().parents[2]
TASK = "shape_context_reconstruction"
FORMULATIONS = ("appearance", "geometry", "reconstruction")


def dump(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    builder._atomic_write_text(path, dump(value) + "\n")


def band(value, edges):
    return next((f"<{edge}" for edge in edges if value < edge), f">={edges[-1]}")


def quotas(capacities, target):
    """Sorted floating-point reductions avoid PYTHONHASHSEED-sensitive quotas."""
    remaining = min(target, sum(capacities.values()))
    result = dict.fromkeys(sorted(capacities), 0)
    while remaining:
        active = [k for k in result if result[k] < capacities[k]]
        weights = {k: math.sqrt(capacities[k]) for k in active}
        total = math.fsum(weights.values())
        desired = {k: remaining * weights[k] / total for k in active}
        takes = {k: min(capacities[k] - result[k], int(desired[k])) for k in active}
        if not any(takes.values()):
            for k in sorted(active, key=lambda k: (-desired[k], k))[:remaining]:
                takes[k] = 1
        for k, take in takes.items():
            result[k] += take
            remaining -= take
    return result


def appearance_stratum(p, bbox):
    x1, y1, x2, y2 = bbox
    width, height = x2 - x1, y2 - y1
    corners = Counter(c.get("type") for c in p.get("corners", []))
    fills = p.get("fill", {})
    fills = fills if isinstance(fills, list) else [fills]
    colors = [f.get("color", "") for f in fills]
    rgb = [
        tuple(int(c[i : i + 2], 16) / 255 for i in (1, 3, 5))
        for c in colors
        if isinstance(c, str) and len(c) == 7 and c.startswith("#")
    ]
    contrast = max((math.dist(a, b) / math.sqrt(3) for a in rgb for b in rgb), default=0)
    luminance = [sum(a * b for a, b in zip(c, (0.2126, 0.7152, 0.0722))) for c in rgb]
    hues = [
        "neutral"
        if colorsys.rgb_to_hsv(*color)[1] < 0.15
        else str(int(colorsys.rgb_to_hsv(*color)[0] * 12) % 12)
        for color in rgb
    ]
    return [
        sorted(corners.items()),
        [c.get("type") for c in p.get("corners", [])],
        [f.get("type") for f in fills],
        hues,
        p.get("border", {}).get("type"),
        p.get("border", {}).get("style"),
        p.get("effect", {}).get("type"),
        band(width / height, [0.25, 0.5, 1 / 1.2, 1.2, 2, 4]),
        band(min(width, height), [16, 32, 64, 128]),
        band(contrast, [0.15, 0.4]),
        [band(v, [0.3, 0.7]) for v in luminance],
    ]


def card_stratum(p, bbox):
    x1, y1, x2, y2 = bbox
    axes, positions = [], []
    for split in p.get("splits", []):
        points = [
            point
            for corner in split.get("split_corners", [])
            for key, point in corner.items()
            if key in {"point", "start", "mid", "end"}
        ]
        if len(points) < 2:
            axes.append("unknown")
            continue
        dx = max(v[0] for v in points) - min(v[0] for v in points)
        dy = max(v[1] for v in points) - min(v[1] for v in points)
        axis = (
            "stacked"
            if dy <= max(1, dx * 0.02) and dx > dy
            else "side_by_side"
            if dx <= max(1, dy * 0.02) and dy > dx
            else "oblique"
        )
        axes.append(axis)
        positions.append(
            (sum(v[1] for v in points) / len(points) - y1) / (y2 - y1)
            if axis == "stacked"
            else (sum(v[0] for v in points) / len(points) - x1) / (x2 - x1)
        )
    layout = axes[0] if axes and len(set(axes)) == 1 else "mixed"
    # Preserve unusual layouts too: never discard oblique/mixed as merely inconvenient.
    protected = len(p.get("fill", [])) >= 3 or layout != "stacked"
    stratum = [
        layout,
        len(p.get("fill", [])),
        band(min(positions), [0.1, 0.2, 0.3, 0.4, 0.5]) if positions else "unknown",
        [(s.get("type"), s.get("style")) for s in p.get("splits", [])],
        appearance_stratum(p, bbox),
    ]
    return protected, dump(stratum)


def scan_source(item):
    root, stem = item
    path = root / "gt_standard" / f"{stem}.json"
    image_path = root / "img" / f"{stem}.png"
    data = json.loads(path.read_text())
    width, height = data["size"]
    with Image.open(image_path) as image:
        if image.size != (width, height):
            raise ValueError(f"GT/image mismatch: {stem}; do not rescale GT implicitly")
    rows, issues = [], Counter()
    for index, obj in enumerate(data["layout"]):
        if obj.get("type") != "shape":
            continue
        p = obj.get("parameters")
        bbox = source_contract._bbox(obj.get("bbox"), width=width, height=height)
        errors = source_contract._validate_shape(p, width=width, height=height)
        if bbox is None or errors:
            issues.update(errors or ["invalid_bbox"])
            issues["excluded_instances"] += 1
            continue
        shape_type = p["shape_type"]
        protected, stratum = True, shape_type
        if shape_type == "rectangle":
            protected, stratum = False, dump(appearance_stratum(p, bbox))
        elif shape_type == "card":
            protected, stratum = card_stratum(p, bbox)
        rows.append((stem, index, bbox, shape_type, protected, stratum))
    hashes = {f"gt_standard/{stem}.json": sha(path), f"img/{stem}.png": sha(image_path)}
    return rows, issues, hashes


def read_ids(path):
    if path.suffix == ".json":
        return sorted(builder._read_excluded_ids(path))
    ids = [Path(line.strip()).stem for line in path.read_text().splitlines() if line.strip()]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate split ids: {path}")
    if any("/" in stem or "\\" in stem or stem in {".", ".."} for stem in ids):
        raise ValueError(f"Unsafe identity: {path}")
    return sorted(ids)


def environment():
    return {
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in ("Pillow", "numpy")},
        "codecs": {name: features.version(name) for name in ("jpg", "zlib", "libjpeg_turbo")},
    }


def prepare(args, recipe, work):
    root = args.synthetic_root
    train, val = read_ids(root / "train.txt"), read_ids(root / "val.txt")
    if set(train) & set(val):
        raise ValueError("Synthetic train/val overlap")
    excluded = set().union(*(set(read_ids(path)) for path in args.exclude_manifests))
    train = [stem for stem in train if stem not in excluded]
    chosen, common = [], defaultdict(list)
    issues, available, input_hashes = Counter(), Counter(), {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = pool.map(scan_source, ((root, stem) for stem in train), chunksize=32)
        for index, (rows, errors, hashes) in enumerate(results, 1):
            issues.update(errors)
            input_hashes.update(hashes)
            for row in rows:
                available[row[3]] += 1
                if row[4]:
                    chosen.append(row)
                else:
                    common[(row[3], row[5])].append(row)
            if index % 5000 == 0:
                print(f"scan {index}/{len(train)} sources", flush=True)
    seed = recipe["seed"]

    def score(row):
        return hashlib.sha256(f"{seed}:shape:{row[0]}:{row[1]}".encode()).digest()

    for shape_type, target in (
        ("rectangle", recipe["shape"]["rectangle_target"]),
        ("card", recipe["shape"]["common_card_target"]),
    ):
        groups = {key[1]: rows for key, rows in common.items() if key[0] == shape_type}
        allocation = quotas({key: len(rows) for key, rows in groups.items()}, target)
        for key in sorted(groups):
            chosen.extend(sorted(groups[key], key=score)[: allocation[key]])
    chosen.sort(key=lambda row: (row[0], row[1]))
    selection = work / "shape.selection.jsonl"
    with selection.open("w") as stream:
        for stem, index, bbox, shape_type, protected, stratum in chosen:
            row = source_contract._selection_row(
                source_contract.Candidate(stem, index, "shape", bbox, stratum)
            )
            row["extra"]["sampling"] = {
                "shape_type": shape_type,
                "protected": protected,
                "stratum": stratum,
            }
            stream.write(dump(row) + "\n")
    input_hashes["train.txt"] = sha(root / "train.txt")
    input_hashes["val.txt"] = sha(root / "val.txt")
    write_json(work / "input_checksums.json", input_hashes)
    files = [
        Path(__file__),
        REPO / "scripts/tasks/build_context_reconstruction_sft.py",
        REPO / "scripts/tasks/prepare_gt_standard_v5_7.py",
        REPO / recipe["shape"]["prompt_pool"],
    ]
    files += sorted((REPO / "src/shaft").rglob("*.py"))
    lock = {
        "recipe": recipe,
        "environment": environment(),
        "code_sha256": {str(p.relative_to(REPO)): sha(p) for p in files},
        "input_checksums_sha256": sha(work / "input_checksums.json"),
        "selection_sha256": sha(selection),
        "excluded_ids": sorted(excluded),
        "exclusion_manifest_sha256": sorted(sha(p) for p in args.exclude_manifests),
        "train_source_count": len(train),
        "held_out_source_count": len(val),
        "available": dict(available),
        "selected": dict(Counter(r[3] for r in chosen)),
        "quality_exclusions": dict(issues),
    }
    write_json(work / "reproduction.lock.json", lock)
    print(dump({"selected": lock["selected"], "issues": dict(issues)}), flush=True)


def check_media(item):
    root, relative, size = item
    path = root / relative
    with Image.open(path) as image:
        image.load()
        if list(image.size) != size:
            raise ValueError(f"Output media size mismatch: {relative}")
    return relative, sha(path)


def verify_task(root, workers):
    media, ids = [], set()
    with ExitStack() as stack:
        structured = stack.enter_context((root / "structured/train.jsonl").open())
        stores = [
            stack.enter_context((root / f"sft/formulations/{f}/train.jsonl").open())
            for f in FORMULATIONS
        ]
        for line in structured:
            row = json.loads(line)
            if row["sample_id"] in ids:
                raise ValueError("Duplicate output identity")
            ids.add(row["sample_id"])
            source = row["instances"][0]["parameters"]
            siblings = []
            for formulation, stream in zip(FORMULATIONS, stores):
                sft = json.loads(next(stream))
                target = json.loads(sft.pop("target_text"))
                expected = {
                    "type": "shape",
                    "parameters": builder._formulation_parameters("shape", formulation, source),
                }
                if target != expected or sft["sample_id"] != row["sample_id"]:
                    raise ValueError("Target/identity mismatch")
                path = (root / f"sft/formulations/{formulation}" / sft["image_path"]).resolve()
                if path != (root / "structured" / row["image_path"]).resolve():
                    raise ValueError("SFT media mismatch")
                siblings.append(sft)
            if not all(s == siblings[0] for s in siblings):
                raise ValueError("Formulation identity mismatch")
            relative = (
                (root / "structured" / row["image_path"]).resolve().relative_to(root.resolve())
            )
            media.append((root, str(relative), [row["image_width"], row["image_height"]]))
        if any(stream.read(1) for stream in stores):
            raise ValueError("Extra formulation rows")
    hashes = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, (relative, digest) in enumerate(pool.map(check_media, media, chunksize=32), 1):
            hashes[relative] = digest
            if index % 50000 == 0:
                print(f"verify media {index}/{len(media)}", flush=True)
    for folder in ("structured", "sft", "selection"):
        for path in sorted((root / folder).rglob("*.jsonl")):
            hashes[str(path.relative_to(root))] = sha(path)
    write_json(root / "reports/content_checksums.json", hashes)
    return {"rows": len(ids), "content_sha256": hashlib.sha256(dump(hashes).encode()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recipe", type=Path, default=REPO / "configs/data/preparation/banana_v5_10.json"
    )
    parser.add_argument("--synthetic-root", type=Path, required=True)
    parser.add_argument("--exclude-manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data"))
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=50)
    parser.add_argument("--stage", choices=("prepare", "build", "all"), default="all")
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    recipe = json.loads(args.recipe.read_text())
    if args.workers < 1:
        raise ValueError("workers must be positive")
    shape = recipe["shape"]
    if (
        shape["augmentation"] != "synthetic_realism_v1"
        or not shape["keep_all_other_types"]
        or not shape["keep_all_card_side_by_side_or_multiregion"]
    ):
        raise ValueError("Unsupported change to the frozen v5.10 shape contract")
    if min(shape["rectangle_target"], shape["common_card_target"]) < 0:
        raise ValueError("Sampling counts must be nonnegative")
    work = args.work_root.resolve()
    work.mkdir(parents=True, exist_ok=True)
    if args.stage in {"prepare", "all"}:
        if (work / "reproduction.lock.json").exists():
            raise FileExistsError(
                "Preparation already frozen; use --stage build or a new work root"
            )
        prepare(args, recipe, work)
    if args.stage == "prepare":
        return
    lock = json.loads((work / "reproduction.lock.json").read_text())
    if lock["recipe"] != recipe or lock["environment"] != environment():
        raise ValueError("Frozen recipe/environment changed")
    if any(sha(REPO / path) != digest for path, digest in lock["code_sha256"].items()):
        raise ValueError("Builder code/prompt changed since preparation")
    if sha(work / "shape.selection.jsonl") != lock["selection_sha256"]:
        raise ValueError("Frozen selection changed")
    # Recheck source hashes even when resuming on a different machine.
    checksums = json.loads((work / "input_checksums.json").read_text())
    if sha(work / "input_checksums.json") != lock["input_checksums_sha256"]:
        raise ValueError("Input hash manifest changed")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        actual = pool.map(sha, (args.synthetic_root / path for path in checksums), chunksize=32)
        if any(value != expected for value, expected in zip(actual, checksums.values())):
            raise ValueError("Source snapshot changed")
    destination = args.output_root.resolve() / TASK
    if destination.exists() and not args.replace:
        raise FileExistsError(f"Pass --replace to replace {destination}")
    staging = Path(tempfile.mkdtemp(prefix="shape-build-", dir=work))
    shape = recipe["shape"]
    spec = builder.TaskSpec(
        TASK,
        "shape",
        work / "shape.selection.jsonl",
        args.synthetic_root,
        REPO / shape["prompt_pool"],
        "synthetic",
        source_dataset_id=recipe["synthetic_snapshot"],
    )
    print("Building shared crops and all three formulations", flush=True)
    builder._build_task(
        spec,
        output_root=staging,
        workers=args.workers,
        chunksize=8,
        clean=False,
        seed=recipe["seed"],
        min_crop_size=shape["min_crop_size"],
        max_aspect_ratio=shape["max_aspect_ratio"],
        png_compress_level=shape["png_compress_level"],
        excluded_ids=set(),
        limit=None,
        shape_attribute_max_rectangle_fraction=1.0,
        preflight_only=False,
    )
    built = staging / TASK
    print("Validating complete formulation alignment and media", flush=True)
    result = verify_task(built, args.workers)
    for name in ("reproduction.lock.json", "input_checksums.json", "shape.selection.jsonl"):
        shutil.copyfile(work / name, built / "reports" / name)
    write_json(built / "reports/reproduction_result.json", result)
    destination.parent.mkdir(parents=True, exist_ok=True)
    backup = None
    if destination.exists():
        backup = Path(tempfile.mkdtemp(prefix=f"{TASK}.previous-", dir=destination.parent))
        backup.rmdir()
        destination.rename(backup)
    try:
        built.rename(destination)
    except BaseException:
        if backup is not None:
            backup.rename(destination)
        raise
    print(dump({"published": str(destination), "backup": str(backup), **result}), flush=True)


if __name__ == "__main__":
    main()
