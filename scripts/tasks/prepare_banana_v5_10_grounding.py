"""Frozen real grounding selection, multiscale/JPEG views, SFT, validation and publication.

All raw/eval/work paths are explicit CLI inputs. Raw truth is never modified. Existing
grounding builders own geometry and target conversion; this recipe owns JPEG pairing.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import os
import random
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version
from itertools import zip_longest
from pathlib import Path

import imagehash
from PIL import Image, ImageOps


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def image_index(root):
    result = defaultdict(list)
    for name in sorted(os.listdir(root)):
        if Path(name).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            result[Path(name).stem].append(root / name)
    return result


def fingerprint(path):
    with Image.open(path) as im:
        oriented = ImageOps.exif_transpose(im)
        try:
            phash = str(imagehash.phash(oriented))
        finally:
            if oriented is not im:
                oriented.close()
    return {"id": path.stem, "image": path.name, "sha256": digest(path), "phash": phash}


def source_fingerprint(item):
    annotation, image = item
    result = fingerprint(image)
    data = json.loads(annotation.read_text())
    result.update(
        json=annotation.name,
        json_sha256=digest(annotation),
        positive=any(x["type"] in {"shape", "line", "icon", "image"} for x in data["layout"]),
    )
    return result


def test_matches(source, tests, threshold):
    return [
        t["id"]
        for t in tests
        if source["id"] == t["id"]
        or source["sha256"] == t["sha256"]
        or (int(source["phash"], 16) ^ int(t["phash"], 16)).bit_count() <= threshold
    ]


def select(args, recipe, task):
    selection = task / "selection"
    if selection.exists():
        raise FileExistsError(selection)
    index = image_index(args.raw_root / "images")
    tests = []
    for directory in args.test_image_dir:
        for paths in image_index(directory).values():
            tests.extend(paths)
    manifest = json.loads(args.test_manifest.read_text())
    for entry in manifest["items"]:
        path = args.test_raw_root / entry["image_path"]
        if not path.exists():
            matches = sorted({p for p in tests if p.name == Path(entry["image_path"]).name})
            if not matches or len({digest(p) for p in matches}) != 1:
                raise ValueError(f"Missing/ambiguous canonical test image: {entry['image_path']}")
            path = matches[0]
            with Image.open(path) as im:
                if im.size != (entry["width"], entry["height"]):
                    raise ValueError(f"Canonical test dimensions differ: {path}")
        tests.append(path)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        test_records = list(pool.map(fingerprint, sorted(set(tests)), chunksize=10))
    annotations = sorted((args.raw_root / "json").glob("*.json"))
    inputs = []
    for annotation in annotations:
        paths = index[annotation.stem]
        if len(paths) != 1:
            raise ValueError(f"Image coverage/ambiguity: {annotation.name}")
        inputs.append((annotation, paths[0]))
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for record in pool.map(source_fingerprint, inputs, chunksize=10):
            record["test_matches"] = test_matches(
                record, test_records, recipe["test_phash_distance"]
            )
            records.append(record)
            if len(records) % 1000 == 0:
                print(f"[selection] {len(records)}/{len(inputs)}", flush=True)
    kept = [r for r in records if not r["test_matches"]]
    if args.limit:
        kept = kept[: args.limit]
    selection.mkdir(parents=True)
    (selection / "train.txt").write_text("".join(f"json/{r['json']}\n" for r in kept))
    (selection / "val.txt").write_text("")
    write_json(selection / "sources.json", records)
    write_json(selection / "tests.json", test_records)
    write_json(selection / "recipe.json", recipe)
    metadata = {
        "version": recipe["version"],
        "input_sources": len(records),
        "excluded_test_candidates": sum(bool(r["test_matches"]) for r in records),
        "selected_sources": len(kept),
        "positive_sources": sum(r["positive"] for r in kept),
        "limit": args.limit,
        "code_hashes": {
            p.name: digest(p)
            for p in [
                Path(__file__),
                Path(__file__).with_name("build_grounding_structured.py"),
                Path(__file__).with_name("build_sft_from_structured.py"),
            ]
        },
        "prompt_sha256": digest(recipe["prompt_config"]),
        "environment": {p: version(p) for p in ["Pillow", "numpy", "ImageHash", "scipy"]},
    }
    write_json(selection / "metadata.json", metadata)
    print(json.dumps(metadata), flush=True)


def jpeg_plan(source_id, recipe):
    seed = int.from_bytes(
        hashlib.sha256(f"{recipe['seed']}:jpeg:{source_id}".encode()).digest()[:8]
    )
    rng = random.Random(seed)
    bands = recipe["jpeg_quality_bands"]
    low, high, _ = rng.choices(bands, weights=[b[2] for b in bands], k=1)[0]
    return {
        "name": "jpeg_compression",
        "quality": rng.randint(low, high),
        "quality_band": f"{low}-{high}",
        "subsampling": recipe["jpeg_subsampling"],
        "seed": seed,
    }


def jpeg_row(item):
    row, task, recipe = item
    image_path = (task / "structured" / row["image_path"]).resolve()
    plan = jpeg_plan(row["extra"]["source_json"], recipe)
    out = copy.deepcopy(row)
    out["sample_id"] = row["sample_id"] + "__jpeg"
    name = out["sample_id"] + ".png"
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=plan["quality"], subsampling=plan["subsampling"])
        buf.seek(0)
        with Image.open(buf) as decoded:
            decoded.save(task / "images/train" / name, format="PNG", compress_level=1)
    out["image_path"] = f"../images/train/{name}"
    out["extra"].update(
        view_type="jpeg_full", pixel_augmentation=plan, clean_twin_sample_id=row["sample_id"]
    )
    return out


def add_jpeg(task, recipe, workers):
    path = task / "structured/train.jsonl"
    rows = [json.loads(line) for line in path.open()]
    if any(r["extra"]["view_type"] == "jpeg_full" for r in rows):
        raise ValueError("JPEG views already added")
    groups = defaultdict(list)
    priority = {"continuous_resize_full": 0, "random_padded_full": 1, "full_image": 2}
    for row in rows:
        if row["instances"] and row["extra"]["view_type"] in priority:
            groups[row["extra"]["source_json"]].append(row)
    twins = [
        min(group, key=lambda r: (priority[r["extra"]["view_type"]], r["sample_id"]))
        for _, group in sorted(groups.items())
    ]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for n, row in enumerate(
            pool.map(jpeg_row, ((r, task, recipe) for r in twins), chunksize=8), 1
        ):
            rows.append(row)
            if n % 1000 == 0:
                print(f"[jpeg] {n}/{len(twins)}", flush=True)
    rows.sort(key=lambda r: (r["extra"]["source_json"], r["sample_id"]))
    temporary = path.with_suffix(".pending")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def verify_media(item):
    path, width, height = item
    with Image.open(path) as im:
        im.load()
        if im.size != (width, height):
            raise ValueError(f"Media size mismatch: {path}")
    return path.name, digest(path)


def verify(task, workers):
    from build_sft_from_structured import _build_grounding_target

    selected = set((task / "selection/train.txt").read_text().splitlines())
    source_records = json.loads((task / "selection/sources.json").read_text())
    assert all(not r["test_matches"] for r in source_records if f"json/{r['json']}" in selected)
    ids = set()
    media = []
    views = Counter()
    bands = Counter()
    native = Counter()
    twins = {}
    pending = []
    positive_native = 0
    for split in ("train", "val"):
        with (
            (task / f"structured/{split}.jsonl").open() as a,
            (task / f"sft/{split}.jsonl").open() as b,
        ):
            for sa, sb in zip_longest(a, b):
                assert sa and sb
                row, sft = json.loads(sa), json.loads(sb)
                for instance in row["instances"]:
                    x1, y1, x2, y2 = instance["bbox"]
                    assert 0 <= x1 < x2 <= row["image_width"]
                    assert 0 <= y1 < y2 <= row["image_height"]
                sid = row["sample_id"]
                assert sid == sft["sample_id"] and sid not in ids
                ids.add(sid)
                assert row["extra"]["source_json"] in selected and split == "train"
                target, _ = _build_grounding_target(
                    row["instances"],
                    image_width=row["image_width"],
                    image_height=row["image_height"],
                    num_bins=1000,
                )
                assert target == json.loads(sft["target_text"])
                assert sft["system_prompt"] == sft["user_prompt"] == ""
                p = (task / "structured" / row["image_path"]).resolve()
                assert p == (task / "sft" / sft["image_path"]).resolve()
                assert p.is_relative_to((task / "images").resolve())
                media.append((p, row["image_width"], row["image_height"]))
                view = row["extra"]["view_type"]
                views[view] += 1
                signature = (row["image_width"], row["image_height"], row["instances"])
                twins[sid] = signature
                if view == "full_image":
                    native[row["extra"]["source_json"]] += 1
                    positive_native += bool(row["instances"])
                if view == "jpeg_full":
                    pending.append((row["extra"]["clean_twin_sample_id"], signature))
                    bands[row["extra"]["pixel_augmentation"]["quality_band"]] += 1
    assert set(native) == selected and set(native.values()) == {1}
    assert all(twins[key] == signature for key, signature in pending)
    assert len(pending) == positive_native
    assert len({p for p, _, _ in media}) == len(ids)
    assert len(list((task / "images/train").glob("*.png"))) == len(ids)
    content = hashlib.sha256()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for n, (name, sha) in enumerate(pool.map(verify_media, media, chunksize=8), 1):
            content.update(f"{name}\t{sha}\n".encode())
            if n % 5000 == 0:
                print(f"[verify media] {n}/{len(media)}", flush=True)
    for relative in (
        "structured/train.jsonl",
        "structured/val.jsonl",
        "sft/train.jsonl",
        "sft/val.jsonl",
    ):
        content.update(f"{relative}\t{digest(task / relative)}\n".encode())
    result = {
        "rows": len(ids),
        "sources": len(selected),
        "views": dict(views),
        "jpeg_bands": dict(bands),
        "content_sha256": content.hexdigest(),
        "status": "passed",
    }
    write_json(task / "reports/reproduction_result.json", result)
    (task / "README.md").write_text(
        "# Banana v5.10 grounding_layout\n\n"
        "Raw truth is unchanged. Native, multiscale, padded, density/hard-negative, mild L1 "
        "blur/noise and paired JPEG views; train-only, empty val.\n\n"
        "Recipe: `selection/recipe.json`; source/test fingerprints: `selection/`; "
        "final counts and content hash: `reports/reproduction_result.json`.\n\n"
        f"Train sources: {len(selected)}; structured/SFT/media rows: {len(ids)}.\n\n"
        f"Views: `{json.dumps(dict(views), sort_keys=True)}`.\n\n"
        "JPEG uses one deterministic quality (40-95, weighted bands), 4:2:0, with exact "
        "clean-twin geometry. PNG storage preserves the JPEG artifact. No degradation stacking. "
        "If no clean resize/padding exists, the native clean view is the JPEG twin.\n\n"
        "Test gate excludes ID, exact bytes, and pHash<=6 candidates; candidates are conservative "
        "training exclusions, not changes to raw. Not a full visual-semantic audit.\n"
    )
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--test-image-dir", action="append", required=True, type=Path)
    parser.add_argument("--test-manifest", required=True, type=Path)
    parser.add_argument("--test-raw-root", required=True, type=Path)
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--recipe", type=Path, default=Path("configs/data/preparation/banana_v5_10_grounding.json")
    )
    parser.add_argument("--workers", type=int, default=50)
    parser.add_argument(
        "--limit", type=int, default=0, help="Canary only; never publish a limited build"
    )
    parser.add_argument(
        "--phase",
        choices=("select", "build", "jpeg", "sft", "verify", "publish", "all"),
        default="all",
    )
    args = parser.parse_args()
    recipe = json.loads(args.recipe.read_text())
    task = args.work_root / "grounding_layout"
    stages = (
        ["select", "build", "jpeg", "sft", "verify", "publish"]
        if args.phase == "all"
        else [args.phase]
    )
    for stage in stages:
        print(f"[phase] {stage}", flush=True)
        if stage == "select":
            select(args, recipe, task)
            continue
        if json.loads((task / "selection/recipe.json").read_text()) != recipe:
            raise ValueError("Recipe changed since selection")
        metadata = json.loads((task / "selection/metadata.json").read_text())
        for name, sha in metadata["code_hashes"].items():
            if digest(Path(__file__).with_name(name)) != sha:
                raise ValueError(f"Code changed since selection: {name}")
        if digest(recipe["prompt_config"]) != metadata["prompt_sha256"]:
            raise ValueError("Prompt changed since selection")
        if stage == "build":
            cmd = [
                sys.executable,
                str(Path(__file__).with_name("build_grounding_structured.py")),
                "--raw-root",
                str(args.raw_root),
                "--output-root",
                str(args.work_root),
                "--train-split",
                str(task / "selection/train.txt"),
                "--val-split",
                str(task / "selection/val.txt"),
                "--task",
                "grounding_layout",
                "--workers",
                str(args.workers),
            ]
            for key in (
                "seed",
                "augmentation_profile",
                "min_pixels",
                "max_pixels",
                "clean_resize_views",
                "padded_full_ratio",
                "degraded_resize_ratio",
                "degradation_max_severity",
                "density_crop_ratio",
                "negative_ratio",
            ):
                cmd.extend(["--" + key.replace("_", "-"), str(recipe[key])])
            subprocess.run(cmd, check=True)
        elif stage == "jpeg":
            add_jpeg(task, recipe, args.workers)
        elif stage == "sft":
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("build_sft_from_structured.py")),
                    "--data-root",
                    str(args.work_root),
                    "--task",
                    "grounding_layout",
                    "--workers",
                    str(args.workers),
                    "--prompt-config",
                    f"grounding_layout={recipe['prompt_config']}",
                    "--prompt-variant",
                    f"grounding_layout={recipe['prompt_variant']}",
                ],
                check=True,
            )
        elif stage == "verify":
            verify(task, args.workers)
        elif stage == "publish":
            metadata = json.loads((task / "selection/metadata.json").read_text())
            if metadata["limit"]:
                print("[canary] verified; publication intentionally skipped", flush=True)
                continue
            assert (
                json.loads((task / "reports/reproduction_result.json").read_text())["status"]
                == "passed"
            )
            destination = args.output_root / "grounding_layout"
            backup = args.output_root / "grounding_layout.before_v5_10"
            if backup.exists():
                raise FileExistsError(backup)
            if destination.exists():
                destination.rename(backup)
            try:
                task.rename(destination)
            except BaseException:
                if backup.exists():
                    backup.rename(destination)
                raise
            print(f"[published] {destination}", flush=True)


if __name__ == "__main__":
    main()
