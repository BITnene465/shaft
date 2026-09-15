#!/usr/bin/env python3
"""Reproducible, complete-only real shape cohort from an explicit filename whitelist."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from dataclasses import replace
from functools import lru_cache
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import shutil
import random

from PIL import Image, ImageOps

import prepare_banana_v5_10 as S
import real_shape_contract as C

B = S.builder


def check_source(root: Path, source: dict) -> dict:
    if (
        S.sha(root / "json" / source["json"]) != source["json_sha256"]
        or S.sha(root / "images" / source["image"]) != source["sha256"]
    ):
        raise ValueError(f"Input changed since test isolation: {source['json']}")
    return json.loads((root / "json" / source["json"]).read_text())


def select_source(job: tuple) -> tuple:
    root, source = job
    doc = check_source(root, source)
    width, height = doc["size"]
    selected, rejected = [], []
    counts = Counter()
    for index, obj in enumerate(doc["layout"]):
        if obj.get("type") != "shape":
            continue
        counts["raw_shape_instances"] += 1
        try:
            if C.V._bbox(obj.get("bbox"), width=width, height=height) is None:
                raise ValueError("shape.invalid_bbox")
            p = C.normalize(obj.get("parameters"), width, height)
        except ValueError as error:
            rejected.append({"source_json": source["json"], "index": index, "reason": str(error)})
            continue
        selected.append(
            {
                "source_json": f"json/{source['json']}",
                "source_image": f"images/{source['image']}",
                "stem": source["id"],
                "index": index,
                "bbox": obj["bbox"],
                "shape_type": p["shape_type"],
            }
        )
        counts["selected:" + p["shape_type"]] += 1
    return selected, rejected, counts


def validate_target(p: dict) -> None:
    issues = C.V._validate_shape(p, width=999, height=999)
    if issues:
        raise ValueError("quantization.schema:" + ";".join(issues))
    C.validate_geometry(p)


def build_source(job: tuple) -> tuple:
    root, source, selections, config = job
    doc = check_source(root, source)
    with Image.open(root / "images" / source["image"]) as opened:
        candidate = opened if list(opened.size) == doc["size"] else ImageOps.exif_transpose(opened)
        try:
            if list(candidate.size) != doc["size"]:
                raise ValueError("Image/GT dimensions mismatch")
            image = candidate.convert("RGB")
        finally:
            if candidate is not opened:
                candidate.close()
    rows, rejected = [], []
    try:
        for selected in selections:
            obj = doc["layout"][selected["index"]]
            if obj["type"] != "shape" or obj["bbox"] != selected["bbox"]:
                raise ValueError("Frozen selected identity changed")
            p = C.normalize(obj["parameters"], image.width, image.height)
            selection = B.Selection(
                f"real__{selected['stem']}__shape_{selected['index']:05d}",
                selected["stem"],
                selected["index"],
                tuple(obj["bbox"]),
                selected["source_image"],
                selected["source_json"],
            )
            kwargs = dict(
                selection=selection,
                source_image=image,
                source_instance_index=selected["index"],
                source_bbox=tuple(obj["bbox"]),
                source_parameters=p,
                source_layout=doc["layout"],
                image_width=image.width,
                image_height=image.height,
            )
            try:
                # Reject bad quantized geometry before writing any output media.
                preview, _, _ = B._build_row(config=replace(config, write_images=False), **kwargs)
                projected = json.loads(preview)
                validate_target(projected["instances"][0]["parameters"])
                left, top, right, bottom = projected["extra"]["crop_box"]
                if not all(
                    left <= x <= right and top <= y <= bottom for x, y in C.geometry_points(p)
                ):
                    raise ValueError("geometry.crop_clips_source")
            except ValueError as error:
                rejected.append(
                    {
                        "source_json": selected["source_json"],
                        "index": selected["index"],
                        "reason": str(error),
                    }
                )
                continue
            structured, sfts, _ = B._build_row(config=config, **kwargs)
            row = json.loads(structured)
            row["extra"]["normalization"] = "complete-shape-clockwise-v1"
            row["extra"]["cohort"] = "real"
            materialized = []
            for formulation, text in sfts:
                sft = json.loads(text)
                sft["extra"]["source_type"] = "human_real_context_shape"
                sft["extra"]["structured_extra"] = row["extra"]
                materialized.append((formulation, S.dump(sft)))
            rows.append((S.dump(row), materialized))
    finally:
        image.close()
    return rows, rejected


def prepare(args: argparse.Namespace, recipe: dict) -> None:
    work = args.work_root
    work.mkdir(parents=True, exist_ok=False)
    gate = args.grounding_root / "selection"
    sources = json.loads((gate / "sources.json").read_text())
    by_name = {r["json"]: r for r in sources}
    if len(by_name) != len(sources):
        raise ValueError("Duplicate source identities")
    split = set((gate / "train.txt").read_text().splitlines())
    if split != {f"json/{r['json']}" for r in sources if not r["test_matches"]}:
        raise ValueError("Grounding test gate and train split disagree")
    names = [s.strip() for s in args.filename_list.read_text().splitlines() if s.strip()]
    if len(names) != len(set(names)) or any(
        Path(s).name != s or not s.endswith(".json") for s in names
    ):
        raise ValueError("Whitelist must contain unique flat JSON filenames")
    missing, excluded, active = [], [], []
    for name in sorted(names):
        if not (args.raw_root / "json" / name).exists():
            if name in by_name:
                raise ValueError(f"Source disappeared since content isolation: {name}")
            missing.append(name)
        elif name not in by_name:
            raise ValueError(f"Source absent from content-isolation gate: {name}")
        elif by_name[name]["test_matches"]:
            excluded.append(name)
        else:
            active.append(by_name[name])
    selected, rejected, counts = [], [], Counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, (rows, failures, stats) in enumerate(
            pool.map(select_source, [(args.raw_root, r) for r in active], chunksize=8), 1
        ):
            selected.extend(rows)
            rejected.extend(failures)
            counts.update(stats)
            if i % 200 == 0:
                print(f"select sources {i}/{len(active)} accepted={len(selected)}", flush=True)
    if args.limit:
        selected = selected[: args.limit]
    if not selected:
        raise ValueError("No complete shapes selected")
    selected_names = {Path(r["source_json"]).name for r in selected}
    frozen = [r for r in active if r["json"] in selected_names]
    B._atomic_write_text(work / "selection.jsonl", "".join(S.dump(r) + "\n" for r in selected))
    S.write_json(work / "sources.json", frozen)
    S.write_json(
        work / "selection_report.json",
        {
            "whitelist_files": len(names),
            "missing_in_active_raw": missing,
            "excluded_test_candidates": excluded,
            "eligible_files": len(active),
            "counts": counts,
            "rejected_instances": rejected,
            "selected": len(selected),
        },
    )
    shutil.copyfile(args.filename_list, work / "filename_list.txt")
    for name in ("sources.json", "tests.json", "train.txt", "metadata.json"):
        shutil.copyfile(gate / name, work / f"grounding_{name}")
    code = [
        Path(__file__),
        Path(C.__file__),
        Path(C.V.__file__),
        Path(B.__file__),
        Path(S.__file__),
        S.REPO / recipe["prompt_pool"],
        *sorted((S.REPO / "src/shaft").rglob("*.py")),
    ]
    S.write_json(
        work / "reproduction.lock.json",
        {
            "recipe": recipe,
            "environment": S.environment(),
            "limit": args.limit,
            "code_sha256": {str(p.resolve().relative_to(S.REPO)): S.sha(p) for p in code},
            "artifacts_sha256": {p.name: S.sha(p) for p in sorted(work.iterdir()) if p.is_file()},
        },
    )
    print(
        S.dump(
            {
                "selected": len(selected),
                "rejected": len(rejected),
                "counts": counts,
                "missing_files": len(missing),
                "test_excluded_files": len(excluded),
            }
        ),
        flush=True,
    )


def jpeg_twin(job: tuple) -> tuple:
    stage, clean_text, clean_sfts, recipe = job
    row = json.loads(clean_text)
    original_id = row["sample_id"]
    quality = random.Random(
        hashlib.sha256(f"{recipe['seed']}:{original_id}:jpeg".encode()).digest()
    ).randint(*recipe["additional_jpeg"]["quality"])
    sample_id = original_id + "__jpeg"
    relative = f"images/train/{B._image_shard(sample_id)}/{sample_id}.png"
    output = stage / relative
    output.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(stage / "structured" / row["image_path"]) as clean:
        with io.BytesIO() as buffer:
            clean.save(
                buffer,
                format="JPEG",
                quality=quality,
                subsampling=recipe["additional_jpeg"]["subsampling"],
            )
            buffer.seek(0)
            with Image.open(buffer) as decoded:
                decoded.save(output, format="PNG", compress_level=recipe["png_compress_level"])
    row["sample_id"], row["image_path"] = sample_id, "../" + relative
    row["extra"]["clean_sample_id"] = original_id
    row["extra"]["pixel_augmentation"] = {
        "profile": recipe["version"] + ":jpeg",
        "operations": [
            {
                "name": "jpeg",
                "quality": quality,
                "subsampling": recipe["additional_jpeg"]["subsampling"],
            }
        ],
    }
    sfts = []
    for formulation, text in clean_sfts:
        record = json.loads(text)
        record["sample_id"], record["image_path"] = sample_id, "../../../" + relative
        record["extra"]["structured_extra"] = row["extra"]
        sfts.append((formulation, S.dump(record)))
    return S.dump(row), sfts


def verify_source_targets(stage: Path, root: Path, recipe: dict) -> dict:
    @lru_cache(maxsize=32)
    def doc(path: str) -> dict:
        return json.loads((root / path).read_text())

    counts, media, signatures = Counter(), set(), {}
    for text in (stage / "structured/train.jsonl").open():
        row = json.loads(text)
        extra = row["extra"]
        raw = doc(extra["source_json"])
        p = C.normalize(raw["layout"][extra["source_instance_index"]]["parameters"], *raw["size"])
        crop = extra["crop_box"]
        if not all(
            crop[0] <= x <= crop[2] and crop[1] <= y <= crop[3] for x, y in C.geometry_points(p)
        ):
            raise ValueError("Published crop clips source geometry")
        expected = B._target_parameters("shape", p, crop_box=tuple(crop))
        if expected != row["instances"][0]["parameters"]:
            raise ValueError("Raw/structured geometry mismatch")
        validate_target(expected)
        if [crop[2] - crop[0], crop[3] - crop[1]] != [row["image_width"], row["image_height"]]:
            raise ValueError("Crop dimension mismatch")
        if extra.get("clean_sample_id"):
            clean = signatures.get(extra["clean_sample_id"])
            twin = deepcopy(row)
            twin.pop("sample_id")
            twin.pop("image_path")
            twin["extra"].pop("clean_sample_id")
            twin["extra"]["pixel_augmentation"] = {"profile": "none", "operations": []}
            if clean != S.dump(twin):
                raise ValueError("JPEG twin changed geometry or metadata")
            ops = extra["pixel_augmentation"]["operations"]
            policy = recipe["additional_jpeg"]
            if (
                len(ops) != 1
                or ops[0]["name"] != "jpeg"
                or ops[0]["subsampling"] != policy["subsampling"]
                or not policy["quality"][0] <= ops[0]["quality"] <= policy["quality"][1]
            ):
                raise ValueError("Invalid JPEG twin policy")
        else:
            if extra["pixel_augmentation"] != {"profile": "none", "operations": []}:
                raise ValueError("Clean sample unexpectedly pixel-augmented")
            clean = deepcopy(row)
            clean.pop("sample_id")
            clean.pop("image_path")
            signatures[row["sample_id"]] = S.dump(clean)
        path = (stage / "structured" / row["image_path"]).resolve().relative_to(stage.resolve())
        if str(path) in media:
            raise ValueError("Duplicate media identity")
        media.add(str(path))
        counts[expected["shape_type"]] += 1
    if media != {str(p.relative_to(stage)) for p in (stage / "images").rglob("*.png")}:
        raise ValueError("Missing or unreferenced media")
    if any(p.stat().st_size for p in stage.rglob("val.jsonl")):
        raise ValueError("Nonempty train-only validation")
    return dict(counts)


def build(args: argparse.Namespace, recipe: dict) -> None:
    work = args.work_root
    lock = json.loads((work / "reproduction.lock.json").read_text())
    if lock["recipe"] != recipe or lock["environment"] != S.environment():
        raise ValueError("Frozen recipe/environment changed")
    for path, digest in lock["code_sha256"].items():
        if S.sha(S.REPO / path) != digest:
            raise ValueError(f"Frozen code changed: {path}")
    for path, digest in lock["artifacts_sha256"].items():
        if S.sha(work / path) != digest:
            raise ValueError(f"Frozen artifact changed: {path}")
    destination = args.output_root.resolve() / recipe["dataset"]
    if destination.exists() and not lock["limit"]:
        raise FileExistsError(f"Refusing to overwrite existing real cohort: {destination}")
    stage = work / recipe["dataset"]
    stage.mkdir(exist_ok=False)
    for folder in (
        "selection",
        "structured",
        "reports",
        *[f"sft/formulations/{f}" for f in recipe["formulations"]],
    ):
        (stage / folder).mkdir(parents=True)
        if folder != "reports":
            (stage / folder / "val.jsonl").touch()
    shutil.copyfile(work / "selection.jsonl", stage / "selection/train.jsonl")
    sources = json.loads((work / "sources.json").read_text())
    groups = defaultdict(list)
    for text in (work / "selection.jsonl").open():
        r = json.loads(text)
        groups[Path(r["source_json"]).name].append(r)
    pool_id, schema, formulations = B._prompt_contract(S.REPO / recipe["prompt_pool"])
    if list(formulations) != recipe["formulations"]:
        raise ValueError("Frozen prompt formulation mismatch")
    spec = B.TaskSpec(
        recipe["dataset"],
        "shape",
        work / "selection.jsonl",
        args.raw_root,
        S.REPO / recipe["prompt_pool"],
        "real",
        source_dataset_id=recipe["version"],
        eligible_formulations=formulations,
    )
    config = B.WorkerConfig(
        spec,
        stage,
        pool_id,
        schema,
        recipe["seed"],
        recipe["min_crop_size"],
        recipe["max_aspect_ratio"],
        recipe["png_compress_level"],
        formulations,
    )
    rejected, count, clean_rows = [], 0, []
    with ExitStack() as stack:
        structured = stack.enter_context((stage / "structured/train.jsonl").open("w"))
        stores = {
            f: stack.enter_context((stage / f"sft/formulations/{f}/train.jsonl").open("w"))
            for f in formulations
        }
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for i, (rows, failures) in enumerate(
                pool.map(
                    build_source,
                    [(args.raw_root, s, groups[s["json"]], config) for s in sources],
                    chunksize=4,
                ),
                1,
            ):
                rejected.extend(failures)
                clean_rows.extend(rows)
                for row, sfts in rows:
                    structured.write(row + "\n")
                    for f, text in sfts:
                        stores[f].write(text + "\n")
                    count += 1
                if i % 100 == 0:
                    print(f"build sources {i}/{len(sources)} rows={count}", flush=True)
        jpeg_count = int(count * recipe["additional_jpeg"]["fraction"])
        chosen = sorted(
            clean_rows,
            key=lambda pair: hashlib.sha256(
                f"{recipe['seed']}:{json.loads(pair[0])['sample_id']}".encode()
            ).digest(),
        )[:jpeg_count]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for row, sfts in pool.map(
                jpeg_twin, [(stage, r, s, recipe) for r, s in chosen], chunksize=8
            ):
                structured.write(row + "\n")
                for f, text in sfts:
                    stores[f].write(text + "\n")
        print(f"JPEG twins {jpeg_count}; clean={count}", flush=True)
    if count + len(rejected) != sum(map(len, groups.values())) or not count:
        raise ValueError("Unexpected build cardinality")
    S.write_json(stage / "reports/rejected_views.json", rejected)
    print("Recomputing every target from raw; validating all media and formulations", flush=True)
    types = verify_source_targets(stage, args.raw_root, recipe)
    result = S.verify_task(stage, args.workers, label="shape", formulations=formulations)
    # Detect source mutation even across a long build/validation run.
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        checks = [
            (args.raw_root / folder / s[key], s[digest])
            for s in sources
            for folder, key, digest in (
                ("json", "json", "json_sha256"),
                ("images", "image", "sha256"),
            )
        ]
        if any(
            actual != expected
            for actual, (_, expected) in zip(
                pool.map(S.sha, [p for p, _ in checks], chunksize=8), checks
            )
        ):
            raise ValueError("Source changed during build")
    result.update(
        shape_types=types,
        rejected_views=len(rejected),
        clean_rows=count,
        jpeg_rows=jpeg_count,
        total_sft_rows=(count + jpeg_count) * len(formulations),
    )
    for p in work.iterdir():
        if p.is_file():
            shutil.copyfile(p, stage / "reports" / p.name)
    S.write_json(stage / "reports/reproduction_result.json", result)
    (stage / "README.md").write_text(
        f"# {recipe['version']}\n\nComplete-only real shape cohort; {count} shared clean crops "
        f"plus {jpeg_count} JPEG twins; {(count + jpeg_count) * len(formulations)} SFT rows. "
        f"Formulations: {list(formulations)}.\n\n"
        "Same existing shape v5.8 prompt pool; independent source for later mixing. "
        "No raw modification. Extra20% JPEG60–90/4:4:4 twins only; no blur/noise/resize. "
        "Train-only, empty validation.\n\n"
        "Input list, source/test hashes, code, recipe and environment are in reports/reproduction.lock.json "
        "and its frozen artifacts. See scripts/tasks/banana_v5_10_real_shape.md for portable rebuilding.\n"
    )
    if lock["limit"]:
        print(S.dump({"canary": str(stage), **result}), flush=True)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        stage.rename(destination)
        print(S.dump({"published": str(destination), **result}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("raw-root", "filename-list", "grounding-root", "work-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--recipe",
        type=Path,
        default=S.REPO / "configs/data/preparation/banana_v5_10_real_shape.json",
    )
    parser.add_argument("--output-root", type=Path, default=Path("data"))
    parser.add_argument("--workers", type=int, default=50)
    parser.add_argument("--limit", type=int, default=0, help="Small canary; cannot publish")
    parser.add_argument("--stage", choices=("prepare", "build", "all"), default="all")
    args = parser.parse_args()
    recipe = json.loads(args.recipe.read_text())
    if (
        args.workers < 1
        or args.limit < 0
        or recipe["normalization"] != "complete-shape-clockwise-v1"
        or recipe["pixel_augmentation"] != "none"
        or not 0 <= recipe["additional_jpeg"]["fraction"] <= 1
        or not 1
        <= recipe["additional_jpeg"]["quality"][0]
        <= recipe["additional_jpeg"]["quality"][1]
        <= 100
        or recipe["additional_jpeg"]["subsampling"] != 0
        or recipe["min_crop_size"] < 1
        or recipe["max_aspect_ratio"] < 1
        or not 0 <= recipe["png_compress_level"] <= 9
        or Path(recipe["dataset"]).name != recipe["dataset"]
        or recipe["dataset"] in (".", "..")
    ):
        raise ValueError("Invalid preparation configuration")
    if args.stage in ("prepare", "all"):
        prepare(args, recipe)
    if args.stage in ("build", "all"):
        build(args, recipe)


if __name__ == "__main__":
    main()
