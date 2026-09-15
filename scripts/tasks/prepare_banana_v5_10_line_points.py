#!/usr/bin/env python3
"""Rebuild real line points plus a V10 multi-path supplement, without editing raw data."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import random
import shutil
from functools import lru_cache

from PIL import Image, ImageOps

import prepare_banana_v5_10 as S
from shaft.data.real_line_points import _bbox, _validate_points

B = S.builder
TASK = "line_context_points"


def rank(value: str, seed: int) -> bytes:
    return hashlib.sha256(f"{seed}:{value}".encode()).digest()


def choose_synthetic(rows: list[dict], limit: int, seed: int) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        if row["group"] == "multi_path":
            groups[row["stratum"]].append(row)
    allocation = S.quotas({k: len(v) for k, v in groups.items()}, limit)
    chosen = []
    for key in sorted(groups):
        chosen.extend(
            sorted(groups[key], key=lambda r: rank(f"{r['stem']}:{r['index']}", seed))[
                : allocation[key]
            ]
        )
    return sorted(chosen, key=lambda r: (r["stem"], r["index"]))


def scan_real(item: tuple[Path, dict]) -> tuple[list[dict], Counter, list[dict]]:
    root, frozen = item
    if (
        S.sha(root / "json" / frozen["json"]) != frozen["json_sha256"]
        or S.sha(root / "images" / frozen["image"]) != frozen["sha256"]
    ):
        raise ValueError(f"Raw source changed since test isolation: {frozen['id']}")
    doc = json.loads((root / "json" / frozen["json"]).read_text())
    width, height = doc["size"]
    rows, counts, rejected = [], Counter(), []
    for index, obj in enumerate(doc["layout"]):
        if obj.get("type") != "line":
            continue
        counts["line_instances"] += 1
        parameters = obj.get("parameters") or {}
        points = parameters.get("points")
        if not points:
            counts["empty_points"] += 1
            continue
        try:
            segments, duplicates = _validate_points(points, width=width, height=height)
            _bbox(obj["bbox"], width=width, height=height)
        except ValueError as error:
            rejected.append({"stem": frozen["id"], "index": index, "reason": str(error)})
            continue
        rows.append(
            {
                "kind": "real",
                "stem": frozen["id"],
                "index": index,
                "bbox": obj["bbox"],
                "source_json": f"json/{frozen['json']}",
                "source_image": f"images/{frozen['image']}",
            }
        )
        counts["multi" if segments > 1 else "single"] += 1
        counts["adjacent_source_duplicates"] += duplicates
    return rows, counts, rejected


def build_source(item: tuple) -> tuple[list[tuple[dict, dict]], list[dict]]:
    root, rows, config, hashes = item
    first = rows[0]
    for key in (first["source_json"], first["source_image"]):
        if S.sha(root / key) != hashes[key]:
            raise ValueError(f"Source changed after selection: {key}")
    doc = json.loads((root / first["source_json"]).read_text())
    with Image.open(root / first["source_image"]) as opened:
        candidate = opened if list(opened.size) == doc["size"] else ImageOps.exif_transpose(opened)
        try:
            if list(candidate.size) != doc["size"]:
                raise ValueError(f"GT/image size mismatch: {first['stem']}")
            image = candidate.convert("RGB")
        finally:
            if candidate is not opened:
                candidate.close()
    built, rejected = [], []
    try:
        for selected in rows:
            obj = doc["layout"][selected["index"]]
            if obj["type"] != "line" or obj["bbox"] != selected["bbox"]:
                raise ValueError("Frozen line identity mismatch")
            points = obj["parameters"]["points"]
            _validate_points(points, width=image.width, height=image.height)
            if selected["kind"] == "synthetic" and len(points) < 2:
                raise ValueError("Synthetic supplement must be multi-path")
            # No style inference, path reordering, source-point deletion, or bbox-only dedupe.
            parameters = {"is_single": len(points) == 1, "points": points}
            bbox = tuple(obj["bbox"])
            sample_id = f"{selected['kind']}__{selected['stem']}__line_{selected['index']:05d}"
            selection = B.Selection(
                sample_id,
                selected["stem"],
                selected["index"],
                bbox,
                selected["source_image"],
                selected["source_json"],
            )
            if selected["kind"] == "synthetic":
                bbox = (
                    max(0, bbox[0] - 2),
                    max(0, bbox[1] - 2),
                    min(image.width, bbox[2] + 2),
                    min(image.height, bbox[3] + 2),
                )
            try:
                structured, sfts, _ = B._build_row(
                    config=config,
                    selection=selection,
                    source_image=image,
                    source_instance_index=selected["index"],
                    source_bbox=bbox,
                    source_parameters=parameters,
                    source_layout=doc["layout"],
                    image_width=image.width,
                    image_height=image.height,
                )
            except ValueError as error:
                # These are target/view eligibility failures, never raw annotation edits.
                if not any(
                    text in str(error)
                    for text in (
                        "collision after Qwen quantization",
                        "collapses after Qwen quantization",
                    )
                ):
                    raise
                rejected.append({"sample_id": sample_id, "reason": str(error)})
                continue
            row, sft = json.loads(structured), json.loads(sfts[0][1])
            row["extra"]["cohort"] = selected["kind"]
            row["extra"]["source_bbox_raw"] = obj["bbox"]
            row["extra"]["locator_padding_px"] = 2 if selected["kind"] == "synthetic" else 0
            sft["extra"]["structured_extra"] = row["extra"]
            built.append((row, sft))
    finally:
        image.close()
    return built, rejected


def jpeg_twin(item: tuple[Path, dict, dict, dict]) -> tuple[dict, dict]:
    root, clean, clean_sft, recipe = item
    row, sft = deepcopy(clean), deepcopy(clean_sft)
    quality = random.Random(rank(clean["sample_id"] + ":jpeg", recipe["seed"])).randint(
        *recipe["real_jpeg"]["quality"]
    )
    sample_id = clean["sample_id"] + "__jpeg"
    relative = f"images/train/{B._image_shard(sample_id)}/{sample_id}.png"
    output = root / relative
    output.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(root / "structured" / clean["image_path"]) as image:
        with io.BytesIO() as buffer:
            image.convert("RGB").save(
                buffer,
                format="JPEG",
                quality=quality,
                subsampling=recipe["real_jpeg"]["subsampling"],
            )
            buffer.seek(0)
            with Image.open(buffer) as compressed:
                compressed.save(output, format="PNG", compress_level=recipe["png_compress_level"])
    row["sample_id"], row["image_path"] = sample_id, f"../{relative}"
    row["extra"]["clean_sample_id"] = clean["sample_id"]
    row["extra"]["pixel_augmentation"] = {
        "profile": recipe["version"] + ":real-jpeg",
        "operations": [
            {"name": "jpeg", "quality": quality, "subsampling": recipe["real_jpeg"]["subsampling"]}
        ],
    }
    sft["sample_id"], sft["image_path"] = sample_id, f"../../../{relative}"
    sft["extra"]["structured_extra"] = row["extra"]
    return row, sft


def verify_points(root: Path, raw_root: Path, synthetic_root: Path) -> dict:
    """Recompute every ordered target from raw, and prove JPEG twin geometry is unchanged."""

    @lru_cache(maxsize=32)
    def document(path: Path) -> dict:
        return json.loads(path.read_text())

    clean_signatures, seen_media, counts = {}, set(), Counter()
    qualities = Counter()
    for line in (root / "structured/train.jsonl").open():
        row = json.loads(line)
        extra = row["extra"]
        source_root = raw_root if extra["cohort"] == "real" else synthetic_root
        doc = document(source_root / extra["source_json"])
        obj = doc["layout"][extra["source_instance_index"]]
        points = obj["parameters"]["points"]
        expected = B._target_parameters(
            "line",
            {"is_single": len(points) == 1, "points": points},
            crop_box=tuple(extra["crop_box"]),
            strict_line_geometry=True,
        )
        if row["instances"][0]["parameters"] != expected:
            raise ValueError("Raw ordered points / derived target mismatch")
        left, top, right, bottom = extra["crop_box"]
        if not all(
            left <= x <= right and top <= y <= bottom for segment in points for x, y in segment
        ):
            raise ValueError("Crop clips source points")
        if [right - left, bottom - top] != [row["image_width"], row["image_height"]]:
            raise ValueError("Crop / image dimensions mismatch")
        signature = S.dump({k: row[k] for k in ("image_width", "image_height", "instances")})
        signature += S.dump(
            {
                k: extra[k]
                for k in ("crop_box", "proposal_bbox_2d", "source_json", "source_instance_index")
            }
        )
        twin = extra.get("clean_sample_id")
        if twin:
            if extra["cohort"] != "real" or clean_signatures.get(twin) != signature:
                raise ValueError("JPEG twin is missing or changes target/geometry")
            op = extra["pixel_augmentation"]["operations"]
            if len(op) != 1 or op[0]["name"] != "jpeg" or op[0]["subsampling"] != 0:
                raise ValueError("Invalid real JPEG operation")
            qualities[op[0]["quality"]] += 1
            counts["real_jpeg"] += 1
        else:
            clean_signatures[row["sample_id"]] = signature
            counts[extra["cohort"]] += 1
        media = (root / "structured" / row["image_path"]).resolve().relative_to(root.resolve())
        if str(media) in seen_media:
            raise ValueError("Duplicate output media")
        seen_media.add(str(media))
    actual_media = {str(p.relative_to(root)) for p in (root / "images").rglob("*.png")}
    if actual_media != seen_media:
        raise ValueError("Unreferenced or missing media")
    for path in root.rglob("val.jsonl"):
        if path.stat().st_size:
            raise ValueError("Train-only dataset has nonempty validation")
    return {"counts": dict(counts), "jpeg_quality_counts": dict(sorted(qualities.items()))}


def prepare(args: argparse.Namespace, recipe: dict) -> None:
    args.work_root.mkdir(parents=True, exist_ok=False)
    gate = args.grounding_root / "selection"
    frozen = json.loads((gate / "sources.json").read_text())
    eligible = [r for r in frozen if not r["test_matches"]]
    split = (gate / "train.txt").read_text().splitlines()
    if set(split) != {f"json/{r['json']}" for r in eligible}:
        raise ValueError("Grounding split and exclusion manifest disagree")
    candidates, stats, failures = [], Counter(), []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, (rows, counts, rejected) in enumerate(
            pool.map(scan_real, [(args.raw_root, r) for r in eligible], chunksize=16), 1
        ):
            candidates.extend(rows)
            stats.update(counts)
            failures.extend(rejected)
            if index % 5000 == 0:
                print(f"scan real {index}/{len(eligible)} paths={len(candidates)}", flush=True)
    synthetic = choose_synthetic(
        [
            json.loads(line)
            for line in (args.synthetic_cohort / "selection/train.jsonl").read_text().splitlines()
        ],
        recipe["synthetic_limit"],
        recipe["seed"],
    )
    if len(synthetic) != recipe["synthetic_limit"]:
        raise ValueError("Insufficient synthetic multi-path candidates")
    for row in synthetic:
        row.update(
            kind="synthetic",
            source_json=f"gt_standard/{row['stem']}.json",
            source_image=f"img/{row['stem']}.png",
        )
    if args.limit:
        candidates, synthetic = candidates[: args.limit], synthetic[: args.limit]
    selections = candidates + synthetic
    B._atomic_write_text(
        args.work_root / "selection.jsonl", "".join(S.dump(row) + "\n" for row in selections)
    )
    raw_hashes = {}
    for row in eligible:
        raw_hashes[f"json/{row['json']}"] = row["json_sha256"]
        raw_hashes[f"images/{row['image']}"] = row["sha256"]
    synth_hashes = json.loads((args.synthetic_cohort / "reports/input_checksums.json").read_text())
    for folder in ("real", "synthetic"):
        selected_paths = {
            r[k] for r in selections if r["kind"] == folder for k in ("source_json", "source_image")
        }
        hashes = raw_hashes if folder == "real" else synth_hashes
        S.write_json(
            args.work_root / f"{folder}_checksums.json",
            {k: hashes[k] for k in sorted(selected_paths)},
        )
    for name in ("sources.json", "tests.json", "train.txt", "metadata.json"):
        shutil.copyfile(gate / name, args.work_root / f"grounding_{name}")
    tracked = [
        Path(__file__),
        Path(S.__file__),
        Path(B.__file__),
        S.REPO / recipe["prompt_pool"],
        *sorted((S.REPO / "src/shaft").rglob("*.py")),
    ]
    S.write_json(
        args.work_root / "reproduction.lock.json",
        {
            "recipe": recipe,
            "environment": S.environment(),
            "limit": args.limit,
            "synthetic_policy": json.loads((S.REPO / recipe["synthetic_recipe"]).read_text()),
            "code_sha256": {str(p.resolve().relative_to(S.REPO)): S.sha(p) for p in tracked},
            "artifacts_sha256": {p.name: S.sha(p) for p in sorted(args.work_root.iterdir())},
            "counts": dict(stats),
            "source_rejections": failures,
            "real_selected": len(candidates),
            "synthetic_selected": len(synthetic),
            "excluded_test_sources": len(frozen) - len(eligible),
        },
    )
    print(
        S.dump(
            {
                "real_selected": len(candidates),
                "synthetic_selected": len(synthetic),
                "counts": stats,
                "source_rejections": len(failures),
            }
        ),
        flush=True,
    )


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
            raise ValueError(f"Frozen selection/input hash changed: {path}")
    destination = args.output_root.resolve() / TASK
    if destination.exists() and not args.replace and not lock["limit"]:
        raise FileExistsError("Pass --replace to back up and replace the derived cohort")
    stage = work / TASK
    stage.mkdir(exist_ok=False)
    for folder in ("selection", "reports", "structured", "sft/formulations/points"):
        (stage / folder).mkdir(parents=True)
        if folder != "reports":
            (stage / folder / "val.jsonl").touch()
    shutil.copyfile(work / "selection.jsonl", stage / "selection/train.jsonl")
    selections = [json.loads(line) for line in (work / "selection.jsonl").read_text().splitlines()]
    pool_id, schema, formulations = B._prompt_contract(
        S.REPO / recipe["prompt_pool"], eligible_formulations=("points",)
    )
    jobs = []
    for kind, root in (("real", args.raw_root), ("synthetic", args.synthetic_root)):
        hashes = json.loads((work / f"{kind}_checksums.json").read_text())
        spec = B.TaskSpec(
            TASK,
            "line",
            work / "selection.jsonl",
            root,
            S.REPO / recipe["prompt_pool"],
            "real_point" if kind == "real" else "synthetic_point_multi",
            source_dataset_id="banana-v5.10-" + kind,
            eligible_formulations=("points",),
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
            True,
            lock["synthetic_policy"]["pixel_policy"] if kind == "synthetic" else None,
            True,
        )
        groups = defaultdict(list)
        for row in selections:
            if row["kind"] == kind:
                groups[row["stem"]].append(row)
        for _, rows in sorted(groups.items()):
            jobs.append(
                (
                    root,
                    rows,
                    config,
                    {k: hashes[k] for k in (rows[0]["source_json"], rows[0]["source_image"])},
                )
            )
    counts, failures, real_rows = Counter(), [], []
    with (
        (stage / "structured/train.jsonl").open("w") as structured,
        (stage / "sft/formulations/points/train.jsonl").open("w") as sft,
    ):
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for index, (rows, rejected) in enumerate(pool.map(build_source, jobs, chunksize=4), 1):
                failures.extend(rejected)
                for row, record in rows:
                    structured.write(S.dump(row) + "\n")
                    sft.write(S.dump(record) + "\n")
                    counts[row["extra"]["cohort"]] += 1
                    if row["extra"]["cohort"] == "real":
                        real_rows.append((row, record))
                if index % 1000 == 0:
                    print(f"build sources {index}/{len(jobs)} counts={dict(counts)}", flush=True)
        count = int(len(real_rows) * recipe["real_jpeg"]["additional_fraction"])
        selected = sorted(real_rows, key=lambda pair: rank(pair[0]["sample_id"], recipe["seed"]))[
            :count
        ]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for index, (row, record) in enumerate(
                pool.map(jpeg_twin, [(stage, r, s, recipe) for r, s in selected], chunksize=16), 1
            ):
                structured.write(S.dump(row) + "\n")
                sft.write(S.dump(record) + "\n")
                counts["real_jpeg"] += 1
                if index % 5000 == 0:
                    print(f"JPEG twins {index}/{count}", flush=True)
    if counts["real"] + counts["synthetic"] + len(failures) != len(selections):
        raise ValueError("Base cardinality mismatch")
    S.write_json(stage / "reports/rejected_views.json", failures)
    S.write_json(stage / "reports/build_summary.json", counts)
    for p in sorted(work.iterdir()):
        if p.is_file():
            shutil.copyfile(p, stage / "reports" / p.name)
    print("Verifying all points targets and decoding all media", flush=True)
    audit = verify_points(stage, args.raw_root, args.synthetic_root)
    if audit["counts"] != dict(counts):
        raise ValueError("Verified cohort counts mismatch")
    S.write_json(stage / "reports/points_audit.json", audit)
    result = S.verify_task(stage, args.workers, label="line", formulations=("points",))
    result["counts"] = dict(counts)
    result["rejected_views"] = len(failures)
    S.write_json(stage / "reports/reproduction_result.json", result)
    (stage / "README.md").write_text(
        "# Banana v5.10 line points\n\n"
        f"Train-only, {result['rows']} rows. Counts: {dict(counts)}.\n\n"
        "Same line reconstruction v5.8 pool, points formulation only. "
        "All eligible real paths plus 15,000 selected V10 multi-path candidates; "
        "view quantization failures are reported, never repaired by dropping points. "
        "Real clean crops retained; extra 20% JPEG 60–90, 4:4:4. "
        "Synthetic single-operation policy inherited from the frozen V10 line recipe.\n\n"
        "Reproduction: scripts/tasks/banana_v5_10_line_points.md; "
        "reports/reproduction.lock.json and input checksums. "
        "Raw and test gate manifests must be byte-identical across machines.\n"
    )
    if lock["limit"]:
        print(S.dump({"canary": str(stage), **result}), flush=True)
        return
    backup = destination.with_name(TASK + ".before_v5_10")
    if destination.exists():
        if backup.exists():
            raise FileExistsError(backup)
        destination.rename(backup)
    try:
        stage.rename(destination)
    except BaseException:
        if backup.exists() and not destination.exists():
            backup.rename(destination)
        raise
    print(S.dump({"published": str(destination), **result}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("raw-root", "synthetic-root", "grounding-root", "synthetic-cohort", "work-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--recipe",
        type=Path,
        default=S.REPO / "configs/data/preparation/banana_v5_10_line_points.json",
    )
    parser.add_argument("--output-root", type=Path, default=Path("data"))
    parser.add_argument("--workers", type=int, default=50)
    parser.add_argument("--stage", choices=("prepare", "build", "all"), default="all")
    parser.add_argument("--limit", type=int, default=0, help="Canary per cohort; cannot publish")
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    recipe = json.loads(args.recipe.read_text())
    policy = recipe["real_jpeg"]
    if (
        args.workers < 1
        or args.limit < 0
        or not 0 <= policy["additional_fraction"] <= 1
        or not 1 <= policy["quality"][0] <= policy["quality"][1] <= 100
        or policy["subsampling"] != 0
        or recipe["synthetic_limit"] < 1
    ):
        raise ValueError("Invalid line points recipe")
    if args.stage in ("prepare", "all"):
        prepare(args, recipe)
    if args.stage in ("build", "all"):
        build(args, recipe)


if __name__ == "__main__":
    main()
