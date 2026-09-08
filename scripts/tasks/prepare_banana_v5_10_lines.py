#!/usr/bin/env python3
"""Reproducible v5.10 synthetic full-capable line cohort; raw points cohort is separate."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

from PIL import Image

import audit_banana_v5_10_lines as audit
import prepare_banana_v5_10 as shared


B = shared.builder
C = shared.source_contract
TASK = "line_context_reconstruction"


def scan(item: tuple[Path, str]) -> tuple[list[dict[str, Any]], dict[str, str], Counter]:
    root, stem = item
    json_path = root / "gt_standard" / f"{stem}.json"
    image_path = root / "img" / f"{stem}.png"
    doc = json.loads(json_path.read_text())
    width, height = doc["size"]
    with Image.open(image_path) as image:
        if image.size != (width, height):
            raise ValueError(f"GT/image mismatch: {stem}")
    rows, counts = [], Counter()
    for index, obj in enumerate(doc["layout"]):
        if obj.get("type") != "line":
            continue
        errors = C._validate_line(obj.get("parameters"), width=width, height=height)
        bbox = obj.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4 or not all(C._is_number(v) for v in bbox):
            errors.append("invalid_bbox")
        elif not (0 <= bbox[0] <= bbox[2] <= width and 0 <= bbox[1] <= bbox[3] <= height):
            errors.append("invalid_bbox")
        if errors:
            counts["invalid_instances"] += 1
            counts.update(errors)
            continue
        p = obj["parameters"]
        rows.append(
            {
                "stem": stem,
                "index": index,
                "bbox": bbox,
                "stratum": C._line_stratum(p),
                "group": audit.classify(p),
            }
        )
    return (
        rows,
        {
            f"gt_standard/{stem}.json": shared.sha(json_path),
            f"img/{stem}.png": shared.sha(image_path),
        },
        counts,
    )


def choose(rows: list[dict[str, Any]], recipe: dict[str, Any]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["group"], row["stratum"]].append(row)
    selected = [r for r in rows if r["group"] == "multi_path"]
    policy = recipe["sampling"]
    for group, target in (
        ("single_complex", policy["complex_single_target"]),
        ("single_simple", policy["simple_single_target"]),
    ):
        strata = {key[1]: values for key, values in groups.items() if key[0] == group}
        protected = {
            key
            for key, values in strata.items()
            if len(values) <= policy["protect_stratum_max_count"]
        }
        kept = sum(len(strata[k]) for k in protected)
        capacities = {k: len(v) for k, v in strata.items() if k not in protected}
        allocation = shared.quotas(capacities, max(0, target - kept))
        for key in sorted(strata):
            take = len(strata[key]) if key in protected else allocation[key]
            selected.extend(
                sorted(
                    strata[key],
                    key=lambda r: hashlib.sha256(
                        f"{recipe['seed']}:line:{r['stem']}:{r['index']}".encode()
                    ).digest(),
                )[:take]
            )
    return sorted(selected, key=lambda r: (r["stem"], r["index"]))


def build_source(item: tuple[Path, list[dict[str, Any]], B.WorkerConfig, dict[str, str]]) -> tuple:
    root, selections, config, expected = item
    stem = selections[0]["stem"]
    source_json, source_image = f"gt_standard/{stem}.json", f"img/{stem}.png"
    if (
        shared.sha(root / source_json) != expected[source_json]
        or shared.sha(root / source_image) != expected[source_image]
    ):
        raise ValueError(f"Input changed after selection: {stem}")
    doc = json.loads((root / source_json).read_text())
    with Image.open(root / source_image) as opened:
        image = opened.convert("RGB")
    rows, failures, counts = [], [], Counter()
    try:
        for selected in selections:
            index = selected["index"]
            obj = doc["layout"][index]
            if obj["type"] != "line" or obj["bbox"] != selected["bbox"]:
                raise ValueError("Frozen identity mismatch")
            bbox = list(obj["bbox"])
            for axis, limit in ((0, image.width), (1, image.height)):
                # Small visible-stroke margin; do not change source points or raw GT.
                bbox[axis] = max(0, bbox[axis] - 2)
                bbox[axis + 2] = min(limit, bbox[axis + 2] + 2)
            sample_id = f"{stem}__line_{index:04d}"
            selection = B.Selection(
                sample_id, stem, index, tuple(obj["bbox"]), source_image, source_json
            )
            try:
                structured, sft, row_counts = B._build_row(
                    config=config,
                    selection=selection,
                    source_image=image,
                    source_instance_index=index,
                    source_bbox=tuple(bbox),
                    source_parameters=obj["parameters"],
                    source_layout=doc["layout"],
                    image_width=image.width,
                    image_height=image.height,
                )
            except ValueError as error:
                if "collision after Qwen quantization" not in str(error):
                    raise
                failures.append({"sample_id": sample_id, "reason": str(error)})
                continue
            row = json.loads(structured)
            # Explicitly distinguish the padded locator from authoritative source bbox.
            row["extra"]["source_bbox_raw"] = obj["bbox"]
            row["extra"]["locator_padding_px"] = 2
            materialized = []
            for formulation, line in sft:
                record = json.loads(line)
                record["extra"]["structured_extra"] = row["extra"]
                materialized.append((formulation, shared.dump(record)))
            rows.append((shared.dump(row), materialized))
            counts.update(row_counts)
    finally:
        image.close()
    return rows, failures, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-root", type=Path, required=True)
    parser.add_argument("--exclude-manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data"))
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument(
        "--recipe",
        type=Path,
        default=shared.REPO / "configs/data/preparation/banana_v5_10_line.json",
    )
    parser.add_argument("--workers", type=int, default=50)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    recipe = json.loads(args.recipe.read_text())
    policy = recipe["pixel_policy"]
    if (
        args.workers < 1
        or not 0 <= policy["jpeg_probability"] <= 1
        or not 0 <= policy["jpeg_quality"][0] <= policy["jpeg_quality"][1] <= 100
        or not 0 <= policy["noise_sigma"][0] <= policy["noise_sigma"][1]
        or policy["clean_short_edge_px"] < 0
        or any(v < 0 for v in recipe["sampling"].values())
    ):
        raise ValueError("Invalid preparation configuration")
    args.work_root.mkdir(parents=True, exist_ok=False)
    destination = args.output_root.resolve() / TASK
    if destination.exists() and not args.replace:
        raise FileExistsError(destination)
    train = audit.read_ids(args.synthetic_root / "train.txt")
    val = audit.read_ids(args.synthetic_root / "val.txt")
    if set(train) & set(val):
        raise ValueError("Train/val overlap")
    excluded = set().union(*(set(audit.read_ids(p)) for p in args.exclude_manifests))
    train = [stem for stem in train if stem not in excluded]
    candidates, hashes, issues = [], {}, Counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, (rows, inputs, errors) in enumerate(
            pool.map(scan, [(args.synthetic_root, s) for s in train], chunksize=32), 1
        ):
            candidates.extend(rows)
            hashes.update(inputs)
            issues.update(errors)
            if index % 10000 == 0:
                print(f"scan {index}/{len(train)}", flush=True)
    selections = choose(candidates, recipe)
    selected_counts = Counter(r["group"] for r in selections)
    print(shared.dump({"selected": selected_counts, "issues": issues}), flush=True)
    for split in ("train", "val"):
        hashes[f"{split}.txt"] = shared.sha(args.synthetic_root / f"{split}.txt")
    shared.write_json(args.work_root / "input_checksums.json", hashes)
    sources = [
        Path(__file__),
        Path(shared.__file__),
        Path(audit.__file__),
        Path(B.__file__),
        Path(C.__file__),
        shared.REPO / recipe["prompt_pool"],
        *sorted((shared.REPO / "src/shaft").rglob("*.py")),
    ]
    lock = {
        "recipe": recipe,
        "environment": shared.environment(),
        "excluded_ids": sorted(excluded),
        "code_sha256": {str(p.resolve().relative_to(shared.REPO)): shared.sha(p) for p in sources},
        "inputs_sha256": shared.sha(args.work_root / "input_checksums.json"),
        "selected": selected_counts,
        "source_issues": issues,
    }
    shared.write_json(args.work_root / "reproduction.lock.json", lock)
    stage = Path(tempfile.mkdtemp(prefix="line-build-", dir=args.work_root)) / TASK
    for name in (
        "selection",
        "structured",
        "reports",
        *[f"sft/formulations/{f}" for f in recipe["formulations"]],
    ):
        (stage / name).mkdir(parents=True)
        if name != "reports":
            (stage / name / "val.jsonl").touch()
    with (stage / "selection/train.jsonl").open("w") as stream:
        for row in selections:
            stream.write(shared.dump(row) + "\n")
    pool_id, schema, formulations = B._prompt_contract(shared.REPO / recipe["prompt_pool"])
    if list(formulations) != recipe["formulations"]:
        raise ValueError("Prompt formulation mismatch")
    spec = B.TaskSpec(
        TASK,
        "line",
        stage / "selection/train.jsonl",
        args.synthetic_root,
        shared.REPO / recipe["prompt_pool"],
        "synthetic",
        source_dataset_id=recipe["snapshot"],
    )
    config = B.WorkerConfig(
        spec, stage, pool_id, schema, recipe["seed"], 32, 60, 1, formulations, True, policy, True
    )
    by_source: dict[str, list] = defaultdict(list)
    for row in selections:
        by_source[row["stem"]].append(row)
    jobs = [
        (
            args.synthetic_root,
            rows,
            config,
            {key: hashes[key] for key in (f"gt_standard/{stem}.json", f"img/{stem}.png")},
        )
        for stem, rows in sorted(by_source.items())
    ]
    counts, failures = Counter(), []
    with ExitStack() as stack:
        structured = stack.enter_context((stage / "structured/train.jsonl").open("w"))
        stores = {
            f: stack.enter_context((stage / f"sft/formulations/{f}/train.jsonl").open("w"))
            for f in formulations
        }
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for index, (rows, rejected, stats) in enumerate(
                pool.map(build_source, jobs, chunksize=8), 1
            ):
                counts.update(stats)
                failures.extend(rejected)
                for line, sft in rows:
                    structured.write(line + "\n")
                    for f, text in sft:
                        stores[f].write(text + "\n")
                if index % 5000 == 0:
                    print(f"build sources {index}/{len(jobs)} rows={counts['rows']}", flush=True)
    if counts["rows"] + len(failures) != len(selections):
        raise ValueError("Build cardinality mismatch")
    shared.write_json(stage / "reports/rejected_views.json", failures)
    shared.write_json(stage / "reports/build_summary.json", dict(counts))
    result = shared.verify_task(stage, args.workers, label="line", formulations=formulations)
    for name in ("input_checksums.json", "reproduction.lock.json"):
        shutil.copyfile(args.work_root / name, stage / "reports" / name)
    shared.write_json(stage / "reports/reproduction_result.json", result)
    (stage / "README.md").write_text(
        f"# v5.10 line reconstruction\n\nTrain-only; {result['rows']} shared crops; appearance/points/reconstruction.\n\nSource, seed, policies and hashes: reports/reproduction.lock.json.\nNo blur/resampling; single JPEG 40–90 (4:4:4) or noise; tiny targets clean.\n"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    backup = None
    if destination.exists():
        backup = Path(tempfile.mkdtemp(prefix=TASK + ".previous-", dir=destination.parent))
        backup.rmdir()
        destination.rename(backup)
    try:
        stage.rename(destination)
    except BaseException:
        if backup is not None:
            backup.rename(destination)
        raise
    print(shared.dump({"published": str(destination), "backup": str(backup), **result}), flush=True)


if __name__ == "__main__":
    main()
