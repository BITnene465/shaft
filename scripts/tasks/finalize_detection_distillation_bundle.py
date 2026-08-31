#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from safetensors import safe_open
from safetensors.torch import save_file
import torch

from shaft.codec.coordinates import dequantize_qwen_bbox
from shaft.offline_kd.artifact import canonical_sha256, file_sha256, offline_kd_artifact_identity


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open() if line.strip()]


def _copy(pair: tuple[Path, Path]) -> None:
    source, target = pair
    if target.exists():
        if target.stat().st_size != source.stat().st_size:
            raise ValueError(f"Image collision differs in size: {target}")
        return
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, target)


def _link_or_copy(source: str, target: str) -> str:
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)
    return target


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _bbox(values: list[int], *, width: int, height: int) -> list[int]:
    left, top, right, bottom = dequantize_qwen_bbox(values, width=width, height=height)
    x1 = min(max(math.floor(left), 0), width - 1)
    y1 = min(max(math.floor(top), 0), height - 1)
    x2 = min(max(math.ceil(right), x1 + 1), width)
    y2 = min(max(math.ceil(bottom), y1 + 1), height)
    return [x1, y1, x2, y2]


def _write_gt(pair: tuple[dict[str, Any], Path]) -> None:
    row, target = pair
    plan = row["offline_kd_media_plan"]
    width, height = int(plan["source_width"]), int(plan["source_height"])
    payload = {
        "size": {"width": width, "height": height},
        "background": "none",
        "layout": [
            {"type": item["label"], "bbox": _bbox(item["bbox_2d"], width=width, height=height)}
            for item in json.loads(row["target_text"])
        ],
    }
    target.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")


def _subset_shard(source: Path, selected: list[tuple[int, dict[str, Any]]], target: Path) -> None:
    with safe_open(source, framework="pt", device="cpu") as handle:
        tensors = {key: handle.get_tensor(key) for key in handle.keys()}
    row_offsets, input_offsets = tensors["row_offsets"], tensors["input_row_offsets"]
    position_parts, input_parts, media_parts = [], [], []
    output_offsets, output_input_offsets = [0], [0]
    distribution_names = [name for name in ("dense_logits", "topk_token_ids", "topk_log_probs", "tail_log_probs") if name in tensors]
    distribution_parts: dict[str, list[torch.Tensor]] = {name: [] for name in distribution_names}
    for row_index, _ in selected:
        start, end = int(row_offsets[row_index]), int(row_offsets[row_index + 1])
        input_start, input_end = int(input_offsets[row_index]), int(input_offsets[row_index + 1])
        position_parts.append(tensors["completion_token_ids"][start:end])
        input_parts.append(tensors["input_token_ids"][input_start:input_end])
        media_parts.append(tensors["media_sha256"][row_index])
        for name in distribution_names:
            distribution_parts[name].append(tensors[name][start:end])
        output_offsets.append(output_offsets[-1] + end - start)
        output_input_offsets.append(output_input_offsets[-1] + input_end - input_start)
    output = {
        "completion_token_ids": torch.cat(position_parts),
        "input_token_ids": torch.cat(input_parts),
        "media_sha256": torch.stack(media_parts),
        "row_offsets": torch.tensor(output_offsets, dtype=torch.long),
        "input_row_offsets": torch.tensor(output_input_offsets, dtype=torch.long),
    }
    output.update({name: torch.cat(parts) for name, parts in distribution_parts.items()})
    save_file(output, str(target))


def main() -> None:
    parser = argparse.ArgumentParser(description="Atomically add a filtered detection artifact to the public distillation bundle.")
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--new-selection", required=True)
    parser.add_argument("--selection-audit", required=True)
    parser.add_argument("--paper-root", required=True)
    parser.add_argument("--new-artifact", required=True)
    parser.add_argument("--new-accepted", required=True)
    parser.add_argument("--new-exclusions", required=True)
    parser.add_argument("--drop-existing", required=True)
    parser.add_argument("--workers", type=int, default=50)
    args = parser.parse_args()

    source_bundle = Path(args.bundle).resolve()
    output = Path(args.output).resolve()
    if output == source_bundle or output.is_relative_to(source_bundle):
        raise ValueError("Finalized bundle output must be outside the source bundle.")
    if output.exists():
        raise FileExistsError(f"Finalized bundle output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    publish_staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent)
    )
    shutil.rmtree(publish_staging)
    shutil.copytree(source_bundle, publish_staging, copy_function=_link_or_copy)
    bundle = publish_staging
    old_root = bundle / "offline_kd" / "detection"
    old_manifest = json.loads((old_root / "manifest.json").read_text())
    old_rows = _rows(old_root / "train.jsonl")
    old_index = _rows(bundle / "index.jsonl")
    new_root = Path(args.new_artifact).resolve()
    new_manifest = json.loads((new_root / "manifest.json").read_text())
    new_rows = _rows(Path(args.new_accepted).resolve())
    selection = _rows(Path(args.new_selection).resolve())
    audit = json.loads(Path(args.selection_audit).read_text())
    filter_exclusions = _rows(Path(args.new_exclusions).resolve())
    drop_existing = {line.strip() for line in Path(args.drop_existing).open() if line.strip()}

    for key in ("teacher", "input_abi", "distribution"):
        if old_manifest[key] != new_manifest[key]:
            raise ValueError(f"Old/new artifact {key} differs.")
    old_contract = dict(old_manifest["input_contract"])
    new_contract = dict(new_manifest["input_contract"])
    old_contract.pop("media_snapshot_id", None)
    new_contract.pop("media_snapshot_id", None)
    if old_contract != new_contract:
        raise ValueError("Old/new input contracts differ beyond media snapshot identity.")
    input_contract = dict(old_manifest["input_contract"])
    input_contract["media_snapshot_id"] = "banana-v5.8-distillation-bundle-detection-v2"
    kept_old = [row for row in old_rows if row["sample_id"] not in drop_existing]
    source_fingerprint = canonical_sha256({"kind": "banana-v5.8-distillation-final-repack-v2", "parents": [old_manifest["artifact_id"], new_manifest["artifact_id"]], "sample_ids": [row["sample_id"] for row in kept_old + new_rows]})
    build = {"source_fingerprint": source_fingerprint, "denylist_fingerprint": old_manifest["build"]["denylist_fingerprint"]}
    artifact_id = offline_kd_artifact_identity(teacher=old_manifest["teacher"], input_abi=old_manifest["input_abi"], input_contract=input_contract, distribution=old_manifest["distribution"], build=build)

    image_root, gt_root = bundle / "images", bundle / "gt_standard"
    paper_root = Path(args.paper_root).resolve()
    new_names = {row["image"] for row in audit["exclusion_rows"]} | {Path(row["image_path"]).name for row in selection}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        list(executor.map(_copy, [(paper_root / name, image_root / name) for name in sorted(new_names)], chunksize=32))
        list(executor.map(_write_gt, [(row, gt_root / f"{Path(row['image_path']).stem}.json") for row in new_rows], chunksize=32))

    artifact_staging = Path(
        tempfile.mkdtemp(prefix=".detection.final-", dir=old_root.parent)
    )
    shards: dict[str, str] = {}
    published_rows: list[dict[str, Any]] = []
    try:
        for source_root, source_rows in ((old_root, kept_old), (new_root, new_rows)):
            grouped: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
            for row in source_rows:
                ref = row["distillation_ref"]
                grouped[ref["shard"]].append((int(ref["row"]), row))
            for source_name, selected in grouped.items():
                output_name = f"shards/teacher-{len(shards) + 1:05d}.safetensors"
                output_path = artifact_staging / output_name
                output_path.parent.mkdir(exist_ok=True)
                source_path = source_root / source_name
                with safe_open(source_path, framework="np") as handle:
                    total_rows = int(handle.get_slice("row_offsets").get_shape()[0]) - 1
                indices = [index for index, _ in selected]
                if indices == list(range(total_rows)):
                    try:
                        os.link(source_path, output_path)
                    except OSError:
                        shutil.copy2(source_path, output_path)
                else:
                    _subset_shard(source_path, selected, output_path)
                shards[output_name] = file_sha256(output_path)
                for output_row, (_, row) in enumerate(selected):
                    payload = {key: value for key, value in row.items() if key != "distillation_ref"}
                    payload["image_path"] = f"../../images/{Path(row['image_path']).name}"
                    payload["distillation_ref"] = {"artifact_id": artifact_id, "shard": output_name, "row": output_row}
                    published_rows.append(payload)
        with (artifact_staging / "train.jsonl").open("w", encoding="utf-8") as handle:
            for row in published_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest = {"version": old_manifest["version"], "artifact_id": artifact_id, "teacher": old_manifest["teacher"], "input_abi": old_manifest["input_abi"], "input_contract": input_contract, "distribution": old_manifest["distribution"], "build": build, "shards": shards}
        (artifact_staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")

        accepted_by_id = {row["sample_id"]: row for row in new_rows}
        published_by_id = {row["sample_id"]: row for row in published_rows}
        excluded_by_id = defaultdict(list)
        for row in filter_exclusions:
            excluded_by_id[row["sample_id"]].append(row)
        new_index = []
        for row in old_index:
            if row["sample_id"] in drop_existing:
                changed = dict(row)
                changed["status"] = "excluded"
                changed["tasks"] = {"detection": {"gt_standard": None, "offline_kd": None}}
                new_index.append(changed)
            else:
                new_index.append(row)
        for row in selection:
            sample_id, name = row["sample_id"], Path(row["image_path"]).name
            accepted = accepted_by_id.get(sample_id)
            new_index.append({"sample_id": sample_id, "image": f"images/{name}", "source_pool": "paper", "source_relative_path": f"paper/{name}", "status": "accepted" if accepted else ("excluded" if sample_id in excluded_by_id else "inference_bad_case"), "tasks": {"detection": {"gt_standard": None if not accepted else f"gt_standard/{Path(name).stem}.json", "offline_kd": None if not accepted else published_by_id[sample_id]["distillation_ref"]}}})
        for row in audit["exclusion_rows"]:
            name = row["image"]
            new_index.append({"sample_id": f"paper:paper/{name}", "image": f"images/{name}", "source_pool": "paper", "source_relative_path": f"paper/{name}", "status": "excluded", "tasks": {"detection": {"gt_standard": None, "offline_kd": None}}})
        index_tmp = bundle / ".index.jsonl.final"
        with index_tmp.open("w", encoding="utf-8") as handle:
            for row in new_index:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        old_backup = old_root.parent / ".detection.previous"
        if old_backup.exists():
            raise FileExistsError(old_backup)
        os.replace(old_root, old_backup)
        os.replace(artifact_staging, old_root)
        os.replace(index_tmp, bundle / "index.jsonl")
        for sample_id in drop_existing:
            old = next(row for row in old_rows if row["sample_id"] == sample_id)
            (gt_root / f"{Path(old['image_path']).stem}.json").unlink(missing_ok=True)
        shutil.rmtree(old_backup)
        summary_path = bundle / "audit" / "build_summary.json"
        summary = json.loads(summary_path.read_text())
        source_counts: dict[str, int] = defaultdict(int)
        accepted_source_counts: dict[str, int] = defaultdict(int)
        for row in new_index:
            source_counts[row["source_pool"]] += 1
            if row["status"] == "accepted":
                accepted_source_counts[row["source_pool"]] += 1
        summary.update({"images": len(new_index), "accepted_detection": len(published_rows), "excluded": len(new_index) - len(published_rows), "new_paper_files": len(new_names), "new_teacher_success": len(_rows(new_root / "train.jsonl")), "new_accepted_detection": len(new_rows), "dropped_existing_near_duplicates": len(drop_existing), "new_artifact_id": artifact_id})
        summary["source_counts"] = dict(sorted(source_counts.items()))
        summary["accepted_source_counts"] = dict(sorted(accepted_source_counts.items()))
        summary["image_bytes"] = sum(path.stat().st_size for path in image_root.iterdir() if path.is_file())
        _atomic_write_text(
            summary_path,
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        )
        _atomic_write_text(
            bundle / "bundle_manifest.json",
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        )
        _atomic_write_text(
            bundle / "audit" / "source_distribution.json",
            json.dumps(
                {
                    "all": summary["source_counts"],
                    "accepted": summary["accepted_source_counts"],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
        )
        exclusion_path = bundle / "audit" / "exclusions.jsonl"
        existing_exclusions = _rows(exclusion_path)
        accepted_ids = set(accepted_by_id)
        generated_ids = {row["sample_id"] for row in selection}
        inference_bad_ids = generated_ids - accepted_ids - set(excluded_by_id)
        exclusion_tmp = exclusion_path.with_name(f".{exclusion_path.name}.final.tmp")
        with exclusion_tmp.open("w", encoding="utf-8") as handle:
            for row in existing_exclusions:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            for row in audit["exclusion_rows"]:
                handle.write(json.dumps({"sample_id": f"paper:paper/{row['image']}", "image": row["image"], "source_pool": "paper", "source_relative_path": f"paper/{row['image']}", "status": "excluded", "reasons": [{key: value for key, value in row.items() if key != "image"}]}, ensure_ascii=False, separators=(",", ":")) + "\n")
            for row in filter_exclusions:
                handle.write(json.dumps({"sample_id": row["sample_id"], "image": row["image"], "source_pool": "paper", "source_relative_path": f"paper/{row['image']}", "status": "excluded", "reasons": [{key: value for key, value in row.items() if key not in {"sample_id", "image"}}]}, ensure_ascii=False, separators=(",", ":")) + "\n")
            selection_by_id = {row["sample_id"]: row for row in selection}
            for sample_id in sorted(inference_bad_ids):
                name = Path(selection_by_id[sample_id]["image_path"]).name
                handle.write(json.dumps({"sample_id": sample_id, "image": name, "source_pool": "paper", "source_relative_path": f"paper/{name}", "status": "inference_bad_case", "reasons": [{"reason": "generation_or_strict_parser_failure"}]}, ensure_ascii=False, separators=(",", ":")) + "\n")
        os.replace(exclusion_tmp, exclusion_path)
        _atomic_write_text(
            bundle / "audit" / "attribute_coverage.json",
            json.dumps(
                {
                    "detection": len(published_rows),
                    "shape_reconstruction": 0,
                    "line_reconstruction": 0,
                    "image_reconstruction": 0,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
        )
        checksum_paths = ["README.md", "bundle_manifest.json", "index.jsonl", "audit/build_summary.json", "audit/exclusions.jsonl", "audit/source_distribution.json", "audit/attribute_coverage.json", "offline_kd/detection/manifest.json", "offline_kd/detection/train.jsonl"]
        _atomic_write_text(
            bundle / "checksums.sha256",
            "\n".join(
                f"{_sha256(bundle / relative)}  {relative}" for relative in checksum_paths
            )
            + "\n",
        )
        os.replace(publish_staging, output)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except Exception:
        if artifact_staging.exists():
            shutil.rmtree(artifact_staging)
        if publish_staging.exists():
            shutil.rmtree(publish_staging)
        raise


if __name__ == "__main__":
    main()
