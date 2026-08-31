#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
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
import torch

from shaft.codec.coordinates import dequantize_qwen_bbox
from shaft.offline_kd import (
    OfflineKDArtifactRow,
    OfflineKDArtifactStore,
    OfflineKDArtifactWriter,
    OfflineKDDistributionSpec,
    ShaftOfflineKDInputContract,
    canonical_sha256,
    file_sha256,
)
from shaft.opd.input_abi import ShaftOPDInputABI
from shaft.training.distribution_loss import TeacherDistribution


PACKAGE_VERSION = "banana-v5.8-distillation-bundle-v1"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_image(pair: tuple[Path, Path]) -> tuple[str, int]:
    source, target = pair
    if target.exists():
        if target.stat().st_size != source.stat().st_size:
            raise ValueError(f"Existing image size differs: {target}")
        return target.name, target.stat().st_size
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target.name, target.stat().st_size


def _covering_bbox(values: list[float], *, width: int, height: int) -> list[int]:
    left, top, right, bottom = dequantize_qwen_bbox(values, width=width, height=height)
    x1 = min(max(math.floor(left), 0), width - 1)
    y1 = min(max(math.floor(top), 0), height - 1)
    x2 = min(max(math.ceil(right), x1 + 1), width)
    y2 = min(max(math.ceil(bottom), y1 + 1), height)
    return [x1, y1, x2, y2]


def _gt_standard(row: dict[str, Any]) -> dict[str, Any]:
    plan = row["offline_kd_media_plan"]
    width, height = int(plan["source_width"]), int(plan["source_height"])
    detections = json.loads(row["target_text"])
    return {
        "size": {"width": width, "height": height},
        "background": "none",
        "layout": [
            {
                "type": item["label"],
                "bbox": _covering_bbox(item["bbox_2d"], width=width, height=height),
            }
            for item in detections
        ],
    }


def _load_exclusions(paths: list[Path]) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for path in paths:
        if not path.is_file():
            continue
        for row in _read_jsonl(path):
            output.setdefault(str(row["sample_id"]), []).append(row)
    return output


def _sanitize_exclusion(row: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "reason",
        "eval_split",
        "phash_distance",
        "ahash_distance",
        "whash_distance",
        "exact_sha256",
        "identity_match",
        "matched_sample_id",
        "representative_sample_id",
        "bbox_index",
        "bbox",
        "label",
        "value",
        "threshold",
    }
    return {key: row[key] for key in sorted(allowed & set(row))}


class _OldShardReader:
    def __init__(self, root: Path, manifest: dict[str, Any]):
        self.root = root
        self.shards = dict(manifest["shards"])
        self.verified: set[str] = set()
        self.name: str | None = None
        self.tensors: dict[str, torch.Tensor] = {}

    def row(self, shard: str, index: int) -> tuple[torch.Tensor, ...]:
        if shard not in self.shards:
            raise ValueError(f"Artifact row references an undeclared shard: {shard!r}")
        if shard != self.name:
            path = (self.root / shard).resolve()
            if not path.is_relative_to(self.root):
                raise ValueError("Artifact shard resolves outside the source artifact.")
            if shard not in self.verified:
                if file_sha256(path) != self.shards[shard]:
                    raise ValueError(f"Artifact shard checksum mismatch: {shard!r}")
                self.verified.add(shard)
            with safe_open(path, framework="pt", device="cpu") as handle:
                self.tensors = {key: handle.get_tensor(key) for key in handle.keys()}
            self.name = shard
        tensors = self.tensors
        row_offsets = tensors["row_offsets"]
        input_offsets = tensors["input_row_offsets"]
        start, end = int(row_offsets[index]), int(row_offsets[index + 1])
        input_start, input_end = int(input_offsets[index]), int(input_offsets[index + 1])
        return (
            tensors["input_token_ids"][input_start:input_end],
            tensors["completion_token_ids"][start:end],
            tensors["media_sha256"][index],
            tensors["topk_token_ids"][start:end],
            tensors["topk_log_probs"][start:end],
            tensors["tail_log_probs"][start:end],
        )


def _write_gt(pair: tuple[dict[str, Any], Path]) -> None:
    row, path = pair
    path.write_text(json.dumps(_gt_standard(row), ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")


def build(args: argparse.Namespace) -> None:
    selection_path = Path(args.selection).resolve()
    accepted_path = Path(args.accepted).resolve()
    old_artifact = Path(args.old_artifact).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    selection = _read_jsonl(selection_path)
    accepted = _read_jsonl(accepted_path)
    by_id = {str(row["sample_id"]): row for row in selection}
    if len(by_id) != len(selection):
        raise ValueError("Selection contains duplicate sample_id values.")
    names = [Path(row["image_path"]).name for row in selection]
    if len(names) != len(set(names)):
        raise ValueError("Flat image publication requires globally unique basenames.")
    accepted_ids = {str(row["sample_id"]) for row in accepted}
    if len(accepted_ids) != len(accepted) or not accepted_ids <= set(by_id):
        raise ValueError("Accepted rows must be a unique subset of selection.")
    exclusions = _load_exclusions([Path(path).resolve() for path in args.exclusions])
    old_manifest = json.loads((old_artifact / "manifest.json").read_text(encoding="utf-8"))
    input_abi = ShaftOPDInputABI.from_mapping(old_manifest["input_abi"])
    input_contract = ShaftOfflineKDInputContract.from_mapping(
        old_manifest["input_contract"]
    )
    OfflineKDArtifactStore(
        old_artifact / "manifest.json",
        student_input_abi=input_abi,
        student_input_contract=input_contract,
        max_cached_shards=1,
    )
    distribution_payload = old_manifest["distribution"]
    if distribution_payload["mode"] != "topk_tail":
        raise ValueError("Detection distillation packaging requires a topk_tail artifact.")
    distribution_spec = OfflineKDDistributionSpec(
        mode="topk_tail",
        temperature=float(distribution_payload["temperature"]),
        top_k=int(distribution_payload["top_k"]),
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent)
    )
    try:
        image_root = staging / "images"
        gt_root = staging / "gt_standard"
        audit_root = staging / "audit"
        detection_root = staging / "offline_kd" / "detection"
        for directory in (image_root, gt_root, audit_root, detection_root.parent):
            directory.mkdir(parents=True, exist_ok=True)

        copy_pairs = [(Path(row["image_path"]), image_root / Path(row["image_path"]).name) for row in selection]
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            copied = list(executor.map(_copy_image, copy_pairs, chunksize=32))

        gt_pairs = [(row, gt_root / f"{Path(row['image_path']).stem}.json") for row in accepted]
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            list(executor.map(_write_gt, gt_pairs, chunksize=32))

        source_fingerprint = canonical_sha256(
            {
                "kind": "banana-v5.8-detection-filtered-repack-v1",
                "parent_artifact_id": old_manifest["artifact_id"],
                "accepted_sample_ids": sorted(accepted_ids),
            }
        )
        reader = _OldShardReader(old_artifact, old_manifest)
        with OfflineKDArtifactWriter(
            detection_root,
            teacher_model=str(old_manifest["teacher"]["model"]),
            teacher_checkpoint_fingerprint=str(
                old_manifest["teacher"]["checkpoint_fingerprint"]
            ),
            input_abi=input_abi,
            input_contract=input_contract,
            distribution_spec=distribution_spec,
            source_fingerprint=source_fingerprint,
            denylist_fingerprint=str(old_manifest["build"]["denylist_fingerprint"]),
            shard_rows=args.shard_rows,
            shard_max_bytes=args.shard_max_bytes,
        ) as writer:
            for row in accepted:
                filename = Path(row["image_path"]).name
                ref = row["distillation_ref"]
                values = reader.row(str(ref["shard"]), int(ref["row"]))
                input_ids, completion_ids, media_digest, token_ids, log_probs, tail = values
                payload = {key: value for key, value in row.items() if key != "distillation_ref"}
                payload["image_path"] = f"../../images/{filename}"
                writer.add(
                    OfflineKDArtifactRow(
                        source_payload=payload,
                        input_token_ids=input_ids,
                        completion_token_ids=completion_ids,
                        media_sha256=bytes(int(value) for value in media_digest.tolist()),
                        distribution=TeacherDistribution(
                            kind="topk_tail",
                            vocab_size=int(distribution_payload["vocab_size"]),
                            topk_token_ids=token_ids,
                            topk_log_probs=log_probs,
                            tail_log_probs=tail,
                            temperature=float(distribution_payload["temperature"]),
                        ),
                    )
                )
            writer.finalize()

        with (audit_root / "exclusions.jsonl").open("w", encoding="utf-8") as handle:
            for sample_id in sorted(set(by_id) - accepted_ids):
                source = by_id[sample_id]
                records = exclusions.get(sample_id, [])
                payload = {
                    "sample_id": sample_id,
                    "image": Path(source["image_path"]).name,
                    "source_pool": source["source_pool"],
                    "source_relative_path": source["source_relative_path"],
                    "status": "excluded" if records else "inference_bad_case",
                    "reasons": [_sanitize_exclusion(record) for record in records]
                    or [{"reason": "generation_or_strict_parser_failure"}],
                }
                handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")

        new_rows = {row["sample_id"]: row for row in _read_jsonl(detection_root / "train.jsonl")}
        source_counts = Counter(row["source_pool"] for row in selection)
        accepted_counts = Counter(row["source_pool"] for row in accepted)
        with (staging / "index.jsonl").open("w", encoding="utf-8") as handle:
            for row in selection:
                sample_id = str(row["sample_id"])
                filename = Path(row["image_path"]).name
                published = new_rows.get(sample_id)
                payload = {
                    "sample_id": sample_id,
                    "image": f"images/{filename}",
                    "source_pool": row["source_pool"],
                    "source_relative_path": row["source_relative_path"],
                    "status": "accepted" if published else ("excluded" if sample_id in exclusions else "inference_bad_case"),
                    "tasks": {
                        "detection": {
                            "gt_standard": None if not published else f"gt_standard/{Path(filename).stem}.json",
                            "offline_kd": None if not published else published["distillation_ref"],
                        }
                    },
                }
                handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")

        summary = {
            "version": PACKAGE_VERSION,
            "images": len(selection),
            "accepted_detection": len(accepted),
            "excluded": len(selection) - len(accepted),
            "source_counts": dict(sorted(source_counts.items())),
            "accepted_source_counts": dict(sorted(accepted_counts.items())),
            "image_bytes": sum(size for _, size in copied),
            "old_artifact_id": old_manifest["artifact_id"],
            "new_artifact_id": writer.artifact_id,
        }
        (audit_root / "build_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (audit_root / "source_distribution.json").write_text(json.dumps({"all": dict(sorted(source_counts.items())), "accepted": dict(sorted(accepted_counts.items()))}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (audit_root / "attribute_coverage.json").write_text(json.dumps({"detection": len(accepted), "shape_reconstruction": 0, "line_reconstruction": 0, "image_reconstruction": 0}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (staging / "bundle_manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (staging / "README.md").write_text(
            "# Banana v5.8 Distillation Bundle\n\n"
            "Self-contained images, strict `gt_standard` pseudo labels, and Offline-KD artifacts.\n"
            "Images with `excluded` or `inference_bad_case` status are retained for audit and future relabeling, but have no pseudo label or KD reference.\n",
            encoding="utf-8",
        )
        checks = []
        for relative in ["README.md", "bundle_manifest.json", "index.jsonl", "audit/build_summary.json", "audit/exclusions.jsonl", "offline_kd/detection/manifest.json", "offline_kd/detection/train.jsonl"]:
            path = staging / relative
            checks.append(f"{_sha256(path)}  {relative}")
        (staging / "checksums.sha256").write_text("\n".join(checks) + "\n", encoding="utf-8")
        os.replace(staging, output)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Publish and compact a filtered detection distillation bundle.")
    value.add_argument("--selection", required=True)
    value.add_argument("--accepted", required=True)
    value.add_argument("--old-artifact", required=True)
    value.add_argument("--exclusions", action="append", default=[])
    value.add_argument("--output", required=True)
    value.add_argument("--workers", type=int, default=50)
    value.add_argument("--shard-rows", type=int, default=4096)
    value.add_argument("--shard-max-bytes", type=int, default=512 * 1024 * 1024)
    return value


if __name__ == "__main__":
    build(parser().parse_args())
