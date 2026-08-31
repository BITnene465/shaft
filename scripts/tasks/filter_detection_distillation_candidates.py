#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from safetensors import safe_open


def _dct_matrix(size: int) -> np.ndarray:
    x = np.arange(size, dtype=np.float64)
    k = x[:, None]
    matrix = np.cos(np.pi * (2 * x + 1) * k / (2 * size))
    matrix[0] *= 1 / np.sqrt(2)
    return matrix * np.sqrt(2 / size)


DCT32 = _dct_matrix(32)
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp"})


def _bits(values: np.ndarray) -> int:
    output = 0
    for bit in values.reshape(-1):
        output = (output << 1) | int(bool(bit))
    return output


def _hashes(path: Path) -> tuple[int, int, int]:
    with Image.open(path) as image:
        image.load()
        gray32 = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
        low = (DCT32 @ gray32 @ DCT32.T)[:8, :8]
        phash = _bits(low >= np.median(low.reshape(-1)[1:]))
        gray8 = np.asarray(image.convert("L").resize((8, 8), Image.Resampling.LANCZOS), dtype=np.float64)
        ahash = _bits(gray8 >= gray8.mean())
        current = gray32
        for _ in range(2):
            current = (current[0::2, 0::2] + current[1::2, 0::2] + current[0::2, 1::2] + current[1::2, 1::2]) / 4.0
        whash = _bits(current >= np.median(current))
    return phash, ahash, whash


def _resolve_image(row: dict[str, Any], jsonl: Path) -> Path:
    path = Path(row["image_path"])
    return path.resolve() if path.is_absolute() else (jsonl.parent / path).resolve()


def _iou(left: list[int], right: list[int]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_left = max(0, left[2] - left[0]) * max(0, left[3] - left[1])
    area_right = max(0, right[2] - right[0]) * max(0, right[3] - right[1])
    union = area_left + area_right - intersection
    return 0.0 if union <= 0 else intersection / union


def _rule_reason(row: dict[str, Any]) -> str | None:
    try:
        boxes = json.loads(row["target_text"])
    except Exception:
        return "malformed_prediction"
    if not boxes:
        return "empty_prediction"
    seen: set[tuple[str, tuple[int, ...]]] = set()
    parsed: list[tuple[str, list[int]]] = []
    for item in boxes:
        if not isinstance(item, dict) or set(item) != {"bbox_2d", "label"}:
            return "malformed_prediction"
        bbox = item["bbox_2d"]
        label = item["label"]
        if label not in {"shape", "icon", "image", "line"} or not isinstance(bbox, list) or len(bbox) != 4:
            return "malformed_prediction"
        if any(type(value) is not int or not 0 <= value <= 999 for value in bbox) or bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
            return "degenerate_or_invalid_bbox"
        key = (label, tuple(bbox))
        if key in seen:
            return "exact_duplicate_bbox_label"
        seen.add(key)
        parsed.append((label, bbox))
    if len(parsed) == 1:
        return "single_box"
    if any(b[0] <= 5 and b[1] <= 5 and b[2] >= 994 and b[3] >= 994 for _, b in parsed):
        return "near_full_frame_box"
    plan = row["offline_kd_media_plan"]
    width, height = int(plan["source_width"]), int(plan["source_height"])
    if max(width / height, height / width) > 10:
        return "extreme_source_aspect_ratio_gt_10"
    if any(label != "line" and min(b[2] - b[0], b[3] - b[1]) <= 2 for label, b in parsed):
        return "ultrathin_nonline_bbox_dim_le_2"
    for index, (_, left) in enumerate(parsed):
        if any(_iou(left, right) >= 0.95 for _, right in parsed[index + 1 :]):
            return "overlapping_duplicate_bbox_iou_ge_095"
    return None


class _DSU:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[right] = left


def _mean_nll(row: dict[str, Any], artifact_roots: dict[str, Path]) -> float:
    ref = row["distillation_ref"]
    root = artifact_roots[ref["artifact_id"]]
    with safe_open(root / ref["shard"], framework="np") as handle:
        offsets = handle.get_tensor("row_offsets")
        start, end = int(offsets[ref["row"]]), int(offsets[ref["row"] + 1])
        completion = handle.get_slice("completion_token_ids")[start:end]
        token_ids = handle.get_slice("topk_token_ids")[start:end]
        log_probs = handle.get_slice("topk_log_probs")[start:end]
    matches = token_ids == completion[:, None]
    if not matches.any(axis=1).all():
        return math.inf
    selected = log_probs[matches]
    return float(-selected.mean()) if selected.size == completion.size else math.inf


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the frozen detection distillation quality policy to a new artifact.")
    parser.add_argument("--existing-jsonl", required=True)
    parser.add_argument("--new-jsonl", required=True)
    parser.add_argument("--eval-image-dir", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--exclusions", required=True)
    parser.add_argument("--drop-existing", required=True)
    parser.add_argument("--workers", type=int, default=50)
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be > 0")
    existing_path, new_path = Path(args.existing_jsonl).resolve(), Path(args.new_jsonl).resolve()
    output_paths = tuple(
        Path(value).resolve()
        for value in (args.output, args.exclusions, args.drop_existing)
    )
    if len(set(output_paths)) != len(output_paths):
        raise ValueError("Filter output paths must be distinct.")
    if any(path.exists() for path in output_paths):
        raise FileExistsError("Detection distillation filter outputs must not already exist.")
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    existing = [json.loads(line) for line in existing_path.open() if line.strip()]
    new = [json.loads(line) for line in new_path.open() if line.strip()]
    excluded: list[dict[str, Any]] = []
    clean_new = []
    for row in new:
        reason = _rule_reason(row)
        if reason:
            excluded.append({"sample_id": row["sample_id"], "image": Path(row["image_path"]).name, "reason": reason})
        else:
            clean_new.append(row)

    rows = existing + clean_new
    paths = [_resolve_image(row, existing_path if index < len(existing) else new_path) for index, row in enumerate(rows)]
    eval_paths = [
        path.resolve()
        for root in args.eval_image_dir
        for path in Path(root).iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    unique_paths = list(dict.fromkeys(paths + eval_paths))
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        hashes = dict(zip(unique_paths, executor.map(_hashes, unique_paths, chunksize=16), strict=True))

    eval_hashes = [(path, hashes[path]) for path in eval_paths]
    keep_rows, keep_paths, keep_hashes = [], [], []
    for row_index, (row, path) in enumerate(zip(rows, paths, strict=True)):
        if row_index < len(existing):
            keep_rows.append(row)
            keep_paths.append(path)
            keep_hashes.append(hashes[path])
            continue
        phash, ahash, whash = hashes[path]
        eval_match = None
        for eval_path, (ep, ea, ew) in eval_hashes:
            pd = (phash ^ ep).bit_count()
            if pd <= 6:
                confirmed = (ahash ^ ea).bit_count() <= 6 and (whash ^ ew).bit_count() <= 6
                eval_match = (eval_path.name, "eval_near_confirmed" if confirmed else "eval_near_review")
                break
        if eval_match:
            excluded.append({"sample_id": row["sample_id"], "image": path.name, "reason": eval_match[1], "matched_eval_image": eval_match[0]})
        else:
            keep_rows.append(row)
            keep_paths.append(path)
            keep_hashes.append(hashes[path])

    dsu = _DSU(len(keep_rows))
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, (phash, ahash, whash) in enumerate(keep_hashes):
        candidates: set[int] = set()
        for block in range(4):
            value = (phash >> (block * 16)) & 0xFFFF
            for neighbor in [value] + [value ^ (1 << bit) for bit in range(16)]:
                candidates.update(buckets[(block, neighbor)])
        for other in candidates:
            op, oa, ow = keep_hashes[other]
            if (phash ^ op).bit_count() <= 6 and (ahash ^ oa).bit_count() <= 6 and (whash ^ ow).bit_count() <= 6:
                dsu.union(index, other)
        for block in range(4):
            buckets[(block, (phash >> (block * 16)) & 0xFFFF)].append(index)
    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(keep_rows)):
        components[dsu.find(index)].append(index)

    artifact_roots = {}
    for path in (existing_path, new_path):
        manifest = json.loads((path.parent / "manifest.json").read_text())
        artifact_roots[manifest["artifact_id"]] = path.parent
    dropped: set[str] = set()
    for members in components.values():
        if len(members) <= 1 or not any(member >= len(existing) for member in members):
            continue
        ranked = sorted(members, key=lambda i: (_mean_nll(keep_rows[i], artifact_roots), -int(keep_rows[i]["offline_kd_media_plan"]["target_width"]) * int(keep_rows[i]["offline_kd_media_plan"]["target_height"]), keep_rows[i]["sample_id"]))
        representative = keep_rows[ranked[0]]["sample_id"]
        for index in ranked[1:]:
            row = keep_rows[index]
            dropped.add(row["sample_id"])
            excluded.append({"sample_id": row["sample_id"], "image": keep_paths[index].name, "reason": "near_duplicate_confirmed", "representative_sample_id": representative})

    existing_ids = {row["sample_id"] for row in existing}
    accepted_new = [row for row in keep_rows if row["sample_id"] not in existing_ids and row["sample_id"] not in dropped]
    drop_existing = sorted(dropped & existing_ids)
    with output_paths[0].open("w", encoding="utf-8") as handle:
        for row in accepted_new:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with output_paths[1].open("w", encoding="utf-8") as handle:
        for row in excluded:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    output_paths[2].write_text("\n".join(drop_existing) + ("\n" if drop_existing else ""), encoding="utf-8")
    summary = {"teacher_success_rows": len(new), "accepted_new": len(accepted_new), "excluded_new": len(new) - len(accepted_new), "drop_existing": len(drop_existing)}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
