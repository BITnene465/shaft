#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from shaft.offline_kd.media_plan import MEDIA_PLAN_FIELD, deterministic_detection_media_plan
from shaft.prompting import load_prompt_template


SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp"})


def _dct_matrix(size: int) -> np.ndarray:
    x = np.arange(size, dtype=np.float64)
    k = x[:, None]
    matrix = np.cos(np.pi * (2 * x + 1) * k / (2 * size))
    matrix[0] *= 1 / np.sqrt(2)
    return matrix * np.sqrt(2 / size)


DCT32 = _dct_matrix(32)


def _bits(values: np.ndarray) -> int:
    output = 0
    for bit in values.reshape(-1):
        output = (output << 1) | int(bool(bit))
    return output


def _hashes(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    with Image.open(path) as image:
        image.load()
        width, height = image.size
        gray32 = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
        low = (DCT32 @ gray32 @ DCT32.T)[:8, :8]
        median = np.median(low.reshape(-1)[1:])
        phash = _bits(low >= median)
        gray8 = np.asarray(image.convert("L").resize((8, 8), Image.Resampling.LANCZOS), dtype=np.float64)
        ahash = _bits(gray8 >= gray8.mean())
        current = gray32
        for _ in range(2):
            current = (current[0::2, 0::2] + current[1::2, 0::2] + current[0::2, 1::2] + current[1::2, 1::2]) / 4.0
        whash = _bits(current >= np.median(current))
    return {"sha256": digest.hexdigest(), "width": int(width), "height": int(height), "phash": phash, "ahash": ahash, "whash": whash}


def _near(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        (left["phash"] ^ right["phash"]).bit_count() <= 6
        and (left["ahash"] ^ right["ahash"]).bit_count() <= 6
        and (left["whash"] ^ right["whash"]).bit_count() <= 6
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze newly arrived paper images for detection distillation.")
    parser.add_argument("--paper-root", required=True)
    parser.add_argument("--old-selection", required=True)
    parser.add_argument("--eval-image-dir", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--workers", type=int, default=50)
    parser.add_argument("--seed", type=int, default=465)
    parser.add_argument("--prompt-pool", default="configs/prompts/pools/grounding_layout.v5.8.yaml")
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be > 0")
    output = Path(args.output).resolve()
    audit_path = Path(args.audit).resolve()
    if output == audit_path:
        raise ValueError("Selection output and audit paths must be distinct.")
    if output.exists() or audit_path.exists():
        raise FileExistsError("New-paper selection outputs must not already exist.")

    old_rows = [json.loads(line) for line in Path(args.old_selection).open() if line.strip()]
    old_names = {Path(row["image_path"]).name for row in old_rows}
    paper = sorted(path.resolve() for path in Path(args.paper_root).iterdir() if path.is_file() and path.suffix.lower() in SUFFIXES)
    candidates = [path for path in paper if path.name not in old_names]
    eval_paths = sorted(
        path.resolve()
        for root in args.eval_image_dir
        for path in Path(root).iterdir()
        if path.is_file() and path.suffix.lower() in SUFFIXES
    )
    old_paths = [Path(row["image_path"]).resolve() for row in old_rows]
    inspected_paths = list(dict.fromkeys(old_paths + candidates + eval_paths))
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        inspected = dict(zip(inspected_paths, executor.map(_hashes, inspected_paths, chunksize=16), strict=True))

    protected_sha = {inspected[path]["sha256"] for path in old_paths + eval_paths}
    eval_hashes = [(path, inspected[path]) for path in eval_paths]
    seen_sha: set[str] = set()
    prompt = load_prompt_template(Path(args.prompt_pool), variant_id="detailed")
    selected: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for path in candidates:
        info = inspected[path]
        reason = None
        matched = None
        if info["sha256"] in protected_sha:
            reason = "existing_or_eval_exact"
        elif info["sha256"] in seen_sha:
            reason = "new_batch_exact_duplicate"
        else:
            for eval_path, eval_info in eval_hashes:
                if _near(info, eval_info):
                    reason = "eval_near_duplicate_confirmed"
                    matched = eval_path.name
                    break
        if reason:
            exclusions.append({"image": path.name, "reason": reason, "matched_eval_image": matched})
            continue
        seen_sha.add(info["sha256"])
        relative = f"paper/{path.name}"
        sample_id = f"paper:{relative}"
        plan = deterministic_detection_media_plan(sample_id=sample_id, width=info["width"], height=info["height"], seed=args.seed)
        selected.append({
            "image_path": str(path), "sample_id": sample_id,
            "system_prompt": prompt.system_prompt, "user_prompt": prompt.user_prompt,
            "target_text": "", "task": "detection", "source_pool": "paper",
            "source_relative_path": relative, "prompt_pool_id": prompt.metadata.get("prompt_pool_id"),
            "prompt_variant_id": prompt.variant_id, "prompt_version": prompt.version,
            MEDIA_PLAN_FIELD: plan.to_dict(),
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    audit = {"paper_files": len(paper), "old_selection_rows": len(old_rows), "new_by_basename": len(candidates), "selected_for_inference": len(selected), "excluded": len(exclusions), "exclusions_by_reason": {reason: sum(row["reason"] == reason for row in exclusions) for reason in sorted({row["reason"] for row in exclusions})}, "exclusion_rows": exclusions}
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in audit.items() if key != "exclusion_rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
