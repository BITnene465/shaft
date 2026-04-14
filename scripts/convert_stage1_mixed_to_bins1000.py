#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

PROMPT_STAGE1_V2 = (
    'Locate every instance that belongs to the following categories: "single_arrow, double_arrow".\n'
    'Report bbox coordinates in JSON format with no markdown and no extra text.\n'
    'Each item must be {"label":"single_arrow"|"double_arrow","bbox_2d":[x1,y1,x2,y2]}'
)


def _quantize(value: float, size: int, num_bins: int = 1000) -> int:
    size = max(int(size), 1)
    if size == 1:
        return 0
    clipped = min(max(float(value), 0.0), float(size - 1))
    return int(round(clipped / float(size - 1) * float(num_bins - 1)))


def _to_abs_image_path(repo_root: Path, raw_path: str, jsonl_path: Path) -> str:
    p = Path(str(raw_path))
    if p.is_absolute():
        return str(p)
    candidate_repo = (repo_root / p).resolve()
    if candidate_repo.exists():
        return str(candidate_repo)
    return str((jsonl_path.parent / p).resolve())


def _convert_file(repo_root: Path, src: Path, dst: Path, *, dataset_id: str) -> int:
    count = 0
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            text = line.strip()
            if not text:
                continue
            raw = json.loads(text)

            image_width = int(raw.get("image_width", 0) or 0)
            image_height = int(raw.get("image_height", 0) or 0)
            if image_width <= 0 or image_height <= 0:
                raise ValueError(f"Missing image_width/image_height at {src}:{line_no}")

            payload = []
            for instance in (raw.get("instances") or []):
                if not isinstance(instance, dict):
                    continue
                bbox = instance.get("bbox", [])
                if not isinstance(bbox, list) or len(bbox) != 4:
                    continue
                x1, y1, x2, y2 = [float(v) for v in bbox]
                payload.append(
                    {
                        "label": str(instance.get("label", "")),
                        "bbox_2d": [
                            _quantize(x1, image_width),
                            _quantize(y1, image_height),
                            _quantize(x2, image_width),
                            _quantize(y2, image_height),
                        ],
                    }
                )

            out = {
                "image_path": _to_abs_image_path(repo_root, str(raw.get("image_path", "")), src),
                "sample_id": str(raw.get("sample_id", "")).strip() or f"line_{line_no:07d}",
                "dataset_id": dataset_id,
                "user_prompt": PROMPT_STAGE1_V2,
                "system_prompt": "",
                "target_text": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                "image_width": image_width,
                "image_height": image_height,
                "source_type": raw.get("source_type", ""),
                "source_sample_id": raw.get("source_sample_id", ""),
                "task_type": raw.get("task_type", "grounding"),
                "domain_type": raw.get("domain_type", "arrow"),
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    stage1_dir = repo_root / "data" / "two_stage" / "stage1"
    train_path = stage1_dir / "train_mixed.jsonl"
    val_path = stage1_dir / "val_full.jsonl"

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError("Expected stage1 train_mixed.jsonl and val_full.jsonl")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_bak = stage1_dir / f"train_mixed.jsonl.bak_{stamp}_pre_bins1000"
    val_bak = stage1_dir / f"val_full.jsonl.bak_{stamp}_pre_bins1000"
    train_bak.write_bytes(train_path.read_bytes())
    val_bak.write_bytes(val_path.read_bytes())

    train_tmp = stage1_dir / "train_mixed.bins1000.tmp.jsonl"
    val_tmp = stage1_dir / "val_full.bins1000.tmp.jsonl"

    train_count = _convert_file(repo_root, train_path, train_tmp, dataset_id="stage1_grounding_arrow")
    val_count = _convert_file(repo_root, val_path, val_tmp, dataset_id="stage1_grounding_arrow")

    train_tmp.replace(train_path)
    val_tmp.replace(val_path)

    shaft_dir = repo_root / "data" / "shaft"
    shaft_dir.mkdir(parents=True, exist_ok=True)
    shaft_train = shaft_dir / "train.jsonl"
    shaft_test = shaft_dir / "test.jsonl"
    shaft_train.write_bytes(train_path.read_bytes())
    shaft_test.write_bytes(val_path.read_bytes())

    print(f"backup_train={train_bak}")
    print(f"backup_val={val_bak}")
    print(f"train_records={train_count}")
    print(f"val_records={val_count}")
    print(f"shaft_train={shaft_train}")
    print(f"shaft_test={shaft_test}")


if __name__ == "__main__":
    main()
