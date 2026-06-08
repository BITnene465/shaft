#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from shaft.prompting import load_prompt_template


ALLOWED_SHAPE_TYPES = {
    "rectangle",
    "rounded_rectangle",
    "ellipse",
    "circle",
    "parallelogram",
    "hexagon",
    "triangle",
    "cylinder",
    "cloud",
    "cube",
    "trapezoid",
    "oval_callout",
    "unknown",
}
ALLOWED_ORIENTATIONS = {
    "default",
    "up",
    "down",
    "left",
    "right",
    "left_slant",
    "right_slant",
    "top_narrow",
    "bottom_narrow",
    "flat_top",
    "point_top",
    "vertical",
    "horizontal",
    "unknown",
}
ALLOWED_STROKE_STYLES = {"solid", "dashed", "dotted", "double", "none", "unknown"}
ALLOWED_FILL_STYLES = {"solid", "none", "unknown"}
TARGET_FIELDS = (
    "label",
    "shape_type",
    "shape_orientation",
    "stroke_visible",
    "fill_visible",
    "stroke_style",
    "fill_style",
    "stroke_color",
    "fill_color",
)
HEX_COLOR_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent) as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
        tmp_path = Path(handle.name)
    os.replace(tmp_path, path)


def _write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    body = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)
    _atomic_write_text(path, body)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_choice(value: Any, allowed: set[str], *, default: str = "unknown") -> str:
    item = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return item if item in allowed else default


def _normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y"}


def _normalize_color(value: Any) -> str:
    text = str(value or "").strip()
    lowered = text.lower()
    if lowered in {"none", "unknown"}:
        return lowered
    if HEX_COLOR_RE.match(text):
        return text.upper()
    return "unknown"


def _target_from_weak(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": "shape",
        "shape_type": _normalize_choice(row.get("shape_type"), ALLOWED_SHAPE_TYPES),
        "shape_orientation": _normalize_choice(row.get("shape_orientation"), ALLOWED_ORIENTATIONS),
        "stroke_visible": _normalize_bool(row.get("stroke_visible")),
        "fill_visible": _normalize_bool(row.get("fill_visible")),
        "stroke_style": _normalize_choice(row.get("stroke_style"), ALLOWED_STROKE_STYLES),
        "fill_style": _normalize_choice(row.get("fill_style"), ALLOWED_FILL_STYLES),
        "stroke_color": _normalize_color(row.get("stroke_color")),
        "fill_color": _normalize_color(row.get("fill_color")),
    }


def _is_clean_weak_row(row: dict[str, Any], *, weak_job_dir: Path) -> tuple[bool, str]:
    target = _target_from_weak(row)
    if target["shape_type"] == "unknown":
        return False, "unknown_shape_type"
    if str(row.get("confidence") or "").strip().lower() not in {"high", "medium"}:
        return False, "low_confidence"
    if str(row.get("abstain_reason") or "").strip():
        return False, "has_abstain_reason"
    crop_path = weak_job_dir / str(row.get("crop_path") or "")
    if not crop_path.is_file():
        return False, "missing_crop"
    return True, "kept"


def _select_rows(
    rows: list[dict[str, Any]],
    *,
    rounded_cap: int,
    rectangle_cap: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_target_from_weak(row)["shape_type"]].append(row)

    selected: list[dict[str, Any]] = []
    for shape_type, group in sorted(grouped.items()):
        group = list(group)
        rng.shuffle(group)
        if shape_type == "rounded_rectangle":
            group = group[:rounded_cap]
        elif shape_type == "rectangle":
            group = group[:rectangle_cap]
        selected.extend(group)
    selected.sort(key=lambda item: int(item.get("id") or 0))
    return selected


def _copy_crop(src: Path, dst: Path) -> tuple[int, int]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    with Image.open(dst) as image:
        return int(image.width), int(image.height)


def _make_sample_id(row: dict[str, Any]) -> str:
    weak_id = int(row.get("id") or 0)
    return f"drawio_shape_{weak_id:06d}"


def _build_rows(
    selected_rows: list[dict[str, Any]],
    *,
    weak_job_dir: Path,
    output_root: Path,
    prompt_config: Path,
    prompt_variant: str,
    source_job_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    prompt = load_prompt_template(prompt_config, variant_id=prompt_variant)
    structured_rows: list[dict[str, Any]] = []
    sft_rows: list[dict[str, Any]] = []
    distribution: Counter[str] = Counter()

    for row in selected_rows:
        sample_id = _make_sample_id(row)
        target = _target_from_weak(row)
        distribution[target["shape_type"]] += 1
        src_crop = weak_job_dir / str(row["crop_path"])
        image_name = f"{sample_id}{src_crop.suffix.lower()}"
        image_rel = f"images/train/{image_name}"
        image_width, image_height = _copy_crop(src_crop, output_root / image_rel)

        extra = {
            "task": "drawio_shape",
            "split": "train",
            "view_type": "weak_shape_crop",
            "label_source": "weak_vlm",
            "source_job_id": source_job_id,
            "source_model": row.get("source"),
            "source_schema_version": row.get("schema_version"),
            "source_json": row.get("json_rel"),
            "source_image": row.get("image_rel"),
            "source_instance_index": row.get("instance_index"),
            "source_bbox": row.get("bbox"),
            "crop_box": row.get("crop_box"),
            "weak_label_id": row.get("id"),
            "weak_confidence": row.get("confidence"),
        }
        structured_row = {
            "sample_id": sample_id,
            "dataset_name": "drawio_shape",
            "task_name": "drawio_shape",
            "schema_version": "drawio_shape.v4.0",
            "split": "train",
            "image_path": image_rel,
            "image_width": image_width,
            "image_height": image_height,
            "target": target,
            "extra": extra,
        }
        structured_rows.append(structured_row)

        sft_rows.append(
            {
                "sample_id": sample_id,
                "dataset_name": "drawio_shape",
                "task_name": "drawio_shape",
                "schema_version": "drawio_shape.sft.v4.0",
                "image_path": f"../{image_rel}",
                "system_prompt": prompt.system_prompt,
                "user_prompt": prompt.user_prompt,
                "target_text": json.dumps(target, ensure_ascii=False, separators=(",", ":")),
                "extra": {
                    "prompt_id": prompt.prompt_id,
                    "prompt_source": prompt.source_path,
                    "source_sample_id": sample_id,
                    "image_width": image_width,
                    "image_height": image_height,
                    "target_fields": list(TARGET_FIELDS),
                    "label_source": "weak_vlm",
                    "source_job_id": source_job_id,
                    "weak_label_id": row.get("id"),
                },
            }
        )

    return structured_rows, sft_rows, distribution


def _make_readme(
    *,
    output_root: Path,
    source_job_id: str,
    prompt_config: Path,
    prompt_variant: str,
    rounded_cap: int,
    rectangle_cap: int,
    source_count: int,
    clean_count: int,
    selected_count: int,
    distribution: Counter[str],
) -> None:
    lines = [
        "# drawio_shape",
        "",
        "Train-only weak-supervised dataset for draw.io shape reconstruction.",
        "",
        "## Version",
        "",
        "- Dataset schema: `drawio_shape.v4.0`",
        "- SFT schema: `drawio_shape.sft.v4.0`",
        f"- Prompt pool: `{prompt_config}` (`{prompt_variant}`)",
        f"- Source weak-label job: `{source_job_id}`",
        "",
        "## Target Fields",
        "",
        "Training targets contain only business fields used by downstream reconstruction:",
        "",
        "```json",
        json.dumps({field: f"<{field}>" for field in TARGET_FIELDS}, ensure_ascii=False, indent=2),
        "```",
        "",
        "`evidence`, `confidence`, and `abstain_reason` are not training targets.",
        "Weak-label provenance is kept only in `extra` for audit and future filtering.",
        "",
        "## Filtering And Sampling",
        "",
        f"- Source rows: {source_count}",
        f"- Clean rows: {clean_count}",
        "- Dropped rows: `shape_type=unknown`, low confidence, nonempty abstain reason, or missing crop.",
        f"- Rounded rectangle cap: {rounded_cap}",
        f"- Rectangle cap: {rectangle_cap}",
        "- All non-rectangle rare classes are retained after cleaning.",
        f"- Selected train rows: {selected_count}",
        "",
        "## Shape Distribution",
        "",
        "| shape_type | rows |",
        "|---|---:|",
    ]
    for shape_type, count in distribution.most_common():
        lines.append(f"| {shape_type} | {count} |")
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `structured/train.jsonl`: structured crop-level records.",
            "- `sft/train.jsonl`: SFT rows using the v4.0 prompt pool.",
            "- `images/train/`: copied crop images selected for training.",
            "- `summary.json`: machine-readable build summary.",
            "",
            "This dataset is derived from weak VLM labels and should not be treated as raw truth.",
            "",
        ]
    )
    _atomic_write_text(output_root / "README.md", "\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weak-job-dir",
        type=Path,
        default=Path("subTasks/drawio_shape_weak/jobs/drawio_shape_weak_v2_qwen36_27b_train30000"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("data/drawio_shape"))
    parser.add_argument(
        "--prompt-config",
        type=Path,
        default=Path("configs/prompts/pools/drawio_shape.v4.0.yaml"),
    )
    parser.add_argument("--prompt-variant", default="main")
    parser.add_argument("--rounded-cap", type=int, default=3500)
    parser.add_argument("--rectangle-cap", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean", action="store_true", help="Remove the output root before building.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weak_job_dir = args.weak_job_dir
    weak_label_path = weak_job_dir / "weak_labels.json"
    manifest_path = weak_job_dir / "job_manifest.json"
    if not weak_label_path.is_file():
        raise FileNotFoundError(weak_label_path)
    if not args.prompt_config.is_file():
        raise FileNotFoundError(args.prompt_config)

    if args.clean and args.output_root.exists():
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    rows = _load_json(weak_label_path)
    if not isinstance(rows, list):
        raise TypeError(f"{weak_label_path} must contain a list")
    manifest = _load_json(manifest_path) if manifest_path.is_file() else {}
    source_job_id = str(manifest.get("job_id") or weak_job_dir.name)

    drop_reasons: Counter[str] = Counter()
    clean_rows: list[dict[str, Any]] = []
    source_distribution: Counter[str] = Counter()
    clean_distribution: Counter[str] = Counter()
    for row in rows:
        target = _target_from_weak(row)
        source_distribution[target["shape_type"]] += 1
        kept, reason = _is_clean_weak_row(row, weak_job_dir=weak_job_dir)
        if kept:
            clean_rows.append(row)
            clean_distribution[target["shape_type"]] += 1
        else:
            drop_reasons[reason] += 1

    selected_rows = _select_rows(
        clean_rows,
        rounded_cap=args.rounded_cap,
        rectangle_cap=args.rectangle_cap,
        seed=args.seed,
    )
    structured_rows, sft_rows, selected_distribution = _build_rows(
        selected_rows,
        weak_job_dir=weak_job_dir,
        output_root=args.output_root,
        prompt_config=args.prompt_config,
        prompt_variant=args.prompt_variant,
        source_job_id=source_job_id,
    )

    _write_jsonl(args.output_root / "structured" / "train.jsonl", structured_rows)
    _write_jsonl(args.output_root / "sft" / "train.jsonl", sft_rows)

    summary = {
        "dataset_name": "drawio_shape",
        "schema_version": "drawio_shape.v4.0",
        "sft_schema_version": "drawio_shape.sft.v4.0",
        "source_job_id": source_job_id,
        "weak_job_dir": str(weak_job_dir),
        "prompt_config": str(args.prompt_config),
        "prompt_variant": args.prompt_variant,
        "seed": args.seed,
        "caps": {
            "rounded_rectangle": args.rounded_cap,
            "rectangle": args.rectangle_cap,
        },
        "counts": {
            "source_rows": len(rows),
            "clean_rows": len(clean_rows),
            "selected_rows": len(selected_rows),
            "structured_train_rows": len(structured_rows),
            "sft_train_rows": len(sft_rows),
            "copied_images": len(list((args.output_root / "images" / "train").glob("*"))),
        },
        "drop_reasons": dict(sorted(drop_reasons.items())),
        "source_distribution": dict(sorted(source_distribution.items())),
        "clean_distribution": dict(sorted(clean_distribution.items())),
        "selected_distribution": dict(sorted(selected_distribution.items())),
        "target_fields": list(TARGET_FIELDS),
    }
    _write_json(args.output_root / "summary.json", summary)
    _make_readme(
        output_root=args.output_root,
        source_job_id=source_job_id,
        prompt_config=args.prompt_config,
        prompt_variant=args.prompt_variant,
        rounded_cap=args.rounded_cap,
        rectangle_cap=args.rectangle_cap,
        source_count=len(rows),
        clean_count=len(clean_rows),
        selected_count=len(selected_rows),
        distribution=selected_distribution,
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
