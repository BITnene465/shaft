#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
import sys
from typing import Any

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from convert_grounding_structured_to_sft import (  # noqa: E402
    PromptConfig,
    _atomic_write_text,
    _clip_bbox,
    _load_prompt_config,
    _normalize_output_image_path,
    _quantize_bbox,
    _resolve_absolute_image_path,
    _write_jsonl_atomic,
)


def _instance_sort_key_row_major_quantized(
    prepared_instance: dict[str, Any],
    *,
    row_bucket_size: int,
) -> tuple[Any, ...]:
    quantized_bbox = prepared_instance["bbox_2d"]
    x1, y1, x2, y2 = prepared_instance["bbox"]
    y_center = prepared_instance["y_center_2d"]
    row_bucket = int(y_center // max(1, row_bucket_size))
    return (
        row_bucket,
        quantized_bbox[0],
        quantized_bbox[1],
        quantized_bbox[3],
        quantized_bbox[2],
        y1,
        x1,
        y2,
        x2,
        str(prepared_instance.get("label", "")),
    )


def _prepare_instance(
    instance: dict[str, Any],
    *,
    image_width: int,
    image_height: int,
    num_bins: int,
) -> dict[str, Any]:
    quantized_bbox = _quantize_bbox(
        instance["bbox"],
        image_width=image_width,
        image_height=image_height,
        num_bins=num_bins,
    )
    x1, y1, x2, y2 = _clip_bbox(instance["bbox"], image_width=image_width, image_height=image_height)
    return {
        "label": str(instance["label"]),
        "bbox": [x1, y1, x2, y2],
        "bbox_2d": quantized_bbox,
        "y_center_2d": float(quantized_bbox[1] + quantized_bbox[3]) / 2.0,
    }


def _resolve_row_bucket_size(prepared_instances: list[dict[str, Any]]) -> int:
    if not prepared_instances:
        return 8
    heights = [max(1, int(instance["bbox_2d"][3]) - int(instance["bbox_2d"][1])) for instance in prepared_instances]
    return max(8, int(round(float(median(heights)) * 0.5)))


def _build_target_instances(
    instances: list[dict[str, Any]],
    *,
    image_width: int,
    image_height: int,
    num_bins: int,
) -> tuple[list[dict[str, Any]], int]:
    prepared_instances = [
        _prepare_instance(
            instance,
            image_width=image_width,
            image_height=image_height,
            num_bins=num_bins,
        )
        for instance in instances
    ]
    row_bucket_size = _resolve_row_bucket_size(prepared_instances)
    sorted_instances = sorted(
        prepared_instances,
        key=lambda instance: _instance_sort_key_row_major_quantized(
            instance,
            row_bucket_size=row_bucket_size,
        ),
    )
    result: list[dict[str, Any]] = []
    for instance in sorted_instances:
        result.append(
            {
                "label": str(instance["label"]),
                "bbox_2d": list(instance["bbox_2d"]),
            }
        )
    return result, row_bucket_size


def _build_output_row(
    record: dict[str, Any],
    *,
    structured_path: Path,
    output_path: Path,
    dataset_name: str,
    prompt: PromptConfig,
    num_bins: int,
) -> dict[str, Any]:
    image_width = int(record["image_width"])
    image_height = int(record["image_height"])
    image_path = _resolve_absolute_image_path(record, structured_path=structured_path)
    target_instances, row_bucket_size = _build_target_instances(
        list(record.get("instances", [])),
        image_width=image_width,
        image_height=image_height,
        num_bins=num_bins,
    )
    extra = {
        "prompt_id": prompt.prompt_id,
        "source_sample_id": str(record.get("source_sample_id", record.get("sample_id", ""))),
        "source_type": str(record.get("source_type", "")),
        "image_width": image_width,
        "image_height": image_height,
        "sort_policy": {
            "type": "row_bucket_center_v2",
            "coordinate_space": "bbox_2d",
            "row_anchor": "y_center",
            "row_bucket_size_2d": row_bucket_size,
            "order": ("row_bucket", "x1", "y1", "y2", "x2", "label"),
            "tie_break": "source_bbox_float",
        },
        "num_bins": num_bins,
        "structured_extra": record.get("extra", {}),
    }
    return {
        "image_path": _normalize_output_image_path(image_path, output_path=output_path),
        "sample_id": str(record["sample_id"]),
        "dataset_name": dataset_name,
        "system_prompt": prompt.system_prompt,
        "user_prompt": prompt.user_prompt,
        "target_text": json.dumps(target_instances, ensure_ascii=False, separators=(",", ":")),
        "extra": extra,
    }


def _build_readme(
    *,
    structured_path: Path,
    output_path: Path,
    dataset_name: str,
    prompt: PromptConfig,
    row_count: int,
    num_bins: int,
) -> str:
    return (
        "# Grounding SFT Data\n\n"
        f"- Source structured GT: `{structured_path}`\n"
        f"- Output SFT jsonl: `{output_path.name}`\n"
        f"- Dataset name: `{dataset_name}`\n"
        f"- Prompt id: `{prompt.prompt_id}`\n"
        f"- Rows: `{row_count}`\n"
        f"- Coordinate bins: `{num_bins}`\n"
        "- Sort policy: "
        "`row_bucket_center_v2(bbox_2d: row_bucket(y_center, size=max(8, round(median_height * 0.5))) "
        "-> x1 -> y1 -> y2 -> x2 -> label)`\n\n"
        "Top-level fields:\n"
        "- `image_path`\n"
        "- `sample_id`\n"
        "- `dataset_name`\n"
        "- `system_prompt`\n"
        "- `user_prompt`\n"
        "- `target_text`\n"
        "- `extra`\n\n"
        "`target_text` is a pure JSON array of grounding instances:\n\n"
        "```json\n"
        "[{\"label\":\"<label>\",\"bbox_2d\":[x1,y1,x2,y2]}]\n"
        "```\n"
    )


def convert_structured_jsonl_to_sft_row_major(
    *,
    structured_path: Path,
    output_path: Path,
    dataset_name: str,
    prompt_config_path: Path,
    num_bins: int = 1000,
    write_readme: bool = True,
) -> int:
    prompt = _load_prompt_config(prompt_config_path)
    rows: list[dict[str, Any]] = []
    with structured_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            record = json.loads(text)
            if not isinstance(record, dict):
                raise TypeError(f"{structured_path}:{line_no} is not a JSON object.")
            rows.append(
                _build_output_row(
                    record,
                    structured_path=structured_path,
                    output_path=output_path,
                    dataset_name=dataset_name,
                    prompt=prompt,
                    num_bins=num_bins,
                )
            )
    _write_jsonl_atomic(output_path, rows)
    if write_readme:
        readme_path = output_path.parent / "README.md"
        _atomic_write_text(
            readme_path,
            _build_readme(
                structured_path=structured_path,
                output_path=output_path,
                dataset_name=dataset_name,
                prompt=prompt,
                row_count=len(rows),
                num_bins=num_bins,
            ),
        )
    return len(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert grounding structured GT jsonl to Shaft SFT jsonl with row-major canonical order."
    )
    parser.add_argument("--input", required=True, help="Path to structured train/val jsonl.")
    parser.add_argument("--output", required=True, help="Path to output SFT jsonl.")
    parser.add_argument("--dataset-name", required=True, help="dataset_name to embed in output rows.")
    parser.add_argument(
        "--prompt-config",
        default="configs/prompts/pools/grounding_arrow.v2.4.yaml",
        help="YAML file defining a prompt or versioned prompt pool; pool defaults to main.",
    )
    parser.add_argument("--num-bins", type=int, default=1000, help="Coordinate quantization bins.")
    parser.add_argument(
        "--no-readme",
        action="store_true",
        help="Skip writing sibling README.md for the generated SFT dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    count = convert_structured_jsonl_to_sft_row_major(
        structured_path=Path(args.input).resolve(),
        output_path=Path(args.output).resolve(),
        dataset_name=str(args.dataset_name),
        prompt_config_path=Path(args.prompt_config).resolve(),
        num_bins=int(args.num_bins),
        write_readme=not bool(args.no_readme),
    )
    print(
        f"Converted {count} row(s): "
        f"{Path(args.input)} -> {Path(args.output)} "
        f"(dataset_name={args.dataset_name}, num_bins={args.num_bins}, sort_policy=row_bucket_center_v2)"
    )


if __name__ == "__main__":
    main()
