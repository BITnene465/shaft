#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shaft.prompting import load_prompt_template


@dataclass(frozen=True)
class PromptConfig:
    prompt_id: str
    system_prompt: str
    user_prompt: str


def _load_prompt_config(path: Path) -> PromptConfig:
    prompt = load_prompt_template(path, variant_id="main")
    return PromptConfig(
        prompt_id=prompt.prompt_id,
        system_prompt=prompt.system_prompt,
        user_prompt=prompt.user_prompt,
    )


def _resolve_absolute_image_path(record: dict[str, Any], structured_path: Path) -> Path:
    raw = str(record["image_path"])
    image_path = Path(raw)
    if image_path.is_absolute():
        return image_path
    candidate_paths = [
        (structured_path.parent / image_path).resolve(),
        (structured_path.parent.parent / image_path).resolve(),
    ]
    for candidate_path in candidate_paths:
        if candidate_path.exists():
            return candidate_path
    return candidate_paths[0]


def _normalize_output_image_path(image_path: Path, output_path: Path) -> str:
    return os.path.relpath(image_path, start=output_path.parent)


def _clip_bbox(bbox: list[float], image_width: int, image_height: int) -> tuple[float, float, float, float]:
    if len(bbox) != 4:
        raise ValueError(f"Expected bbox with 4 values, got: {bbox!r}")
    x1, y1, x2, y2 = [float(value) for value in bbox]
    x1 = min(max(x1, 0.0), float(image_width - 1))
    y1 = min(max(y1, 0.0), float(image_height - 1))
    x2 = min(max(x2, 0.0), float(image_width - 1))
    y2 = min(max(y2, 0.0), float(image_height - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def _quantize_coord(value: float, size: int, num_bins: int) -> int:
    if size <= 1:
        return 0
    clipped = min(max(value, 0.0), float(size - 1))
    normalized = clipped / float(size - 1)
    return int(round(normalized * float(num_bins - 1)))


def _quantize_bbox(bbox: list[float], image_width: int, image_height: int, num_bins: int) -> list[int]:
    x1, y1, x2, y2 = _clip_bbox(bbox, image_width=image_width, image_height=image_height)
    return [
        _quantize_coord(x1, image_width, num_bins),
        _quantize_coord(y1, image_height, num_bins),
        _quantize_coord(x2, image_width, num_bins),
        _quantize_coord(y2, image_height, num_bins),
    ]


def _bbox_area_ratio(bbox: list[float], image_width: int, image_height: int) -> float:
    x1, y1, x2, y2 = _clip_bbox(bbox, image_width=image_width, image_height=image_height)
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    image_area = max(1.0, float(image_width * image_height))
    area_ratio = (width * height) / image_area
    return max(area_ratio, 1e-12)


def _log_area_bucket(bbox: list[float], image_width: int, image_height: int, bucket_base: float) -> int:
    if bucket_base <= 1.0:
        raise ValueError(f"bucket_base must be > 1.0, got {bucket_base!r}")
    area_ratio = _bbox_area_ratio(bbox, image_width=image_width, image_height=image_height)
    return int(math.floor(math.log(area_ratio) / math.log(bucket_base)))


def _instance_sort_key(instance: dict[str, Any], *, image_width: int, image_height: int, bucket_base: float) -> tuple[Any, ...]:
    x1, y1, x2, y2 = _clip_bbox(instance["bbox"], image_width=image_width, image_height=image_height)
    bucket = _log_area_bucket(
        instance["bbox"],
        image_width=image_width,
        image_height=image_height,
        bucket_base=bucket_base,
    )
    return (-bucket, y1, x1, y2, x2, str(instance.get("label", "")))


def _build_target_instances(
    instances: list[dict[str, Any]],
    *,
    image_width: int,
    image_height: int,
    num_bins: int,
    bucket_base: float,
) -> list[dict[str, Any]]:
    sorted_instances = sorted(
        instances,
        key=lambda instance: _instance_sort_key(
            instance,
            image_width=image_width,
            image_height=image_height,
            bucket_base=bucket_base,
        ),
    )
    result: list[dict[str, Any]] = []
    for instance in sorted_instances:
        result.append(
            {
                "label": str(instance["label"]),
                "bbox_2d": _quantize_bbox(
                    instance["bbox"],
                    image_width=image_width,
                    image_height=image_height,
                    num_bins=num_bins,
                ),
            }
        )
    return result


def _build_output_row(
    record: dict[str, Any],
    *,
    structured_path: Path,
    output_path: Path,
    dataset_name: str,
    prompt: PromptConfig,
    num_bins: int,
    bucket_base: float,
) -> dict[str, Any]:
    image_width = int(record["image_width"])
    image_height = int(record["image_height"])
    image_path = _resolve_absolute_image_path(record, structured_path=structured_path)
    target_instances = _build_target_instances(
        list(record.get("instances", [])),
        image_width=image_width,
        image_height=image_height,
        num_bins=num_bins,
        bucket_base=bucket_base,
    )
    extra = {
        "prompt_id": prompt.prompt_id,
        "source_sample_id": str(record.get("source_sample_id", record.get("sample_id", ""))),
        "source_type": str(record.get("source_type", "")),
        "image_width": image_width,
        "image_height": image_height,
        "sort_policy": {
            "type": "log_area_bucket",
            "bucket_base": bucket_base,
            "within_bucket": "bbox_coordinate",
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


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent) as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
        tmp_path = Path(handle.name)
    os.replace(tmp_path, path)


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    body = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)
    _atomic_write_text(path, body)


def _build_readme(
    *,
    structured_path: Path,
    output_path: Path,
    dataset_name: str,
    prompt: PromptConfig,
    row_count: int,
    num_bins: int,
    bucket_base: float,
) -> str:
    return (
        "# Grounding Arrow SFT Data\n\n"
        f"- Source structured GT: `{structured_path}`\n"
        f"- Output SFT jsonl: `{output_path.name}`\n"
        f"- Dataset name: `{dataset_name}`\n"
        f"- Prompt id: `{prompt.prompt_id}`\n"
        f"- Rows: `{row_count}`\n"
        f"- Coordinate bins: `{num_bins}`\n"
        f"- Sort policy: `log_area_bucket(base={bucket_base}) -> bbox_coordinate`\n\n"
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
        "[{\"label\":\"arrow\",\"bbox_2d\":[x1,y1,x2,y2]}]\n"
        "```\n"
    )


def convert_structured_jsonl_to_sft(
    *,
    structured_path: Path,
    output_path: Path,
    dataset_name: str,
    prompt_config_path: Path,
    num_bins: int = 1000,
    bucket_base: float = 1.5,
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
                    bucket_base=bucket_base,
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
                bucket_base=bucket_base,
            ),
        )
    return len(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert grounding structured GT jsonl to Shaft SFT jsonl.")
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
        "--area-bucket-base",
        type=float,
        default=1.5,
        help="Log-area bucket base. Larger boxes sort first; within bucket uses bbox coordinates.",
    )
    parser.add_argument(
        "--no-readme",
        action="store_true",
        help="Skip writing sibling README.md for the generated SFT dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    count = convert_structured_jsonl_to_sft(
        structured_path=Path(args.input).resolve(),
        output_path=Path(args.output).resolve(),
        dataset_name=str(args.dataset_name),
        prompt_config_path=Path(args.prompt_config).resolve(),
        num_bins=int(args.num_bins),
        bucket_base=float(args.area_bucket_base),
        write_readme=not bool(args.no_readme),
    )
    print(
        f"Converted {count} row(s): "
        f"{Path(args.input)} -> {Path(args.output)} "
        f"(dataset_name={args.dataset_name}, num_bins={args.num_bins}, area_bucket_base={args.area_bucket_base})"
    )


if __name__ == "__main__":
    main()
