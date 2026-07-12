#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shaft.codec.coordinates import quantize_qwen_coordinate
from shaft.prompting import load_prompt_template


GROUNDING_ROW_BUCKET_SIZE_2D = 20


@dataclass(frozen=True)
class TaskSpec:
    name: str
    kind: str
    prompt_config: str


@dataclass(frozen=True)
class PromptConfig:
    prompt_id: str
    system_prompt: str
    user_prompt: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ConvertConfig:
    task: TaskSpec
    prompt: PromptConfig
    structured_path: Path
    output_path: Path
    num_bins: int


TASKS: tuple[TaskSpec, ...] = (
    TaskSpec("grounding_arrow", "grounding", "configs/prompts/pools/grounding_arrow.v2.4.yaml"),
    TaskSpec("grounding_layout", "grounding", "configs/prompts/pools/grounding_layout.v5.0.yaml"),
    TaskSpec("grounding_shape", "grounding", "configs/prompts/pools/grounding_shape.v2.4.yaml"),
    TaskSpec(
        "grounding_icon_image",
        "grounding",
        "configs/prompts/pools/grounding_icon_image.v2.4.yaml",
    ),
    TaskSpec("point_arrow", "point_arrow", "configs/prompts/pools/point_arrow.v2.4.yaml"),
    TaskSpec("point_line", "point_line", "configs/prompts/pools/point_line.v5.0.yaml"),
)

DEFAULT_TASK_NAMES: tuple[str, ...] = ("grounding_layout", "point_line")


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent) as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
        tmp_path = Path(handle.name)
    os.replace(tmp_path, path)


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    body = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    )
    _atomic_write_text(path, body)


def _load_prompt_config(path: Path) -> PromptConfig:
    prompt = load_prompt_template(path, variant_id="main")
    return PromptConfig(
        prompt_id=prompt.prompt_id,
        system_prompt=prompt.system_prompt,
        user_prompt=prompt.user_prompt,
        metadata=dict(prompt.metadata),
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


def _quantize_coord(value: float, size: int, num_bins: int) -> int:
    return quantize_qwen_coordinate(value, size=size, num_bins=num_bins)


def _clip_bbox(
    bbox: list[float],
    *,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
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


def _quantize_bbox(
    bbox: list[float],
    *,
    image_width: int,
    image_height: int,
    num_bins: int,
) -> list[int]:
    x1, y1, x2, y2 = _clip_bbox(bbox, image_width=image_width, image_height=image_height)
    return [
        _quantize_coord(x1, image_width, num_bins),
        _quantize_coord(y1, image_height, num_bins),
        _quantize_coord(x2, image_width, num_bins),
        _quantize_coord(y2, image_height, num_bins),
    ]


def _prepare_grounding_instance(
    instance: dict[str, Any],
    *,
    image_width: int,
    image_height: int,
    num_bins: int,
) -> dict[str, Any]:
    bbox = _clip_bbox(instance["bbox"], image_width=image_width, image_height=image_height)
    bbox_2d = _quantize_bbox(
        instance["bbox"],
        image_width=image_width,
        image_height=image_height,
        num_bins=num_bins,
    )
    return {
        "label": str(instance["label"]),
        "bbox": list(bbox),
        "bbox_2d": bbox_2d,
        "area_2d": max(0, int(bbox_2d[2]) - int(bbox_2d[0]))
        * max(0, int(bbox_2d[3]) - int(bbox_2d[1])),
    }


def _grounding_sort_key(instance: dict[str, Any]) -> tuple[Any, ...]:
    quantized_bbox = instance["bbox_2d"]
    row_bucket = int(int(quantized_bbox[1]) // GROUNDING_ROW_BUCKET_SIZE_2D)
    return (
        row_bucket,
        int(quantized_bbox[0]),
        int(quantized_bbox[1]),
        -int(instance["area_2d"]),
        int(quantized_bbox[2]),
        int(quantized_bbox[3]),
        str(instance.get("label", "")),
    )


def _build_grounding_target(
    instances: list[dict[str, Any]],
    *,
    image_width: int,
    image_height: int,
    num_bins: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prepared_instances = [
        _prepare_grounding_instance(
            instance,
            image_width=image_width,
            image_height=image_height,
            num_bins=num_bins,
        )
        for instance in instances
    ]
    sorted_instances = sorted(
        prepared_instances,
        key=_grounding_sort_key,
    )
    target = [
        {
            "bbox_2d": list(instance["bbox_2d"]),
            "label": str(instance["label"]),
        }
        for instance in sorted_instances
    ]
    return target, {
        "type": "row_bucket_top_area_v3",
        "coordinate_space": "bbox_2d",
        "row_anchor": "y1",
        "row_bucket_size_2d": GROUNDING_ROW_BUCKET_SIZE_2D,
        "order": ("row_bucket", "x1", "y1", "-area", "x2", "y2", "label"),
        "tie_break": "qwen_bbox_2d",
    }


def _build_point_target(
    instances: list[dict[str, Any]],
    *,
    image_width: int,
    image_height: int,
    num_bins: int,
) -> dict[str, Any]:
    if len(instances) != 1:
        raise ValueError(f"point_arrow expects one arrow instance, got {len(instances)}")
    linestrip = instances[0].get("linestrip")
    if not isinstance(linestrip, list) or len(linestrip) < 2:
        raise ValueError("point_arrow requires a linestrip with at least two points.")
    keypoints: list[list[int]] = []
    for point in linestrip:
        if not isinstance(point, list | tuple) or len(point) != 2:
            raise ValueError(f"Invalid linestrip point: {point!r}")
        keypoints.append(
            [
                _quantize_coord(float(point[0]), image_width, num_bins),
                _quantize_coord(float(point[1]), image_height, num_bins),
            ]
        )
    return {"keypoints_2d": keypoints}


def _build_point_line_target(
    instances: list[dict[str, Any]],
    *,
    image_width: int,
    image_height: int,
    num_bins: int,
) -> dict[str, Any]:
    if len(instances) != 1:
        raise ValueError(f"point_line expects one line instance, got {len(instances)}")
    linestrip = instances[0].get("linestrip")
    segments = _normalize_line_segments(linestrip)
    if not segments:
        raise ValueError("point_line requires at least one linestrip segment with two points.")
    quantized_segments: list[list[list[int]]] = []
    for segment in segments:
        quantized_segment: list[list[int]] = []
        for point in segment:
            quantized_segment.append(
                [
                    _quantize_coord(float(point[0]), image_width, num_bins),
                    _quantize_coord(float(point[1]), image_height, num_bins),
                ]
            )
        quantized_segments.append(quantized_segment)
    raw_is_single = instances[0].get("is_single")
    is_single = bool(raw_is_single) if isinstance(raw_is_single, bool) else len(quantized_segments) == 1
    return {
        "type": "line",
        "parameters": {
            "is_single": is_single,
            "points": quantized_segments,
        },
    }


def _normalize_line_segments(value: Any) -> list[list[list[float]]]:
    if not isinstance(value, list):
        return []
    if _is_point_list(value):
        return [[_coerce_point(point) for point in value]]

    segments: list[list[list[float]]] = []
    for segment in value:
        if not _is_point_list(segment):
            continue
        coerced_segment = [_coerce_point(point) for point in segment]
        if len(coerced_segment) >= 2:
            segments.append(coerced_segment)
    return segments


def _is_point_list(value: Any) -> bool:
    return (
        isinstance(value, list | tuple)
        and len(value) >= 2
        and all(isinstance(point, list | tuple) and len(point) == 2 for point in value)
        and all(not isinstance(point[0], list | tuple) for point in value)
    )


def _coerce_point(point: Any) -> list[float]:
    if not isinstance(point, list | tuple) or len(point) != 2:
        raise ValueError(f"Invalid linestrip point: {point!r}")
    return [float(point[0]), float(point[1])]


def _build_output_row(item: tuple[int, str, ConvertConfig]) -> dict[str, Any]:
    line_no, line, config = item
    record = json.loads(line)
    if not isinstance(record, dict):
        raise TypeError(f"{config.structured_path}:{line_no} is not a JSON object.")
    image_width = int(record["image_width"])
    image_height = int(record["image_height"])
    image_path = _resolve_absolute_image_path(record, structured_path=config.structured_path)
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    instances = list(record.get("instances", []))
    if config.task.kind == "grounding":
        target, sort_policy = _build_grounding_target(
            instances,
            image_width=image_width,
            image_height=image_height,
            num_bins=config.num_bins,
        )
        target_text = json.dumps(target, ensure_ascii=False, separators=(",", ":"))
        task_extra: dict[str, Any] = {"sort_policy": sort_policy}
        if config.prompt.metadata.get("output_schema"):
            task_extra["output_schema"] = str(config.prompt.metadata["output_schema"])
    elif config.task.kind == "point_arrow":
        target_text = json.dumps(
            _build_point_target(
                instances,
                image_width=image_width,
                image_height=image_height,
                num_bins=config.num_bins,
            ),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        task_extra = {
            "target_policy": {
                "type": "full_ordered_linestrip",
                "coordinate_space": "keypoints_2d",
                "order": "arrow_tail_to_head",
            }
        }
    elif config.task.kind == "point_line":
        target_text = json.dumps(
            _build_point_line_target(
                instances,
                image_width=image_width,
                image_height=image_height,
                num_bins=config.num_bins,
            ),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        task_extra = {
            "target_policy": {
                "type": "line_points_segments",
                "coordinate_space": "parameters.points",
                "order": "source_linestrip_segment_order",
            }
        }
    else:
        raise ValueError(f"Unknown task kind: {config.task.kind}")

    extra = {
        "prompt_id": config.prompt.prompt_id,
        "source_sample_id": str(record.get("source_sample_id", record.get("sample_id", ""))),
        "source_type": str(record.get("source_type", "")),
        "image_width": image_width,
        "image_height": image_height,
        "num_bins": config.num_bins,
        "structured_extra": record.get("extra", {}),
        **task_extra,
    }
    return {
        "image_path": _normalize_output_image_path(image_path, output_path=config.output_path),
        "sample_id": str(record["sample_id"]),
        "dataset_name": config.task.name,
        "system_prompt": "",
        "user_prompt": "",
        "target_text": target_text,
        "extra": extra,
    }


def _read_structured_lines(path: Path) -> list[tuple[int, str]]:
    result: list[tuple[int, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if text:
                result.append((line_no, text))
    return result


def _convert_split(
    *,
    task: TaskSpec,
    split: str,
    data_root: Path,
    num_bins: int,
    workers: int,
) -> int:
    task_root = data_root / task.name
    structured_path = task_root / "structured" / f"{split}.jsonl"
    output_path = task_root / "sft" / f"{split}.jsonl"
    if not structured_path.exists():
        raise FileNotFoundError(structured_path)
    prompt = _load_prompt_config(Path(task.prompt_config))
    config = ConvertConfig(
        task=task,
        prompt=prompt,
        structured_path=structured_path.resolve(),
        output_path=output_path.resolve(),
        num_bins=num_bins,
    )
    lines = _read_structured_lines(structured_path)
    work_items = [(line_no, line, config) for line_no, line in lines]
    if workers <= 1:
        rows = [_build_output_row(item) for item in work_items]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(_build_output_row, work_items, chunksize=256))
    _write_jsonl_atomic(output_path, rows)
    _write_readme(task_root / "sft" / "README.md", task=task, prompt=prompt, rows=rows, split=split)
    return len(rows)


def _count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _target_contract_summary(task: TaskSpec) -> str:
    if task.kind == "grounding":
        return (
            "- Target schema: JSON array of `{bbox_2d, label}` objects.\n"
            "- Canonical order: `row_bucket(y1, 20) -> x1 -> y1 -> -area -> x2 -> y2 -> label`; "
            "labels are mixed in visual order, and `line` is not forced to the end.\n"
        )
    if task.kind == "point_line":
        return (
            "- Target schema: `{\"type\":\"line\",\"parameters\":{\"is_single\":...,\"points\":[...]}}`.\n"
            "- Canonical order: `parameters.points` preserves the source `linestrip` segment order.\n"
        )
    if task.kind == "point_arrow":
        return (
            "- Target schema: `{\"keypoints_2d\":[...]}`.\n"
            "- Canonical order: arrow tail to arrow head.\n"
        )
    return ""


def _write_readme(
    path: Path,
    *,
    task: TaskSpec,
    prompt: PromptConfig,
    rows: list[dict[str, Any]],
    split: str,
) -> None:
    split_counts = Counter(str(row["extra"]["structured_extra"].get("split", "")) for row in rows)
    current_file_counts = {
        jsonl_path.stem: _count_jsonl_rows(jsonl_path)
        for jsonl_path in sorted(path.parent.glob("*.jsonl"))
    }
    content = (
        f"# {task.name} SFT\n\n"
        f"- Task kind: `{task.kind}`\n"
        f"- Prompt id: `{prompt.prompt_id}`\n"
        "- `system_prompt` and `user_prompt` are intentionally empty; runtime prompt pools are "
        "the train prompt source.\n"
        f"- Last converted split: `{split}` (`{len(rows)}` row(s))\n"
        f"- Current SFT split files: `{current_file_counts}`\n"
        f"- Last converted split counts from structured extra: `{dict(split_counts)}`\n"
        f"{_target_contract_summary(task)}"
        "- Coordinate bins: `1000`\n"
        "- Image paths are relative to this `sft/` directory and point back to task-local images.\n"
    )
    _atomic_write_text(path, content)


def _clean_outputs(tasks: list[TaskSpec], data_root: Path) -> None:
    for task in tasks:
        sft_dir = data_root / task.name / "sft"
        if sft_dir.exists():
            shutil.rmtree(sft_dir)
        sft_dir.mkdir(parents=True, exist_ok=True)


def _task_by_name(names: list[str] | None) -> list[TaskSpec]:
    task_map = {task.name: task for task in TASKS}
    if not names:
        return [task_map[name] for name in DEFAULT_TASK_NAMES]
    missing = sorted(set(names) - set(task_map))
    if missing:
        raise ValueError(f"Unknown task(s): {', '.join(missing)}")
    return [task_map[name] for name in names]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SFT JSONL from structured task data.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--task", action="append", choices=[task.name for task in TASKS])
    parser.add_argument("--workers", type=int, default=30)
    parser.add_argument("--num-bins", type=int, default=1000)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    tasks = _task_by_name(args.task)
    if args.clean:
        _clean_outputs(tasks, data_root)

    for task in tasks:
        for split in ("train", "val"):
            count = _convert_split(
                task=task,
                split=split,
                data_root=data_root,
                num_bins=int(args.num_bins),
                workers=int(args.workers),
            )
            print(f"{task.name}/{split}: {count} row(s)")


if __name__ == "__main__":
    main()
