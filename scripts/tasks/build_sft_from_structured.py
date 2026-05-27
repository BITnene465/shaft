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
from statistics import median
from typing import Any

import yaml


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


@dataclass(frozen=True)
class ConvertConfig:
    task: TaskSpec
    prompt: PromptConfig
    structured_path: Path
    output_path: Path
    num_bins: int


TASKS: tuple[TaskSpec, ...] = (
    TaskSpec("grounding_arrow", "grounding", "configs/prompts/grounding_arrow.yaml"),
    TaskSpec("grounding_layout", "grounding", "configs/prompts/grounding_layout.yaml"),
    TaskSpec("grounding_shape", "grounding", "configs/prompts/grounding_shape.yaml"),
    TaskSpec("grounding_icon_image", "grounding", "configs/prompts/grounding_icon_image.yaml"),
    TaskSpec("point_arrow", "point", "configs/prompts/point_arrow.yaml"),
)


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
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    metadata = payload.get("metadata") or {}
    prompt = payload.get("prompt") or {}
    prompt_id = str(metadata.get("id") or payload.get("prompt_id") or "").strip()
    if not prompt_id:
        raise ValueError(f"Missing prompt id in {path}.")
    system_prompt = str(prompt.get("system_prompt") or payload.get("system_prompt") or "").strip()
    user_prompt = str(prompt.get("user_prompt") or payload.get("user_prompt") or "").strip()
    if not user_prompt:
        raise ValueError(f"Missing user_prompt in {path}.")
    return PromptConfig(
        prompt_id=prompt_id,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
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
    if size <= 1:
        return 0
    clipped = min(max(float(value), 0.0), float(size - 1))
    normalized = clipped / float(size - 1)
    return int(round(normalized * float(num_bins - 1)))


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
        "y_center_2d": float(bbox_2d[1] + bbox_2d[3]) / 2.0,
    }


def _resolve_row_bucket_size(prepared_instances: list[dict[str, Any]]) -> int:
    if not prepared_instances:
        return 8
    heights = [
        max(1, int(instance["bbox_2d"][3]) - int(instance["bbox_2d"][1]))
        for instance in prepared_instances
    ]
    return max(8, int(round(float(median(heights)) * 0.5)))


def _grounding_sort_key(instance: dict[str, Any], *, row_bucket_size: int) -> tuple[Any, ...]:
    quantized_bbox = instance["bbox_2d"]
    x1, y1, x2, y2 = instance["bbox"]
    y_center = instance["y_center_2d"]
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
    row_bucket_size = _resolve_row_bucket_size(prepared_instances)
    sorted_instances = sorted(
        prepared_instances,
        key=lambda instance: _grounding_sort_key(instance, row_bucket_size=row_bucket_size),
    )
    target = [
        {
            "label": str(instance["label"]),
            "bbox_2d": list(instance["bbox_2d"]),
        }
        for instance in sorted_instances
    ]
    return target, {
        "type": "row_bucket_center_v2",
        "coordinate_space": "bbox_2d",
        "row_anchor": "y_center",
        "row_bucket_size_2d": row_bucket_size,
        "order": ("row_bucket", "x1", "y1", "y2", "x2", "label"),
        "tie_break": "source_bbox_float",
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
    elif config.task.kind == "point":
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
        "system_prompt": config.prompt.system_prompt,
        "user_prompt": config.prompt.user_prompt,
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
    _write_readme(task_root / "sft" / "README.md", task=task, prompt=prompt, rows=rows)
    return len(rows)


def _write_readme(path: Path, *, task: TaskSpec, prompt: PromptConfig, rows: list[dict[str, Any]]) -> None:
    split_counts = Counter(str(row["extra"]["structured_extra"].get("split", "")) for row in rows)
    content = (
        f"# {task.name} SFT\n\n"
        f"- Task kind: `{task.kind}`\n"
        f"- Prompt id: `{prompt.prompt_id}`\n"
        f"- Rows in last converted split: `{len(rows)}`\n"
        f"- Last converted split counts from structured extra: `{dict(split_counts)}`\n"
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
    if not names:
        return list(TASKS)
    task_map = {task.name: task for task in TASKS}
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
