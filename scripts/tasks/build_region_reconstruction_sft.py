#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from shaft.codec.coordinates import quantize_qwen_bbox, quantize_qwen_point
from shaft.prompting import load_prompt_pool


@dataclass(frozen=True)
class TaskSpec:
    name: str
    label: str
    selection_path: Path
    source_root: Path
    prompt_path: Path
    source_kind: str


@dataclass(frozen=True)
class Selection:
    sample_id: str
    stem: str
    instance_index: int
    source_bbox: tuple[float, float, float, float]
    source_image: str
    source_json: str


@dataclass(frozen=True)
class WorkerConfig:
    spec: TaskSpec
    output_root: Path
    prompt_pool_id: str
    output_schema: str | None


@dataclass(frozen=True)
class WorkerResult:
    rows: tuple[tuple[str, str], ...]
    counts: dict[str, int]


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent) as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
        temp_path = Path(f.name)
    os.replace(temp_path, path)


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _bbox(value: Any) -> tuple[float, float, float, float]:
    if not isinstance(value, list | tuple) or len(value) != 4:
        raise ValueError(f"Invalid bbox: {value!r}")
    if not all(_is_number(item) for item in value):
        raise ValueError(f"Invalid bbox: {value!r}")
    x1, y1, x2, y2 = [float(item) for item in value]
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Degenerate bbox: {value!r}")
    return x1, y1, x2, y2


def _bbox_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    intersection_width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    intersection_height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = intersection_width * intersection_height
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _clip_bbox(
    bbox: tuple[float, float, float, float],
    *,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    return (
        min(max(bbox[0], 0.0), float(image_width)),
        min(max(bbox[1], 0.0), float(image_height)),
        min(max(bbox[2], 0.0), float(image_width)),
        min(max(bbox[3], 0.0), float(image_height)),
    )


def _resolve_source_instance(
    selection: Selection,
    *,
    layout: list[Any],
    label: str,
    image_width: int,
    image_height: int,
) -> tuple[int, dict[str, Any], tuple[float, float, float, float], bool]:
    def matches_label(instance: dict[str, Any]) -> bool:
        return instance.get("type", instance.get("label")) == label

    if 0 <= selection.instance_index < len(layout):
        indexed = layout[selection.instance_index]
        if isinstance(indexed, dict) and matches_label(indexed):
            indexed_bbox = _clip_bbox(
                _bbox(indexed.get("bbox")),
                image_width=image_width,
                image_height=image_height,
            )
            if _bbox_iou(selection.source_bbox, indexed_bbox) >= 0.90:
                return (
                    selection.instance_index,
                    indexed,
                    indexed_bbox,
                    indexed_bbox != selection.source_bbox,
                )

    candidates: list[tuple[float, int, dict[str, Any], tuple[float, float, float, float]]] = []
    for index, instance in enumerate(layout):
        if not isinstance(instance, dict) or not matches_label(instance):
            continue
        source_bbox = _clip_bbox(
            _bbox(instance.get("bbox")),
            image_width=image_width,
            image_height=image_height,
        )
        candidates.append(
            (_bbox_iou(selection.source_bbox, source_bbox), index, instance, source_bbox)
        )
    candidates.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    if not candidates or candidates[0][0] < 0.90:
        raise ValueError(
            f"Cannot match selected {label} bbox {selection.source_bbox} "
            f"at source index {selection.instance_index}"
        )
    best_iou, best_index, best_instance, best_bbox = candidates[0]
    if len(candidates) > 1 and candidates[1][0] == best_iou:
        raise ValueError(
            f"Ambiguous selected {label} bbox {selection.source_bbox}: "
            f"indices {best_index} and {candidates[1][1]}"
        )
    return best_index, best_instance, best_bbox, True


def _load_selections(spec: TaskSpec) -> list[Selection]:
    selections: list[Selection] = []
    seen_sample_ids: set[str] = set()
    with spec.selection_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            extra = row.get("extra")
            instances = row.get("instances")
            if (
                not isinstance(extra, dict)
                or not isinstance(instances, list)
                or len(instances) != 1
            ):
                raise ValueError(f"Invalid selection row: {spec.selection_path}:{line_no}")
            instance = instances[0]
            if not isinstance(instance, dict) or instance.get("label") != spec.label:
                raise ValueError(f"Wrong selection label: {spec.selection_path}:{line_no}")
            sample_id = str(row.get("sample_id") or "")
            if not sample_id or sample_id in seen_sample_ids:
                raise ValueError(f"Missing/duplicate sample id: {spec.selection_path}:{line_no}")
            seen_sample_ids.add(sample_id)
            source_json = str(extra.get("source_json") or "")
            source_image = str(extra.get("source_image") or "")
            stem = Path(source_json).stem
            source_bbox = _bbox(extra.get("source_bbox"))
            selections.append(
                Selection(
                    sample_id=sample_id,
                    stem=stem,
                    instance_index=int(extra["source_instance_index"]),
                    source_bbox=source_bbox,
                    source_image=source_image,
                    source_json=source_json,
                )
            )
    return selections


def _full_image_point(point: Any, *, image_width: int, image_height: int) -> list[int]:
    if not isinstance(point, list | tuple) or len(point) != 2:
        raise ValueError(f"Invalid point: {point!r}")
    if not all(_is_number(item) for item in point):
        raise ValueError(f"Invalid point: {point!r}")
    return quantize_qwen_point(point, width=image_width, height=image_height)


def _full_image_bbox(value: Any, *, image_width: int, image_height: int) -> list[int]:
    source = _bbox(value)
    return quantize_qwen_bbox(source, width=image_width, height=image_height)


def _full_image_corner(
    value: Any,
    *,
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Invalid corner: {value!r}")
    result = json.loads(json.dumps(value))
    for key in ("point", "start", "mid", "end"):
        if key in result:
            result[key] = _full_image_point(
                result[key],
                image_width=image_width,
                image_height=image_height,
            )
    return result


def _shape_parameters(
    parameters: dict[str, Any],
    *,
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    result = json.loads(json.dumps(parameters))
    for key in ("corners", "body_corners"):
        if isinstance(result.get(key), list):
            result[key] = [
                _full_image_corner(
                    corner,
                    image_width=image_width,
                    image_height=image_height,
                )
                for corner in result[key]
            ]
    if isinstance(result.get("body_bbox"), list):
        result["body_bbox"] = _full_image_bbox(
            result["body_bbox"],
            image_width=image_width,
            image_height=image_height,
        )
    tail = result.get("tail")
    if isinstance(tail, dict) and isinstance(tail.get("points"), list):
        tail["points"] = [
            _full_image_point(
                point,
                image_width=image_width,
                image_height=image_height,
            )
            for point in tail["points"]
        ]
    return result


def _line_parameters(
    parameters: dict[str, Any],
    *,
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    result = json.loads(json.dumps(parameters))
    points = result.get("points")
    if not isinstance(points, list) or not points:
        raise ValueError("Line parameters require points.")
    converted_segments: list[list[list[int]]] = []
    for segment_index, segment in enumerate(points):
        if not isinstance(segment, list) or not segment:
            raise ValueError(f"Invalid line point segment at index {segment_index}: {segment!r}")
        converted_segments.append(
            [
                _full_image_point(
                    point,
                    image_width=image_width,
                    image_height=image_height,
                )
                for point in segment
            ]
        )
    result["points"] = converted_segments
    return result


def _source_image_path(spec: TaskSpec, selection: Selection) -> Path:
    path = spec.source_root / selection.source_image
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


def _relative_image(path: Path, *, output_root: Path, task: str, kind: str) -> str:
    return os.path.relpath(path, start=(output_root / task / kind).resolve())


def _build_rows(
    *,
    config: WorkerConfig,
    selection: Selection,
    image_path: Path,
    image_width: int,
    image_height: int,
    source_bbox: tuple[float, float, float, float],
    target_parameters: dict[str, Any],
    source_instance_index: int | None = None,
) -> tuple[str, str]:
    spec = config.spec
    prompt_bbox = quantize_qwen_bbox(source_bbox, width=image_width, height=image_height)
    selection_manifest = os.path.relpath(
        spec.selection_path.resolve(),
        start=(config.output_root / spec.name).resolve(),
    )
    structured_extra = {
        "task": spec.name,
        "split": "train",
        "view_type": "full_image_bbox_conditioned",
        "source_dataset": spec.source_root.name,
        "source_json": selection.source_json,
        "source_image": selection.source_image,
        "source_instance_index": (
            selection.instance_index if source_instance_index is None else source_instance_index
        ),
        "selection_source_instance_index": selection.instance_index,
        "selection_source_bbox": list(selection.source_bbox),
        "source_bbox": list(source_bbox),
        "prompt_bbox_2d": prompt_bbox,
        "prompt_coordinate_space": "qwen_0_999_full_image",
        "target_coordinate_space": "qwen_0_999_full_image",
        "target_num_bins": 1000,
        "selection_manifest": selection_manifest,
        "augmentation": {"name": "none"},
    }
    target = {"type": spec.label, "parameters": target_parameters}
    structured = {
        "sample_id": selection.sample_id,
        "image_path": _relative_image(
            image_path,
            output_root=config.output_root,
            task=spec.name,
            kind="structured",
        ),
        "image_width": image_width,
        "image_height": image_height,
        "instances": [
            {
                "label": spec.label,
                "bbox": list(source_bbox),
                "parameters": target_parameters,
            }
        ],
        "extra": structured_extra,
    }
    sft_extra: dict[str, Any] = {
        "prompt_pool_id": config.prompt_pool_id,
        "source_sample_id": selection.stem,
        "source_type": (
            "synthetic_gt_standard_region"
            if spec.source_kind == "synthetic"
            else "reviewed_raw_region"
        ),
        "image_width": image_width,
        "image_height": image_height,
        "prompt_coordinate_space": "qwen_0_999_full_image",
        "target_coordinate_space": "qwen_0_999_full_image",
        "num_bins": 1000,
        "structured_extra": structured_extra,
    }
    if config.output_schema:
        sft_extra["output_schema"] = config.output_schema
    sft = {
        "image_path": _relative_image(
            image_path,
            output_root=config.output_root,
            task=spec.name,
            kind="sft",
        ),
        "sample_id": selection.sample_id,
        "dataset_name": spec.name,
        "system_prompt": "",
        "user_prompt": "",
        "prompt_args": {"bbox_2d": prompt_bbox},
        "target_text": _json_dumps(target),
        "extra": sft_extra,
    }
    return _json_dumps(structured), _json_dumps(sft)


def _build_synthetic_source(
    item: tuple[tuple[str, str], tuple[Selection, ...], WorkerConfig],
) -> WorkerResult:
    source_key, selections, config = item
    spec = config.spec
    source_json, source_image = source_key
    if any(
        (selection.source_json, selection.source_image) != source_key for selection in selections
    ):
        raise ValueError(f"Mixed source group: {source_key!r}")
    source_json_path = spec.source_root / source_json
    payload = json.loads(source_json_path.read_text(encoding="utf-8"))
    size = payload.get("size")
    layout = payload.get("layout")
    if not isinstance(size, list) or len(size) != 2 or not isinstance(layout, list):
        raise ValueError(f"Invalid gt_standard source: {source_json_path}")
    image_width, image_height = int(size[0]), int(size[1])
    image_path = _source_image_path(spec, selections[0])
    with Image.open(image_path) as image:
        if image.size != (image_width, image_height):
            raise ValueError(
                f"Image size mismatch for {source_image}: {image.size} != {tuple(size)}"
            )

    rows: list[tuple[str, str]] = []
    counts: Counter[str] = Counter()
    for selection in selections:
        source_index, source_instance, source_bbox, source_drift = _resolve_source_instance(
            selection,
            layout=layout,
            label=spec.label,
            image_width=image_width,
            image_height=image_height,
        )
        source_parameters = source_instance.get("parameters")
        if not isinstance(source_parameters, dict):
            raise ValueError(f"Missing source parameters: {source_json}:{selection.instance_index}")
        target_parameters = (
            _shape_parameters(
                source_parameters,
                image_width=image_width,
                image_height=image_height,
            )
            if spec.label == "shape"
            else _line_parameters(
                source_parameters,
                image_width=image_width,
                image_height=image_height,
            )
        )
        rows.append(
            _build_rows(
                config=config,
                selection=selection,
                image_path=image_path,
                image_width=image_width,
                image_height=image_height,
                source_bbox=source_bbox,
                target_parameters=target_parameters,
                source_instance_index=source_index,
            )
        )
        counts["rows"] += 1
        counts[f"label_{spec.label}"] += 1
        counts["source_bbox_drift"] += int(source_drift)
        counts["source_index_remap"] += int(source_index != selection.instance_index)
    return WorkerResult(tuple(rows), dict(counts))


def _build_image_source(
    item: tuple[tuple[str, str], tuple[Selection, ...], WorkerConfig],
) -> WorkerResult:
    source_key, selections, config = item
    spec = config.spec
    source_json, _ = source_key
    if any(
        (selection.source_json, selection.source_image) != source_key for selection in selections
    ):
        raise ValueError(f"Mixed source group: {source_key!r}")
    source_json_path = spec.source_root / source_json
    payload = json.loads(source_json_path.read_text(encoding="utf-8"))
    instances = payload.get("instances")
    if not isinstance(instances, list):
        raise ValueError(f"Invalid raw source: {source_json_path}")
    image_path = _source_image_path(spec, selections[0])
    with Image.open(image_path) as image:
        image_width, image_height = image.size
    annotated_size = (payload.get("image_width"), payload.get("image_height"))
    if all(_is_number(item) for item in annotated_size) and tuple(map(int, annotated_size)) != (
        image_width,
        image_height,
    ):
        raise ValueError(
            f"Image size mismatch for {source_json_path}: "
            f"{(image_width, image_height)} != {annotated_size}"
        )
    rows: list[tuple[str, str]] = []
    counts: Counter[str] = Counter()
    for selection in selections:
        source_index, source_instance, source_bbox, source_drift = _resolve_source_instance(
            selection,
            layout=instances,
            label=spec.label,
            image_width=image_width,
            image_height=image_height,
        )
        source_extra = source_instance.get("extra")
        source_parameters = (
            source_extra.get("parameters") if isinstance(source_extra, dict) else None
        )
        image_type = (
            source_parameters.get("image_type") if isinstance(source_parameters, dict) else None
        )
        if not isinstance(image_type, str) or not image_type:
            raise ValueError(f"Missing raw image_type: {source_json_path}:{source_index}")
        rows.append(
            _build_rows(
                config=config,
                selection=selection,
                image_path=image_path,
                image_width=image_width,
                image_height=image_height,
                source_bbox=source_bbox,
                target_parameters={"image_type": image_type},
                source_instance_index=source_index,
            )
        )
        counts["rows"] += 1
        counts[f"image_type_{image_type}"] += 1
        counts["source_bbox_drift"] += int(source_drift)
        counts["source_index_remap"] += int(source_index != selection.instance_index)
    return WorkerResult(tuple(rows), dict(counts))


def _prepare_staging_output(spec: TaskSpec, output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    task_root = Path(tempfile.mkdtemp(prefix=f".{spec.name}.staging.", dir=output_root))
    for kind in ("structured", "sft"):
        (task_root / kind).mkdir(parents=True, exist_ok=True)
        (task_root / kind / "val.jsonl").write_text("", encoding="utf-8")
    return task_root


def _publish_staging_output(
    staging_root: Path,
    task_root: Path,
    *,
    clean: bool,
) -> None:
    if clean:
        backup_root: Path | None = None
        if task_root.exists():
            backup_root = Path(
                tempfile.mkdtemp(prefix=f".{task_root.name}.backup.", dir=task_root.parent)
            )
            backup_root.rmdir()
            os.replace(task_root, backup_root)
        try:
            os.replace(staging_root, task_root)
        except BaseException:
            if backup_root is not None:
                os.replace(backup_root, task_root)
            raise
        if backup_root is not None:
            shutil.rmtree(backup_root, ignore_errors=True)
        return

    for relative_path in (
        Path("structured/train.jsonl"),
        Path("structured/val.jsonl"),
        Path("sft/train.jsonl"),
        Path("sft/val.jsonl"),
        Path("README.md"),
    ):
        destination = task_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging_root / relative_path, destination)
    shutil.rmtree(staging_root)


def _prompt_contract(path: Path) -> tuple[str, str | None]:
    prompts = load_prompt_pool(path)
    if not prompts:
        raise ValueError(f"Empty prompt pool: {path}")
    for prompt in prompts:
        if "bbox_2d" not in prompt.program.schema.names:
            raise ValueError(f"Prompt does not declare bbox_2d: {prompt.prompt_id}")
        if "[1,2,3,4]" not in prompt.render({"bbox_2d": [1, 2, 3, 4]}):
            raise ValueError(f"Prompt does not render bbox_2d: {prompt.prompt_id}")
    output_schema = prompts[0].metadata.get("output_schema")
    return prompts[0].prompt_id.rsplit(".", 1)[0], (
        str(output_schema) if output_schema is not None else None
    )


def _write_readme(spec: TaskSpec, task_root: Path, counts: Counter[str]) -> None:
    content = f"""# {spec.name}

- Input contract: full image plus `prompt_args.bbox_2d` in full-image Qwen `0..999` coordinates.
- Target contract: compact `{spec.label}` reconstruction JSON.
- Target geometry: Qwen `0..999` coordinates normalized against the same full image.
- Selection manifest: `{spec.selection_path}`; filtering and sampling are unchanged from the crop task.
- Source root: `{spec.source_root}`
- Prompt pool: `{spec.prompt_path}`
- Image policy: source full images are referenced directly; no crop image is generated or copied.
- Augmentation: none.
- Split: train only; validation remains empty.
- Rows: {counts["rows"]}
"""
    _atomic_write_text(task_root / "README.md", content)


def _build_task(spec: TaskSpec, *, output_root: Path, workers: int, clean: bool) -> Counter[str]:
    prompt_pool_id, output_schema = _prompt_contract(spec.prompt_path)
    config = WorkerConfig(
        spec=spec,
        output_root=output_root.resolve(),
        prompt_pool_id=prompt_pool_id,
        output_schema=output_schema,
    )
    selections = _load_selections(spec)
    by_source: dict[tuple[str, str], list[Selection]] = defaultdict(list)
    for selection in selections:
        by_source[(selection.source_json, selection.source_image)].append(selection)
    work_items = [
        (source_key, tuple(items), config) for source_key, items in sorted(by_source.items())
    ]
    worker = _build_synthetic_source if spec.source_kind == "synthetic" else _build_image_source
    task_root = output_root / spec.name
    staging_root = _prepare_staging_output(spec, output_root)
    counters: Counter[str] = Counter()
    try:
        with (
            (staging_root / "structured/train.jsonl").open(
                "w", encoding="utf-8"
            ) as structured_file,
            (staging_root / "sft/train.jsonl").open("w", encoding="utf-8") as sft_file,
            ProcessPoolExecutor(max_workers=workers) as executor,
        ):
            for result in executor.map(worker, work_items, chunksize=16):
                counters.update(result.counts)
                for structured, sft in result.rows:
                    structured_file.write(structured + "\n")
                    sft_file.write(sft + "\n")
        if counters["rows"] != len(selections):
            raise RuntimeError(
                f"{spec.name}: generated {counters['rows']} != selected {len(selections)}"
            )
        _write_readme(spec, staging_root, counters)
        _publish_staging_output(staging_root, task_root, clean=clean)
    except BaseException:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise
    return counters


def _task_specs(args: argparse.Namespace) -> dict[str, TaskSpec]:
    synthetic_root = Path(args.synthetic_root)
    raw_root = Path(args.raw_root)
    return {
        "shape_region_reconstruction": TaskSpec(
            "shape_region_reconstruction",
            "shape",
            Path(args.shape_selection),
            synthetic_root,
            Path(args.shape_prompt_pool),
            "synthetic",
        ),
        "line_region_reconstruction": TaskSpec(
            "line_region_reconstruction",
            "line",
            Path(args.line_selection),
            synthetic_root,
            Path(args.line_prompt_pool),
            "synthetic",
        ),
        "image_region_reconstruction": TaskSpec(
            "image_region_reconstruction",
            "image",
            Path(args.image_selection),
            raw_root,
            Path(args.image_prompt_pool),
            "real",
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build full-image bbox-conditioned region reconstruction SFT data."
    )
    parser.add_argument("--synthetic-root", default="data/regulated_layout_dataset_v8_20260709")
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--output-root", default="data")
    parser.add_argument(
        "--shape-selection", default="data/shape_reconstruction/structured/train.jsonl"
    )
    parser.add_argument(
        "--line-selection", default="data/line_reconstruction/structured/train.jsonl"
    )
    parser.add_argument(
        "--image-selection", default="data/image_reconstruction/structured/train.jsonl"
    )
    parser.add_argument(
        "--shape-prompt-pool",
        default="configs/prompts/pools/shape_region_reconstruction.v5.2.yaml",
    )
    parser.add_argument(
        "--line-prompt-pool",
        default="configs/prompts/pools/line_region_reconstruction.v5.2.yaml",
    )
    parser.add_argument(
        "--image-prompt-pool",
        default="configs/prompts/pools/image_region_reconstruction.v5.2.yaml",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=[
            "shape_region_reconstruction",
            "line_region_reconstruction",
            "image_region_reconstruction",
        ],
    )
    parser.add_argument("--workers", type=int, default=50)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("workers must be positive")
    specs = _task_specs(args)
    unknown = sorted(set(args.tasks) - set(specs))
    if unknown:
        parser.error(f"unknown tasks: {unknown}")
    summary = {}
    for task in args.tasks:
        summary[task] = dict(
            sorted(
                _build_task(
                    specs[task],
                    output_root=Path(args.output_root),
                    workers=int(args.workers),
                    clean=bool(args.clean),
                ).items()
            )
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
