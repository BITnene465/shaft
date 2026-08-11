from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import random
import re
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable
from urllib import request

from PIL import Image, ImageDraw, ImageFont

from scripts.tasks.build_context_reconstruction_sft import _expand_crop, _local_bbox
from shaft.codec import decode_with_codec
from shaft.codec.coordinates import (
    dequantize_qwen_point,
    quantize_qwen_bbox,
)
from shaft.prompting import ShaftPromptTemplate, load_prompt_pool
from shaft.utils.qwen_pixel_budget import image_to_data_url_with_qwen_pixel_budget


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_JSON = ROOT / "data" / "raw" / "json_20260706"
DEFAULT_IMAGE_DIR = ROOT / "data" / "raw" / "images"
DEFAULT_OUTPUT = ROOT / "temp" / "json_20260706_line_ckpt10000"
DEFAULT_MODEL = ROOT / "outputs" / "qwen3vl-sft" / "4b" / "banana-v5.3" / "checkpoint-10000"
RECONSTRUCTION_PROMPT = (
    ROOT / "configs" / "prompts" / "pools" / "line_context_reconstruction.v5.3.yaml"
)
POINTS_PROMPT = ROOT / "configs" / "prompts" / "pools" / "line_context_points.v5.3.yaml"
ARROW_HEAD_PROMPT = (
    ROOT / "configs" / "prompts" / "pools" / "line_arrow_head_recovery.v5.3.yaml"
)
ARROW_END_PROMPT = (
    ROOT / "configs" / "prompts" / "pools" / "line_arrow_end_type_recovery.v5.3.yaml"
)
TARGET_LABELS = {"arrow", "line"}
RECONSTRUCTION_VARIANTS = (
    "main",
    "visible_object",
    "schema_first",
    "attribute_extraction",
    "minimal_contract",
)
POINTS_VARIANTS = (
    "main",
    "visible_path",
    "schema_first",
    "path_extraction",
    "minimal_contract",
)
ARROW_HEAD_VARIANTS = ("main", "strict", "forced_choice")
ARROW_TYPES = {"none", "line", "stealth", "triangle", "pointy", "tee", "circle"}
NON_NONE_ARROW_TYPES = ARROW_TYPES - {"none"}
HEX_COLOR = re.compile(r"^#[0-9A-Fa-f]{6}$")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{random.randrange(1 << 30):08x}.tmp"
    )
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{random.randrange(1 << 30):08x}.tmp"
    )
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def stable_hash(text: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{text}".encode("utf-8")).hexdigest()


def image_map(image_dir: Path) -> dict[str, Path]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in image_dir.iterdir():
        if path.is_file():
            grouped[path.stem].append(path)
    duplicated = {stem: paths for stem, paths in grouped.items() if len(paths) != 1}
    if duplicated:
        examples = {stem: [str(path) for path in paths] for stem, paths in list(duplicated.items())[:5]}
        raise RuntimeError(f"image stems are not unique: {examples}")
    return {stem: paths[0] for stem, paths in grouped.items()}


def bbox(value: Any) -> tuple[float, float, float, float]:
    if not isinstance(value, list | tuple) or len(value) != 4:
        raise ValueError(f"invalid bbox: {value!r}")
    if not all(isinstance(item, int | float) and not isinstance(item, bool) for item in value):
        raise ValueError(f"invalid bbox: {value!r}")
    x1, y1, x2, y2 = [float(item) for item in value]
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"degenerate bbox: {value!r}")
    return x1, y1, x2, y2


def clean_medium_context_view(
    source_bbox: tuple[float, float, float, float],
    *,
    image_width: int,
    image_height: int,
    sample_id: str,
    seed: int,
    min_crop_size: int = 16,
    max_aspect_ratio: float = 60.0,
) -> tuple[tuple[int, int, int, int], list[int], list[float]]:
    x1, y1, x2, y2 = source_bbox
    if x1 < 0 or y1 < 0 or x2 > image_width or y2 > image_height:
        raise ValueError(f"bbox outside image: {source_bbox}")
    rng = random.Random(f"{seed}:clean-medium:{sample_id}")
    width, height = x2 - x1, y2 - y1
    longest = max(width, height)
    reference_x = max(width, min(64.0, longest * 0.10))
    reference_y = max(height, min(64.0, longest * 0.10))
    ratios = tuple(rng.uniform(0.45, 0.90) for _ in range(4))
    crop_box = (
        max(0, int(math.floor(x1 - ratios[0] * reference_x))),
        max(0, int(math.floor(y1 - ratios[1] * reference_y))),
        min(image_width, int(math.ceil(x2 + ratios[2] * reference_x))),
        min(image_height, int(math.ceil(y2 + ratios[3] * reference_y))),
    )
    crop_box = _expand_crop(
        crop_box,
        image_width=image_width,
        image_height=image_height,
        min_crop_size=min_crop_size,
        max_aspect_ratio=max_aspect_ratio,
    )
    local = _local_bbox(source_bbox, crop_box)
    crop_width = crop_box[2] - crop_box[0]
    crop_height = crop_box[3] - crop_box[1]
    prompt_bbox = quantize_qwen_bbox(
        local,
        width=crop_width,
        height=crop_height,
        minimum_extent_bins=1,
    )
    return crop_box, prompt_bbox, [round(value, 6) for value in ratios]


def scan_targets(input_json_dir: Path, image_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    images = image_map(image_dir)
    records: list[dict[str, Any]] = []
    label_counts: Counter[str] = Counter()
    target_images: set[str] = set()
    json_paths = sorted(input_json_dir.glob("*.json"))
    for json_path in json_paths:
        document = read_json(json_path)
        image_path = images.get(json_path.stem)
        if image_path is None:
            raise FileNotFoundError(f"missing source image for {json_path}")
        image_width = int(document["image_width"])
        image_height = int(document["image_height"])
        for instance_index, instance in enumerate(document.get("instances") or []):
            label = str(instance.get("label") or "").lower()
            label_counts[label] += 1
            if label not in TARGET_LABELS:
                continue
            source_bbox = bbox(instance.get("bbox"))
            if source_bbox[2] > image_width or source_bbox[3] > image_height:
                raise ValueError(f"bbox outside image: {json_path}/{instance_index}")
            target_images.add(json_path.stem)
            target_id = f"{json_path.stem}__line_{instance_index:05d}"
            records.append(
                {
                    "target_id": target_id,
                    "json_name": json_path.name,
                    "image_stem": json_path.stem,
                    "instance_index": instance_index,
                    "source_label": label,
                    "source_bbox": list(source_bbox),
                    "image_width": image_width,
                    "image_height": image_height,
                    "image_path": str(image_path.relative_to(ROOT)),
                }
            )
    summary = {
        "json_files": len(json_paths),
        "target_images": len(target_images),
        "target_instances": len(records),
        "arrow_instances": sum(record["source_label"] == "arrow" for record in records),
        "line_instances": sum(record["source_label"] == "line" for record in records),
        "all_label_counts": dict(label_counts),
    }
    return records, summary


def select_pilot(records: list[dict[str, Any]], *, limit: int, seed: int) -> list[dict[str, Any]]:
    if limit <= 0 or limit >= len(records):
        return records
    grouped = {
        label: sorted(
            (record for record in records if record["source_label"] == label),
            key=lambda record: stable_hash(str(record["target_id"]), seed),
        )
        for label in ("arrow", "line")
    }
    line_quota = min(len(grouped["line"]), limit // 2)
    arrow_quota = min(len(grouped["arrow"]), limit - line_quota)
    remaining = limit - line_quota - arrow_quota
    if remaining:
        arrow_quota += min(remaining, len(grouped["arrow"]) - arrow_quota)
    selected = grouped["arrow"][:arrow_quota] + grouped["line"][:line_quota]
    return sorted(selected, key=lambda record: str(record["target_id"]))


def prepare_image(arguments: tuple[list[dict[str, Any]], str, str, int]) -> list[dict[str, Any]]:
    records, image_path_text, output_root_text, seed = arguments
    image_path = Path(image_path_text)
    output_root = Path(output_root_text)
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    prepared = []
    try:
        for record in records:
            crop_box, prompt_bbox, ratios = clean_medium_context_view(
                tuple(float(value) for value in record["source_bbox"]),
                image_width=int(record["image_width"]),
                image_height=int(record["image_height"]),
                sample_id=str(record["target_id"]),
                seed=seed,
            )
            shard = hashlib.sha1(str(record["target_id"]).encode("utf-8")).hexdigest()[:2]
            crop_path = output_root / "crops" / shard / f"{record['target_id']}.png"
            if not crop_path.exists():
                crop_path.parent.mkdir(parents=True, exist_ok=True)
                crop = image.crop(crop_box)
                try:
                    crop.save(crop_path, format="PNG", compress_level=1)
                finally:
                    crop.close()
            prepared.append(
                {
                    **record,
                    "crop_path": str(crop_path.relative_to(ROOT)),
                    "crop_box": list(crop_box),
                    "crop_width": crop_box[2] - crop_box[0],
                    "crop_height": crop_box[3] - crop_box[1],
                    "proposal_bbox_2d": prompt_bbox,
                    "padding_ratios": ratios,
                    "proposal_policy": "exact_source_bbox",
                    "padding_policy": "deterministic_clean_medium_0.45_0.90",
                }
            )
    finally:
        image.close()
    return prepared


def prepare(args: argparse.Namespace) -> None:
    records, inventory = scan_targets(args.input_json_dir, args.image_dir)
    selected = select_pilot(records, limit=args.limit, seed=args.seed) if args.limit else records
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in selected:
        grouped[str(record["image_stem"])].append(record)
    jobs = [
        (group, str(ROOT / group[0]["image_path"]), str(args.output_root), args.seed)
        for _, group in sorted(grouped.items())
    ]
    prepared: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        for completed, output in enumerate(executor.map(prepare_image, jobs, chunksize=1), start=1):
            prepared.extend(output)
            if completed == 1 or completed % 500 == 0 or completed == len(jobs):
                print(
                    json.dumps(
                        {"prepared_images": completed, "total_images": len(jobs), "targets": len(prepared)}
                    ),
                    flush=True,
                )
    prepared.sort(key=lambda record: str(record["target_id"]))
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "manifest.jsonl"
    atomic_write_text(
        manifest_path,
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in prepared),
    )
    manifest = {
        "schema_version": 1,
        "input_json_dir": str(args.input_json_dir.relative_to(ROOT)),
        "image_dir": str(args.image_dir.relative_to(ROOT)),
        "output_root": str(args.output_root.relative_to(ROOT)),
        "inventory": inventory,
        "selected_targets": len(prepared),
        "selected_images": len(grouped),
        "selection": "balanced_pilot" if args.limit else "all",
        "limit": args.limit or None,
        "seed": args.seed,
        "crop_policy": "clean exact-bbox proposal + deterministic medium contextual padding",
        "coordinate_space": "crop-local Qwen integer 0..999",
    }
    atomic_write_json(args.output_root / "input_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def read_manifest(output_root: Path) -> list[dict[str, Any]]:
    path = output_root / "manifest.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"prepare first: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def valid_point(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(item, int) and not isinstance(item, bool) and 0 <= item <= 999 for item in value)
    )


def validate_segments(value: Any) -> list[str]:
    if not isinstance(value, list) or not value:
        return ["points:must_be_nonempty_segment_list"]
    errors = []
    for segment_index, segment in enumerate(value):
        if not isinstance(segment, list) or len(segment) < 2:
            errors.append(f"points[{segment_index}]:must_have_at_least_two_points")
            continue
        if len({tuple(point) for point in segment if valid_point(point)}) < 2:
            errors.append(f"points[{segment_index}]:must_have_two_distinct_points")
        for point_index, point in enumerate(segment):
            if not valid_point(point):
                errors.append(f"points[{segment_index}][{point_index}]:invalid")
    return errors


def validate_is_single(is_single: Any, points: Any) -> list[str]:
    if not isinstance(is_single, bool) or not isinstance(points, list):
        return []
    if is_single and len(points) != 1:
        return ["is_single:true_requires_exactly_one_segment"]
    if not is_single and len(points) <= 1:
        return ["is_single:false_requires_multiple_segments"]
    return []


def valid_color(value: Any) -> bool:
    if isinstance(value, str):
        return bool(HEX_COLOR.fullmatch(value))
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(item, str) and HEX_COLOR.fullmatch(item) for item in value)
    )


def validate_reconstruction(
    payload: Any,
    *,
    source_label: str,
    enforce_source_prior: bool = True,
) -> list[str]:
    if not isinstance(payload, dict) or set(payload) != {"type", "parameters"}:
        return ["root:schema_mismatch"]
    if payload.get("type") != "line" or not isinstance(payload.get("parameters"), dict):
        return ["root:expected_line_with_parameters"]
    parameters = payload["parameters"]
    required = {
        "line_type",
        "line_style",
        "is_single",
        "points",
        "begin_arrow",
        "end_arrow",
        "dash_style",
        "fill_color",
    }
    allowed = required | {"corner_style", "has_border", "border_style", "border_color"}
    errors = []
    if not required.issubset(parameters):
        errors.append(f"parameters:missing={sorted(required - set(parameters))}")
    if set(parameters) - allowed:
        errors.append(f"parameters:extra={sorted(set(parameters) - allowed)}")
    if parameters.get("line_type") not in {"straight", "curved"}:
        errors.append("line_type:invalid")
    line_style = parameters.get("line_style")
    if line_style not in {"path", "shape"}:
        errors.append("line_style:invalid")
    if not isinstance(parameters.get("is_single"), bool):
        errors.append("is_single:invalid")
    errors.extend(validate_segments(parameters.get("points")))
    errors.extend(validate_is_single(parameters.get("is_single"), parameters.get("points")))
    if parameters.get("begin_arrow") not in ARROW_TYPES:
        errors.append("begin_arrow:invalid")
    if parameters.get("end_arrow") not in ARROW_TYPES:
        errors.append("end_arrow:invalid")
    if parameters.get("dash_style") not in {"solid", "dash"}:
        errors.append("dash_style:invalid")
    if not valid_color(parameters.get("fill_color")):
        errors.append("fill_color:invalid")
    if "corner_style" in parameters and parameters["corner_style"] not in {"sharp", "round"}:
        errors.append("corner_style:invalid")
    border_keys = {"has_border", "border_style", "border_color"}
    if line_style == "path" and border_keys & set(parameters):
        errors.append("path:border_fields_not_allowed")
    if line_style == "shape":
        if not isinstance(parameters.get("has_border"), bool):
            errors.append("shape:has_border_required")
        elif parameters["has_border"]:
            if parameters.get("border_style") not in {"solid", "dash"}:
                errors.append("border_style:invalid")
            if not isinstance(parameters.get("border_color"), str) or not HEX_COLOR.fullmatch(
                parameters["border_color"]
            ):
                errors.append("border_color:invalid")
        elif {"border_style", "border_color"} & set(parameters):
            errors.append("shape:no_border_must_omit_border_fields")
    if (
        enforce_source_prior
        and source_label == "arrow"
        and parameters.get("begin_arrow") == parameters.get("end_arrow") == "none"
    ):
        errors.append("source_prior:arrow_requires_at_least_one_head")
    return errors


def normalize_attribute_parameters(
    payload: Any,
    *,
    source_label: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    if isinstance(payload, dict) and payload.get("type") == "line":
        raw_parameters = payload.get("parameters")
    else:
        raw_parameters = payload
    if not isinstance(raw_parameters, dict):
        return None, ["attributes:expected_object"]
    attribute_keys = {
        "line_type",
        "line_style",
        "begin_arrow",
        "end_arrow",
        "dash_style",
        "fill_color",
        "corner_style",
        "has_border",
        "border_style",
        "border_color",
    }
    parameters = {
        key: json.loads(json.dumps(value))
        for key, value in raw_parameters.items()
        if key in attribute_keys
    }
    if parameters.get("begin_arrow") == "round":
        parameters["begin_arrow"] = "circle"
    if parameters.get("end_arrow") == "round":
        parameters["end_arrow"] = "circle"
    if parameters.get("line_style") == "path":
        for key in ("has_border", "border_style", "border_color"):
            parameters.pop(key, None)
    if parameters.get("line_style") == "shape" and parameters.get("has_border") is False:
        parameters.pop("border_style", None)
        parameters.pop("border_color", None)
    if source_label == "line":
        parameters["begin_arrow"] = "none"
        parameters["end_arrow"] = "none"
    candidate = {
        "type": "line",
        "parameters": {
            **parameters,
            "is_single": True,
            "points": [[[0, 0], [999, 999]]],
        },
    }
    errors = validate_reconstruction(
        candidate,
        source_label=source_label,
        enforce_source_prior=False,
    )
    return (parameters if not errors else None), errors


def select_reconstruction_attributes(
    reconstruction: dict[str, Any],
    *,
    source_label: str,
) -> dict[str, Any] | None:
    attempts = reconstruction.get("attempts") or []
    candidates = attempts or [reconstruction.get("selected") or {}]
    for attempt_index, attempt in enumerate(candidates):
        parameters, errors = normalize_attribute_parameters(
            attempt.get("prediction"),
            source_label=source_label,
        )
        if not errors and parameters is not None:
            prediction = attempt.get("prediction") or {}
            raw_parameters = prediction.get("parameters") or {}
            return {
                "parameters": parameters,
                "source": "line_context_reconstruction",
                "variant": attempt.get("variant"),
                "attempt_index": attempt_index,
                "reference_is_single": raw_parameters.get("is_single"),
                "reference_points": raw_parameters.get("points"),
            }
    return None


def validate_arrow_heads(payload: Any) -> list[str]:
    if not isinstance(payload, dict) or set(payload) != {"begin_arrow", "end_arrow"}:
        return ["root:expected_only_begin_arrow_and_end_arrow"]
    errors = []
    if payload.get("begin_arrow") not in ARROW_TYPES:
        errors.append("begin_arrow:invalid")
    if payload.get("end_arrow") not in ARROW_TYPES:
        errors.append("end_arrow:invalid")
    if payload.get("begin_arrow") == payload.get("end_arrow") == "none":
        errors.append("source_prior:arrow_requires_at_least_one_head")
    return errors


def validate_forced_end_arrow(payload: Any) -> list[str]:
    if not isinstance(payload, dict) or set(payload) != {"end_arrow"}:
        return ["root:expected_only_end_arrow"]
    if payload.get("end_arrow") not in NON_NONE_ARROW_TYPES:
        return ["end_arrow:must_be_non_none_arrow_type"]
    return []


def validate_points(payload: Any) -> list[str]:
    if not isinstance(payload, dict) or set(payload) != {"type", "parameters"}:
        return ["root:schema_mismatch"]
    if payload.get("type") != "line" or not isinstance(payload.get("parameters"), dict):
        return ["root:expected_line_with_parameters"]
    parameters = payload["parameters"]
    errors = []
    if set(parameters) != {"is_single", "points"}:
        errors.append("parameters:must_only_contain_is_single_and_points")
    if not isinstance(parameters.get("is_single"), bool):
        errors.append("is_single:invalid")
    errors.extend(validate_segments(parameters.get("points")))
    errors.extend(validate_is_single(parameters.get("is_single"), parameters.get("points")))
    return errors


def geometry_metrics(points: Any, proposal_bbox: list[int]) -> dict[str, Any]:
    if not isinstance(points, list):
        return {"valid": False, "score": -1.0, "fraction_inside": 0.0}
    flattened = [point for segment in points if isinstance(segment, list) for point in segment if valid_point(point)]
    if not flattened:
        return {"valid": False, "score": -1.0, "fraction_inside": 0.0}
    x1, y1, x2, y2 = [float(value) for value in proposal_bbox]
    margin_x = max(35.0, (x2 - x1) * 0.20)
    margin_y = max(35.0, (y2 - y1) * 0.20)
    expanded = (
        max(0.0, x1 - margin_x),
        max(0.0, y1 - margin_y),
        min(999.0, x2 + margin_x),
        min(999.0, y2 + margin_y),
    )
    inside = sum(expanded[0] <= point[0] <= expanded[2] and expanded[1] <= point[1] <= expanded[3] for point in flattened)
    fraction_inside = inside / len(flattened)
    px1 = min(point[0] for point in flattened)
    py1 = min(point[1] for point in flattened)
    px2 = max(point[0] for point in flattened)
    py2 = max(point[1] for point in flattened)
    intersects = min(px2, expanded[2]) >= max(px1, expanded[0]) and min(py2, expanded[3]) >= max(py1, expanded[1])
    proposal_center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    point_center = ((px1 + px2) / 2.0, (py1 + py2) / 2.0)
    center_distance = math.dist(proposal_center, point_center) / math.sqrt(2 * 999**2)
    score = fraction_inside - 0.15 * center_distance
    return {
        "valid": bool(fraction_inside >= 0.5 and intersects),
        "score": round(score, 6),
        "fraction_inside": round(fraction_inside, 6),
        "center_distance_normalized": round(center_distance, 6),
        "point_bbox_2d": [px1, py1, px2, py2],
    }


def prompt_variants(path: Path) -> dict[str, ShaftPromptTemplate]:
    prompts = load_prompt_pool(path)
    return {prompt.variant_id: prompt for prompt in prompts}


def prior_text(source_label: str, task: str) -> str:
    if task == "reconstruction" and source_label == "line":
        return (
            "\nTrusted source-label prior: this target is a headless line. "
            "Set both begin_arrow and end_arrow to none."
        )
    if task == "reconstruction" and source_label == "arrow":
        return (
            "\nTrusted source-label prior: this target is an arrow. At least one of "
            "begin_arrow or end_arrow must be non-none; inspect whether it is single- or double-headed."
        )
    if source_label == "arrow":
        return (
            "\nTrusted source-label prior: this target is an arrow. For a one-way arrow, keep "
            "the center-path order from tail to arrowhead."
        )
    return "\nTrusted source-label prior: this target is a headless line; trace that line only."


def call_vllm(
    *,
    endpoint: str,
    served_model: str,
    image_url: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    timeout_s: float,
) -> tuple[str, str | None, float]:
    payload = {
        "model": served_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
    }
    http_request = request.Request(
        endpoint.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with request.urlopen(http_request, timeout=timeout_s) as response:
        response_payload = json.loads(response.read().decode("utf-8"))
    choices = response_payload.get("choices") or []
    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    content = message.get("content", "")
    if isinstance(content, list):
        content = "".join(str(item.get("text") or "") for item in content if isinstance(item, dict))
    return str(content), choice.get("finish_reason"), (time.perf_counter() - started) * 1000.0


def infer_head(
    *,
    task: str,
    variants: tuple[str, ...],
    prompts: dict[str, ShaftPromptTemplate],
    record: dict[str, Any],
    image_url: str,
    endpoint: str,
    served_model: str,
    max_tokens: int,
    timeout_s: float,
) -> dict[str, Any]:
    attempts = []
    for variant_id in variants:
        prompt = prompts[variant_id]
        user_prompt = prompt.render({"proposal_bbox_2d": record["proposal_bbox_2d"]}) + prior_text(
            str(record["source_label"]), task
        )
        raw_text = ""
        finish_reason = None
        latency_ms = 0.0
        request_error = None
        for request_attempt in range(1, 4):
            try:
                raw_text, finish_reason, latency_ms = call_vllm(
                    endpoint=endpoint,
                    served_model=served_model,
                    image_url=image_url,
                    system_prompt=prompt.system_prompt,
                    user_prompt=user_prompt,
                    max_tokens=max_tokens,
                    timeout_s=timeout_s,
                )
                request_error = None
                break
            except Exception as exc:  # noqa: BLE001
                request_error = repr(exc)
                if request_attempt < 3:
                    time.sleep(request_attempt)
        decoded = decode_with_codec("json_any", raw_text) if raw_text else None
        parsed = decoded.parsed if decoded is not None and decoded.valid else None
        errors = (
            validate_reconstruction(
                parsed,
                source_label=str(record["source_label"]),
                enforce_source_prior=False,
            )
            if task == "reconstruction"
            else validate_points(parsed)
        )
        geometry = geometry_metrics(
            ((parsed or {}).get("parameters") or {}).get("points"),
            record["proposal_bbox_2d"],
        )
        attempt = {
            "variant": variant_id,
            "prompt_id": prompt.prompt_id,
            "raw_text": raw_text,
            "raw_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
            "prediction": parsed,
            "json_valid": bool(decoded is not None and decoded.valid),
            "partial": bool(decoded is not None and decoded.partial),
            "decode_error": decoded.error_type if decoded is not None else request_error,
            "contract_errors": errors,
            "contract_valid": not errors,
            "geometry": geometry,
            "finish_reason": finish_reason,
            "latency_ms": round(latency_ms, 3),
            "request_error": request_error,
        }
        attempts.append(attempt)
        if not errors and (task == "reconstruction" or geometry["valid"]):
            break
    acceptable = [
        attempt
        for attempt in attempts
        if attempt["contract_valid"] and (task == "reconstruction" or attempt["geometry"]["valid"])
    ]
    contract_valid = [attempt for attempt in attempts if attempt["contract_valid"]]
    pool = acceptable or contract_valid or attempts
    selected = max(pool, key=lambda attempt: float(attempt["geometry"]["score"]))
    return {"task": task, "selected": selected, "attempts": attempts}


def infer_arrow_heads(
    *,
    prompts: dict[str, ShaftPromptTemplate],
    record: dict[str, Any],
    image_url: str,
    endpoint: str,
    served_model: str,
    max_tokens: int,
    timeout_s: float,
) -> dict[str, Any]:
    attempts = []
    for variant_id in ARROW_HEAD_VARIANTS:
        prompt = prompts[variant_id]
        raw_text = ""
        finish_reason = None
        latency_ms = 0.0
        request_error = None
        for request_attempt in range(1, 4):
            try:
                raw_text, finish_reason, latency_ms = call_vllm(
                    endpoint=endpoint,
                    served_model=served_model,
                    image_url=image_url,
                    system_prompt=prompt.system_prompt,
                    user_prompt=prompt.render(
                        {"proposal_bbox_2d": record["proposal_bbox_2d"]}
                    ),
                    max_tokens=min(max_tokens, 256),
                    timeout_s=timeout_s,
                )
                request_error = None
                break
            except Exception as exc:  # noqa: BLE001
                request_error = repr(exc)
                if request_attempt < 3:
                    time.sleep(request_attempt)
        decoded = decode_with_codec("json_any", raw_text) if raw_text else None
        parsed = decoded.parsed if decoded is not None and decoded.valid else None
        errors = validate_arrow_heads(parsed)
        attempt = {
            "variant": variant_id,
            "prompt_id": prompt.prompt_id,
            "raw_text": raw_text,
            "raw_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
            "prediction": parsed,
            "json_valid": bool(decoded is not None and decoded.valid),
            "partial": bool(decoded is not None and decoded.partial),
            "decode_error": decoded.error_type if decoded is not None else request_error,
            "contract_errors": errors,
            "contract_valid": not errors,
            "finish_reason": finish_reason,
            "latency_ms": round(latency_ms, 3),
            "request_error": request_error,
        }
        attempts.append(attempt)
        if not errors:
            break
    valid = [attempt for attempt in attempts if attempt["contract_valid"]]
    selected = valid[0] if valid else attempts[-1]
    return {"task": "arrow_head_recovery", "selected": selected, "attempts": attempts}


def infer_forced_end_arrow(
    *,
    prompt: ShaftPromptTemplate,
    record: dict[str, Any],
    image_url: str,
    endpoint: str,
    served_model: str,
    timeout_s: float,
) -> dict[str, Any]:
    raw_text = ""
    finish_reason = None
    latency_ms = 0.0
    request_error = None
    for request_attempt in range(1, 4):
        try:
            raw_text, finish_reason, latency_ms = call_vllm(
                endpoint=endpoint,
                served_model=served_model,
                image_url=image_url,
                system_prompt=prompt.system_prompt,
                user_prompt=prompt.render(
                    {"proposal_bbox_2d": record["proposal_bbox_2d"]}
                ),
                max_tokens=128,
                timeout_s=timeout_s,
            )
            request_error = None
            break
        except Exception as exc:  # noqa: BLE001
            request_error = repr(exc)
            if request_attempt < 3:
                time.sleep(request_attempt)
    decoded = decode_with_codec("json_any", raw_text) if raw_text else None
    parsed = decoded.parsed if decoded is not None and decoded.valid else None
    errors = validate_forced_end_arrow(parsed)
    selected = {
        "variant": prompt.variant_id,
        "prompt_id": prompt.prompt_id,
        "raw_text": raw_text,
        "raw_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
        "prediction": parsed,
        "json_valid": bool(decoded is not None and decoded.valid),
        "partial": bool(decoded is not None and decoded.partial),
        "decode_error": decoded.error_type if decoded is not None else request_error,
        "contract_errors": errors,
        "contract_valid": not errors,
        "finish_reason": finish_reason,
        "latency_ms": round(latency_ms, 3),
        "request_error": request_error,
    }
    return {"task": "forced_end_arrow_recovery", "selected": selected, "attempts": [selected]}


def endpoint_cost(first: list[Any], second: list[Any]) -> float:
    return math.dist((float(first[0]), float(first[1])), (float(second[0]), float(second[1])))


def align_points_to_reconstruction(
    point_segments: list[list[list[int]]],
    reconstruction_segments: Any,
    *,
    source_label: str,
) -> tuple[list[list[list[int]]], bool]:
    copied = json.loads(json.dumps(point_segments))
    if source_label != "arrow" or len(copied) != 1:
        return copied, False
    if not isinstance(reconstruction_segments, list) or len(reconstruction_segments) != 1:
        return copied, False
    point_segment = copied[0]
    reconstruction_segment = reconstruction_segments[0]
    if len(point_segment) < 2 or not isinstance(reconstruction_segment, list) or len(reconstruction_segment) < 2:
        return copied, False
    same = endpoint_cost(point_segment[0], reconstruction_segment[0]) + endpoint_cost(
        point_segment[-1], reconstruction_segment[-1]
    )
    reversed_cost = endpoint_cost(point_segment[-1], reconstruction_segment[0]) + endpoint_cost(
        point_segment[0], reconstruction_segment[-1]
    )
    if reversed_cost + 1e-9 < same:
        copied[0].reverse()
        return copied, True
    return copied, False


def proposal_axis_geometry(proposal_bbox_2d: list[int]) -> dict[str, Any]:
    x1, y1, x2, y2 = [int(value) for value in proposal_bbox_2d]
    if x2 - x1 >= y2 - y1:
        center = int(round((y1 + y2) / 2))
        points = [[[x1, center], [x2, center]]]
    else:
        center = int(round((x1 + x2) / 2))
        points = [[[center, y1], [center, y2]]]
    return {"is_single": True, "points": points}


def fuse_heads(
    record: dict[str, Any],
    reconstruction: dict[str, Any],
    points: dict[str, Any],
    arrow_head_recovery: dict[str, Any] | None = None,
    arrow_head_recovery_full_image: dict[str, Any] | None = None,
    forced_end_arrow_recovery: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reconstruction_selected = reconstruction["selected"]
    points_selected = points["selected"]
    attribute_selection = select_reconstruction_attributes(
        reconstruction,
        source_label=str(record["source_label"]),
    )
    if attribute_selection is None:
        return {"status": "unresolved", "reason": "no_contract_valid_reconstruction"}
    reconstruction_parameters = attribute_selection["parameters"]
    points_contract = points_selected["contract_valid"]
    points_geometry = points_selected["geometry"]["valid"]
    reconstruction_geometry = (
        reconstruction_selected["contract_valid"]
        and reconstruction_selected["geometry"]["valid"]
    )
    reconstruction_selected_parameters = (
        (reconstruction_selected.get("prediction") or {}).get("parameters") or {}
    )
    if points_contract and points_geometry:
        geometry_source = "line_context_points"
        selected_points = points_selected["prediction"]["parameters"]
    elif reconstruction_geometry:
        geometry_source = "line_context_reconstruction_fallback"
        selected_points = {
            "is_single": reconstruction_selected_parameters["is_single"],
            "points": reconstruction_selected_parameters["points"],
        }
    elif points_contract:
        geometry_source = "line_context_points_geometry_warning"
        selected_points = points_selected["prediction"]["parameters"]
    else:
        geometry_source = "source_bbox_axis_high_risk_fallback"
        selected_points = proposal_axis_geometry(record["proposal_bbox_2d"])
    aligned, reversed_for_endpoint_alignment = align_points_to_reconstruction(
        selected_points["points"],
        attribute_selection.get("reference_points"),
        source_label=str(record["source_label"]),
    )
    parameters = json.loads(json.dumps(reconstruction_parameters))
    parameters["is_single"] = bool(selected_points["is_single"])
    parameters["points"] = aligned
    if record["source_label"] == "line":
        parameters["begin_arrow"] = "none"
        parameters["end_arrow"] = "none"
        arrow_head_source = "source_line_hard_prior"
    else:
        crop_recovery_selected = (arrow_head_recovery or {}).get("selected") or {}
        full_recovery_selected = (arrow_head_recovery_full_image or {}).get("selected") or {}
        forced_end_selected = (forced_end_arrow_recovery or {}).get("selected") or {}
        recovery_selected = (
            crop_recovery_selected
            if crop_recovery_selected.get("contract_valid")
            else full_recovery_selected
        )
        if recovery_selected.get("contract_valid"):
            recovery_prediction = recovery_selected["prediction"]
            parameters["begin_arrow"] = recovery_prediction["begin_arrow"]
            parameters["end_arrow"] = recovery_prediction["end_arrow"]
            arrow_head_source = (
                "focused_arrow_head_recovery"
                if crop_recovery_selected.get("contract_valid")
                else "full_image_arrow_head_recovery"
            )
        elif forced_end_selected.get("contract_valid"):
            parameters["begin_arrow"] = "none"
            parameters["end_arrow"] = forced_end_selected["prediction"]["end_arrow"]
            arrow_head_source = "trusted_arrow_prior_forced_path_end_type"
        else:
            parameters["begin_arrow"] = "none"
            parameters["end_arrow"] = "stealth"
            arrow_head_source = "dataset_empirical_map_prior_end_stealth"
    if record["source_label"] == "arrow" and parameters["begin_arrow"] == parameters["end_arrow"] == "none":
        return {"status": "unresolved", "reason": "arrow_prior_not_satisfied_after_fusion"}
    return {
        "status": "success",
        "parameters_crop_qwen": parameters,
        "geometry_source": geometry_source,
        "geometry_warning": not bool(points_geometry or reconstruction_geometry),
        "geometry_high_risk_fallback": geometry_source == "source_bbox_axis_high_risk_fallback",
        "attribute_source": attribute_selection["source"],
        "attribute_variant": attribute_selection["variant"],
        "attribute_attempt_index": attribute_selection["attempt_index"],
        "arrow_head_source": arrow_head_source,
        "points_reversed_for_endpoint_alignment": reversed_for_endpoint_alignment,
        "topology_disagreement": (
            attribute_selection.get("reference_is_single") is not None
            and bool(attribute_selection["reference_is_single"])
            != bool(selected_points["is_single"])
        ),
    }


def result_path(output_root: Path, target_id: str) -> Path:
    shard = hashlib.sha1(target_id.encode("utf-8")).hexdigest()[:2]
    return output_root / "results" / shard / f"{target_id}.json"


def result_complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = read_json(path)
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("fusion", {}).get("status") == "success"


def infer_one(
    record: dict[str, Any],
    *,
    output_root: Path,
    endpoint: str,
    served_model: str,
    reconstruction_prompts: dict[str, ShaftPromptTemplate],
    points_prompts: dict[str, ShaftPromptTemplate],
    arrow_head_prompts: dict[str, ShaftPromptTemplate],
    arrow_end_prompt: ShaftPromptTemplate,
    min_pixels: int,
    max_pixels: int,
    max_tokens: int,
    timeout_s: float,
) -> dict[str, Any]:
    path = result_path(output_root, str(record["target_id"]))
    if result_complete(path):
        return {"target_id": record["target_id"], "status": "skipped"}
    image_url, image_request = image_to_data_url_with_qwen_pixel_budget(
        ROOT / record["crop_path"],
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    reconstruction = infer_head(
        task="reconstruction",
        variants=RECONSTRUCTION_VARIANTS,
        prompts=reconstruction_prompts,
        record=record,
        image_url=image_url,
        endpoint=endpoint,
        served_model=served_model,
        max_tokens=max_tokens,
        timeout_s=timeout_s,
    )
    reconstruction_parameters = (
        (reconstruction.get("selected") or {}).get("prediction") or {}
    ).get("parameters") or {}
    needs_arrow_head_recovery = (
        record["source_label"] == "arrow"
        and reconstruction_parameters.get("begin_arrow") == "none"
        and reconstruction_parameters.get("end_arrow") == "none"
    )
    arrow_head_recovery = (
        infer_arrow_heads(
            prompts=arrow_head_prompts,
            record=record,
            image_url=image_url,
            endpoint=endpoint,
            served_model=served_model,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
        )
        if needs_arrow_head_recovery
        else None
    )
    crop_recovery_selected = (arrow_head_recovery or {}).get("selected") or {}
    needs_full_image_arrow_head_recovery = (
        needs_arrow_head_recovery and not crop_recovery_selected.get("contract_valid")
    )
    full_image_request = None
    if needs_full_image_arrow_head_recovery:
        full_image_url, full_image_request = image_to_data_url_with_qwen_pixel_budget(
            ROOT / record["image_path"],
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        global_record = {
            **record,
            "proposal_bbox_2d": quantize_qwen_bbox(
                record["source_bbox"],
                width=int(record["image_width"]),
                height=int(record["image_height"]),
                minimum_extent_bins=1,
            ),
        }
        arrow_head_recovery_full_image = infer_arrow_heads(
            prompts=arrow_head_prompts,
            record=global_record,
            image_url=full_image_url,
            endpoint=endpoint,
            served_model=served_model,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
        )
    else:
        arrow_head_recovery_full_image = None
    full_recovery_selected = (arrow_head_recovery_full_image or {}).get("selected") or {}
    needs_forced_end_arrow_recovery = (
        needs_full_image_arrow_head_recovery
        and not full_recovery_selected.get("contract_valid")
    )
    if needs_forced_end_arrow_recovery:
        if full_image_request is None:
            full_image_url, full_image_request = image_to_data_url_with_qwen_pixel_budget(
                ROOT / record["image_path"],
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
        global_record = {
            **record,
            "proposal_bbox_2d": quantize_qwen_bbox(
                record["source_bbox"],
                width=int(record["image_width"]),
                height=int(record["image_height"]),
                minimum_extent_bins=1,
            ),
        }
        forced_end_arrow_recovery = infer_forced_end_arrow(
            prompt=arrow_end_prompt,
            record=global_record,
            image_url=full_image_url,
            endpoint=endpoint,
            served_model=served_model,
            timeout_s=timeout_s,
        )
    else:
        forced_end_arrow_recovery = None
    points = infer_head(
        task="points",
        variants=POINTS_VARIANTS,
        prompts=points_prompts,
        record=record,
        image_url=image_url,
        endpoint=endpoint,
        served_model=served_model,
        max_tokens=max_tokens,
        timeout_s=timeout_s,
    )
    fusion = fuse_heads(
        record,
        reconstruction,
        points,
        arrow_head_recovery,
        arrow_head_recovery_full_image,
        forced_end_arrow_recovery,
    )
    payload = {
        "schema_version": 1,
        "target": record,
        "model": served_model,
        "message_order": "image_first",
        "request_content_order": ["image_url", "text"],
        "generation": {
            "do_sample": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": max_tokens,
        },
        "image_request": image_request.to_dict(),
        "reconstruction": reconstruction,
        "arrow_head_recovery": arrow_head_recovery,
        "arrow_head_recovery_full_image": arrow_head_recovery_full_image,
        "forced_end_arrow_recovery": forced_end_arrow_recovery,
        "full_image_request": (
            full_image_request.to_dict() if full_image_request is not None else None
        ),
        "points": points,
        "fusion": fusion,
    }
    atomic_write_json(path, payload)
    return {
        "target_id": record["target_id"],
        "status": fusion["status"],
        "geometry_source": fusion.get("geometry_source"),
        "geometry_warning": fusion.get("geometry_warning", False),
        "geometry_high_risk_fallback": fusion.get("geometry_high_risk_fallback", False),
        "topology_disagreement": fusion.get("topology_disagreement", False),
        "reconstruction_attempts": len(reconstruction["attempts"]),
        "arrow_head_recovery_attempts": (
            len(arrow_head_recovery["attempts"]) if arrow_head_recovery else 0
        ),
        "arrow_head_recovery_full_image_attempts": (
            len(arrow_head_recovery_full_image["attempts"])
            if arrow_head_recovery_full_image
            else 0
        ),
        "forced_end_arrow_recovery_attempts": (
            len(forced_end_arrow_recovery["attempts"])
            if forced_end_arrow_recovery
            else 0
        ),
        "points_attempts": len(points["attempts"]),
    }


def bounded_map(
    executor: concurrent.futures.ThreadPoolExecutor,
    records: Iterable[dict[str, Any]],
    function: Any,
    *,
    window: int,
) -> Iterable[dict[str, Any]]:
    iterator = iter(records)
    pending: set[concurrent.futures.Future] = set()
    for _ in range(window):
        try:
            pending.add(executor.submit(function, next(iterator)))
        except StopIteration:
            break
    while pending:
        done, pending = concurrent.futures.wait(
            pending,
            return_when=concurrent.futures.FIRST_COMPLETED,
        )
        for future in done:
            yield future.result()
            try:
                pending.add(executor.submit(function, next(iterator)))
            except StopIteration:
                pass


def verify_endpoint(endpoint: str, served_model: str, expected_model_path: Path) -> dict[str, Any]:
    with request.urlopen(endpoint.rstrip("/") + "/v1/models", timeout=15) as response:
        payload = json.loads(response.read().decode("utf-8"))
    candidates = [item for item in payload.get("data", []) if isinstance(item, dict)]
    matches = [item for item in candidates if item.get("id") == served_model]
    if len(matches) != 1:
        raise RuntimeError(f"served model mismatch: wanted {served_model}, got {candidates}")
    model = matches[0]
    root_value = str(model.get("root") or "")
    expected = str(expected_model_path)
    if root_value and expected not in root_value and str(expected_model_path.relative_to(ROOT)) not in root_value:
        raise RuntimeError(f"endpoint root mismatch: wanted {expected}, got {root_value}")
    return model


def infer(args: argparse.Namespace) -> None:
    records = read_manifest(args.output_root)
    endpoint_model = verify_endpoint(args.endpoint, args.served_model, args.model_path)
    reconstruction_prompts = prompt_variants(args.reconstruction_prompt)
    points_prompts = prompt_variants(args.points_prompt)
    arrow_head_prompts = prompt_variants(args.arrow_head_prompt)
    arrow_end_prompts = prompt_variants(args.arrow_end_prompt)
    missing_variants = set(RECONSTRUCTION_VARIANTS) - set(reconstruction_prompts)
    missing_variants |= set(POINTS_VARIANTS) - set(points_prompts)
    missing_variants |= set(ARROW_HEAD_VARIANTS) - set(arrow_head_prompts)
    if "main" not in arrow_end_prompts:
        missing_variants.add("arrow_end:main")
    if missing_variants:
        raise RuntimeError(f"missing prompt variants: {sorted(missing_variants)}")
    pending = [record for record in records if not result_complete(result_path(args.output_root, record["target_id"]))]
    counters: Counter[str] = Counter()
    started = time.perf_counter()

    def run(record: dict[str, Any]) -> dict[str, Any]:
        return infer_one(
            record,
            output_root=args.output_root,
            endpoint=args.endpoint,
            served_model=args.served_model,
            reconstruction_prompts=reconstruction_prompts,
            points_prompts=points_prompts,
            arrow_head_prompts=arrow_head_prompts,
            arrow_end_prompt=arrow_end_prompts["main"],
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            max_tokens=args.max_tokens,
            timeout_s=args.timeout,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        for completed, result in enumerate(
            bounded_map(executor, pending, run, window=max(args.workers * 3, 1)), start=1
        ):
            counters[f"status_{result['status']}"] += 1
            if result.get("geometry_source"):
                counters[f"geometry_{result['geometry_source']}"] += 1
            counters["geometry_warning"] += int(bool(result.get("geometry_warning")))
            counters["geometry_high_risk_fallback"] += int(
                bool(result.get("geometry_high_risk_fallback"))
            )
            counters["topology_disagreement"] += int(bool(result.get("topology_disagreement")))
            counters["reconstruction_retry"] += int(result.get("reconstruction_attempts", 1) > 1)
            counters["arrow_head_recovery"] += int(
                result.get("arrow_head_recovery_attempts", 0) > 0
            )
            counters["arrow_head_recovery_retry"] += int(
                result.get("arrow_head_recovery_attempts", 0) > 1
            )
            counters["arrow_head_recovery_full_image"] += int(
                result.get("arrow_head_recovery_full_image_attempts", 0) > 0
            )
            counters["arrow_head_recovery_full_image_retry"] += int(
                result.get("arrow_head_recovery_full_image_attempts", 0) > 1
            )
            counters["forced_end_arrow_recovery"] += int(
                result.get("forced_end_arrow_recovery_attempts", 0) > 0
            )
            counters["points_retry"] += int(result.get("points_attempts", 1) > 1)
            if completed == 1 or completed % 500 == 0 or completed == len(pending):
                elapsed = time.perf_counter() - started
                print(
                    json.dumps(
                        {
                            "done": completed,
                            "pending_total": len(pending),
                            "manifest_total": len(records),
                            "rate_targets_s": round(completed / elapsed, 3) if elapsed else None,
                            "counts": dict(counters),
                        }
                    ),
                    flush=True,
                )
    result_counts = Counter()
    for record in records:
        path = result_path(args.output_root, record["target_id"])
        if not path.exists():
            result_counts["missing"] += 1
            continue
        result = read_json(path)
        result_counts[result.get("fusion", {}).get("status", "unknown")] += 1
        source = result.get("fusion", {}).get("geometry_source")
        if source:
            result_counts[f"geometry_{source}"] += 1
        result_counts["geometry_warning"] += int(
            bool(result.get("fusion", {}).get("geometry_warning"))
        )
    summary = {
        "schema_version": 1,
        "model_path": str(args.model_path.relative_to(ROOT)),
        "served_model": args.served_model,
        "endpoint_model": endpoint_model,
        "endpoint": args.endpoint,
        "manifest_targets": len(records),
        "workers": args.workers,
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "max_tokens": args.max_tokens,
        "message_order": "image_first",
        "decoding": {"do_sample": False, "temperature": 0.0, "top_p": 1.0},
        "result_counts": dict(result_counts),
        "elapsed_s_this_run": round(time.perf_counter() - started, 3),
    }
    atomic_write_json(args.output_root / "inference_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def globalize_parameters(
    parameters: dict[str, Any], record: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    output = json.loads(json.dumps(parameters))
    left, top, _, _ = [int(value) for value in record["crop_box"]]
    crop_width = int(record["crop_width"])
    crop_height = int(record["crop_height"])
    global_segments = []
    seen_segments: set[tuple[tuple[float, float], ...]] = set()
    duplicate_points_removed = 0
    duplicate_segments_removed = 0
    for segment in output["points"]:
        global_segment = []
        for point in segment:
            local_x, local_y = dequantize_qwen_point(
                point,
                width=crop_width,
                height=crop_height,
            )
            global_point = [
                round(min(max(local_x + left, 0.0), float(record["image_width"] - 1)), 2),
                round(min(max(local_y + top, 0.0), float(record["image_height"] - 1)), 2),
            ]
            if global_segment and global_point == global_segment[-1]:
                duplicate_points_removed += 1
                continue
            global_segment.append(global_point)
        if len(global_segment) < 2:
            continue
        segment_key = tuple((point[0], point[1]) for point in global_segment)
        if segment_key in seen_segments:
            duplicate_segments_removed += 1
            continue
        seen_segments.add(segment_key)
        global_segments.append(global_segment)
    bbox_fallback = not global_segments
    if bbox_fallback:
        x1, y1, x2, y2 = [float(value) for value in record["source_bbox"]]
        if x2 - x1 >= y2 - y1:
            center = round((y1 + y2) / 2.0, 2)
            global_segments = [[[round(x1, 2), center], [round(x2, 2), center]]]
        else:
            center = round((x1 + x2) / 2.0, 2)
            global_segments = [[[center, round(y1, 2)], [center, round(y2, 2)]]]
    output["points"] = global_segments
    output["is_single"] = len(global_segments) == 1
    return output, {
        "consecutive_duplicates_removed": duplicate_points_removed,
        "duplicate_segments_removed": duplicate_segments_removed,
        "bbox_axis_fallback": bbox_fallback,
        "changed": bool(
            duplicate_points_removed or duplicate_segments_removed or bbox_fallback
        ),
    }


def gt_standard_bbox(
    bbox: list[Any], *, image_width: int, image_height: int
) -> list[int]:
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError(f"invalid bbox: {bbox!r}")
    x1, y1, x2, y2 = [float(value) for value in bbox]
    left = min(max(math.floor(min(x1, x2)), 0), image_width - 1)
    top = min(max(math.floor(min(y1, y2)), 0), image_height - 1)
    right = min(max(math.ceil(max(x1, x2)), left + 1), image_width)
    bottom = min(max(math.ceil(max(y1, y2)), top + 1), image_height)
    return [left, top, right, bottom]


def integerize_line_parameters(
    parameters: dict[str, Any], record: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    output = json.loads(json.dumps(parameters))
    image_width = int(record["image_width"])
    image_height = int(record["image_height"])
    segments: list[list[list[int]]] = []
    seen_segments: set[tuple[tuple[int, int], ...]] = set()
    duplicate_points_removed = 0
    duplicate_segments_removed = 0
    collapsed_segments = 0
    for raw_segment in output.get("points") or []:
        segment: list[list[int]] = []
        for raw_point in raw_segment:
            point = [
                min(max(int(round(float(raw_point[0]))), 0), image_width - 1),
                min(max(int(round(float(raw_point[1]))), 0), image_height - 1),
            ]
            if segment and point == segment[-1]:
                duplicate_points_removed += 1
                continue
            segment.append(point)
        if len(segment) < 2:
            collapsed_segments += 1
            continue
        segment_key = tuple((point[0], point[1]) for point in segment)
        if segment_key in seen_segments:
            duplicate_segments_removed += 1
            continue
        seen_segments.add(segment_key)
        segments.append(segment)
    bbox_fallback = not segments
    if bbox_fallback:
        x1, y1, x2, y2 = gt_standard_bbox(
            record["source_bbox"],
            image_width=image_width,
            image_height=image_height,
        )
        if x2 - x1 >= y2 - y1:
            center = min(max(int(round((y1 + y2 - 1) / 2.0)), 0), image_height - 1)
            segments = [[[x1, center], [max(x1 + 1, x2 - 1), center]]]
        else:
            center = min(max(int(round((x1 + x2 - 1) / 2.0)), 0), image_width - 1)
            segments = [[[center, y1], [center, max(y1 + 1, y2 - 1)]]]
    output["points"] = segments
    output["is_single"] = len(segments) == 1
    parameter_order = (
        "line_type",
        "line_style",
        "is_single",
        "points",
        "dash_style",
        "begin_arrow",
        "end_arrow",
        "corner_style",
        "has_border",
        "fill_color",
        "border_style",
        "border_color",
    )
    output = {key: output[key] for key in parameter_order if key in output}
    return output, {
        "consecutive_duplicates_removed": duplicate_points_removed,
        "duplicate_segments_removed": duplicate_segments_removed,
        "collapsed_segments_removed": collapsed_segments,
        "bbox_axis_fallback": bbox_fallback,
        "changed": bool(
            duplicate_points_removed
            or duplicate_segments_removed
            or collapsed_segments
            or bbox_fallback
        ),
    }


def point_to_segment_distance(
    point: list[int],
    start: list[int],
    end: list[int],
) -> float:
    px, py = float(point[0]), float(point[1])
    sx, sy = float(start[0]), float(start[1])
    ex, ey = float(end[0]), float(end[1])
    dx, dy = ex - sx, ey - sy
    if dx == 0.0 and dy == 0.0:
        return math.hypot(px - sx, py - sy)
    ratio = min(1.0, max(0.0, ((px - sx) * dx + (py - sy) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (sx + ratio * dx), py - (sy + ratio * dy))


def simplify_segment_rdp(segment: list[list[int]], epsilon: float) -> list[list[int]]:
    if len(segment) <= 2:
        return json.loads(json.dumps(segment))
    start, end = segment[0], segment[-1]
    distances = [point_to_segment_distance(point, start, end) for point in segment[1:-1]]
    maximum = max(distances, default=0.0)
    if maximum <= epsilon:
        return [list(start), list(end)]
    split = distances.index(maximum) + 1
    left = simplify_segment_rdp(segment[: split + 1], epsilon)
    right = simplify_segment_rdp(segment[split:], epsilon)
    return left[:-1] + right


def sample_segment_by_arclength(segment: list[list[int]], count: int = 4) -> list[list[int]]:
    if len(segment) <= 2 or count <= 2:
        return json.loads(json.dumps(segment))
    cumulative = [0.0]
    for first, second in zip(segment, segment[1:]):
        cumulative.append(cumulative[-1] + endpoint_cost(first, second))
    total = cumulative[-1]
    if total <= 0.0:
        return [list(segment[0]), list(segment[-1])]
    sampled = []
    edge_index = 0
    for sample_index in range(count):
        target = total * sample_index / (count - 1)
        while edge_index + 1 < len(cumulative) - 1 and cumulative[edge_index + 1] < target:
            edge_index += 1
        start_distance, end_distance = cumulative[edge_index], cumulative[edge_index + 1]
        ratio = 0.0 if end_distance == start_distance else (target - start_distance) / (
            end_distance - start_distance
        )
        start, end = segment[edge_index], segment[edge_index + 1]
        sampled.append(
            [
                int(round(float(start[0]) + ratio * (float(end[0]) - float(start[0])))),
                int(round(float(start[1]) + ratio * (float(end[1]) - float(start[1])))),
            ]
        )
    return sampled


def remove_consecutive_duplicate_points(segment: list[list[int]]) -> list[list[int]]:
    output = []
    for point in segment:
        normalized = [int(point[0]), int(point[1])]
        if not output or normalized != output[-1]:
            output.append(normalized)
    return output


def clean_parameters_geometry(
    parameters: dict[str, Any],
    record: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Defensively reduce pathological model traces without changing the raw audit result."""
    output = json.loads(json.dumps(parameters))
    original_segments = output.get("points") if isinstance(output.get("points"), list) else []
    original_total = sum(len(segment) for segment in original_segments if isinstance(segment, list))
    invalid_removed = 0
    duplicate_removed = 0
    duplicate_segments_removed = 0
    simplified_segments = 0
    cleaned_segments: list[list[list[int]]] = []
    seen_segments: set[tuple[tuple[int, int], ...]] = set()
    for raw_segment in original_segments:
        if not isinstance(raw_segment, list):
            continue
        valid = []
        for point in raw_segment:
            if valid_point(point):
                valid.append([int(point[0]), int(point[1])])
            else:
                invalid_removed += 1
        deduplicated = remove_consecutive_duplicate_points(valid)
        duplicate_removed += len(valid) - len(deduplicated)
        if len(deduplicated) < 2:
            continue
        if output.get("line_type") == "curved":
            cleaned = (
                remove_consecutive_duplicate_points(
                    sample_segment_by_arclength(deduplicated, count=4)
                )
                if len(deduplicated) > 4
                else deduplicated
            )
        else:
            epsilon = 1.5
            cleaned = simplify_segment_rdp(deduplicated, epsilon)
            while len(cleaned) > 64 and epsilon < 96.0:
                epsilon *= 2.0
                cleaned = simplify_segment_rdp(deduplicated, epsilon)
        if len(cleaned) < 2:
            continue
        simplified_segments += int(cleaned != raw_segment)
        segment_key = tuple((point[0], point[1]) for point in cleaned)
        if segment_key in seen_segments:
            duplicate_segments_removed += 1
            continue
        seen_segments.add(segment_key)
        cleaned_segments.append(cleaned)
    used_bbox_fallback = not cleaned_segments
    if used_bbox_fallback:
        fallback = proposal_axis_geometry(record["proposal_bbox_2d"])
        cleaned_segments = fallback["points"]
    output["points"] = cleaned_segments
    output["is_single"] = len(cleaned_segments) == 1
    final_total = sum(len(segment) for segment in cleaned_segments)
    return output, {
        "version": 1,
        "methods": [
            "remove_invalid_points",
            "remove_consecutive_duplicates",
            "curved_arclength_four_samples",
            "straight_rdp_epsilon_1_5_with_64_point_cap",
        ],
        "original_total_points": original_total,
        "final_total_points": final_total,
        "invalid_points_removed": invalid_removed,
        "consecutive_duplicates_removed": duplicate_removed,
        "duplicate_segments_removed": duplicate_segments_removed,
        "segments_simplified": simplified_segments,
        "bbox_axis_fallback": used_bbox_fallback,
        "changed": bool(
            invalid_removed
            or duplicate_removed
            or duplicate_segments_removed
            or simplified_segments
            or used_bbox_fallback
            or original_total != final_total
        ),
    }


def finalize(args: argparse.Namespace) -> None:
    records = read_manifest(args.output_root)
    by_json: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unresolved = []
    for record in records:
        path = result_path(args.output_root, record["target_id"])
        if not path.exists():
            unresolved.append({"target_id": record["target_id"], "reason": "missing_result"})
            continue
        result = read_json(path)
        if result.get("fusion", {}).get("status") != "success":
            unresolved.append(
                {
                    "target_id": record["target_id"],
                    "reason": result.get("fusion", {}).get("reason", "unknown"),
                }
            )
            continue
        by_json[str(record["json_name"])].append({"record": record, "result": result, "path": path})
    atomic_write_json(args.output_root / "unresolved.json", unresolved)
    if unresolved:
        raise RuntimeError(f"cannot finalize with {len(unresolved)} unresolved targets")
    staging = args.output_root / "json.staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    input_json_paths = sorted(args.input_json_dir.glob("*.json"))
    counters: Counter[str] = Counter()
    for input_path in input_json_paths:
        source_document = read_json(input_path)
        image_width = int(source_document["image_width"])
        image_height = int(source_document["image_height"])
        document = {
            "size": [image_width, image_height],
            "background": "image" if source_document.get("background") is True else "none",
            "layout": [],
        }
        targets = {int(item["record"]["instance_index"]): item for item in by_json.get(input_path.name, [])}
        for instance_index, source_instance in enumerate(source_document.get("instances") or []):
            item = targets.get(instance_index)
            original_label = str(source_instance.get("label") or "").lower()
            output_instance: dict[str, Any] = {
                "type": "line" if item is not None else original_label,
                "bbox": gt_standard_bbox(
                    source_instance["bbox"],
                    image_width=image_width,
                    image_height=image_height,
                ),
            }
            if item is None:
                source_extra = (
                    source_instance.get("extra")
                    if isinstance(source_instance.get("extra"), dict)
                    else {}
                )
                if isinstance(source_extra.get("parameters"), dict):
                    output_instance["parameters"] = json.loads(
                        json.dumps(source_extra["parameters"])
                    )
                document["layout"].append(output_instance)
                counters[f"source_{original_label}"] += 1
                continue
            record, result = item["record"], item["result"]
            if original_label != record["source_label"]:
                raise RuntimeError(f"source label drift: {input_path}/{instance_index}")
            cleaned_parameters, cleanup = clean_parameters_geometry(
                result["fusion"]["parameters_crop_qwen"], record
            )
            parameters, global_cleanup = globalize_parameters(cleaned_parameters, record)
            parameters, integer_cleanup = integerize_line_parameters(parameters, record)
            cleanup["global_coordinate_cleanup"] = global_cleanup
            cleanup["integer_coordinate_cleanup"] = integer_cleanup
            cleanup["final_total_points_global"] = sum(
                len(segment) for segment in parameters["points"]
            )
            cleanup["changed"] = bool(
                cleanup["changed"]
                or global_cleanup["changed"]
                or integer_cleanup["changed"]
            )
            output_instance["parameters"] = parameters
            document["layout"].append(output_instance)
            counters[f"source_{original_label}"] += 1
            counters[f"geometry_{result['fusion']['geometry_source']}"] += 1
            counters["geometry_warning"] += int(bool(result["fusion"].get("geometry_warning")))
            counters["geometry_high_risk_fallback"] += int(
                bool(result["fusion"].get("geometry_high_risk_fallback"))
            )
            counters["geometry_cleanup_changed"] += int(bool(cleanup["changed"]))
            counters["geometry_cleanup_bbox_fallback"] += int(bool(cleanup["bbox_axis_fallback"]))
            counters["integer_cleanup_changed"] += int(bool(integer_cleanup["changed"]))
            counters["integer_cleanup_bbox_fallback"] += int(
                bool(integer_cleanup["bbox_axis_fallback"])
            )
            counters["geometry_cleanup_points_removed"] += int(
                cleanup["original_total_points"] - cleanup["final_total_points_global"]
            )
            counters[f"arrow_head_{result['fusion'].get('arrow_head_source')}"] += 1
            counters["topology_disagreement"] += int(
                bool(result["fusion"].get("topology_disagreement"))
            )
        atomic_write_json(staging / input_path.name, document)
    output_json = args.output_root / "json"
    if output_json.exists():
        backup = args.output_root / "json.previous"
        if backup.exists():
            shutil.rmtree(backup)
        os.replace(output_json, backup)
    os.replace(staging, output_json)
    if "backup" in locals() and backup.exists():
        shutil.rmtree(backup)
    summary = {
        "schema_version": 1,
        "input_json_files": len(input_json_paths),
        "output_json_files": len(list(output_json.glob("*.json"))),
        "target_instances": len(records),
        "unresolved": 0,
        "counts": dict(counters),
        "output_json_dir": str(output_json.relative_to(ROOT)),
        "model_path": str(args.model_path.relative_to(ROOT)),
        "served_model": args.served_model,
        "fusion_policy": (
            "line_context_points geometry when contract-valid and spatially sane; "
            "line_context_reconstruction fallback; reconstruction attributes retained"
        ),
        "source_priors": {
            "line": "begin_arrow=end_arrow=none",
            "arrow": "at least one endpoint head is non-none",
        },
        "gt_standard_schema": {
            "root_keys": ["size", "background", "layout"],
            "instance_keys": ["type", "bbox", "parameters?"],
            "background_mapping": {"true": "image", "missing_or_false": "none"},
            "audit_metadata_in_annotation": False,
        },
    }
    atomic_write_json(args.output_root / "finalize_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def preview_font(size: int) -> ImageFont.ImageFont:
    candidates = (
        ROOT / "assets" / "fonts" / "DejaVuSans.ttf",
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def render_preview(arguments: tuple[str, str, str, int, dict[int, str]]) -> str:
    json_path_text, image_path_text, preview_path_text, max_side, source_labels = arguments
    document = read_json(Path(json_path_text))
    with Image.open(image_path_text) as opened:
        image = opened.convert("RGB")
    scale = min(1.0, max_side / max(image.size))
    if scale < 1.0:
        image = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.Resampling.LANCZOS,
        )
    draw = ImageDraw.Draw(image, "RGBA")
    font = preview_font(max(12, min(24, round(min(image.size) / 55))))
    for index, instance in enumerate(document.get("layout") or []):
        source_label = source_labels.get(index)
        if source_label not in TARGET_LABELS or instance.get("type") != "line":
            continue
        box = [float(value) * scale for value in instance["bbox"]]
        color = (255, 45, 85, 220) if source_label == "arrow" else (0, 160, 255, 220)
        draw.rectangle(box, outline=color, width=max(2, round(3 * scale)))
        parameters = instance.get("parameters") or {}
        for segment in parameters.get("points") or []:
            points = [(float(point[0]) * scale, float(point[1]) * scale) for point in segment]
            if len(points) >= 2:
                draw.line(points, fill=(139, 92, 246, 235), width=max(2, round(4 * scale)), joint="curve")
                radius = max(3, round(5 * scale))
                for point, fill in ((points[0], (34, 197, 94, 255)), (points[-1], (236, 72, 153, 255))):
                    draw.ellipse(
                        (point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius),
                        fill=fill,
                        outline="white",
                    )
        label = f"{index}:{source_label}->line {parameters.get('begin_arrow')}/{parameters.get('end_arrow')}"
        text_box = draw.textbbox((box[0], box[1]), label, font=font)
        draw.rectangle(text_box, fill=(17, 24, 39, 210))
        draw.text((box[0], box[1]), label, font=font, fill="white")
    preview_path = Path(preview_path_text)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(preview_path, format="JPEG", quality=90, optimize=True)
    image.close()
    return preview_path.name


def previews(args: argparse.Namespace) -> None:
    output_json = args.output_root / "json"
    if not output_json.exists():
        raise FileNotFoundError("finalize first")
    images = image_map(args.image_dir)
    records_by_json: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in read_manifest(args.output_root):
        records_by_json[str(record["json_name"])].append(record)
    jobs = []
    review_records = []
    for json_path in sorted(output_json.glob("*.json")):
        records = records_by_json.get(json_path.name) or []
        if not records:
            continue
        source_labels = {
            int(record["instance_index"]): str(record["source_label"]) for record in records
        }
        preview_path = args.output_root / "previews" / f"{json_path.stem}.jpg"
        jobs.append(
            (
                str(json_path),
                str(images[json_path.stem]),
                str(preview_path),
                args.max_preview_side,
                source_labels,
            )
        )
        metadata = []
        for record in records:
            result = read_json(result_path(args.output_root, str(record["target_id"])))
            fusion = result["fusion"]
            cleaned, cleanup = clean_parameters_geometry(
                fusion["parameters_crop_qwen"], record
            )
            globalized, global_cleanup = globalize_parameters(cleaned, record)
            _, integer_cleanup = integerize_line_parameters(globalized, record)
            metadata.append(
                {
                    "geometry_high_risk_fallback": bool(
                        fusion.get("geometry_high_risk_fallback")
                        or integer_cleanup["bbox_axis_fallback"]
                    ),
                    "geometry_warning": fusion.get("geometry_warning"),
                    "arrow_head_source": fusion.get("arrow_head_source"),
                    "cleanup_changed": bool(
                        cleanup["changed"]
                        or global_cleanup["changed"]
                        or integer_cleanup["changed"]
                    ),
                }
            )
        review_records.append(
            {
                "stem": json_path.stem,
                "targets": len(records),
                "high_risk": any(item.get("geometry_high_risk_fallback") for item in metadata),
                "warning": any(item.get("geometry_warning") for item in metadata),
                "empirical_head": any(
                    item.get("arrow_head_source") == "dataset_empirical_map_prior_end_stealth"
                    for item in metadata
                ),
                "cleanup": any(
                    item.get("cleanup_changed") for item in metadata
                ),
            }
        )
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        for completed, _ in enumerate(executor.map(render_preview, jobs, chunksize=1), start=1):
            if completed == 1 or completed % 500 == 0 or completed == len(jobs):
                print(json.dumps({"previews": completed, "total": len(jobs)}), flush=True)
    review_data = json.dumps(review_records, ensure_ascii=False).replace("</", "<\\/")
    review_html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>json_20260706 line prelabel review</title>
<style>
body{{margin:0;background:#f7f7f5;color:#171717;font:14px/1.45 system-ui,sans-serif}}
header{{position:sticky;top:0;z-index:2;padding:12px 18px;background:#fff;border-bottom:1px solid #ddd}}
.controls{{display:flex;gap:10px;align-items:center;flex-wrap:wrap}} input,select,button{{padding:7px 9px}}
#grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:14px;padding:14px}}
.card{{background:#fff;border:1px solid #ddd;border-radius:8px;overflow:hidden}} .card img{{display:block;width:100%;height:auto;background:#fff}}
.meta{{padding:8px 10px;display:flex;gap:8px;flex-wrap:wrap}} .tag{{padding:2px 6px;border-radius:10px;background:#eee}}
.risk{{background:#fee2e2}} .warn{{background:#fef3c7}} footer{{padding:12px 18px 24px}}
</style></head><body><header><div class="controls">
<strong>Line prelabel review（仅预测，原图叠加 bbox/path）</strong>
<input id="query" placeholder="按文件名筛选"><select id="filter"><option value="all">全部</option><option value="high_risk">高风险几何</option><option value="warning">几何警告</option><option value="empirical_head">先验兜底 head</option><option value="cleanup">几何清理过</option></select>
<button id="prev">上一页</button><span id="status"></span><button id="next">下一页</button>
</div></header><main id="grid"></main><footer>绿色点为起点，粉色点为终点；红框表示原始 arrow，蓝框表示原始 line。</footer>
<script>const DATA={review_data};const PAGE=50;let page=0;
const grid=document.querySelector('#grid'),status=document.querySelector('#status'),query=document.querySelector('#query'),filter=document.querySelector('#filter');
function selected(){{const q=query.value.trim().toLowerCase(),f=filter.value;return DATA.filter(x=>(!q||x.stem.toLowerCase().includes(q))&&(f==='all'||x[f]));}}
function render(){{const data=selected(),pages=Math.max(1,Math.ceil(data.length/PAGE));page=Math.min(page,pages-1);const rows=data.slice(page*PAGE,(page+1)*PAGE);grid.innerHTML=rows.map(x=>`<article class="card"><img loading="lazy" src="../previews/${{encodeURIComponent(x.stem)}}.jpg"><div class="meta"><strong>${{x.stem}}</strong><span class="tag">${{x.targets}} targets</span>${{x.high_risk?'<span class="tag risk">高风险几何</span>':''}}${{x.warning?'<span class="tag warn">几何警告</span>':''}}${{x.empirical_head?'<span class="tag warn">先验 head</span>':''}}${{x.cleanup?'<span class="tag">已清理点</span>':''}}</div></article>`).join('');status.textContent=`${{data.length}} 张 · ${{page+1}}/${{pages}} 页`;}}
query.oninput=()=>{{page=0;render()}};filter.onchange=()=>{{page=0;render()}};document.querySelector('#prev').onclick=()=>{{page=Math.max(0,page-1);render()}};document.querySelector('#next').onclick=()=>{{page++;render()}};render();</script>
</body></html>"""
    review_path = args.output_root / "review" / "index.html"
    atomic_write_text(review_path, review_html)
    summary = {
        "preview_images": len(jobs),
        "preview_dir": str((args.output_root / "previews").relative_to(ROOT)),
        "review_page": str(review_path.relative_to(ROOT)),
    }
    atomic_write_json(args.output_root / "preview_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def validate_final(args: argparse.Namespace) -> None:
    records = read_manifest(args.output_root)
    record_by_key = {(record["json_name"], int(record["instance_index"])): record for record in records}
    output_json = args.output_root / "json"
    input_paths = sorted(args.input_json_dir.glob("*.json"))
    output_paths = sorted(output_json.glob("*.json"))
    if [path.name for path in input_paths] != [path.name for path in output_paths]:
        raise RuntimeError("input/output JSON coverage differs")
    counters: Counter[str] = Counter()
    for input_path, output_path in zip(input_paths, output_paths):
        source = read_json(input_path)
        output = read_json(output_path)
        image_width = int(source["image_width"])
        image_height = int(source["image_height"])
        if set(output) != {"size", "background", "layout"}:
            raise RuntimeError(f"gt_standard root schema failed: {output_path.name}")
        if output["size"] != [image_width, image_height]:
            raise RuntimeError(f"size mismatch: {output_path.name}")
        expected_background = "image" if source.get("background") is True else "none"
        if output["background"] != expected_background:
            raise RuntimeError(f"background mismatch: {output_path.name}")
        if len(source.get("instances") or []) != len(output.get("layout") or []):
            raise RuntimeError(f"instance count changed: {input_path.name}")
        for index, (before, after) in enumerate(zip(source["instances"], output["layout"])):
            key = (input_path.name, index)
            expected_bbox = gt_standard_bbox(
                before["bbox"], image_width=image_width, image_height=image_height
            )
            if key not in record_by_key:
                expected: dict[str, Any] = {
                    "type": str(before.get("label") or "").lower(),
                    "bbox": expected_bbox,
                }
                source_extra = before.get("extra") if isinstance(before.get("extra"), dict) else {}
                if isinstance(source_extra.get("parameters"), dict):
                    expected["parameters"] = source_extra["parameters"]
                if after != expected:
                    raise RuntimeError(
                        f"non-target gt_standard conversion failed: {input_path.name}/{index}"
                    )
                counters[f"source_{expected['type']}"] += 1
                continue
            record = record_by_key[key]
            if set(after) != {"type", "bbox", "parameters"}:
                raise RuntimeError(f"target instance schema failed: {input_path.name}/{index}")
            if after.get("type") != "line" or after.get("bbox") != expected_bbox:
                raise RuntimeError(f"target label/bbox invariant failed: {input_path.name}/{index}")
            parameters = after.get("parameters") or {}
            required_parameters = {
                "line_type",
                "line_style",
                "is_single",
                "points",
                "begin_arrow",
                "end_arrow",
                "dash_style",
                "fill_color",
            }
            optional_parameters = {
                "corner_style",
                "has_border",
                "border_style",
                "border_color",
            }
            if not required_parameters.issubset(parameters) or set(parameters) - (
                required_parameters | optional_parameters
            ):
                raise RuntimeError(f"line parameters schema failed: {input_path.name}/{index}")
            for segment in parameters.get("points") or []:
                if len(segment) < 2:
                    raise RuntimeError(f"short segment: {input_path.name}/{index}")
                if any(first == second for first, second in zip(segment, segment[1:])):
                    raise RuntimeError(
                        f"consecutive duplicate point: {input_path.name}/{index}"
                    )
                if len(segment) > 64:
                    raise RuntimeError(f"over-dense segment: {input_path.name}/{index}")
                for point in segment:
                    if not (
                        isinstance(point, list)
                        and len(point) == 2
                        and all(isinstance(value, int) and not isinstance(value, bool) for value in point)
                        and 0 <= point[0] < image_width
                        and 0 <= point[1] < image_height
                    ):
                        raise RuntimeError(f"invalid global point: {input_path.name}/{index}/{point}")
            if record["source_label"] == "line":
                if parameters.get("begin_arrow") != "none" or parameters.get("end_arrow") != "none":
                    raise RuntimeError(f"line head prior failed: {input_path.name}/{index}")
            else:
                if parameters.get("begin_arrow") == parameters.get("end_arrow") == "none":
                    raise RuntimeError(f"arrow head prior failed: {input_path.name}/{index}")
            if bool(parameters.get("is_single")) != (len(parameters.get("points") or []) == 1):
                raise RuntimeError(f"is_single relation failed: {input_path.name}/{index}")
            counters[f"source_{record['source_label']}"] += 1
            result = read_json(result_path(args.output_root, str(record["target_id"])))
            fusion = result["fusion"]
            counters[f"geometry_{fusion.get('geometry_source')}"] += 1
            counters["geometry_warning"] += int(bool(fusion.get("geometry_warning")))
    expected_previews = len({record["image_stem"] for record in records})
    preview_count = len(list((args.output_root / "previews").glob("*.jpg")))
    validation = {
        "schema_version": 1,
        "status": "ready_for_human_review",
        "input_json_files": len(input_paths),
        "output_json_files": len(output_paths),
        "target_instances": len(records),
        "counts": dict(counters),
        "unresolved": len(read_json(args.output_root / "unresolved.json")),
        "expected_preview_images": expected_previews,
        "preview_images": preview_count,
        "preview_complete": preview_count == expected_previews,
        "invariants": {
            "root_schema_size_background_layout": True,
            "instances_only_type_bbox_optional_parameters": True,
            "no_extra_fields": True,
            "non_target_parameters_preserved": True,
            "target_bboxes_cover_source_bbox_after_integerization": True,
            "all_target_labels_line": True,
            "source_line_heads_none": True,
            "source_arrow_has_head": True,
            "is_single_matches_segment_count": True,
            "points_in_original_image_coordinates": True,
        },
    }
    if validation["unresolved"] or not validation["preview_complete"]:
        validation["status"] = "needs_revision"
    atomic_write_json(args.output_root / "validation.json", validation)
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    if validation["status"] != "ready_for_human_review":
        raise SystemExit(1)


def common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-json-dir", type=Path, default=DEFAULT_INPUT_JSON)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dual-head v5.3 line reconstruction prelabeling")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    common_paths(prepare_parser)
    prepare_parser.add_argument("--workers", type=int, default=8)
    prepare_parser.add_argument("--seed", type=int, default=42)
    prepare_parser.add_argument("--limit", type=int, default=0)

    infer_parser = subparsers.add_parser("infer")
    common_paths(infer_parser)
    infer_parser.add_argument("--endpoint", default="http://127.0.0.1:18954")
    infer_parser.add_argument("--served-model", default="banana-v5.3-ckpt10000-line-prelabel")
    infer_parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    infer_parser.add_argument("--reconstruction-prompt", type=Path, default=RECONSTRUCTION_PROMPT)
    infer_parser.add_argument("--points-prompt", type=Path, default=POINTS_PROMPT)
    infer_parser.add_argument("--arrow-head-prompt", type=Path, default=ARROW_HEAD_PROMPT)
    infer_parser.add_argument("--arrow-end-prompt", type=Path, default=ARROW_END_PROMPT)
    infer_parser.add_argument("--workers", type=int, default=256)
    infer_parser.add_argument("--min-pixels", type=int, default=200704)
    infer_parser.add_argument("--max-pixels", type=int, default=2000000)
    infer_parser.add_argument("--max-tokens", type=int, default=4000)
    infer_parser.add_argument("--timeout", type=float, default=900.0)

    finalize_parser = subparsers.add_parser("finalize")
    common_paths(finalize_parser)
    finalize_parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    finalize_parser.add_argument("--served-model", default="banana-v5.3-ckpt10000-line-prelabel")

    preview_parser = subparsers.add_parser("previews")
    common_paths(preview_parser)
    preview_parser.add_argument("--workers", type=int, default=8)
    preview_parser.add_argument("--max-preview-side", type=int, default=2000)

    validate_parser = subparsers.add_parser("validate")
    common_paths(validate_parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for key in (
        "input_json_dir",
        "image_dir",
        "output_root",
        "model_path",
        "reconstruction_prompt",
        "points_prompt",
        "arrow_head_prompt",
        "arrow_end_prompt",
    ):
        value = getattr(args, key, None)
        if isinstance(value, Path) and not value.is_absolute():
            setattr(args, key, ROOT / value)
    if hasattr(args, "workers") and args.workers <= 0:
        raise ValueError("workers must be positive")
    if args.command == "prepare":
        prepare(args)
    elif args.command == "infer":
        infer(args)
    elif args.command == "finalize":
        finalize(args)
    elif args.command == "previews":
        previews(args)
    elif args.command == "validate":
        validate_final(args)


if __name__ == "__main__":
    main()
