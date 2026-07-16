#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import tempfile
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

import boto3
from PIL import Image, ImageDraw, ImageFont


IMAGE_TYPES = {
    "photo",
    "screenshot",
    "chart",
    "table",
    "diagram",
    "document",
    "map",
    "medical",
    "microscopy",
    "rendering",
    "illustration",
    "infographic",
    "other",
}

SHAPE_TYPES = {
    "rectangle",
    "oval",
    "triangle",
    "trapezoid",
    "parallelogram",
    "diamond",
    "step",
    "regular_pentagon",
    "regular_hexagon",
    "arrow_pentagon",
    "other_polygon",
    "callout",
    "other",
}

FILL_TYPES = {"none", "solid", "linear_gradient", "radial_gradient"}
FILL_DIRECTIONS = {"left_to_right", "right_to_left", "top_to_bottom", "bottom_to_top", "center_to_edge"}
EFFECT_TYPES = {"none", "shadow", "glow"}
BORDER_TYPES = {"none", "uniform"}
BORDER_STYLES = {"solid", "dot"}

IMAGE_TYPE_ALIASES = {
    "graph": "chart",
    "plot": "chart",
    "data_visualization": "chart",
    "visualization": "chart",
    "matrix": "table",
    "spreadsheet": "table",
    "flowchart": "diagram",
    "technical_diagram": "diagram",
    "architecture": "diagram",
    "network": "diagram",
    "paper": "document",
    "pdf": "document",
    "scan": "document",
    "scanned_document": "document",
    "form": "document",
    "ui": "screenshot",
    "interface": "screenshot",
    "screen": "screenshot",
    "remote_sensing": "map",
    "satellite": "map",
    "geographic": "map",
    "mri": "medical",
    "ct": "medical",
    "xray": "medical",
    "x_ray": "medical",
    "ultrasound": "medical",
    "endoscopy": "medical",
    "pathology": "medical",
    "anatomy": "medical",
    "sem": "microscopy",
    "tem": "microscopy",
    "micrograph": "microscopy",
    "microscope": "microscopy",
    "microscopic": "microscopy",
    "materials": "microscopy",
    "3d": "rendering",
    "3d_render": "rendering",
    "cad": "rendering",
    "product_rendering": "rendering",
    "poster": "infographic",
    "advertisement": "infographic",
    "ad": "infographic",
    "comic": "illustration",
    "cartoon": "illustration",
    "drawing": "illustration",
    "art": "illustration",
}

SHAPE_TYPE_ALIASES = {
    "rect": "rectangle",
    "rounded_rectangle": "rectangle",
    "rounded_rect": "rectangle",
    "circle": "oval",
    "ellipse": "oval",
    "hexagon": "regular_hexagon",
    "pentagon": "regular_pentagon",
    "arrow": "arrow_pentagon",
    "right_arrow": "arrow_pentagon",
    "polygon": "other_polygon",
    "speech_bubble": "callout",
    "bubble": "callout",
    "caption": "callout",
}


@dataclass(frozen=True)
class TaskRecord:
    task_type: str
    task_id: str
    source_json: str
    source_image: str
    image_stem: str
    instance_index: int
    image_size: list[int]
    bbox: list[int]
    crop_box: list[int]
    area: int


_thread_local = threading.local()
_write_lock = threading.Lock()


def atomic_write_json(path: Path, payload: Any, *, pretty: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
    ) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent) as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())
        tmp = Path(handle.name)
    os.replace(tmp, path)


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _write_lock:
        with path.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_image(images_dir: Path, stem: str) -> Path | None:
    for suffix in (".png", ".jpg", ".jpeg", ".webp"):
        path = images_dir / f"{stem}{suffix}"
        if path.exists():
            return path
    return None


def normalize_token(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def clean_hex_color(value: Any, default: str = "#000000") -> str:
    text = str(value or "").strip()
    if re.fullmatch(r"#[0-9a-fA-F]{6}", text):
        return text.upper()
    if re.fullmatch(r"[0-9a-fA-F]{6}", text):
        return f"#{text.upper()}"
    rgb_match = re.fullmatch(r"rgb\((\d{1,3}),\s*(\d{1,3}),\s*(\d{1,3})\)", text)
    if rgb_match:
        vals = [max(0, min(255, int(v))) for v in rgb_match.groups()]
        return "#{:02X}{:02X}{:02X}".format(*vals)
    return default


def clean_bbox(bbox: Any, width: int, height: int) -> list[int] | None:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None
    x1 = min(max(x1, 0.0), width)
    y1 = min(max(y1, 0.0), height)
    x2 = min(max(x2, 0.0), width)
    y2 = min(max(y2, 0.0), height)
    if x2 <= x1 or y2 <= y1:
        return None
    return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]


def crop_box_for_bbox(bbox: list[int], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    pad = max(10, int(round(max(bw, bh) * 0.10)))
    return [
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(width, x2 + pad),
        min(height, y2 + pad),
    ]


def has_existing_image_type(inst: dict[str, Any]) -> bool:
    params = inst.get("parameters")
    extra = inst.get("extra")
    if isinstance(params, dict) and params.get("image_type"):
        return True
    if isinstance(extra, dict):
        if extra.get("image_type"):
            return True
        nested = extra.get("parameters")
        if isinstance(nested, dict) and nested.get("image_type"):
            return True
    return False


def has_existing_shape_parameters(inst: dict[str, Any]) -> bool:
    if isinstance(inst.get("parameters"), dict) and inst["parameters"]:
        return True
    if isinstance(inst.get("subattr"), dict) and inst["subattr"]:
        return True
    extra = inst.get("extra")
    if isinstance(extra, dict) and isinstance(extra.get("parameters"), dict) and extra["parameters"]:
        return True
    return False


def collect_tasks(
    raw_json_dir: Path,
    images_dir: Path,
    task_type: str,
    *,
    skip_existing: bool,
    min_area: int,
) -> list[TaskRecord]:
    tasks: list[TaskRecord] = []
    label = "image" if task_type == "image" else "shape"
    for json_path in sorted(raw_json_dir.glob("*.json")):
        image_path = find_image(images_dir, json_path.stem)
        if image_path is None:
            continue
        obj = load_json(json_path)
        with Image.open(image_path) as image:
            width, height = image.size
        for index, inst in enumerate(obj.get("instances", [])):
            if not isinstance(inst, dict) or inst.get("label") != label:
                continue
            if skip_existing:
                if task_type == "image" and has_existing_image_type(inst):
                    continue
                if task_type == "shape" and has_existing_shape_parameters(inst):
                    continue
            bbox = clean_bbox(inst.get("bbox"), width, height)
            if bbox is None:
                continue
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if area < min_area:
                continue
            task_id = f"{json_path.stem}__{task_type}_{index:04d}"
            tasks.append(
                TaskRecord(
                    task_type=task_type,
                    task_id=task_id,
                    source_json=str(json_path),
                    source_image=str(image_path),
                    image_stem=json_path.stem,
                    instance_index=index,
                    image_size=[width, height],
                    bbox=bbox,
                    crop_box=crop_box_for_bbox(bbox, width, height),
                    area=area,
                )
            )
    return tasks


def task_to_dict(task: TaskRecord) -> dict[str, Any]:
    return {
        "task_type": task.task_type,
        "id": task.task_id,
        "source_json": task.source_json,
        "source_image": task.source_image,
        "image_stem": task.image_stem,
        "instance_index": task.instance_index,
        "image_size": task.image_size,
        "bbox": task.bbox,
        "crop_box": task.crop_box,
        "area": task.area,
    }


def batched(items: list[TaskRecord], batch_size: int) -> list[list[TaskRecord]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def fit_image_to_tile(image: Image.Image, *, tile_size: int, header: int) -> Image.Image:
    canvas = Image.new("RGB", (tile_size, tile_size), "white")
    available = tile_size - header - 8
    work = image.convert("RGB")
    work.thumbnail((tile_size - 8, available), Image.Resampling.LANCZOS)
    x = (tile_size - work.width) // 2
    y = header + (available - work.height) // 2
    canvas.paste(work, (x, y))
    return canvas


def build_sheet_bytes(
    batch: list[TaskRecord],
    *,
    tile_size: int,
    max_api_side: int,
    sheet_path: Path | None = None,
) -> bytes:
    columns = 2 if len(batch) <= 4 else min(4, len(batch))
    rows = math.ceil(len(batch) / columns)
    header = 30
    sheet = Image.new("RGB", (columns * tile_size, rows * tile_size), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for local_index, task in enumerate(batch):
        with Image.open(task.source_image) as source:
            source = source.convert("RGB")
            crop = source.crop(tuple(task.crop_box))
        tile = fit_image_to_tile(crop, tile_size=tile_size, header=header)
        col = local_index % columns
        row = local_index // columns
        x = col * tile_size
        y = row * tile_size
        sheet.paste(tile, (x, y))
        draw.rectangle((x, y, x + tile_size - 1, y + tile_size - 1), outline=(80, 80, 80), width=2)
        draw.rectangle((x + 4, y + 4, x + 58, y + 27), fill=(210, 0, 0))
        draw.text((x + 10, y + 9), str(local_index), fill="white", font=font)
    if sheet_path is not None:
        sheet_path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(sheet_path, quality=94)
    if max(sheet.size) > max_api_side:
        sheet.thumbnail((max_api_side, max_api_side), Image.Resampling.LANCZOS)
    buffer = BytesIO()
    sheet.save(buffer, format="JPEG", quality=92, optimize=True)
    return buffer.getvalue()


def build_image_prompt(batch_size: int) -> str:
    return (
        "Return only valid compact JSON. No markdown, comments, or explanation. "
        f"The image is a contact sheet with {batch_size} numbered tiles. "
        "Each tile has a red index label in the top-left corner. Classify each tile independently "
        "for editable document reconstruction. "
        "Output exactly one JSON array with one object per tile, sorted by index: "
        '[{"index":0,"image_type":"photo"}]. '
        "Use exactly one lowercase image_type from this schema: "
        "photo, screenshot, chart, table, diagram, document, map, medical, microscopy, rendering, "
        "illustration, infographic, other. Do not invent labels. "
        "Priority: table/chart > document > screenshot > map > medical > microscopy > diagram > "
        "infographic > rendering > photo > illustration > other. "
        "table: row-column tables, matrices, spreadsheets, forms dominated by cells, or grid-like data. "
        "chart: plots, axes, legends, heatmaps, spectra, or data visualizations. "
        "document: scanned/report/paper/PDF pages, certificates, ID cards, receipts, forms, text-dominant pages. "
        "screenshot: software, web, app, OS, code-editor, dashboard, or digital UI screenshots. "
        "map: maps, routes, geographic/spatial/satellite figures. "
        "medical: MRI, CT, X-ray, ultrasound, endoscopy, pathology, anatomy, clinical diagnostics. "
        "microscopy: SEM/TEM, cells under microscope, micrographs, microstructures, microscope scale-bar images. "
        "diagram: flowcharts, architecture/model diagrams, networks, circuits, mechanisms, process diagrams. "
        "infographic: posters, ads, promotional panels, or designed information panels mixing text/icons/images. "
        "rendering: 3D/CAD/product/engineering/building/mechanical simulation renderings. "
        "photo: camera-like natural imagery. "
        "illustration: drawings, cartoons, decorative art. "
        "other: ambiguous, too small/blurry, pure texture, icon-like residue, or none of the above. "
        "If a tile is mainly a chart/table inside a document or UI, prefer chart/table."
    )


def build_shape_prompt(batch_size: int) -> str:
    return (
        "Return only valid compact JSON. No markdown, comments, or explanation. "
        f"The image is a contact sheet with {batch_size} numbered tiles. "
        "Each tile has a red index label in the top-left corner. Classify the main shape in each tile "
        "for editable slide/document reconstruction. Ignore nearby text, arrows, and unrelated objects. "
        "Output exactly one JSON array with one object per tile, sorted by index. "
        "Use this compact schema: "
        '[{"index":0,"shape_type":"rectangle","border":{"type":"none"},'
        '"fill":{"type":"solid","color":"#FFFFFF"},"effect":{"type":"none"},"corners":[]}]. '
        "Allowed shape_type values: rectangle, oval, triangle, trapezoid, parallelogram, diamond, step, "
        "regular_pentagon, regular_hexagon, arrow_pentagon, other_polygon, callout, other. "
        "Use other only when the tile is not a recoverable standard shape. "
        "For shape_type other, output only index and shape_type. "
        "For all non-other shapes, include border, fill, effect, and corners. "
        "Do not output coordinate control points: corners must always be [] in this task. "
        "For callout, also include body_type if visible as rectangle, oval, or other; set tail to {} "
        "instead of guessing tail points. "
        "border.type is none or uniform. If uniform, include style solid or dot and an approximate #RRGGBB color. "
        "Use dot for dotted/dashed borders. "
        "fill.type is none, solid, linear_gradient, or radial_gradient. "
        "For solid fill include color. For gradient include direction and two approximate #RRGGBB colors. "
        "Allowed gradient directions: left_to_right, right_to_left, top_to_bottom, bottom_to_top, center_to_edge. "
        "effect.type is none, shadow, or glow. "
        "Prefer rectangle for rounded rectangles and simple panels. Prefer callout for speech bubbles or labels "
        "with a visible pointer/tail. Prefer arrow_pentagon for filled block arrows, not thin connector arrows."
    )


def extract_array(text: str) -> list[dict[str, Any]]:
    stripped = re.sub(r"^```(?:json)?", "", text.strip()).strip()
    stripped = re.sub(r"```$", "", stripped).strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", stripped, flags=re.S)
        if match is None:
            raise
        payload = json.loads(match.group(0))
    if not isinstance(payload, list):
        raise ValueError("response is not a JSON array")
    return [item for item in payload if isinstance(item, dict)]


def normalize_image_prediction(payload: dict[str, Any] | None) -> dict[str, str] | None:
    if not isinstance(payload, dict):
        return None
    image_type = normalize_token(payload.get("image_type"))
    image_type = IMAGE_TYPE_ALIASES.get(image_type, image_type)
    if image_type not in IMAGE_TYPES:
        image_type = "other"
    return {"image_type": image_type}


def normalize_border(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"type": "none"}
    border_type = normalize_token(payload.get("type"))
    border_type = "uniform" if border_type in {"solid", "dash", "dashed", "dot", "dotted", "line"} else border_type
    if border_type not in BORDER_TYPES:
        border_type = "none"
    if border_type == "none":
        return {"type": "none"}
    style = normalize_token(payload.get("style"))
    if style in {"dash", "dashed", "dotted"}:
        style = "dot"
    if style not in BORDER_STYLES:
        style = "solid"
    return {"type": "uniform", "style": style, "color": clean_hex_color(payload.get("color"))}


def normalize_fill(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"type": "none"}
    fill_type = normalize_token(payload.get("type"))
    if fill_type in {"gradient", "linear"}:
        fill_type = "linear_gradient"
    if fill_type in {"radial"}:
        fill_type = "radial_gradient"
    if fill_type not in FILL_TYPES:
        fill_type = "none"
    if fill_type == "none":
        return {"type": "none"}
    if fill_type == "solid":
        return {"type": "solid", "color": clean_hex_color(payload.get("color"), default="#FFFFFF")}
    direction = normalize_token(payload.get("direction"))
    if fill_type == "radial_gradient":
        direction = "center_to_edge"
    elif direction not in FILL_DIRECTIONS:
        direction = "left_to_right"
    raw_colors = payload.get("colors")
    colors: list[str] = []
    if isinstance(raw_colors, list):
        colors = [clean_hex_color(item, default="#FFFFFF") for item in raw_colors[:2]]
    if len(colors) < 2:
        colors = [clean_hex_color(payload.get("color"), default="#FFFFFF"), "#FFFFFF"]
    return {"type": fill_type, "direction": direction, "colors": colors[:2]}


def normalize_effect(payload: Any) -> dict[str, str]:
    if not isinstance(payload, dict):
        return {"type": "none"}
    effect_type = normalize_token(payload.get("type"))
    if effect_type not in EFFECT_TYPES:
        effect_type = "none"
    return {"type": effect_type}


def normalize_shape_prediction(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    shape_type = normalize_token(payload.get("shape_type"))
    shape_type = SHAPE_TYPE_ALIASES.get(shape_type, shape_type)
    if shape_type not in SHAPE_TYPES:
        shape_type = "other"
    if shape_type == "other":
        return {"shape_type": "other"}
    output: dict[str, Any] = {
        "shape_type": shape_type,
        "border": normalize_border(payload.get("border")),
        "fill": normalize_fill(payload.get("fill")),
        "corners": [],
        "effect": normalize_effect(payload.get("effect")),
    }
    if shape_type == "callout":
        body_type = normalize_token(payload.get("body_type"))
        if body_type not in {"rectangle", "oval", "other"}:
            body_type = "rectangle"
        output["body_type"] = body_type
        output["tail"] = {}
        output["body_corners"] = []
    return output


def normalize_batch(raw_items: list[dict[str, Any]] | None, batch_size: int, task_type: str) -> dict[int, dict[str, Any] | None]:
    output: dict[int, dict[str, Any] | None] = {idx: None for idx in range(batch_size)}
    if raw_items is None:
        return output
    for raw in raw_items:
        try:
            index = int(raw.get("index"))
        except (TypeError, ValueError):
            continue
        if index not in output:
            continue
        if task_type == "image":
            output[index] = normalize_image_prediction(raw)
        else:
            output[index] = normalize_shape_prediction(raw)
    return output


def bedrock_client(region: str) -> Any:
    client = getattr(_thread_local, "client", None)
    client_region = getattr(_thread_local, "region", None)
    if client is None or client_region != region:
        client = boto3.client("bedrock-runtime", region_name=region)
        _thread_local.client = client
        _thread_local.region = region
    return client


def call_sheet(
    batch: list[TaskRecord],
    *,
    task_type: str,
    model_id: str,
    region: str,
    tile_size: int,
    max_api_side: int,
    keep_sheet_path: Path | None,
    max_tokens: int,
    retries: int,
) -> dict[str, Any]:
    sheet_bytes = build_sheet_bytes(
        batch,
        tile_size=tile_size,
        max_api_side=max_api_side,
        sheet_path=keep_sheet_path,
    )
    prompt = build_image_prompt(len(batch)) if task_type == "image" else build_shape_prompt(len(batch))
    raw_response: str | None = None
    error: str | None = None
    raw_items: list[dict[str, Any]] | None = None
    for attempt in range(retries + 1):
        try:
            response = bedrock_client(region).converse(
                modelId=model_id,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"image": {"format": "jpeg", "source": {"bytes": sheet_bytes}}},
                            {"text": prompt},
                        ],
                    }
                ],
                inferenceConfig={"maxTokens": max_tokens},
            )
            raw_response = response["output"]["message"]["content"][0]["text"]
            raw_items = extract_array(raw_response)
            error = None
            break
        except Exception as exc:  # noqa: BLE001 - keep resumable API job alive.
            error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(min(30, 2**attempt + random.random()))
    predictions = normalize_batch(raw_items, len(batch), task_type)
    records: list[dict[str, Any]] = []
    for local_index, task in enumerate(batch):
        prediction = predictions[local_index]
        records.append(
            {
                **task_to_dict(task),
                "sheet_local_index": local_index,
                "prediction": prediction,
                "raw_sheet_prediction": raw_items,
                "raw_response": raw_response,
                "error": error if prediction is None else None,
            }
        )
    return {
        "task_type": task_type,
        "task_ids": [task.task_id for task in batch],
        "raw_prediction": raw_items,
        "raw_response": raw_response,
        "error": error,
        "sheet_path": str(keep_sheet_path) if keep_sheet_path else None,
        "records": records,
    }


def completed_ids(results_path: Path) -> set[str]:
    return {str(row.get("id")) for row in read_jsonl(results_path) if row.get("prediction") is not None}


def result_records_by_id(results_path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(results_path):
        if row.get("prediction") is not None:
            records[str(row["id"])] = row
    return records


def write_task_manifest(path: Path, tasks: list[TaskRecord]) -> None:
    atomic_write_json(path.with_suffix(".summary.json"), {"count": len(tasks)}, pretty=True)
    with path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task_to_dict(task), ensure_ascii=False, separators=(",", ":")) + "\n")


def run_prelabel(
    tasks: list[TaskRecord],
    *,
    task_type: str,
    output_dir: Path,
    model_id: str,
    region: str,
    batch_size: int,
    tile_size: int,
    max_api_side: int,
    workers: int,
    keep_sheet_limit: int,
    max_tokens: int,
    retries: int,
) -> dict[str, Any]:
    task_dir = output_dir / task_type
    task_dir.mkdir(parents=True, exist_ok=True)
    results_path = task_dir / "results.jsonl"
    sheet_results_path = task_dir / "sheet_results.jsonl"
    manifest_path = task_dir / "tasks.jsonl"
    write_task_manifest(manifest_path, tasks)

    done = completed_ids(results_path)
    remaining = [task for task in tasks if task.task_id not in done]
    batches = batched(remaining, batch_size)
    print(
        f"[{task_type}] total={len(tasks)} done={len(done)} remaining={len(remaining)} sheets={len(batches)}",
        flush=True,
    )
    if not batches:
        return summarize_results(task_type, task_dir, model_id, batch_size)

    next_sheet_base = len(read_jsonl(sheet_results_path))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_index = {}
        for offset, batch in enumerate(batches):
            sheet_index = next_sheet_base + offset
            keep_path = None
            if sheet_index < keep_sheet_limit:
                keep_path = task_dir / "sheets" / f"sheet_{sheet_index:06d}.jpg"
            future = executor.submit(
                call_sheet,
                batch,
                task_type=task_type,
                model_id=model_id,
                region=region,
                tile_size=tile_size,
                max_api_side=max_api_side,
                keep_sheet_path=keep_path,
                max_tokens=max_tokens,
                retries=retries,
            )
            future_to_index[future] = sheet_index
        completed = 0
        for future in as_completed(future_to_index):
            sheet_index = future_to_index[future]
            result = future.result()
            for record in result["records"]:
                record["sheet_index"] = sheet_index
            sheet_row = {
                "sheet_index": sheet_index,
                "task_type": task_type,
                "task_ids": result["task_ids"],
                "sheet_path": result["sheet_path"],
                "raw_prediction": result["raw_prediction"],
                "raw_response": result["raw_response"],
                "error": result["error"],
            }
            append_jsonl(sheet_results_path, [sheet_row])
            append_jsonl(results_path, result["records"])
            completed += 1
            if completed == 1 or completed % 50 == 0 or completed == len(batches):
                print(
                    f"[{task_type}] sheets_done={completed}/{len(batches)} "
                    f"records_done={len(done) + completed * batch_size}",
                    flush=True,
                )
    return summarize_results(task_type, task_dir, model_id, batch_size)


def summarize_results(task_type: str, task_dir: Path, model_id: str, batch_size: int) -> dict[str, Any]:
    rows = read_jsonl(task_dir / "results.jsonl")
    counter = Counter()
    parse_failed = 0
    for row in rows:
        pred = row.get("prediction")
        if not pred:
            parse_failed += 1
            counter["parse_failed"] += 1
        elif task_type == "image":
            counter[pred.get("image_type", "unknown")] += 1
        else:
            counter[pred.get("shape_type", "unknown")] += 1
    sheet_rows = read_jsonl(task_dir / "sheet_results.jsonl")
    summary = {
        "task": task_type,
        "model_id": model_id,
        "batch_size": batch_size,
        "result_records": len(rows),
        "unique_success": len(result_records_by_id(task_dir / "results.jsonl")),
        "sheet_count": len(sheet_rows),
        "parse_failed": parse_failed,
        "sheet_errors": sum(1 for row in sheet_rows if row.get("error")),
        "distribution": dict(sorted(counter.items())),
    }
    atomic_write_json(task_dir / "summary.json", summary, pretty=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def merge_results(raw_json_dir: Path, results_path: Path, task_type: str) -> dict[str, Any]:
    records = result_records_by_id(results_path)
    by_json: dict[Path, list[dict[str, Any]]] = {}
    for record in records.values():
        by_json.setdefault(Path(record["source_json"]), []).append(record)
    changed_files = 0
    merged = 0
    skipped_missing = 0
    for json_path, file_records in sorted(by_json.items()):
        if not json_path.exists() or raw_json_dir not in json_path.parents:
            skipped_missing += len(file_records)
            continue
        obj = load_json(json_path)
        instances = obj.get("instances", [])
        changed = False
        for record in file_records:
            index = int(record["instance_index"])
            if index < 0 or index >= len(instances):
                skipped_missing += 1
                continue
            inst = instances[index]
            if task_type == "image" and inst.get("label") != "image":
                skipped_missing += 1
                continue
            if task_type == "shape" and inst.get("label") != "shape":
                skipped_missing += 1
                continue
            pred = record.get("prediction")
            if not isinstance(pred, dict):
                continue
            extra = inst.get("extra")
            if not isinstance(extra, dict):
                extra = {}
                inst["extra"] = extra
            params = extra.get("parameters")
            if not isinstance(params, dict):
                params = {}
                extra["parameters"] = params
            if task_type == "image":
                params["image_type"] = pred["image_type"]
            else:
                params.clear()
                params.update(pred)
            changed = True
            merged += 1
        if changed:
            atomic_write_json(json_path, obj, pretty=True)
            changed_files += 1
    summary = {
        "task": task_type,
        "result_records": len(records),
        "merged_instances": merged,
        "changed_files": changed_files,
        "skipped_missing": skipped_missing,
    }
    return summary


def build_review_sample(
    output_dir: Path,
    *,
    task_type: str,
    results_path: Path,
    sample_size: int,
    seed: int,
) -> None:
    rows = list(result_records_by_id(results_path).values())
    rng = random.Random(seed)
    rng.shuffle(rows)
    rows = rows[:sample_size]
    review_dir = output_dir / task_type / "review"
    if review_dir.exists():
        shutil.rmtree(review_dir)
    review_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    html = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<style>body{font-family:Arial,sans-serif;margin:20px;color:#222}"
        "table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:8px;vertical-align:top}"
        "img{max-width:360px;max-height:280px}pre{font-size:12px;white-space:pre-wrap}</style></head><body>",
        f"<h1>{task_type} attribute review sample</h1>",
        "<table><tr><th>id</th><th>prediction</th><th>crop</th><th>json</th></tr>",
    ]
    for row in rows:
        item_dir = review_dir / str(row["id"])
        item_dir.mkdir(parents=True, exist_ok=True)
        with Image.open(row["source_image"]) as source:
            crop = source.convert("RGB").crop(tuple(row["crop_box"]))
        crop_path = item_dir / "crop.jpg"
        crop.save(crop_path, quality=92)
        annotation = {
            "id": row["id"],
            "source_json": row["source_json"],
            "source_image": row["source_image"],
            "instance_index": row["instance_index"],
            "bbox": row["bbox"],
            "crop_box": row["crop_box"],
            "parameters": row["prediction"],
            "raw_response": row.get("raw_response"),
            "error": row.get("error"),
        }
        atomic_write_json(item_dir / "annotation.json", annotation, pretty=True)
        manifest.append(annotation)
        html.append(
            "<tr>"
            f"<td>{row['id']}</td>"
            f"<td><pre>{json.dumps(row['prediction'], ensure_ascii=False, indent=2)}</pre></td>"
            f"<td><img src='{row['id']}/crop.jpg'></td>"
            f"<td><a href='{row['id']}/annotation.json'>annotation.json</a></td>"
            "</tr>"
        )
    html.append("</table></body></html>")
    (review_dir / "index.html").write_text("\n".join(html), encoding="utf-8")
    atomic_write_json(review_dir / "manifest.json", manifest, pretty=True)


def active_tasks(args: argparse.Namespace) -> list[str]:
    if args.task == "all":
        return ["image", "shape"]
    return [args.task]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-json-dir", default="data/raw/json_20260706")
    parser.add_argument("--images-dir", default="data/raw/images")
    parser.add_argument("--output-dir", default="temp/attr_sheet_prelabel_20260706")
    parser.add_argument("--task", choices=["image", "shape", "all"], default="all")
    parser.add_argument("--model-id", default="au.anthropic.claude-opus-4-8")
    parser.add_argument("--region", default="ap-southeast-2")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--tile-size", type=int, default=360)
    parser.add_argument("--max-api-side", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="0 means all collected tasks.")
    parser.add_argument("--min-area", type=int, default=16)
    parser.add_argument("--keep-sheet-limit", type=int, default=200)
    parser.add_argument("--review-sample", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()

    raw_json_dir = Path(args.raw_json_dir).resolve()
    images_dir = Path(args.images_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_summaries: dict[str, Any] = {}
    merge_summaries: dict[str, Any] = {}
    for task_type in active_tasks(args):
        tasks = collect_tasks(
            raw_json_dir,
            images_dir,
            task_type,
            skip_existing=args.skip_existing,
            min_area=args.min_area,
        )
        if args.limit > 0:
            tasks = tasks[: args.limit]
        run_summaries[task_type] = run_prelabel(
            tasks,
            task_type=task_type,
            output_dir=output_dir,
            model_id=args.model_id,
            region=args.region,
            batch_size=args.batch_size,
            tile_size=args.tile_size,
            max_api_side=args.max_api_side,
            workers=args.workers,
            keep_sheet_limit=args.keep_sheet_limit,
            max_tokens=args.max_tokens,
            retries=args.retries,
        )
        build_review_sample(
            output_dir,
            task_type=task_type,
            results_path=output_dir / task_type / "results.jsonl",
            sample_size=args.review_sample,
            seed=args.seed,
        )
        if args.merge:
            merge_summaries[task_type] = merge_results(
                raw_json_dir,
                output_dir / task_type / "results.jsonl",
                task_type,
            )
            print(json.dumps(merge_summaries[task_type], ensure_ascii=False, indent=2), flush=True)

    summary = {
        "model_id": args.model_id,
        "region": args.region,
        "batch_size": args.batch_size,
        "raw_json_dir": str(raw_json_dir),
        "images_dir": str(images_dir),
        "output_dir": str(output_dir),
        "run": run_summaries,
        "merge": merge_summaries,
    }
    atomic_write_json(output_dir / "summary.json", summary, pretty=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
