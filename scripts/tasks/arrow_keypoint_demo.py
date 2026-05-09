#!/usr/bin/env python3
from __future__ import annotations

import base64
import io
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import yaml

# Gradio/httpx may fail to import when SOCKS proxy env vars are set but socksio is
# not installed. This demo only talks to local vLLM by default, so ignore proxies.
for _name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
    os.environ.pop(_name, None)

import gradio as gr  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from shaft.codec import decode_with_codec  # noqa: E402


FALLBACK_SYSTEM_PROMPT = "You are a visual annotation assistant. Return only valid JSON with no markdown or extra text."
FALLBACK_ARROW_GROUNDING_PROMPT = (
    "Detect every arrow instance in the image.\n"
    "Return all detections as a JSON array of objects with label arrow and bbox_2d."
)
FALLBACK_LAYOUT_GROUNDING_PROMPT = (
    "Detect every top-level layout element labeled as icon, image, or shape.\n"
    "Use image for one whole picture region and do not split its internal content.\n"
    "Use icon only for small standalone symbols.\n"
    "Return all detections as a JSON array of objects with label and bbox_2d."
)
FALLBACK_ARROW_KEYPOINT_PROMPT = (
    "Predict the ordered keypoints of the central arrow and return a JSON object with keypoints_2d."
)

DEFAULT_ARROW_PROMPT_PATH = REPO_ROOT / "configs/prompts/grounding_arrow.yaml"
DEFAULT_LAYOUT_PROMPT_PATH = REPO_ROOT / "configs/prompts/grounding_layout.yaml"
DEFAULT_KEYPOINT_PROMPT_PATH = REPO_ROOT / "configs/prompts/keypoint_arrow.yaml"
NUM_BINS = 1000
OUTPUT_DIR = REPO_ROOT / "temp/arrow_keypoint_demo/outputs"
LABELS = ("arrow", "icon", "image", "shape")
COLORS = {
    "arrow": (230, 70, 70),
    "icon": (60, 120, 240),
    "image": (40, 170, 90),
    "shape": (190, 120, 30),
}
ARROW_PALETTE = (
    (255, 85, 85),
    (20, 184, 166),
    (255, 190, 64),
    (120, 160, 255),
    (255, 120, 220),
    (130, 230, 90),
    (255, 140, 80),
    (170, 130, 255),
)


def _load_prompt(path: Path, *, fallback_user_prompt: str) -> tuple[str, str, str]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return FALLBACK_SYSTEM_PROMPT, fallback_user_prompt, path.stem
    prompt = payload.get("prompt") if isinstance(payload, dict) else {}
    metadata = payload.get("metadata") if isinstance(payload, dict) else {}
    if not isinstance(prompt, dict):
        prompt = {}
    if not isinstance(metadata, dict):
        metadata = {}
    system_prompt = str(prompt.get("system_prompt") or FALLBACK_SYSTEM_PROMPT).strip()
    user_prompt = str(prompt.get("user_prompt") or fallback_user_prompt).strip()
    prompt_id = str(metadata.get("id") or path.stem).strip()
    return system_prompt, user_prompt, prompt_id


DEFAULT_SYSTEM_PROMPT, DEFAULT_ARROW_GROUNDING_PROMPT, DEFAULT_ARROW_PROMPT_ID = _load_prompt(
    DEFAULT_ARROW_PROMPT_PATH,
    fallback_user_prompt=FALLBACK_ARROW_GROUNDING_PROMPT,
)
_, DEFAULT_LAYOUT_GROUNDING_PROMPT, DEFAULT_LAYOUT_PROMPT_ID = _load_prompt(
    DEFAULT_LAYOUT_PROMPT_PATH,
    fallback_user_prompt=FALLBACK_LAYOUT_GROUNDING_PROMPT,
)
_, DEFAULT_ARROW_KEYPOINT_PROMPT, DEFAULT_KEYPOINT_PROMPT_ID = _load_prompt(
    DEFAULT_KEYPOINT_PROMPT_PATH,
    fallback_user_prompt=FALLBACK_ARROW_KEYPOINT_PROMPT,
)


def _decode_json_any(text: str) -> tuple[Any, bool, str | None]:
    decoded = decode_with_codec("json_any", str(text or ""))
    if decoded.valid:
        return decoded.parsed, bool(decoded.partial), None
    return None, False, decoded.error


def _extract_json(text: str) -> Any:
    parsed, _partial, _error = _decode_json_any(text)
    if parsed is not None:
        return parsed

    text = str(text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    candidates = []
    for left, right in (("[", "]"), ("{", "}")):
        start = text.find(left)
        end = text.rfind(right)
        if start >= 0 and end > start:
            candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _items_from_payload(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("detections", "instances", "objects", "items", "result"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        if "bbox_2d" in payload:
            return [payload]
    return []


def _bbox_from_item(item: Any) -> list[float] | None:
    if not isinstance(item, dict):
        return None
    for key in ("bbox_2d", "bbox", "box_2d", "box"):
        value = item.get(key)
        if isinstance(value, list) and len(value) >= 4:
            try:
                return [float(value[0]), float(value[1]), float(value[2]), float(value[3])]
            except (TypeError, ValueError):
                return None
    return None


def _bbox_to_pixels(bbox: list[float], width: int, height: int) -> list[int] | None:
    x1, y1, x2, y2 = bbox
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= float(NUM_BINS):
        x1 = x1 / float(NUM_BINS) * width
        x2 = x2 / float(NUM_BINS) * width
        y1 = y1 / float(NUM_BINS) * height
        y2 = y2 / float(NUM_BINS) * height
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    left = max(0, min(width, int(round(left))))
    right = max(0, min(width, int(round(right))))
    top = max(0, min(height, int(round(top))))
    bottom = max(0, min(height, int(round(bottom))))
    if right - left < 2 or bottom - top < 2:
        return None
    return [left, top, right, bottom]


def _iou(a: list[int], b: list[int]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def _dedupe(instances: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for instance in instances:
        bbox = instance["bbox"]
        label = str(instance.get("label"))
        if any(str(old.get("label")) == label and _iou(bbox, old["bbox"]) >= threshold for old in kept):
            continue
        kept.append(instance)
    return kept


def _image_to_data_url(image: Image.Image, *, mime_type: str = "image/png") -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return f"data:{mime_type};base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"


def _post_chat(
    *,
    endpoint: str,
    model_name: str,
    system_prompt: str,
    image: Image.Image,
    prompt: str,
    max_tokens: int,
    min_pixels: int,
    max_pixels: int,
    timeout: float,
) -> str:
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _image_to_data_url(image)}},
                    {"type": "text", "text": prompt},
                ],
            },
        ],
        "max_tokens": int(max_tokens),
        "temperature": 0.0,
        "top_p": 1.0,
        "mm_processor_kwargs": {"min_pixels": int(min_pixels), "max_pixels": int(max_pixels)},
    }
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=float(timeout)) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body[:500]}") from exc
    return str(data["choices"][0]["message"].get("content", "")).strip()


def _parse_arrow_boxes(text: str, width: int, height: int, iou_threshold: float) -> list[dict[str, Any]]:
    payload = _extract_json(text)
    instances: list[dict[str, Any]] = []
    for item in _items_from_payload(payload):
        bbox = _bbox_from_item(item)
        if bbox is None:
            continue
        pixel_bbox = _bbox_to_pixels(bbox, width, height)
        if pixel_bbox is None:
            continue
        instances.append(
            {
                "label": "arrow",
                "bbox": pixel_bbox,
                "bbox_2d_raw": bbox,
            }
        )
    return _dedupe(instances, float(iou_threshold))


def _normalize_layout_label(value: Any) -> str | None:
    label = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if label == "shape_combination":
        label = "icon"
    return label if label in {"icon", "image", "shape"} else None


def _parse_layout_boxes(text: str, width: int, height: int, iou_threshold: float) -> list[dict[str, Any]]:
    payload = _extract_json(text)
    instances: list[dict[str, Any]] = []
    for item in _items_from_payload(payload):
        if not isinstance(item, dict):
            continue
        label = _normalize_layout_label(item.get("label"))
        bbox = _bbox_from_item(item)
        if label is None or bbox is None:
            continue
        pixel_bbox = _bbox_to_pixels(bbox, width, height)
        if pixel_bbox is None:
            continue
        instances.append(
            {
                "label": label,
                "bbox": pixel_bbox,
                "bbox_2d_raw": bbox,
            }
        )
    return _dedupe(instances, float(iou_threshold))


def _crop_box_for_bbox(bbox: list[int], width: int, height: int, padding: float) -> list[int]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)
    pad_x = bw * float(padding)
    pad_y = bh * float(padding)
    return [
        int(math.floor(max(0.0, x1 - pad_x))),
        int(math.floor(max(0.0, y1 - pad_y))),
        int(math.ceil(min(float(width), x2 + pad_x))),
        int(math.ceil(min(float(height), y2 + pad_y))),
    ]


def _parse_keypoints(text: str) -> list[list[float]]:
    payload = _extract_json(text)
    if isinstance(payload, dict):
        raw_points = payload.get("keypoints_2d")
    elif isinstance(payload, list):
        raw_points = payload
    else:
        raw_points = None
    if not isinstance(raw_points, list):
        return []
    points: list[list[float]] = []
    for item in raw_points:
        if not isinstance(item, list) or len(item) != 2:
            continue
        try:
            x = max(0.0, min(float(NUM_BINS), float(item[0])))
            y = max(0.0, min(float(NUM_BINS), float(item[1])))
        except (TypeError, ValueError):
            continue
        points.append([x, y])
    return points


def _points_to_global(points_2d: list[list[float]], crop_box: list[int]) -> list[list[float]]:
    x1, y1, x2, y2 = crop_box
    crop_w = max(x2 - x1, 1)
    crop_h = max(y2 - y1, 1)
    global_points: list[list[float]] = []
    for x, y in points_2d:
        gx = x1 + x / float(NUM_BINS) * crop_w
        gy = y1 + y / float(NUM_BINS) * crop_h
        global_points.append([round(gx, 2), round(gy, 2)])
    return global_points


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def _draw_result(
    image: Image.Image,
    instances: list[dict[str, Any]],
    *,
    label_filter: str | None = None,
) -> Image.Image:
    arrow_blackout = label_filter == "arrow"
    canvas = Image.new("RGB", image.size, (0, 0, 0)) if arrow_blackout else image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    w, h = canvas.size
    line_w = max(2, round(max(w, h) / 500))
    radius = max(3, round(max(w, h) / 350))
    label_font = _font(max(12, round(max(w, h) / 90)))
    point_color = (20, 184, 166)
    text_color = (255, 255, 255)
    filtered = [
        instance for instance in instances if label_filter is None or str(instance.get("label")) == label_filter
    ]

    for idx, instance in enumerate(filtered, start=1):
        instance_label = str(instance.get("label", "unknown"))
        if arrow_blackout:
            color = ARROW_PALETTE[(idx - 1) % len(ARROW_PALETTE)]
        else:
            color = COLORS.get(instance_label, (240, 190, 40))
        x1, y1, x2, y2 = [int(v) for v in instance["bbox"]]
        if not arrow_blackout:
            draw.rectangle([x1, y1, x2, y2], outline=color, width=line_w)
        label_text = f"{instance_label} {idx}"
        if not arrow_blackout:
            tb = draw.textbbox((0, 0), label_text, font=label_font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
            ly = max(0, y1 - th - 2 * line_w)
            draw.rectangle([x1, ly, x1 + tw + 3 * line_w, ly + th + 2 * line_w], fill=color)
            draw.text((x1 + line_w, ly + line_w), label_text, fill=text_color, font=label_font)

        points = instance.get("keypoints_xy") or []
        if len(points) >= 2:
            if arrow_blackout:
                draw.line(
                    [(float(x), float(y)) for x, y in points],
                    fill=(255, 255, 255),
                    width=line_w + 5,
                )
            draw.line(
                [(float(x), float(y)) for x, y in points],
                fill=color if arrow_blackout else point_color,
                width=line_w,
            )
        for point_idx, point in enumerate(points, start=1):
            px, py = float(point[0]), float(point[1])
            if arrow_blackout:
                draw.ellipse(
                    [px - radius - 3, py - radius - 3, px + radius + 3, py + radius + 3],
                    fill=(255, 255, 255),
                )
            draw.ellipse(
                [px - radius, py - radius, px + radius, py + radius],
                fill=color if arrow_blackout else point_color,
            )
            draw.text(
                (px + radius + 1, py + radius + 1),
                str(point_idx),
                fill=color if arrow_blackout else point_color,
                font=label_font,
            )
    if label_filter is not None:
        title = f"{label_filter}: {len(filtered)}"
        tb = draw.textbbox((0, 0), title, font=label_font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        margin = max(4, line_w * 2)
        color = COLORS.get(label_filter, (240, 190, 40))
        draw.rectangle([margin, margin, margin + tw + 3 * line_w, margin + th + 2 * line_w], fill=color)
        draw.text((margin + line_w, margin + line_w), title, fill=text_color, font=label_font)
    return canvas


def predict(
    image: Image.Image | None,
    endpoint: str,
    model_name: str,
    system_prompt: str,
    layout_prompt: str,
    arrow_prompt: str,
    keypoint_prompt: str,
    stage1_max_tokens: int,
    stage2_max_tokens: int,
    layout_max_tokens: int,
    stage1_max_pixels: int,
    stage2_max_pixels: int,
    layout_max_pixels: int,
    crop_padding: float,
    iou_threshold: float,
    timeout: float,
) -> tuple[Image.Image | None, list[tuple[Image.Image, str]], dict[str, Any], str]:
    if image is None:
        return None, [], {}, "no image"

    started = time.perf_counter()
    image = image.convert("RGB")
    width, height = image.size

    layout_text = _post_chat(
        endpoint=endpoint,
        model_name=model_name,
        system_prompt=system_prompt,
        image=image,
        prompt=layout_prompt,
        max_tokens=int(layout_max_tokens),
        min_pixels=200704,
        max_pixels=int(layout_max_pixels),
        timeout=float(timeout),
    )
    layout_parsed, layout_partial, layout_parse_error = _decode_json_any(layout_text)
    layout_instances = _parse_layout_boxes(layout_text, width, height, float(iou_threshold))

    stage1_text = _post_chat(
        endpoint=endpoint,
        model_name=model_name,
        system_prompt=system_prompt,
        image=image,
        prompt=arrow_prompt,
        max_tokens=int(stage1_max_tokens),
        min_pixels=200704,
        max_pixels=int(stage1_max_pixels),
        timeout=float(timeout),
    )
    stage1_parsed, stage1_partial, stage1_parse_error = _decode_json_any(stage1_text)
    arrow_instances = _parse_arrow_boxes(stage1_text, width, height, float(iou_threshold))

    for index, instance in enumerate(arrow_instances, start=1):
        crop_box = _crop_box_for_bbox(instance["bbox"], width, height, float(crop_padding))
        crop = image.crop(tuple(crop_box))
        stage2_text = _post_chat(
            endpoint=endpoint,
            model_name=model_name,
            system_prompt=system_prompt,
            image=crop,
            prompt=keypoint_prompt,
            max_tokens=int(stage2_max_tokens),
            min_pixels=50176,
            max_pixels=int(stage2_max_pixels),
            timeout=float(timeout),
        )
        stage2_parsed, stage2_partial, stage2_parse_error = _decode_json_any(stage2_text)
        keypoints_2d = _parse_keypoints(stage2_text)
        instance.update(
            {
                "index": index,
                "crop_box": crop_box,
                "keypoints_2d": keypoints_2d,
                "keypoints_xy": _points_to_global(keypoints_2d, crop_box),
                "stage2_raw_text": stage2_text,
                "stage2_parse_partial": stage2_partial,
                "stage2_parse_error": stage2_parse_error,
                "stage2_parsed_type": type(stage2_parsed).__name__,
            }
        )

    instances = layout_instances + arrow_instances
    rendered = _draw_result(image, instances)
    label_images: list[tuple[Image.Image, str]] = []
    label_counts: dict[str, int] = {}
    for label in LABELS:
        count = sum(1 for instance in instances if str(instance.get("label")) == label)
        label_counts[label] = count
        label_rendered = _draw_result(image, instances, label_filter=label)
        label_images.append((label_rendered, f"{label}: {count}"))
    result = {
        "image_size": [width, height],
        "instances": instances,
        "label_counts": label_counts,
        "layout_raw_text": layout_text,
        "layout_parse_partial": layout_partial,
        "layout_parse_error": layout_parse_error,
        "layout_parsed_type": type(layout_parsed).__name__,
        "stage1_raw_text": stage1_text,
        "stage1_parse_partial": stage1_partial,
        "stage1_parse_error": stage1_parse_error,
        "stage1_parsed_type": type(stage1_parsed).__name__,
        "config": {
            "endpoint": endpoint,
            "model_name": model_name,
            "stage1_max_tokens": int(stage1_max_tokens),
            "stage2_max_tokens": int(stage2_max_tokens),
            "layout_max_tokens": int(layout_max_tokens),
            "stage1_max_pixels": int(stage1_max_pixels),
            "stage2_max_pixels": int(stage2_max_pixels),
            "layout_max_pixels": int(layout_max_pixels),
            "crop_padding": float(crop_padding),
            "iou_threshold": float(iou_threshold),
            "prompt_ids": {
                "layout": DEFAULT_LAYOUT_PROMPT_ID,
                "arrow": DEFAULT_ARROW_PROMPT_ID,
                "keypoint": DEFAULT_KEYPOINT_PROMPT_ID,
            },
            "prompts": {
                "system": system_prompt,
                "layout": layout_prompt,
                "arrow": arrow_prompt,
                "keypoint": keypoint_prompt,
            },
        },
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    rendered.save(run_dir / "all.jpg", quality=90)
    for label, (label_rendered, _caption) in zip(LABELS, label_images, strict=True):
        label_rendered.save(run_dir / f"{label}.jpg", quality=90)
    (run_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    status = (
        f"ok: counts={label_counts} elapsed={time.perf_counter() - started:.2f}s "
        f"saved={run_dir}"
    )
    return rendered, label_images, result, status


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Arrow bbox + keypoint demo") as demo:
        gr.Markdown("## Arrow bbox + keypoint demo")
        with gr.Row():
            image = gr.Image(label="Input image", type="pil")
            rendered = gr.Image(label="Preview", type="pil")
        with gr.Row():
            endpoint = gr.Textbox(label="vLLM endpoint", value="http://127.0.0.1:8100")
            model_name = gr.Textbox(label="served model", value="prelabel_latest")
        with gr.Accordion("Prompts", open=False):
            system_prompt = gr.Textbox(label="system prompt", value=DEFAULT_SYSTEM_PROMPT, lines=2)
            layout_prompt = gr.Textbox(label=f"layout prompt ({DEFAULT_LAYOUT_PROMPT_ID})", value=DEFAULT_LAYOUT_GROUNDING_PROMPT, lines=5)
            arrow_prompt = gr.Textbox(label=f"arrow bbox prompt ({DEFAULT_ARROW_PROMPT_ID})", value=DEFAULT_ARROW_GROUNDING_PROMPT, lines=4)
            keypoint_prompt = gr.Textbox(label=f"arrow keypoint prompt ({DEFAULT_KEYPOINT_PROMPT_ID})", value=DEFAULT_ARROW_KEYPOINT_PROMPT, lines=4)
        with gr.Row():
            stage1_max_tokens = gr.Number(label="bbox max tokens", value=4096, precision=0)
            stage2_max_tokens = gr.Number(label="keypoint max tokens", value=256, precision=0)
            layout_max_tokens = gr.Number(label="layout max tokens", value=4096, precision=0)
            timeout = gr.Number(label="timeout seconds", value=300, precision=0)
        with gr.Row():
            stage1_max_pixels = gr.Number(label="bbox max pixels", value=1048576, precision=0)
            stage2_max_pixels = gr.Number(label="keypoint max pixels", value=262144, precision=0)
            layout_max_pixels = gr.Number(label="layout max pixels", value=1048576, precision=0)
        with gr.Row():
            crop_padding = gr.Slider(label="crop padding", minimum=0.0, maximum=0.6, step=0.05, value=0.25)
            iou_threshold = gr.Slider(label="bbox dedupe IoU", minimum=0.5, maximum=1.0, step=0.01, value=0.95)
        run = gr.Button("Predict")
        status = gr.Textbox(label="Status")
        label_gallery = gr.Gallery(label="Per-label previews", columns=4, object_fit="contain")
        output_json = gr.JSON(label="JSON")
        run.click(
            predict,
            inputs=[
                image,
                endpoint,
                model_name,
                system_prompt,
                layout_prompt,
                arrow_prompt,
                keypoint_prompt,
                stage1_max_tokens,
                stage2_max_tokens,
                layout_max_tokens,
                stage1_max_pixels,
                stage2_max_pixels,
                layout_max_pixels,
                crop_padding,
                iou_threshold,
                timeout,
            ],
            outputs=[rendered, label_gallery, output_json, status],
        )
    return demo


if __name__ == "__main__":
    build_demo().launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7861")),
        show_error=True,
    )
