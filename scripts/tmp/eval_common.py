#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import gc
import json
import re
import time
import warnings
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageColor, ImageDraw, ImageFont
from tqdm.auto import tqdm

warnings.filterwarnings(
    "ignore",
    message=r"TRL currently supports vLLM versions: .*",
    category=UserWarning,
)

from shaft.codec import CODEC_REGISTRY, decode_with_codec  # noqa: E402
from shaft.codec.base import ShaftCodecResult  # noqa: E402
from shaft.config import RuntimeConfig, load_config  # noqa: E402
from shaft.data import SFTRecord, load_jsonl_sft_records  # noqa: E402
from shaft.infer import InferGenerationConfig, ShaftInferRequest, ShaftInferResponse  # noqa: E402
from shaft.infer.engine import HFLocalInferAdapter  # noqa: E402
from shaft.metrics import EVAL_METRIC_REGISTRY, build_eval_metric  # noqa: E402
from shaft.model import build_model_tokenizer_processor  # noqa: E402
from shaft.training.checkpointing import inspect_checkpoint_layout  # noqa: E402

_BOX_PALETTE = (
    "#2563EB",
    "#DC2626",
    "#059669",
    "#D97706",
    "#7C3AED",
    "#0F766E",
    "#DB2777",
    "#65A30D",
)
_LABEL_COLOR_OFFSETS = {
    "image": 0,
    "icon": 3,
}


def _optional_bool(text: str | None) -> bool | None:
    if text is None:
        return None
    normalized = str(text).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {text!r}")


def build_parser(
    *,
    task_name: str,
    default_input: str,
    default_dataset_name: str,
    default_codec: str,
    default_metrics: tuple[str, ...],
    default_output_subdir: str,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Temporary offline eval for {task_name}")
    parser.add_argument("--config", required=True, help="Path to training YAML config.")
    parser.add_argument("--input", default=default_input, help="SFT JSONL to evaluate.")
    parser.add_argument("--dataset-name", default=default_dataset_name, help="Fallback dataset name.")
    parser.add_argument("--codec", default=default_codec, help="Codec name.")
    parser.add_argument("--metrics", default=",".join(default_metrics), help="Comma-separated metrics.")
    parser.add_argument("--checkpoint", action="append", default=[], help="Checkpoint directory.")
    parser.add_argument("--checkpoint-root", default=None, help="Directory to scan checkpoint-*.")
    parser.add_argument("--include-best", action="store_true", default=False, help="Include best/.")
    parser.add_argument("--include-final", action="store_true", default=False, help="Include root ckpt.")
    parser.add_argument("--output-root", default=None, help="Directory for eval outputs.")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit.")
    parser.add_argument("--batch-size", type=int, default=1, help="Offline eval batch size.")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Override max_new_tokens.")
    parser.add_argument("--do-sample", type=_optional_bool, default=None, help="Override do_sample.")
    parser.add_argument("--temperature", type=float, default=None, help="Override temperature.")
    parser.add_argument("--top-p", type=float, default=None, help="Override top_p.")
    parser.add_argument("--top-k", type=int, default=None, help="Override top_k.")
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="Override repetition_penalty.",
    )
    parser.add_argument("--min-pixels", type=int, default=None, help="Override min_pixels.")
    parser.add_argument("--max-pixels", type=int, default=None, help="Override max_pixels.")
    parser.add_argument("--device", default=None, help="Override device, for example cuda:0.")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=False,
        help="Continue when one checkpoint fails.",
    )
    parser.add_argument(
        "--save-visualizations",
        type=_optional_bool,
        default=True,
        help="Save prediction visualizations.",
    )
    parser.set_defaults(task_name=task_name, default_output_subdir=default_output_subdir)
    return parser


def _parse_metric_names(raw: str) -> tuple[str, ...]:
    metrics = tuple(item.strip().lower() for item in str(raw).split(",") if item.strip())
    if not metrics:
        raise ValueError("metrics cannot be empty.")
    unknown = [item for item in metrics if item not in EVAL_METRIC_REGISTRY.keys()]
    if unknown:
        raise KeyError(f"Unknown metrics: {unknown}. Available: {sorted(EVAL_METRIC_REGISTRY.keys())}")
    return metrics


def _parse_target(raw: Any) -> Any:
    if isinstance(raw, str):
        try:
            return json.loads(raw.strip())
        except Exception:
            return raw
    return raw


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list | tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _resolve_codec(name: str) -> str:
    codec = str(name).strip().lower()
    if not CODEC_REGISTRY.has(codec):
        raise ValueError(f"Unknown codec {codec!r}. Available: {sorted(CODEC_REGISTRY.keys())}")
    return codec


def _load_records(input_jsonl: Path, dataset_name: str) -> list[SFTRecord]:
    if not input_jsonl.exists():
        raise FileNotFoundError(f"Input jsonl not found: {input_jsonl}")
    return load_jsonl_sft_records(input_jsonl, dataset_name=dataset_name)


def _collect_checkpoints(
    *,
    explicit: Iterable[Path],
    root: Path | None,
    include_best: bool,
    include_final: bool,
) -> list[Path]:
    checkpoints: list[Path] = []
    seen: set[Path] = set()

    for raw_path in explicit:
        path = raw_path.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint path not found: {path}")
        if inspect_checkpoint_layout(path).kind not in {"full", "adapter"}:
            raise ValueError(f"Checkpoint is not usable: {path}")
        if path not in seen:
            checkpoints.append(path)
            seen.add(path)

    if root is None:
        return checkpoints

    root = root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"checkpoint-root not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"checkpoint-root is not a directory: {root}")

    for child in sorted(root.iterdir(), key=lambda item: item.name):
        if not child.is_dir():
            continue
        if child.name.startswith("checkpoint-") or (include_best and child.name == "best"):
            if inspect_checkpoint_layout(child).kind in {"full", "adapter"} and child not in seen:
                checkpoints.append(child)
                seen.add(child)

    if include_final and inspect_checkpoint_layout(root).kind in {"full", "adapter"} and root not in seen:
        checkpoints.append(root)

    return checkpoints


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def _sanitize_filename(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]", "_", text.strip())
    return cleaned[:96] or "sample"


def _coerce_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, list | tuple) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if not (x2 > x1 and y2 > y1):
        return None
    return x1, y1, x2, y2


def _coerce_keypoints(value: Any) -> list[tuple[float, float]] | None:
    if not isinstance(value, list | tuple):
        return None
    points: list[tuple[float, float]] = []
    for item in value:
        if not isinstance(item, list | tuple) or len(item) != 2:
            continue
        try:
            points.append((float(item[0]), float(item[1])))
        except (TypeError, ValueError):
            continue
    return points or None


def _scale_from_1000(x: float, y: float, width: int, height: int) -> tuple[float, float]:
    return float(x) * float(width) / 1000.0, float(y) * float(height) / 1000.0


def _resolve_box_line_width(image_width: int, image_height: int) -> int:
    short_edge = max(1, min(int(image_width), int(image_height)))
    return max(3, int(round(short_edge / 180.0)))


def _resolve_point_radius(image_width: int, image_height: int) -> int:
    short_edge = max(1, min(int(image_width), int(image_height)))
    return max(4, int(round(short_edge / 140.0)))


def _resolve_annotation_font_size(image_width: int, image_height: int) -> int:
    short_edge = max(1, min(int(image_width), int(image_height)))
    return max(12, min(28, int(round(short_edge / 42.0))))


def _load_annotation_font(font_size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=int(font_size))
    except Exception:
        return ImageFont.load_default()


def _resolve_box_color(label: str, index: int) -> tuple[int, int, int]:
    palette_index = (_LABEL_COLOR_OFFSETS.get(label, 0) + max(0, index - 1)) % len(_BOX_PALETTE)
    return ImageColor.getrgb(_BOX_PALETTE[palette_index])


def _draw_labeled_box(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: tuple[int, int, int, int],
    label: str,
    index: int,
    image_width: int,
    image_height: int,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
) -> None:
    x1, y1, x2, y2 = bbox
    box_label = label or "box"
    color = _resolve_box_color(box_label, index)
    line_width = _resolve_box_line_width(image_width, image_height)
    halo_width = line_width + max(2, line_width // 2)
    draw.rectangle([x1, y1, x2, y2], outline="white", width=halo_width)
    draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

    label_text = f"{index}:{box_label}"
    text_padding_x = max(4, line_width)
    text_padding_y = max(2, line_width // 2)
    text_bbox = draw.textbbox((0, 0), label_text, font=font)
    text_width = max(1, text_bbox[2] - text_bbox[0])
    text_height = max(1, text_bbox[3] - text_bbox[1])
    label_x1 = max(0, min(x1, image_width - text_width - (text_padding_x * 2)))
    preferred_y = y1 - text_height - (text_padding_y * 2) - line_width
    if preferred_y >= 0:
        label_y1 = preferred_y
    else:
        label_y1 = min(max(0, y1 + line_width), max(0, image_height - text_height - (text_padding_y * 2)))
    label_x2 = min(image_width, label_x1 + text_width + (text_padding_x * 2))
    label_y2 = min(image_height, label_y1 + text_height + (text_padding_y * 2))
    draw.rectangle([label_x1, label_y1, label_x2, label_y2], fill="white")
    draw.rectangle([label_x1, label_y1, label_x2, label_y2], outline=color, width=max(2, line_width - 1))
    draw.text((label_x1 + text_padding_x, label_y1 + text_padding_y), label_text, fill=color, font=font)


def _build_footer_image(image: Image.Image, lines: list[str]) -> Image.Image:
    footer_lines = [line for line in lines if line.strip()]
    if not footer_lines:
        return image
    draw = ImageDraw.Draw(image)
    spacing = 4
    footer_text = "\n".join(footer_lines)
    bbox = draw.multiline_textbbox((0, 0), footer_text, spacing=spacing)
    footer_height = max(32, (bbox[3] - bbox[1]) + 20)
    canvas = Image.new("RGB", (image.width, image.height + footer_height), "white")
    canvas.paste(image, (0, 0))
    canvas_draw = ImageDraw.Draw(canvas)
    canvas_draw.multiline_text((8, image.height + 8), footer_text, fill="black", spacing=spacing)
    return canvas


def _render_prediction_visualization(
    *,
    image_path: str,
    sample_id: str,
    sample_index: int,
    prediction: ShaftCodecResult,
    out_dir: Path,
) -> str | None:
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception:
        return None

    payload = prediction.parsed
    boxes: list[tuple[str, tuple[float, float, float, float]]] = []
    points: list[tuple[float, float]] | None = None
    summary_parts: list[str] = []

    if isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                continue
            bbox = _coerce_bbox(item.get("bbox_2d"))
            if bbox is None:
                continue
            x1, y1 = _scale_from_1000(bbox[0], bbox[1], image.width, image.height)
            x2, y2 = _scale_from_1000(bbox[2], bbox[3], image.width, image.height)
            boxes.append((str(item.get("label", "")).strip().lower(), (x1, y1, x2, y2)))
    elif isinstance(payload, dict):
        raw_points = _coerce_keypoints(payload.get("keypoints_2d"))
        if raw_points is not None:
            points = [_scale_from_1000(x, y, image.width, image.height) for x, y in raw_points]
        if payload.get("stroke_pattern") is not None:
            summary_parts.append(f"stroke={payload['stroke_pattern']}")
        if payload.get("geometry_style") is not None:
            summary_parts.append(f"geometry={payload['geometry_style']}")

    if not boxes and points is None and not summary_parts:
        return None

    draw = ImageDraw.Draw(image)
    font = _load_annotation_font(_resolve_annotation_font_size(image.width, image.height))
    point_radius = _resolve_point_radius(image.width, image.height)
    point_line_width = max(2, _resolve_box_line_width(image.width, image.height))

    for idx, (label, bbox) in enumerate(boxes, start=1):
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        _draw_labeled_box(
            draw,
            bbox=(x1, y1, x2, y2),
            label=label,
            index=idx,
            image_width=image.width,
            image_height=image.height,
            font=font,
        )

    if points is not None:
        for idx, (x, y) in enumerate(points, start=1):
            cx = int(round(x))
            cy = int(round(y))
            point_color = _resolve_box_color("keypoint", idx)
            draw.ellipse(
                [cx - point_radius, cy - point_radius, cx + point_radius, cy + point_radius],
                outline="white",
                fill="white",
                width=point_line_width + 1,
            )
            draw.ellipse(
                [cx - point_radius, cy - point_radius, cx + point_radius, cy + point_radius],
                outline=point_color,
                fill=point_color,
                width=point_line_width,
            )
        if len(points) >= 2:
            draw.line(
                [(int(round(x)), int(round(y))) for x, y in points],
                fill=ImageColor.getrgb("#14B8A6"),
                width=point_line_width,
                joint="curve",
            )

    footer_lines = [f"id={sample_id} idx={sample_index:06d}"]
    if not prediction.valid:
        footer_lines.append("pred: invalid")
    elif summary_parts:
        footer_lines.append(f"pred: {' '.join(summary_parts)}")
    if prediction.error_type:
        footer_lines.append(f"error: {prediction.error_type}")
    if boxes:
        box_parts: list[str] = []
        for idx, (label, bbox) in enumerate(boxes, start=1):
            x1, y1, x2, y2 = [int(round(v)) for v in bbox]
            box_parts.append(f"{idx}:{label or 'box'}[{x1},{y1},{x2},{y2}]")
        footer_lines.append("boxes: " + " ".join(box_parts))
    if points is not None:
        point_parts = [f"{idx}=({int(round(x))},{int(round(y))})" for idx, (x, y) in enumerate(points, start=1)]
        footer_lines.append("points: " + " ".join(point_parts))

    image = _build_footer_image(image, footer_lines)

    out_dir = out_dir / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{_sanitize_filename(f'{sample_id}_{sample_index:06d}')}.jpg"
    image.save(output_path, format="JPEG", quality=90, optimize=True)
    return str(output_path)


def _run_infer_batch(
    infer: HFLocalInferAdapter,
    records: list[SFTRecord],
    generation: InferGenerationConfig,
) -> list[ShaftInferResponse]:
    prompts: list[str] = []
    images: list[Image.Image] = []
    try:
        for record in records:
            messages = record.messages or infer._build_messages(
                user_prompt=record.user_prompt,
                system_prompt=record.system_prompt,
            )
            prompts.append(infer._apply_chat_template(messages))
            with Image.open(record.image_path) as image_obj:
                images.append(image_obj.convert("RGB"))
        batch = infer.model_adapter.build_processor_inputs(
            processor=infer.processor,
            tokenizer=infer.tokenizer,
            prompt_texts=prompts,
            images=images,
            min_pixels=infer.min_pixels,
            max_pixels=infer.max_pixels,
            padding_side="left",
        )
        batch = infer._move_batch_to_device(batch)
        generated = infer._generate(batch=batch, generation=generation)
        prompt_len = int(batch["input_ids"].shape[1])
        responses: list[ShaftInferResponse] = []
        for index in range(len(records)):
            output_ids = generated[index][prompt_len:].detach().cpu()
            responses.append(
                ShaftInferResponse(
                    text=infer._decode(output_ids),
                    prompt=prompts[index],
                    output_ids=[int(x) for x in output_ids.tolist()],
                    backend="hf_local",
                )
            )
        return responses
    finally:
        for image in images:
            image.close()


def _build_generation_config(config: RuntimeConfig, args: argparse.Namespace) -> InferGenerationConfig:
    return InferGenerationConfig(
        max_new_tokens=(
            int(args.max_new_tokens)
            if args.max_new_tokens is not None
            else int(config.eval.max_new_tokens)
        ),
        do_sample=bool(args.do_sample) if args.do_sample is not None else bool(config.eval.do_sample),
        temperature=(
            float(args.temperature)
            if args.temperature is not None
            else float(config.eval.temperature)
        ),
        top_p=float(args.top_p) if args.top_p is not None else 1.0,
        top_k=int(args.top_k) if args.top_k is not None else 50,
        repetition_penalty=(
            float(args.repetition_penalty) if args.repetition_penalty is not None else 1.0
        ),
    )


def _evaluate_checkpoint(
    *,
    checkpoint: Path,
    base_config: RuntimeConfig,
    records: list[SFTRecord],
    codec: str,
    metrics: tuple[str, ...],
    generation: InferGenerationConfig,
    output_root: Path,
    device: str | None,
    batch_size: int,
    limit: int | None,
    min_pixels: int | None,
    max_pixels: int | None,
    save_visualizations: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    infer = _build_infer_adapter(
        checkpoint=checkpoint,
        base_config=base_config,
        generation=generation,
        device=device,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    try:
        return _evaluate_records_with_infer(
            infer=infer,
            checkpoint=checkpoint,
            records=records,
            codec=codec,
            metrics=metrics,
            generation=generation,
            output_dir=output_root / checkpoint.name,
            batch_size=batch_size,
            limit=limit,
            save_visualizations=save_visualizations,
        )
    finally:
        del infer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _build_infer_adapter(
    *,
    checkpoint: Path,
    base_config: RuntimeConfig,
    generation: InferGenerationConfig,
    device: str | None,
    min_pixels: int | None,
    max_pixels: int | None,
) -> HFLocalInferAdapter:
    config = copy.deepcopy(base_config)
    config.train.init_from_checkpoint = str(checkpoint)
    artifacts = build_model_tokenizer_processor(config, init_from_checkpoint=config.train.init_from_checkpoint)
    return HFLocalInferAdapter(
        model=artifacts.model,
        tokenizer=artifacts.tokenizer,
        processor=artifacts.processor,
        model_adapter=artifacts.model_adapter,
        template=artifacts.template,
        device=device,
        min_pixels=int(base_config.data.min_pixels) if min_pixels is None else int(min_pixels),
        max_pixels=(
            int(base_config.data.max_pixels)
            if max_pixels is None and base_config.data.max_pixels is not None
            else (int(max_pixels) if max_pixels is not None else None)
        ),
        default_generation=generation,
    )


def _evaluate_records_with_infer(
    *,
    infer: HFLocalInferAdapter,
    checkpoint: Path,
    records: list[SFTRecord],
    codec: str,
    metrics: tuple[str, ...],
    generation: InferGenerationConfig,
    output_dir: Path,
    batch_size: int,
    limit: int | None,
    save_visualizations: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metric_objects = [build_eval_metric(name=name) for name in metrics]
    rows: list[dict[str, Any]] = []
    latencies_ms: list[float] = []
    selected_records = records[: len(records) if limit is None else min(len(records), int(limit))]
    total_batches = (len(selected_records) + batch_size - 1) // batch_size

    with tqdm(
        total=len(selected_records),
        desc=f"{checkpoint.name}",
        unit="sample",
        dynamic_ncols=True,
    ) as progress:
        for batch_start in range(0, len(selected_records), batch_size):
            batch_records = selected_records[batch_start : batch_start + batch_size]
            batch_index = (batch_start // batch_size) + 1
            progress.set_postfix_str(f"batch {batch_index}/{total_batches}")
            progress.refresh()
            start = time.perf_counter()
            batch_error: str | None = None
            responses: list[ShaftInferResponse] = []
            try:
                if len(batch_records) == 1:
                    record = batch_records[0]
                    responses = [
                        infer.run(
                            request=ShaftInferRequest(
                                image_path=record.image_path,
                                system_prompt=record.system_prompt,
                                user_prompt=record.user_prompt,
                                messages=record.messages,
                                generation=generation,
                            )
                        )
                    ]
                else:
                    responses = _run_infer_batch(infer, batch_records, generation)
            except Exception as exc:  # noqa: BLE001
                batch_error = str(exc)

            batch_latency_ms = (time.perf_counter() - start) * 1000.0
            per_sample_latency_ms = batch_latency_ms / max(1, len(batch_records))
            progress.update(len(batch_records))

            for offset, record in enumerate(batch_records):
                sample_index = batch_start + offset + 1
                sample_id = record.sample_id or f"sample_{sample_index:06d}"
                target = _parse_target(record.target_text)
                raw_text = ""
                error: str | None = batch_error
                visualization_path: str | None = None
                if batch_error is None:
                    raw_text = str(responses[offset].text)
                    parsed = decode_with_codec(codec, raw_text)
                    if save_visualizations:
                        visualization_path = _render_prediction_visualization(
                            image_path=record.image_path,
                            sample_id=sample_id,
                            sample_index=sample_index,
                            prediction=parsed,
                            out_dir=output_dir,
                        )
                else:
                    parsed = ShaftCodecResult(
                        raw_text="",
                        parsed=None,
                        valid=False,
                        partial=False,
                        error_type="inference_error",
                        error=batch_error,
                    )

                latencies_ms.append(per_sample_latency_ms)
                sample_meta = {
                    "dataset_name": record.dataset_name,
                    "sample_id": sample_id,
                    "index": sample_index,
                    "extra": dict(record.extra),
                }
                for metric in metric_objects:
                    metric.update(prediction=parsed, target=target, sample_meta=sample_meta)

                rows.append(
                    {
                        "sample_id": sample_id,
                        "dataset_name": record.dataset_name,
                        "image_path": record.image_path,
                        "target": _to_jsonable(target),
                        "prediction_raw": raw_text,
                        "prediction_valid": bool(parsed.valid),
                        "prediction_parsed": _to_jsonable(parsed.parsed),
                        "prediction_error": _to_jsonable(parsed.error),
                        "decode_error": _to_jsonable(parsed.error_type),
                        "visualization_path": visualization_path,
                        "error": error,
                        "latency_ms": float(per_sample_latency_ms),
                    }
                )

    summary = {
        "checkpoint": str(checkpoint),
        "checkpoint_kind": inspect_checkpoint_layout(checkpoint).kind,
        "codec": codec,
        "dataset_name": records[0].dataset_name if records else "",
        "num_samples": len(selected_records),
        "num_records_total": len(records),
        "metrics": {name: float(metric_objects[i].compute()) for i, name in enumerate(metrics)},
        "latency_ms_mean": float(sum(latencies_ms) / len(latencies_ms)) if latencies_ms else 0.0,
        "latency_ms_total": float(sum(latencies_ms)),
    }

    _write_jsonl(output_dir / "predictions.jsonl", rows)
    _write_json(output_dir / "summary.json", summary)
    return summary, rows


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    base_config = load_config(args.config)
    codec = _resolve_codec(args.codec)
    metrics = _parse_metric_names(args.metrics)
    input_jsonl = Path(args.input).expanduser().resolve()
    records = _load_records(input_jsonl, str(args.dataset_name))
    if not records:
        raise ValueError(f"No records loaded from {input_jsonl}")

    explicit = [Path(path).expanduser() for path in args.checkpoint]
    checkpoint_root = Path(args.checkpoint_root).expanduser() if args.checkpoint_root else None
    if checkpoint_root is None and not explicit:
        checkpoint_root = Path(base_config.experiment.output_dir).resolve()
    checkpoints = _collect_checkpoints(
        explicit=explicit,
        root=checkpoint_root,
        include_best=bool(args.include_best),
        include_final=bool(args.include_final),
    )
    if not checkpoints:
        raise ValueError("No valid checkpoints found.")

    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else Path(base_config.experiment.output_dir).resolve() / args.default_output_subdir
    )
    generation = _build_generation_config(base_config, args)

    summaries: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    for checkpoint in checkpoints:
        start = time.perf_counter()
        try:
            summary, _ = _evaluate_checkpoint(
                checkpoint=checkpoint,
                base_config=base_config,
                records=records,
                codec=codec,
                metrics=metrics,
                generation=generation,
                output_root=output_root,
                device=args.device,
                batch_size=max(1, int(args.batch_size)),
                limit=args.limit,
                min_pixels=args.min_pixels,
                max_pixels=args.max_pixels,
                save_visualizations=bool(args.save_visualizations),
            )
            summary["runtime_sec"] = float(time.perf_counter() - start)
            summary["status"] = "ok"
            summaries.append(summary)
            print(f"[{args.task_name}] {checkpoint.name} done in {summary['runtime_sec']:.2f}s")
        except Exception as exc:  # noqa: BLE001
            failed.append(
                {
                    "checkpoint": str(checkpoint),
                    "status": "failed",
                    "error": str(exc),
                }
            )
            print(f"[{args.task_name}] {checkpoint.name} failed: {exc}")
            if not args.continue_on_error:
                raise

    aggregate = {
        "task_name": args.task_name,
        "codec": codec,
        "input_jsonl": str(input_jsonl),
        "dataset_name": str(args.dataset_name),
        "metrics": list(metrics),
        "checkpoints": summaries,
        "failed_checkpoints": failed,
        "checkpoint_count": len(summaries),
        "failed_count": len(failed),
    }
    _write_json(output_root / "summary.json", aggregate)
    print(f"[summary] {args.task_name}: wrote {output_root / 'summary.json'}")
    return aggregate
