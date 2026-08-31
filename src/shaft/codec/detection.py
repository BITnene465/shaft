from __future__ import annotations

from collections.abc import Collection
import json
from typing import Any

from .base import ShaftCodecResult
from .registry import register_codec


def decode_qwen_bbox_2d_list(
    raw_text: str,
    *,
    allowed_labels: Collection[str] | None = None,
) -> ShaftCodecResult:
    raw = str(raw_text)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return ShaftCodecResult(raw, None, False, False, "json_decode_error", str(exc))
    if not isinstance(payload, list):
        return ShaftCodecResult(
            raw, None, False, False, "json_type_error", "Expected one JSON list."
        )
    labels = None if allowed_labels is None else frozenset(str(label) for label in allowed_labels)
    for index, item in enumerate(payload):
        if not isinstance(item, dict) or set(item) != {"bbox_2d", "label"}:
            return ShaftCodecResult(
                raw,
                None,
                False,
                False,
                "schema_error",
                f"Item {index} must contain exactly bbox_2d and label.",
            )
        bbox: Any = item["bbox_2d"]
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or any(type(value) is not int or value < 0 or value > 999 for value in bbox)
        ):
            return ShaftCodecResult(
                raw,
                None,
                False,
                False,
                "schema_error",
                f"Item {index} bbox_2d must contain four integers in 0..999.",
            )
        if bbox[2] < bbox[0] or bbox[3] < bbox[1]:
            return ShaftCodecResult(
                raw, None, False, False, "schema_error", f"Item {index} bbox_2d is inverted."
            )
        label = item["label"]
        if not isinstance(label, str) or not label:
            return ShaftCodecResult(
                raw, None, False, False, "schema_error", f"Item {index} label is invalid."
            )
        if labels is not None and label not in labels:
            return ShaftCodecResult(
                raw,
                None,
                False,
                False,
                "schema_error",
                f"Item {index} label {label!r} is not allowed.",
            )
    return ShaftCodecResult(raw, payload, True, False, None, None)


@register_codec("qwen_bbox_2d_list")
def codec_qwen_bbox_2d_list(raw_text: str) -> ShaftCodecResult:
    return decode_qwen_bbox_2d_list(raw_text)
