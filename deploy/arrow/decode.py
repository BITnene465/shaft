from __future__ import annotations

import json
import re
from typing import Any

from .config import ArrowProtocolSpec, load_arrow_config

_JSON_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_DEFAULT_PROTOCOL = load_arrow_config().protocol

__all__ = [
    "decode_stage1_output",
    "decode_stage2_output",
]


def decode_stage1_output(
    text: str,
    *,
    image_width: int,
    image_height: int,
    strict: bool = False,
    protocol: ArrowProtocolSpec | None = None,
) -> dict[str, Any]:
    protocol = protocol or _DEFAULT_PROTOCOL
    payload, _recovered_prefix = _parse_json_payload(text, strict=strict)
    if isinstance(payload, dict):
        if strict:
            raise ValueError("Stage1 decoded payload must be a JSON array.")
        payload = [payload]
    if not isinstance(payload, list):
        raise ValueError("Stage1 decoded payload must be a JSON array or object.")

    instances: list[dict[str, Any]] = []
    for item_index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Stage1 item[{item_index}] must be a JSON object.")
        label = item.get("label")
        if label not in protocol.labels:
            raise ValueError(f"Stage1 item[{item_index}] label must be one of {list(protocol.labels)}.")
        bbox_values = item.get("bbox_2d")
        if not isinstance(bbox_values, list) or len(bbox_values) != 4:
            raise ValueError(f"Stage1 item[{item_index}] bbox_2d must contain 4 values.")
        bbox = [
            _dequantize(_parse_coord(bbox_values[0], "x", protocol.num_bins, strict=strict), image_width, protocol.num_bins),
            _dequantize(_parse_coord(bbox_values[1], "y", protocol.num_bins, strict=strict), image_height, protocol.num_bins),
            _dequantize(_parse_coord(bbox_values[2], "x", protocol.num_bins, strict=strict), image_width, protocol.num_bins),
            _dequantize(_parse_coord(bbox_values[3], "y", protocol.num_bins, strict=strict), image_height, protocol.num_bins),
        ]
        if strict and (bbox[0] >= bbox[2] or bbox[1] >= bbox[3]):
            raise ValueError(f"Stage1 item[{item_index}] bbox must satisfy x1 < x2 and y1 < y2.")
        instances.append({"label": str(label), "bbox": bbox})

    return {"instances": instances}


def decode_stage2_output(
    text: str,
    *,
    image_width: int,
    image_height: int,
    strict: bool = False,
    protocol: ArrowProtocolSpec | None = None,
) -> dict[str, Any]:
    protocol = protocol or _DEFAULT_PROTOCOL
    payload, _recovered_prefix = _parse_json_payload(text, strict=strict)
    if isinstance(payload, dict):
        payload = payload.get("keypoints_2d")
    if not isinstance(payload, list):
        raise ValueError("Stage2 decoded payload must be an object with keypoints_2d or a JSON array of points.")

    keypoints_2d: list[list[int]] = []
    keypoints: list[list[float]] = []
    for point_index, raw_point in enumerate(payload):
        if not isinstance(raw_point, list) or len(raw_point) != 2:
            raise ValueError(f"Stage2 point[{point_index}] must be [x, y].")
        x_value = _parse_coord(raw_point[0], "x", protocol.num_bins, strict=strict)
        y_value = _parse_coord(raw_point[1], "y", protocol.num_bins, strict=strict)
        keypoints_2d.append([x_value, y_value])
        keypoints.append([
            _dequantize(x_value, image_width, protocol.num_bins),
            _dequantize(y_value, image_height, protocol.num_bins),
        ])

    if len(keypoints_2d) < 2:
        raise ValueError("Stage2 keypoint list must contain at least 2 points.")
    return {"keypoints": keypoints, "keypoints_2d": keypoints_2d}


def _parse_coord(value: Any, axis: str, num_bins: int, *, strict: bool = False) -> int:
    if strict:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"Expected integer {axis} coordinate, got {value!r}.")
        parsed = int(value)
    else:
        try:
            parsed = int(round(float(value)))
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Expected numeric {axis} coordinate, got {value!r}.") from exc
    if parsed < 0 or parsed >= num_bins:
        raise ValueError(f"{axis} coordinate {parsed} out of range [0, {num_bins - 1}].")
    return parsed


def _dequantize(value: int, size: int, num_bins: int) -> float:
    size = max(int(size), 1)
    if size == 1:
        return 0.0
    return float(value) / float(num_bins - 1) * float(size - 1)


def _extract_balanced_json(text: str) -> str | None:
    for opener, closer in (("[", "]"), ("{", "}")):
        payload = _extract_balanced_json_with_delimiters(text, opener, closer)
        if payload is not None:
            return payload
    return None


def _extract_balanced_json_with_delimiters(text: str, opener: str, closer: str) -> str | None:
    start = text.find(opener)
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _recover_truncated_json_array(text: str) -> str | None:
    start = text.find("[")
    if start < 0:
        return None

    items: list[Any] = []
    in_string = False
    escape = False
    depth = 1
    item_start = start + 1

    for index in range(start + 1, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char in "[{":
            depth += 1
            continue
        if char in "]}":
            depth -= 1
            if depth == 0:
                item_text = text[item_start:index].strip()
                if item_text:
                    try:
                        items.append(json.loads(item_text))
                    except json.JSONDecodeError:
                        pass
                return json.dumps(items, ensure_ascii=False, separators=(",", ":"))
            continue
        if char == "," and depth == 1:
            item_text = text[item_start:index].strip()
            if item_text:
                try:
                    items.append(json.loads(item_text))
                except json.JSONDecodeError:
                    pass
            item_start = index + 1

    tail_text = text[item_start:].strip()
    if tail_text:
        try:
            items.append(json.loads(tail_text))
        except json.JSONDecodeError:
            pass
    if not items:
        return None
    return json.dumps(items, ensure_ascii=False, separators=(",", ":"))


def _parse_json_payload(text: str, *, strict: bool = False) -> tuple[Any, bool]:
    stripped = text.strip()
    if not stripped:
        raise ValueError("Decoded text is empty.")
    if strict:
        try:
            return json.loads(stripped), False
        except json.JSONDecodeError as exc:
            raise ValueError(f"Strict JSON payload must occupy the entire decoded text: {exc.msg}.") from exc
    fenced = _JSON_FENCE_PATTERN.search(stripped)
    if fenced is not None:
        stripped = fenced.group(1).strip()
    payload_text = _extract_balanced_json(stripped)
    recovered_prefix = False
    if payload_text is None:
        payload_text = _recover_truncated_json_array(stripped)
        recovered_prefix = payload_text is not None
    if payload_text is None:
        raise ValueError("No JSON payload found in decoded text.")
    try:
        return json.loads(payload_text), recovered_prefix
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON payload: {exc.msg}.") from exc
