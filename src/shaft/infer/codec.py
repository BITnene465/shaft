from __future__ import annotations

import json
import re
from typing import Any

from shaft.plugins import Registry

CODEC_REGISTRY: Registry = Registry("infer_codec")


def register_codec(name: str):
    return CODEC_REGISTRY.register(name)


def _strip_code_fence(text: str) -> str:
    value = str(text).strip()
    if not value.startswith("```"):
        return value
    lines = value.splitlines()
    if not lines:
        return value
    if lines[0].startswith("```"):
        lines = lines[1:]
    while lines and lines[-1].strip().startswith("```"):
        lines.pop()
    return "\n".join(lines).strip()


def _extract_primary_json_fragment(text: str) -> str:
    value = _strip_code_fence(text)
    first_obj = value.find("{")
    first_arr = value.find("[")
    starts = [idx for idx in (first_obj, first_arr) if idx >= 0]
    if not starts:
        return value.strip()
    return value[min(starts) :].strip()


def _close_open_structures(fragment: str) -> str:
    value = str(fragment).strip()
    if not value:
        return value
    stack: list[str] = []
    out: list[str] = []
    in_string = False
    escape = False
    for ch in value:
        if in_string:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            out.append(ch)
            continue
        if ch in "{[":
            stack.append(ch)
            out.append(ch)
            continue
        if ch in "}]":
            if stack and ((stack[-1] == "{" and ch == "}") or (stack[-1] == "[" and ch == "]")):
                stack.pop()
                out.append(ch)
            # Unmatched closing bracket is ignored during salvage.
            continue
        out.append(ch)

    if in_string:
        if escape:
            out.append("\\")
        out.append('"')
    while stack:
        opener = stack.pop()
        out.append("}" if opener == "{" else "]")
    repaired = "".join(out).strip()
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    return repaired


def _extract_safe_prefix(fragment: str) -> str:
    value = str(fragment).strip()
    if not value:
        return value
    stack: list[str] = []
    in_string = False
    escape = False
    safe_end = 0
    for idx, ch in enumerate(value):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue
        if ch in "{[":
            stack.append(ch)
            continue
        if ch in "}]":
            if stack and ((stack[-1] == "{" and ch == "}") or (stack[-1] == "[" and ch == "]")):
                stack.pop()
                safe_end = idx + 1
            continue
        if ch == "," and stack:
            safe_end = idx + 1
    if safe_end <= 0:
        return value
    return value[:safe_end].strip()


def _try_loads_or_raw_decode(text: str) -> Any:
    value = str(text).strip()
    if not value:
        raise ValueError("Empty text cannot be parsed as JSON.")
    try:
        return json.loads(value)
    except Exception:
        decoder = json.JSONDecoder()
        obj, _end = decoder.raw_decode(value)
        return obj


def _normalize_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _normalize_json_value(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, list):
        return [_normalize_json_value(v) for v in value]
    return value


def _decode_json_lenient(raw_text: str) -> Any:
    primary = _extract_primary_json_fragment(raw_text)
    attempts: list[str] = []

    def _push(candidate: str) -> None:
        text = str(candidate).strip()
        if text and text not in attempts:
            attempts.append(text)

    _push(primary)
    _push(_close_open_structures(primary))
    safe_prefix = _extract_safe_prefix(primary)
    _push(safe_prefix)
    _push(_close_open_structures(safe_prefix))
    if primary.startswith("{"):
        _push("{}")
    if primary.startswith("["):
        _push("[]")

    last_error: Exception | None = None
    for candidate in attempts:
        try:
            parsed = _try_loads_or_raw_decode(candidate)
            return _normalize_json_value(parsed)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue
    raise ValueError(f"Failed to decode JSON from model output. last_error={last_error}")


@register_codec("text")
def codec_text(raw_text: str) -> str:
    return str(raw_text)


@register_codec("json_any")
def codec_json_any(raw_text: str) -> Any:
    return _decode_json_lenient(raw_text)


@register_codec("json_object")
def codec_json_object(raw_text: str) -> dict[str, Any]:
    parsed = codec_json_any(raw_text)
    if not isinstance(parsed, dict):
        raise TypeError(f"codec=json_object expects JSON object, got {type(parsed).__name__}.")
    return parsed


@register_codec("json_list")
def codec_json_list(raw_text: str) -> list[Any]:
    parsed = codec_json_any(raw_text)
    if not isinstance(parsed, list):
        raise TypeError(f"codec=json_list expects JSON list, got {type(parsed).__name__}.")
    return parsed


def decode_with_codec(codec: str, raw_text: str) -> Any:
    codec_name = str(codec).strip().lower()
    decoder = CODEC_REGISTRY.get(codec_name)
    return decoder(raw_text)
