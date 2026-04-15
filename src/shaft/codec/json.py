from __future__ import annotations

import json
import re
from typing import Any

from .base import ShaftCodecResult
from .registry import register_codec


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


def _decode_json_lenient(raw_text: str) -> ShaftCodecResult:
    primary = _extract_primary_json_fragment(raw_text)
    attempts: list[tuple[str, bool]] = []

    def _push(candidate: str, *, partial: bool) -> None:
        text = str(candidate).strip()
        if text and all(existing != text for existing, _ in attempts):
            attempts.append((text, partial))

    if not primary:
        return ShaftCodecResult(
            raw_text=str(raw_text),
            parsed=None,
            valid=False,
            partial=False,
            error_type="json_decode_error",
            error="Empty text cannot be parsed as JSON.",
        )

    _push(primary, partial=False)
    _push(_close_open_structures(primary), partial=True)
    safe_prefix = _extract_safe_prefix(primary)
    _push(safe_prefix, partial=True)
    _push(_close_open_structures(safe_prefix), partial=True)
    if primary.startswith("{"):
        _push("{}", partial=True)
    if primary.startswith("["):
        _push("[]", partial=True)

    last_error: Exception | None = None
    for candidate, partial in attempts:
        try:
            parsed = _try_loads_or_raw_decode(candidate)
            return ShaftCodecResult(
                raw_text=str(raw_text),
                parsed=_normalize_json_value(parsed),
                valid=True,
                partial=partial,
                error_type=None,
                error=None,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue
    return ShaftCodecResult(
        raw_text=str(raw_text),
        parsed=None,
        valid=False,
        partial=False,
        error_type="json_decode_error",
        error=f"Failed to decode JSON from model output. last_error={last_error}",
    )


@register_codec("text")
def codec_text(raw_text: str) -> ShaftCodecResult:
    return ShaftCodecResult(
        raw_text=str(raw_text),
        parsed=str(raw_text).strip(),
        valid=True,
        partial=False,
        error_type=None,
        error=None,
    )


@register_codec("json_any")
def codec_json_any(raw_text: str) -> ShaftCodecResult:
    return _decode_json_lenient(raw_text)


@register_codec("json_object")
def codec_json_object(raw_text: str) -> ShaftCodecResult:
    decoded = codec_json_any(raw_text)
    if not decoded.valid:
        return decoded
    if not isinstance(decoded.parsed, dict):
        return ShaftCodecResult(
            raw_text=decoded.raw_text,
            parsed=None,
            valid=False,
            partial=decoded.partial,
            error_type="json_type_error",
            error=f"codec=json_object expects JSON object, got {type(decoded.parsed).__name__}.",
        )
    return decoded


@register_codec("json_list")
def codec_json_list(raw_text: str) -> ShaftCodecResult:
    decoded = codec_json_any(raw_text)
    if not decoded.valid:
        return decoded
    if not isinstance(decoded.parsed, list):
        return ShaftCodecResult(
            raw_text=decoded.raw_text,
            parsed=None,
            valid=False,
            partial=decoded.partial,
            error_type="json_type_error",
            error=f"codec=json_list expects JSON list, got {type(decoded.parsed).__name__}.",
        )
    return decoded
