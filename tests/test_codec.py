from __future__ import annotations

import pytest

from shaft.codec import ShaftCodecResult, decode_with_codec


def test_codec_text_passthrough() -> None:
    raw = "hello"
    decoded = decode_with_codec("text", raw)
    assert isinstance(decoded, ShaftCodecResult)
    assert decoded.valid is True
    assert decoded.partial is False
    assert decoded.parsed == "hello"


def test_codec_json_object_with_markdown_fence() -> None:
    raw = """```json
{"a": 1, "b": true}
```"""
    decoded = decode_with_codec("json_object", raw)
    assert decoded.valid is True
    assert decoded.partial is False
    assert decoded.parsed["a"] == 1
    assert decoded.parsed["b"] is True


def test_codec_json_list_type_mismatch_reports_invalid() -> None:
    decoded = decode_with_codec("json_list", '{"a":1}')
    assert decoded.valid is False
    assert decoded.error_type == "json_type_error"


def test_codec_json_any_extracts_json_from_prefixed_text() -> None:
    raw = 'Model output: {"ok": true, "score": 0.9} trailing words'
    decoded = decode_with_codec("json_any", raw)
    assert decoded.valid is True
    assert decoded.partial is False
    assert decoded.parsed["ok"] is True
    assert decoded.parsed["score"] == pytest.approx(0.9)


def test_codec_json_any_salvages_truncated_list_as_partial() -> None:
    raw = '[{"a":1},{"b":2'
    decoded = decode_with_codec("json_any", raw)
    assert decoded.valid is True
    assert decoded.partial is True
    assert isinstance(decoded.parsed, list)
    assert decoded.parsed[0]["a"] == 1
    assert decoded.parsed[1]["b"] == 2
