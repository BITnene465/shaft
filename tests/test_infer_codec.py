from __future__ import annotations

import pytest

from shaft.infer import decode_with_codec


def test_codec_text_passthrough() -> None:
    raw = "hello"
    assert decode_with_codec("text", raw) == "hello"


def test_codec_json_object_with_markdown_fence() -> None:
    raw = """```json
{"a": 1, "b": true}
```"""
    parsed = decode_with_codec("json_object", raw)
    assert parsed["a"] == 1
    assert parsed["b"] is True


def test_codec_json_list_type_mismatch_raises() -> None:
    with pytest.raises(TypeError, match="json_list"):
        decode_with_codec("json_list", '{"a":1}')


def test_codec_json_any_extracts_json_from_prefixed_text() -> None:
    raw = 'Model output: {"ok": true, "score": 0.9} trailing words'
    parsed = decode_with_codec("json_any", raw)
    assert parsed["ok"] is True
    assert parsed["score"] == pytest.approx(0.9)


def test_codec_json_any_salvages_truncated_list() -> None:
    raw = '[{"a":1},{"b":2'
    parsed = decode_with_codec("json_any", raw)
    assert isinstance(parsed, list)
    assert parsed[0]["a"] == 1
    assert parsed[1]["b"] == 2
