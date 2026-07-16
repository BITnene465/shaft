from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from shaft.codec import ShaftCodecResult, decode_with_codec


def _load_inference_contract_smoke_module():
    script = Path(__file__).resolve().parents[1] / "docker" / "inference" / "contract-smoke.py"
    spec = importlib.util.spec_from_file_location("shaft_inference_contract_smoke", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_inference_contract_smoke_uses_shared_json_codec() -> None:
    module = _load_inference_contract_smoke_module()

    contract = module._codec_contract('[{"label": "arrow", "bbox": [1, 2, 3, 4]}')

    assert contract["codec"] == "json_any"
    assert contract["valid"] is True
    assert contract["partial"] is True
    assert contract["parsed"] == [{"label": "arrow", "bbox": [1, 2, 3, 4]}]


def test_inference_contract_smoke_reports_decode_errors() -> None:
    module = _load_inference_contract_smoke_module()

    contract = module._codec_contract("not json")

    assert contract["codec"] == "json_any"
    assert contract["valid"] is False
    assert contract["parsed"] is None
    assert contract["error_type"] == "json_decode_error"
