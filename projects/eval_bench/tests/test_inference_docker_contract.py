from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_contract_smoke_module():
    script = Path(__file__).resolve().parents[3] / "docker" / "inference" / "contract-smoke.py"
    spec = importlib.util.spec_from_file_location("shaft_inference_contract_smoke", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_inference_contract_smoke_uses_shared_json_codec() -> None:
    module = _load_contract_smoke_module()

    contract = module._codec_contract('[{"label": "arrow", "bbox": [1, 2, 3, 4]}')

    assert contract["codec"] == "json_any"
    assert contract["valid"] is True
    assert contract["partial"] is True
    assert contract["parsed"] == [{"label": "arrow", "bbox": [1, 2, 3, 4]}]


def test_inference_contract_smoke_reports_decode_errors() -> None:
    module = _load_contract_smoke_module()

    contract = module._codec_contract("not json")

    assert contract["codec"] == "json_any"
    assert contract["valid"] is False
    assert contract["parsed"] is None
    assert contract["error_type"] == "json_decode_error"
