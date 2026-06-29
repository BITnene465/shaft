from __future__ import annotations

import json
from pathlib import Path

import pytest


pytestmark = pytest.mark.contract


def test_deepspeed_example_configs_cover_zero_stages() -> None:
    root = Path(__file__).resolve().parents[1]
    expected = {
        "zero1_bf16.json": 1,
        "zero2_bf16.json": 2,
        "zero3_bf16.json": 3,
    }
    for filename, stage in expected.items():
        payload = json.loads((root / "configs" / "deepspeed" / filename).read_text(encoding="utf-8"))
        assert payload["zero_optimization"]["stage"] == stage
        assert payload["bf16"]["enabled"] == "auto"
        assert payload["train_micro_batch_size_per_gpu"] == "auto"
        assert payload["train_batch_size"] == "auto"
