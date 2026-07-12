from __future__ import annotations

import json
from pathlib import Path

import pytest

from shaft.config import load_config


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


@pytest.mark.parametrize(
    "filename",
    [
        "banana_sft_4b_v5_0_re.yaml",
        "banana_sft_4b_v5_0_re2.yaml",
    ],
)
def test_banana_re_configs_use_bounded_qwen_vision_contract(filename: str) -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "train" / filename)
    batching = config.data.batching

    assert batching.strategy == "bounded_cost_aware"
    assert config.data.media_snapshot_id
    assert batching.buffer_size == 64
    assert batching.max_samples_per_microbatch == 2
    assert batching.max_vision_patches == 16384
    assert config.data.max_pixels is not None
    assert (int(config.data.max_pixels) + 255) // 256 <= 16384
    assert config.data.num_workers == 4
    assert config.train.gradient_accumulation_steps == 4
    assert not hasattr(config.train, "optimizer_batch")
