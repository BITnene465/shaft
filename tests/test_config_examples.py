from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from shaft.config import load_config, load_config_from_text, to_resolved_payload


pytestmark = pytest.mark.contract


def test_all_train_configs_load_with_explicit_batch_axes() -> None:
    root = Path(__file__).resolve().parents[1]
    train_config_dir = root / "configs" / "train"
    config_paths = sorted(train_config_dir.glob("*.yaml"))

    assert config_paths
    for config_path in config_paths:
        raw_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        raw_batching = raw_payload["data"]["batching"]
        assert "grouping" in raw_batching
        assert "cardinality" in raw_batching
        assert "layout" in raw_batching
        assert isinstance(raw_batching.get("packing"), dict)
        assert "mode" in raw_batching["packing"]

        config = load_config(config_path)
        assert config.data.batching.grouping
        assert config.data.batching.cardinality
        assert config.data.batching.packing.mode
        assert config.data.batching.layout
        resolved_payload = to_resolved_payload(config)
        reloaded = load_config_from_text(
            yaml.safe_dump(resolved_payload, sort_keys=False),
            config_path=config_path,
        )
        assert to_resolved_payload(reloaded) == resolved_payload


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
        "banana_sft_4b_v5_1.yaml",
    ],
)
def test_banana_re_configs_use_bounded_qwen_vision_contract(filename: str) -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "train" / filename)
    batching = config.data.batching

    assert batching.grouping == "bounded_cost"
    assert batching.cardinality == "token_budget"
    assert batching.packing.mode == "none"
    assert batching.layout == "padded"
    assert config.data.media_snapshot_id
    assert batching.buffer_size == 64
    assert batching.max_tokens_per_microbatch == 10000
    assert batching.resource_budgets == {"vision_patches": 16384}
    assert config.data.max_pixels is not None
    assert (int(config.data.max_pixels) + 255) // 256 <= 16384
    assert config.data.num_workers == 4
    assert config.train.per_device_train_batch_size == 2
    assert config.train.gradient_accumulation_steps == 4
    assert not hasattr(config.train, "optimizer_batch")
