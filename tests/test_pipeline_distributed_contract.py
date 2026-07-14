from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from transformers.integrations.deepspeed import deepspeed_config

from shaft.algorithms.rlhf_utils import build_trl_dpo_config
from shaft.config import load_config
from shaft.pipeline import run_rlhf, run_sft
from shaft.pipeline.training_args import build_hf_training_args
from tests.support.pipeline import FakePipelineTrainer
from tests.support.pipeline import build_fake_model_artifacts
from tests.support.pipeline import write_sft_pipeline_config
from tests.support.rlhf import write_dpo_config


pytestmark = pytest.mark.component


def _deepspeed_config(stage: int = 3) -> dict[str, object]:
    return {
        "bf16": {"enabled": "auto"},
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "train_batch_size": "auto",
        "zero_optimization": {"stage": stage},
    }


def _assert_zero3_is_active() -> None:
    active_config = deepspeed_config()
    assert active_config is not None
    assert active_config["zero_optimization"]["stage"] == 3


def test_sft_pipeline_initializes_deepspeed_before_model_loading(tmp_path: Path) -> None:
    config = write_sft_pipeline_config(tmp_path)
    config.model.model_type = "smoke_vlm"
    config.model.model_name_or_path = "models/Smoke-VLM"
    config.train.distributed.strategy = "deepspeed"
    config.train.distributed.deepspeed.config = _deepspeed_config()

    def _fake_build_model(
        runtime_config,
        *,
        init_from_checkpoint=None,
        sequence_execution_contract=None,
    ):
        assert runtime_config is config
        assert init_from_checkpoint is None
        assert sequence_execution_contract is not None
        _assert_zero3_is_active()
        return build_fake_model_artifacts()

    with patch("shaft.pipeline.sft.build_model_tokenizer_processor", _fake_build_model):
        with patch("shaft.algorithms.sft.ShaftSFTTrainer", FakePipelineTrainer):
            metrics = run_sft(config)

    assert "train_loss" in metrics
    assert FakePipelineTrainer.last_kwargs["args"].deepspeed == config.train.distributed.deepspeed.config


def test_rlhf_pipeline_initializes_deepspeed_before_model_loading(tmp_path: Path) -> None:
    config = load_config(write_dpo_config(tmp_path))
    config.train.distributed.strategy = "deepspeed"
    config.train.distributed.deepspeed.config = _deepspeed_config()

    def _fake_build_model(runtime_config, *, init_from_checkpoint=None):
        assert runtime_config is config
        assert init_from_checkpoint is None
        _assert_zero3_is_active()
        return build_fake_model_artifacts()

    with patch("shaft.pipeline.rlhf.build_model_tokenizer_processor", _fake_build_model):
        with patch("shaft.algorithms.dpo.ShaftDPOTrainer", FakePipelineTrainer):
            metrics = run_rlhf(config)

    assert "train_loss" in metrics
    assert FakePipelineTrainer.last_kwargs["args"].deepspeed == config.train.distributed.deepspeed.config


def test_training_args_expose_active_deepspeed_contract(tmp_path: Path) -> None:
    config = write_sft_pipeline_config(tmp_path)
    config.train.distributed.strategy = "deepspeed"
    config.train.distributed.deepspeed.config = _deepspeed_config(stage=2)

    args = build_hf_training_args(config)

    assert args.deepspeed == config.train.distributed.deepspeed.config
    assert getattr(args, "hf_deepspeed_config", None) is not None
    assert deepspeed_config()["zero_optimization"]["stage"] == 2


def test_non_deepspeed_training_args_clear_global_deepspeed_state(tmp_path: Path) -> None:
    deepspeed_dir = tmp_path / "deepspeed"
    deepspeed_dir.mkdir()
    deepspeed_runtime = write_sft_pipeline_config(deepspeed_dir)
    deepspeed_runtime.train.distributed.strategy = "deepspeed"
    deepspeed_runtime.train.distributed.deepspeed.config = _deepspeed_config(stage=2)
    _ = build_hf_training_args(deepspeed_runtime)
    assert deepspeed_config()["zero_optimization"]["stage"] == 2

    ddp_dir = tmp_path / "ddp"
    ddp_dir.mkdir()
    ddp_runtime = write_sft_pipeline_config(ddp_dir)
    ddp_args = build_hf_training_args(ddp_runtime)

    assert ddp_args.deepspeed is None
    assert deepspeed_config() is None


def test_dpo_trl_config_preserves_deepspeed_contract(tmp_path: Path) -> None:
    config = load_config(write_dpo_config(tmp_path))
    config.train.distributed.strategy = "deepspeed"
    config.train.distributed.deepspeed.config = _deepspeed_config(stage=2)

    train_args = build_hf_training_args(config)
    dpo_args = build_trl_dpo_config(train_args=train_args, rlhf_config=config.rlhf.dpo)

    assert dpo_args.deepspeed == config.train.distributed.deepspeed.config
    assert getattr(dpo_args, "hf_deepspeed_config", None) is not None
