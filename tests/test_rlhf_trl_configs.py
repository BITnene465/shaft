from __future__ import annotations

from pathlib import Path

import pytest

from shaft.algorithms.rlhf_utils import build_trl_dpo_config, build_trl_grpo_config
from shaft.config import load_config
from shaft.pipeline.training_args import build_hf_training_args
from tests.support.pipeline import fsdp_enabled as _fsdp_enabled
from tests.support.pipeline import fsdp_option_values as _fsdp_option_values
from tests.support.rlhf import write_dpo_config as _write_dpo_config
from tests.support.rlhf import write_grpo_config as _write_grpo_config


pytestmark = pytest.mark.component


def test_dpo_trl_config_preserves_deepspeed_args(tmp_path: Path) -> None:
    cfg = load_config(_write_dpo_config(tmp_path))
    cfg.train.distributed.strategy = "deepspeed"
    cfg.train.distributed.deepspeed.config = {
        "bf16": {"enabled": "auto"},
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "train_batch_size": "auto",
        "zero_optimization": {"stage": 2},
    }

    train_args = build_hf_training_args(cfg)
    dpo_args = build_trl_dpo_config(train_args=train_args, rlhf_config=cfg.rlhf.dpo)

    assert dpo_args.deepspeed == cfg.train.distributed.deepspeed.config
    assert getattr(dpo_args, "hf_deepspeed_config", None) is not None


def test_grpo_trl_config_preserves_fsdp_args(tmp_path: Path) -> None:
    cfg = load_config(_write_grpo_config(tmp_path))
    cfg.train.distributed.strategy = "fsdp"
    cfg.train.distributed.fsdp.auto_wrap_policy = "none"

    train_args = build_hf_training_args(cfg)
    grpo_args = build_trl_grpo_config(train_args=train_args, rlhf_config=cfg.rlhf.grpo)

    assert _fsdp_enabled(grpo_args.fsdp) is True
    option_values = _fsdp_option_values(grpo_args.fsdp)
    if option_values:
        assert option_values == ["full_shard"]
    assert grpo_args.fsdp_config["activation_checkpointing"] is True
