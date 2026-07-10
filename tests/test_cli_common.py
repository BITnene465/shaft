from __future__ import annotations

from unittest.mock import patch

import pytest

from shaft.cli.common import apply_common_overrides, run_from_args
from shaft.config import RuntimeConfig
from tests.support.cli import build_common_train_args


pytestmark = pytest.mark.component


def test_apply_common_overrides() -> None:
    cfg = RuntimeConfig()
    args = build_common_train_args(
        run_id="rid-1",
        seed=7,
        max_steps=8,
        learning_rate=2.5e-5,
        train_batch_size=2,
        eval_batch_size=4,
        mix_strategy="concat",
        optimizer_name="adamw_torch",
        scheduler_name="cosine_with_restarts",
        scheduler_num_cycles=2.0,
        scheduler_power=1.5,
        loss_name="auto",
        loss_scale="all",
        finetune_mode="lora",
        lora_r=24,
        lora_alpha=48,
        lora_dropout=0.15,
        qlora_load_in_4bit=False,
        use_cpu=True,
        init_from="init-ckpt-a",
        resume_from="ckpt-a",
    )
    out = apply_common_overrides(cfg, args)
    assert out.experiment.run_id == "rid-1"
    assert out.experiment.seed == 7
    assert out.train.duration.unit == "steps"
    assert out.train.duration.value == 8
    assert out.train.learning_rate == pytest.approx(2.5e-5)
    assert out.train.per_device_train_batch_size == 2
    assert out.eval.per_device_eval_batch_size == 4
    assert out.data.mix_strategy == "concat"
    assert out.train.optimizer_name == "adamw_torch"
    assert out.train.scheduler_name == "cosine_with_restarts"
    assert out.train.scheduler_num_cycles == pytest.approx(2.0)
    assert out.train.scheduler_power == pytest.approx(1.5)
    assert out.train.loss_name == "auto"
    assert out.train.loss_scale == "all"
    assert out.model.finetune.mode == "lora"
    assert out.model.finetune.lora_r == 24
    assert out.model.finetune.lora_alpha == 48
    assert out.model.finetune.lora_dropout == pytest.approx(0.15)
    assert out.model.finetune.qlora_load_in_4bit is False
    assert out.train.use_cpu is True
    assert out.train.init_from_checkpoint == "init-ckpt-a"
    assert out.train.resume_from_checkpoint == "ckpt-a"


def test_apply_common_overrides_rejects_two_duration_units() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        apply_common_overrides(
            RuntimeConfig(),
            build_common_train_args(epochs=2, max_steps=10),
        )


def test_apply_common_overrides_rejects_non_positive_steps() -> None:
    with pytest.raises(ValueError, match="--max-steps must be > 0"):
        apply_common_overrides(RuntimeConfig(), build_common_train_args(max_steps=0))


def test_run_from_args_forced_algorithm() -> None:
    args = build_common_train_args()
    cfg = RuntimeConfig()
    with patch("shaft.cli.common.load_config", return_value=cfg):
        with patch("shaft.cli.common.run_sft", return_value={"train_loss": 0.1}) as mocked:
            run_from_args(args, forced_algorithm="sft")
    assert cfg.algorithm.name == "sft"
    mocked.assert_called_once()


def test_run_from_args_allowed_algorithms() -> None:
    args = build_common_train_args(algorithm="ppo")
    cfg = RuntimeConfig()
    with patch("shaft.cli.common.load_config", return_value=cfg):
        with patch("shaft.cli.common.run_rlhf", return_value={}) as mocked:
            run_from_args(args, allowed_algorithms={"dpo", "ppo", "grpo"})
    assert cfg.algorithm.name == "ppo"
    mocked.assert_called_once()


def test_run_from_args_supports_grpo() -> None:
    args = build_common_train_args(algorithm="grpo")
    cfg = RuntimeConfig()
    with patch("shaft.cli.common.load_config", return_value=cfg):
        with patch("shaft.cli.common.run_rlhf", return_value={}) as mocked:
            run_from_args(args, allowed_algorithms={"dpo", "ppo", "grpo"})
    assert cfg.algorithm.name == "grpo"
    mocked.assert_called_once()


def test_run_from_args_rejects_disallowed_algorithm() -> None:
    args = build_common_train_args(algorithm="sft")
    cfg = RuntimeConfig()
    with patch("shaft.cli.common.load_config", return_value=cfg):
        with pytest.raises(ValueError):
            run_from_args(args, allowed_algorithms={"dpo", "ppo", "grpo"})
