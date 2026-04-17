from __future__ import annotations

import argparse
from unittest.mock import patch

import pytest

from shaft.cli.common import apply_common_overrides, run_from_args
from shaft.config import RuntimeConfig


def test_apply_common_overrides() -> None:
    cfg = RuntimeConfig()
    args = argparse.Namespace(
        run_id="rid-1",
        seed=7,
        epochs=3,
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
    assert out.train.epochs == 3
    assert out.train.max_steps == 8
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


def _build_min_args(**kwargs):
    base = dict(
        config="dummy.yaml",
        run_id=None,
        seed=None,
        epochs=None,
        max_steps=None,
        learning_rate=None,
        train_batch_size=None,
        eval_batch_size=None,
        mix_strategy=None,
        optimizer_name=None,
        scheduler_name=None,
        scheduler_num_cycles=None,
        scheduler_power=None,
        loss_name=None,
        loss_scale=None,
        finetune_mode=None,
        lora_r=None,
        lora_alpha=None,
        lora_dropout=None,
        qlora_load_in_4bit=None,
        use_cpu=None,
        init_from=None,
        resume_from=None,
        algorithm=None,
    )
    base.update(kwargs)
    return argparse.Namespace(**base)


def test_run_from_args_forced_algorithm() -> None:
    args = _build_min_args()
    cfg = RuntimeConfig()
    with patch("shaft.cli.common.load_config", return_value=cfg):
        with patch("shaft.cli.common.run_sft", return_value={"train_loss": 0.1}) as mocked:
            run_from_args(args, forced_algorithm="sft")
    assert cfg.algorithm.name == "sft"
    mocked.assert_called_once()


def test_run_from_args_allowed_algorithms() -> None:
    args = _build_min_args(algorithm="ppo")
    cfg = RuntimeConfig()
    with patch("shaft.cli.common.load_config", return_value=cfg):
        with patch("shaft.cli.common.run_rlhf", return_value={}) as mocked:
            run_from_args(args, allowed_algorithms={"dpo", "ppo"})
    assert cfg.algorithm.name == "ppo"
    mocked.assert_called_once()


def test_run_from_args_rejects_disallowed_algorithm() -> None:
    args = _build_min_args(algorithm="sft")
    cfg = RuntimeConfig()
    with patch("shaft.cli.common.load_config", return_value=cfg):
        with pytest.raises(ValueError):
            run_from_args(args, allowed_algorithms={"dpo", "ppo"})
