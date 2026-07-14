from __future__ import annotations

from unittest.mock import patch

import pytest

from shaft.cli.common import apply_common_overrides, run_from_args
from shaft.config import DatasetSourceConfig, RuntimeConfig
from tests.support.cli import build_common_train_args


pytestmark = pytest.mark.component


def _valid_runtime_config() -> RuntimeConfig:
    config = RuntimeConfig()
    config.data.datasets = [
        DatasetSourceConfig(
            dataset_name="fixture",
            train_paths=["train.jsonl"],
            val_paths=["val.jsonl"],
        )
    ]
    return config


def _enable_bounded_batching(config: RuntimeConfig) -> None:
    config.data.media_snapshot_id = "cli-fixture-v1"
    config.data.batching.grouping = "bounded_cost"
    config.data.batching.max_tokens_per_microbatch = 1024


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
    assert out.data.schedule.mixing == "concat"
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
    cfg = _valid_runtime_config()
    with patch("shaft.cli.common.load_config", return_value=cfg):
        with patch("shaft.cli.common.run_sft", return_value={"train_loss": 0.1}) as mocked:
            run_from_args(args, forced_algorithm="sft")
    assert cfg.algorithm.name == "sft"
    mocked.assert_called_once()


def test_run_from_args_resolves_one_run_id_for_logging_and_pipeline() -> None:
    args = build_common_train_args()
    cfg = _valid_runtime_config()
    cfg.experiment.name = "canonical-run"
    cfg.experiment.run_id = None
    with patch("shaft.cli.common.load_config", return_value=cfg):
        with patch("shaft.cli.common.configure_logging") as mocked_logging:
            with patch("shaft.cli.common.run_sft", return_value={}):
                run_from_args(args, forced_algorithm="sft")

    assert cfg.experiment.run_id == "canonical-run"
    mocked_logging.assert_called_once_with(
        cfg.logging,
        run_id="canonical-run",
    )


def test_run_from_args_allowed_algorithms() -> None:
    args = build_common_train_args(algorithm="ppo")
    cfg = _valid_runtime_config()
    cfg.data.datasets[0].source_type = "jsonl_ppo"
    with patch("shaft.cli.common.load_config", return_value=cfg):
        with patch("shaft.cli.common.run_rlhf", return_value={}) as mocked:
            run_from_args(args, allowed_algorithms={"dpo", "ppo", "grpo"})
    assert cfg.algorithm.name == "ppo"
    mocked.assert_called_once()


def test_run_from_args_supports_grpo() -> None:
    args = build_common_train_args(algorithm="grpo")
    cfg = _valid_runtime_config()
    with patch("shaft.cli.common.load_config", return_value=cfg):
        with patch("shaft.cli.common.run_rlhf", return_value={}) as mocked:
            run_from_args(args, allowed_algorithms={"dpo", "ppo", "grpo"})
    assert cfg.algorithm.name == "grpo"
    mocked.assert_called_once()


def test_run_from_args_rejects_disallowed_algorithm() -> None:
    args = build_common_train_args(algorithm="sft")
    cfg = _valid_runtime_config()
    with patch("shaft.cli.common.load_config", return_value=cfg):
        with pytest.raises(ValueError):
            run_from_args(args, allowed_algorithms={"dpo", "ppo", "grpo"})


def test_run_from_args_revalidates_bounded_duration_override() -> None:
    args = build_common_train_args(epochs=1)
    config = _valid_runtime_config()
    _enable_bounded_batching(config)

    with patch("shaft.cli.common.load_config", return_value=config):
        with patch("shaft.cli.common.run_sft") as run_sft:
            with pytest.raises(ValueError, match="requires train.duration.unit='steps'"):
                run_from_args(args, forced_algorithm="sft")

    run_sft.assert_not_called()


def test_run_from_args_revalidates_schedule_mixing_override() -> None:
    args = build_common_train_args(mix_strategy="weighted")
    config = _valid_runtime_config()
    _enable_bounded_batching(config)
    config.data.schedule.mixing = "concat"
    config.data.schedule.shuffle = False

    with patch("shaft.cli.common.load_config", return_value=config):
        with patch("shaft.cli.common.run_sft") as run_sft:
            with pytest.raises(ValueError, match="requires data.schedule.shuffle=true"):
                run_from_args(args, forced_algorithm="sft")

    run_sft.assert_not_called()
