from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from safetensors.torch import load_file
import torch

from shaft.algorithms.grpo_rewards import GRPO_REWARD_REGISTRY
from shaft.config import EvalDatasetPolicyConfig, load_config
from shaft.pipeline import run_rlhf
from shaft.training import (
    ShaftDPOTrainer,
    ShaftGRPOTrainer,
    ShaftPPOTrainer,
    load_batching_run_metadata,
    load_checkpoint_batching_metadata,
)
from shaft.training.checkpointing import (
    ShaftCheckpointProtocol,
    resolve_resume_checkpoint,
    validate_training_checkpoint_commit,
)
from tests.support.pipeline import FakePipelineTrainer as _FakeTrainer
from tests.support.rlhf import write_dpo_config as _write_dpo_config
from tests.support.rlhf import write_grpo_config as _write_grpo_config
from tests.support.rlhf import write_ppo_config as _write_ppo_config


pytestmark = [pytest.mark.component, pytest.mark.smoke]


_GRPO_SMOKE_REWARD_NAME = "test_smoke_generation_slot"


def _grpo_smoke_generation_slot_reward(*, prediction_texts, **kwargs):
    """Give every second completion a reward so the smoke model must update."""

    _ = kwargs
    return [float(index % 2) for index, _ in enumerate(prediction_texts)]


if not GRPO_REWARD_REGISTRY.has(_GRPO_SMOKE_REWARD_NAME):
    GRPO_REWARD_REGISTRY.register(
        _GRPO_SMOKE_REWARD_NAME,
        _grpo_smoke_generation_slot_reward,
    )


def test_run_rlhf_dpo_smoke(tmp_path: Path) -> None:
    cfg = load_config(_write_dpo_config(tmp_path))
    cfg.train.save_strategy = "steps"
    cfg.train.save_steps = 1
    cfg.train.save_total_limit = 1
    metrics = run_rlhf(cfg)
    assert "train_loss" in metrics
    checkpoint = Path(cfg.experiment.output_dir) / "checkpoint-1"
    validate_training_checkpoint_commit(checkpoint)
    root_metadata = load_batching_run_metadata(cfg.experiment.output_dir)
    checkpoint_metadata = load_checkpoint_batching_metadata(checkpoint)
    assert root_metadata.sample_execution_fingerprint
    assert (
        checkpoint_metadata.sample_execution_fingerprint
        == root_metadata.sample_execution_fingerprint
    )
    assert resolve_resume_checkpoint(
        cfg.experiment.output_dir,
        protocol=ShaftCheckpointProtocol.COMMITTED_MANIFEST,
    ) == str(checkpoint)


def test_run_rlhf_dpo_weighted_exact_resume_matches_uninterrupted(
    tmp_path: Path,
) -> None:
    config_path = _write_dpo_config(tmp_path)
    cfg = load_config(config_path)
    cfg.train.duration.value = 2
    cfg.train.save_strategy = "steps"
    cfg.train.save_steps = 1
    cfg.train.save_total_limit = 2
    run_rlhf(cfg)

    uninterrupted_output = Path(cfg.experiment.output_dir)
    checkpoint_one = uninterrupted_output / "checkpoint-1"
    expected_final = uninterrupted_output / "checkpoint-2"
    validate_training_checkpoint_commit(checkpoint_one)
    validate_training_checkpoint_commit(expected_final)

    resumed = load_config(config_path)
    resumed.experiment.output_dir = str(tmp_path / "resumed_outputs_dpo")
    resumed.train.duration.value = 2
    resumed.train.save_strategy = "steps"
    resumed.train.save_steps = 1
    resumed.train.save_total_limit = 2
    resumed.train.resume_from_checkpoint = str(checkpoint_one)
    run_rlhf(resumed)

    actual_final = Path(resumed.experiment.output_dir) / "checkpoint-2"
    validate_training_checkpoint_commit(actual_final)
    _assert_checkpoint_training_state_equal(expected_final, actual_final)


def test_run_rlhf_dpo_eval_uses_distinct_eval_collator(tmp_path: Path) -> None:
    cfg = load_config(_write_dpo_config(tmp_path))
    cfg.eval.enabled = True
    cfg.eval.eval_strategy = "steps"
    cfg.eval.eval_steps = 1
    cfg.eval.min_pixels = 200
    cfg.eval.max_pixels = 2000
    cfg.eval.datasets = {
        "dpo_ds": EvalDatasetPolicyConfig(min_pixels=300, max_pixels=3000)
    }
    cfg.eval.metric_for_best_model = "eval_final_loss"
    captured = {}
    original_get_eval_dataloader = ShaftDPOTrainer.get_eval_dataloader

    def _capture_eval_collator(self, eval_dataset=None):
        captured["train_collator"] = self._shaft_train_data_collator
        captured["eval_collator"] = self.eval_data_collator
        return original_get_eval_dataloader(self, eval_dataset)

    with patch.object(ShaftDPOTrainer, "get_eval_dataloader", _capture_eval_collator):
        metrics = run_rlhf(cfg)

    assert "train_loss" in metrics
    assert captured["train_collator"] is not captured["eval_collator"]
    assert (
        captured["eval_collator"].min_pixels,
        captured["eval_collator"].max_pixels,
    ) == (200, 2000)
    assert captured["eval_collator"]._resolve_pixel_budget(["dpo_ds"]) == (300, 3000)


def test_run_rlhf_ppo_smoke(tmp_path: Path) -> None:
    cfg = load_config(_write_ppo_config(tmp_path))
    cfg.eval.enabled = True
    captured: dict[str, object] = {}
    original_train = ShaftPPOTrainer.train

    def _capture_duration(self, *args, **kwargs):
        captured["num_total_batches"] = int(self.args.num_total_batches)
        captured["collator_input_mode"] = self.data_collator.input_mode
        captured["collator_padding_side"] = self.data_collator.padding_side
        return original_train(self, *args, **kwargs)

    with patch("shaft.pipeline.rlhf.resolve_eval_input_policy") as resolve_eval_policy:
        with patch.object(ShaftPPOTrainer, "train", _capture_duration):
            metrics = run_rlhf(cfg)
    assert "episode" in metrics
    assert "objective/rlhf_reward" in metrics
    assert captured["num_total_batches"] == int(cfg.train.duration.value)
    assert captured["collator_input_mode"] == "generation"
    assert captured["collator_padding_side"] == "left"
    assert load_batching_run_metadata(
        cfg.experiment.output_dir
    ).sample_execution_fingerprint
    resolve_eval_policy.assert_not_called()


def test_run_rlhf_ppo_rejects_resume_before_checkpoint_resolution(
    tmp_path: Path,
) -> None:
    cfg = load_config(_write_ppo_config(tmp_path))
    cfg.train.resume_from_checkpoint = str(tmp_path / "does-not-exist")

    with pytest.raises(ValueError, match="PPOTrainer does not support resume"):
        run_rlhf(cfg)


def test_run_rlhf_ppo_rejects_periodic_checkpoint_save(tmp_path: Path) -> None:
    cfg = load_config(_write_ppo_config(tmp_path))
    cfg.train.save_strategy = "steps"
    cfg.train.save_steps = 1

    with pytest.raises(ValueError, match="does not publish resumable training checkpoints"):
        run_rlhf(cfg)


def test_run_rlhf_grpo_smoke(tmp_path: Path) -> None:
    cfg = load_config(_write_grpo_config(tmp_path))
    with patch("shaft.algorithms.grpo.ShaftGRPOTrainer", _FakeTrainer):
        metrics = run_rlhf(cfg)
    assert "train_loss" in metrics


def test_run_rlhf_grpo_publishes_checkpoint_commit(tmp_path: Path) -> None:
    cfg = load_config(_write_grpo_config(tmp_path))
    cfg.train.save_strategy = "steps"
    cfg.train.save_steps = 2
    cfg.train.save_total_limit = 1

    def _save_only(self, *, resume_from_checkpoint=None):
        assert resume_from_checkpoint is None
        self.create_optimizer_and_scheduler(num_training_steps=2)
        self.state.global_step = 2
        self._save_checkpoint(self.model, trial=None)
        self.control = self.callback_handler.on_save(
            self.args,
            self.state,
            self.control,
        )
        return SimpleNamespace(metrics={"train_loss": 0.0})

    with patch.object(ShaftGRPOTrainer, "train", _save_only):
        metrics = run_rlhf(cfg)

    checkpoint = Path(cfg.experiment.output_dir) / "checkpoint-2"
    assert metrics["train_loss"] == 0.0
    validate_training_checkpoint_commit(checkpoint)
    root_metadata = load_batching_run_metadata(cfg.experiment.output_dir)
    checkpoint_metadata = load_checkpoint_batching_metadata(checkpoint)
    assert root_metadata.sample_execution_fingerprint
    assert (
        checkpoint_metadata.sample_execution_fingerprint
        == root_metadata.sample_execution_fingerprint
    )
    assert resolve_resume_checkpoint(
        cfg.experiment.output_dir,
        protocol=ShaftCheckpointProtocol.COMMITTED_MANIFEST,
    ) == str(checkpoint)


def test_run_rlhf_grpo_generation_safe_exact_resume_matches_uninterrupted(
    tmp_path: Path,
) -> None:
    config_path = _write_grpo_config(tmp_path, sample_count=4)
    uninterrupted = load_config(config_path)
    uninterrupted.experiment.output_dir = str(tmp_path / "outputs_grpo_full")
    uninterrupted.train.duration.value = 4
    uninterrupted.train.save_strategy = "steps"
    uninterrupted.train.save_steps = 2
    uninterrupted.train.save_total_limit = 2
    uninterrupted.rlhf.grpo.reward_functions[0].name = _GRPO_SMOKE_REWARD_NAME
    run_rlhf(uninterrupted)

    checkpoint_two = Path(uninterrupted.experiment.output_dir) / "checkpoint-2"
    expected_final = Path(uninterrupted.experiment.output_dir) / "checkpoint-4"
    validate_training_checkpoint_commit(checkpoint_two)
    validate_training_checkpoint_commit(expected_final)

    checkpoint_two_adapter = load_file(
        str(checkpoint_two / "adapter_model.safetensors")
    )
    expected_adapter = load_file(str(expected_final / "adapter_model.safetensors"))
    assert checkpoint_two_adapter.keys() == expected_adapter.keys()
    assert any(
        not torch.equal(checkpoint_two_adapter[name], expected_adapter[name])
        for name in expected_adapter
    ), "GRPO smoke reward must produce a real parameter update after checkpoint-2."

    resumed = load_config(config_path)
    resumed.experiment.output_dir = str(tmp_path / "outputs_grpo_resumed")
    resumed.train.duration.value = 4
    resumed.train.save_strategy = "steps"
    resumed.train.save_steps = 2
    resumed.train.save_total_limit = 2
    resumed.train.resume_from_checkpoint = str(checkpoint_two)
    resumed.rlhf.grpo.reward_functions[0].name = _GRPO_SMOKE_REWARD_NAME
    run_rlhf(resumed)

    actual_final = Path(resumed.experiment.output_dir) / "checkpoint-4"
    validate_training_checkpoint_commit(actual_final)
    _assert_checkpoint_training_state_equal(expected_final, actual_final)


def _assert_checkpoint_training_state_equal(expected: Path, actual: Path) -> None:
    expected_adapter = load_file(str(expected / "adapter_model.safetensors"))
    actual_adapter = load_file(str(actual / "adapter_model.safetensors"))
    assert expected_adapter.keys() == actual_adapter.keys()
    for name in expected_adapter:
        assert torch.equal(expected_adapter[name], actual_adapter[name]), name

    for filename in ("optimizer.pt", "scheduler.pt"):
        _assert_nested_state_equal(
            torch.load(expected / filename, map_location="cpu", weights_only=True),
            torch.load(actual / filename, map_location="cpu", weights_only=True),
        )
    expected_rng = sorted(expected.glob("rng_state*.pth"))
    actual_rng = sorted(actual.glob("rng_state*.pth"))
    assert [path.name for path in expected_rng] == [path.name for path in actual_rng]
    for expected_path, actual_path in zip(expected_rng, actual_rng, strict=True):
        _assert_nested_state_equal(
            torch.load(expected_path, map_location="cpu", weights_only=False),
            torch.load(actual_path, map_location="cpu", weights_only=False),
        )


def _assert_nested_state_equal(expected, actual) -> None:
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert torch.equal(expected, actual)
        return
    if isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray)
        assert np.array_equal(expected, actual)
        return
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert expected.keys() == actual.keys()
        for key in expected:
            _assert_nested_state_equal(expected[key], actual[key])
        return
    if isinstance(expected, (list, tuple)):
        assert type(expected) is type(actual)
        assert len(expected) == len(actual)
        for expected_item, actual_item in zip(expected, actual, strict=True):
            _assert_nested_state_equal(expected_item, actual_item)
        return
    assert expected == actual
