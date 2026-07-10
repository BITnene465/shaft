from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from shaft.config import load_config
from shaft.pipeline import run_rlhf
from shaft.training import ShaftPPOTrainer
from tests.support.pipeline import FakePipelineTrainer as _FakeTrainer
from tests.support.rlhf import write_dpo_config as _write_dpo_config
from tests.support.rlhf import write_grpo_config as _write_grpo_config
from tests.support.rlhf import write_ppo_config as _write_ppo_config


pytestmark = [pytest.mark.component, pytest.mark.smoke]


def test_run_rlhf_dpo_smoke(tmp_path: Path) -> None:
    cfg = load_config(_write_dpo_config(tmp_path))
    metrics = run_rlhf(cfg)
    assert "train_loss" in metrics


def test_run_rlhf_ppo_smoke(tmp_path: Path) -> None:
    cfg = load_config(_write_ppo_config(tmp_path))
    captured: dict[str, int] = {}
    original_train = ShaftPPOTrainer.train

    def _capture_duration(self, *args, **kwargs):
        captured["num_total_batches"] = int(self.args.num_total_batches)
        return original_train(self, *args, **kwargs)

    with patch.object(ShaftPPOTrainer, "train", _capture_duration):
        metrics = run_rlhf(cfg)
    assert "episode" in metrics
    assert "objective/rlhf_reward" in metrics
    assert captured["num_total_batches"] == int(cfg.train.duration.value)


def test_run_rlhf_grpo_smoke(tmp_path: Path) -> None:
    cfg = load_config(_write_grpo_config(tmp_path))
    with patch("shaft.algorithms.grpo.ShaftGRPOTrainer", _FakeTrainer):
        metrics = run_rlhf(cfg)
    assert "train_loss" in metrics
