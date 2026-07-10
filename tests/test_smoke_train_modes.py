from __future__ import annotations

import json
from pathlib import Path

import pytest

from shaft.config import load_config
from shaft.model.smoke_vlm import SmokeProcessor
from shaft.pipeline import run_sft
from tests.support.configs import write_sft_smoke_config


pytestmark = pytest.mark.smoke


def _run_mode(tmp_path: Path, mode: str, *, online_eval: bool = False) -> tuple[Path, dict[str, float]]:
    cfg_path = write_sft_smoke_config(tmp_path, finetune_mode=mode, online_eval=online_eval)
    cfg = load_config(cfg_path)
    metrics = run_sft(cfg)
    assert "train_loss" in metrics
    assert "epoch" in metrics
    return cfg_path, metrics


def test_smoke_full(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    processor_batch_sizes: list[int] = []
    original_call = SmokeProcessor.__call__

    def _counting_call(self, *args, **kwargs):
        processor_batch_sizes.append(len(kwargs["text"]))
        return original_call(self, *args, **kwargs)

    monkeypatch.setattr(SmokeProcessor, "__call__", _counting_call)
    _run_mode(tmp_path, "full")
    assert processor_batch_sizes == [1, 1]


def test_smoke_lora(tmp_path: Path) -> None:
    _run_mode(tmp_path, "lora")


def test_smoke_dora(tmp_path: Path) -> None:
    _run_mode(tmp_path, "dora")


def test_smoke_qlora(tmp_path: Path) -> None:
    _run_mode(tmp_path, "qlora")


def test_smoke_online_eval_canary(tmp_path: Path) -> None:
    cfg_path, _ = _run_mode(tmp_path, "full", online_eval=True)
    cfg = load_config(cfg_path)
    trainer_state_path = Path(cfg.experiment.output_dir) / "checkpoint-1" / "trainer_state.json"
    assert trainer_state_path.exists()
    trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    assert float(trainer_state["best_metric"]) == 1.0
    assert str(trainer_state["best_model_checkpoint"]).endswith("checkpoint-1")
