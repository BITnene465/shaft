from __future__ import annotations

from pathlib import Path

import pytest

from shaft.config import RuntimeConfig, load_config


def test_load_minimal_config(tmp_path: Path) -> None:
    payload = """
experiment:
  name: demo
data:
  datasets:
    - name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(payload, encoding="utf-8")
    cfg = load_config(config_path)
    assert isinstance(cfg, RuntimeConfig)
    assert cfg.experiment.name == "demo"
    assert len(cfg.data.datasets) == 1
    assert cfg.data.datasets[0].name == "ds1"
    assert cfg.model.finetune.mode == "full"


def test_normalization(tmp_path: Path) -> None:
    payload = """
algorithm:
  name: SFT
data:
  mix_strategy: INTERLEAVE_OVER
  datasets:
    - name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
train:
  scheduler_name: auto
  lr_scheduler_type: LINEAR
model:
  finetune:
    mode: DORA
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(payload, encoding="utf-8")
    cfg = load_config(config_path)
    assert cfg.algorithm.name == "sft"
    assert cfg.data.mix_strategy == "interleave_over"
    assert cfg.train.scheduler_name == "linear"
    assert cfg.model.finetune.mode == "dora"


def test_unknown_key_raises(tmp_path: Path) -> None:
    payload = """
experiment:
  name: demo
  unknown: true
data:
  datasets:
    - name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(config_path)


def test_algorithm_source_type_mismatch_raises(tmp_path: Path) -> None:
    payload = """
algorithm:
  name: dpo
data:
  datasets:
    - name: ds1
      source_type: jsonl_sft
      train_path: train.jsonl
      val_path: val.jsonl
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(config_path)


def test_rlhf_numeric_validation_raises(tmp_path: Path) -> None:
    payload = """
algorithm:
  name: ppo
data:
  datasets:
    - name: ds1
      source_type: jsonl_ppo
      train_path: train.jsonl
      val_path: val.jsonl
rlhf:
  ppo:
    cliprange: 1.5
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(config_path)
