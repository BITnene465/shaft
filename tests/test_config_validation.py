from __future__ import annotations

from pathlib import Path

import pytest

from shaft.config import load_config
from tests.support.configs import write_config_yaml


pytestmark = pytest.mark.component


def test_invalid_loss_scale_raises(tmp_path: Path) -> None:
    payload = """
data:
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
train:
  loss_scale: missing_strategy
"""
    config_path = write_config_yaml(tmp_path, payload)

    with pytest.raises(ValueError, match="Unsupported train.loss_scale"):
        load_config(config_path)


def test_invalid_param_group_lr_key_raises(tmp_path: Path) -> None:
    payload = """
data:
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
train:
  param_group_lrs:
    bad_group: 1.0e-5
"""
    config_path = write_config_yaml(tmp_path, payload)

    with pytest.raises(ValueError, match="Unsupported train.param_group_lrs key"):
        load_config(config_path)


def test_invalid_param_group_lr_value_raises(tmp_path: Path) -> None:
    payload = """
data:
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
train:
  param_group_lrs:
    aligner: 0
"""
    config_path = write_config_yaml(tmp_path, payload)

    with pytest.raises(ValueError, match="train.param_group_lrs\\['aligner'\\] must be > 0"):
        load_config(config_path)


def test_invalid_mix_strategy_raises(tmp_path: Path) -> None:
    payload = """
data:
  mix_strategy: interleave_over
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
"""
    config_path = write_config_yaml(tmp_path, payload)

    with pytest.raises(ValueError, match="Unsupported data.mix_strategy"):
        load_config(config_path)


def test_step_duration_requires_integer_value(tmp_path: Path) -> None:
    payload = """
data:
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
train:
  duration:
    unit: steps
    value: 1.5
"""
    config_path = write_config_yaml(tmp_path, payload)

    with pytest.raises(ValueError, match="must be an integer"):
        load_config(config_path)


def test_dataset_weight_must_be_finite_and_non_negative(tmp_path: Path) -> None:
    payload = """
data:
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
      weight: -1
"""
    config_path = write_config_yaml(tmp_path, payload)

    with pytest.raises(ValueError, match="weight must be finite and >= 0"):
        load_config(config_path)


def test_invalid_data_max_length_raises(tmp_path: Path) -> None:
    payload = """
data:
  max_length: 0
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
"""
    config_path = write_config_yaml(tmp_path, payload)

    with pytest.raises(ValueError, match="data.max_length must be > 0"):
        load_config(config_path)


def test_invalid_eval_epoch_interval_raises(tmp_path: Path) -> None:
    payload = """
data:
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
eval:
  epoch_interval: 0
"""
    config_path = write_config_yaml(tmp_path, payload)

    with pytest.raises(ValueError, match="eval.epoch_interval must be > 0"):
        load_config(config_path)


def test_invalid_save_epoch_interval_raises(tmp_path: Path) -> None:
    payload = """
data:
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
train:
  save_epoch_interval: 0
"""
    config_path = write_config_yaml(tmp_path, payload)

    with pytest.raises(ValueError, match="train.save_epoch_interval must be > 0"):
        load_config(config_path)


def test_unknown_key_raises(tmp_path: Path) -> None:
    payload = """
experiment:
  name: demo
  unknown: true
data:
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
"""
    config_path = write_config_yaml(tmp_path, payload)

    with pytest.raises(ValueError):
        load_config(config_path)


def test_algorithm_source_type_mismatch_raises(tmp_path: Path) -> None:
    payload = """
algorithm:
  name: dpo
data:
  datasets:
    - dataset_name: ds1
      source_type: jsonl_sft
      train_path: train.jsonl
      val_path: val.jsonl
"""
    config_path = write_config_yaml(tmp_path, payload)

    with pytest.raises(ValueError):
        load_config(config_path)
