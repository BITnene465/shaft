from __future__ import annotations

from pathlib import Path

import pytest

from shaft.config import load_config
from tests.support.configs import load_config_from_yaml, write_config_yaml


pytestmark = pytest.mark.component


def test_eval_enabled_allows_train_only_dataset(tmp_path: Path) -> None:
    payload = """
algorithm:
  name: sft
data:
  datasets:
    - dataset_name: eval_ds
      train_path: train.jsonl
      val_path: val.jsonl
    - dataset_name: train_only_ds
      train_path: train2.jsonl
      use_for_eval: false
eval:
  enabled: true
"""
    cfg = load_config_from_yaml(tmp_path, payload)
    assert cfg.data.datasets[0].use_for_eval is True
    assert cfg.data.datasets[1].use_for_eval is False
    assert cfg.data.datasets[1].val_paths == []


def test_eval_enabled_requires_val_for_eval_dataset(tmp_path: Path) -> None:
    payload = """
algorithm:
  name: sft
data:
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      use_for_eval: true
eval:
  enabled: true
"""
    config_path = write_config_yaml(tmp_path, payload)

    with pytest.raises(ValueError, match="val_paths cannot be empty"):
        load_config(config_path)


def test_eval_enabled_requires_at_least_one_eval_dataset(tmp_path: Path) -> None:
    payload = """
algorithm:
  name: sft
data:
  datasets:
    - dataset_name: train_only_ds
      train_path: train.jsonl
      use_for_eval: false
eval:
  enabled: true
"""
    config_path = write_config_yaml(tmp_path, payload)

    with pytest.raises(ValueError, match="at least one dataset with use_for_eval=true"):
        load_config(config_path)
