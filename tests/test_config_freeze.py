from __future__ import annotations

from pathlib import Path

import pytest

from shaft.config import load_config
from tests.support.configs import load_config_from_yaml, write_config_yaml


pytestmark = pytest.mark.component


def test_normalization_supports_freeze_groups_and_prefixes(tmp_path: Path) -> None:
    payload = """
data:
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
model:
  finetune:
    freeze:
      groups: [Vision_Tower, aligner, vision_tower]
      prefixes: [" model.visual ", "", "model.visual"]
      trainable_prefixes: [" lm_head ", ""]
      regex: ".*visual.*"
      trainable_regex: ".*lm_head.*"
"""
    cfg = load_config_from_yaml(tmp_path, payload)
    freeze = cfg.model.finetune.freeze
    assert freeze.groups == ["vision_tower", "aligner"]
    assert freeze.prefixes == ["model.visual"]
    assert freeze.trainable_prefixes == ["lm_head"]
    assert freeze.regex == ".*visual.*"
    assert freeze.trainable_regex == ".*lm_head.*"


def test_normalization_preserves_explicit_peft_parameter_targets(tmp_path: Path) -> None:
    payload = """
data:
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
model:
  finetune:
    mode: lora
    target_modules: []
    target_parameters: [auto, " mlp.gate.weight ", auto]
"""

    cfg = load_config_from_yaml(tmp_path, payload)

    assert cfg.model.finetune.target_modules == []
    assert cfg.model.finetune.target_parameters == ["auto", "mlp.gate.weight"]


def test_invalid_freeze_group_raises(tmp_path: Path) -> None:
    payload = """
data:
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
model:
  finetune:
    freeze:
      groups: [unknown_group]
"""
    config_path = write_config_yaml(tmp_path, payload)

    with pytest.raises(ValueError, match="Unsupported model.finetune.freeze.groups"):
        load_config(config_path)


def test_invalid_freeze_regex_raises(tmp_path: Path) -> None:
    payload = """
data:
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
model:
  finetune:
    freeze:
      regex: "*invalid"
"""
    config_path = write_config_yaml(tmp_path, payload)

    with pytest.raises(ValueError, match="model.finetune.freeze.regex"):
        load_config(config_path)


def test_invalid_trainable_freeze_regex_raises(tmp_path: Path) -> None:
    payload = """
data:
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
model:
  finetune:
    freeze:
      trainable_regex: "*invalid"
"""
    config_path = write_config_yaml(tmp_path, payload)

    with pytest.raises(ValueError, match="model.finetune.freeze.trainable_regex"):
        load_config(config_path)
