from __future__ import annotations

from pathlib import Path

import pytest

from shaft.config import load_config
from tests.support.configs import load_config_from_yaml, write_config_yaml


pytestmark = pytest.mark.component


def test_deepspeed_strategy_requires_config(tmp_path: Path) -> None:
    payload = """
data:
  batching:
    strategy: fixed
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
train:
  distributed:
    strategy: deepspeed
"""
    config_path = write_config_yaml(tmp_path, payload)

    with pytest.raises(ValueError, match="requires either"):
        load_config(config_path)


def test_deepspeed_strategy_accepts_inline_config(tmp_path: Path) -> None:
    payload = """
data:
  batching:
    strategy: fixed
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
train:
  distributed:
    strategy: deepspeed
    deepspeed:
      config:
        zero_optimization:
          stage: 3
"""
    cfg = load_config_from_yaml(tmp_path, payload)

    assert cfg.train.distributed.strategy == "deepspeed"
    assert cfg.train.distributed.deepspeed.config == {"zero_optimization": {"stage": 3}}


def test_deepspeed_config_path_resolves_relative_to_train_config(tmp_path: Path) -> None:
    train_dir = tmp_path / "configs" / "train"
    train_dir.mkdir(parents=True)
    deepspeed_dir = tmp_path / "configs" / "deepspeed"
    deepspeed_dir.mkdir(parents=True)
    deepspeed_path = deepspeed_dir / "zero3.json"
    deepspeed_path.write_text('{"zero_optimization": {"stage": 3}}\n', encoding="utf-8")
    payload = """
data:
  batching:
    strategy: fixed
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
train:
  distributed:
    strategy: deepspeed
    deepspeed:
      config_path: ../deepspeed/zero3.json
"""
    config_path = train_dir / "config.yaml"
    config_path.write_text(payload, encoding="utf-8")

    cfg = load_config(config_path)

    assert cfg.train.distributed.deepspeed.config_path == str(deepspeed_path.resolve())


def test_deepspeed_inline_config_rejects_optimizer_ownership_conflict(tmp_path: Path) -> None:
    payload = """
data:
  batching:
    strategy: fixed
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
train:
  distributed:
    strategy: deepspeed
    deepspeed:
      config:
        optimizer:
          type: AdamW
        zero_optimization:
          stage: 2
"""
    config_path = write_config_yaml(tmp_path, payload)

    with pytest.raises(ValueError, match="Shaft owns optimizer/scheduler"):
        load_config(config_path)


def test_deepspeed_config_path_rejects_optimizer_ownership_conflict(tmp_path: Path) -> None:
    train_dir = tmp_path / "configs" / "train"
    train_dir.mkdir(parents=True)
    deepspeed_dir = tmp_path / "configs" / "deepspeed"
    deepspeed_dir.mkdir(parents=True)
    (deepspeed_dir / "zero2.json").write_text(
        '{"optimizer": {"type": "AdamW"}, "zero_optimization": {"stage": 2}}\n',
        encoding="utf-8",
    )
    payload = """
data:
  batching:
    strategy: fixed
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
train:
  distributed:
    strategy: deepspeed
    deepspeed:
      config_path: ../deepspeed/zero2.json
"""
    config_path = train_dir / "config.yaml"
    config_path.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match="Shaft owns optimizer/scheduler"):
        load_config(config_path)
