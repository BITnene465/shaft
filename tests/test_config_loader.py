from __future__ import annotations

from pathlib import Path

import pytest

from shaft.config import RuntimeConfig, TrainDistributedConfig
from tests.support.configs import load_config_from_yaml


pytestmark = pytest.mark.component


def test_load_minimal_config(tmp_path: Path) -> None:
    payload = """
experiment:
  name: demo
data:
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
"""
    cfg = load_config_from_yaml(tmp_path, payload)
    assert isinstance(cfg, RuntimeConfig)
    assert cfg.experiment.name == "demo"
    assert len(cfg.data.datasets) == 1
    assert cfg.data.datasets[0].dataset_name == "ds1"
    assert cfg.model.finetune.mode == "full"
    assert cfg.model.attn_implementation is None
    assert isinstance(cfg.train.distributed, TrainDistributedConfig)
    assert cfg.train.distributed.strategy == "ddp"


def test_normalization(tmp_path: Path) -> None:
    payload = """
algorithm:
  name: SFT
data:
  mix_strategy: WEIGHTED
  record_cache_dir: .cache/records
  batching:
    cost_plan_cache_dir: .cache/cost-plans
  max_length: 4096
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
      help: "  demo dataset  "
      tags: [" a ", "", "b"]
train:
  scheduler_name: auto
  lr_scheduler_type: LINEAR
  loss_scale: ALL
  gradient_checkpointing: true
  distributed:
    strategy: FSDP
    fsdp:
      sharding_strategy: FULL_SHARD
      auto_wrap_policy: TRANSFORMER
      transformer_layer_cls_to_wrap: [" auto ", "auto"]
      activation_checkpointing: true
      state_dict_type: FULL_STATE_DICT
      backward_prefetch: BACKWARD_PRE
  save_epoch_interval: 2
  param_group_lrs:
    Language_Model: 1.0e-5
    modules_to_save: 2.5e-5
  no_decay_name_patterns: [" Embed_Tokens.Weight ", "", "LM_HEAD.WEIGHT", "embed_tokens.weight"]
eval:
  epoch_interval: 3
model:
  finetune:
    mode: DORA
"""
    cfg = load_config_from_yaml(tmp_path, payload)
    assert cfg.algorithm.name == "sft"
    assert cfg.data.mix_strategy == "weighted"
    assert cfg.data.record_cache_dir == str((tmp_path / ".cache/records").resolve())
    assert cfg.data.batching.cost_plan_cache_dir == str(
        (tmp_path / ".cache/cost-plans").resolve()
    )
    assert cfg.data.max_length == 4096
    assert cfg.train.scheduler_name == "linear"
    assert cfg.train.loss_scale == "all"
    assert cfg.train.gradient_checkpointing is True
    assert cfg.train.distributed.strategy == "fsdp"
    assert cfg.train.distributed.fsdp.sharding_strategy == "full_shard"
    assert cfg.train.distributed.fsdp.auto_wrap_policy == "transformer"
    assert cfg.train.distributed.fsdp.transformer_layer_cls_to_wrap == ["auto"]
    assert cfg.train.distributed.fsdp.state_dict_type == "full_state_dict"
    assert cfg.train.distributed.fsdp.backward_prefetch == "backward_pre"
    assert cfg.train.save_epoch_interval == 2
    assert cfg.eval.epoch_interval == 3
    assert cfg.train.param_group_lrs == {
        "language_model": pytest.approx(1.0e-5),
        "modules_to_save": pytest.approx(2.5e-5),
    }
    assert cfg.train.no_decay_name_patterns == ["embed_tokens.weight", "lm_head.weight"]
    assert cfg.model.finetune.mode == "dora"
    assert cfg.data.datasets[0].help == "demo dataset"
    assert cfg.data.datasets[0].tags == ["a", "b"]
