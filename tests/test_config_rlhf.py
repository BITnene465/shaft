from __future__ import annotations

from pathlib import Path

import pytest

from shaft.config import load_config
from tests.support.configs import load_config_from_yaml, write_config_yaml


pytestmark = pytest.mark.component


def test_grpo_requires_jsonl_sft_dataset(tmp_path: Path) -> None:
    payload = """
algorithm:
  name: grpo
data:
  datasets:
    - dataset_name: ds1
      source_type: jsonl_ppo
      train_path: train.jsonl
      val_path: val.jsonl
eval:
  enabled: false
"""
    config_path = write_config_yaml(tmp_path, payload)

    with pytest.raises(ValueError, match="GRPO currently expects jsonl_sft data"):
        load_config(config_path)


def test_rlhf_numeric_validation_raises(tmp_path: Path) -> None:
    payload = """
algorithm:
  name: ppo
train:
  save_strategy: no
data:
  datasets:
    - dataset_name: ds1
      source_type: jsonl_ppo
      train_path: train.jsonl
      val_path: val.jsonl
rlhf:
  ppo:
    cliprange: 1.5
"""
    config_path = write_config_yaml(tmp_path, payload)

    with pytest.raises(ValueError):
        load_config(config_path)


def test_ppo_rejects_periodic_checkpoint_strategy_during_normalize(
    tmp_path: Path,
) -> None:
    payload = """
algorithm:
  name: ppo
train:
  save_strategy: steps
data:
  datasets:
    - dataset_name: ds1
      source_type: jsonl_ppo
      train_path: train.jsonl
      val_path: val.jsonl
"""

    with pytest.raises(ValueError, match="does not publish resumable training checkpoints"):
        load_config(write_config_yaml(tmp_path, payload))


def test_load_config_supports_grpo_reward_config(tmp_path: Path) -> None:
    payload = """
algorithm:
  name: grpo
data:
  datasets:
    - dataset_name: ds1
      source_type: jsonl_sft
      train_path: train.jsonl
      val_path: val.jsonl
eval:
  enabled: false
rlhf:
  enabled: true
  grpo:
    beta: 0.01
    rollout:
      num_generations: 4
      max_completion_length: 96
      temperature: 0.8
      top_p: 0.95
      generation_kwargs:
        frequency_penalty: 0.1
    vllm:
      enabled: true
      mode: colocate
      model_impl: transformers
      gpu_memory_utilization: 0.25
      max_model_length: 4096
    reward_functions:
      - name: exact_match
        codec: json_any
        weight: 2.0
"""
    cfg = load_config_from_yaml(tmp_path, payload)
    assert cfg.algorithm.name == "grpo"
    assert cfg.rlhf.grpo.num_generations == 4
    assert cfg.rlhf.grpo.rollout.num_generations == 4
    assert cfg.rlhf.grpo.rollout.max_completion_length == 96
    assert cfg.rlhf.grpo.rollout.generation_kwargs == {"frequency_penalty": 0.1}
    assert cfg.rlhf.grpo.use_vllm is True
    assert cfg.rlhf.grpo.vllm.enabled is True
    assert cfg.rlhf.grpo.vllm.mode == "colocate"
    assert cfg.rlhf.grpo.vllm.model_impl == "transformers"
    assert cfg.rlhf.grpo.vllm.gpu_memory_utilization == pytest.approx(0.25)
    assert cfg.rlhf.grpo.vllm.max_model_length == 4096
    assert cfg.rlhf.grpo.reward_functions[0].name == "exact_match"
    assert cfg.rlhf.grpo.reward_functions[0].codec == "json_any"
    assert cfg.rlhf.grpo.reward_functions[0].weight == pytest.approx(2.0)
