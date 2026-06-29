from __future__ import annotations

from pathlib import Path

import pytest

from tests.support.configs import load_config_from_yaml


pytestmark = pytest.mark.component


def test_load_config_supports_online_eval_dataset_policies(tmp_path: Path) -> None:
    payload = """
algorithm:
  name: sft
data:
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
eval:
  enabled: true
  online_metrics_enabled: true
  metric_for_best_model: eval_final_score
  greater_is_better: true
  datasets:
    ds1:
      prediction_codec: json_object
      target_adapter: target_text
      target_adapter_params:
        codec: json_object
      metrics:
        - name: parse_success
        - name: exact_match
      primary_metric: exact_match
      normalizer:
        type: identity
      weight: 1.0
"""
    cfg = load_config_from_yaml(tmp_path, payload)
    assert cfg.eval.online_metrics_enabled is True
    assert cfg.eval.loss_metrics_enabled is True
    assert cfg.eval.metric_for_best_model == "eval_final_score"
    assert cfg.eval.greater_is_better is True
    assert "ds1" in cfg.eval.datasets
    policy = cfg.eval.datasets["ds1"]
    assert policy.prediction_codec == "json_object"
    assert policy.target_adapter == "target_text"
    assert policy.target_adapter_params == {"codec": "json_object"}
    assert [metric.name for metric in policy.metrics] == ["parse_success", "exact_match"]
    assert policy.primary_metric == "exact_match"
    assert policy.normalizer.type == "identity"


def test_load_config_supports_grpo_online_eval_dataset_policies(tmp_path: Path) -> None:
    payload = """
algorithm:
  name: grpo
data:
  datasets:
    - dataset_name: ds1
      source_type: jsonl_sft
      train_path: train.jsonl
      val_path: val.jsonl
  mix_refresh: static
train:
  load_best_model_at_end: false
eval:
  enabled: true
  loss_metrics_enabled: false
  online_metrics_enabled: true
  metric_for_best_model: eval_final_score
  greater_is_better: true
  datasets:
    ds1:
      prediction_codec: json_object
      target_adapter: target_text
      target_adapter_params:
        codec: json_object
      metrics:
        - name: parse_success
        - name: exact_match
      primary_metric: exact_match
rlhf:
  grpo:
    reward_functions:
      - name: exact_match
        codec: json_object
"""
    cfg = load_config_from_yaml(tmp_path, payload)
    assert cfg.algorithm.name == "grpo"
    assert cfg.eval.online_metrics_enabled is True
    assert cfg.eval.metric_for_best_model == "eval_final_score"
    assert "ds1" in cfg.eval.datasets
