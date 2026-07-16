from __future__ import annotations

from pathlib import Path

import pytest

from tests.support.configs import load_config_from_yaml


pytestmark = pytest.mark.component


def test_online_eval_forces_final_score_as_best_metric(tmp_path: Path) -> None:
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
  metric_for_best_model: eval_loss
  greater_is_better: false
  datasets:
    ds1:
      prediction_codec: text
      target_adapter: target_text
      metrics:
        - name: exact_match
      primary_metric: exact_match
      normalizer:
        type: identity
      weight: 1.0
"""
    config = load_config_from_yaml(tmp_path, payload)
    assert config.eval.metric_for_best_model == "eval_final_score"
    assert config.eval.greater_is_better is True


def test_dataset_policy_eval_defaults_to_final_loss_without_online_metrics(tmp_path: Path) -> None:
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
  metric_for_best_model: eval_loss
  datasets:
    ds1:
      weight: 1.0
"""
    config = load_config_from_yaml(tmp_path, payload)
    assert config.eval.loss_metrics_enabled is True
    assert config.eval.online_metrics_enabled is False
    assert config.eval.metric_for_best_model == "eval_final_loss"
    assert config.eval.greater_is_better is False


def test_dataset_policy_eval_accepts_final_loss_as_best_metric(tmp_path: Path) -> None:
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
  loss_metrics_enabled: true
  metric_for_best_model: eval_final_loss
  datasets:
    ds1:
      weight: 1.0
"""
    config = load_config_from_yaml(tmp_path, payload)
    assert config.eval.metric_for_best_model == "eval_final_loss"
    assert config.eval.greater_is_better is False
