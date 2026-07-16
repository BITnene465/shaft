from __future__ import annotations

from pathlib import Path

import pytest

from shaft.config import resolve_eval_input_policy
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
    resolved_input_policy = resolve_eval_input_policy(
        cfg.eval,
        train_min_pixels=cfg.data.min_pixels,
        train_max_pixels=cfg.data.max_pixels,
    )
    assert resolved_input_policy.pixel_budgets_by_dataset() == {}


def test_grpo_eval_rejects_loss_metrics_until_final_loss_is_supported(
    tmp_path: Path,
) -> None:
    payload = """
algorithm:
  name: grpo
data:
  datasets:
    - dataset_name: ds1
      source_type: jsonl_sft
      train_path: train.jsonl
      val_path: val.jsonl
train:
  load_best_model_at_end: false
eval:
  enabled: true
"""

    with pytest.raises(
        ValueError,
        match=r"GRPO evaluation currently supports online metrics only.*loss_metrics_enabled=false",
    ):
        load_config_from_yaml(tmp_path, payload)


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


def test_eval_pixel_budget_resolves_train_fallback_default_and_dataset_override(
    tmp_path: Path,
) -> None:
    payload = """
algorithm:
  name: sft
data:
  min_pixels: 100
  max_pixels: 1000
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
    - dataset_name: ds2
      train_path: train2.jsonl
      val_path: val2.jsonl
eval:
  enabled: true
  min_pixels: 200
  max_pixels: 2000
  datasets:
    ds1:
      max_pixels: 3000
    ds2: {}
"""
    cfg = load_config_from_yaml(tmp_path, payload)

    policy = resolve_eval_input_policy(
        cfg.eval,
        train_min_pixels=cfg.data.min_pixels,
        train_max_pixels=cfg.data.max_pixels,
    )

    assert (policy.default_pixel_budget.min_pixels, policy.default_pixel_budget.max_pixels) == (
        200,
        2000,
    )
    dataset_budget = policy.pixel_budget_for("ds1")
    assert (dataset_budget.min_pixels, dataset_budget.max_pixels) == (200, 3000)
    fallback_budget = policy.pixel_budget_for("ds2")
    assert (fallback_budget.min_pixels, fallback_budget.max_pixels) == (200, 2000)
    assert policy.pixel_budgets_by_dataset() == {"ds1": (200, 3000)}


def test_eval_pixel_budget_preserves_legacy_data_budget_fallback(tmp_path: Path) -> None:
    payload = """
algorithm:
  name: sft
data:
  min_pixels: 128
  max_pixels: 1024
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
eval:
  enabled: true
"""
    cfg = load_config_from_yaml(tmp_path, payload)

    policy = resolve_eval_input_policy(
        cfg.eval,
        train_min_pixels=cfg.data.min_pixels,
        train_max_pixels=cfg.data.max_pixels,
    )

    assert (policy.default_pixel_budget.min_pixels, policy.default_pixel_budget.max_pixels) == (
        128,
        1024,
    )


def test_eval_pixel_budget_rejects_invalid_resolved_dataset_override(tmp_path: Path) -> None:
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
  min_pixels: 200
  max_pixels: 1000
  datasets:
    ds1:
      max_pixels: 100
"""

    with pytest.raises(ValueError, match=r"eval\.datasets\.ds1 pixel budget"):
        load_config_from_yaml(tmp_path, payload)


def test_ppo_rejects_explicit_eval_pixel_budget_as_not_applicable(tmp_path: Path) -> None:
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
eval:
  enabled: true
  max_pixels: 1000
"""

    with pytest.raises(ValueError, match="not applicable"):
        load_config_from_yaml(tmp_path, payload)
