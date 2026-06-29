from __future__ import annotations

from pathlib import Path

import pytest

from shaft.config import load_config
from tests.support.configs import load_config_from_yaml, write_config_yaml


pytestmark = pytest.mark.component


def test_online_eval_requires_policy_for_each_dataset(tmp_path: Path) -> None:
    payload = """
algorithm:
  name: sft
data:
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
    - dataset_name: ds2
      train_path: train2.jsonl
      val_path: val2.jsonl
eval:
  enabled: true
  online_metrics_enabled: true
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
    config_path = write_config_yaml(tmp_path, payload)

    with pytest.raises(ValueError, match="dataset-policy eval is missing policies"):
        load_config(config_path)


def test_online_eval_ignores_train_only_dataset_policy_requirement(tmp_path: Path) -> None:
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
  online_metrics_enabled: true
  datasets:
    eval_ds:
      prediction_codec: text
      target_adapter: target_text
      metrics:
        - name: exact_match
      primary_metric: exact_match
      normalizer:
        type: identity
      weight: 1.0
"""
    cfg = load_config_from_yaml(tmp_path, payload)
    assert set(cfg.eval.datasets.keys()) == {"eval_ds"}


def test_online_eval_rejects_sampling(tmp_path: Path) -> None:
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
  do_sample: true
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
    config_path = write_config_yaml(tmp_path, payload)

    with pytest.raises(ValueError, match="greedy decoding"):
        load_config(config_path)


def test_online_eval_rejects_unregistered_prediction_codec(tmp_path: Path) -> None:
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
  datasets:
    ds1:
      prediction_codec: not_registered
      target_adapter: target_text
      metrics:
        - name: exact_match
      primary_metric: exact_match
      normalizer:
        type: identity
      weight: 1.0
"""
    config_path = write_config_yaml(tmp_path, payload)

    with pytest.raises(
        ValueError, match="prediction_codec='not_registered'|prediction_codec=.*not_registered"
    ):
        load_config(config_path)


def test_online_eval_rejects_unregistered_target_adapter(tmp_path: Path) -> None:
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
  datasets:
    ds1:
      prediction_codec: text
      target_adapter: not_registered
      metrics:
        - name: exact_match
      primary_metric: exact_match
      normalizer:
        type: identity
      weight: 1.0
"""
    config_path = write_config_yaml(tmp_path, payload)

    with pytest.raises(
        ValueError, match="target_adapter='not_registered'|target_adapter=.*not_registered"
    ):
        load_config(config_path)


def test_online_eval_rejects_unregistered_metric(tmp_path: Path) -> None:
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
  datasets:
    ds1:
      prediction_codec: text
      target_adapter: target_text
      metrics:
        - name: not_registered
      primary_metric: not_registered
      normalizer:
        type: identity
      weight: 1.0
"""
    config_path = write_config_yaml(tmp_path, payload)

    with pytest.raises(
        ValueError,
        match="unregistered metric 'not_registered'|unregistered metric .*not_registered",
    ):
        load_config(config_path)
