from __future__ import annotations

from pathlib import Path

import pytest

from shaft.config import RuntimeConfig, load_config


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
    config_path = tmp_path / "config.yaml"
    config_path.write_text(payload, encoding="utf-8")
    cfg = load_config(config_path)
    assert isinstance(cfg, RuntimeConfig)
    assert cfg.experiment.name == "demo"
    assert len(cfg.data.datasets) == 1
    assert cfg.data.datasets[0].dataset_name == "ds1"
    assert cfg.model.finetune.mode == "full"
    assert cfg.model.attn_implementation is None


def test_normalization(tmp_path: Path) -> None:
    payload = """
algorithm:
  name: SFT
data:
  mix_strategy: INTERLEAVE_OVER
  mix_refresh: EPOCH_REFRESH
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
  param_group_lrs:
    Language_Model: 1.0e-5
    modules_to_save: 2.5e-5
model:
  finetune:
    mode: DORA
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(payload, encoding="utf-8")
    cfg = load_config(config_path)
    assert cfg.algorithm.name == "sft"
    assert cfg.data.mix_strategy == "interleave_over"
    assert cfg.data.mix_refresh == "epoch_refresh"
    assert cfg.train.scheduler_name == "linear"
    assert cfg.train.loss_scale == "all"
    assert cfg.train.gradient_checkpointing is True
    assert cfg.train.param_group_lrs == {
        "language_model": pytest.approx(1.0e-5),
        "modules_to_save": pytest.approx(2.5e-5),
    }
    assert cfg.model.finetune.mode == "dora"
    assert cfg.data.datasets[0].help == "demo dataset"
    assert cfg.data.datasets[0].tags == ["a", "b"]


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
    config_path = tmp_path / "config.yaml"
    config_path.write_text(payload, encoding="utf-8")
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
    config_path = tmp_path / "config.yaml"
    config_path.write_text(payload, encoding="utf-8")
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
    config_path = tmp_path / "config.yaml"
    config_path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match="train.param_group_lrs\\['aligner'\\] must be > 0"):
        load_config(config_path)


def test_invalid_mix_refresh_raises(tmp_path: Path) -> None:
    payload = """
data:
  mix_refresh: every_step
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported data.mix_refresh"):
        load_config(config_path)


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
    config_path = tmp_path / "config.yaml"
    config_path.write_text(payload, encoding="utf-8")
    cfg = load_config(config_path)
    freeze = cfg.model.finetune.freeze
    assert freeze.groups == ["vision_tower", "aligner"]
    assert freeze.prefixes == ["model.visual"]
    assert freeze.trainable_prefixes == ["lm_head"]
    assert freeze.regex == ".*visual.*"
    assert freeze.trainable_regex == ".*lm_head.*"


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
    config_path = tmp_path / "config.yaml"
    config_path.write_text(payload, encoding="utf-8")
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
    config_path = tmp_path / "config.yaml"
    config_path.write_text(payload, encoding="utf-8")
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
    config_path = tmp_path / "config.yaml"
    config_path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match="model.finetune.freeze.trainable_regex"):
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
    config_path = tmp_path / "config.yaml"
    config_path.write_text(payload, encoding="utf-8")
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
    config_path = tmp_path / "config.yaml"
    config_path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(config_path)


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
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match="GRPO currently expects jsonl_sft data"):
        load_config(config_path)


def test_grpo_requires_static_mix_refresh(tmp_path: Path) -> None:
    payload = """
algorithm:
  name: grpo
data:
  mix_refresh: epoch_refresh
  datasets:
    - dataset_name: ds1
      source_type: jsonl_sft
      train_path: train.jsonl
      val_path: val.jsonl
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match="GRPO currently requires data.mix_refresh='static'"):
        load_config(config_path)


def test_rlhf_numeric_validation_raises(tmp_path: Path) -> None:
    payload = """
algorithm:
  name: ppo
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
    config_path = tmp_path / "config.yaml"
    config_path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(config_path)


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
rlhf:
  enabled: true
  grpo:
    beta: 0.01
    num_generations: 4
    max_completion_length: 96
    temperature: 0.8
    top_p: 0.95
    reward_functions:
      - name: exact_match
        codec: json_any
        weight: 2.0
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(payload, encoding="utf-8")
    cfg = load_config(config_path)
    assert cfg.algorithm.name == "grpo"
    assert cfg.rlhf.grpo.num_generations == 4
    assert cfg.rlhf.grpo.reward_functions[0].name == "exact_match"
    assert cfg.rlhf.grpo.reward_functions[0].codec == "json_any"
    assert cfg.rlhf.grpo.reward_functions[0].weight == pytest.approx(2.0)


def test_load_config_resolves_catalog_entries(tmp_path: Path) -> None:
    catalog_path = tmp_path / "datasets.yaml"
    catalog_path.write_text(
        """
datasets:
  registry_ds:
    source_type: jsonl_sft
    train_path: registry/train.jsonl
    val_path: registry/val.jsonl
    weight: 2.0
    help: demo
    tags: [base, train]
""",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
data:
  catalog_path: {catalog_path.name}
  catalog_names: [registry_ds]
""",
        encoding="utf-8",
    )

    cfg = load_config(config_path)
    assert len(cfg.data.datasets) == 1
    dataset = cfg.data.datasets[0]
    assert dataset.dataset_name == "registry_ds"
    assert dataset.weight == 2.0
    assert dataset.help == "demo"
    assert dataset.tags == ["base", "train"]
    assert dataset.train_paths == [str((tmp_path / "registry" / "train.jsonl").resolve())]
    assert dataset.val_paths == [str((tmp_path / "registry" / "val.jsonl").resolve())]
    assert cfg.data.catalog_path == str(catalog_path.resolve())
    assert cfg.data.catalog_names == ["registry_ds"]


def test_load_config_merges_catalog_entries_and_inline_datasets(tmp_path: Path) -> None:
    catalog_path = tmp_path / "datasets.yaml"
    catalog_path.write_text(
        """
datasets:
  ds_from_catalog:
    source_type: jsonl_sft
    train_path: train_a.jsonl
    val_path: val_a.jsonl
""",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
data:
  catalog_path: {catalog_path.name}
  catalog_names: [ds_from_catalog]
  datasets:
    - dataset_name: inline_ds
      train_path: inline_train.jsonl
      val_path: inline_val.jsonl
""",
        encoding="utf-8",
    )

    cfg = load_config(config_path)
    assert [dataset.dataset_name for dataset in cfg.data.datasets] == ["ds_from_catalog", "inline_ds"]
    assert cfg.data.datasets[1].train_paths == [str((tmp_path / "inline_train.jsonl").resolve())]
    assert cfg.data.datasets[1].val_paths == [str((tmp_path / "inline_val.jsonl").resolve())]


def test_load_config_raises_for_missing_catalog_entry(tmp_path: Path) -> None:
    catalog_path = tmp_path / "datasets.yaml"
    catalog_path.write_text("datasets: {}\n", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
data:
  catalog_path: {catalog_path.name}
  catalog_names: [missing_ds]
""",
        encoding="utf-8",
    )

    with pytest.raises(KeyError):
        load_config(config_path)


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
    config_path = tmp_path / "config.yaml"
    config_path.write_text(payload, encoding="utf-8")
    cfg = load_config(config_path)
    assert cfg.eval.online_metrics_enabled is True
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
    config_path = tmp_path / "config.yaml"
    config_path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match="missing online eval policies"):
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
    config_path = tmp_path / "config.yaml"
    config_path.write_text(payload, encoding="utf-8")
    cfg = load_config(config_path)
    assert set(cfg.eval.datasets.keys()) == {"eval_ds"}


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
    config_path = tmp_path / "config.yaml"
    config_path.write_text(payload, encoding="utf-8")
    config = load_config(config_path)
    assert config.eval.metric_for_best_model == "eval_final_score"
    assert config.eval.greater_is_better is True


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
    config_path = tmp_path / "config.yaml"
    config_path.write_text(payload, encoding="utf-8")
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
    config_path = tmp_path / "config.yaml"
    config_path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match="prediction_codec='not_registered'|prediction_codec=.*not_registered"):
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
    config_path = tmp_path / "config.yaml"
    config_path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match="target_adapter='not_registered'|target_adapter=.*not_registered"):
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
    config_path = tmp_path / "config.yaml"
    config_path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match="unregistered metric 'not_registered'|unregistered metric .*not_registered"):
        load_config(config_path)


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
    config_path = tmp_path / "config.yaml"
    config_path.write_text(payload, encoding="utf-8")
    cfg = load_config(config_path)
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
    config_path = tmp_path / "config.yaml"
    config_path.write_text(payload, encoding="utf-8")
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
    config_path = tmp_path / "config.yaml"
    config_path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match="at least one dataset with use_for_eval=true"):
        load_config(config_path)
