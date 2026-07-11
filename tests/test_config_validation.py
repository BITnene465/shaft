from __future__ import annotations

from pathlib import Path

import pytest

from shaft.config import load_config
from tests.support.configs import write_config_yaml


pytestmark = pytest.mark.component


def test_full_determinism_config_is_normalized(tmp_path: Path) -> None:
    config = load_config(
        write_config_yaml(
            tmp_path,
            """
data:
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
train:
  full_determinism: true
""",
        )
    )

    assert config.train.full_determinism is True


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
    config_path = write_config_yaml(tmp_path, payload)

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
    config_path = write_config_yaml(tmp_path, payload)

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
    config_path = write_config_yaml(tmp_path, payload)

    with pytest.raises(ValueError, match="train.param_group_lrs\\['aligner'\\] must be > 0"):
        load_config(config_path)


def test_invalid_mix_strategy_raises(tmp_path: Path) -> None:
    payload = """
data:
  mix_strategy: interleave_over
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
"""
    config_path = write_config_yaml(tmp_path, payload)

    with pytest.raises(ValueError, match="Unsupported data.mix_strategy"):
        load_config(config_path)


def test_cost_aware_batching_config_is_normalized(tmp_path: Path) -> None:
    payload = """
data:
  batching:
    strategy: COST_AWARE
    planning_window: 64
    image_size_cache_size: 0
    cost_plan_cache_dir: "  /tmp/shaft-cost-plans  "
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
train:
  duration:
    unit: steps
    value: 2
"""
    config = load_config(write_config_yaml(tmp_path, payload))

    assert config.data.batching.strategy == "cost_aware"
    assert config.data.batching.planning_window == 64
    assert config.data.batching.image_size_cache_size == 0
    assert config.data.batching.cost_plan_cache_dir == "/tmp/shaft-cost-plans"


def test_dynamic_cost_aware_batching_config_is_normalized(tmp_path: Path) -> None:
    payload = """
data:
  batching:
    strategy: DYNAMIC_COST_AWARE
    planning_window: 64
    max_samples_per_microbatch: 4
    max_padded_tokens: 512
    max_vision_patches: 1024
    rank_balance: true
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
train:
  duration:
    unit: steps
    value: 2
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 2
  optimizer_batch:
    target_samples: 6
"""
    config = load_config(write_config_yaml(tmp_path, payload))

    assert config.data.batching.strategy == "dynamic_cost_aware"
    assert config.data.batching.max_samples_per_microbatch == 4
    assert config.data.batching.max_padded_tokens == 512
    assert config.data.batching.max_vision_patches == 1024
    assert config.data.batching.rank_balance is True
    assert config.train.optimizer_batch.target_samples == 6
    assert config.train.optimizer_batch.target_supervised_tokens is None


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("strategy", "unknown", "Unsupported data.batching.strategy"),
        ("planning_window", "0", "data.batching.planning_window must be > 0"),
        (
            "image_size_cache_size",
            "-1",
            "data.batching.image_size_cache_size must be >= 0",
        ),
    ],
)
def test_invalid_batching_config_raises(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    payload = f"""
data:
  batching:
    {field}: {value}
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
"""

    with pytest.raises(ValueError, match=message):
        load_config(write_config_yaml(tmp_path, payload))


def test_cost_aware_batching_rejects_non_sft_algorithm(tmp_path: Path) -> None:
    payload = """
algorithm:
  name: dpo
data:
  batching:
    strategy: cost_aware
  datasets:
    - dataset_name: ds1
      source_type: jsonl_dpo
      train_path: train.jsonl
      val_path: val.jsonl
train:
  duration:
    unit: steps
    value: 1
"""

    with pytest.raises(ValueError, match="supports algorithm.name='sft' only"):
        load_config(write_config_yaml(tmp_path, payload))


def test_cost_aware_batching_rejects_epoch_duration(tmp_path: Path) -> None:
    payload = """
data:
  batching:
    strategy: cost_aware
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
train:
  duration:
    unit: epochs
    value: 1
"""

    with pytest.raises(ValueError, match="requires train.duration.unit='steps'"):
        load_config(write_config_yaml(tmp_path, payload))


@pytest.mark.parametrize(
    ("optimizer_batch", "message"),
    [
        (
            "target_samples: 4\n    target_supervised_tokens: 100",
            "at most one of target_samples and target_supervised_tokens",
        ),
        ("target_samples: 0", "target_samples must be > 0"),
        ("target_supervised_tokens: 0", "target_supervised_tokens must be > 0"),
    ],
)
def test_dynamic_optimizer_batch_rejects_invalid_config(
    tmp_path: Path,
    optimizer_batch: str,
    message: str,
) -> None:
    payload = f"""
data:
  batching:
    strategy: dynamic_cost_aware
    max_samples_per_microbatch: 4
    max_padded_tokens: 512
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
train:
  duration:
    unit: steps
    value: 2
  optimizer_batch:
    {optimizer_batch}
"""

    with pytest.raises(ValueError, match=message):
        load_config(write_config_yaml(tmp_path, payload))


@pytest.mark.parametrize(
    ("batching_fields", "message"),
    [
        ("max_samples_per_microbatch: 0\n    max_padded_tokens: 512", "max_samples_per_microbatch"),
        ("max_samples_per_microbatch: 4", "max_padded_tokens"),
        ("max_samples_per_microbatch: 4\n    max_padded_tokens: 0", "max_padded_tokens"),
        (
            "max_samples_per_microbatch: 4\n    max_padded_tokens: 512\n    max_vision_patches: 0",
            "max_vision_patches",
        ),
    ],
)
def test_dynamic_batching_rejects_invalid_hard_budgets(
    tmp_path: Path,
    batching_fields: str,
    message: str,
) -> None:
    payload = f"""
data:
  batching:
    strategy: dynamic_cost_aware
    {batching_fields}
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
train:
  duration:
    unit: steps
    value: 2
"""

    with pytest.raises(ValueError, match=message):
        load_config(write_config_yaml(tmp_path, payload))


def test_fixed_batching_rejects_dynamic_only_budget_fields(tmp_path: Path) -> None:
    payload = """
data:
  batching:
    strategy: fixed
    max_padded_tokens: 512
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
"""

    with pytest.raises(ValueError, match="Dynamic data.batching budget fields"):
        load_config(write_config_yaml(tmp_path, payload))


def test_dynamic_token_target_rejects_horizon_dependent_weighted_plan(
    tmp_path: Path,
) -> None:
    payload = """
data:
  mix_strategy: weighted
  shuffle: false
  batching:
    strategy: dynamic_cost_aware
    max_samples_per_microbatch: 4
    max_padded_tokens: 512
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
    - dataset_name: ds2
      train_path: train2.jsonl
      val_path: val2.jsonl
train:
  duration:
    unit: steps
    value: 2
  optimizer_batch:
    target_supervised_tokens: 100
"""

    with pytest.raises(ValueError, match="horizon-dependent"):
        load_config(write_config_yaml(tmp_path, payload))


@pytest.mark.parametrize("distributed_strategy", ["fsdp", "deepspeed"])
def test_dynamic_batching_rejects_unvalidated_distributed_strategies(
    tmp_path: Path,
    distributed_strategy: str,
) -> None:
    payload = f"""
data:
  batching:
    strategy: dynamic_cost_aware
    max_samples_per_microbatch: 4
    max_padded_tokens: 512
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
train:
  duration:
    unit: steps
    value: 2
  distributed:
    strategy: {distributed_strategy}
"""

    with pytest.raises(ValueError, match="supports.*ddp.*only"):
        load_config(write_config_yaml(tmp_path, payload))


def test_step_duration_requires_integer_value(tmp_path: Path) -> None:
    payload = """
data:
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
train:
  duration:
    unit: steps
    value: 1.5
"""
    config_path = write_config_yaml(tmp_path, payload)

    with pytest.raises(ValueError, match="must be an integer"):
        load_config(config_path)


def test_dataset_weight_must_be_finite_and_non_negative(tmp_path: Path) -> None:
    payload = """
data:
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
      weight: -1
"""
    config_path = write_config_yaml(tmp_path, payload)

    with pytest.raises(ValueError, match="weight must be finite and >= 0"):
        load_config(config_path)


def test_invalid_data_max_length_raises(tmp_path: Path) -> None:
    payload = """
data:
  max_length: 0
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
"""
    config_path = write_config_yaml(tmp_path, payload)

    with pytest.raises(ValueError, match="data.max_length must be > 0"):
        load_config(config_path)


def test_invalid_eval_epoch_interval_raises(tmp_path: Path) -> None:
    payload = """
data:
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
eval:
  epoch_interval: 0
"""
    config_path = write_config_yaml(tmp_path, payload)

    with pytest.raises(ValueError, match="eval.epoch_interval must be > 0"):
        load_config(config_path)


def test_invalid_save_epoch_interval_raises(tmp_path: Path) -> None:
    payload = """
data:
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
train:
  save_epoch_interval: 0
"""
    config_path = write_config_yaml(tmp_path, payload)

    with pytest.raises(ValueError, match="train.save_epoch_interval must be > 0"):
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
    config_path = write_config_yaml(tmp_path, payload)

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
    config_path = write_config_yaml(tmp_path, payload)

    with pytest.raises(ValueError):
        load_config(config_path)
