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


def test_training_yaml_requires_explicit_batching_strategy(tmp_path: Path) -> None:
    config_path = write_config_yaml(
        tmp_path,
        """
data:
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
""",
        ensure_explicit_batching=False,
    )

    with pytest.raises(ValueError, match="data.batching.strategy.*explicit"):
        load_config(config_path)


def test_fixed_batching_needs_no_secondary_guard_config(tmp_path: Path) -> None:
    config = load_config(
        write_config_yaml(
            tmp_path,
            """
data:
  media_snapshot_id: fixture-v1
  batching:
    strategy: FIXED
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
""",
        )
    )

    assert config.data.batching.strategy == "fixed"


def test_bounded_cost_aware_config_is_normalized(tmp_path: Path) -> None:
    config = load_config(
        write_config_yaml(
            tmp_path,
            """
data:
  media_snapshot_id: fixture-v1
  batching:
    strategy: BOUNDED_COST_AWARE
    buffer_size: 64
    cost_cache_size: 1024
    max_samples_per_microbatch: 4
    max_padded_tokens: 512
    max_vision_patches: 1024
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
""",
        )
    )

    batching = config.data.batching
    assert batching.strategy == "bounded_cost_aware"
    assert batching.buffer_size == 64
    assert batching.cost_cache_size == 1024
    assert batching.max_samples_per_microbatch == 4
    assert batching.max_padded_tokens == 512
    assert batching.max_vision_patches == 1024
    assert not hasattr(config.train, "optimizer_batch")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("strategy", "unknown", "Unsupported data.batching.strategy"),
        ("buffer_size", "0", "data.batching.buffer_size must be > 0"),
        ("cost_cache_size", "-1", "data.batching.cost_cache_size must be >= 0"),
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


@pytest.mark.parametrize(
    ("algorithm_block", "duration_block", "message"),
    [
        (
            """
algorithm:
  name: dpo
""",
            """
  duration:
    unit: steps
    value: 2
""",
            "supports algorithm.name='sft' only",
        ),
        (
            "",
            """
  duration:
    unit: epochs
    value: 2
""",
            "requires train.duration.unit='steps'",
        ),
    ],
)
def test_bounded_batching_rejects_unsupported_algorithm_or_duration(
    tmp_path: Path,
    algorithm_block: str,
    duration_block: str,
    message: str,
) -> None:
    source_type = "jsonl_dpo" if "dpo" in algorithm_block else "jsonl_sft"
    payload = f"""
{algorithm_block}
data:
  batching:
    strategy: bounded_cost_aware
    max_samples_per_microbatch: 4
    max_padded_tokens: 512
  datasets:
    - dataset_name: ds1
      source_type: {source_type}
      train_path: train.jsonl
      val_path: val.jsonl
train:
{duration_block}
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
def test_bounded_batching_rejects_invalid_hard_budgets(
    tmp_path: Path,
    batching_fields: str,
    message: str,
) -> None:
    payload = f"""
data:
  batching:
    strategy: bounded_cost_aware
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


def test_fixed_batching_rejects_bounded_only_budget_fields(tmp_path: Path) -> None:
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
    with pytest.raises(ValueError, match="Bounded data.batching budget fields"):
        load_config(write_config_yaml(tmp_path, payload))


def test_bounded_weighted_mixing_requires_horizon_independent_shuffle(
    tmp_path: Path,
) -> None:
    payload = """
data:
  media_snapshot_id: fixture-v1
  mix_strategy: weighted
  shuffle: false
  batching:
    strategy: bounded_cost_aware
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
"""
    with pytest.raises(ValueError, match="horizon-independent"):
        load_config(write_config_yaml(tmp_path, payload))


@pytest.mark.parametrize("distributed_strategy", ["fsdp", "deepspeed"])
def test_bounded_batching_rejects_unvalidated_distributed_strategies(
    tmp_path: Path,
    distributed_strategy: str,
) -> None:
    payload = f"""
data:
  media_snapshot_id: fixture-v1
  batching:
    strategy: bounded_cost_aware
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


def test_bounded_batching_requires_immutable_media_snapshot_id(
    tmp_path: Path,
) -> None:
    payload = """
data:
  batching:
    strategy: bounded_cost_aware
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
"""
    with pytest.raises(ValueError, match="media_snapshot_id"):
        load_config(write_config_yaml(tmp_path, payload))


@pytest.mark.parametrize(
    ("legacy_block", "path"),
    [
        ("    fixed_guard:\n      policy: off", "data.batching"),
        ("    planning_window: 64", "data.batching"),
        ("    cost_plan_cache_dir: /tmp/plans", "data.batching"),
        ("    image_size_cache_size: 1024", "data.batching"),
        ("    rank_balance: true", "data.batching"),
    ],
)
def test_removed_batching_keys_fail_loudly(
    tmp_path: Path,
    legacy_block: str,
    path: str,
) -> None:
    payload = f"""
data:
  batching:
    strategy: fixed
{legacy_block}
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
"""
    with pytest.raises(ValueError, match=f"Unknown config keys at {path}"):
        load_config(write_config_yaml(tmp_path, payload))


@pytest.mark.parametrize("strategy", ["cost_aware", "dynamic_cost_aware"])
def test_removed_batching_strategies_fail_loudly(
    tmp_path: Path,
    strategy: str,
) -> None:
    payload = f"""
data:
  batching:
    strategy: {strategy}
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
"""
    with pytest.raises(ValueError, match="Unsupported data.batching.strategy"):
        load_config(write_config_yaml(tmp_path, payload))


def test_removed_optimizer_batch_config_fails_loudly(tmp_path: Path) -> None:
    payload = """
data:
  batching:
    strategy: fixed
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
train:
  optimizer_batch:
    target_samples: 64
"""
    with pytest.raises(ValueError, match="Unknown config keys at train"):
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
