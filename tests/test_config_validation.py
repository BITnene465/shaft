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


def test_invalid_schedule_mixing_raises(tmp_path: Path) -> None:
    payload = """
data:
  schedule:
    mixing: interleave_over
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
"""
    config_path = write_config_yaml(tmp_path, payload)

    with pytest.raises(ValueError, match="Unsupported data.schedule.mixing"):
        load_config(config_path)


def test_invalid_schedule_boolean_is_rejected(tmp_path: Path) -> None:
    payload = """
data:
  schedule:
    shuffle: sometimes
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
"""

    with pytest.raises(ValueError, match="data.schedule.shuffle must be a boolean"):
        load_config(write_config_yaml(tmp_path, payload))


def test_quoted_false_dataset_enabled_does_not_activate_source(tmp_path: Path) -> None:
    payload = """
data:
  datasets:
    - dataset_name: disabled
      train_path: disabled.jsonl
      enabled: "false"
    - dataset_name: active
      train_path: active.jsonl
      val_path: active-val.jsonl
"""
    config = load_config(write_config_yaml(tmp_path, payload))

    assert config.data.datasets[0].enabled is False
    assert config.data.datasets[1].enabled is True


def test_quoted_false_eval_enabled_does_not_require_validation_paths(
    tmp_path: Path,
) -> None:
    payload = """
data:
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
eval:
  enabled: "false"
"""
    config = load_config(write_config_yaml(tmp_path, payload))

    assert config.eval.enabled is False


def test_training_yaml_requires_explicit_batching_contract(tmp_path: Path) -> None:
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

    with pytest.raises(ValueError, match="data.batching.*explicit"):
        load_config(config_path)


def test_training_yaml_requires_explicit_batch_layout_contract(
    tmp_path: Path,
) -> None:
    config_path = write_config_yaml(
        tmp_path,
        """
data:
  batching:
    grouping: none
    cardinality: fixed
    packing:
      mode: none
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
""",
        ensure_explicit_batching=False,
    )

    with pytest.raises(ValueError, match="data.batching.layout.*explicit"):
        load_config(config_path)


def test_training_yaml_requires_explicit_packing_contract(tmp_path: Path) -> None:
    config_path = write_config_yaml(
        tmp_path,
        """
data:
  batching:
    grouping: none
    cardinality: fixed
    layout: padded
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
""",
        ensure_explicit_batching=False,
    )

    with pytest.raises(ValueError, match="data.batching.packing.*explicit"):
        load_config(config_path)


@pytest.mark.parametrize("missing_field", ["grouping", "cardinality"])
def test_training_yaml_requires_each_explicit_batch_axis(
    tmp_path: Path,
    missing_field: str,
) -> None:
    batching_lines = {
        "grouping": "    grouping: none",
        "cardinality": "    cardinality: fixed",
    }
    included_lines = "\n".join(
        line for name, line in batching_lines.items() if name != missing_field
    )
    payload = f"""
data:
  batching:
{included_lines}
    packing:
      mode: none
    layout: padded
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
"""

    with pytest.raises(ValueError, match=rf"data\.batching\.{missing_field}.*explicit"):
        load_config(
            write_config_yaml(
                tmp_path,
                payload,
                ensure_explicit_batching=False,
            )
        )


def test_unknown_batching_key_is_reported_before_missing_axis(tmp_path: Path) -> None:
    payload = """
data:
  batching:
    strategy: fixed
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
"""

    with pytest.raises(ValueError, match=r"Unknown config keys at data\.batching: \['strategy'\]"):
        load_config(
            write_config_yaml(
                tmp_path,
                payload,
                ensure_explicit_batching=False,
            )
        )


def test_invalid_grouping_is_reported_before_bounded_field_combination(
    tmp_path: Path,
) -> None:
    payload = """
data:
  batching:
    grouping: nonsense
    cardinality: fixed
    packing:
      mode: none
    layout: padded
    buffer_size: 64
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
"""

    with pytest.raises(ValueError, match="Unsupported data.batching.grouping='nonsense'"):
        load_config(
            write_config_yaml(
                tmp_path,
                payload,
                ensure_explicit_batching=False,
            )
        )


def test_fixed_batching_needs_no_secondary_guard_config(tmp_path: Path) -> None:
    config = load_config(
        write_config_yaml(
            tmp_path,
            """
data:
  media_snapshot_id: fixture-v1
  batching:
    grouping: NONE
    cardinality: FIXED
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
""",
        )
    )

    assert config.data.batching.grouping == "none"
    assert config.data.batching.cardinality == "fixed"


@pytest.mark.parametrize(
    "legacy_field",
    [
        "mix_strategy: weighted",
        "shuffle: true",
        "prompt_sampling: {enabled: false}",
        "sequence_layout: {mode: padded}",
    ],
)
def test_legacy_flat_data_axes_are_rejected(tmp_path: Path, legacy_field: str) -> None:
    payload = f"""
data:
  {legacy_field}
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
"""

    with pytest.raises(ValueError, match="Unknown config keys at data"):
        load_config(write_config_yaml(tmp_path, payload))


@pytest.mark.parametrize(
    ("config_block", "message"),
    [
        (
            """
  batching:
    grouping: none
    cardinality: token_budget
    packing:
      mode: none
    layout: padded
""",
            "cardinality='token_budget'.*grouping='bounded_cost'",
        ),
        (
            """
  batching:
    grouping: none
    cardinality: fixed
    packing:
      mode: none
    layout: varlen
""",
            "layout='varlen'.*grouping='length'",
        ),
        (
            """
  batching:
    grouping: none
    cardinality: fixed
    packing:
      mode: greedy
    layout: padded
""",
            "packing.mode='greedy'.*grouping='length'",
        ),
    ],
)
def test_declared_future_batch_modes_fail_instead_of_silently_degrading(
    tmp_path: Path,
    config_block: str,
    message: str,
) -> None:
    payload = f"""
data:
{config_block}
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
"""

    with pytest.raises(ValueError, match=message):
        load_config(write_config_yaml(tmp_path, payload))


def test_length_grouping_config_is_normalized(tmp_path: Path) -> None:
    config = load_config(
        write_config_yaml(
            tmp_path,
            """
data:
  media_snapshot_id: fixture-v1
  max_length: 128
  batching:
    grouping: LENGTH
    cardinality: FIXED
    packing:
      mode: none
    layout: padded
    buffer_size: 64
    cost_cache_size: 1024
  schedule:
    mixing: concat
    shuffle: false
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
train:
  duration:
    unit: steps
    value: 2
""",
        )
    )

    assert config.data.batching.grouping == "length"
    assert config.data.batching.cardinality == "fixed"
    assert config.data.batching.buffer_size == 64
    assert config.data.batching.cost_cache_size == 1024


def test_qwen3vl_greedy_varlen_config_is_normalized(tmp_path: Path) -> None:
    config = load_config(
        write_config_yaml(
            tmp_path,
            """
model:
  model_type: qwen3vl
  attn_implementation: flash_attention_2
  torch_dtype: bfloat16
data:
  media_snapshot_id: fixture-v1
  max_length: 128
  batching:
    grouping: length
    cardinality: fixed
    packing:
      mode: greedy
    layout: varlen
    buffer_size: 64
    cost_cache_size: 1024
    resource_budgets:
      vision_patches: 4096
  schedule:
    mixing: concat
    shuffle: false
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
train:
  duration:
    unit: steps
    value: 2
  distributed:
    strategy: ddp
""",
        )
    )

    assert config.data.batching.packing.mode == "greedy"
    assert config.data.batching.layout == "varlen"
    assert config.data.batching.resource_budgets == {"vision_patches": 4096}


@pytest.mark.parametrize(
    ("data_fields", "message"),
    [
        ("  media_snapshot_id: fixture-v1\n", "data.max_length"),
        ("  max_length: 128\n", "media_snapshot_id"),
    ],
)
def test_length_grouping_requires_stable_cost_contract(
    tmp_path: Path,
    data_fields: str,
    message: str,
) -> None:
    payload = f"""
data:
{data_fields}  batching:
    grouping: length
    cardinality: fixed
    packing:
      mode: none
    layout: padded
  schedule:
    mixing: concat
    shuffle: false
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
train:
  duration:
    unit: steps
    value: 1
"""

    with pytest.raises(ValueError, match=message):
        load_config(write_config_yaml(tmp_path, payload))


def test_greedy_varlen_requires_vision_guard(tmp_path: Path) -> None:
    payload = """
model:
  model_type: qwen3vl
  attn_implementation: flash_attention_2
data:
  media_snapshot_id: fixture-v1
  max_length: 128
  batching:
    grouping: length
    cardinality: fixed
    packing:
      mode: greedy
    layout: varlen
  schedule:
    mixing: concat
    shuffle: false
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
train:
  duration:
    unit: steps
    value: 1
"""

    with pytest.raises(ValueError, match="resource_budgets.vision_patches"):
        load_config(write_config_yaml(tmp_path, payload))


def test_bounded_cost_grouping_config_is_normalized(tmp_path: Path) -> None:
    config = load_config(
        write_config_yaml(
            tmp_path,
            """
data:
  media_snapshot_id: fixture-v1
  batching:
    grouping: BOUNDED_COST
    cardinality: FIXED
    buffer_size: 64
    cost_cache_size: 1024
    max_tokens_per_microbatch: 512
    resource_budgets:
      vision_patches: 1024
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
    assert batching.grouping == "bounded_cost"
    assert batching.cardinality == "fixed"
    assert batching.buffer_size == 64
    assert batching.cost_cache_size == 1024
    assert batching.max_tokens_per_microbatch == 512
    assert batching.resource_budgets == {"vision_patches": 1024}
    assert not hasattr(config.train, "optimizer_batch")


def test_bounded_token_budget_cardinality_is_normalized(tmp_path: Path) -> None:
    config = load_config(
        write_config_yaml(
            tmp_path,
            """
data:
  media_snapshot_id: fixture-v1
  batching:
    grouping: BOUNDED_COST
    cardinality: TOKEN_BUDGET
    buffer_size: 64
    max_tokens_per_microbatch: 512
    resource_budgets:
      vision_patches: 1024
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

    assert config.data.batching.cardinality == "token_budget"
    assert config.train.per_device_train_batch_size == 2


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("grouping", "unknown", "Unsupported data.batching.grouping"),
        ("cardinality", "unknown", "Unsupported data.batching.cardinality"),
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
    bounded_context = ""
    if field in {"buffer_size", "cost_cache_size"}:
        bounded_context = (
            "    grouping: bounded_cost\n"
            "    cardinality: fixed\n"
            "    max_tokens_per_microbatch: 512\n"
        )
    payload = f"""
data:
  batching:
{bounded_context}\
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
  media_snapshot_id: fixture-v1
  batching:
    grouping: bounded_cost
    cardinality: fixed
    packing:
      mode: none
    layout: padded
    max_tokens_per_microbatch: 512
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
        ("max_tokens_per_microbatch: 0", "max_tokens_per_microbatch"),
        ("buffer_size: 64", "max_tokens_per_microbatch"),
        (
            "max_tokens_per_microbatch: 512\n    resource_budgets:\n      vision_patches: 0",
            "vision_patches",
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
  media_snapshot_id: fixture-v1
  batching:
    grouping: bounded_cost
    cardinality: fixed
    packing:
      mode: none
    layout: padded
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
    grouping: none
    cardinality: fixed
    packing:
      mode: none
    layout: padded
    max_tokens_per_microbatch: 512
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
"""
    with pytest.raises(ValueError, match="only valid when grouping='bounded_cost'"):
        load_config(write_config_yaml(tmp_path, payload))


@pytest.mark.parametrize(
    "planned_only_field",
    [
        "buffer_size: 64",
        "cost_cache_size: 65536",
    ],
)
def test_unplanned_grouping_rejects_explicit_planning_fields(
    tmp_path: Path,
    planned_only_field: str,
) -> None:
    payload = f"""
data:
  batching:
    grouping: none
    cardinality: fixed
    packing:
      mode: none
    layout: padded
    {planned_only_field}
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
"""

    with pytest.raises(ValueError, match="require grouping='length'.*grouping='bounded_cost'"):
        load_config(write_config_yaml(tmp_path, payload))


def test_unplanned_packing_fields_are_not_reserved_in_schema(tmp_path: Path) -> None:
    payload = """
data:
  batching:
    grouping: none
    cardinality: fixed
    packing:
      mode: none
      target_length: 4096
    layout: padded
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
"""

    with pytest.raises(
        ValueError,
        match=r"Unknown config keys at data\.batching\.packing: \['target_length'\]",
    ):
        load_config(write_config_yaml(tmp_path, payload))


def test_unknown_batch_resource_fails_during_config_validation(tmp_path: Path) -> None:
    payload = """
data:
  media_snapshot_id: fixture-v1
  batching:
    grouping: bounded_cost
    cardinality: fixed
    packing:
      mode: none
    layout: padded
    max_tokens_per_microbatch: 512
    resource_budgets:
      audio_frames: 1024
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
train:
  duration:
    unit: steps
    value: 2
"""

    with pytest.raises(ValueError, match="Unsupported.*resource.*audio_frames"):
        load_config(write_config_yaml(tmp_path, payload))


def test_bounded_weighted_mixing_requires_horizon_independent_shuffle(
    tmp_path: Path,
) -> None:
    payload = """
data:
  media_snapshot_id: fixture-v1
  schedule:
    mixing: weighted
    shuffle: false
  batching:
    grouping: bounded_cost
    cardinality: fixed
    packing:
      mode: none
    layout: padded
    max_tokens_per_microbatch: 512
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
    grouping: bounded_cost
    cardinality: fixed
    packing:
      mode: none
    layout: padded
    max_tokens_per_microbatch: 512
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
    grouping: bounded_cost
    cardinality: fixed
    packing:
      mode: none
    layout: padded
    max_tokens_per_microbatch: 512
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
    grouping: none
    cardinality: fixed
    packing:
      mode: none
    layout: padded
{legacy_block}
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
"""
    with pytest.raises(ValueError, match=f"Unknown config keys at {path}"):
        load_config(write_config_yaml(tmp_path, payload))


@pytest.mark.parametrize("strategy", ["fixed", "cost_aware", "dynamic_cost_aware"])
def test_removed_batching_strategy_key_fails_loudly(
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
    with pytest.raises(ValueError, match="Unknown config keys at data.batching"):
        load_config(write_config_yaml(tmp_path, payload))


def test_removed_optimizer_batch_config_fails_loudly(tmp_path: Path) -> None:
    payload = """
data:
  batching:
    grouping: none
    cardinality: fixed
    packing:
      mode: none
    layout: padded
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
