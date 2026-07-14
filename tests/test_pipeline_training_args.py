from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from shaft.data import ShaftBatchPlanningSpec
from shaft.model import build_model_tokenizer_processor
from shaft.pipeline.training_args import build_hf_training_args
from shaft.training.batch_planning import (
    build_batch_contract,
    build_batching_run_metadata,
)
from shaft.training.reproducibility import initialize_training_randomness
from tests.support.pipeline import fsdp_enabled as _fsdp_enabled
from tests.support.pipeline import fsdp_option_values as _fsdp_option_values
from tests.support.pipeline import write_sft_pipeline_config as _write_config


pytestmark = pytest.mark.component


def test_build_hf_training_args_supports_gradient_checkpointing(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    config.train.gradient_checkpointing = True

    args = build_hf_training_args(config)

    assert args.gradient_checkpointing is True
    assert args.average_tokens_across_devices is True


def test_build_hf_training_args_exposes_full_determinism(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    config.train.full_determinism = True

    args = build_hf_training_args(config)

    assert args.full_determinism is True


def test_initialize_training_randomness_dispatches_full_determinism() -> None:
    with patch(
        "shaft.training.reproducibility.enable_full_determinism"
    ) as enable_determinism:
        with patch("shaft.training.reproducibility.set_seed") as set_seed:
            initialize_training_randomness(
                seed=23,
                full_determinism=True,
            )
    enable_determinism.assert_called_once_with(23)
    set_seed.assert_not_called()


@pytest.mark.parametrize("finetune_mode", ["full", "lora"])
def test_training_seed_reproduces_fresh_smoke_model_and_adapter_initialization(
    tmp_path: Path,
    finetune_mode: str,
) -> None:
    config = _write_config(tmp_path)
    config.model.model_type = "smoke_vlm"
    config.model.finetune.mode = finetune_mode
    config.model.finetune.target_modules = ["all-linear"]
    config.experiment.seed = 29
    training_args = build_hf_training_args(config)

    initialize_training_randomness(
        seed=training_args.seed,
        full_determinism=training_args.full_determinism,
    )
    first_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in build_model_tokenizer_processor(config).model.state_dict().items()
    }
    _ = torch.rand(17)
    initialize_training_randomness(
        seed=training_args.seed,
        full_determinism=training_args.full_determinism,
    )
    second_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in build_model_tokenizer_processor(config).model.state_dict().items()
    }

    assert first_state.keys() == second_state.keys()
    for name in first_state:
        assert torch.equal(first_state[name], second_state[name]), name


def test_build_hf_training_args_uses_warmup_steps_ratio(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    config.train.warmup_ratio = 0.03
    config.train.duration.unit = "steps"
    config.train.duration.value = 10000

    args = build_hf_training_args(config)

    assert args.warmup_ratio is None
    assert args.warmup_steps == 300


def test_build_hf_training_args_keeps_warmup_ratio_for_epoch_duration(
    tmp_path: Path,
) -> None:
    config = _write_config(tmp_path)
    config.train.warmup_ratio = 0.03
    config.train.duration.unit = "epochs"
    config.train.duration.value = 3

    args = build_hf_training_args(config)

    assert args.warmup_ratio == 0.03
    assert args.warmup_steps == 0.03
    assert args.max_steps == -1
    assert args.num_train_epochs == 3


def test_step_duration_resolves_exact_global_sample_budget(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    config.train.duration.unit = "steps"
    config.train.duration.value = 10
    config.train.per_device_train_batch_size = 2
    config.train.gradient_accumulation_steps = 4

    args = build_hf_training_args(config)
    contract = build_batch_contract(config=config, training_args=args)

    assert contract.finite_sample_plan_size(max_steps=10) == 80

    eight_rank_contract = replace(contract, data_world_size=8)
    assert eight_rank_contract.finite_sample_plan_size(max_steps=10) == 640

    config.train.duration.unit = "epochs"
    config.train.duration.value = 2
    assert eight_rank_contract.finite_sample_plan_size(max_steps=-1) is None


def test_resolved_batch_contract_uses_per_device_batch_as_physical_truth(
    tmp_path: Path,
) -> None:
    config = _write_config(tmp_path)
    config.train.per_device_train_batch_size = 3
    config.train.gradient_accumulation_steps = 2
    args = build_hf_training_args(config)

    contract = build_batch_contract(config=config, training_args=args)

    assert contract.per_device_microbatch_size == 3
    assert contract.global_pack_count == 3
    assert contract.optimizer_pack_count == 6
    assert contract.cardinality == "fixed"


def test_length_contract_keeps_physical_batch_slots_and_derives_local_token_capacity(
    tmp_path: Path,
) -> None:
    config = _write_config(tmp_path)
    config.data.max_length = 128
    config.data.media_snapshot_id = "fixture-v1"
    config.data.batching.grouping = "length"
    config.data.batching.cardinality = "fixed"
    config.data.batching.packing.mode = "greedy"
    config.data.batching.layout = "varlen"
    config.data.batching.buffer_size = 16
    config.data.batching.resource_budgets = {"vision_patches": 4096}
    config.train.per_device_train_batch_size = 3
    config.train.gradient_accumulation_steps = 2
    args = build_hf_training_args(config)

    contract = build_batch_contract(config=config, training_args=args)

    assert contract.is_planned is True
    assert contract.is_bounded is False
    assert contract.per_device_microbatch_size == 3
    assert contract.local_pack_count_bounds == (3, 3)
    assert contract.max_sequence_length == 128
    assert contract.max_tokens_per_microbatch is None
    assert contract.local_token_capacity == 384
    assert contract.global_pack_count == 3
    assert contract.finite_sample_plan_size(max_steps=10) is None


def test_length_contract_rejects_planner_buffer_smaller_than_one_global_microbatch(
    tmp_path: Path,
) -> None:
    config = _write_config(tmp_path)
    config.data.max_length = 128
    config.data.batching.grouping = "length"
    config.data.batching.buffer_size = 1
    config.train.per_device_train_batch_size = 2
    args = build_hf_training_args(config)

    with pytest.raises(ValueError, match="one complete global microbatch"):
        build_batch_contract(config=config, training_args=args)


def test_resolved_token_budget_contract_uses_per_device_batch_as_upper_bound(
    tmp_path: Path,
) -> None:
    config = _write_config(tmp_path)
    config.data.batching.grouping = "bounded_cost"
    config.data.batching.cardinality = "token_budget"
    config.data.batching.buffer_size = 8
    config.data.batching.max_tokens_per_microbatch = 512
    config.data.media_snapshot_id = "fixture-v1"
    config.train.duration.unit = "steps"
    config.train.duration.value = 2
    config.train.per_device_train_batch_size = 2
    config.train.gradient_accumulation_steps = 4
    args = build_hf_training_args(config)

    contract = build_batch_contract(config=config, training_args=args)

    assert contract.cardinality == "token_budget"
    assert contract.per_device_microbatch_size == 2
    assert contract.local_pack_count_bounds == (1, 2)
    assert contract.global_pack_count_bounds == (1, 2)
    assert contract.optimizer_pack_count_bounds == (4, 8)
    with pytest.raises(ValueError, match="not exact"):
        _ = contract.global_pack_count
    with pytest.raises(ValueError, match="not exact"):
        _ = contract.optimizer_pack_count


def test_resolved_bounded_contract_rejects_too_small_buffer_before_data_loading(
    tmp_path: Path,
) -> None:
    config = _write_config(tmp_path)
    config.data.batching.grouping = "bounded_cost"
    config.data.batching.cardinality = "fixed"
    config.data.batching.buffer_size = 1
    config.data.batching.max_tokens_per_microbatch = 1024
    config.data.media_snapshot_id = "fixture-v1"
    config.train.per_device_train_batch_size = 2
    args = build_hf_training_args(config)

    with pytest.raises(ValueError, match="one complete global microbatch"):
        build_batch_contract(config=config, training_args=args)


def test_batching_metadata_rejects_a_spec_that_drifted_from_resolved_contract(
    tmp_path: Path,
) -> None:
    config = _write_config(tmp_path)
    config.data.media_snapshot_id = "fixture-v1"
    config.data.batching.grouping = "bounded_cost"
    config.data.batching.cardinality = "fixed"
    config.data.batching.buffer_size = 64
    config.data.batching.max_tokens_per_microbatch = 1024
    config.data.batching.resource_budgets = {"vision_patches": 2048}
    config.train.duration.unit = "steps"
    config.train.duration.value = 2
    config.train.per_device_train_batch_size = 1
    args = build_hf_training_args(config)
    contract = build_batch_contract(config=config, training_args=args)
    drifted = ShaftBatchPlanningSpec(
        data_world_size=contract.data_world_size,
        buffer_size=64,
        per_device_microbatch_size=2,
        max_tokens_per_microbatch=1024,
        resource_budgets=(("vision_patches", 2048),),
        seed=42,
        sample_schedule_fingerprint="schedule-v1",
        cost_fingerprint="cost-v1",
    )

    with pytest.raises(ValueError, match="differs from the resolved batch contract"):
        build_batching_run_metadata(
            config=config,
            training_args=args,
            planning_spec=drifted,
            batch_contract=contract,
        )


def test_bounded_step_duration_does_not_build_finite_sample_plan(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    config.data.batching.grouping = "bounded_cost"
    config.data.batching.cardinality = "fixed"
    config.data.batching.buffer_size = 64
    config.data.batching.max_tokens_per_microbatch = 512
    config.train.duration.value = 10
    config.train.per_device_train_batch_size = 2
    config.train.gradient_accumulation_steps = 2

    args = build_hf_training_args(config)
    contract = build_batch_contract(config=config, training_args=args)
    assert contract.finite_sample_plan_size(max_steps=2) is None
    args = build_hf_training_args(config)
    assert args.accelerator_config.even_batches is True
    assert args.accelerator_config.split_batches is False
    assert args.accelerator_config.dispatch_batches is False
    assert args.per_device_train_batch_size == 2


def test_bounded_per_device_batch_does_not_change_schedule_horizon(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    config.data.batching.grouping = "bounded_cost"
    config.data.batching.cardinality = "fixed"
    config.data.batching.buffer_size = 8
    config.data.batching.max_tokens_per_microbatch = 512
    config.train.duration.value = 10
    config.train.per_device_train_batch_size = 3
    config.train.gradient_accumulation_steps = 2

    args = build_hf_training_args(config)
    contract = build_batch_contract(config=config, training_args=args)
    assert contract.finite_sample_plan_size(max_steps=2) is None


def test_build_hf_training_args_supports_fsdp_strategy(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    config.train.distributed.strategy = "fsdp"
    config.train.distributed.fsdp.transformer_layer_cls_to_wrap = ["auto"]

    args = build_hf_training_args(config)

    assert _fsdp_enabled(args.fsdp) is True
    option_values = _fsdp_option_values(args.fsdp)
    if option_values:
        assert option_values == ["full_shard", "auto_wrap"]
    assert args.fsdp_config["transformer_layer_cls_to_wrap"] == [
        "Qwen3VLTextDecoderLayer",
        "Qwen3VLVisionBlock",
    ]
    assert args.fsdp_config["activation_checkpointing"] is True
    assert args.fsdp_config["state_dict_type"] == "full_state_dict"


def test_fsdp_activation_checkpointing_disables_trainer_gradient_checkpointing(
    tmp_path: Path,
) -> None:
    config = _write_config(tmp_path)
    config.train.gradient_checkpointing = True
    config.train.distributed.strategy = "fsdp"
    config.train.distributed.fsdp.activation_checkpointing = True
    config.train.distributed.fsdp.transformer_layer_cls_to_wrap = ["auto"]

    args = build_hf_training_args(config)

    assert args.gradient_checkpointing is False
    assert args.fsdp_config["activation_checkpointing"] is True


def test_build_hf_training_args_resolves_qwen36vl_fsdp_auto_layers(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    config.model.model_type = "qwen36vl"
    config.model.model_name_or_path = "models/Qwen3.6-27B"
    config.train.distributed.strategy = "fsdp"
    config.train.distributed.fsdp.transformer_layer_cls_to_wrap = ["auto"]

    args = build_hf_training_args(config)

    assert _fsdp_enabled(args.fsdp) is True
    assert args.fsdp_config["transformer_layer_cls_to_wrap"] == [
        "Qwen3_5DecoderLayer",
        "Qwen3_5VisionBlock",
    ]


def test_fsdp_auto_layers_consume_descriptor_driven_model_plan(tmp_path: Path) -> None:
    model_dir = tmp_path / "custom-qwen-moe"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5_moe",
                "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            }
        ),
        encoding="utf-8",
    )
    config = _write_config(tmp_path)
    config.model.model_type = "qwen36vl"
    config.model.model_name_or_path = str(model_dir)
    config.train.distributed.strategy = "fsdp"
    config.train.distributed.fsdp.transformer_layer_cls_to_wrap = ["auto"]
    from shaft.model import resolve_model_plan

    model_plan = resolve_model_plan(config)
    args = build_hf_training_args(config, resolved_model_plan=model_plan)

    assert model_plan.model_adapter.group_name == "moe"
    assert args.fsdp_config["transformer_layer_cls_to_wrap"] == [
        "Qwen3_5MoeDecoderLayer",
        "Qwen3_5MoeVisionBlock",
    ]


def test_fsdp_auto_layers_follow_full_init_checkpoint_descriptor(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-moe"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5_moe",
                "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            }
        ),
        encoding="utf-8",
    )
    config = _write_config(tmp_path)
    config.model.model_type = "qwen36vl"
    config.model.model_name_or_path = "models/Qwen3.6-27B"
    config.train.init_from_checkpoint = str(checkpoint)
    config.train.distributed.strategy = "fsdp"
    config.train.distributed.fsdp.transformer_layer_cls_to_wrap = ["auto"]
    from shaft.model import resolve_model_plan

    model_plan = resolve_model_plan(
        config,
        init_from_checkpoint=config.train.init_from_checkpoint,
    )
    args = build_hf_training_args(config, resolved_model_plan=model_plan)

    assert model_plan.model_adapter.group_name == "moe"
    assert args.fsdp_config["transformer_layer_cls_to_wrap"] == [
        "Qwen3_5MoeDecoderLayer",
        "Qwen3_5MoeVisionBlock",
    ]


def test_fsdp_auto_layers_require_model_default(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    config.model.model_type = "unknown_model"
    config.train.distributed.strategy = "fsdp"
    config.train.distributed.fsdp.transformer_layer_cls_to_wrap = ["auto"]

    with pytest.raises(ValueError, match=r"transformer_layer_cls_to_wrap=\['auto'\]"):
        build_hf_training_args(config)
