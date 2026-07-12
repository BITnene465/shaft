from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from shaft.config import load_config
from shaft.model import build_model_tokenizer_processor
from shaft.pipeline.training_args import build_hf_training_args, resolve_step_sample_budget
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

    assert resolve_step_sample_budget(config, world_size=8) == 640

    config.train.duration.unit = "epochs"
    config.train.duration.value = 2
    assert resolve_step_sample_budget(config, world_size=8) is None


def test_bounded_step_duration_does_not_build_finite_sample_plan(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    config.data.batching.strategy = "bounded_cost_aware"
    config.data.batching.buffer_size = 64
    config.data.batching.max_samples_per_microbatch = 4
    config.data.batching.max_padded_tokens = 512
    config.train.duration.value = 10
    config.train.per_device_train_batch_size = 2
    config.train.gradient_accumulation_steps = 2

    assert resolve_step_sample_budget(config, world_size=2) is None
    args = build_hf_training_args(config)
    assert args.accelerator_config.even_batches is True
    assert args.accelerator_config.split_batches is False
    assert args.accelerator_config.dispatch_batches is False


def test_bounded_sample_cap_does_not_change_schedule_horizon(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    config.data.batching.strategy = "bounded_cost_aware"
    config.data.batching.buffer_size = 8
    config.data.batching.max_samples_per_microbatch = None
    config.data.batching.max_padded_tokens = 512
    config.train.duration.value = 10
    config.train.per_device_train_batch_size = 3
    config.train.gradient_accumulation_steps = 2

    assert resolve_step_sample_budget(config, world_size=2) is None


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


def test_qwen36_sft_27b_fsdp_example_config_loads() -> None:
    config = load_config(Path("configs/train/qwen36_sft_27b_fsdp_example.yaml"))

    assert config.model.model_type == "qwen36vl"
    assert config.model.template == "qwen35vl"
    assert config.model.finetune.mode == "lora"
    assert config.train.distributed.strategy == "fsdp"

    args = build_hf_training_args(config)
    assert _fsdp_enabled(args.fsdp) is True
    assert args.fsdp_config["activation_checkpointing"] is False
    assert args.gradient_checkpointing is True
    assert args.fsdp_config["transformer_layer_cls_to_wrap"] == [
        "Qwen3_5DecoderLayer",
        "Qwen3_5VisionBlock",
    ]


def test_fsdp_auto_layers_require_model_default(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    config.model.model_type = "unknown_model"
    config.train.distributed.strategy = "fsdp"
    config.train.distributed.fsdp.transformer_layer_cls_to_wrap = ["auto"]

    with pytest.raises(ValueError, match=r"transformer_layer_cls_to_wrap=\['auto'\]"):
        build_hf_training_args(config)
