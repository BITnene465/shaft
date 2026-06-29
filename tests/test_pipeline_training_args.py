from __future__ import annotations

from pathlib import Path

import pytest

from shaft.config import load_config
from shaft.pipeline.training_args import (
    _build_deepspeed_arg,
    _build_fsdp_args,
    build_hf_training_args,
)
from tests.support.pipeline import fsdp_enabled as _fsdp_enabled
from tests.support.pipeline import fsdp_option_values as _fsdp_option_values
from tests.support.pipeline import write_sft_pipeline_config as _write_config


pytestmark = pytest.mark.component


def test_build_hf_training_args_supports_gradient_checkpointing(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    config.train.gradient_checkpointing = True

    args = build_hf_training_args(config)

    assert args.gradient_checkpointing is True


def test_build_hf_training_args_uses_warmup_steps_ratio(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    config.train.warmup_ratio = 0.03
    config.train.max_steps = 10000

    args = build_hf_training_args(config)

    assert args.warmup_ratio is None
    assert args.warmup_steps == 300


def test_build_hf_training_args_keeps_warmup_ratio_when_max_steps_unknown(
    tmp_path: Path,
) -> None:
    config = _write_config(tmp_path)
    config.train.warmup_ratio = 0.03
    config.train.max_steps = -1

    args = build_hf_training_args(config)

    assert args.warmup_ratio == 0.03
    assert args.warmup_steps == 0.03


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

    try:
        _build_fsdp_args(config)
    except ValueError as exc:
        message = str(exc)
    else:  # pragma: no cover - defensive assertion path
        raise AssertionError("FSDP auto layer resolution should require a registered default")

    assert "transformer_layer_cls_to_wrap=['auto']" in message


def test_deepspeed_training_arg_prefers_inline_config(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    config.train.distributed.strategy = "deepspeed"
    config.train.distributed.deepspeed.config_path = "configs/deepspeed/zero3_bf16.json"
    config.train.distributed.deepspeed.config = {"zero_optimization": {"stage": 3}}

    assert _build_deepspeed_arg(config) == {"zero_optimization": {"stage": 3}}


def test_deepspeed_training_arg_uses_config_path(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    config.train.distributed.strategy = "deepspeed"
    config.train.distributed.deepspeed.config_path = "configs/deepspeed/zero3_bf16.json"

    assert _build_deepspeed_arg(config) == "configs/deepspeed/zero3_bf16.json"


def test_build_hf_training_args_supports_deepspeed_strategy(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    config.train.distributed.strategy = "deepspeed"
    config.train.distributed.deepspeed.config = {
        "bf16": {"enabled": "auto"},
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "train_batch_size": "auto",
        "zero_optimization": {"stage": 2},
    }

    args = build_hf_training_args(config)

    assert args.deepspeed == config.train.distributed.deepspeed.config
    assert getattr(args, "hf_deepspeed_config", None) is not None
    assert _fsdp_enabled(args.fsdp) is False


def test_build_hf_training_args_resets_deepspeed_state_for_non_deepspeed(
    tmp_path: Path,
) -> None:
    from transformers.integrations.deepspeed import deepspeed_config

    deepspeed_config_payload = {
        "bf16": {"enabled": "auto"},
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "train_batch_size": "auto",
        "zero_optimization": {"stage": 2},
    }
    deepspeed_dir = tmp_path / "deepspeed"
    deepspeed_dir.mkdir()
    deepspeed_runtime = _write_config(deepspeed_dir)
    deepspeed_runtime.train.distributed.strategy = "deepspeed"
    deepspeed_runtime.train.distributed.deepspeed.config = deepspeed_config_payload
    _ = build_hf_training_args(deepspeed_runtime)
    assert deepspeed_config()["zero_optimization"]["stage"] == 2

    ddp_dir = tmp_path / "ddp"
    ddp_dir.mkdir()
    ddp_runtime = _write_config(ddp_dir)
    ddp_args = build_hf_training_args(ddp_runtime)

    assert ddp_args.deepspeed is None
    assert deepspeed_config() is None
