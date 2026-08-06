from __future__ import annotations

from unittest.mock import patch
from types import SimpleNamespace

from peft import PeftModel, get_peft_model_state_dict
import torch

from shaft.config import FinetuneConfig
from shaft.model import build_model_meta
from shaft.model.finetune import (
    apply_resolved_finetune_plan,
    apply_finetune_strategy,
    load_peft_checkpoint,
    summarize_finetune,
)
from shaft.model.finetune_plan import build_resolved_finetune_plan
from shaft.model.generation import export_model_cache
from shaft.model.smoke_vlm import SmokeVLMConfig, SmokeVLMModel


def _build_model():
    return SmokeVLMModel(SmokeVLMConfig())


def _build_adapter():
    return build_model_meta("smoke_vlm").resolve_adapter(model_name_or_path="models/Smoke-VLM")


def test_full_mode_trainable() -> None:
    model = apply_finetune_strategy(
        _build_model(),
        FinetuneConfig(mode="full"),
        model_adapter=_build_adapter(),
    )
    summary = summarize_finetune(model, "full")
    assert summary.trainable_params == summary.total_params


def test_lora_mode_trainable() -> None:
    model = apply_finetune_strategy(
        _build_model(),
        FinetuneConfig(mode="lora", target_modules=["all-linear"]),
        model_adapter=_build_adapter(),
    )
    assert isinstance(model, PeftModel)
    summary = summarize_finetune(model, "lora")
    assert summary.trainable_params > 0
    assert summary.trainable_params < summary.total_params


def test_dora_mode_trainable() -> None:
    model = apply_finetune_strategy(
        _build_model(),
        FinetuneConfig(mode="dora", target_modules=["all-linear"]),
        model_adapter=_build_adapter(),
    )
    assert isinstance(model, PeftModel)
    summary = summarize_finetune(model, "dora")
    assert summary.trainable_params > 0
    assert summary.trainable_params < summary.total_params


def test_qlora_mode_trainable_for_smoke_model() -> None:
    model = apply_finetune_strategy(
        _build_model(),
        FinetuneConfig(mode="qlora", target_modules=["all-linear"], qlora_load_in_4bit=False),
        model_adapter=_build_adapter(),
    )
    assert isinstance(model, PeftModel)
    summary = summarize_finetune(model, "qlora")
    assert summary.trainable_params > 0
    assert summary.trainable_params < summary.total_params


def test_training_finetune_disables_use_cache_even_without_gradient_checkpointing() -> None:
    model = _build_model()
    model.config.use_cache = True
    model.model = SimpleNamespace(
        language_model=SimpleNamespace(config=SimpleNamespace(use_cache=True))
    )
    model = apply_finetune_strategy(
        model,
        FinetuneConfig(mode="full"),
        model_adapter=_build_adapter(),
        gradient_checkpointing=False,
    )
    assert model.config.use_cache is False
    assert model.model.language_model.config.use_cache is False


def test_training_cache_defaults_are_restored_only_while_exporting_peft() -> None:
    model = _build_model()
    model.config.use_cache = True
    model.generation_config.use_cache = None
    wrapped = apply_finetune_strategy(
        model,
        FinetuneConfig(mode="lora", target_modules=["all-linear"]),
        model_adapter=_build_adapter(),
    )

    assert model.config.use_cache is False
    assert model.generation_config.use_cache is False
    with export_model_cache(wrapped):
        assert model.config.use_cache is True
        assert model.generation_config.use_cache is None
    assert model.config.use_cache is False
    assert model.generation_config.use_cache is False


def test_peft_checkpoint_load_is_exact_before_distributed_wrapping(tmp_path) -> None:
    torch.manual_seed(13)
    source = apply_finetune_strategy(
        _build_model(),
        FinetuneConfig(mode="lora", target_modules=["all-linear"]),
        model_adapter=_build_adapter(),
    )
    for name, parameter in source.named_parameters():
        if "lora_" in name:
            parameter.data.fill_(0.125 if "lora_A" in name else -0.25)
    checkpoint = tmp_path / "checkpoint-1"
    source.save_pretrained(checkpoint)

    torch.manual_seed(29)
    target = apply_finetune_strategy(
        _build_model(),
        FinetuneConfig(mode="lora", target_modules=["all-linear"]),
        model_adapter=_build_adapter(),
    )
    tensor_count, parameter_count = load_peft_checkpoint(target, checkpoint)
    source_state = get_peft_model_state_dict(source)
    target_state = get_peft_model_state_dict(target)

    assert tensor_count == len(source_state)
    assert parameter_count == sum(tensor.numel() for tensor in source_state.values())
    assert source_state.keys() == target_state.keys()
    assert all(torch.equal(source_state[name], target_state[name]) for name in source_state)


def test_finetune_summary_uses_deepspeed_global_parameter_counts() -> None:
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.empty(0), requires_grad=True)
    model.weight.ds_numel = 16
    model.frozen = torch.nn.Parameter(torch.empty(0), requires_grad=False)
    model.frozen.ds_shape = (2, 3)

    summary = summarize_finetune(model, "full")

    assert summary.total_params == 22
    assert summary.trainable_params == 16


def test_qlora_gradient_checkpointing_is_forwarded_to_prepare_model_for_kbit_training() -> None:
    model = _build_model()
    model.config.use_cache = True
    adapter = _build_adapter()
    finetune = FinetuneConfig(mode="qlora", target_modules=["all-linear"], qlora_load_in_4bit=False)
    plan = build_resolved_finetune_plan(model, finetune, model_adapter=adapter)

    with patch("shaft.model.finetune.prepare_model_for_kbit_training", side_effect=lambda m, **_: m) as mocked:
        wrapped = apply_resolved_finetune_plan(
            model,
            plan,
            finetune=finetune,
            gradient_checkpointing=True,
        )

    assert isinstance(wrapped, PeftModel)
    mocked.assert_called_once()
    _, kwargs = mocked.call_args
    assert kwargs["use_gradient_checkpointing"] is True
    assert model.config.use_cache is False
