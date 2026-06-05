from __future__ import annotations

from unittest.mock import patch
from types import SimpleNamespace

from peft import PeftModel
import torch

from shaft.config import FinetuneConfig
from shaft.model import build_model_meta
from shaft.model.finetune import apply_resolved_finetune_plan, apply_finetune_strategy, summarize_finetune
from shaft.model.finetune_plan import build_resolved_finetune_plan
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


def test_gradient_checkpointing_disables_use_cache_for_full_mode() -> None:
    model = _build_model()
    model.config.use_cache = True
    model.model = SimpleNamespace(
        language_model=SimpleNamespace(config=SimpleNamespace(use_cache=True))
    )
    model = apply_finetune_strategy(
        model,
        FinetuneConfig(mode="full"),
        model_adapter=_build_adapter(),
        gradient_checkpointing=True,
    )
    assert model.config.use_cache is False
    assert model.model.language_model.config.use_cache is False


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
