from __future__ import annotations

from peft import PeftModel

from shaft.config import FinetuneConfig
from shaft.model import build_model_meta
from shaft.model.finetune import apply_finetune_strategy, summarize_finetune
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
