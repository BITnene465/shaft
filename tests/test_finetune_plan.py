from __future__ import annotations

from unittest.mock import patch

from peft import PeftModel

from shaft.config import FinetuneConfig, FreezeConfig, RuntimeConfig
from shaft.model import build_model_meta, build_model_tokenizer_processor
from shaft.model.finetune_plan import build_resolved_finetune_plan
from shaft.model.smoke_vlm import SmokeVLMConfig, SmokeVLMModel


def _build_smoke_model() -> SmokeVLMModel:
    return SmokeVLMModel(SmokeVLMConfig())


def _build_smoke_adapter():
    return build_model_meta("smoke_vlm").resolve_adapter(model_name_or_path="models/smoke-vlm")


def test_build_resolved_finetune_plan_for_full_mode_resolves_parameter_names() -> None:
    model = _build_smoke_model()
    adapter = _build_smoke_adapter()
    plan = build_resolved_finetune_plan(
        model,
        FinetuneConfig(mode="full", freeze=FreezeConfig(groups=["language_model"])),
        model_adapter=adapter,
    )

    assert plan.mode == "full"
    assert "embed_tokens.weight" in plan.parameter_plan.frozen_parameter_names
    assert "proj.weight" in plan.parameter_plan.frozen_parameter_names
    assert "lm_head.weight" in plan.parameter_plan.trainable_parameter_names
    assert plan.adapter_plan is None


def test_build_resolved_finetune_plan_for_lora_mode_resolves_adapter_signature() -> None:
    model = _build_smoke_model()
    adapter = _build_smoke_adapter()
    plan = build_resolved_finetune_plan(
        model,
        FinetuneConfig(
            mode="lora",
            target_modules=["all-linear"],
            freeze=FreezeConfig(trainable_prefixes=["lm_head"]),
        ),
        model_adapter=adapter,
    )

    assert plan.adapter_plan is not None
    assert plan.adapter_plan.resolved_target_modules == ("proj",)
    assert plan.adapter_plan.modules_to_save == ("lm_head",)
    assert plan.adapter_plan.peft_signature.modules_to_save == ("lm_head",)


def test_loader_populates_finetune_plan_on_artifacts() -> None:
    cfg = RuntimeConfig()
    cfg.model.model_type = "smoke_vlm"
    cfg.model.model_name_or_path = "models/smoke-vlm"
    cfg.model.finetune.mode = "lora"
    cfg.model.finetune.target_modules = ["all-linear"]
    cfg.model.finetune.freeze.trainable_prefixes = ["lm_head"]

    artifacts = build_model_tokenizer_processor(cfg)

    assert isinstance(artifacts.model, PeftModel)
    assert artifacts.finetune_plan is not None
    assert artifacts.finetune_plan.adapter_plan is not None
    assert artifacts.finetune_plan.adapter_plan.peft_signature.modules_to_save == ("lm_head",)


def test_builder_adapter_init_prefers_resolved_finetune_plan_for_expected_signature(tmp_path) -> None:
    cfg_src = RuntimeConfig()
    cfg_src.model.model_type = "smoke_vlm"
    cfg_src.model.model_name_or_path = "models/smoke-vlm"
    cfg_src.model.finetune.mode = "lora"
    cfg_src.model.finetune.target_modules = ["all-linear"]
    cfg_src.model.finetune.freeze.trainable_prefixes = ["lm_head"]
    artifacts_src = build_model_tokenizer_processor(cfg_src)
    adapter_dir = tmp_path / "adapter"
    artifacts_src.model.save_pretrained(adapter_dir)

    cfg_tgt = RuntimeConfig()
    cfg_tgt.model.model_type = "smoke_vlm"
    cfg_tgt.model.model_name_or_path = "models/smoke-vlm"
    cfg_tgt.model.finetune.mode = "lora"
    cfg_tgt.model.finetune.target_modules = ["all-linear"]
    cfg_tgt.model.finetune.freeze.trainable_prefixes = ["lm_head"]

    with patch("shaft.model.builder._resolve_default_peft_config", side_effect=AssertionError("fallback used")):
        artifacts_tgt = build_model_tokenizer_processor(cfg_tgt, init_from_checkpoint=str(adapter_dir))

    assert isinstance(artifacts_tgt.model, PeftModel)
