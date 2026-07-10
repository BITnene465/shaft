from __future__ import annotations

from unittest.mock import patch

import pytest

from shaft.model import (
    ModelCapabilities,
    ModelGroup,
    ModelMeta,
    build_model_meta,
)
from shaft.model.policies import build_peft_policy, build_processor_policy
from shaft.model.qwen3vl import _resolve_attn_implementation
from shaft.template import resolve_template_meta


def test_qwen3vl_meta_exposes_family_and_policies() -> None:
    model_meta = build_model_meta("qwen3vl")
    assert model_meta.family == "qwen"
    assert model_meta.processor_policy.supports_pixel_budget is True
    assert model_meta.module_groups.language_model == ("model",)
    assert model_meta.module_groups.vision_tower == ("model.visual",)
    assert model_meta.module_groups.aligner == (
        "model.visual.merger",
        "model.visual.deepstack_merger_list",
    )
    assert model_meta.module_groups.generator == ("lm_head",)
    assert model_meta.default_target_modules() == ["all-linear"]
    assert model_meta.default_template == "qwen3vl"
    assert model_meta.requires == ()
    assert model_meta.additional_saved_files == ()
    assert len(model_meta.model_groups) == 1
    assert model_meta.candidate_templates == ("qwen3vl",)


def test_qwen35vl_meta_exposes_family_and_policies() -> None:
    model_meta = build_model_meta("qwen35vl")
    assert model_meta.family == "qwen"
    assert model_meta.hf_model_types == ("qwen3_5", "qwen3_5_moe")
    assert model_meta.processor_policy.supports_pixel_budget is True
    assert model_meta.module_groups.language_model == ("model.language_model",)
    assert model_meta.module_groups.vision_tower == ("model.visual",)
    assert model_meta.module_groups.aligner == (
        "model.visual.merger",
        "model.visual.deepstack_merger_list",
    )
    assert model_meta.module_groups.generator == ("lm_head",)
    assert model_meta.default_target_modules() == ["all-linear"]
    assert model_meta.default_template == "qwen35vl"
    assert model_meta.requires == (
        "transformers>=5.10.1",
        "module:transformers.models.qwen3_5",
    )
    assert model_meta.candidate_templates == ("qwen35vl",)


def test_qwen36vl_alias_uses_same_template() -> None:
    model_meta = build_model_meta("qwen36vl")
    assert model_meta.default_template == "qwen35vl"
    assert model_meta.hf_model_types == ("qwen3_5", "qwen3_5_moe")
    assert model_meta.resolve_template_type("models/Qwen3.6-27B") == "qwen35vl"


def test_qwen35vl_dense_fsdp_auto_layers() -> None:
    adapter = build_model_meta("qwen35vl").resolve_adapter(
        model_name_or_path="models/Qwen3.6-27B"
    )
    assert adapter.resolve_fsdp_transformer_layer_cls_to_wrap(["auto"]) == [
        "Qwen3_5DecoderLayer",
        "Qwen3_5VisionBlock",
    ]


def test_qwen35vl_moe_fsdp_auto_layers() -> None:
    adapter = build_model_meta("qwen35vl").resolve_adapter(
        model_name_or_path="models/Qwen3.6-35B-A3B"
    )
    assert adapter.resolve_fsdp_transformer_layer_cls_to_wrap(["auto"]) == [
        "Qwen3_5MoeDecoderLayer",
        "Qwen3_5MoeVisionBlock",
    ]
    assert adapter.requires == (
        "transformers>=5.10.1",
        "module:transformers.models.qwen3_5",
        "module:transformers.models.qwen3_5_moe",
    )


def test_model_requires_check_validates_minimum_versions() -> None:
    model_meta = ModelMeta(
        model_type="dummy",
        family="dummy",
        default_template="smoke_vlm",
        requires=("transformers>=999.0.0",),
    )
    with pytest.raises(ImportError, match="transformers>=999.0.0"):
        model_meta.check_requires()


def test_model_requires_check_validates_required_modules() -> None:
    model_meta = ModelMeta(
        model_type="dummy",
        family="dummy",
        default_template="smoke_vlm",
        requires=("module:package_that_does_not_exist_for_shaft_tests.submodule",),
    )
    with pytest.raises(ImportError, match="package_that_does_not_exist"):
        model_meta.check_requires()


def test_model_meta_can_match_registered_model_name() -> None:
    model_meta = build_model_meta("smoke_vlm")
    matched = model_meta.get_matched_model_group("models/Smoke-VLM")
    assert matched is not None
    assert matched.name == "default"


def test_model_meta_resolves_template_from_matched_group() -> None:
    model_meta = build_model_meta("smoke_vlm")
    assert model_meta.resolve_template_type("models/Smoke-VLM") == "smoke_vlm"


def test_model_meta_check_requires_raises_for_missing_package() -> None:
    model_meta = ModelMeta(
        model_type="dummy",
        family="dummy",
        default_template="smoke_vlm",
        model_groups=(ModelGroup(name="default"),),
        requires=("package_that_does_not_exist_for_shaft_tests>=1.0",),
    )
    with pytest.raises(ImportError):
        model_meta.check_requires()


def test_model_meta_can_resolve_unified_model_adapter() -> None:
    model_meta = build_model_meta("smoke_vlm")
    adapter = model_meta.resolve_adapter(model_name_or_path="models/Smoke-VLM")
    assert adapter.model_type == "smoke_vlm"
    assert adapter.group_name == "default"
    assert adapter.template_type == "smoke_vlm"
    assert adapter.default_target_modules() == ["all-linear"]
    assert adapter.required_saved_files() == ("smoke_tokenizer.json", "smoke_processor.json")


def test_model_group_can_override_template_and_policies() -> None:
    model_meta = ModelMeta(
        model_type="dummy",
        family="dummy",
        default_template="smoke_vlm",
        capabilities=ModelCapabilities(is_multimodal=True),
        processor_policy=build_processor_policy("qwen_vl"),
        peft_policy=build_peft_policy("all_linear"),
        model_groups=(
            ModelGroup(
                name="compact",
                model_ids=("dummy-compact",),
                template="qwen3vl",
                capabilities=ModelCapabilities(is_multimodal=False),
                processor_policy=build_processor_policy("identity"),
                requires=("pkg_a>=1.0",),
                additional_saved_files=("extra.json",),
            ),
        ),
    )
    adapter = model_meta.resolve_adapter(model_name_or_path="dummy-compact")
    assert adapter.template_type == "qwen3vl"
    assert adapter.processor_policy.supports_pixel_budget is False
    assert adapter.capabilities.is_multimodal is False
    assert adapter.requires == ("pkg_a>=1.0",)
    assert adapter.required_saved_files() == ("extra.json",)


def test_qwen3vl_flash_attention_falls_back_without_flash_attn() -> None:
    with patch("shaft.model.qwen3vl.importlib.util.find_spec", return_value=None):
        with pytest.warns(UserWarning, match="flash-attn"):
            resolved = _resolve_attn_implementation("flash_attention_2")
    assert resolved is None


def test_model_meta_can_resolve_template_meta() -> None:
    model_adapter = build_model_meta("qwen3vl").resolve_adapter(
        model_name_or_path="models/Qwen3-VL-4B-Instruct"
    )
    template_meta = resolve_template_meta(model_adapter=model_adapter)
    assert template_meta.template_type == "qwen3vl"
    assert template_meta.template_cls.__name__ == "Qwen3VLTemplate"
