from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from shaft.config import RuntimeConfig
from shaft.model import (
    ModelCapabilities,
    ModelGroup,
    ModelMeta,
    build_model_meta,
    resolve_local_model_descriptor,
    resolve_model_plan,
)
from shaft.model.policies import build_peft_policy, build_processor_policy
from shaft.model.qwen3vl import _resolve_attn_implementation
from shaft.model.generation import align_model_generation_config
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


def test_generation_alignment_accepts_a_scalar_existing_eos_token() -> None:
    target = SimpleNamespace(
        config=SimpleNamespace(eos_token_id=None, bos_token_id=None, pad_token_id=None),
        generation_config=SimpleNamespace(
            do_sample=False,
            eos_token_id=2,
            bos_token_id=None,
            pad_token_id=None,
        ),
    )
    tokenizer = SimpleNamespace(eos_token_id=2, bos_token_id=1, pad_token_id=0)

    align_model_generation_config(target, tokenizer=tokenizer)

    assert target.config.eos_token_id == 2
    assert target.generation_config.eos_token_id == [2]


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


def test_qwen35vl_unknown_local_name_selects_variant_from_hf_config(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "arbitrary-release-name"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5_moe",
                "architectures": ["Qwen3_5MoeForConditionalGeneration"],
                "text_config": {"layer_types": ["linear_attention", "full_attention"]},
            }
        ),
        encoding="utf-8",
    )
    descriptor = resolve_local_model_descriptor(model_dir)
    assert descriptor is not None
    assert descriptor.hf_model_type == "qwen3_5_moe"

    adapter = build_model_meta("qwen35vl").resolve_adapter(
        model_name_or_path=str(model_dir),
        descriptor=descriptor,
    )
    assert adapter.group_name == "moe"
    assert adapter.resolve_fsdp_transformer_layer_cls_to_wrap(["auto"]) == [
        "Qwen3_5MoeDecoderLayer",
        "Qwen3_5MoeVisionBlock",
    ]


def test_qwen35vl_descriptor_overrides_misleading_catalog_basename(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "qwen3.6-27b"
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
    descriptor = resolve_local_model_descriptor(model_dir)
    assert descriptor is not None

    adapter = build_model_meta("qwen36vl").resolve_adapter(
        model_name_or_path=str(model_dir),
        descriptor=descriptor,
    )

    assert adapter.group_name == "moe"


def test_qwen35vl_custom_hub_checkpoint_resolves_config_before_group_selection() -> None:
    config = RuntimeConfig()
    config.model.model_type = "qwen36vl"
    config.model.model_name_or_path = "my-org/Qwen3.6-domain-sft"
    config.model.revision = "release-v2"

    with patch(
        "shaft.model.descriptor.PretrainedConfig.get_config_dict",
        return_value=(
            {
                "model_type": "qwen3_5_moe",
                "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            },
            {},
        ),
    ) as resolver:
        plan = resolve_model_plan(config)

    assert plan.model_adapter.group_name == "moe"
    assert plan.descriptor is not None
    assert plan.descriptor.source == "hf://my-org/Qwen3.6-domain-sft@release-v2"
    assert plan.effective_model_name_or_path == "my-org/Qwen3.6-domain-sft"
    assert plan.fingerprint
    resolver.assert_called_once_with(
        "my-org/Qwen3.6-domain-sft",
        revision="release-v2",
        cache_dir=None,
        local_files_only=False,
    )


def test_hub_descriptor_overrides_a_misleading_catalog_basename() -> None:
    config = RuntimeConfig()
    config.model.model_type = "qwen36vl"
    config.model.model_name_or_path = "my-org/Qwen3.6-27B"

    with patch(
        "shaft.model.descriptor.PretrainedConfig.get_config_dict",
        return_value=(
            {
                "model_type": "qwen3_5_moe",
                "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            },
            {},
        ),
    ):
        plan = resolve_model_plan(config)

    assert plan.model_adapter.group_name == "moe"


@pytest.mark.parametrize(
    "repo_id",
    [
        "custom-org/not-actually-qwen",
        "models/not-actually-qwen",
        "outputs/not-actually-qwen",
        "checkpoints/not-actually-qwen",
        "artifacts/not-actually-qwen",
    ],
)
def test_single_variant_custom_hub_checkpoint_still_validates_hf_config(
    repo_id: str,
) -> None:
    config = RuntimeConfig()
    config.model.model_type = "qwen3vl"
    config.model.model_name_or_path = repo_id

    with patch(
        "shaft.model.descriptor.PretrainedConfig.get_config_dict",
        return_value=(
            {
                "model_type": "unrelated_vlm",
                "architectures": ["UnrelatedVisionLanguageModel"],
            },
            {},
        ),
    ) as resolver:
        with pytest.raises(ValueError, match="not a registered variant"):
            resolve_model_plan(config)

    resolver.assert_called_once()


def test_qwen35vl_conflicting_model_type_and_architecture_fail_closed(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "conflicting-qwen"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5",
                "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            }
        ),
        encoding="utf-8",
    )
    config = RuntimeConfig()
    config.model.model_type = "qwen35vl"
    config.model.model_name_or_path = str(model_dir)

    with pytest.raises(ValueError, match="does not match any registered model group"):
        resolve_model_plan(config)


def test_full_init_checkpoint_is_the_model_plan_truth_source(tmp_path: Path) -> None:
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
    config = RuntimeConfig()
    config.model.model_type = "qwen36vl"
    config.model.model_name_or_path = "models/Qwen3.6-27B"

    plan = resolve_model_plan(config, init_from_checkpoint=str(checkpoint))

    assert plan.init_kind == "full_checkpoint"
    assert plan.effective_model_name_or_path == str(checkpoint)
    assert plan.model_adapter.model_name_or_path == str(checkpoint)
    assert plan.model_adapter.group_name == "moe"


def test_adapter_init_keeps_base_artifact_as_model_plan_truth_source(
    tmp_path: Path,
) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "models/Qwen3.6-27B"}),
        encoding="utf-8",
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"")
    config = RuntimeConfig()
    config.model.model_type = "qwen36vl"
    config.model.model_name_or_path = "models/Qwen3.6-27B"

    plan = resolve_model_plan(config, init_from_checkpoint=str(adapter))

    assert plan.init_kind == "adapter"
    assert plan.effective_model_name_or_path == "models/Qwen3.6-27B"
    assert plan.model_adapter.group_name == "dense"
    assert plan.adapter_init is not None
    assert plan.adapter_init.base_model_name_or_path == "models/Qwen3.6-27B"


def test_adapter_init_fingerprint_binds_the_adapter_artifact(tmp_path: Path) -> None:
    config = RuntimeConfig()
    config.model.model_type = "qwen36vl"
    config.model.model_name_or_path = "models/Qwen3.6-27B"
    fingerprints = []
    for name in ("adapter-a", "adapter-b"):
        adapter = tmp_path / name
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": "models/Qwen3.6-27B"}),
            encoding="utf-8",
        )
        (adapter / "adapter_model.safetensors").write_bytes(name.encode("utf-8"))
        fingerprints.append(
            resolve_model_plan(config, init_from_checkpoint=str(adapter)).fingerprint
        )

    assert fingerprints[0] != fingerprints[1]


def test_adapter_init_rejects_a_different_declared_base_variant(
    tmp_path: Path,
) -> None:
    dense = tmp_path / "dense"
    dense.mkdir()
    (dense / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5"}),
        encoding="utf-8",
    )
    moe = tmp_path / "moe"
    moe.mkdir()
    (moe / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5_moe",
                "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            }
        ),
        encoding="utf-8",
    )
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": str(moe)}),
        encoding="utf-8",
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"placeholder")
    config = RuntimeConfig()
    config.model.model_type = "qwen36vl"
    config.model.model_name_or_path = str(dense)

    with pytest.raises(ValueError, match="base variant differs"):
        resolve_model_plan(config, init_from_checkpoint=str(adapter))


def test_qwen35vl_unknown_hf_architecture_fails_closed(tmp_path: Path) -> None:
    model_dir = tmp_path / "future-qwen"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen_future_vl",
                "architectures": ["QwenFutureForConditionalGeneration"],
            }
        ),
        encoding="utf-8",
    )
    descriptor = resolve_local_model_descriptor(model_dir)
    assert descriptor is not None

    with pytest.raises(ValueError, match="not a registered variant"):
        build_model_meta("qwen35vl").resolve_adapter(
            model_name_or_path=str(model_dir),
            descriptor=descriptor,
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
    assert model_meta.uses_hf_artifacts is False
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


def test_qwen3vl_required_flash_attention_never_silently_falls_back() -> None:
    with patch("shaft.model.qwen3vl.importlib.util.find_spec", return_value=None):
        with pytest.raises(ImportError, match="varlen.*flash-attn"):
            _resolve_attn_implementation("flash_attention_2", required=True)


def test_model_meta_can_resolve_template_meta() -> None:
    model_adapter = build_model_meta("qwen3vl").resolve_adapter(
        model_name_or_path="models/Qwen3-VL-4B-Instruct"
    )
    template_meta = resolve_template_meta(model_adapter=model_adapter)
    assert template_meta.template_type == "qwen3vl"
    assert template_meta.template_cls.__name__ == "Qwen3VLTemplate"
