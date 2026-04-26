from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from peft import PeftModel
import pytest

from shaft.config import RuntimeConfig
from shaft.model import (
    MODEL_REGISTRY,
    ModelCapabilities,
    ModelGroup,
    ModelMeta,
    PEFT_POLICY_REGISTRY,
    PROCESSOR_POLICY_REGISTRY,
    build_model_meta,
    build_model_tokenizer_processor,
)
from shaft.model.policies import build_peft_policy, build_processor_policy
from shaft.model.qwen3vl import _resolve_attn_implementation
from shaft.template import resolve_template_meta


def test_qwen3vl_registered() -> None:
    assert MODEL_REGISTRY.has("qwen3vl")


def test_builtin_model_policies_registered() -> None:
    assert PROCESSOR_POLICY_REGISTRY.has("pixel_budget")
    assert PROCESSOR_POLICY_REGISTRY.has("no_pixel_budget")
    assert PEFT_POLICY_REGISTRY.has("all_linear")


def test_unknown_model_type_raises() -> None:
    config = RuntimeConfig()
    config.model.model_type = "unknown_model"
    with pytest.raises(KeyError):
        build_model_tokenizer_processor(config)


def test_builder_dispatches_registry() -> None:
    config = RuntimeConfig()
    config.model.model_type = "qwen3vl"
    fake_artifacts = object()
    fake_meta = type(
        "Meta",
        (),
        {
            "resolve_adapter": lambda self, *, model_name_or_path, template_type=None: type(
                "Adapter", (), {"check_requires": lambda self: None}
            )(),
            "loader": type(
                "Loader",
                (),
                {
                    "build": lambda self, cfg, *, model_meta, model_adapter: fake_artifacts,
                },
            )(),
        },
    )()
    with patch("shaft.model.builder.build_model_meta", return_value=fake_meta) as mocked:
        out = build_model_tokenizer_processor(config)
    mocked.assert_called_once_with("qwen3vl")
    assert out is fake_artifacts


def test_smoke_artifacts_expose_meta_and_template() -> None:
    cfg = RuntimeConfig()
    cfg.model.model_type = "smoke_vlm"
    artifacts = build_model_tokenizer_processor(cfg)
    assert artifacts.model_meta.model_type == "smoke_vlm"
    assert artifacts.model_adapter.model_type == "smoke_vlm"
    assert artifacts.model_info.model_type == "smoke_vlm"
    assert artifacts.model_info.model_dir == cfg.model.model_name_or_path
    assert artifacts.model_info.is_multimodal is True
    assert artifacts.model_adapter.default_target_modules() == ["all-linear"]
    assert artifacts.model_adapter.capabilities.supports_pixel_budget is False
    assert artifacts.template.name == "smoke_vlm"
    assert artifacts.template.template_meta.template_type == "smoke_vlm"


def test_qwen3vl_meta_exposes_family_and_policies() -> None:
    model_meta = build_model_meta("qwen3vl")
    assert model_meta.family == "qwen"
    assert model_meta.capabilities.supports_pixel_budget is True
    assert model_meta.module_groups.language_model == ("model",)
    assert model_meta.module_groups.vision_tower == ("model.visual",)
    assert model_meta.module_groups.aligner == ("model.visual.merger", "model.visual.deepstack_merger_list")
    assert model_meta.module_groups.generator == ("lm_head",)
    assert model_meta.default_target_modules() == ["all-linear"]
    assert model_meta.default_template == "qwen3vl"
    assert model_meta.requires == ()
    assert model_meta.additional_saved_files == ()
    assert len(model_meta.model_groups) == 1
    assert model_meta.candidate_templates == ("qwen3vl",)


def test_model_meta_supports_model_group_override() -> None:
    model_meta = build_model_meta("qwen3vl")
    group = model_meta.model_groups[0]
    assert group.name == "default"
    assert group.template == "qwen3vl"
    assert isinstance(group.model_ids, tuple)


def test_model_meta_collects_requires_from_groups() -> None:
    model_meta = build_model_meta("qwen3vl")
    assert model_meta.all_requires() == []


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


def test_processor_policy_controls_pixel_budget_forwarding() -> None:
    model_adapter = build_model_meta("qwen3vl").resolve_adapter(model_name_or_path="models/Qwen3-VL-4B-Instruct")
    captured = {}

    class _Processor:
        def __call__(self, **kwargs):
            captured.update(kwargs)
            return {"ok": True}

    _ = model_adapter.build_processor_inputs(
        processor=_Processor(),
        prompt_texts=["hello"],
        images=["img"],
        min_pixels=16,
        max_pixels=32,
    )
    assert captured["min_pixels"] == 16
    assert captured["max_pixels"] == 32


def test_processor_policy_can_disable_pixel_budget() -> None:
    model_adapter = build_model_meta("smoke_vlm").resolve_adapter(model_name_or_path="models/Smoke-VLM")
    captured = {}

    class _Processor:
        def __call__(self, **kwargs):
            captured.update(kwargs)
            return {"ok": True}

    _ = model_adapter.build_processor_inputs(
        processor=_Processor(),
        prompt_texts=["hello"],
        images=["img"],
        min_pixels=16,
        max_pixels=32,
    )
    assert "min_pixels" not in captured
    assert "max_pixels" not in captured


def test_processor_policy_temporarily_controls_padding_side() -> None:
    model_adapter = build_model_meta("qwen3vl").resolve_adapter(model_name_or_path="models/Qwen3-VL-4B-Instruct")
    captured = {}

    class _Tokenizer:
        def __init__(self) -> None:
            self.padding_side = "right"

    tokenizer = _Tokenizer()

    class _Processor:
        def __init__(self, tokenizer_obj) -> None:
            self.tokenizer = tokenizer_obj

        def __call__(self, **kwargs):
            captured["padding_side_during_call"] = self.tokenizer.padding_side
            return {"ok": True, "kwargs": kwargs}

    processor = _Processor(tokenizer)
    _ = model_adapter.build_processor_inputs(
        processor=processor,
        tokenizer=tokenizer,
        prompt_texts=["hello"],
        images=["img"],
        min_pixels=16,
        max_pixels=32,
        padding_side="left",
    )
    assert captured["padding_side_during_call"] == "left"
    assert tokenizer.padding_side == "right"


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
        capabilities=ModelCapabilities(supports_pixel_budget=True, is_multimodal=True),
        processor_policy=build_processor_policy("pixel_budget"),
        peft_policy=build_peft_policy("all_linear"),
        model_groups=(
            ModelGroup(
                name="compact",
                model_ids=("dummy-compact",),
                template="qwen3vl",
                capabilities=ModelCapabilities(supports_pixel_budget=False, is_multimodal=False),
                processor_policy=build_processor_policy("no_pixel_budget"),
                requires=("pkg_a>=1.0",),
                additional_saved_files=("extra.json",),
            ),
        ),
    )
    adapter = model_meta.resolve_adapter(model_name_or_path="dummy-compact")
    assert adapter.template_type == "qwen3vl"
    assert adapter.capabilities.supports_pixel_budget is False
    assert adapter.capabilities.is_multimodal is False
    assert adapter.requires == ("pkg_a>=1.0",)
    assert adapter.required_saved_files() == ("extra.json",)


def test_qwen3vl_flash_attention_falls_back_without_flash_attn() -> None:
    with patch("shaft.model.qwen3vl.importlib.util.find_spec", return_value=None):
        with pytest.warns(UserWarning, match="flash-attn"):
            resolved = _resolve_attn_implementation("flash_attention_2")
    assert resolved is None


def test_qwen3vl_attn_implementation_keeps_non_flash_value() -> None:
    assert _resolve_attn_implementation("sdpa") == "sdpa"


def test_init_from_full_checkpoint_overrides_model_path(tmp_path: Path) -> None:
    config = RuntimeConfig()
    config.model.model_type = "qwen3vl"
    init_ckpt = tmp_path / "full_ckpt"
    init_ckpt.mkdir(parents=True, exist_ok=True)
    (init_ckpt / "config.json").write_text("{}", encoding="utf-8")

    captured = {}

    class _Adapter:
        loader = None

        def __init__(self):
            self.loader = type("Loader", (), {"build": self._build})()

        def resolve_adapter(self, *, model_name_or_path, template_type=None):
            _ = template_type
            return type(
                "ResolvedAdapter",
                (),
                {
                    "check_requires": lambda self: None,
                    "model_name_or_path": model_name_or_path,
                },
            )()

        def _build(self, cfg, *, model_meta, model_adapter):
            captured["cfg"] = cfg
            captured["model_meta"] = model_meta
            captured["model_adapter"] = model_adapter
            return object()

    with patch("shaft.model.builder.build_model_meta", return_value=_Adapter()):
        _ = build_model_tokenizer_processor(config, init_from_checkpoint=str(init_ckpt))
    called_cfg = captured["cfg"]
    assert called_cfg is not config
    assert called_cfg.model.model_name_or_path == str(init_ckpt)
    assert captured["model_meta"] is not None
    assert captured["model_adapter"].model_name_or_path == str(init_ckpt)


def test_init_from_adapter_requires_peft_mode(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"placeholder")

    cfg = RuntimeConfig()
    cfg.model.model_type = "smoke_vlm"
    cfg.model.finetune.mode = "full"
    with pytest.raises(ValueError):
        build_model_tokenizer_processor(cfg, init_from_checkpoint=str(adapter_dir))


def test_init_from_adapter_lora_loads_weights(tmp_path: Path) -> None:
    cfg_src = RuntimeConfig()
    cfg_src.model.model_type = "smoke_vlm"
    cfg_src.model.model_name_or_path = "models/smoke-vlm"
    cfg_src.model.finetune.mode = "lora"
    cfg_src.model.finetune.target_modules = ["all-linear"]
    cfg_src.model.finetune.lora_r = 8
    cfg_src.model.finetune.lora_alpha = 16
    artifacts_src = build_model_tokenizer_processor(cfg_src)
    assert isinstance(artifacts_src.model, PeftModel)
    adapter_dir = tmp_path / "adapter"
    artifacts_src.model.save_pretrained(adapter_dir)

    src_lora = {k: v.detach().cpu().clone() for k, v in artifacts_src.model.named_parameters() if "lora_" in k}
    assert src_lora, "source adapter has no lora parameters"

    cfg_tgt = RuntimeConfig()
    cfg_tgt.model.model_type = "smoke_vlm"
    cfg_tgt.model.model_name_or_path = "models/smoke-vlm"
    cfg_tgt.model.finetune.mode = "lora"
    cfg_tgt.model.finetune.target_modules = ["all-linear"]
    cfg_tgt.model.finetune.lora_r = 8
    cfg_tgt.model.finetune.lora_alpha = 16
    artifacts_tgt = build_model_tokenizer_processor(cfg_tgt, init_from_checkpoint=str(adapter_dir))
    assert isinstance(artifacts_tgt.model, PeftModel)
    tgt_lora = {k: v.detach().cpu() for k, v in artifacts_tgt.model.named_parameters() if "lora_" in k}
    assert tgt_lora.keys() == src_lora.keys()
    for key in src_lora:
        assert (src_lora[key] == tgt_lora[key]).all()


def test_init_from_adapter_mismatch_raises(tmp_path: Path) -> None:
    cfg_src = RuntimeConfig()
    cfg_src.model.model_type = "smoke_vlm"
    cfg_src.model.model_name_or_path = "models/smoke-vlm"
    cfg_src.model.finetune.mode = "lora"
    cfg_src.model.finetune.target_modules = ["all-linear"]
    cfg_src.model.finetune.lora_r = 8
    cfg_src.model.finetune.lora_alpha = 16
    artifacts_src = build_model_tokenizer_processor(cfg_src)
    adapter_dir = tmp_path / "adapter"
    artifacts_src.model.save_pretrained(adapter_dir)

    cfg_tgt = RuntimeConfig()
    cfg_tgt.model.model_type = "smoke_vlm"
    cfg_tgt.model.model_name_or_path = "models/smoke-vlm"
    cfg_tgt.model.finetune.mode = "lora"
    cfg_tgt.model.finetune.target_modules = ["all-linear"]
    cfg_tgt.model.finetune.lora_r = 16  # mismatch
    cfg_tgt.model.finetune.lora_alpha = 16
    with pytest.raises(ValueError):
        build_model_tokenizer_processor(cfg_tgt, init_from_checkpoint=str(adapter_dir))


def test_init_from_adapter_with_modules_to_save_loads_weights(tmp_path: Path) -> None:
    cfg_src = RuntimeConfig()
    cfg_src.model.model_type = "smoke_vlm"
    cfg_src.model.model_name_or_path = "models/smoke-vlm"
    cfg_src.model.finetune.mode = "lora"
    cfg_src.model.finetune.target_modules = ["all-linear"]
    cfg_src.model.finetune.freeze.trainable_prefixes = ["lm_head"]
    artifacts_src = build_model_tokenizer_processor(cfg_src)
    assert isinstance(artifacts_src.model, PeftModel)

    for name, parameter in artifacts_src.model.named_parameters():
        if "lora_A" in name:
            parameter.data.fill_(0.125)
        if "lora_B" in name:
            parameter.data.fill_(0.25)
        if "modules_to_save" in name:
            parameter.data.fill_(0.5)

    adapter_dir = tmp_path / "adapter-with-modules-to-save"
    artifacts_src.model.save_pretrained(adapter_dir)
    payload = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    assert payload["modules_to_save"] == ["lm_head"]

    src_state = {
        name: parameter.detach().cpu().clone()
        for name, parameter in artifacts_src.model.named_parameters()
        if "lora_" in name or "modules_to_save" in name
    }
    assert any("modules_to_save" in name for name in src_state)

    cfg_tgt = RuntimeConfig()
    cfg_tgt.model.model_type = "smoke_vlm"
    cfg_tgt.model.model_name_or_path = "models/smoke-vlm"
    cfg_tgt.model.finetune.mode = "lora"
    cfg_tgt.model.finetune.target_modules = ["all-linear"]
    cfg_tgt.model.finetune.freeze.trainable_prefixes = ["lm_head"]
    artifacts_tgt = build_model_tokenizer_processor(cfg_tgt, init_from_checkpoint=str(adapter_dir))
    assert isinstance(artifacts_tgt.model, PeftModel)
    tgt_state = {
        name: parameter.detach().cpu()
        for name, parameter in artifacts_tgt.model.named_parameters()
        if "lora_" in name or "modules_to_save" in name
    }
    assert tgt_state.keys() == src_state.keys()
    for key in src_state:
        assert (src_state[key] == tgt_state[key]).all()


def test_init_from_adapter_modules_to_save_mismatch_raises(tmp_path: Path) -> None:
    cfg_src = RuntimeConfig()
    cfg_src.model.model_type = "smoke_vlm"
    cfg_src.model.model_name_or_path = "models/smoke-vlm"
    cfg_src.model.finetune.mode = "lora"
    cfg_src.model.finetune.target_modules = ["all-linear"]
    cfg_src.model.finetune.freeze.trainable_prefixes = ["lm_head"]
    artifacts_src = build_model_tokenizer_processor(cfg_src)
    adapter_dir = tmp_path / "adapter-with-modules-to-save"
    artifacts_src.model.save_pretrained(adapter_dir)

    cfg_tgt = RuntimeConfig()
    cfg_tgt.model.model_type = "smoke_vlm"
    cfg_tgt.model.model_name_or_path = "models/smoke-vlm"
    cfg_tgt.model.finetune.mode = "lora"
    cfg_tgt.model.finetune.target_modules = ["all-linear"]
    with pytest.raises(ValueError, match="modules_to_save mismatch"):
        build_model_tokenizer_processor(cfg_tgt, init_from_checkpoint=str(adapter_dir))


def test_model_meta_can_resolve_template_meta() -> None:
    model_adapter = build_model_meta("qwen3vl").resolve_adapter(model_name_or_path="models/Qwen3-VL-4B-Instruct")
    template_meta = resolve_template_meta(model_adapter=model_adapter)
    assert template_meta.template_type == "qwen3vl"
    assert template_meta.template_cls.__name__ == "Qwen3VLTemplate"
