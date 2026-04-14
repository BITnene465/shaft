from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from peft import PeftModel
import pytest

from shaft.config import RuntimeConfig
from shaft.model import MODEL_REGISTRY, ModelGroup, ModelMeta, build_model_meta, build_model_tokenizer_processor
from shaft.template import build_template_meta, resolve_template_meta


def test_qwen3vl_registered() -> None:
    assert MODEL_REGISTRY.has("qwen3vl")


def test_unknown_model_type_raises() -> None:
    config = RuntimeConfig()
    config.model.model_type = "unknown_model"
    with pytest.raises(KeyError):
        build_model_tokenizer_processor(config)


def test_builder_dispatches_registry() -> None:
    config = RuntimeConfig()
    config.model.model_type = "qwen3vl"
    fake_artifacts = object()
    fake_meta = type("Meta", (), {"loader": type("Loader", (), {"build": lambda self, cfg, *, model_meta: fake_artifacts})()})()
    with patch("shaft.model.builder.build_model_meta", return_value=fake_meta) as mocked:
        out = build_model_tokenizer_processor(config)
    mocked.assert_called_once_with("qwen3vl")
    assert out is fake_artifacts


def test_smoke_artifacts_expose_meta_and_template() -> None:
    cfg = RuntimeConfig()
    cfg.model.model_type = "smoke_vlm"
    artifacts = build_model_tokenizer_processor(cfg)
    assert artifacts.model_meta.model_type == "smoke_vlm"
    assert artifacts.model_info.model_type == "smoke_vlm"
    assert artifacts.model_info.model_dir == cfg.model.model_name_or_path
    assert artifacts.model_info.is_multimodal is True
    assert artifacts.model_meta.default_target_modules() == ["all-linear"]
    assert artifacts.model_meta.capabilities.supports_pixel_budget is False
    assert artifacts.template.name == "smoke_vlm"
    assert artifacts.template.template_meta.template_type == "smoke_vlm"


def test_qwen3vl_meta_exposes_family_and_policies() -> None:
    model_meta = build_model_meta("qwen3vl")
    assert model_meta.family == "qwen"
    assert model_meta.capabilities.supports_pixel_budget is True
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
    model_meta = build_model_meta("qwen3vl")
    captured = {}

    class _Processor:
        def __call__(self, **kwargs):
            captured.update(kwargs)
            return {"ok": True}

    _ = model_meta.processor_policy.build_inputs(processor=_Processor(), prompt_texts=["hello"], images=["img"], min_pixels=16, max_pixels=32)
    assert captured["min_pixels"] == 16
    assert captured["max_pixels"] == 32


def test_processor_policy_can_disable_pixel_budget() -> None:
    model_meta = build_model_meta("smoke_vlm")
    captured = {}

    class _Processor:
        def __call__(self, **kwargs):
            captured.update(kwargs)
            return {"ok": True}

    _ = model_meta.processor_policy.build_inputs(processor=_Processor(), prompt_texts=["hello"], images=["img"], min_pixels=16, max_pixels=32)
    assert "min_pixels" not in captured
    assert "max_pixels" not in captured


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

        def _build(self, cfg, *, model_meta):
            captured["cfg"] = cfg
            captured["model_meta"] = model_meta
            return object()

    with patch("shaft.model.builder.build_model_meta", return_value=_Adapter()):
        _ = build_model_tokenizer_processor(config, init_from_checkpoint=str(init_ckpt))
    called_cfg = captured["cfg"]
    assert called_cfg is not config
    assert called_cfg.model.model_name_or_path == str(init_ckpt)
    assert captured["model_meta"] is not None


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
    cfg_src.model.finetune.mode = "lora"
    cfg_src.model.finetune.target_modules = ["all-linear"]
    cfg_src.model.finetune.lora_r = 8
    cfg_src.model.finetune.lora_alpha = 16
    artifacts_src = build_model_tokenizer_processor(cfg_src)
    adapter_dir = tmp_path / "adapter"
    artifacts_src.model.save_pretrained(adapter_dir)

    cfg_tgt = RuntimeConfig()
    cfg_tgt.model.model_type = "smoke_vlm"
    cfg_tgt.model.finetune.mode = "lora"
    cfg_tgt.model.finetune.target_modules = ["all-linear"]
    cfg_tgt.model.finetune.lora_r = 16  # mismatch
    cfg_tgt.model.finetune.lora_alpha = 16
    with pytest.raises(ValueError):
        build_model_tokenizer_processor(cfg_tgt, init_from_checkpoint=str(adapter_dir))


def test_model_meta_can_resolve_template_meta() -> None:
    model_meta = build_model_meta("qwen3vl")
    template_meta = resolve_template_meta(model_meta=model_meta)
    assert template_meta.template_type == "qwen3vl"
    assert template_meta.template_cls.__name__ == "Qwen3VLTemplate"
