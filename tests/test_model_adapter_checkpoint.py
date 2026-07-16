from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from peft import PeftModel
import pytest
from safetensors.torch import load_file, save_file

from shaft.config import RuntimeConfig
from shaft.model import build_model_tokenizer_processor, resolve_model_plan
from shaft.model.builder import _expected_adapter_names_from_artifacts


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

    src_lora = {
        name: value.detach().cpu().clone()
        for name, value in artifacts_src.model.named_parameters()
        if "lora_" in name
    }
    assert src_lora, "source adapter has no lora parameters"

    cfg_tgt = RuntimeConfig()
    cfg_tgt.model.model_type = "smoke_vlm"
    cfg_tgt.model.model_name_or_path = "models/smoke-vlm"
    cfg_tgt.model.finetune.mode = "lora"
    cfg_tgt.model.finetune.target_modules = ["all-linear"]
    cfg_tgt.model.finetune.lora_r = 8
    cfg_tgt.model.finetune.lora_alpha = 16
    artifacts_tgt = build_model_tokenizer_processor(
        cfg_tgt,
        init_from_checkpoint=str(adapter_dir),
    )
    assert isinstance(artifacts_tgt.model, PeftModel)
    tgt_lora = {
        name: value.detach().cpu()
        for name, value in artifacts_tgt.model.named_parameters()
        if "lora_" in name
    }
    assert tgt_lora.keys() == src_lora.keys()
    for key in src_lora:
        assert (src_lora[key] == tgt_lora[key]).all()


def test_adapter_compatibility_uses_peft_persisted_target_canonicalization() -> None:
    artifacts = SimpleNamespace(
        model=SimpleNamespace(
            peft_config={
                "default": SimpleNamespace(
                    target_modules={"q_proj", "v_proj"},
                    modules_to_save=None,
                )
            }
        ),
        finetune_plan=SimpleNamespace(
            adapter_plan=SimpleNamespace(
                resolved_target_modules=(
                    "model.layers.0.self_attn.q_proj",
                    "model.layers.0.self_attn.v_proj",
                ),
                modules_to_save=(),
            )
        ),
    )

    targets, modules_to_save = _expected_adapter_names_from_artifacts(artifacts)

    assert sorted(targets or ()) == ["q_proj", "v_proj"]
    assert modules_to_save == []


def test_init_from_adapter_rejects_missing_state_keys(tmp_path: Path) -> None:
    cfg = RuntimeConfig()
    cfg.model.model_type = "smoke_vlm"
    cfg.model.model_name_or_path = "models/smoke-vlm"
    cfg.model.finetune.mode = "lora"
    cfg.model.finetune.target_modules = ["all-linear"]
    artifacts = build_model_tokenizer_processor(cfg)
    adapter_dir = tmp_path / "adapter"
    artifacts.model.save_pretrained(adapter_dir)
    weights_path = adapter_dir / "adapter_model.safetensors"
    state = load_file(weights_path)
    state.pop(next(iter(state)))
    save_file(state, weights_path)

    with pytest.raises(ValueError, match="does not exactly match"):
        build_model_tokenizer_processor(
            cfg,
            init_from_checkpoint=str(adapter_dir),
        )


def test_adapter_plan_detects_same_size_weight_replacement_before_load(
    tmp_path: Path,
) -> None:
    cfg = RuntimeConfig()
    cfg.model.model_type = "smoke_vlm"
    cfg.model.model_name_or_path = "models/smoke-vlm"
    cfg.model.finetune.mode = "lora"
    cfg.model.finetune.target_modules = ["all-linear"]
    source = build_model_tokenizer_processor(cfg)
    adapter_dir = tmp_path / "adapter"
    source.model.save_pretrained(adapter_dir)
    plan = resolve_model_plan(cfg, init_from_checkpoint=str(adapter_dir))

    weights_path = adapter_dir / "adapter_model.safetensors"
    payload = bytearray(weights_path.read_bytes())
    payload[-1] ^= 1
    weights_path.write_bytes(payload)

    with pytest.raises(ValueError, match="changed after ResolvedModelPlan"):
        build_model_tokenizer_processor(
            cfg,
            init_from_checkpoint=str(adapter_dir),
            resolved_model_plan=plan,
        )


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
    cfg_tgt.model.finetune.lora_r = 16
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
    artifacts_tgt = build_model_tokenizer_processor(
        cfg_tgt,
        init_from_checkpoint=str(adapter_dir),
    )
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
