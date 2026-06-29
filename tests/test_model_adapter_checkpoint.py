from __future__ import annotations

import json
from pathlib import Path

from peft import PeftModel
import pytest

from shaft.config import RuntimeConfig
from shaft.model import build_model_tokenizer_processor


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
