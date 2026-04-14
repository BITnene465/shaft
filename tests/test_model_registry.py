from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from peft import PeftModel
import pytest

from shaft.config import RuntimeConfig
from shaft.model import MODEL_REGISTRY, build_model_tokenizer_processor


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
    with patch.object(MODEL_REGISTRY, "create", return_value=fake_artifacts) as mocked:
        out = build_model_tokenizer_processor(config)
    mocked.assert_called_once_with("qwen3vl", config)
    assert out is fake_artifacts


def test_init_from_full_checkpoint_overrides_model_path(tmp_path: Path) -> None:
    config = RuntimeConfig()
    config.model.model_type = "qwen3vl"
    fake_artifacts = object()
    init_ckpt = tmp_path / "full_ckpt"
    init_ckpt.mkdir(parents=True, exist_ok=True)
    (init_ckpt / "config.json").write_text("{}", encoding="utf-8")
    with patch.object(MODEL_REGISTRY, "create", return_value=fake_artifacts) as mocked:
        out = build_model_tokenizer_processor(config, init_from_checkpoint=str(init_ckpt))
    _, called_cfg = mocked.call_args[0]
    assert called_cfg is not config
    assert called_cfg.model.model_name_or_path == str(init_ckpt)
    assert out is fake_artifacts


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
