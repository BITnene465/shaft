from __future__ import annotations

from pathlib import Path

import pytest

from shaft.config import RuntimeConfig
from shaft.model import build_model_meta
from shaft.training.checkpointing import (
    ensure_hf_export_layout,
    resolve_resume_checkpoint,
    validate_training_state_policy,
)


def test_validate_training_state_policy_requires_eval_for_best_model() -> None:
    cfg = RuntimeConfig()
    cfg.sft.train.load_best_model_at_end = True
    cfg.sft.eval.enabled = False
    with pytest.raises(ValueError):
        validate_training_state_policy(cfg)


def test_validate_training_state_policy_requires_matching_strategies() -> None:
    cfg = RuntimeConfig()
    cfg.sft.train.load_best_model_at_end = True
    cfg.sft.train.save_strategy = "epoch"
    cfg.sft.eval.enabled = True
    cfg.sft.eval.eval_strategy = "steps"
    with pytest.raises(ValueError):
        validate_training_state_policy(cfg)


def test_resolve_resume_checkpoint_picks_last_checkpoint(tmp_path: Path) -> None:
    root = tmp_path / "run"
    ckpt1 = root / "checkpoint-1"
    ckpt2 = root / "checkpoint-2"
    ckpt1.mkdir(parents=True)
    ckpt2.mkdir(parents=True)
    (ckpt1 / "trainer_state.json").write_text("{}", encoding="utf-8")
    (ckpt2 / "trainer_state.json").write_text("{}", encoding="utf-8")
    resolved = resolve_resume_checkpoint(str(root))
    assert resolved == str(ckpt2)


def test_ensure_hf_export_layout_full(tmp_path: Path) -> None:
    export_dir = tmp_path / "full"
    export_dir.mkdir()
    (export_dir / "config.json").write_text("{}", encoding="utf-8")
    (export_dir / "model.safetensors").write_bytes(b"ok")
    ensure_hf_export_layout(export_dir, finetune_mode="full")


def test_ensure_hf_export_layout_adapter(tmp_path: Path) -> None:
    export_dir = tmp_path / "adapter"
    export_dir.mkdir()
    (export_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    (export_dir / "adapter_model.safetensors").write_bytes(b"ok")
    ensure_hf_export_layout(export_dir, finetune_mode="lora")


def test_ensure_hf_export_layout_validates_additional_saved_files(tmp_path: Path) -> None:
    export_dir = tmp_path / "full"
    export_dir.mkdir()
    (export_dir / "config.json").write_text("{}", encoding="utf-8")
    (export_dir / "model.safetensors").write_bytes(b"ok")
    model_meta = build_model_meta("smoke_vlm")
    with pytest.raises(ValueError):
        ensure_hf_export_layout(export_dir, finetune_mode="full", model_meta=model_meta)
    (export_dir / "smoke_tokenizer.json").write_text("{}", encoding="utf-8")
    (export_dir / "smoke_processor.json").write_text("{}", encoding="utf-8")
    ensure_hf_export_layout(export_dir, finetune_mode="full", model_meta=model_meta)
