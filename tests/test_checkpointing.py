from __future__ import annotations

from pathlib import Path

import pytest

from shaft.config import RuntimeConfig
from shaft.model import build_model_meta
from shaft.training.checkpointing import (
    ensure_hf_export_layout,
    resolve_resume_checkpoint,
    validate_resume_checkpoint,
    validate_training_state_policy,
)


def test_validate_training_state_policy_requires_eval_for_best_model() -> None:
    cfg = RuntimeConfig()
    cfg.train.load_best_model_at_end = True
    cfg.eval.enabled = False
    with pytest.raises(ValueError):
        validate_training_state_policy(cfg)


def test_validate_training_state_policy_requires_matching_strategies() -> None:
    cfg = RuntimeConfig()
    cfg.train.load_best_model_at_end = True
    cfg.train.save_strategy = "epoch"
    cfg.eval.enabled = True
    cfg.eval.eval_strategy = "steps"
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


def test_ensure_hf_export_layout_rejects_mismatched_mode(tmp_path: Path) -> None:
    full_dir = tmp_path / "full"
    full_dir.mkdir()
    (full_dir / "config.json").write_text("{}", encoding="utf-8")
    (full_dir / "model.safetensors").write_bytes(b"ok")
    with pytest.raises(ValueError):
        ensure_hf_export_layout(full_dir, finetune_mode="lora")

    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"ok")
    with pytest.raises(ValueError):
        ensure_hf_export_layout(adapter_dir, finetune_mode="full")


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


def _make_full_checkpoint(path: Path, *, with_state: bool = True) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"ok")
    if with_state:
        (path / "trainer_state.json").write_text("{}", encoding="utf-8")
    return path


def _make_adapter_checkpoint(path: Path, *, with_state: bool = True) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (path / "adapter_model.safetensors").write_bytes(b"ok")
    if with_state:
        (path / "trainer_state.json").write_text("{}", encoding="utf-8")
    return path


def test_validate_resume_checkpoint_full_mode_accepts_full_checkpoint(tmp_path: Path) -> None:
    ckpt = _make_full_checkpoint(tmp_path / "ckpt-full")
    validate_resume_checkpoint(ckpt, finetune_mode="full")


def test_validate_resume_checkpoint_full_mode_rejects_adapter_checkpoint(tmp_path: Path) -> None:
    ckpt = _make_adapter_checkpoint(tmp_path / "ckpt-adapter")
    with pytest.raises(ValueError):
        validate_resume_checkpoint(ckpt, finetune_mode="full")


@pytest.mark.parametrize("mode", ["lora", "dora", "qlora"])
def test_validate_resume_checkpoint_adapter_modes_accept_adapter_checkpoint(tmp_path: Path, mode: str) -> None:
    ckpt = _make_adapter_checkpoint(tmp_path / f"ckpt-{mode}")
    validate_resume_checkpoint(ckpt, finetune_mode=mode)


@pytest.mark.parametrize("mode", ["lora", "dora", "qlora"])
def test_validate_resume_checkpoint_adapter_modes_reject_full_checkpoint(tmp_path: Path, mode: str) -> None:
    ckpt = _make_full_checkpoint(tmp_path / f"ckpt-full-{mode}")
    with pytest.raises(ValueError):
        validate_resume_checkpoint(ckpt, finetune_mode=mode)


def test_validate_resume_checkpoint_requires_trainer_state(tmp_path: Path) -> None:
    ckpt = _make_full_checkpoint(tmp_path / "ckpt-no-state", with_state=False)
    with pytest.raises(ValueError):
        validate_resume_checkpoint(ckpt, finetune_mode="full")
