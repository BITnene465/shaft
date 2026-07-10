from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from shaft.config import RuntimeConfig
from shaft.data import ShaftBatchPlanningSignature, ShaftFixedBatchPlanningSpec
from shaft.data.cost_plan import COST_PLAN_REFERENCE_FILENAME
from shaft.model import build_model_meta
from shaft.training.batch_planning import (
    BATCH_PLANNING_SIGNATURE_FILENAME,
    ShaftBatchPlanningCallback,
    load_batch_planning_signature,
    validate_batch_planning_resume,
    validate_batch_planning_resume_geometry,
    write_batch_planning_signature,
)
from shaft.training.checkpointing import (
    ensure_hf_export_layout,
    prune_root_output_layout,
    resolve_best_export_dir,
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


def test_resolve_resume_checkpoint_ignores_root_final_state_when_checkpoints_exist(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    checkpoint = root / "checkpoint-2"
    checkpoint.mkdir(parents=True)
    (root / "trainer_state.json").write_text("{}", encoding="utf-8")
    (checkpoint / "trainer_state.json").write_text("{}", encoding="utf-8")

    assert resolve_resume_checkpoint(root) == str(checkpoint)


def test_prune_root_output_layout_preserves_run_metadata(tmp_path: Path) -> None:
    root = tmp_path / "run"
    (root / "best").mkdir(parents=True)
    (root / "config.json").write_text("{}", encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"legacy-root-export")
    metadata = {
        "trainer_state.json": "{}",
        "shaft_finetune_summary.json": "{}",
        "shaft_optimizer_summary.json": "{}",
        BATCH_PLANNING_SIGNATURE_FILENAME: "{}",
        COST_PLAN_REFERENCE_FILENAME: "{}",
    }
    for name, payload in metadata.items():
        (root / name).write_text(payload, encoding="utf-8")

    prune_root_output_layout(root)

    assert (root / "best").is_dir()
    assert not (root / "config.json").exists()
    assert not (root / "model.safetensors").exists()
    for name in metadata:
        assert (root / name).is_file()


def _batch_planning_signature() -> ShaftBatchPlanningSignature:
    return ShaftBatchPlanningSignature(
        planner_version="planner-v1",
        sample_plan_fingerprint="sample-v1",
        cost_fingerprint="cost-v1",
        source_sample_count=8,
        sample_count=8,
        per_device_batch_size=2,
        data_world_size=2,
        gradient_accumulation_steps=1,
        planning_window=8,
        effective_planning_window=8,
        seed=42,
        drop_last=False,
    )


def test_batch_planning_signature_roundtrip_and_resume_validation(tmp_path: Path) -> None:
    signature = _batch_planning_signature()
    checkpoint = tmp_path / "run" / "checkpoint-1"
    write_batch_planning_signature(checkpoint, signature)

    assert load_batch_planning_signature(checkpoint) == signature
    validate_batch_planning_resume(checkpoint, expected=signature)

    extended_horizon = replace(
        signature,
        source_sample_count=12,
        sample_count=12,
    )
    with pytest.raises(ValueError, match="source_sample_count.*sample_count"):
        validate_batch_planning_resume(checkpoint, expected=extended_horizon)
    with pytest.raises(ValueError, match="planning geometry changed"):
        validate_batch_planning_resume_geometry(
            checkpoint,
            expected=ShaftFixedBatchPlanningSpec(
                sample_plan_fingerprint=signature.sample_plan_fingerprint,
                source_sample_count=12,
                usable_sample_count=12,
                per_device_batch_size=signature.per_device_batch_size,
                data_world_size=signature.data_world_size,
                gradient_accumulation_steps=signature.gradient_accumulation_steps,
                global_microstep_samples=4,
                planning_window=signature.planning_window,
                effective_planning_window=signature.effective_planning_window,
                seed=signature.seed,
                drop_last=signature.drop_last,
            ),
        )


def test_batch_planning_callback_persists_checkpoint_signature(tmp_path: Path) -> None:
    signature = _batch_planning_signature()
    callback = ShaftBatchPlanningCallback(signature)
    checkpoint = tmp_path / "checkpoint-3"
    checkpoint.mkdir()

    control = object()
    returned = callback.on_save(
        SimpleNamespace(output_dir=str(tmp_path)),
        SimpleNamespace(global_step=3, is_world_process_zero=True),
        control,
    )

    assert returned is control
    assert load_batch_planning_signature(checkpoint) == signature


def test_batch_planning_resume_never_borrows_parent_signature(tmp_path: Path) -> None:
    signature = _batch_planning_signature()
    run_root = tmp_path / "run"
    checkpoint = run_root / "checkpoint-1"
    checkpoint.mkdir(parents=True)
    write_batch_planning_signature(run_root, signature)

    with pytest.raises(FileNotFoundError, match="checkpoint-1"):
        validate_batch_planning_resume(checkpoint, expected=signature)


def test_batch_planning_resume_from_run_root_validates_latest_checkpoint(
    tmp_path: Path,
) -> None:
    signature = _batch_planning_signature()
    run_root = tmp_path / "run"
    checkpoint = run_root / "checkpoint-2"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text("{}", encoding="utf-8")
    write_batch_planning_signature(
        run_root,
        replace(signature, source_sample_count=12, sample_count=12),
    )
    write_batch_planning_signature(checkpoint, signature)

    validate_batch_planning_resume(run_root, expected=signature)


def test_ensure_hf_export_layout_full(tmp_path: Path) -> None:
    export_dir = tmp_path / "full"
    export_dir.mkdir()
    (export_dir / "config.json").write_text("{}", encoding="utf-8")
    (export_dir / "model.safetensors").write_bytes(b"ok")
    ensure_hf_export_layout(export_dir, finetune_mode="full")


def test_resolve_best_export_dir(tmp_path: Path) -> None:
    assert resolve_best_export_dir(tmp_path) == tmp_path / "best"
    assert resolve_best_export_dir(f"{tmp_path}") == tmp_path / "best"


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
