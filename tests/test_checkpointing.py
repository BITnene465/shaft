from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from shaft.config import RuntimeConfig
from shaft.data import ShaftBoundedBatchingSpec, ShaftBoundedBatchingState
from shaft.observability import PROGRESS_SNAPSHOT_FILENAME
from shaft.training.batch_planning import (
    BATCHING_RUN_METADATA_FILENAME,
    BOUNDED_BATCHING_CALLBACK_NAME,
    ShaftBatchingMetadataCallback,
    ShaftBatchingRunMetadata,
    ShaftBoundedBatchingCallback,
    checkpoint_has_bounded_batching_state,
    load_batching_run_metadata,
    load_bounded_batching_state,
    write_batching_run_metadata,
)
from shaft.training.checkpointing import (
    ensure_hf_export_layout,
    prune_root_output_layout,
    resolve_best_export_dir,
    resolve_resume_checkpoint,
    validate_resume_checkpoint,
    validate_training_state_policy,
)


def _spec(**changes) -> ShaftBoundedBatchingSpec:
    values = {
        "data_world_size": 2,
        "buffer_size": 16,
        "max_samples_per_microbatch": 4,
        "max_padded_tokens": 1024,
        "max_vision_patches": 2048,
        "seed": 42,
        "sample_schedule_fingerprint": "schedule-v1",
        "cost_fingerprint": "cost-v1",
    }
    values.update(changes)
    return ShaftBoundedBatchingSpec(**values)


def _write_bounded_trainer_state(
    path: Path,
    *,
    spec: ShaftBoundedBatchingSpec,
    state: ShaftBoundedBatchingState,
    resume_contract_fingerprint: str = "resume-v1",
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    payload = {
        "stateful_callbacks": {
            BOUNDED_BATCHING_CALLBACK_NAME: {
                "args": {
                    "spec": spec.to_dict(),
                    "gradient_accumulation_steps": 2,
                    "resume_contract_fingerprint": resume_contract_fingerprint,
                },
                "attributes": {"bounded_state": state.to_dict()},
            }
        }
    }
    (path / "trainer_state.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
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
    for step in (1, 2):
        checkpoint = root / f"checkpoint-{step}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "trainer_state.json").write_text("{}", encoding="utf-8")

    assert resolve_resume_checkpoint(root) == str(root / "checkpoint-2")


def test_prune_root_output_layout_preserves_runtime_metadata(tmp_path: Path) -> None:
    root = tmp_path / "run"
    (root / "best").mkdir(parents=True)
    (root / "config.json").write_text("{}", encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"legacy")
    names = (
        "trainer_state.json",
        "shaft_finetune_summary.json",
        "shaft_optimizer_summary.json",
        BATCHING_RUN_METADATA_FILENAME,
        PROGRESS_SNAPSHOT_FILENAME,
    )
    for name in names:
        (root / name).write_text("{}", encoding="utf-8")

    prune_root_output_layout(root)

    assert not (root / "config.json").exists()
    assert not (root / "model.safetensors").exists()
    assert all((root / name).is_file() for name in names)


def test_bounded_state_roundtrip_and_resume_validation(tmp_path: Path) -> None:
    spec = _spec()
    state = ShaftBoundedBatchingState(
        contract_fingerprint=spec.fingerprint,
        global_microstep=6,
        next_draw_id=20,
        emitted_samples=20,
    )
    _write_bounded_trainer_state(tmp_path, spec=spec, state=state)

    assert load_bounded_batching_state(
        tmp_path,
        expected_spec=spec,
        expected_global_step=3,
        gradient_accumulation_steps=2,
        expected_resume_contract_fingerprint="resume-v1",
    ) == state


def test_bounded_resume_rejects_contract_or_optimizer_boundary_drift(
    tmp_path: Path,
) -> None:
    spec = _spec()
    state = ShaftBoundedBatchingState(
        contract_fingerprint=spec.fingerprint,
        global_microstep=4,
        next_draw_id=10,
        emitted_samples=10,
    )
    _write_bounded_trainer_state(tmp_path, spec=spec, state=state)

    with pytest.raises(ValueError, match="changed fields.*buffer_size"):
        load_bounded_batching_state(
            tmp_path,
            expected_spec=_spec(buffer_size=32),
            expected_global_step=2,
            gradient_accumulation_steps=2,
            expected_resume_contract_fingerprint="resume-v1",
        )
    with pytest.raises(ValueError, match="not aligned"):
        load_bounded_batching_state(
            tmp_path,
            expected_spec=spec,
            expected_global_step=3,
            gradient_accumulation_steps=2,
            expected_resume_contract_fingerprint="resume-v1",
        )

    with pytest.raises(ValueError, match="training contract changed"):
        load_bounded_batching_state(
            tmp_path,
            expected_spec=spec,
            expected_global_step=2,
            gradient_accumulation_steps=2,
            expected_resume_contract_fingerprint="changed-resume",
        )


def test_bounded_callback_saves_only_committed_step_state(tmp_path: Path) -> None:
    spec = _spec()
    committed = ShaftBoundedBatchingState(
        contract_fingerprint=spec.fingerprint,
        global_microstep=4,
        next_draw_id=12,
        emitted_samples=12,
    )

    class _Sampler:
        committed_state = committed

        def commit_global_microstep(self, global_microstep):
            assert global_microstep == 4
            return self.committed_state

    callback = ShaftBoundedBatchingCallback(
        _Sampler(),
        spec,
        gradient_accumulation_steps=2,
        resume_contract_fingerprint="resume-v1",
    )
    control = object()
    state = SimpleNamespace(global_step=2, is_world_process_zero=True)
    callback.on_step_end(SimpleNamespace(), state, control)
    checkpoint = tmp_path / "checkpoint-2"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text(
        json.dumps(
            {
                "stateful_callbacks": {
                    BOUNDED_BATCHING_CALLBACK_NAME: callback.state()
                }
            }
        ),
        encoding="utf-8",
    )

    assert load_bounded_batching_state(
        tmp_path / "checkpoint-2",
        expected_spec=spec,
        expected_global_step=2,
        gradient_accumulation_steps=2,
        expected_resume_contract_fingerprint="resume-v1",
    ) == committed
    assert checkpoint_has_bounded_batching_state(checkpoint)


def test_batching_run_metadata_roundtrip(tmp_path: Path) -> None:
    metadata = ShaftBatchingRunMetadata(
        strategy="bounded_cost_aware",
        per_device_train_batch_size=1,
        data_world_size=2,
        gradient_accumulation_steps=2,
        min_pixels=200704,
        max_pixels=2_000_000,
        source_weights=(("a", 2.0), ("b", 1.0)),
        media_snapshot_id="banana-media-v1",
        buffer_size=64,
        cost_cache_size=65536,
        max_samples_per_microbatch=4,
        max_padded_tokens=10000,
        max_vision_patches=16384,
        contract_fingerprint="contract-v1",
    )
    write_batching_run_metadata(tmp_path, metadata)
    assert load_batching_run_metadata(tmp_path) == metadata


def test_batching_metadata_callback_publishes_wandb_config(monkeypatch) -> None:
    updates = []
    run = SimpleNamespace(
        config=SimpleNamespace(
            update=lambda payload, allow_val_change: updates.append(
                (payload, allow_val_change)
            )
        )
    )
    monkeypatch.setitem(__import__("sys").modules, "wandb", SimpleNamespace(run=run))
    metadata = ShaftBatchingRunMetadata(
        strategy="fixed",
        per_device_train_batch_size=1,
        data_world_size=1,
        gradient_accumulation_steps=1,
        min_pixels=None,
        max_pixels=None,
        source_weights=(("a", 1.0),),
    )
    callback = ShaftBatchingMetadataCallback(metadata)
    callback.on_train_begin(
        SimpleNamespace(report_to=["wandb"]),
        SimpleNamespace(is_world_process_zero=True),
        object(),
    )
    assert updates == [({"shaft_batching": metadata.to_dict()}, True)]


def _write_full_checkpoint(path: Path, *, trainer_state: bool = True) -> None:
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"model")
    if trainer_state:
        (path / "trainer_state.json").write_text("{}", encoding="utf-8")


def _write_adapter_checkpoint(path: Path, *, trainer_state: bool = True) -> None:
    path.mkdir(parents=True)
    (path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (path / "adapter_model.safetensors").write_bytes(b"adapter")
    if trainer_state:
        (path / "trainer_state.json").write_text("{}", encoding="utf-8")


def test_bounded_resume_resolver_skips_newer_incomplete_checkpoint(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    complete = root / "checkpoint-1"
    incomplete = root / "checkpoint-2"
    _write_full_checkpoint(complete)
    _write_full_checkpoint(incomplete)
    spec = _spec()
    _write_bounded_trainer_state(
        complete,
        spec=spec,
        state=ShaftBoundedBatchingState(
            contract_fingerprint=spec.fingerprint,
            global_microstep=2,
            next_draw_id=4,
            emitted_samples=4,
        ),
    )

    assert resolve_resume_checkpoint(
        root,
        require_bounded_state=True,
    ) == str(complete)


def test_ensure_hf_export_layout_and_best_dir(tmp_path: Path) -> None:
    full = tmp_path / "full"
    _write_full_checkpoint(full, trainer_state=False)
    ensure_hf_export_layout(full, finetune_mode="full")
    assert resolve_best_export_dir(tmp_path) == tmp_path / "best"


def test_ensure_hf_export_layout_validates_model_specific_files(tmp_path: Path) -> None:
    full = tmp_path / "full"
    _write_full_checkpoint(full, trainer_state=False)
    model_meta = SimpleNamespace(required_saved_files=lambda: ("processor_config.json",))
    with pytest.raises(ValueError, match="Missing additional saved files"):
        ensure_hf_export_layout(full, finetune_mode="full", model_meta=model_meta)


@pytest.mark.parametrize("mode", ["lora", "dora", "qlora"])
def test_validate_resume_checkpoint_accepts_matching_adapter_mode(
    tmp_path: Path,
    mode: str,
) -> None:
    checkpoint = tmp_path / mode
    _write_adapter_checkpoint(checkpoint)
    validate_resume_checkpoint(checkpoint, finetune_mode=mode)


def test_validate_resume_checkpoint_rejects_mismatched_or_missing_state(
    tmp_path: Path,
) -> None:
    full = tmp_path / "full"
    _write_full_checkpoint(full)
    with pytest.raises(ValueError, match="adapter"):
        validate_resume_checkpoint(full, finetune_mode="lora")

    export = tmp_path / "export"
    _write_full_checkpoint(export, trainer_state=False)
    with pytest.raises(ValueError, match="trainer_state"):
        validate_resume_checkpoint(export, finetune_mode="full")
