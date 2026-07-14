from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from shaft.config import RuntimeConfig
from shaft.data import ShaftBatchPlanningSpec, ShaftBatchPlanningState
from shaft.observability import PROGRESS_SNAPSHOT_FILENAME
from shaft.observability import TRAINING_EFFICIENCY_FILENAME
from shaft.training.batch_planning import (
    BATCHING_METADATA_CALLBACK_NAME,
    BATCHING_RUN_METADATA_FILENAME,
    BATCH_PLANNING_CALLBACK_NAME,
    BATCH_PLANNING_CHECKPOINT_COMPLETION_FILENAME,
    ShaftBatchContract,
    ShaftBatchingMetadataCallback,
    ShaftBatchingRunMetadata,
    ShaftBatchPlanningCallback,
    checkpoint_has_batch_planning_state,
    load_batching_run_metadata,
    load_checkpoint_batching_metadata,
    load_batch_planning_state,
    validate_batching_resume_contract,
    validate_batch_planning_resume_contract,
    write_batching_run_metadata,
    write_batch_planning_checkpoint_completion,
)
from shaft.training.checkpointing import (
    ensure_hf_export_layout,
    prune_root_output_layout,
    resolve_best_export_dir,
    resolve_resume_checkpoint,
    validate_resume_checkpoint,
    validate_training_state_policy,
)


def _spec(**changes) -> ShaftBatchPlanningSpec:
    values = {
        "data_world_size": 2,
        "buffer_size": 16,
        "per_device_microbatch_size": 1,
        "max_tokens_per_microbatch": 1024,
        "resource_budgets": (("vision_patches", 2048),),
        "seed": 42,
        "sample_schedule_fingerprint": "schedule-v1",
        "cost_fingerprint": "cost-v1",
    }
    values.update(changes)
    return ShaftBatchPlanningSpec(**values)


def _metadata_for_spec(
    spec: ShaftBatchPlanningSpec,
    *,
    gradient_accumulation_steps: int,
    cost_cache_size: int = 0,
) -> ShaftBatchingRunMetadata:
    return ShaftBatchingRunMetadata(
        grouping="bounded_cost",
        cardinality=spec.cardinality,
        packing="none",
        layout="padded",
        per_device_train_batch_size=spec.per_device_microbatch_size,
        data_world_size=spec.data_world_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        min_pixels=None,
        max_pixels=None,
        source_weights=(("fixture", 1.0),),
        media_snapshot_id="fixture-media-v1",
        buffer_size=spec.buffer_size,
        cost_cache_size=cost_cache_size,
        max_tokens_per_microbatch=spec.max_tokens_per_microbatch,
        resource_budgets=spec.resource_budgets,
        planner_spec_fingerprint=spec.fingerprint,
    )


def _write_bounded_trainer_state(
    path: Path,
    *,
    spec: ShaftBatchPlanningSpec,
    state: ShaftBatchPlanningState,
    resume_contract_fingerprint: str = "resume-v1",
    gradient_accumulation_steps: int = 2,
    global_step: int | None = None,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    resolved_global_step = (
        int(state.global_microstep) // int(gradient_accumulation_steps)
        if global_step is None
        else int(global_step)
    )
    payload = {
        "global_step": resolved_global_step,
        "stateful_callbacks": {
            BATCH_PLANNING_CALLBACK_NAME: {
                "args": {
                    "spec": spec.to_dict(),
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                    "resume_contract_fingerprint": resume_contract_fingerprint,
                },
                "attributes": {"planning_state": state.to_dict()},
            },
            BATCHING_METADATA_CALLBACK_NAME: ShaftBatchingMetadataCallback(
                _metadata_for_spec(
                    spec,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                )
            ).state(),
        }
    }
    (path / "trainer_state.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    _write_exact_resume_artifacts(path, world_size=spec.data_world_size)
    write_batch_planning_checkpoint_completion(path)


def _write_exact_resume_artifacts(path: Path, *, world_size: int) -> None:
    (path / "optimizer.pt").write_bytes(b"optimizer")
    (path / "scheduler.pt").write_bytes(b"scheduler")
    if int(world_size) <= 1:
        (path / "rng_state.pth").write_bytes(b"rng")
        return
    for rank in range(int(world_size)):
        (path / f"rng_state_{rank}.pth").write_bytes(b"rng")


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
        TRAINING_EFFICIENCY_FILENAME,
    )
    for name in names:
        (root / name).write_text("{}", encoding="utf-8")

    prune_root_output_layout(root)

    assert not (root / "config.json").exists()
    assert not (root / "model.safetensors").exists()
    assert all((root / name).is_file() for name in names)


def test_bounded_state_roundtrip_and_resume_validation(tmp_path: Path) -> None:
    spec = _spec()
    state = ShaftBatchPlanningState(
        contract_fingerprint=spec.fingerprint,
        global_microstep=6,
        next_draw_id=12,
        emitted_samples=12,
    )
    _write_bounded_trainer_state(tmp_path, spec=spec, state=state)

    assert load_batch_planning_state(
        tmp_path,
        expected_spec=spec,
        expected_global_step=3,
        gradient_accumulation_steps=2,
        expected_resume_contract_fingerprint="resume-v1",
    ) == state


def test_completion_manifest_binds_the_committed_cursor(tmp_path: Path) -> None:
    spec = _spec(
        cardinality="token_budget",
        per_device_microbatch_size=2,
    )
    committed = ShaftBatchPlanningState(
        contract_fingerprint=spec.fingerprint,
        global_microstep=2,
        next_draw_id=4,
        emitted_samples=4,
    )
    _write_bounded_trainer_state(tmp_path, spec=spec, state=committed)
    replacement = ShaftBatchPlanningState(
        contract_fingerprint=spec.fingerprint,
        global_microstep=2,
        next_draw_id=8,
        emitted_samples=8,
    )
    trainer_state_path = tmp_path / "trainer_state.json"
    trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    trainer_state["stateful_callbacks"][BATCH_PLANNING_CALLBACK_NAME]["attributes"][
        "planning_state"
    ] = replacement.to_dict()
    trainer_state_path.write_text(json.dumps(trainer_state), encoding="utf-8")

    assert not checkpoint_has_batch_planning_state(tmp_path)


def test_bounded_resume_rejects_contract_or_optimizer_boundary_drift(
    tmp_path: Path,
) -> None:
    spec = _spec()
    state = ShaftBatchPlanningState(
        contract_fingerprint=spec.fingerprint,
        global_microstep=4,
        next_draw_id=8,
        emitted_samples=8,
    )
    _write_bounded_trainer_state(tmp_path, spec=spec, state=state)

    with pytest.raises(ValueError, match="changed fields.*buffer_size"):
        load_batch_planning_state(
            tmp_path,
            expected_spec=_spec(buffer_size=32),
            expected_global_step=2,
            gradient_accumulation_steps=2,
            expected_resume_contract_fingerprint="resume-v1",
        )
    with pytest.raises(ValueError, match="changed fields.*cost_fingerprint"):
        load_batch_planning_state(
            tmp_path,
            expected_spec=_spec(cost_fingerprint="cost-v2"),
            expected_global_step=2,
            gradient_accumulation_steps=2,
            expected_resume_contract_fingerprint="resume-v1",
        )
    with pytest.raises(ValueError, match="global_step differs"):
        load_batch_planning_state(
            tmp_path,
            expected_spec=spec,
            expected_global_step=3,
            gradient_accumulation_steps=2,
            expected_resume_contract_fingerprint="resume-v1",
        )

    with pytest.raises(ValueError, match="training contract changed"):
        load_batch_planning_state(
            tmp_path,
            expected_spec=spec,
            expected_global_step=2,
            gradient_accumulation_steps=2,
            expected_resume_contract_fingerprint="changed-resume",
        )


def test_bounded_callback_saves_only_committed_step_state(tmp_path: Path) -> None:
    spec = _spec()
    committed = ShaftBatchPlanningState(
        contract_fingerprint=spec.fingerprint,
        global_microstep=4,
        next_draw_id=8,
        emitted_samples=8,
    )

    class _Sampler:
        committed_state = committed

        def commit_global_microstep(self, global_microstep):
            assert global_microstep == 4
            return self.committed_state

    callback = ShaftBatchPlanningCallback(
        _Sampler(),
        spec,
        gradient_accumulation_steps=2,
        resume_contract_fingerprint="resume-v1",
    )
    control = object()
    state = SimpleNamespace(
        global_step=2,
        max_steps=4,
        epoch=2.0,
        num_train_epochs=4,
        is_world_process_zero=True,
    )
    callback.on_step_end(SimpleNamespace(), state, control)
    assert state.epoch == 0.5
    assert state.num_train_epochs == 1
    checkpoint = tmp_path / "checkpoint-2"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text(
        json.dumps(
            {
                "global_step": 2,
                "stateful_callbacks": {
                    BATCH_PLANNING_CALLBACK_NAME: callback.state(),
                    BATCHING_METADATA_CALLBACK_NAME: ShaftBatchingMetadataCallback(
                        _metadata_for_spec(
                            spec,
                            gradient_accumulation_steps=2,
                        )
                    ).state(),
                }
            }
        ),
        encoding="utf-8",
    )
    _write_exact_resume_artifacts(checkpoint, world_size=spec.data_world_size)
    write_batch_planning_checkpoint_completion(checkpoint)

    assert load_batch_planning_state(
        tmp_path / "checkpoint-2",
        expected_spec=spec,
        expected_global_step=2,
        gradient_accumulation_steps=2,
        expected_resume_contract_fingerprint="resume-v1",
    ) == committed
    assert checkpoint_has_batch_planning_state(checkpoint)


def test_batching_run_metadata_roundtrip(tmp_path: Path) -> None:
    metadata = ShaftBatchingRunMetadata(
        grouping="bounded_cost",
        cardinality="fixed",
        packing="none",
        layout="padded",
        per_device_train_batch_size=1,
        data_world_size=2,
        gradient_accumulation_steps=2,
        min_pixels=200704,
        max_pixels=2_000_000,
        source_weights=(("a", 2.0), ("b", 1.0)),
        media_snapshot_id="banana-media-v1",
        buffer_size=64,
        cost_cache_size=65536,
        max_tokens_per_microbatch=10000,
        resource_budgets=(("vision_patches", 16384),),
        planner_spec_fingerprint="planner-v1",
    )
    assert metadata.to_dict()["batch_contract"] == metadata.batch_contract.to_dict()
    write_batching_run_metadata(tmp_path, metadata)
    assert load_batching_run_metadata(tmp_path) == metadata


def test_batching_run_metadata_rejects_missing_canonical_contract(
    tmp_path: Path,
) -> None:
    metadata = ShaftBatchingRunMetadata(
        grouping="none",
        cardinality="fixed",
        packing="none",
        layout="padded",
        per_device_train_batch_size=1,
        data_world_size=1,
        gradient_accumulation_steps=1,
        min_pixels=None,
        max_pixels=None,
        source_weights=(("a", 1.0),),
    )
    payload = metadata.to_dict()
    payload.pop("batch_contract")
    (tmp_path / BATCHING_RUN_METADATA_FILENAME).write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="canonical batch_contract"):
        load_batching_run_metadata(tmp_path)


def test_batch_contract_canonical_roundtrip_and_fingerprint() -> None:
    contract = ShaftBatchContract(
        grouping="none",
        cardinality="fixed",
        packing="none",
        layout="padded",
        per_device_microbatch_size=2,
        data_world_size=4,
        gradient_accumulation_steps=3,
    )

    assert ShaftBatchContract.from_dict(contract.to_dict()) == contract
    assert ShaftBatchContract.from_dict(contract.to_dict()).fingerprint == contract.fingerprint
    changed = ShaftBatchContract(
        grouping="none",
        cardinality="fixed",
        packing="none",
        layout="padded",
        per_device_microbatch_size=1,
        data_world_size=4,
        gradient_accumulation_steps=3,
    )
    assert changed.fingerprint != contract.fingerprint


def test_length_greedy_varlen_batch_contract_roundtrip() -> None:
    contract = ShaftBatchContract(
        grouping="length",
        cardinality="fixed",
        packing="greedy",
        layout="varlen",
        per_device_microbatch_size=2,
        data_world_size=8,
        gradient_accumulation_steps=4,
        buffer_size=64,
        max_sequence_length=10_000,
        resource_budgets=(("vision_patches", 16_384),),
    )

    restored = ShaftBatchContract.from_dict(contract.to_dict())

    assert restored == contract
    assert restored.is_planned is True
    assert restored.is_bounded is False
    assert restored.local_token_capacity == 20_000


def test_length_batching_metadata_uses_unified_planner_fingerprint() -> None:
    spec = ShaftBatchPlanningSpec(
        grouping="length",
        cardinality="fixed",
        packing="greedy",
        layout="varlen",
        max_sequence_length=128,
        data_world_size=2,
        buffer_size=16,
        per_device_microbatch_size=2,
        max_tokens_per_microbatch=256,
        resource_budgets=(("vision_patches", 4096),),
        seed=42,
        sample_schedule_fingerprint="schedule-v1",
        cost_fingerprint="cost-v1",
    )
    metadata = ShaftBatchingRunMetadata(
        grouping="length",
        cardinality="fixed",
        packing="greedy",
        layout="varlen",
        per_device_train_batch_size=2,
        data_world_size=2,
        gradient_accumulation_steps=4,
        min_pixels=None,
        max_pixels=None,
        source_weights=(("fixture", 1.0),),
        media_snapshot_id="fixture-media-v1",
        buffer_size=16,
        cost_cache_size=32,
        max_sequence_length=128,
        resource_budgets=(("vision_patches", 4096),),
        planner_spec_fingerprint=spec.fingerprint,
    )

    assert metadata.batch_contract.grouping == "length"
    assert metadata.batch_contract.local_token_capacity == 256


def test_cost_cache_size_is_not_part_of_exact_batch_contract() -> None:
    spec = _spec()
    small_cache = _metadata_for_spec(
        spec,
        gradient_accumulation_steps=2,
        cost_cache_size=0,
    )
    large_cache = _metadata_for_spec(
        spec,
        gradient_accumulation_steps=2,
        cost_cache_size=65536,
    )

    assert small_cache.batch_contract == large_cache.batch_contract
    assert (
        small_cache.batch_contract_fingerprint
        == large_cache.batch_contract_fingerprint
    )


def test_token_budget_metadata_reports_pack_count_ranges(tmp_path: Path) -> None:
    metadata = ShaftBatchingRunMetadata(
        grouping="bounded_cost",
        cardinality="token_budget",
        packing="none",
        layout="padded",
        per_device_train_batch_size=2,
        data_world_size=8,
        gradient_accumulation_steps=4,
        min_pixels=200704,
        max_pixels=4_000_000,
        source_weights=(("a", 1.0),),
        media_snapshot_id="media-v1",
        buffer_size=64,
        cost_cache_size=65536,
        max_tokens_per_microbatch=10000,
        resource_budgets=(("vision_patches", 16384),),
        planner_spec_fingerprint="planner-v3",
    )

    payload = metadata.to_dict()

    assert payload["local_pack_count_range"] == [1, 2]
    assert payload["global_pack_count_range"] == [8, 16]
    assert payload["optimizer_pack_count_range"] == [32, 64]
    assert payload["global_pack_count"] is None
    assert payload["optimizer_pack_count"] is None
    write_batching_run_metadata(tmp_path, metadata)
    assert load_batching_run_metadata(tmp_path) == metadata


@pytest.mark.parametrize(
    "field_name",
    ["global_pack_count", "optimizer_pack_count"],
)
def test_batching_run_metadata_rejects_tampered_derived_counts(
    tmp_path: Path,
    field_name: str,
) -> None:
    metadata = ShaftBatchingRunMetadata(
        grouping="none",
        cardinality="fixed",
        packing="none",
        layout="padded",
        per_device_train_batch_size=2,
        data_world_size=4,
        gradient_accumulation_steps=3,
        min_pixels=None,
        max_pixels=None,
        source_weights=(("a", 1.0),),
    )
    payload = metadata.to_dict()
    payload[field_name] += 1
    (tmp_path / BATCHING_RUN_METADATA_FILENAME).write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=field_name):
        load_batching_run_metadata(tmp_path)


def test_batching_run_metadata_rejects_tampered_token_budget_range(
    tmp_path: Path,
) -> None:
    metadata = ShaftBatchingRunMetadata(
        grouping="bounded_cost",
        cardinality="token_budget",
        packing="none",
        layout="padded",
        per_device_train_batch_size=2,
        data_world_size=4,
        gradient_accumulation_steps=3,
        min_pixels=None,
        max_pixels=None,
        source_weights=(("a", 1.0),),
        media_snapshot_id="media-v1",
        buffer_size=8,
        cost_cache_size=32,
        max_tokens_per_microbatch=512,
        planner_spec_fingerprint="planner-v3",
    )
    payload = metadata.to_dict()
    payload["optimizer_pack_count_range"] = [12, 25]
    (tmp_path / BATCHING_RUN_METADATA_FILENAME).write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="optimizer_pack_count_range"):
        load_batching_run_metadata(tmp_path)


def test_batching_run_metadata_rejects_missing_batch_contract_fingerprint(
    tmp_path: Path,
) -> None:
    metadata = ShaftBatchingRunMetadata(
        grouping="none",
        cardinality="fixed",
        packing="none",
        layout="padded",
        per_device_train_batch_size=1,
        data_world_size=1,
        gradient_accumulation_steps=1,
        min_pixels=None,
        max_pixels=None,
        source_weights=(("a", 1.0),),
    )
    payload = metadata.to_dict()
    payload.pop("batch_contract_fingerprint")
    (tmp_path / BATCHING_RUN_METADATA_FILENAME).write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing batch_contract_fingerprint"):
        load_batching_run_metadata(tmp_path)


def test_batching_run_metadata_reuses_executable_batch_contract_validation() -> None:
    with pytest.raises(
        ValueError,
        match="Non-bounded ShaftBatchContract cannot carry bounded planner fields",
    ):
        ShaftBatchingRunMetadata(
            grouping="none",
            cardinality="fixed",
            packing="none",
            layout="padded",
            per_device_train_batch_size=1,
            data_world_size=1,
            gradient_accumulation_steps=1,
            min_pixels=None,
            max_pixels=None,
            source_weights=(("a", 1.0),),
            buffer_size=64,
        )


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
        grouping="none",
        cardinality="fixed",
        packing="none",
        layout="padded",
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
    assert callback.state() == {
        "args": {"metadata": metadata.to_dict()},
        "attributes": {},
    }


def test_batching_metadata_callback_exportable_state_roundtrip() -> None:
    metadata = ShaftBatchingRunMetadata(
        grouping="none",
        cardinality="fixed",
        packing="none",
        layout="padded",
        per_device_train_batch_size=2,
        data_world_size=4,
        gradient_accumulation_steps=3,
        min_pixels=None,
        max_pixels=None,
        source_weights=(("a", 1.0),),
    )
    original = ShaftBatchingMetadataCallback(metadata)

    restored = ShaftBatchingMetadataCallback.from_state(original.state())

    assert restored.metadata == metadata
    assert restored.state() == original.state()


def test_checkpoint_batch_contract_roundtrip_and_resume_drift_rejection(
    tmp_path: Path,
) -> None:
    metadata = ShaftBatchingRunMetadata(
        grouping="none",
        cardinality="fixed",
        packing="none",
        layout="padded",
        per_device_train_batch_size=2,
        data_world_size=1,
        gradient_accumulation_steps=4,
        min_pixels=None,
        max_pixels=None,
        source_weights=(("a", 1.0),),
        sample_execution_fingerprint="sample-v1",
    )
    callback = ShaftBatchingMetadataCallback(metadata)
    (tmp_path / "trainer_state.json").write_text(
        json.dumps(
            {
                "stateful_callbacks": {
                    BATCHING_METADATA_CALLBACK_NAME: callback.state(),
                }
            }
        ),
        encoding="utf-8",
    )

    assert load_checkpoint_batching_metadata(tmp_path) == metadata
    assert validate_batching_resume_contract(
        tmp_path,
        expected_contract=metadata.batch_contract,
    ) == metadata
    assert validate_batching_resume_contract(
        tmp_path,
        expected_contract=metadata.batch_contract,
        expected_sample_execution_fingerprint="sample-v1",
    ) == metadata
    with pytest.raises(ValueError, match="sample execution changed"):
        validate_batching_resume_contract(
            tmp_path,
            expected_contract=metadata.batch_contract,
            expected_sample_execution_fingerprint="sample-v2",
        )
    changed_contract = ShaftBatchContract(
        grouping="none",
        cardinality="fixed",
        packing="none",
        layout="padded",
        per_device_microbatch_size=1,
        data_world_size=1,
        gradient_accumulation_steps=4,
    )
    with pytest.raises(ValueError, match="Training batch contract changed"):
        validate_batching_resume_contract(
            tmp_path,
            expected_contract=changed_contract,
        )


def test_bounded_completion_requires_canonical_metadata_callback(
    tmp_path: Path,
) -> None:
    spec = _spec()
    state = ShaftBatchPlanningState(
        contract_fingerprint=spec.fingerprint,
        global_microstep=2,
        next_draw_id=4,
        emitted_samples=4,
    )
    _write_bounded_trainer_state(tmp_path, spec=spec, state=state)
    trainer_state_path = tmp_path / "trainer_state.json"
    payload = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    payload["stateful_callbacks"].pop(BATCHING_METADATA_CALLBACK_NAME)
    trainer_state_path.write_text(json.dumps(payload), encoding="utf-8")
    (tmp_path / BATCH_PLANNING_CHECKPOINT_COMPLETION_FILENAME).unlink()

    with pytest.raises(ValueError, match=BATCHING_METADATA_CALLBACK_NAME):
        write_batch_planning_checkpoint_completion(tmp_path)
    assert not checkpoint_has_batch_planning_state(tmp_path)


def test_bounded_completion_rejects_planner_metadata_drift(tmp_path: Path) -> None:
    spec = _spec()
    state = ShaftBatchPlanningState(
        contract_fingerprint=spec.fingerprint,
        global_microstep=2,
        next_draw_id=4,
        emitted_samples=4,
    )
    _write_bounded_trainer_state(tmp_path, spec=spec, state=state)
    trainer_state_path = tmp_path / "trainer_state.json"
    payload = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    metadata = payload["stateful_callbacks"][BATCHING_METADATA_CALLBACK_NAME][
        "args"
    ]["metadata"]
    metadata["planner_spec_fingerprint"] = "different-planner"
    trainer_state_path.write_text(json.dumps(payload), encoding="utf-8")

    assert not checkpoint_has_batch_planning_state(tmp_path)


def test_bounded_training_contract_rejects_optimizer_schedule_drift(
    tmp_path: Path,
) -> None:
    spec = _spec()
    state = ShaftBatchPlanningState(
        contract_fingerprint=spec.fingerprint,
        global_microstep=2,
        next_draw_id=4,
        emitted_samples=4,
    )
    _write_bounded_trainer_state(
        tmp_path,
        spec=spec,
        state=state,
        resume_contract_fingerprint="resume-v1",
    )

    with pytest.raises(ValueError, match="training contract changed"):
        validate_batch_planning_resume_contract(
            tmp_path,
            expected_resume_contract_fingerprint="resume-v2",
        )


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
        state=ShaftBatchPlanningState(
            contract_fingerprint=spec.fingerprint,
            global_microstep=2,
            next_draw_id=4,
            emitted_samples=4,
        ),
    )

    assert resolve_resume_checkpoint(
        root,
        require_planning_state=True,
    ) == str(complete)


@pytest.mark.parametrize(
    "corruption",
    [
        "missing_completion",
        "missing_peer_rng",
        "empty_optimizer",
        "step_misalignment",
        "empty_resume_contract",
    ],
)
def test_bounded_resume_resolver_skips_newer_internally_incomplete_checkpoint(
    tmp_path: Path,
    corruption: str,
) -> None:
    root = tmp_path / "run"
    older = root / "checkpoint-1"
    newer = root / "checkpoint-2"
    _write_full_checkpoint(older)
    _write_full_checkpoint(newer)
    spec = _spec()
    older_state = ShaftBatchPlanningState(
        contract_fingerprint=spec.fingerprint,
        global_microstep=2,
        next_draw_id=4,
        emitted_samples=4,
    )
    newer_state = ShaftBatchPlanningState(
        contract_fingerprint=spec.fingerprint,
        global_microstep=4,
        next_draw_id=8,
        emitted_samples=8,
    )
    _write_bounded_trainer_state(older, spec=spec, state=older_state)
    _write_bounded_trainer_state(newer, spec=spec, state=newer_state)

    if corruption == "missing_completion":
        (newer / BATCH_PLANNING_CHECKPOINT_COMPLETION_FILENAME).unlink()
    elif corruption == "missing_peer_rng":
        (newer / "rng_state_1.pth").unlink()
    elif corruption == "empty_optimizer":
        (newer / "optimizer.pt").write_bytes(b"")
    else:
        payload = json.loads((newer / "trainer_state.json").read_text(encoding="utf-8"))
        if corruption == "step_misalignment":
            payload["global_step"] = 3
        else:
            payload["stateful_callbacks"][BATCH_PLANNING_CALLBACK_NAME]["args"][
                "resume_contract_fingerprint"
            ] = ""
        (newer / "trainer_state.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )

    assert not checkpoint_has_batch_planning_state(newer)
    assert resolve_resume_checkpoint(
        root,
        require_planning_state=True,
    ) == str(older)


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
