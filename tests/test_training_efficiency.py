from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest
import torch

from shaft.data import (
    ShaftCollatedBatchStats,
    ShaftVarlenLayoutPlan,
    ShaftVarlenSegmentLayout,
)
from shaft.observability import (
    TRAINING_EFFICIENCY_FILENAME,
    TRAINING_EFFICIENCY_SCHEMA_VERSION,
    ShaftEfficiencyFrame,
    ShaftTrainingEfficiencyContract,
)
from shaft.cli.efficiency import build_comparison
from shaft.training.efficiency import (
    ShaftTrainingEfficiencyMonitor,
    invalidate_training_efficiency_summary,
    prepare_training_efficiency_checkpoint,
)


def _stats(
    *,
    useful: int = 6,
    materialized: int = 8,
    supervised: int = 3,
    vision: int | None = 4,
) -> ShaftCollatedBatchStats:
    return ShaftCollatedBatchStats(
        logical_segments=2,
        physical_packs=1,
        useful_tokens=useful,
        materialized_tokens=materialized,
        supervised_tokens=supervised,
        weighted_supervision_mass=float(supervised),
        sequence_length_sum=useful,
        sequence_length_square_sum=useful * useful,
        vision_patches=vision,
    )


def _contract(**overrides) -> ShaftTrainingEfficiencyContract:
    values = {
        "algorithm": "sft",
        "model_type": "qwen3vl",
        "model_name_or_path": "models/Qwen3-VL-4B",
        "model_plan_fingerprint": "model-plan-v1",
        "finetune_mode": "full",
        "torch_dtype": "bfloat16",
        "attention_implementation": "flash_attention_2",
        "seed": 42,
        "max_steps": 10,
        "num_train_epochs": 1.0,
        "data_world_size": 2,
        "gradient_accumulation_steps": 1,
        "max_length": 4096,
        "min_pixels": 200704,
        "max_pixels": 4000000,
        "optimizer_name": "adamw_torch",
        "scheduler_name": "cosine",
        "learning_rate": 1e-5,
        "source_fingerprint": "source-v1",
        "source_contract_complete": True,
        "sample_execution_fingerprint": "execution-v1",
        "software_fingerprint": "software-v1",
        "hardware_fingerprint": "hardware-v1",
        "measurement_protocol": "shaft-efficiency-optimizer-frame-v2",
        "timing_mode": "host_optimizer_frame",
        "batch_contract_fingerprint": "batch-v1",
        "sequence_contract_fingerprint": "sequence-v1",
    }
    values.update(overrides)
    return ShaftTrainingEfficiencyContract(**values)


def test_efficiency_monitor_commits_only_successful_optimizer_frames(
    tmp_path: Path,
) -> None:
    monitor = ShaftTrainingEfficiencyMonitor(output_dir=tmp_path)
    monitor.stage(
        [{"_shaft_batch_stats": _stats()}, {"_shaft_batch_stats": _stats()}],
        host_batch_acquire_seconds=0.25,
    )
    monitor.record_training_step(0.5)
    monitor.record_training_step(0.75)
    monitor.commit(global_step=1, update_applied=True)

    monitor.stage(
        [{"_shaft_batch_stats": _stats()}],
        host_batch_acquire_seconds=10.0,
    )
    monitor.discard()

    summary, metrics = monitor.finalize(
        final_global_step=1,
        device=torch.device("cpu"),
    )
    aggregate = summary.aggregate
    assert aggregate is not None
    assert aggregate.optimizer_steps == 1
    assert aggregate.microbatches == 2
    assert aggregate.logical_segments == 4
    assert aggregate.physical_packs == 2
    assert aggregate.useful_tokens == 12
    assert aggregate.materialized_tokens == 16
    assert aggregate.host_batch_acquire_seconds == 0.25
    assert aggregate.training_step_seconds == 1.25
    assert metrics["train_efficiency/padding_fraction"] == 0.25

    payload = json.loads(
        (tmp_path / TRAINING_EFFICIENCY_FILENAME).read_text(encoding="utf-8")
    )
    assert payload["schema_version"] == TRAINING_EFFICIENCY_SCHEMA_VERSION
    assert payload["final_global_step"] == 1
    assert payload["aggregate"]["useful_tokens"] == 12
    comparison = build_comparison([tmp_path])
    assert comparison[0]["steps"] == 1
    assert comparison[0]["padding_fraction"] == 0.25


def test_efficiency_resume_attempt_has_explicit_partial_coverage(tmp_path: Path) -> None:
    monitor = ShaftTrainingEfficiencyMonitor(
        output_dir=tmp_path,
        initial_global_step=2,
    )
    monitor.stage(
        [{"_shaft_batch_stats": _stats(vision=None)}],
        host_batch_acquire_seconds=0.0,
    )
    monitor.record_training_step(0.1)
    monitor.commit(global_step=3, update_applied=False)

    summary, _ = monitor.finalize(
        final_global_step=3,
        device=torch.device("cpu"),
    )
    assert summary.initial_global_step == 2
    assert summary.final_global_step == 3
    assert summary.complete_history is False
    assert summary.aggregate is not None
    assert summary.aggregate.update_applied_steps == 0
    assert summary.aggregate.vision_coverage_batches == 0


def test_efficiency_checkpoint_snapshot_restores_without_double_counting(
    tmp_path: Path,
) -> None:
    first = ShaftTrainingEfficiencyMonitor(output_dir=tmp_path)
    for step in (1, 2):
        first.stage(
            [{"_shaft_batch_stats": _stats()}],
            host_batch_acquire_seconds=0.01,
        )
        first.record_training_step(0.02)
        first.commit(global_step=step)
    checkpoint = tmp_path / "checkpoint-2"
    prepare_training_efficiency_checkpoint(
        checkpoint,
        global_step=2,
        generation=first.snapshot_generation,
    )
    first.write_checkpoint_snapshot(checkpoint, global_step=2)
    transaction = json.loads(
        (
            checkpoint
            / "shaft_training_efficiency_checkpoint_transaction.json"
        ).read_text(encoding="utf-8")
    )
    assert transaction["state"] == "committed"
    assert transaction["generation"] == first.snapshot_generation

    resumed = ShaftTrainingEfficiencyMonitor.from_checkpoint(
        output_dir=tmp_path,
        checkpoint_dir=checkpoint,
        checkpoint_global_step=2,
    )
    resumed.stage(
        [{"_shaft_batch_stats": _stats()}],
        host_batch_acquire_seconds=0.01,
    )
    resumed.record_training_step(0.02)
    resumed.commit(global_step=3)
    summary, _ = resumed.finalize(
        final_global_step=3,
        device=torch.device("cpu"),
    )
    assert summary.complete_history is True
    assert summary.aggregate is not None
    assert summary.aggregate.optimizer_steps == 3
    assert summary.aggregate.useful_tokens == 18


def test_efficiency_checkpoint_contract_mismatch_restarts_partial_coverage(
    tmp_path: Path,
) -> None:
    first = ShaftTrainingEfficiencyMonitor(
        output_dir=tmp_path,
        contract=_contract(),
    )
    first.stage(
        [{"_shaft_batch_stats": _stats()}],
        host_batch_acquire_seconds=0.01,
    )
    first.record_training_step(0.02)
    first.commit(global_step=1)
    checkpoint = tmp_path / "checkpoint-1"
    prepare_training_efficiency_checkpoint(
        checkpoint,
        global_step=1,
        generation=first.snapshot_generation,
    )
    first.write_checkpoint_snapshot(checkpoint, global_step=1)

    resumed = ShaftTrainingEfficiencyMonitor.from_checkpoint(
        output_dir=tmp_path,
        checkpoint_dir=checkpoint,
        checkpoint_global_step=1,
        contract=replace(_contract(), seed=99),
    )

    assert resumed.complete_history is False
    assert resumed.committed_frames == ()
    resumed.stage(
        [{"_shaft_batch_stats": _stats()}],
        host_batch_acquire_seconds=0.01,
    )
    resumed.record_training_step(0.02)
    resumed.commit(global_step=2)


def test_efficiency_checkpoint_snapshot_set_revokes_stale_generation(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint-1"
    checkpoint.mkdir()
    stale_snapshot = checkpoint / "shaft_training_efficiency_rank7.json"
    stale_snapshot.write_text('{"generation": "stale"}\n', encoding="utf-8")
    stale_manifest = checkpoint / "shaft_training_efficiency_snapshot_set.json"
    stale_manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "global_step": 1,
                "world_size": 8,
                "generation": "stale",
            }
        ),
        encoding="utf-8",
    )
    monitor = ShaftTrainingEfficiencyMonitor(output_dir=tmp_path)
    monitor.stage(
        [{"_shaft_batch_stats": _stats()}],
        host_batch_acquire_seconds=0.0,
    )
    monitor.record_training_step(0.01)
    monitor.commit(global_step=1)
    prepare_training_efficiency_checkpoint(
        checkpoint,
        global_step=1,
        generation=monitor.snapshot_generation,
    )

    monitor.write_checkpoint_snapshot(checkpoint, global_step=1)

    assert not stale_snapshot.exists()
    manifest = json.loads(stale_manifest.read_text(encoding="utf-8"))
    snapshot = json.loads(
        (checkpoint / "shaft_training_efficiency_rank0.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["generation"] == monitor.snapshot_generation
    assert snapshot["generation"] == monitor.snapshot_generation


def test_efficiency_checkpoint_rejects_snapshot_generation_mismatch(
    tmp_path: Path,
) -> None:
    monitor = ShaftTrainingEfficiencyMonitor(output_dir=tmp_path)
    monitor.stage(
        [{"_shaft_batch_stats": _stats()}],
        host_batch_acquire_seconds=0.0,
    )
    monitor.record_training_step(0.01)
    monitor.commit(global_step=1)
    checkpoint = tmp_path / "checkpoint-1"
    prepare_training_efficiency_checkpoint(
        checkpoint,
        global_step=1,
        generation=monitor.snapshot_generation,
    )
    monitor.write_checkpoint_snapshot(checkpoint, global_step=1)
    manifest_path = checkpoint / "shaft_training_efficiency_snapshot_set.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["generation"] = "different-generation"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    resumed = ShaftTrainingEfficiencyMonitor.from_checkpoint(
        output_dir=tmp_path,
        checkpoint_dir=checkpoint,
        checkpoint_global_step=1,
    )

    assert resumed.complete_history is False
    assert resumed.committed_frames == ()


def test_efficiency_checkpoint_rejects_non_contiguous_or_invalid_frames(
    tmp_path: Path,
) -> None:
    monitor = ShaftTrainingEfficiencyMonitor(output_dir=tmp_path)
    for step in (1, 2):
        monitor.stage(
            [{"_shaft_batch_stats": _stats()}],
            host_batch_acquire_seconds=0.01,
        )
        monitor.record_training_step(0.02)
        monitor.commit(global_step=step)
    checkpoint = tmp_path / "checkpoint-2"
    prepare_training_efficiency_checkpoint(
        checkpoint,
        global_step=2,
        generation=monitor.snapshot_generation,
    )
    snapshot = monitor.write_checkpoint_snapshot(checkpoint, global_step=2)
    assert snapshot is not None
    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    payload["frames"] = [payload["frames"][1]]
    snapshot.write_text(json.dumps(payload), encoding="utf-8")

    resumed = ShaftTrainingEfficiencyMonitor.from_checkpoint(
        output_dir=tmp_path,
        checkpoint_dir=checkpoint,
        checkpoint_global_step=2,
    )

    assert resumed.initial_global_step == 2
    assert resumed.complete_history is False
    assert resumed.committed_frames == ()


def test_efficiency_frame_uses_device_optimizer_frame_for_critical_path() -> None:
    frame = ShaftEfficiencyFrame(
        global_step=1,
        logical_segments=1,
        physical_packs=1,
        useful_tokens=4,
        materialized_tokens=4,
        supervised_tokens=2,
        weighted_supervision_mass=2.0,
        weighted_supervision_coverage_microbatches=1,
        sequence_length_sum=4,
        sequence_length_square_sum=16,
        vision_patches=0,
        vision_coverage_batches=1,
        microbatches=1,
        host_batch_acquire_seconds=0.1,
        batch_prepare_seconds=0.2,
        training_step_seconds=99.0,
        optimizer_step_seconds=88.0,
        device_training_seconds=0.3,
        update_applied=True,
    )

    assert frame.critical_path_seconds == pytest.approx(0.6)


def test_efficiency_summary_rejects_missing_cuda_event_coverage(
    tmp_path: Path,
) -> None:
    monitor = ShaftTrainingEfficiencyMonitor(
        output_dir=tmp_path,
        device_timing=True,
        persist=False,
    )
    monitor.stage(
        [{"_shaft_batch_stats": _stats()}],
        host_batch_acquire_seconds=0.01,
    )
    monitor.record_training_step(0.02)
    monitor.commit(global_step=1)

    with pytest.raises(RuntimeError, match="CUDA event coverage"):
        monitor.finalize(final_global_step=1, device=torch.device("cuda"))


def test_efficiency_cuda_events_cover_the_whole_optimizer_frame(
    tmp_path: Path,
) -> None:
    operations = []

    class _FakeEvent:
        def __init__(self, *, enable_timing: bool) -> None:
            assert enable_timing is True
            self.index = len([item for item in operations if item.startswith("create")])
            operations.append(f"create-{self.index}")

        def record(self) -> None:
            operations.append(f"record-{self.index}")

        def synchronize(self) -> None:
            operations.append(f"synchronize-{self.index}")

        def elapsed_time(self, other) -> float:
            operations.append(f"elapsed-{self.index}-{other.index}")
            return 250.0

    monitor = ShaftTrainingEfficiencyMonitor(
        output_dir=tmp_path,
        device_timing=True,
        persist=False,
    )
    monitor.stage(
        [
            {"_shaft_batch_stats": _stats()},
            {"_shaft_batch_stats": _stats()},
        ],
        host_batch_acquire_seconds=0.01,
    )
    with patch("shaft.training.efficiency.torch.cuda.Event", _FakeEvent):
        monitor.start_device_frame()
        monitor.record_training_step(0.02)
        monitor.start_device_frame()
        monitor.record_training_step(0.03)
        monitor.start_optimizer_step()
        monitor.finish_optimizer_step()
        monitor.commit(global_step=1)
        summary, _ = monitor.finalize(
            final_global_step=1,
            device=torch.device("cuda"),
        )

    assert operations[:4] == ["create-0", "record-0", "create-1", "record-1"]
    assert operations[-2:] == ["synchronize-1", "elapsed-0-1"]
    assert summary.aggregate is not None
    assert summary.aggregate.microbatches == 2
    assert summary.aggregate.device_training_seconds == pytest.approx(0.25)


def test_pipeline_owned_efficiency_invalidation_removes_stale_root_summary(
    tmp_path: Path,
) -> None:
    stale = tmp_path / TRAINING_EFFICIENCY_FILENAME
    stale.write_text('{"stale": true}\n', encoding="utf-8")

    invalidate_training_efficiency_summary(tmp_path)

    assert not stale.exists()


def test_preinit_nonzero_rank_does_not_race_root_summary_invalidation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stale = tmp_path / TRAINING_EFFICIENCY_FILENAME
    stale.write_text('{"stale": true}\n', encoding="utf-8")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")

    invalidate_training_efficiency_summary(tmp_path)

    assert stale.exists()


def test_efficiency_persist_false_revokes_stale_checkpoint_snapshot_set(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint-1"
    checkpoint.mkdir()
    stale_snapshot = checkpoint / "shaft_training_efficiency_rank0.json"
    stale_manifest = checkpoint / "shaft_training_efficiency_snapshot_set.json"
    stale_snapshot.write_text('{"generation": "stale"}\n', encoding="utf-8")
    stale_manifest.write_text('{"generation": "stale"}\n', encoding="utf-8")
    monitor = ShaftTrainingEfficiencyMonitor(output_dir=tmp_path, persist=False)
    prepare_training_efficiency_checkpoint(
        checkpoint,
        global_step=1,
        generation=None,
    )

    result = monitor.write_checkpoint_snapshot(checkpoint, global_step=1)

    assert result is None
    assert not stale_snapshot.exists()
    assert not stale_manifest.exists()
    transaction = json.loads(
        (
            checkpoint
            / "shaft_training_efficiency_checkpoint_transaction.json"
        ).read_text(encoding="utf-8")
    )
    assert transaction["state"] == "revoked"
    assert transaction["generation"] is None


def test_efficiency_comparator_enforces_identity_but_allows_batch_axes(
    tmp_path: Path,
) -> None:
    paths = []
    for name, contract in (
        ("padded", _contract()),
        (
            "varlen",
            replace(
                _contract(),
                batch_contract_fingerprint="batch-v2",
                sequence_contract_fingerprint="sequence-v2",
            ),
        ),
    ):
        output_dir = tmp_path / name
        monitor = ShaftTrainingEfficiencyMonitor(
            output_dir=output_dir,
            contract=contract,
        )
        monitor.stage(
            [{"_shaft_batch_stats": _stats()}],
            host_batch_acquire_seconds=0.01,
        )
        monitor.record_training_step(0.02)
        monitor.commit(global_step=1)
        monitor.finalize(final_global_step=1, device=torch.device("cpu"))
        paths.append(output_dir)

    assert len(build_comparison(paths)) == 2

    incompatible_dir = tmp_path / "other-model"
    monitor = ShaftTrainingEfficiencyMonitor(
        output_dir=incompatible_dir,
        contract=replace(_contract(), model_type="qwen35vl"),
    )
    monitor.stage(
        [{"_shaft_batch_stats": _stats()}],
        host_batch_acquire_seconds=0.01,
    )
    monitor.record_training_step(0.02)
    monitor.commit(global_step=1)
    monitor.finalize(final_global_step=1, device=torch.device("cpu"))

    with pytest.raises(ValueError, match="model_type"):
        build_comparison([paths[0], incompatible_dir])
    assert len(
        build_comparison(
            [paths[0], incompatible_dir],
            allow_incompatible=True,
        )
    ) == 2


def test_efficiency_comparator_rejects_incomplete_source_identity(
    tmp_path: Path,
) -> None:
    paths = []
    for name in ("first", "second"):
        output_dir = tmp_path / name
        monitor = ShaftTrainingEfficiencyMonitor(
            output_dir=output_dir,
            contract=replace(_contract(), source_contract_complete=False),
        )
        monitor.stage(
            [{"_shaft_batch_stats": _stats()}],
            host_batch_acquire_seconds=0.0,
        )
        monitor.record_training_step(0.01)
        monitor.commit(global_step=1)
        monitor.finalize(final_global_step=1, device=torch.device("cpu"))
        paths.append(output_dir)

    with pytest.raises(ValueError, match="incomplete source identity"):
        build_comparison(paths)


def test_efficiency_comparator_rejects_different_committed_workloads(
    tmp_path: Path,
) -> None:
    paths = []
    for name, useful in (("first", 6), ("second", 7)):
        output_dir = tmp_path / name
        monitor = ShaftTrainingEfficiencyMonitor(
            output_dir=output_dir,
            contract=_contract(),
        )
        monitor.stage(
            [
                {
                    "_shaft_batch_stats": _stats(
                        useful=useful,
                        materialized=useful,
                    )
                }
            ],
            host_batch_acquire_seconds=0.0,
        )
        monitor.record_training_step(0.01)
        monitor.commit(global_step=1)
        monitor.finalize(final_global_step=1, device=torch.device("cpu"))
        paths.append(output_dir)

    with pytest.raises(ValueError, match="committed workload"):
        build_comparison(paths)


def test_efficiency_comparator_tolerates_roundoff_but_checks_applied_updates(
    tmp_path: Path,
) -> None:
    paths = []
    for name in ("first", "second"):
        output_dir = tmp_path / name
        monitor = ShaftTrainingEfficiencyMonitor(
            output_dir=output_dir,
            contract=_contract(),
        )
        monitor.stage(
            [{"_shaft_batch_stats": _stats()}],
            host_batch_acquire_seconds=0.0,
        )
        monitor.record_training_step(0.01)
        monitor.commit(global_step=1)
        monitor.finalize(final_global_step=1, device=torch.device("cpu"))
        paths.append(output_dir)

    second_path = paths[1] / TRAINING_EFFICIENCY_FILENAME
    payload = json.loads(second_path.read_text(encoding="utf-8"))
    payload["aggregate"]["weighted_supervision_mass"] += 5e-7
    second_path.write_text(json.dumps(payload), encoding="utf-8")
    assert len(build_comparison(paths)) == 2

    payload["aggregate"]["update_applied_steps"] = 0
    second_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="update_applied_steps"):
        build_comparison(paths)


def test_efficiency_comparator_rejects_different_timing_protocols(
    tmp_path: Path,
) -> None:
    paths = []
    for name, timing_mode in (
        ("host", "host_optimizer_frame"),
        ("cuda", "cuda_optimizer_frame"),
    ):
        output_dir = tmp_path / name
        monitor = ShaftTrainingEfficiencyMonitor(
            output_dir=output_dir,
            contract=replace(_contract(), timing_mode=timing_mode),
        )
        monitor.stage(
            [{"_shaft_batch_stats": _stats()}],
            host_batch_acquire_seconds=0.0,
        )
        monitor.record_training_step(0.01)
        monitor.commit(global_step=1)
        monitor.finalize(final_global_step=1, device=torch.device("cpu"))
        paths.append(output_dir)

    with pytest.raises(ValueError, match="timing_mode"):
        build_comparison(paths)


def test_collated_batch_stats_use_actual_shifted_training_tensors() -> None:
    stats = ShaftCollatedBatchStats.from_training_inputs(
        sequence_inputs={
            "input_ids": torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
            "labels": torch.tensor([[-100, 2, 3, -100], [-100, 5, -100, -100]]),
            "loss_scale": torch.tensor(
                [[0.0, 1.0, 0.5, 0.0], [0.0, 2.0, 0.0, 0.0]]
            ),
        },
        vision_patches=None,
    )
    assert stats.logical_segments == 2
    assert stats.physical_packs == 2
    assert stats.useful_tokens == 5
    assert stats.materialized_tokens == 8
    assert stats.supervised_tokens == 3
    assert stats.weighted_supervision_mass == 3.5
    assert stats.sequence_length_sum == 5
    assert stats.sequence_length_square_sum == 13
    assert stats.vision_patches is None


def test_varlen_batch_stats_measure_logical_segment_lengths() -> None:
    stats = ShaftCollatedBatchStats.from_training_inputs(
        sequence_inputs={
            "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7]]),
            "labels": torch.tensor([[-100, 2, 3, 4, -100, 6, 7]]),
        },
        varlen_plan=ShaftVarlenLayoutPlan(
            global_microstep=0,
            plan_fingerprint="stats-varlen-v1",
            local_batch_id=0,
            pack_lengths=(7,),
            segments=(
                ShaftVarlenSegmentLayout(0, 0, 0, 0, 4),
                ShaftVarlenSegmentLayout(1, 0, 1, 4, 7),
            ),
        ),
    )

    assert stats.logical_segments == 2
    assert stats.physical_packs == 1
    assert stats.sequence_length_sum == 7
    assert stats.sequence_length_square_sum == 25


def test_weighted_supervision_partial_coverage_is_not_reported_as_full_mass(
    tmp_path: Path,
) -> None:
    monitor = ShaftTrainingEfficiencyMonitor(output_dir=tmp_path)
    monitor.stage(
        [
            {"_shaft_batch_stats": _stats()},
            {
                "_shaft_batch_stats": replace(
                    _stats(),
                    weighted_supervision_mass=None,
                )
            },
        ],
        host_batch_acquire_seconds=0.0,
    )
    monitor.record_training_step(0.01)
    monitor.commit(global_step=1)
    summary, _ = monitor.finalize(final_global_step=1, device=torch.device("cpu"))

    assert summary.aggregate is not None
    assert summary.aggregate.weighted_supervision_mass == 0.0
    assert summary.aggregate.weighted_supervision_coverage_microbatches == 1
    assert summary.aggregate.ratios()["weighted_supervision_coverage_fraction"] == 0.5


def test_compare_efficiency_script_has_help_and_json_contract(
    tmp_path: Path,
    repo_root: Path,
) -> None:
    monitor = ShaftTrainingEfficiencyMonitor(
        output_dir=tmp_path,
        contract=_contract(),
    )
    monitor.stage(
        [{"_shaft_batch_stats": _stats()}],
        host_batch_acquire_seconds=0.01,
    )
    monitor.record_training_step(0.02)
    monitor.commit(global_step=1)
    monitor.finalize(final_global_step=1, device=torch.device("cpu"))

    help_result = subprocess.run(
        [sys.executable, "scripts/compare_efficiency.py", "--help"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert help_result.returncode == 0
    assert "Compare committed Shaft" in help_result.stdout

    json_result = subprocess.run(
        [
            sys.executable,
            "scripts/compare_efficiency.py",
            str(tmp_path),
            "--json",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert json_result.returncode == 0
    rows = json.loads(json_result.stdout)
    assert rows[0]["mean_sequence_length"] == 3.0
    assert rows[0]["useful_tokens"] == 6

    incompatible_dir = tmp_path / "incompatible"
    incompatible_dir.mkdir()
    payload = json.loads(
        (tmp_path / TRAINING_EFFICIENCY_FILENAME).read_text(encoding="utf-8")
    )
    payload["contract"]["model_type"] = "qwen35vl"
    (incompatible_dir / TRAINING_EFFICIENCY_FILENAME).write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    rejected = subprocess.run(
        [
            sys.executable,
            "scripts/compare_efficiency.py",
            str(tmp_path),
            str(incompatible_dir),
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert rejected.returncode != 0
    assert "not a fair A/B comparison" in rejected.stderr
    allowed = subprocess.run(
        [
            sys.executable,
            "scripts/compare_efficiency.py",
            str(tmp_path),
            str(incompatible_dir),
            "--allow-incompatible",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert allowed.returncode == 0
