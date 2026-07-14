from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch
import torch.distributed as dist

from shaft.data import ShaftCollatedBatchStats
from shaft.observability import ShaftTrainingEfficiencyContract
from shaft.training.efficiency import (
    ShaftTrainingEfficiencyMonitor,
    prepare_training_efficiency_checkpoint,
)


def _stats(rank: int) -> ShaftCollatedBatchStats:
    useful = 4 + int(rank)
    return ShaftCollatedBatchStats(
        logical_segments=1,
        physical_packs=1,
        useful_tokens=useful,
        materialized_tokens=useful,
        supervised_tokens=2,
        weighted_supervision_mass=2.0,
        sequence_length_sum=useful,
        sequence_length_square_sum=useful * useful,
        vision_patches=8,
    )


def _contract(rank: int) -> ShaftTrainingEfficiencyContract:
    return ShaftTrainingEfficiencyContract(
        algorithm="sft",
        model_type="qwen3vl" if rank == 0 else "qwen35vl",
        model_name_or_path="models/test",
        model_plan_fingerprint="model-plan-v1",
        finetune_mode="full",
        torch_dtype="float32",
        attention_implementation="eager",
        seed=42,
        max_steps=1,
        num_train_epochs=1.0,
        data_world_size=2,
        gradient_accumulation_steps=1,
        max_length=32,
        min_pixels=None,
        max_pixels=None,
        optimizer_name="adamw_torch",
        scheduler_name="linear",
        learning_rate=1e-4,
        source_fingerprint="source-v1",
        source_contract_complete=True,
        sample_execution_fingerprint="execution-v1",
        software_fingerprint="software-v1",
        hardware_fingerprint="hardware-v1",
        measurement_protocol="shaft-efficiency-optimizer-frame-v2",
        timing_mode="host_optimizer_frame",
        batch_contract_fingerprint="batch-v1",
        sequence_contract_fingerprint="sequence-v1",
    )


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: distributed_efficiency_snapshot_fault.py OUTPUT MODE")
    output_dir = Path(sys.argv[1])
    mode = str(sys.argv[2]).strip().lower()
    if mode not in {
        "missing",
        "corrupt",
        "timing_mismatch",
        "update_mismatch",
        "contract_mismatch",
        "revoke_fail",
        "snapshot_write_fail",
        "manifest_write_fail",
        "transaction_commit_fail",
        "summary_write_fail",
    }:
        raise ValueError(f"Unsupported fault mode: {mode!r}.")

    dist.init_process_group("gloo")
    try:
        rank = dist.get_rank()
        if mode == "contract_mismatch":
            try:
                ShaftTrainingEfficiencyMonitor(
                    output_dir=output_dir,
                    contract=_contract(rank),
                )
            except RuntimeError as exc:
                if "contract differs across distributed ranks" not in str(exc):
                    raise
            else:
                raise AssertionError("Rank-divergent efficiency contracts were accepted.")
            dist.barrier()
            if rank == 0:
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "contract_mismatch_rejected.txt").write_text(
                    "ok\n",
                    encoding="utf-8",
                )
            return
        checkpoint_dir = output_dir / "checkpoint-1"
        monitor = ShaftTrainingEfficiencyMonitor(output_dir=output_dir)
        monitor.stage(
            [{"_shaft_batch_stats": _stats(rank)}],
            host_batch_acquire_seconds=0.01,
            batch_prepare_seconds=0.005,
        )
        monitor.record_training_step(0.02)
        monitor.commit(global_step=1)
        if mode == "summary_write_fail":
            with patch(
                "shaft.training.efficiency.write_training_efficiency_summary",
                side_effect=OSError("injected root summary write failure"),
            ):
                try:
                    monitor.finalize(
                        final_global_step=1,
                        device=torch.device("cpu"),
                    )
                except RuntimeError as exc:
                    if "root summary write" not in str(exc):
                        raise
                else:
                    raise AssertionError(
                        "Rank-zero summary write failure did not converge."
                    )
            dist.barrier()
            if rank == 0:
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "summary_write_fail_rejected.txt").write_text(
                    "ok\n",
                    encoding="utf-8",
                )
            return
        if mode == "revoke_fail":
            prepare_training_efficiency_checkpoint(
                checkpoint_dir,
                global_step=1,
                generation=monitor.snapshot_generation,
            )
            monitor.write_checkpoint_snapshot(checkpoint_dir, global_step=1)
            monitor = ShaftTrainingEfficiencyMonitor(output_dir=output_dir)
            monitor.stage(
                [{"_shaft_batch_stats": _stats(rank)}],
                host_batch_acquire_seconds=0.01,
                batch_prepare_seconds=0.005,
            )
            monitor.record_training_step(0.02)
            monitor.commit(global_step=1)
        prepare_training_efficiency_checkpoint(
            checkpoint_dir,
            global_step=1,
            generation=monitor.snapshot_generation,
        )
        original_unlink = Path.unlink
        original_write_text = Path.write_text

        def _faulting_unlink(path: Path, *args, **kwargs):
            if mode == "revoke_fail" and rank == 0 and path.name.endswith("snapshot_set.json"):
                raise OSError("injected revoke failure")
            return original_unlink(path, *args, **kwargs)

        def _faulting_write_text(path: Path, *args, **kwargs):
            if (
                mode == "snapshot_write_fail"
                and rank == 1
                and path.name == "shaft_training_efficiency_rank1.json.tmp"
            ):
                raise OSError("injected rank snapshot write failure")
            if (
                mode == "manifest_write_fail"
                and rank == 0
                and path.name.endswith("snapshot_set.json.tmp")
            ):
                raise OSError("injected manifest write failure")
            if (
                mode == "transaction_commit_fail"
                and rank == 0
                and path.name
                == "shaft_training_efficiency_checkpoint_transaction.json.tmp"
            ):
                raise OSError("injected checkpoint transaction commit failure")
            return original_write_text(path, *args, **kwargs)

        if mode in {
            "revoke_fail",
            "snapshot_write_fail",
            "manifest_write_fail",
            "transaction_commit_fail",
        }:
            try:
                with (
                    patch.object(Path, "unlink", _faulting_unlink),
                    patch.object(Path, "write_text", _faulting_write_text),
                ):
                    monitor.write_checkpoint_snapshot(checkpoint_dir, global_step=1)
            except RuntimeError as exc:
                expected_phase = {
                    "revoke_fail": "snapshot set revoke",
                    "snapshot_write_fail": "rank snapshot write",
                    "manifest_write_fail": "snapshot set manifest commit",
                    "transaction_commit_fail": (
                        "checkpoint telemetry transaction commit"
                    ),
                }[mode]
                if expected_phase not in str(exc):
                    raise
            else:
                raise AssertionError(f"Injected {mode} did not fail on every rank.")
            if mode in {"revoke_fail", "transaction_commit_fail"}:
                rejected = ShaftTrainingEfficiencyMonitor.from_checkpoint(
                    output_dir=output_dir,
                    checkpoint_dir=checkpoint_dir,
                    checkpoint_global_step=1,
                )
                if rejected.complete_history or rejected.committed_frames:
                    raise AssertionError(
                        "A pending checkpoint generation accepted stale telemetry."
                    )
            dist.barrier()
            if rank == 0:
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / f"{mode}_rejected.txt").write_text(
                    "ok\n",
                    encoding="utf-8",
                )
            return

        snapshot_path = monitor.write_checkpoint_snapshot(checkpoint_dir, global_step=1)
        if snapshot_path is None:
            raise AssertionError("Expected a persisted rank-local efficiency snapshot.")
        dist.barrier()

        if rank == 1 and mode in {"missing", "corrupt"}:
            if mode == "missing":
                snapshot_path.unlink()
            else:
                snapshot_path.write_text("{not-json", encoding="utf-8")
        dist.barrier()

        resumed = ShaftTrainingEfficiencyMonitor.from_checkpoint(
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
            checkpoint_global_step=1,
        )
        if mode in {"timing_mismatch", "update_mismatch"}:
            if not resumed.complete_history or len(resumed.committed_frames) != 1:
                raise AssertionError("A valid snapshot set must restore on every rank.")
        else:
            if resumed.complete_history or resumed.committed_frames:
                raise AssertionError("All ranks must discard asymmetric snapshot history.")
            if resumed.initial_global_step != 1:
                raise AssertionError("Partial coverage must restart at the checkpoint step.")

        resumed.stage(
            [{"_shaft_batch_stats": _stats(rank)}],
            host_batch_acquire_seconds=0.01,
            batch_prepare_seconds=0.005,
        )
        resumed.record_training_step(0.02)
        resumed.commit(
            global_step=2,
            update_applied=not (mode == "update_mismatch" and rank == 1),
        )
        if mode == "timing_mismatch":
            if rank == 0:
                resumed._committed[-1] = replace(
                    resumed._committed[-1],
                    device_training_seconds=0.01,
                )
            try:
                resumed.finalize(
                    final_global_step=2,
                    device=torch.device("cuda"),
                )
            except RuntimeError as exc:
                if "CUDA event coverage" not in str(exc):
                    raise
            else:
                raise AssertionError("Rank-divergent CUDA timing coverage was not rejected.")
            dist.barrier()
            if rank == 0:
                (output_dir / "timing_mismatch_rejected.txt").write_text(
                    "ok\n",
                    encoding="utf-8",
                )
            return
        if mode == "update_mismatch":
            try:
                resumed.finalize(
                    final_global_step=2,
                    device=torch.device("cpu"),
                )
            except RuntimeError as exc:
                if "step/update states differ" not in str(exc):
                    raise
            else:
                raise AssertionError("Rank-divergent optimizer state was not rejected.")
            dist.barrier()
            if rank == 0:
                (output_dir / "mismatch_rejected.txt").write_text(
                    "ok\n",
                    encoding="utf-8",
                )
            return
        summary, _ = resumed.finalize(
            final_global_step=2,
            device=torch.device("cpu"),
        )
        if summary.complete_history:
            raise AssertionError("Fault-recovered telemetry must remain explicitly partial.")
        if summary.initial_global_step != 1 or summary.final_global_step != 2:
            raise AssertionError("Fault-recovered telemetry has the wrong committed span.")
        if summary.aggregate is None or summary.aggregate.optimizer_steps != 1:
            raise AssertionError("Fault-recovered telemetry did not aggregate the new step.")
        if summary.aggregate.useful_tokens != 9:
            raise AssertionError("Rank-local useful token counts were not summed.")
        if summary.aggregate.logical_segments != 2:
            raise AssertionError("Rank-local logical segment counts were not summed.")

        dist.barrier()
        if rank == 0:
            (output_dir / "fault_result.json").write_text(
                json.dumps(summary.to_dict(), sort_keys=True) + "\n",
                encoding="utf-8",
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
