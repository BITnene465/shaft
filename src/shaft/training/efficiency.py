from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import logging
import os
from pathlib import Path
import secrets
import time
from typing import Any, Callable, Sequence

import torch
from transformers import TrainerCallback

from shaft.data import ShaftCollatedBatchStats
from shaft.observability import (
    ShaftEfficiencyAggregate,
    ShaftEfficiencyFrame,
    ShaftTrainingEfficiencyContract,
    ShaftTrainingEfficiencySummary,
    write_training_efficiency_summary,
)


logger = logging.getLogger(__name__)
_EFFICIENCY_SNAPSHOT_VERSION = 2
_EFFICIENCY_SNAPSHOT_SET_FILENAME = "shaft_training_efficiency_snapshot_set.json"
_EFFICIENCY_CHECKPOINT_TRANSACTION_FILENAME = (
    "shaft_training_efficiency_checkpoint_transaction.json"
)
_EFFICIENCY_CHECKPOINT_TRANSACTION_VERSION = 1
_CHECKPOINT_TRANSACTION_PENDING = "pending"
_CHECKPOINT_TRANSACTION_COMMITTED = "committed"
_CHECKPOINT_TRANSACTION_REVOKED = "revoked"


@dataclass(slots=True)
class _StagedEfficiencyFrame:
    stats: tuple[ShaftCollatedBatchStats, ...]
    host_batch_acquire_seconds: float
    batch_prepare_seconds: float
    training_step_seconds: float = 0.0
    optimizer_step_seconds: float = 0.0
    device_event_start: torch.cuda.Event | None = None
    device_event_end: torch.cuda.Event | None = None


class ShaftTrainingEfficiencyMonitor:
    """Stage actual microbatches and commit only successful optimizer frames."""

    def __init__(
        self,
        *,
        output_dir: str | Path,
        initial_global_step: int = 0,
        complete_history: bool | None = None,
        enabled: bool = True,
        device_timing: bool = True,
        persist: bool = True,
        contract: ShaftTrainingEfficiencyContract | None = None,
        snapshot_generation: str | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.initial_global_step = max(int(initial_global_step), 0)
        self.complete_history = (
            self.initial_global_step == 0
            if complete_history is None
            else bool(complete_history)
        )
        self.enabled = bool(enabled)
        self.device_timing = bool(device_timing)
        self.persist = bool(persist)
        self.contract = contract
        self.snapshot_generation = (
            _new_snapshot_generation()
            if snapshot_generation is None and self.enabled
            else snapshot_generation
        )
        self._staged: _StagedEfficiencyFrame | None = None
        self._committed: list[ShaftEfficiencyFrame] = []
        self._reported_count = 0
        self._optimizer_started_at: float | None = None
        self._device_events: dict[int, tuple[torch.cuda.Event, torch.cuda.Event]] = {}
        self._update_applied_provider: Callable[[], bool] | None = None
        if self.enabled and self.contract is not None:
            _validate_distributed_contract_consensus(self.contract)

    def bind_update_applied_provider(self, provider: Callable[[], bool]) -> None:
        self._update_applied_provider = provider

    def update_applied(self) -> bool:
        if self._update_applied_provider is None:
            return True
        return bool(self._update_applied_provider())

    @classmethod
    def from_checkpoint(
        cls,
        *,
        output_dir: str | Path,
        checkpoint_dir: str | Path | None,
        checkpoint_global_step: int,
        enabled: bool = True,
        device_timing: bool = True,
        persist: bool = True,
        contract: ShaftTrainingEfficiencyContract | None = None,
    ) -> "ShaftTrainingEfficiencyMonitor":
        step = max(int(checkpoint_global_step), 0)
        if not enabled:
            return cls(
                output_dir=output_dir,
                initial_global_step=step,
                complete_history=step == 0,
                enabled=False,
                device_timing=device_timing,
                persist=persist,
                contract=contract,
            )
        if checkpoint_dir is None or step == 0:
            return cls(
                output_dir=output_dir,
                initial_global_step=step,
                enabled=enabled,
                device_timing=device_timing,
                persist=persist,
                contract=contract,
            )
        path = cls._snapshot_path(checkpoint_dir)
        set_manifest_path = Path(checkpoint_dir) / _EFFICIENCY_SNAPSHOT_SET_FILENAME
        payload: dict[str, Any] | None = None
        frames: list[ShaftEfficiencyFrame] | None = None
        local_error: Exception | None = None
        initial_global_step = step
        complete_history = False
        snapshot_generation: str | None = None
        try:
            transaction = _read_training_efficiency_checkpoint_transaction(
                checkpoint_dir
            )
            if transaction.get("state") != _CHECKPOINT_TRANSACTION_COMMITTED:
                raise ValueError("checkpoint telemetry transaction is not committed")
            if int(transaction.get("global_step", -1)) != step:
                raise ValueError(
                    "checkpoint telemetry transaction global_step differs from checkpoint"
                )
            if int(transaction.get("world_size", -1)) != _world_size():
                raise ValueError(
                    "checkpoint telemetry transaction world_size differs from runtime"
                )
            snapshot_generation = str(transaction.get("generation") or "").strip()
            if not snapshot_generation:
                raise ValueError("committed checkpoint telemetry has no generation")
            set_manifest = json.loads(set_manifest_path.read_text(encoding="utf-8"))
            if not isinstance(set_manifest, dict):
                raise TypeError("snapshot set manifest must be a JSON object")
            if int(set_manifest.get("schema_version", -1)) != _EFFICIENCY_SNAPSHOT_VERSION:
                raise ValueError("unsupported snapshot set schema version")
            if int(set_manifest.get("global_step", -1)) != step:
                raise ValueError("snapshot set global_step differs from checkpoint")
            if int(set_manifest.get("world_size", -1)) != _world_size():
                raise ValueError("snapshot set world_size differs from current runtime")
            manifest_generation = str(set_manifest.get("generation") or "").strip()
            if not manifest_generation:
                raise ValueError("snapshot set has no generation")
            if manifest_generation != snapshot_generation:
                raise ValueError(
                    "snapshot set generation differs from checkpoint telemetry transaction"
                )
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise TypeError("snapshot root must be a JSON object")
            if int(payload.get("schema_version", -1)) != _EFFICIENCY_SNAPSHOT_VERSION:
                raise ValueError("unsupported schema version")
            if int(payload.get("global_step", -1)) != step:
                raise ValueError("snapshot global_step differs from checkpoint")
            if int(payload.get("world_size", -1)) != _world_size():
                raise ValueError("snapshot world_size differs from current runtime")
            if int(payload.get("rank", -1)) != _rank():
                raise ValueError("snapshot rank differs from current runtime")
            if str(payload.get("generation") or "").strip() != snapshot_generation:
                raise ValueError("snapshot generation differs from its set manifest")
            raw_contract = payload.get("contract")
            snapshot_contract = (
                None
                if raw_contract is None
                else ShaftTrainingEfficiencyContract.from_dict(dict(raw_contract))
            )
            if snapshot_contract != contract:
                raise ValueError("snapshot training contract differs from current runtime")
            raw_frames = payload.get("frames")
            if not isinstance(raw_frames, list):
                raise TypeError("snapshot frames must be a list")
            frames = [ShaftEfficiencyFrame(**dict(value)) for value in raw_frames]
            initial_global_step = int(payload.get("initial_global_step", -1))
            complete_history = bool(payload.get("complete_history", False))
            cls._validate_snapshot_span(
                frames=frames,
                checkpoint_global_step=step,
                initial_global_step=initial_global_step,
                complete_history=complete_history,
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            local_error = exc
            payload = None
            frames = None

        local_valid = payload is not None and frames is not None
        all_valid = local_valid
        world_size = _world_size()
        if world_size > 1:
            collective_device = _collective_device_for_runtime()
            local_state = torch.tensor(
                [
                    int(local_valid),
                    initial_global_step if local_valid else step,
                    int(complete_history) if local_valid else 0,
                    len(frames) if frames is not None else 0,
                    (
                        _generation_numeric(snapshot_generation)
                        if snapshot_generation is not None
                        else 0
                    ),
                ],
                dtype=torch.int64,
                device=collective_device,
            )
            state_min = local_state.clone()
            state_max = local_state.clone()
            torch.distributed.all_reduce(state_min, op=torch.distributed.ReduceOp.MIN)
            torch.distributed.all_reduce(state_max, op=torch.distributed.ReduceOp.MAX)
            all_valid = bool(state_min[0].item()) and torch.equal(
                state_min[1:], state_max[1:]
            )

        if not all_valid:
            detail = (
                str(local_error)
                if local_error is not None
                else "another rank has no compatible snapshot"
            )
            logger.warning(
                "[training-efficiency] checkpoint snapshot set is incomplete or invalid; "
                "all ranks restart coverage at step %s: %s",
                step,
                detail,
            )
            return cls(
                output_dir=output_dir,
                initial_global_step=step,
                complete_history=False,
                enabled=enabled,
                device_timing=device_timing,
                persist=persist,
                contract=contract,
            )
        assert payload is not None
        assert frames is not None
        monitor = cls(
            output_dir=output_dir,
            initial_global_step=initial_global_step,
            complete_history=complete_history,
            enabled=enabled,
            device_timing=device_timing,
            persist=persist,
            contract=contract,
            snapshot_generation=snapshot_generation,
        )
        monitor._committed = frames
        monitor._reported_count = len(frames)
        logger.info(
            "[training-efficiency] restored rank=%s frames=%s through_step=%s",
            _rank(),
            len(frames),
            step,
        )
        return monitor

    @staticmethod
    def _validate_snapshot_span(
        *,
        frames: Sequence[ShaftEfficiencyFrame],
        checkpoint_global_step: int,
        initial_global_step: int,
        complete_history: bool,
    ) -> None:
        step = int(checkpoint_global_step)
        initial = int(initial_global_step)
        if initial < 0 or initial >= step:
            raise ValueError("snapshot initial_global_step is outside the covered span")
        if complete_history and initial != 0:
            raise ValueError("a complete snapshot history must begin at global step zero")
        expected_steps = tuple(range(initial + 1, step + 1))
        actual_steps = tuple(int(frame.global_step) for frame in frames)
        if actual_steps != expected_steps:
            raise ValueError("snapshot frames do not form the declared contiguous step span")

    @property
    def committed_frames(self) -> tuple[ShaftEfficiencyFrame, ...]:
        return tuple(self._committed)

    def stage(
        self,
        batches: Sequence[dict[str, Any]],
        *,
        host_batch_acquire_seconds: float,
        batch_prepare_seconds: float = 0.0,
    ) -> None:
        if not self.enabled or not batches:
            return
        if self._staged is not None:
            raise RuntimeError("An efficiency optimizer frame is already staged.")
        stats: list[ShaftCollatedBatchStats] = []
        for batch in batches:
            value = batch.get("_shaft_batch_stats")
            if not isinstance(value, ShaftCollatedBatchStats):
                raise ValueError(
                    "Training efficiency requires collator-owned _shaft_batch_stats."
                )
            stats.append(value)
        self._staged = _StagedEfficiencyFrame(
            stats=tuple(stats),
            host_batch_acquire_seconds=max(float(host_batch_acquire_seconds), 0.0),
            batch_prepare_seconds=max(float(batch_prepare_seconds), 0.0),
        )

    def start_device_frame(self) -> None:
        if (
            not self.enabled
            or not self.device_timing
            or self._staged is None
            or self._staged.device_event_start is not None
        ):
            return
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self._staged.device_event_start = event

    def record_training_step(
        self,
        seconds: float,
    ) -> None:
        if not self.enabled:
            return
        if self._staged is None:
            raise RuntimeError("Cannot record training time without a staged frame.")
        self._staged.training_step_seconds += max(float(seconds), 0.0)

    def start_optimizer_step(self) -> None:
        if self.enabled and self._staged is not None:
            self._optimizer_started_at = time.perf_counter()

    def finish_optimizer_step(self) -> None:
        if not self.enabled or self._optimizer_started_at is None:
            return
        if self._staged is None:
            raise RuntimeError("Cannot record optimizer time without a staged frame.")
        self._staged.optimizer_step_seconds += max(
            time.perf_counter() - self._optimizer_started_at,
            0.0,
        )
        if self._staged.device_event_start is not None:
            event_end = torch.cuda.Event(enable_timing=True)
            event_end.record()
            self._staged.device_event_end = event_end
        self._optimizer_started_at = None

    def commit(self, *, global_step: int, update_applied: bool = True) -> None:
        if not self.enabled:
            return
        if self._staged is None:
            raise RuntimeError("Cannot commit efficiency telemetry without a staged frame.")
        step = int(global_step)
        expected = (
            self._committed[-1].global_step + 1
            if self._committed
            else self.initial_global_step + 1
        )
        if step != expected:
            raise ValueError(
                f"Efficiency global step is not contiguous: actual={step}, expected={expected}."
            )
        stats = self._staged.stats
        weighted_values = [
            value.weighted_supervision_mass
            for value in stats
            if value.weighted_supervision_mass is not None
        ]
        frame = ShaftEfficiencyFrame(
            global_step=step,
            logical_segments=sum(value.logical_segments for value in stats),
            physical_packs=sum(value.physical_packs for value in stats),
            useful_tokens=sum(value.useful_tokens for value in stats),
            materialized_tokens=sum(value.materialized_tokens for value in stats),
            supervised_tokens=sum(value.supervised_tokens for value in stats),
            weighted_supervision_mass=(
                None
                if len(weighted_values) != len(stats)
                else float(sum(weighted_values))
            ),
            weighted_supervision_coverage_microbatches=len(weighted_values),
            sequence_length_sum=sum(value.sequence_length_sum for value in stats),
            sequence_length_square_sum=sum(
                value.sequence_length_square_sum for value in stats
            ),
            vision_patches=sum(
                int(value.vision_patches or 0) for value in stats
            ),
            vision_coverage_batches=sum(
                value.vision_patches is not None for value in stats
            ),
            microbatches=len(stats),
            host_batch_acquire_seconds=self._staged.host_batch_acquire_seconds,
            batch_prepare_seconds=self._staged.batch_prepare_seconds,
            training_step_seconds=self._staged.training_step_seconds,
            optimizer_step_seconds=self._staged.optimizer_step_seconds,
            device_training_seconds=None,
            update_applied=bool(update_applied),
        )
        if (
            self._staged.device_event_start is not None
            and self._staged.device_event_end is not None
        ):
            self._device_events[step] = (
                self._staged.device_event_start,
                self._staged.device_event_end,
            )
        self._committed.append(frame)
        self._staged = None
        self._optimizer_started_at = None

    def discard(self) -> None:
        self._staged = None
        self._optimizer_started_at = None

    def write_checkpoint_snapshot(
        self,
        checkpoint_dir: str | Path,
        *,
        global_step: int,
    ) -> Path | None:
        if not self.enabled:
            return None
        if not self.persist:
            invalidate_training_efficiency_snapshot_set(
                checkpoint_dir,
                synchronize=True,
            )
            return None
        _validate_training_efficiency_checkpoint_transaction(
            checkpoint_dir,
            global_step=int(global_step),
            generation=self.snapshot_generation,
            expected_state=_CHECKPOINT_TRANSACTION_PENDING,
        )
        self._resolve_device_events()
        step = int(global_step)
        if not self._committed or self._committed[-1].global_step != step:
            raise ValueError(
                "Efficiency snapshot can only be written for the latest committed step."
            )
        path = self._snapshot_path(checkpoint_dir)
        payload = {
            "schema_version": _EFFICIENCY_SNAPSHOT_VERSION,
            "global_step": step,
            "initial_global_step": self.initial_global_step,
            "complete_history": self.complete_history,
            "contract": None if self.contract is None else self.contract.to_dict(),
            "world_size": _world_size(),
            "rank": _rank(),
            "generation": self.snapshot_generation,
            "frames": [asdict(frame) for frame in self._committed],
        }
        invalidate_training_efficiency_snapshot_set(
            checkpoint_dir,
            synchronize=True,
        )
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        snapshot_error: Exception | None = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary.replace(path)
        except Exception as exc:  # noqa: BLE001 - converge rank-local I/O failures
            snapshot_error = exc
        snapshot_failure = _converge_distributed_io_failure(
            phase="rank snapshot write",
            local_error=snapshot_error,
        )
        if snapshot_failure is not None:
            _revoke_after_failed_snapshot_commit(checkpoint_dir)
            raise snapshot_failure from snapshot_error

        manifest_error: Exception | None = None
        if _is_rank_zero():
            try:
                manifest_path = Path(checkpoint_dir) / _EFFICIENCY_SNAPSHOT_SET_FILENAME
                manifest_payload = {
                    "schema_version": _EFFICIENCY_SNAPSHOT_VERSION,
                    "global_step": step,
                    "world_size": _world_size(),
                    "generation": self.snapshot_generation,
                }
                manifest_temporary = manifest_path.with_suffix(
                    f"{manifest_path.suffix}.tmp"
                )
                manifest_temporary.write_text(
                    json.dumps(
                        manifest_payload,
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                manifest_temporary.replace(manifest_path)
            except Exception as exc:  # noqa: BLE001 - converge rank-local I/O failures
                manifest_error = exc
        manifest_failure = _converge_distributed_io_failure(
            phase="snapshot set manifest commit",
            local_error=manifest_error,
        )
        if manifest_failure is not None:
            _revoke_after_failed_snapshot_commit(checkpoint_dir)
            raise manifest_failure from manifest_error
        try:
            commit_training_efficiency_checkpoint(
                checkpoint_dir,
                global_step=step,
                generation=self.snapshot_generation,
            )
        except RuntimeError:
            _revoke_after_failed_snapshot_commit(checkpoint_dir)
            raise
        return path

    @staticmethod
    def _snapshot_path(checkpoint_dir: str | Path) -> Path:
        return Path(checkpoint_dir) / f"shaft_training_efficiency_rank{_rank()}.json"

    def report_pending(self, *, device: torch.device) -> dict[str, float]:
        if not self.enabled or self._reported_count >= len(self._committed):
            return {}
        self._resolve_device_events()
        frames = self._committed[self._reported_count :]
        summary = self._distributed_summary(frames, device=device)
        self._reported_count = len(self._committed)
        return summary.log_metrics()

    def finalize(
        self,
        *,
        final_global_step: int,
        device: torch.device,
    ) -> tuple[ShaftTrainingEfficiencySummary, dict[str, float]]:
        if self._staged is not None:
            self.discard()
        self._resolve_device_events()
        summary = self._distributed_summary(self._committed, device=device)
        if int(final_global_step) != (
            self._committed[-1].global_step
            if self._committed
            else self.initial_global_step
        ):
            raise ValueError(
                "Efficiency committed steps do not match Trainer global_step."
            )
        summary_error: Exception | None = None
        summary_path: Path | None = None
        if self.persist and _is_rank_zero():
            try:
                summary_path = write_training_efficiency_summary(self.output_dir, summary)
            except Exception as exc:  # noqa: BLE001 - converge rank-zero I/O failures
                summary_error = exc
        if self.persist:
            summary_failure = _converge_distributed_io_failure(
                phase="root summary write",
                local_error=summary_error,
            )
            if summary_failure is not None:
                raise summary_failure from summary_error
        if summary_path is not None:
            logger.info("[training-efficiency] summary=%s", summary_path)
        return summary, summary.log_metrics(prefix="train_efficiency")

    def _distributed_summary(
        self,
        frames: Sequence[ShaftEfficiencyFrame],
        *,
        device: torch.device,
    ) -> ShaftTrainingEfficiencySummary:
        world_size = _world_size()
        complete_history = self.complete_history
        if world_size > 1:
            collective_device = _collective_device(device)
            local_state = torch.tensor(
                [
                    len(frames),
                    self.initial_global_step,
                    int(self.complete_history),
                ],
                dtype=torch.int64,
                device=collective_device,
            )
            state_min = local_state.clone()
            state_max = local_state.clone()
            torch.distributed.all_reduce(state_min, op=torch.distributed.ReduceOp.MIN)
            torch.distributed.all_reduce(state_max, op=torch.distributed.ReduceOp.MAX)
            if int(state_min[0].item()) != int(state_max[0].item()):
                raise RuntimeError(
                    "Efficiency ranks have different committed frame counts; "
                    "refusing a collective that could deadlock."
                )
            if int(state_min[1].item()) != int(state_max[1].item()):
                raise RuntimeError(
                    "Efficiency ranks have different initial global steps."
                )
            complete_history = bool(state_min[2].item())
        if not frames:
            return ShaftTrainingEfficiencySummary(
                initial_global_step=self.initial_global_step,
                final_global_step=self.initial_global_step,
                complete_history=complete_history,
                world_size=world_size,
                aggregate=None,
                rank_time_min_seconds=0.0,
                rank_time_mean_seconds=0.0,
                rank_time_max_seconds=0.0,
                contract=self.contract,
            )
        local = ShaftEfficiencyAggregate.from_frames(frames)
        expected_device_timing_steps = (
            len(frames)
            if self.device_timing and torch.device(device).type == "cuda"
            else 0
        )
        if world_size == 1 and local.device_timing_steps != expected_device_timing_steps:
            raise RuntimeError(
                "Efficiency CUDA event coverage differs from the committed frame span: "
                f"actual={local.device_timing_steps}, "
                f"expected={expected_device_timing_steps}."
            )
        if world_size == 1:
            rank_time = local.critical_path_seconds
            return ShaftTrainingEfficiencySummary(
                initial_global_step=self.initial_global_step,
                final_global_step=local.last_step,
                complete_history=complete_history,
                world_size=1,
                aggregate=local,
                rank_time_min_seconds=rank_time,
                rank_time_mean_seconds=rank_time,
                rank_time_max_seconds=rank_time,
                contract=self.contract,
            )

        timing_coverage_state = torch.tensor(
            [local.device_timing_steps, expected_device_timing_steps],
            dtype=torch.int64,
            device=collective_device,
        )
        timing_coverage_min = timing_coverage_state.clone()
        timing_coverage_max = timing_coverage_state.clone()
        torch.distributed.all_reduce(
            timing_coverage_min,
            op=torch.distributed.ReduceOp.MIN,
        )
        torch.distributed.all_reduce(
            timing_coverage_max,
            op=torch.distributed.ReduceOp.MAX,
        )
        if (
            not torch.equal(timing_coverage_min, timing_coverage_max)
            or int(timing_coverage_min[0].item())
            != int(timing_coverage_min[1].item())
        ):
            raise RuntimeError(
                "Efficiency ranks have incomplete or different CUDA event coverage."
            )

        step_state_tensor = torch.tensor(
            [
                [frame.global_step, int(frame.update_applied)]
                for frame in frames
            ],
            dtype=torch.int64,
            device=collective_device,
        )
        gathered_step_states = [
            torch.empty_like(step_state_tensor) for _ in range(world_size)
        ]
        torch.distributed.all_gather(gathered_step_states, step_state_tensor)
        if any(
            not torch.equal(step_state_tensor, other)
            for other in gathered_step_states
        ):
            raise RuntimeError(
                "Efficiency rank step/update states differ; refusing misaligned metrics."
            )

        count_values = torch.tensor(
            [
                local.logical_segments,
                local.physical_packs,
                local.useful_tokens,
                local.materialized_tokens,
                local.supervised_tokens,
                local.weighted_supervision_coverage_microbatches,
                local.sequence_length_sum,
                local.sequence_length_square_sum,
                local.vision_patches,
                local.vision_coverage_batches,
                local.microbatches,
            ],
            dtype=torch.int64,
            device=collective_device,
        )
        torch.distributed.all_reduce(count_values, op=torch.distributed.ReduceOp.SUM)

        weighted_mass = torch.tensor(
            [local.weighted_supervision_mass],
            dtype=torch.float64,
            device=collective_device,
        )
        torch.distributed.all_reduce(weighted_mass, op=torch.distributed.ReduceOp.SUM)

        component_tensor = torch.tensor(
            [
                [
                    frame.host_batch_acquire_seconds,
                    frame.batch_prepare_seconds,
                    frame.training_step_seconds,
                    frame.optimizer_step_seconds,
                    float(frame.device_training_seconds or 0.0),
                    float(frame.device_training_seconds is not None),
                    frame.critical_path_seconds,
                ]
                for frame in frames
            ],
            dtype=torch.float64,
            device=collective_device,
        )
        gathered_components = [
            torch.empty_like(component_tensor) for _ in range(world_size)
        ]
        torch.distributed.all_gather(gathered_components, component_tensor)
        stacked_components = torch.stack(gathered_components, dim=0)
        stacked_critical = stacked_components[:, :, 6]
        critical_rank = stacked_critical.argmax(dim=0)
        step_index = torch.arange(len(frames), device=collective_device)
        critical_components = stacked_components[critical_rank, step_index]
        critical_per_step = critical_components[:, 6].cpu().tolist()
        component_totals = critical_components[:, :6].sum(dim=0)
        rank_totals = stacked_critical.sum(dim=1)

        aggregate = ShaftEfficiencyAggregate(
            first_step=local.first_step,
            last_step=local.last_step,
            optimizer_steps=local.optimizer_steps,
            logical_segments=int(count_values[0].item()),
            physical_packs=int(count_values[1].item()),
            useful_tokens=int(count_values[2].item()),
            materialized_tokens=int(count_values[3].item()),
            supervised_tokens=int(count_values[4].item()),
            weighted_supervision_mass=float(weighted_mass.item()),
            weighted_supervision_coverage_microbatches=int(count_values[5].item()),
            sequence_length_sum=int(count_values[6].item()),
            sequence_length_square_sum=int(count_values[7].item()),
            vision_patches=int(count_values[8].item()),
            vision_coverage_batches=int(count_values[9].item()),
            microbatches=int(count_values[10].item()),
            update_applied_steps=local.update_applied_steps,
            host_batch_acquire_seconds=float(component_totals[0].item()),
            batch_prepare_seconds=float(component_totals[1].item()),
            training_step_seconds=float(component_totals[2].item()),
            optimizer_step_seconds=float(component_totals[3].item()),
            device_training_seconds=float(component_totals[4].item()),
            device_timing_steps=int(component_totals[5].item()),
            critical_path_seconds=float(sum(critical_per_step)),
            critical_path_p50_seconds=float(
                torch.quantile(torch.tensor(critical_per_step), 0.5).item()
            ),
            critical_path_p95_seconds=float(
                torch.quantile(torch.tensor(critical_per_step), 0.95).item()
            ),
        )
        return ShaftTrainingEfficiencySummary(
            initial_global_step=self.initial_global_step,
            final_global_step=aggregate.last_step,
            complete_history=complete_history,
            world_size=world_size,
            aggregate=aggregate,
            rank_time_min_seconds=float(rank_totals.min().item()),
            rank_time_mean_seconds=float(rank_totals.mean().item()),
            rank_time_max_seconds=float(rank_totals.max().item()),
            contract=self.contract,
        )

    def _resolve_device_events(self) -> None:
        if not self._device_events:
            return
        pending_steps = sorted(self._device_events)
        last_events = self._device_events[pending_steps[-1]]
        last_events[1].synchronize()
        by_step = {frame.global_step: index for index, frame in enumerate(self._committed)}
        for step in pending_steps:
            index = by_step.get(step)
            if index is None:
                continue
            start, end = self._device_events[step]
            seconds = float(start.elapsed_time(end)) / 1000.0
            self._committed[index] = replace(
                self._committed[index],
                device_training_seconds=max(seconds, 0.0),
            )
        self._device_events.clear()


class ShaftTrainingEfficiencyCallback(TrainerCallback):
    def __init__(self, monitor: ShaftTrainingEfficiencyMonitor) -> None:
        self.monitor = monitor

    def on_pre_optimizer_step(self, args, state, control, **kwargs):  # noqa: ANN001
        _ = args, state, kwargs
        self.monitor.start_optimizer_step()
        return control

    def on_optimizer_step(self, args, state, control, **kwargs):  # noqa: ANN001
        _ = args, state, kwargs
        self.monitor.finish_optimizer_step()
        return control

    def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001
        _ = args, kwargs
        self.monitor.commit(
            global_step=int(state.global_step),
            update_applied=self.monitor.update_applied(),
        )
        return control

    def on_save(self, args, state, control, **kwargs):  # noqa: ANN001
        _ = kwargs
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{int(state.global_step)}"
        self.monitor.write_checkpoint_snapshot(
            checkpoint_dir,
            global_step=int(state.global_step),
        )
        return control


class ShaftTrainingEfficiencySnapshotInvalidationCallback(TrainerCallback):
    """Revoke stale telemetry when checkpoints are saved with telemetry disabled."""

    def on_save(self, args, state, control, **kwargs):  # noqa: ANN001
        _ = kwargs
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{int(state.global_step)}"
        invalidate_training_efficiency_snapshot_set(
            checkpoint_dir,
            synchronize=True,
        )
        return control


def prepare_training_efficiency_checkpoint(
    checkpoint_dir: str | Path,
    *,
    global_step: int,
    generation: str | None,
) -> None:
    """Publish the telemetry generation before HF mutates a checkpoint directory."""

    resolved_generation = None if generation is None else str(generation).strip()
    if generation is not None and not resolved_generation:
        raise ValueError("Training efficiency checkpoint generation must not be empty.")
    state = (
        _CHECKPOINT_TRANSACTION_PENDING
        if resolved_generation is not None
        else _CHECKPOINT_TRANSACTION_REVOKED
    )
    _write_training_efficiency_checkpoint_transaction(
        checkpoint_dir,
        global_step=int(global_step),
        generation=resolved_generation,
        state=state,
        phase="checkpoint telemetry generation prepare",
    )


def commit_training_efficiency_checkpoint(
    checkpoint_dir: str | Path,
    *,
    global_step: int,
    generation: str | None,
) -> None:
    resolved_generation = str(generation or "").strip()
    if not resolved_generation:
        raise ValueError("Committed training efficiency checkpoint requires a generation.")
    _validate_training_efficiency_checkpoint_transaction(
        checkpoint_dir,
        global_step=int(global_step),
        generation=resolved_generation,
        expected_state=_CHECKPOINT_TRANSACTION_PENDING,
    )
    _write_training_efficiency_checkpoint_transaction(
        checkpoint_dir,
        global_step=int(global_step),
        generation=resolved_generation,
        state=_CHECKPOINT_TRANSACTION_COMMITTED,
        phase="checkpoint telemetry transaction commit",
    )


def _write_training_efficiency_checkpoint_transaction(
    checkpoint_dir: str | Path,
    *,
    global_step: int,
    generation: str | None,
    state: str,
    phase: str,
) -> None:
    path = Path(checkpoint_dir) / _EFFICIENCY_CHECKPOINT_TRANSACTION_FILENAME
    local_error: Exception | None = None
    if _is_rank_zero():
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema_version": _EFFICIENCY_CHECKPOINT_TRANSACTION_VERSION,
                "global_step": int(global_step),
                "world_size": _world_size(),
                "generation": generation,
                "state": state,
            }
            temporary = path.with_suffix(f"{path.suffix}.tmp")
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary.replace(path)
        except Exception as exc:  # noqa: BLE001 - converge rank-zero I/O failures
            local_error = exc
    failure = _converge_distributed_io_failure(
        phase=phase,
        local_error=local_error,
    )
    if failure is not None:
        raise failure from local_error


def _read_training_efficiency_checkpoint_transaction(
    checkpoint_dir: str | Path,
) -> dict[str, Any]:
    path = Path(checkpoint_dir) / _EFFICIENCY_CHECKPOINT_TRANSACTION_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("checkpoint telemetry transaction must be a JSON object")
    if (
        int(payload.get("schema_version", -1))
        != _EFFICIENCY_CHECKPOINT_TRANSACTION_VERSION
    ):
        raise ValueError("unsupported checkpoint telemetry transaction schema")
    state = str(payload.get("state") or "").strip()
    if state not in {
        _CHECKPOINT_TRANSACTION_PENDING,
        _CHECKPOINT_TRANSACTION_COMMITTED,
        _CHECKPOINT_TRANSACTION_REVOKED,
    }:
        raise ValueError("unsupported checkpoint telemetry transaction state")
    return payload


def _validate_training_efficiency_checkpoint_transaction(
    checkpoint_dir: str | Path,
    *,
    global_step: int,
    generation: str | None,
    expected_state: str,
) -> None:
    local_error: Exception | None = None
    try:
        payload = _read_training_efficiency_checkpoint_transaction(checkpoint_dir)
        if str(payload.get("state")) != expected_state:
            raise ValueError(
                "checkpoint telemetry transaction state differs from expected state"
            )
        if int(payload.get("global_step", -1)) != int(global_step):
            raise ValueError(
                "checkpoint telemetry transaction global_step differs from runtime"
            )
        if int(payload.get("world_size", -1)) != _world_size():
            raise ValueError(
                "checkpoint telemetry transaction world_size differs from runtime"
            )
        if payload.get("generation") != generation:
            raise ValueError(
                "checkpoint telemetry transaction generation differs from runtime"
            )
    except Exception as exc:  # noqa: BLE001 - converge rank-local read failures
        local_error = exc
    failure = _converge_distributed_io_failure(
        phase="checkpoint telemetry transaction validation",
        local_error=local_error,
    )
    if failure is not None:
        raise failure from local_error


def invalidate_training_efficiency_summary(output_dir: str | Path) -> None:
    from shaft.observability import TRAINING_EFFICIENCY_FILENAME

    if not (
        torch.distributed.is_available() and torch.distributed.is_initialized()
    ):
        rank = int(os.environ.get("RANK", "0"))
        if rank != 0:
            return
        try:
            (Path(output_dir) / TRAINING_EFFICIENCY_FILENAME).unlink(missing_ok=True)
        except OSError as exc:
            logger.warning(
                "[training-efficiency] could not revoke the stale root summary; "
                "continuing without changing training semantics: %s",
                exc,
            )
        return

    local_error: Exception | None = None
    if _is_rank_zero():
        try:
            (Path(output_dir) / TRAINING_EFFICIENCY_FILENAME).unlink(missing_ok=True)
        except Exception as exc:  # noqa: BLE001 - converge rank-local I/O failures
            local_error = exc
    failure = _converge_distributed_io_failure(
        phase="root summary revoke",
        local_error=local_error,
    )
    if failure is not None:
        raise failure from local_error


def invalidate_training_efficiency_snapshot_set(
    checkpoint_dir: str | Path,
    *,
    synchronize: bool,
) -> None:
    directory = Path(checkpoint_dir)
    local_error: Exception | None = None
    if _is_rank_zero():
        try:
            for pattern in (
                "shaft_training_efficiency_rank*.json",
                "shaft_training_efficiency_rank*.json.tmp",
            ):
                for path in directory.glob(pattern):
                    path.unlink(missing_ok=True)
            manifest_path = directory / _EFFICIENCY_SNAPSHOT_SET_FILENAME
            manifest_path.unlink(missing_ok=True)
            manifest_path.with_suffix(f"{manifest_path.suffix}.tmp").unlink(
                missing_ok=True
            )
        except Exception as exc:  # noqa: BLE001 - converge rank-local I/O failures
            local_error = exc
    if not synchronize:
        if local_error is not None:
            raise local_error
        return
    failure = _converge_distributed_io_failure(
        phase="snapshot set revoke",
        local_error=local_error,
    )
    if failure is not None:
        raise failure from local_error


def _revoke_after_failed_snapshot_commit(checkpoint_dir: str | Path) -> None:
    """Best-effort distributed rollback after every rank has observed a failed phase."""

    try:
        invalidate_training_efficiency_snapshot_set(
            checkpoint_dir,
            synchronize=True,
        )
    except RuntimeError as exc:
        logger.error(
            "[training-efficiency] failed to revoke an incomplete snapshot set: %s",
            exc,
        )


def _converge_distributed_io_failure(
    *,
    phase: str,
    local_error: Exception | None,
) -> RuntimeError | None:
    """Return one shared failure after a fallible rank-local snapshot phase."""

    world_size = _world_size()
    if world_size == 1:
        if local_error is None:
            return None
        return RuntimeError(f"Training efficiency {phase} failed on rank 0: {local_error}")
    device = _collective_device_for_runtime()
    local_state = torch.tensor(
        [
            int(local_error is None),
            world_size if local_error is None else _rank(),
        ],
        dtype=torch.int64,
        device=device,
    )
    global_state = local_state.clone()
    torch.distributed.all_reduce(global_state, op=torch.distributed.ReduceOp.MIN)
    if bool(global_state[0].item()):
        return None
    failed_rank = int(global_state[1].item())
    detail = str(local_error) if _rank() == failed_rank and local_error is not None else ""
    suffix = f": {detail}" if detail else ""
    return RuntimeError(
        f"Training efficiency {phase} failed on rank {failed_rank}{suffix}"
    )


def _world_size() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_world_size())
    return 1


def _is_rank_zero() -> bool:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank()) == 0
    return True


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank())
    return 0


def _collective_device(device: torch.device) -> torch.device:
    backend = str(torch.distributed.get_backend()).lower()
    if "nccl" in backend:
        return torch.device(device)
    return torch.device("cpu")


def _collective_device_for_runtime() -> torch.device:
    backend = str(torch.distributed.get_backend()).lower()
    if "nccl" in backend:
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL telemetry restore requires an available CUDA device.")
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _barrier_if_distributed() -> None:
    if _world_size() > 1:
        torch.distributed.barrier()


def _new_snapshot_generation() -> str:
    values = [secrets.randbits(62), secrets.randbits(62)] if _is_rank_zero() else [0, 0]
    if _world_size() > 1:
        device = _collective_device_for_runtime()
        tensor = torch.tensor(values, dtype=torch.int64, device=device)
        torch.distributed.broadcast(tensor, src=0)
        values = [int(item) for item in tensor.cpu().tolist()]
    return "".join(f"{value:016x}" for value in values)


def _generation_numeric(value: str) -> int:
    digest = hashlib.sha256(str(value).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _validate_distributed_contract_consensus(
    contract: ShaftTrainingEfficiencyContract,
) -> None:
    if _world_size() <= 1:
        return
    canonical = json.dumps(
        contract.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(canonical).digest()
    values = [
        int.from_bytes(digest[offset : offset + 8], "big") & ((1 << 63) - 1)
        for offset in range(0, len(digest), 8)
    ]
    device = _collective_device_for_runtime()
    local = torch.tensor(values, dtype=torch.int64, device=device)
    minimum = local.clone()
    maximum = local.clone()
    torch.distributed.all_reduce(minimum, op=torch.distributed.ReduceOp.MIN)
    torch.distributed.all_reduce(maximum, op=torch.distributed.ReduceOp.MAX)
    if not torch.equal(minimum, maximum):
        raise RuntimeError(
            "Training-efficiency contract differs across distributed ranks."
        )
