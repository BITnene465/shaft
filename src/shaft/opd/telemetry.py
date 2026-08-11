from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Iterator

import torch
from transformers import TrainerCallback


OPD_TELEMETRY_FILENAME = "shaft_opd_telemetry.json"
_SNAPSHOT_VERSION = 2
_PHASES = (
    "rollout_weight_sync",
    "rollout_generate",
    "student_score",
    "teacher_score",
    "objective",
)


@dataclass(frozen=True, slots=True)
class OPDTelemetryContract:
    training_resume_fingerprint: str
    rollout_backend: str
    teacher_provider: str
    teacher_artifact_fingerprint: str
    objective_mode: str
    timing_mode: str = "wall"
    measurement_protocol: str = "shaft-opd-optimizer-frame-v2"

    def __post_init__(self) -> None:
        for name in (
            "training_resume_fingerprint",
            "rollout_backend",
            "teacher_provider",
            "teacher_artifact_fingerprint",
            "objective_mode",
            "timing_mode",
            "measurement_protocol",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"OPD telemetry contract {name} must not be empty.")


@dataclass(frozen=True, slots=True)
class OPDTelemetryFrame:
    global_step: int
    microbatches: int
    prompt_tokens: int
    materialized_prompt_tokens: int
    completion_tokens: int
    vision_patches: int
    dense_teacher_elements: int
    topk_teacher_elements: int
    teacher_request_bytes: int
    teacher_response_bytes: int
    rollout_weight_sync_seconds: float
    rollout_generate_seconds: float
    student_score_seconds: float
    teacher_score_seconds: float
    objective_seconds: float
    backward_and_trainer_overhead_seconds: float
    optimizer_seconds: float
    update_applied: bool
    device_rollout_weight_sync_seconds: float | None = None
    device_rollout_generate_seconds: float | None = None
    device_student_score_seconds: float | None = None
    device_teacher_score_seconds: float | None = None
    device_objective_seconds: float | None = None
    device_optimizer_frame_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.global_step <= 0 or self.microbatches <= 0:
            raise ValueError("A committed OPD telemetry frame must be non-empty.")
        for name in (
            "prompt_tokens",
            "materialized_prompt_tokens",
            "completion_tokens",
            "vision_patches",
            "dense_teacher_elements",
            "topk_teacher_elements",
            "teacher_request_bytes",
            "teacher_response_bytes",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"OPD telemetry {name} must be >= 0.")
        for name in (
            *(f"{phase}_seconds" for phase in _PHASES),
            "backward_and_trainer_overhead_seconds",
            "optimizer_seconds",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"OPD telemetry {name} must be finite and >= 0.")
        for name in (
            *(f"device_{phase}_seconds" for phase in _PHASES),
            "device_optimizer_frame_seconds",
        ):
            value = getattr(self, name)
            if value is not None and (
                not math.isfinite(float(value)) or float(value) < 0
            ):
                raise ValueError(f"OPD telemetry {name} must be finite and >= 0.")

    @property
    def critical_path_seconds(self) -> float:
        return sum(
            float(getattr(self, f"{phase}_seconds")) for phase in _PHASES
        ) + float(self.backward_and_trainer_overhead_seconds) + float(self.optimizer_seconds)


@dataclass(slots=True)
class _OPDWindow:
    microbatches: int = 0
    prompt_tokens: int = 0
    materialized_prompt_tokens: int = 0
    completion_tokens: int = 0
    vision_patches: int = 0
    dense_teacher_elements: int = 0
    topk_teacher_elements: int = 0
    teacher_request_bytes: int = 0
    teacher_response_bytes: int = 0
    phases: dict[str, float] = field(default_factory=lambda: {name: 0.0 for name in _PHASES})
    training_step_seconds: float = 0.0
    optimizer_seconds: float = 0.0
    device_frame_start: torch.cuda.Event | None = None
    device_frame_end: torch.cuda.Event | None = None
    device_phase_events: dict[
        str,
        list[tuple[torch.cuda.Event, torch.cuda.Event]],
    ] = field(default_factory=lambda: {name: [] for name in _PHASES})


@dataclass(slots=True)
class _OPDPendingDeviceFrame:
    frame_start: torch.cuda.Event
    frame_end: torch.cuda.Event
    phase_events: dict[str, tuple[tuple[torch.cuda.Event, torch.cuda.Event], ...]]


class OPDTelemetryMonitor:
    """OPD-native phase telemetry committed only after one optimizer attempt."""

    def __init__(
        self,
        *,
        output_dir: str | Path,
        contract: OPDTelemetryContract,
        initial_global_step: int = 0,
        frames: tuple[OPDTelemetryFrame, ...] = (),
        persist: bool = True,
        device_timing: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.contract = contract
        self.initial_global_step = int(initial_global_step)
        self.frames = list(frames)
        self.persist = bool(persist)
        self.device_timing = bool(device_timing)
        self._window: _OPDWindow | None = None
        self._optimizer_started_at: float | None = None
        self._device_frames: dict[int, _OPDPendingDeviceFrame] = {}
        self._update_applied_provider = lambda: True

    @classmethod
    def from_checkpoint(
        cls,
        *,
        output_dir: str | Path,
        checkpoint_dir: str | Path | None,
        checkpoint_global_step: int,
        contract: OPDTelemetryContract,
        persist: bool,
        device_timing: bool = False,
    ) -> "OPDTelemetryMonitor":
        step = int(checkpoint_global_step)
        if checkpoint_dir is None or step == 0:
            return cls(
                output_dir=output_dir,
                contract=contract,
                persist=persist,
                device_timing=device_timing,
            )
        path = cls.snapshot_path(checkpoint_dir)
        local_error = None
        frames: tuple[OPDTelemetryFrame, ...] = ()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            expected = {"schema_version", "rank", "world_size", "global_step", "contract", "frames"}
            if not isinstance(payload, dict) or set(payload) != expected:
                raise ValueError("snapshot fields differ from the OPD telemetry schema")
            if int(payload["schema_version"]) != _SNAPSHOT_VERSION:
                raise ValueError("unsupported OPD telemetry snapshot version")
            if int(payload["rank"]) != _rank() or int(payload["world_size"]) != _world_size():
                raise ValueError("OPD telemetry snapshot topology differs from runtime")
            if int(payload["global_step"]) != step:
                raise ValueError("OPD telemetry snapshot global step differs from checkpoint")
            if OPDTelemetryContract(**dict(payload["contract"])) != contract:
                raise ValueError("OPD telemetry snapshot contract differs from runtime")
            frames = tuple(OPDTelemetryFrame(**dict(value)) for value in payload["frames"])
            if tuple(frame.global_step for frame in frames) != tuple(range(1, step + 1)):
                raise ValueError("OPD telemetry snapshot is not a contiguous complete history")
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            local_error = f"{type(exc).__name__}: {exc}"
        errors = _all_gather_object(local_error)
        if any(error is not None for error in errors):
            raise ValueError(f"OPD telemetry exact resume validation failed: {errors}")
        return cls(
            output_dir=output_dir,
            contract=contract,
            initial_global_step=step,
            frames=frames,
            persist=persist,
            device_timing=device_timing,
        )

    def bind_update_applied_provider(self, provider) -> None:
        self._update_applied_provider = provider

    def bind_component(self, component: Any) -> None:
        binder = getattr(component, "bind_telemetry", None)
        if callable(binder):
            binder(self)

    def stage_microbatch(self, stats: dict[str, int]) -> None:
        if self._window is None:
            self._window = _OPDWindow()
            if self.device_timing:
                event = torch.cuda.Event(enable_timing=True)
                event.record()
                self._window.device_frame_start = event
        window = self._window
        window.microbatches += 1
        for name in ("prompt_tokens", "materialized_prompt_tokens", "vision_patches"):
            value = int(stats.get(name, 0))
            if value < 0:
                raise ValueError(f"OPD telemetry batch stat {name} must be >= 0.")
            setattr(window, name, int(getattr(window, name)) + value)

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        if name not in _PHASES:
            raise ValueError(f"Unknown OPD telemetry phase {name!r}.")
        started = time.perf_counter()
        device_start = None
        if self.device_timing and self._window is not None:
            device_start = torch.cuda.Event(enable_timing=True)
            device_start.record()
        try:
            yield
        finally:
            if self._window is not None:
                self._window.phases[name] += max(time.perf_counter() - started, 0.0)
                if device_start is not None:
                    device_end = torch.cuda.Event(enable_timing=True)
                    device_end.record()
                    self._window.device_phase_events[name].append(
                        (device_start, device_end)
                    )

    def record_completion_tokens(self, count: int) -> None:
        if self._window is not None:
            self._window.completion_tokens += max(int(count), 0)

    def record_teacher_distribution(self, distribution: Any) -> None:
        if self._window is None:
            return
        if distribution.kind == "dense_logits":
            assert distribution.dense_logits is not None
            self._window.dense_teacher_elements += int(distribution.dense_logits.numel())
        else:
            assert distribution.topk_token_ids is not None
            self._window.topk_teacher_elements += int(distribution.topk_token_ids.numel())

    def record_teacher_transfer(self, *, request_bytes: int, response_bytes: int) -> None:
        if self._window is not None:
            self._window.teacher_request_bytes += max(int(request_bytes), 0)
            self._window.teacher_response_bytes += max(int(response_bytes), 0)

    def finish_training_step(self, seconds: float) -> None:
        if self._window is not None:
            self._window.training_step_seconds += max(float(seconds), 0.0)

    def start_optimizer_step(self) -> None:
        if self._window is not None:
            self._optimizer_started_at = time.perf_counter()

    def finish_optimizer_step(self) -> None:
        if self._window is not None and self._optimizer_started_at is not None:
            self._window.optimizer_seconds += max(
                time.perf_counter() - self._optimizer_started_at,
                0.0,
            )
        self._optimizer_started_at = None
        if self.device_timing and self._window is not None:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self._window.device_frame_end = event

    def discard_window(self) -> None:
        self._window = None
        self._optimizer_started_at = None

    def commit(self, *, global_step: int) -> None:
        if self._window is None:
            raise RuntimeError("Cannot commit OPD telemetry without a staged optimizer window.")
        expected = self.frames[-1].global_step + 1 if self.frames else self.initial_global_step + 1
        if int(global_step) != expected:
            raise ValueError(
                f"OPD telemetry step is not contiguous: actual={global_step} expected={expected}."
            )
        measured = sum(self._window.phases.values())
        overhead = max(self._window.training_step_seconds - measured, 0.0)
        frame = OPDTelemetryFrame(
            global_step=int(global_step),
            microbatches=self._window.microbatches,
            prompt_tokens=self._window.prompt_tokens,
            materialized_prompt_tokens=self._window.materialized_prompt_tokens,
            completion_tokens=self._window.completion_tokens,
            vision_patches=self._window.vision_patches,
            dense_teacher_elements=self._window.dense_teacher_elements,
            topk_teacher_elements=self._window.topk_teacher_elements,
            teacher_request_bytes=self._window.teacher_request_bytes,
            teacher_response_bytes=self._window.teacher_response_bytes,
            rollout_weight_sync_seconds=self._window.phases["rollout_weight_sync"],
            rollout_generate_seconds=self._window.phases["rollout_generate"],
            student_score_seconds=self._window.phases["student_score"],
            teacher_score_seconds=self._window.phases["teacher_score"],
            objective_seconds=self._window.phases["objective"],
            backward_and_trainer_overhead_seconds=overhead,
            optimizer_seconds=self._window.optimizer_seconds,
            update_applied=bool(self._update_applied_provider()),
        )
        pending_device_frame = None
        if self.device_timing:
            if (
                self._window.device_frame_start is None
                or self._window.device_frame_end is None
            ):
                raise RuntimeError("OPD CUDA timing frame is incomplete at commit.")
            pending_device_frame = _OPDPendingDeviceFrame(
                frame_start=self._window.device_frame_start,
                frame_end=self._window.device_frame_end,
                phase_events={
                    name: tuple(events)
                    for name, events in self._window.device_phase_events.items()
                },
            )
        self.frames.append(frame)
        if pending_device_frame is not None:
            self._device_frames[int(global_step)] = pending_device_frame
        self.discard_window()

    @staticmethod
    def snapshot_path(checkpoint_dir: str | Path) -> Path:
        return Path(checkpoint_dir) / f"shaft_opd_telemetry_rank{_rank()}.json"

    def write_checkpoint_snapshot(self, checkpoint_dir: str | Path, *, global_step: int) -> None:
        if not self.persist:
            return
        if not self.frames or self.frames[-1].global_step != int(global_step):
            raise ValueError("OPD telemetry snapshot does not cover the checkpoint step.")
        self._resolve_device_frames()
        path = self.snapshot_path(checkpoint_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": _SNAPSHOT_VERSION,
            "rank": _rank(),
            "world_size": _world_size(),
            "global_step": int(global_step),
            "contract": asdict(self.contract),
            "frames": [asdict(frame) for frame in self.frames],
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(temporary, path)

    def finalize(self, *, final_global_step: int) -> dict[str, float]:
        if self._window is not None:
            raise RuntimeError("OPD telemetry has an uncommitted optimizer window at finalize.")
        expected = self.frames[-1].global_step if self.frames else self.initial_global_step
        if expected != int(final_global_step):
            raise ValueError("OPD telemetry committed span differs from Trainer global_step.")
        self._resolve_device_frames()
        rank_frames = _all_gather_object([asdict(frame) for frame in self.frames])
        rank_steps = [
            tuple(int(frame["global_step"]) for frame in frames)
            for frames in rank_frames
        ]
        if any(steps != rank_steps[0] for steps in rank_steps[1:]):
            raise ValueError("OPD telemetry rank histories have different optimizer steps.")
        metrics: dict[str, float] = {}
        if _rank() == 0:
            frames_by_rank = [
                [OPDTelemetryFrame(**frame) for frame in frames]
                for frames in rank_frames
            ]
            flat = [frame for frames in frames_by_rank for frame in frames]
            frames_by_step = list(zip(*frames_by_rank, strict=True))
            total_seconds = sum(
                max(frame.critical_path_seconds for frame in step_frames)
                for step_frames in frames_by_step
            )
            total_completion = sum(frame.completion_tokens for frame in flat)
            metrics = {
                "opd_efficiency/completion_tokens_per_second": (
                    total_completion / total_seconds if total_seconds > 0 else 0.0
                ),
                "opd_efficiency/rollout_seconds": sum(
                    max(frame.rollout_generate_seconds for frame in step_frames)
                    for step_frames in frames_by_step
                ),
                "opd_efficiency/teacher_seconds": sum(
                    max(frame.teacher_score_seconds for frame in step_frames)
                    for step_frames in frames_by_step
                ),
                "opd_efficiency/student_seconds": sum(
                    max(frame.student_score_seconds for frame in step_frames)
                    for step_frames in frames_by_step
                ),
                "opd_efficiency/local_device_seconds": sum(
                    max(
                        float(frame.device_optimizer_frame_seconds or 0.0)
                        for frame in step_frames
                    )
                    for step_frames in frames_by_step
                ),
            }
            if self.persist:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                path = self.output_dir / OPD_TELEMETRY_FILENAME
                temporary = path.with_suffix(path.suffix + ".tmp")
                temporary.write_text(
                    json.dumps(
                        {
                            "schema_version": _SNAPSHOT_VERSION,
                            "world_size": _world_size(),
                            "final_global_step": int(final_global_step),
                            "contract": asdict(self.contract),
                            "rank_frames": rank_frames,
                            "metrics": metrics,
                        },
                        sort_keys=True,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                os.replace(temporary, path)
        return metrics

    def _resolve_device_frames(self) -> None:
        if not self._device_frames:
            return
        pending_steps = sorted(self._device_frames)
        self._device_frames[pending_steps[-1]].frame_end.synchronize()
        by_step = {frame.global_step: index for index, frame in enumerate(self.frames)}
        for step in pending_steps:
            index = by_step.get(step)
            if index is None:
                raise RuntimeError(f"OPD CUDA timing has no committed frame for step {step}.")
            pending = self._device_frames[step]
            phase_seconds = {
                name: sum(
                    max(float(start.elapsed_time(end)) / 1000.0, 0.0)
                    for start, end in events
                )
                for name, events in pending.phase_events.items()
            }
            self.frames[index] = replace(
                self.frames[index],
                **{
                    f"device_{name}_seconds": phase_seconds[name]
                    for name in _PHASES
                },
                device_optimizer_frame_seconds=max(
                    float(pending.frame_start.elapsed_time(pending.frame_end)) / 1000.0,
                    0.0,
                ),
            )
        self._device_frames.clear()


class OPDTelemetryCallback(TrainerCallback):
    def __init__(self, monitor: OPDTelemetryMonitor) -> None:
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
        self.monitor.commit(global_step=int(state.global_step))
        return control

    def on_save(self, args, state, control, **kwargs):  # noqa: ANN001
        _ = kwargs
        self.monitor.write_checkpoint_snapshot(
            Path(args.output_dir) / f"checkpoint-{int(state.global_step)}",
            global_step=int(state.global_step),
        )
        return control


def _rank() -> int:
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0


def _world_size() -> int:
    return torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1


def _all_gather_object(value: Any) -> list[Any]:
    if not torch.distributed.is_initialized():
        return [value]
    output = [None for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather_object(output, value)
    return output
