from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
import logging
import math
import time

from torch.utils.data import Sampler

from shaft.utils.distributed import get_rank, get_world_size

from .batching import ShaftLocalMicroBatchPlan
from .cost import ShaftSampleCostProvider
from .dynamic_batching import (
    ShaftBoundedBatchPlanner,
    ShaftBoundedBatchingSpec,
    ShaftBoundedBatchingState,
    ShaftBoundedMicrobatchPlan,
)
from .mixing import ShaftSamplePlan, ShaftSampleRef, ShaftSampleSchedule, _affine_permute, _splitmix64


logger = logging.getLogger(__name__)


class ShaftSampleSampler(Sampler[ShaftSampleRef]):
    """Lazily emit immutable refs; no Python tuple plan is materialized or copied."""

    def __init__(
        self,
        plan: ShaftSamplePlan,
        *,
        rank: int | None = None,
        world_size: int | None = None,
        drop_last: bool = False,
    ) -> None:
        self.plan = plan
        self.rank = int(get_rank() if rank is None else rank)
        self.world_size = int(get_world_size() if world_size is None else world_size)
        self.drop_last = bool(drop_last)
        self.plan_cycle = 0
        if self.rank < 0 or self.rank >= self.world_size:
            raise ValueError(f"Invalid sampler rank/world_size: {self.rank}/{self.world_size}.")

    def __iter__(self) -> Iterator[ShaftSampleRef]:
        total_size = len(self.plan)
        if self.drop_last:
            total_size -= total_size % self.world_size
        for position in range(self.rank, total_size, self.world_size):
            yield self.plan.ref_at(position, plan_cycle=self.plan_cycle)

    def __len__(self) -> int:
        total_size = len(self.plan)
        if self.drop_last:
            return total_size // self.world_size
        if total_size <= self.rank:
            return 0
        return int(math.ceil((total_size - self.rank) / self.world_size))

    def set_epoch(self, epoch: int) -> None:
        self.plan_cycle = int(epoch)


class ShaftBoundedBatchSampler(Sampler[list[ShaftSampleRef]]):
    """Yield variable-cardinality global batches for Accelerate rank sharding.

    The flattened order is ``[planning frame][microstep][rank]``.  A complete
    planning frame is planned before any of its batches are yielded, so a cost or
    capacity failure cannot surface after a partial frame.
    """

    batch_size = None
    drop_last = True

    def __init__(
        self,
        schedule: ShaftSampleSchedule,
        *,
        cost_provider: ShaftSampleCostProvider,
        spec: ShaftBoundedBatchingSpec,
        global_microstep_count: int,
        planning_frame_size: int,
        initial_state: ShaftBoundedBatchingState | None = None,
    ) -> None:
        self.schedule = schedule
        self.cost_provider = cost_provider
        self.spec = spec
        self.global_microstep_count = int(global_microstep_count)
        self.planning_frame_size = int(planning_frame_size)
        if self.global_microstep_count <= 0:
            raise ValueError("global_microstep_count must be > 0.")
        if self.planning_frame_size <= 0:
            raise ValueError("planning_frame_size must be > 0.")
        if initial_state is None:
            initial_state = ShaftBoundedBatchingState(
                contract_fingerprint=spec.fingerprint,
            )
        if initial_state.contract_fingerprint != spec.fingerprint:
            raise ValueError("Initial bounded batching state has a different contract.")
        if int(initial_state.global_microstep) % self.planning_frame_size != 0:
            raise ValueError(
                "Bounded batching can only resume at a planning-frame boundary."
            )
        if int(initial_state.global_microstep) > self.global_microstep_count:
            raise ValueError("Bounded batching resume state is beyond configured duration.")
        self.initial_state = initial_state
        self._committed_state = initial_state
        self._snapshots: dict[int, ShaftBoundedBatchingState] = {
            int(initial_state.global_microstep): initial_state
        }

    @property
    def sampler(self) -> "ShaftBoundedBatchSampler":
        """Expose set_epoch through Accelerate's BatchSamplerShard."""

        return self

    def __len__(self) -> int:
        remaining = self.global_microstep_count - int(self.initial_state.global_microstep)
        return remaining * int(self.spec.data_world_size)

    def __iter__(self) -> Iterator[list[ShaftSampleRef]]:
        planner = ShaftBoundedBatchPlanner(
            schedule=self.schedule,
            cost_provider=self.cost_provider,
            spec=self.spec,
            state=self.initial_state,
        )
        remaining_microsteps = self.global_microstep_count - int(
            self.initial_state.global_microstep
        )
        if remaining_microsteps % self.planning_frame_size != 0:
            raise RuntimeError(
                "Remaining bounded microsteps do not form complete planning frames."
            )

        planning_seconds = 0.0
        emitted_samples = 0
        emitted_llm_tokens = 0
        emitted_padded_tokens = 0
        emitted_supervised_tokens = 0
        emitted_vision_patches = 0
        max_local_batch_size = 0
        max_rank_skew = 0.0
        max_frame_rank_skew = 0.0
        frame_count = remaining_microsteps // self.planning_frame_size
        try:
            for _ in range(frame_count):
                started = time.perf_counter()
                frame, frame_rank_skew = self._balance_planning_frame(tuple(
                    planner.next_global_microbatch()
                    for _ in range(self.planning_frame_size)
                ))
                planning_seconds += time.perf_counter() - started
                max_frame_rank_skew = max(max_frame_rank_skew, frame_rank_skew)
                for plan in frame:
                    stats = plan.stats
                    emitted_samples += stats.sample_count
                    emitted_llm_tokens += stats.useful_llm_tokens
                    emitted_padded_tokens += stats.padded_llm_tokens
                    emitted_supervised_tokens += stats.supervised_tokens
                    emitted_vision_patches += stats.vision_patches
                    max_local_batch_size = max(
                        max_local_batch_size,
                        stats.max_local_batch_size,
                    )
                    max_rank_skew = max(max_rank_skew, stats.max_rank_cost_skew)
                    if plan.global_microstep == int(self.initial_state.global_microstep):
                        logger.info(
                            "[bounded-batch] buffer=%s world=%s max_samples=%s "
                            "max_padded_tokens=%s max_vision_patches=%s "
                            "first_samples=%s first_padding=%.4f first_rank_skew=%.4f",
                            self.spec.buffer_size,
                            self.spec.data_world_size,
                            self.spec.max_samples_per_microbatch,
                            self.spec.max_padded_tokens,
                            self.spec.max_vision_patches,
                            stats.sample_count,
                            stats.padding_ratio,
                            stats.max_rank_cost_skew,
                        )
                state = frame[-1].state_after
                self._snapshots[int(state.global_microstep)] = state
                for plan in frame:
                    for local_batch in plan.rank_microbatches:
                        yield list(local_batch.sample_refs)
        finally:
            if emitted_samples:
                logger.info(
                    "[bounded-batch-planned-summary] microsteps=%s samples=%s "
                    "useful_llm_tokens=%s padded_llm_tokens=%s supervised_tokens=%s "
                    "vision_patches=%s max_local_batch=%s/%s padding=%.4f "
                    "microstep_rank_skew_max=%.4f frame_rank_skew_max=%.4f "
                    "planning_seconds=%.6f",
                    remaining_microsteps,
                    emitted_samples,
                    emitted_llm_tokens,
                    emitted_padded_tokens,
                    emitted_supervised_tokens,
                    emitted_vision_patches,
                    max_local_batch_size,
                    self.spec.max_samples_per_microbatch,
                    1.0 - emitted_llm_tokens / max(emitted_padded_tokens, 1),
                    max_rank_skew,
                    max_frame_rank_skew,
                    planning_seconds,
                )

    def commit_global_microstep(
        self,
        global_microstep: int,
    ) -> ShaftBoundedBatchingState:
        target_microstep = int(global_microstep)
        if target_microstep < int(self._committed_state.global_microstep):
            raise ValueError("Cannot move the committed bounded batching state backwards.")
        try:
            committed = self._snapshots[target_microstep]
        except KeyError as exc:
            raise RuntimeError(
                "No bounded batching snapshot exists for completed planning frame at "
                f"global_microstep={target_microstep}; producer/consumer state drifted."
            ) from exc
        self._committed_state = committed
        self._snapshots = {
            microstep: state
            for microstep, state in self._snapshots.items()
            if microstep >= target_microstep
        }
        return committed

    @property
    def committed_state(self) -> ShaftBoundedBatchingState:
        return self._committed_state

    @property
    def latest_planned_state(self) -> ShaftBoundedBatchingState:
        return self._snapshots[max(self._snapshots)]

    @property
    def executed_sample_count(self) -> int:
        return int(self._committed_state.emitted_samples) - int(
            self.initial_state.emitted_samples
        )

    def _balance_planning_frame(
        self,
        frame: tuple[ShaftBoundedMicrobatchPlan, ...],
    ) -> tuple[tuple[ShaftBoundedMicrobatchPlan, ...], float]:
        cumulative_load = [0.0] * int(self.spec.data_world_size)
        balanced: list[ShaftBoundedMicrobatchPlan] = []
        for plan in frame:
            batches = sorted(
                plan.rank_microbatches,
                key=lambda batch: (
                    self._normalized_batch_load(batch),
                    -min(ref.context.draw_id for ref in batch.sample_refs),
                ),
                reverse=True,
            )
            available_ranks = sorted(
                range(len(cumulative_load)),
                key=lambda rank: (
                    cumulative_load[rank],
                    _splitmix64(
                        int(self.spec.seed)
                        ^ int(plan.global_microstep)
                        ^ int(rank)
                    ),
                ),
            )
            assigned: list[ShaftLocalMicroBatchPlan | None] = [None] * len(
                cumulative_load
            )
            for batch, rank in zip(batches, available_ranks, strict=True):
                assigned[rank] = batch
                cumulative_load[rank] += self._normalized_batch_load(batch)
            balanced.append(
                replace(
                    plan,
                    rank_microbatches=tuple(
                        batch for batch in assigned if batch is not None
                    ),
                )
            )
        mean_load = sum(cumulative_load) / max(len(cumulative_load), 1)
        frame_rank_skew = (
            0.0
            if mean_load <= 0
            else max(
                abs(load - mean_load) / mean_load for load in cumulative_load
            )
        )
        return tuple(balanced), frame_rank_skew

    def _normalized_batch_load(self, batch: ShaftLocalMicroBatchPlan) -> float:
        text = batch.padded_llm_tokens / int(self.spec.max_padded_tokens)
        vision = (
            0.0
            if self.spec.max_vision_patches is None
            else batch.vision_patches / int(self.spec.max_vision_patches)
        )
        return text + vision

    def set_epoch(self, epoch: int) -> None:
        # A step-bounded loader is one deterministic stream.  HF may report a
        # synthetic epoch number on resume; draw ids, not epochs, own randomness.
        _ = epoch


class ShaftGroupedSampleSampler(Sampler[ShaftSampleRef]):
    """GRPO-compatible grouped repeats with deterministic, resumable plan cycles."""

    def __init__(
        self,
        plan: ShaftSamplePlan,
        *,
        mini_repeat_count: int,
        batch_size: int,
        repeat_count: int,
        shuffle: bool,
        seed: int,
    ) -> None:
        self.plan = plan
        self.mini_repeat_count = int(mini_repeat_count)
        self.batch_size = int(batch_size)
        self.repeat_count = int(repeat_count)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.plan_cycle = 0
        if min(self.mini_repeat_count, self.batch_size, self.repeat_count) <= 0:
            raise ValueError("Grouped sampler repeat counts and batch_size must be > 0.")

    def __iter__(self) -> Iterator[ShaftSampleRef]:
        usable_size = (len(self.plan) // self.batch_size) * self.batch_size
        permutation_seed = _splitmix64(self.seed ^ self.plan_cycle)
        for chunk_start in range(0, usable_size, self.batch_size):
            chunk: list[ShaftSampleRef] = []
            for offset in range(self.batch_size):
                position = chunk_start + offset
                if self.shuffle:
                    position = _affine_permute(
                        position,
                        len(self.plan),
                        seed=permutation_seed,
                    )
                chunk.append(self.plan.ref_at(position, plan_cycle=self.plan_cycle))
            for _ in range(self.repeat_count):
                for sample_ref in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield sample_ref

    def __len__(self) -> int:
        usable_size = (len(self.plan) // self.batch_size) * self.batch_size
        return usable_size * self.mini_repeat_count * self.repeat_count

    def set_epoch(self, epoch: int) -> None:
        self.plan_cycle = int(epoch)
