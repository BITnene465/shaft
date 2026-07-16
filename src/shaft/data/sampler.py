from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace
import hashlib
import logging
import math
import time

from torch.utils.data import Sampler

from shaft.utils.distributed import get_rank, get_world_size

from .batching import ShaftLocalMicroBatchPlan
from .cost import ShaftSampleCostProvider
from .dynamic_batching import (
    ShaftBatchPlanner,
    ShaftBatchPlanningSpec,
    ShaftBatchPlanningState,
    ShaftBatchMicrobatchPlan,
)
from .mixing import ShaftSamplePlan, ShaftSampleRef, ShaftSampleSchedule, _splitmix64
from .planned import ShaftPlannedSampleRef


logger = logging.getLogger(__name__)

_GROUPED_SAMPLE_CONTRACT_VERSION = "shaft-grouped-sample-contract-v1"


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

    def validate_epoch_sharding(
        self,
        *,
        per_device_batch_size: int,
        data_world_size: int,
        dataloader_drop_last: bool,
        require_equal_rank_batch_cardinality: bool = False,
    ) -> int:
        """Validate that rank sharding can preserve the canonical epoch exactly.

        Accelerate's default ``even_batches=True`` repeats samples from the start
        when the number of local batches is not divisible by the data world size.
        Shaft disables that padding for canonical sample plans, so an unequal
        number of rank-local steps would instead deadlock DDP. Reject only that
        geometry; a smaller final local batch is valid when every rank still owns
        the same number of steps.
        """

        batch_size = int(per_device_batch_size)
        world_size = int(data_world_size)
        self.plan.validate_data_world_size(world_size)
        if batch_size <= 0 or world_size <= 0:
            raise ValueError("Epoch sharding batch size and world size must be > 0.")
        sample_count = len(self.plan)
        if bool(dataloader_drop_last):
            if sample_count % batch_size != 0:
                raise ValueError(
                    "Shaft canonical sample plans cannot use dataloader_drop_last=True "
                    "when an incomplete local batch would be silently discarded."
                )
            local_batch_count = sample_count // batch_size
        else:
            local_batch_count = int(math.ceil(sample_count / batch_size))
        if local_batch_count % world_size != 0:
            raise ValueError(
                "Canonical epoch sharding would produce unequal per-rank train step "
                "counts without repeating or dropping samples: "
                f"samples={sample_count}, per_device_batch_size={batch_size}, "
                f"world_size={world_size}, local_batches={local_batch_count}. "
                "Use train.duration.unit='steps' or choose an epoch dataset/batch "
                "geometry whose local batch count is divisible by world_size."
            )
        if (
            bool(require_equal_rank_batch_cardinality)
            and world_size > 1
            and sample_count % (batch_size * world_size) != 0
        ):
            raise ValueError(
                "This training algorithm requires equal rank-local batch cardinality "
                "at every synchronized step, but the canonical epoch has a partial "
                "global tail: "
                f"samples={sample_count}, per_device_batch_size={batch_size}, "
                f"world_size={world_size}. Use train.duration.unit='steps' or align "
                "the epoch sample count to per_device_batch_size * world_size."
            )
        return local_batch_count // world_size


class ShaftPlannedBatchSampler(Sampler[list[ShaftPlannedSampleRef]]):
    """Yield globally planned batches for Accelerate rank sharding.

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
        spec: ShaftBatchPlanningSpec,
        global_microstep_count: int,
        planning_frame_size: int,
        initial_state: ShaftBatchPlanningState | None = None,
        preflight_plan: ShaftBatchMicrobatchPlan | None = None,
    ) -> None:
        self.schedule = schedule
        self.cost_provider = cost_provider
        self.spec = spec
        self.global_microstep_count = int(global_microstep_count)
        self.planning_frame_size = int(planning_frame_size)
        self.schedule.validate_data_world_size(int(spec.data_world_size))
        if self.global_microstep_count <= 0:
            raise ValueError("global_microstep_count must be > 0.")
        if self.planning_frame_size <= 0:
            raise ValueError("planning_frame_size must be > 0.")
        if initial_state is None:
            initial_state = ShaftBatchPlanningState(
                contract_fingerprint=spec.fingerprint,
            )
        initial_state.validate_against_spec(spec)
        if int(initial_state.global_microstep) % self.planning_frame_size != 0:
            raise ValueError(
                "Planned batching can only resume at a planning-frame boundary."
            )
        if int(initial_state.global_microstep) > self.global_microstep_count:
            raise ValueError("Planned batching resume state is beyond configured duration.")
        self.initial_state = initial_state
        if preflight_plan is not None:
            if int(preflight_plan.global_microstep) != int(
                initial_state.global_microstep
            ):
                raise ValueError(
                    "Planning preflight does not start at the initial state."
                )
            preflight_plan.state_after.validate_against_spec(spec)
            if int(preflight_plan.state_after.global_microstep) != (
                int(initial_state.global_microstep) + 1
            ):
                raise ValueError(
                    "Planning preflight must advance exactly one global microstep."
                )
        self.preflight_plan = preflight_plan
        self._committed_state = initial_state
        self._snapshots: dict[int, ShaftBatchPlanningState] = {
            int(initial_state.global_microstep): initial_state
        }

    @property
    def sampler(self) -> "ShaftPlannedBatchSampler":
        """Expose set_epoch through Accelerate's BatchSamplerShard."""

        return self

    def __len__(self) -> int:
        remaining = self.global_microstep_count - int(self.initial_state.global_microstep)
        return remaining * int(self.spec.data_world_size)

    def __iter__(self) -> Iterator[list[ShaftPlannedSampleRef]]:
        planner_state = (
            self.initial_state
            if self.preflight_plan is None
            else self.preflight_plan.state_after
        )
        planner = ShaftBatchPlanner(
            schedule=self.schedule,
            cost_provider=self.cost_provider,
            spec=self.spec,
            state=planner_state,
        )
        remaining_microsteps = self.global_microstep_count - int(
            self.initial_state.global_microstep
        )
        if remaining_microsteps % self.planning_frame_size != 0:
            raise RuntimeError(
                "Remaining microsteps do not form complete planning frames."
            )

        planning_seconds = 0.0
        planned_microsteps = 0
        emitted_logical_segments = 0
        emitted_physical_packs = 0
        emitted_llm_tokens = 0
        emitted_padded_tokens = 0
        emitted_supervised_tokens = 0
        emitted_vision_patches = 0
        max_local_pack_count = 0
        max_rank_skew = 0.0
        max_frame_rank_skew = 0.0
        frame_count = remaining_microsteps // self.planning_frame_size
        try:
            for frame_index in range(frame_count):
                started = time.perf_counter()
                frame_prefix = (
                    (self.preflight_plan,)
                    if frame_index == 0 and self.preflight_plan is not None
                    else ()
                )
                unbalanced_frame = frame_prefix + tuple(
                    planner.next_global_microbatch()
                    for _ in range(self.planning_frame_size - len(frame_prefix))
                )
                frame, frame_rank_skew = self._balance_planning_frame(
                    unbalanced_frame
                )
                planning_seconds += time.perf_counter() - started
                planned_microsteps += len(frame)
                max_frame_rank_skew = max(max_frame_rank_skew, frame_rank_skew)
                for plan in frame:
                    stats = plan.stats
                    emitted_logical_segments += stats.logical_segment_count
                    emitted_physical_packs += stats.physical_pack_count
                    emitted_llm_tokens += stats.useful_llm_tokens
                    emitted_padded_tokens += stats.padded_llm_tokens
                    emitted_supervised_tokens += stats.supervised_tokens
                    emitted_vision_patches += stats.vision_patches
                    max_local_pack_count = max(
                        max_local_pack_count,
                        stats.max_local_pack_count,
                    )
                    max_rank_skew = max(max_rank_skew, stats.max_rank_cost_skew)
                    if plan.global_microstep == int(self.initial_state.global_microstep):
                        logger.info(
                            "[batch-plan] grouping=%s packing=%s layout=%s "
                            "buffer=%s world=%s cardinality=%s "
                            "local_pack_range=%s "
                            "max_tokens=%s resource_budgets=%s "
                            "first_logical_segments=%s first_physical_packs=%s "
                            "first_padding=%.4f first_rank_skew=%.4f",
                            self.spec.grouping,
                            self.spec.packing,
                            self.spec.layout,
                            self.spec.buffer_size,
                            self.spec.data_world_size,
                            self.spec.cardinality,
                            self.spec.local_pack_count_bounds,
                            self.spec.max_tokens_per_microbatch,
                            dict(self.spec.resource_budgets),
                            stats.logical_segment_count,
                            stats.physical_pack_count,
                            stats.padding_ratio,
                            stats.max_rank_cost_skew,
                        )
                state = frame[-1].state_after
                self._snapshots[int(state.global_microstep)] = state
                for plan in frame:
                    for rank_index in range(len(plan.rank_microbatches)):
                        yield list(plan.planned_refs_for_rank(rank_index))
        finally:
            if emitted_logical_segments:
                logger.info(
                    "[batch-plan-summary] microsteps=%s logical_segments=%s "
                    "physical_packs=%s segments_per_pack=%.4f "
                    "useful_llm_tokens=%s padded_llm_tokens=%s supervised_tokens=%s "
                    "vision_patches=%s max_local_packs=%s/%s padding=%.4f "
                    "microstep_rank_skew_max=%.4f frame_rank_skew_max=%.4f "
                    "planning_seconds=%.6f",
                    planned_microsteps,
                    emitted_logical_segments,
                    emitted_physical_packs,
                    emitted_logical_segments / max(emitted_physical_packs, 1),
                    emitted_llm_tokens,
                    emitted_padded_tokens,
                    emitted_supervised_tokens,
                    emitted_vision_patches,
                    max_local_pack_count,
                    self.spec.local_pack_count_bounds[1],
                    1.0 - emitted_llm_tokens / max(emitted_padded_tokens, 1),
                    max_rank_skew,
                    max_frame_rank_skew,
                    planning_seconds,
                )

    def commit_global_microstep(
        self,
        global_microstep: int,
    ) -> ShaftBatchPlanningState:
        target_microstep = int(global_microstep)
        if target_microstep < int(self._committed_state.global_microstep):
            raise ValueError("Cannot move committed batch-planning state backwards.")
        try:
            committed = self._snapshots[target_microstep]
        except KeyError as exc:
            raise RuntimeError(
                "No batch-planning snapshot exists for completed planning frame at "
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
    def committed_state(self) -> ShaftBatchPlanningState:
        return self._committed_state

    @property
    def latest_planned_state(self) -> ShaftBatchPlanningState:
        return self._snapshots[max(self._snapshots)]

    @property
    def executed_sample_count(self) -> int:
        return int(self._committed_state.emitted_samples) - int(
            self.initial_state.emitted_samples
        )

    def _balance_planning_frame(
        self,
        frame: tuple[ShaftBatchMicrobatchPlan, ...],
    ) -> tuple[tuple[ShaftBatchMicrobatchPlan, ...], float]:
        cumulative_load = [0.0] * int(self.spec.data_world_size)
        balanced: list[ShaftBatchMicrobatchPlan] = []
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
        materialized_tokens = (
            batch.useful_llm_tokens
            if self.spec.layout == "varlen"
            else batch.padded_llm_tokens
        )
        text = materialized_tokens / int(self.spec.max_tokens_per_microbatch)
        resources = 0.0
        for resource_name, budget in self.spec.resource_budgets:
            value = batch.resource_total(resource_name)
            resources += value / int(budget)
        return text + resources

    def set_epoch(self, epoch: int) -> None:
        # A step-bounded loader is one deterministic stream.  HF may report a
        # synthetic epoch number on resume; draw ids, not epochs, own randomness.
        _ = epoch


@dataclass(frozen=True, slots=True)
class ShaftGroupedSampleContract:
    """Algorithm-agnostic grouped-repeat geometry and exact-resume identity."""

    mini_repeat_count: int
    batch_size: int
    iteration_count: int
    steps_per_iteration: int

    def __post_init__(self) -> None:
        if min(
            self.mini_repeat_count,
            self.batch_size,
            self.iteration_count,
            self.steps_per_iteration,
        ) <= 0:
            raise ValueError("Grouped sampler counts and batch_size must be > 0.")

    @property
    def repeat_count(self) -> int:
        return self.iteration_count * self.steps_per_iteration

    def finite_sample_plan_size(
        self,
        *,
        max_steps: int,
        gradient_accumulation_steps: int,
    ) -> int | None:
        """Resolve unique samples needed by a step-bounded grouped stream.

        ``max_steps`` counts optimizer updates, while the grouped sampler consumes
        one repeated generation group for ``repeat_count`` train microsteps.  The
        canonical plan therefore owns unique prompts, not the expanded samples
        seen by the dataloader.
        """

        optimizer_steps = int(max_steps)
        if optimizer_steps < 0:
            return None
        accumulation_steps = int(gradient_accumulation_steps)
        if accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be > 0.")
        required_microsteps = optimizer_steps * accumulation_steps
        generation_group_count = math.ceil(required_microsteps / self.repeat_count)
        return generation_group_count * self.batch_size

    def validate_plan_size(self, sample_count: int) -> None:
        sample_count = int(sample_count)
        if sample_count <= 0:
            raise ValueError("GRPO canonical sample plan must not be empty.")
        if sample_count % self.batch_size != 0:
            raise ValueError(
                "GRPO canonical sample plan must contain complete grouped batches; "
                "refusing to silently discard the epoch remainder: "
                f"samples={sample_count}, unique_prompts_per_group={self.batch_size}. "
                "Use train.duration.unit='steps' or align the epoch sample count."
            )

    def validate_epoch_sharding(
        self,
        *,
        sample_count: int,
        per_device_generation_batch_size: int,
        data_world_size: int,
        dataloader_drop_last: bool,
    ) -> int:
        """Return the proven number of rank-local GRPO microsteps per epoch."""

        self.validate_plan_size(sample_count)
        local_batch_size = int(per_device_generation_batch_size)
        world_size = int(data_world_size)
        if local_batch_size <= 0 or world_size <= 0:
            raise ValueError("Grouped epoch batch size and world size must be > 0.")
        expected_global_batch = self.batch_size * self.mini_repeat_count
        actual_global_batch = local_batch_size * world_size
        if actual_global_batch != expected_global_batch:
            raise ValueError(
                "GRPO grouped sampler contract differs from the distributed "
                "generation batch: "
                f"contract_global_batch={expected_global_batch}, "
                f"loader_global_batch={actual_global_batch}."
            )
        expanded_count = (
            int(sample_count) * self.mini_repeat_count * self.repeat_count
        )
        global_batch_count, remainder = divmod(expanded_count, local_batch_size)
        if remainder:
            raise ValueError(
                "GRPO grouped sampler does not form complete rank-local generation batches."
            )
        rank_local_microsteps, rank_remainder = divmod(
            global_batch_count,
            world_size,
        )
        if rank_remainder:
            raise ValueError(
                "GRPO grouped sampler would produce unequal per-rank train step counts."
            )
        if bool(dataloader_drop_last):
            raise ValueError(
                "GRPO grouped execution requires dataloader_drop_last=False; "
                "all grouped batches are already complete."
            )
        return rank_local_microsteps

    def execution_fingerprint(self, base_fingerprint: str) -> str:
        base_fingerprint = str(base_fingerprint).strip()
        if not base_fingerprint:
            raise ValueError("Grouped sample execution requires a base fingerprint.")
        payload = (
            _GROUPED_SAMPLE_CONTRACT_VERSION,
            base_fingerprint,
            self.mini_repeat_count,
            self.batch_size,
            self.iteration_count,
            self.steps_per_iteration,
        )
        return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


class ShaftGroupedSampleSampler(Sampler[ShaftSampleRef]):
    """GRPO grouped repeats over the canonical, resumable sample-plan order.

    The plan is the only shuffle source. Reordering finite-plan positions here can
    cross source permutation cycles and defeat source-local without-replacement.
    """

    def __init__(
        self,
        plan: ShaftSamplePlan,
        *,
        contract: ShaftGroupedSampleContract,
    ) -> None:
        self.plan = plan
        self.contract = contract
        self.mini_repeat_count = contract.mini_repeat_count
        self.batch_size = contract.batch_size
        self.repeat_count = contract.repeat_count
        self.plan_cycle = 0
        self.contract.validate_plan_size(len(self.plan))

    def __iter__(self) -> Iterator[ShaftSampleRef]:
        for chunk_start in range(0, len(self.plan), self.batch_size):
            chunk: list[ShaftSampleRef] = []
            for offset in range(self.batch_size):
                position = chunk_start + offset
                chunk.append(self.plan.ref_at(position, plan_cycle=self.plan_cycle))
            for _ in range(self.repeat_count):
                for sample_ref in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield sample_ref

    def __len__(self) -> int:
        return len(self.plan) * self.mini_repeat_count * self.repeat_count

    def set_epoch(self, epoch: int) -> None:
        self.plan_cycle = int(epoch)

    def validate_epoch_sharding(
        self,
        *,
        per_device_generation_batch_size: int,
        data_world_size: int,
        dataloader_drop_last: bool,
    ) -> int:
        """Prove grouped epochs shard into equal, complete rank-local steps."""

        self.plan.validate_data_world_size(int(data_world_size))
        return self.contract.validate_epoch_sharding(
            sample_count=len(self.plan),
            per_device_generation_batch_size=per_device_generation_batch_size,
            data_world_size=data_world_size,
            dataloader_drop_last=dataloader_drop_last,
        )
