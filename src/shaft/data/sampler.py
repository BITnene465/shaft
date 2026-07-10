from __future__ import annotations

from collections.abc import Iterator
import logging
import math
import time

from torch.utils.data import Sampler

from shaft.utils.distributed import get_rank, get_world_size

from .mixing import ShaftSamplePlan, ShaftSampleRef, _affine_permute, _splitmix64
from .batching import ShaftFixedBatchPlanner
from .cost import ShaftSampleCostProvider


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


class ShaftCostAwareSampler(Sampler[ShaftSampleRef]):
    """Emit a fixed-cardinality BatchPlan as an ordered sample-ref stream.

    HF's regular BatchSampler still owns physical grouping. Accelerate then shards
    consecutive local batches across data ranks, so Phase 1 can balance global
    microsteps without changing Trainer's fixed batch/accumulation semantics.
    """

    def __init__(
        self,
        plan: ShaftSamplePlan,
        *,
        cost_provider: ShaftSampleCostProvider,
        per_device_batch_size: int,
        data_world_size: int,
        planning_window: int,
        gradient_accumulation_steps: int = 1,
        seed: int = 42,
        drop_last: bool = False,
    ) -> None:
        self.plan = plan
        self.plan_cycle = 0
        self.planner = ShaftFixedBatchPlanner(
            plan=plan,
            cost_provider=cost_provider,
            per_device_batch_size=per_device_batch_size,
            data_world_size=data_world_size,
            planning_window=planning_window,
            seed=seed,
            drop_last=drop_last,
        )
        self.signature = self.planner.build_signature(
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

    def __iter__(self) -> Iterator[ShaftSampleRef]:
        window_iterator = iter(
            self.planner.iter_window_plans(plan_cycle=self.plan_cycle)
        )
        window_count = 0
        sample_count = 0
        useful_llm_tokens = 0
        baseline_padded_llm_tokens = 0
        planned_padded_llm_tokens = 0
        supervised_tokens = 0
        loss_weight_sum = 0.0
        loss_weight_sum_known = True
        vision_patches = 0
        inexact_sample_count = 0
        rank_skew_weighted_sum = 0.0
        global_microstep_count = 0
        max_rank_cost_skew = 0.0
        planning_seconds = 0.0
        try:
            while True:
                planning_started = time.perf_counter()
                try:
                    window_plan = next(window_iterator)
                except StopIteration:
                    break
                window_planning_seconds = time.perf_counter() - planning_started
                planning_seconds += window_planning_seconds
                stats = window_plan.stats
                log = logger.info if window_count == 0 else logger.debug
                log(
                    "[batch-plan] cycle=%s window=%s:%s samples=%s "
                    "useful_llm_tokens=%s supervised_tokens=%s loss_weight_sum=%s "
                    "vision_patches=%s "
                    "inexact_samples=%s planning_seconds=%.6f "
                    "padding=%.4f baseline_padding=%.4f rank_skew_max=%.4f "
                    "plan_fingerprint=%s",
                    self.plan_cycle,
                    window_plan.window_start,
                    window_plan.window_stop,
                    stats.sample_count,
                    stats.useful_llm_tokens,
                    stats.supervised_tokens,
                    stats.loss_weight_sum,
                    stats.vision_patches,
                    stats.inexact_sample_count,
                    window_planning_seconds,
                    stats.padding_ratio,
                    stats.baseline_padding_ratio,
                    stats.max_rank_cost_skew,
                    window_plan.fingerprint[:12],
                )
                window_count += 1
                sample_count += stats.sample_count
                useful_llm_tokens += stats.useful_llm_tokens
                baseline_padded_llm_tokens += stats.baseline_padded_llm_tokens
                planned_padded_llm_tokens += stats.planned_padded_llm_tokens
                supervised_tokens += stats.supervised_tokens
                if stats.loss_weight_sum is None:
                    loss_weight_sum_known = False
                else:
                    loss_weight_sum += stats.loss_weight_sum
                vision_patches += stats.vision_patches
                inexact_sample_count += stats.inexact_sample_count
                rank_skew_weighted_sum += (
                    stats.average_rank_cost_skew * stats.global_microstep_count
                )
                global_microstep_count += stats.global_microstep_count
                max_rank_cost_skew = max(
                    max_rank_cost_skew,
                    stats.max_rank_cost_skew,
                )
                yield from window_plan.sample_refs
        finally:
            if window_count > 0:
                baseline_padding_ratio = 1.0 - useful_llm_tokens / max(
                    baseline_padded_llm_tokens,
                    1,
                )
                padding_ratio = 1.0 - useful_llm_tokens / max(
                    planned_padded_llm_tokens,
                    1,
                )
                average_rank_cost_skew = rank_skew_weighted_sum / max(
                    global_microstep_count,
                    1,
                )
                logger.info(
                    "[batch-plan-summary] cycle=%s windows=%s samples=%s "
                    "useful_llm_tokens=%s supervised_tokens=%s loss_weight_sum=%s "
                    "vision_patches=%s "
                    "inexact_samples=%s planning_seconds=%.6f padding=%.4f "
                    "baseline_padding=%.4f rank_skew_avg=%.4f rank_skew_max=%.4f "
                    "signature=%s",
                    self.plan_cycle,
                    window_count,
                    sample_count,
                    useful_llm_tokens,
                    supervised_tokens,
                    loss_weight_sum if loss_weight_sum_known else None,
                    vision_patches,
                    inexact_sample_count,
                    planning_seconds,
                    padding_ratio,
                    baseline_padding_ratio,
                    average_rank_cost_skew,
                    max_rank_cost_skew,
                    self.signature.fingerprint[:12],
                )

    def __len__(self) -> int:
        return len(self.planner)

    def set_epoch(self, epoch: int) -> None:
        self.plan_cycle = int(epoch)


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
