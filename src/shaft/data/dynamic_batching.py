from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterator

from .batching import (
    ShaftBatchPlanningSignature,
    ShaftGlobalMicroBatchPlan,
    ShaftLocalMicroBatchPlan,
)
from .cost import ShaftSampleCost, ShaftSampleCostProvider
from .mixing import ShaftSamplePlan, ShaftSampleRef, _affine_permute, _splitmix64


SHAFT_DYNAMIC_BATCH_PLAN_VERSION = "shaft-dynamic-cost-batch-plan-v1"
_DYNAMIC_PARTITION_SEARCH_NODE_LIMIT = 500_000


class ShaftDynamicPartitionSearchBudgetExceeded(RuntimeError):
    """Feasibility is unknown because the bounded exact fallback exhausted its budget."""


@dataclass(slots=True)
class _DynamicPartitionSearchFrame:
    entry_index: int
    state: tuple[object, ...]
    candidate_bins: tuple[int, ...]
    next_candidate: int = 0
    assigned_bin: int | None = None


@dataclass(frozen=True, slots=True)
class ShaftDynamicBatchPlanningContract:
    """Single source of dynamic optimizer-batch geometry and draw horizon."""

    optimizer_step_count: int
    data_world_size: int
    gradient_accumulation_steps: int
    max_samples_per_microbatch: int
    max_padded_tokens: int
    max_vision_patches: int | None
    target_samples: int | None
    target_supervised_tokens: int | None
    planning_window: int
    seed: int
    rank_balance: bool

    def __post_init__(self) -> None:
        for field_name in (
            "optimizer_step_count",
            "data_world_size",
            "gradient_accumulation_steps",
            "max_samples_per_microbatch",
            "max_padded_tokens",
            "planning_window",
        ):
            if int(getattr(self, field_name)) <= 0:
                raise ValueError(f"{field_name} must be > 0.")
        if self.max_vision_patches is not None and int(self.max_vision_patches) <= 0:
            raise ValueError("max_vision_patches must be > 0 when set.")
        if (self.target_samples is None) == (self.target_supervised_tokens is None):
            raise ValueError(
                "Dynamic planning requires exactly one optimizer target: "
                "target_samples or target_supervised_tokens."
            )
        if self.target_samples is not None and int(self.target_samples) <= 0:
            raise ValueError("target_samples must be > 0.")
        if (
            self.target_supervised_tokens is not None
            and int(self.target_supervised_tokens) <= 0
        ):
            raise ValueError("target_supervised_tokens must be > 0.")

        if self.target_samples is not None:
            if int(self.target_samples) < self.local_microbatch_slots:
                raise ValueError(
                    "target_samples must provide at least one sample per local "
                    "microbatch slot."
                )
            if int(self.target_samples) > self.max_samples_per_optimizer_step:
                raise ValueError(
                    "target_samples exceeds the per-step sample capacity."
                )
            if int(self.target_samples) > int(self.planning_window):
                raise ValueError(
                    "target_samples exceeds the dynamic planning_window draw cap."
                )
        if int(self.planning_window) < self.local_microbatch_slots:
            raise ValueError(
                "planning_window must provide at least one sample per local "
                "microbatch slot: "
                f"required={self.local_microbatch_slots}, "
                f"configured={self.planning_window}."
            )

    @property
    def local_microbatch_slots(self) -> int:
        return int(self.data_world_size) * int(self.gradient_accumulation_steps)

    @property
    def max_samples_per_optimizer_step(self) -> int:
        return self.local_microbatch_slots * int(self.max_samples_per_microbatch)

    @property
    def draw_capacity_per_optimizer_step(self) -> int:
        if self.target_samples is not None:
            return int(self.target_samples)
        return min(
            self.max_samples_per_optimizer_step,
            int(self.planning_window),
        )

    @property
    def sample_plan_horizon(self) -> int:
        return int(self.optimizer_step_count) * self.draw_capacity_per_optimizer_step

    def build_spec(self, plan: ShaftSamplePlan) -> ShaftDynamicBatchPlanningSpec:
        return ShaftDynamicBatchPlanningSpec(
            optimizer_step_count=self.optimizer_step_count,
            data_world_size=self.data_world_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            max_samples_per_microbatch=self.max_samples_per_microbatch,
            max_padded_tokens=self.max_padded_tokens,
            max_vision_patches=self.max_vision_patches,
            target_samples=self.target_samples,
            target_supervised_tokens=self.target_supervised_tokens,
            planning_window=self.planning_window,
            seed=self.seed,
            rank_balance=self.rank_balance,
            sample_plan_fingerprint=str(plan.fingerprint),
            source_sample_count=len(plan),
        )

    @classmethod
    def resolve(
        cls,
        *,
        optimizer_step_count: int,
        per_device_train_batch_size: int,
        data_world_size: int,
        gradient_accumulation_steps: int,
        max_samples_per_microbatch: int | None,
        max_padded_tokens: int,
        max_vision_patches: int | None,
        target_samples: int | None,
        target_supervised_tokens: int | None,
        planning_window: int,
        seed: int,
        rank_balance: bool,
    ) -> ShaftDynamicBatchPlanningContract:
        resolved_max_samples = (
            int(per_device_train_batch_size)
            if max_samples_per_microbatch is None
            else int(max_samples_per_microbatch)
        )
        if target_samples is None and target_supervised_tokens is None:
            target_samples = (
                int(per_device_train_batch_size)
                * int(data_world_size)
                * int(gradient_accumulation_steps)
            )
        return cls(
            optimizer_step_count=int(optimizer_step_count),
            data_world_size=int(data_world_size),
            gradient_accumulation_steps=int(gradient_accumulation_steps),
            max_samples_per_microbatch=resolved_max_samples,
            max_padded_tokens=int(max_padded_tokens),
            max_vision_patches=(
                None if max_vision_patches is None else int(max_vision_patches)
            ),
            target_samples=(None if target_samples is None else int(target_samples)),
            target_supervised_tokens=(
                None
                if target_supervised_tokens is None
                else int(target_supervised_tokens)
            ),
            planning_window=int(planning_window),
            seed=int(seed),
            rank_balance=bool(rank_balance),
        )


@dataclass(frozen=True, slots=True)
class ShaftDynamicBatchPlanningSpec(ShaftDynamicBatchPlanningContract):
    """Dynamic planning contract bound to one immutable SamplePlan."""

    sample_plan_fingerprint: str
    source_sample_count: int

    def __post_init__(self) -> None:
        ShaftDynamicBatchPlanningContract.__post_init__(self)
        if not str(self.sample_plan_fingerprint).strip():
            raise ValueError("sample_plan_fingerprint must not be empty.")
        expected_source_samples = (
            int(self.optimizer_step_count) * self.draw_capacity_per_optimizer_step
        )
        if int(self.source_sample_count) != expected_source_samples:
            raise ValueError(
                "Dynamic SamplePlan size does not match the optimizer-step draw horizon: "
                f"{self.source_sample_count} != {expected_source_samples}."
            )

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(repr(self._payload()).encode("utf-8")).hexdigest()

    def _payload(self) -> tuple[object, ...]:
        return (
            SHAFT_DYNAMIC_BATCH_PLAN_VERSION,
            self.sample_plan_fingerprint,
            self.source_sample_count,
            self.optimizer_step_count,
            self.data_world_size,
            self.gradient_accumulation_steps,
            self.max_samples_per_microbatch,
            self.max_padded_tokens,
            self.max_vision_patches,
            self.target_samples,
            self.target_supervised_tokens,
            self.planning_window,
            self.seed,
            self.rank_balance,
        )

    @classmethod
    def from_plan(
        cls,
        plan: ShaftSamplePlan,
        *,
        optimizer_step_count: int,
        data_world_size: int,
        gradient_accumulation_steps: int,
        max_samples_per_microbatch: int,
        max_padded_tokens: int,
        max_vision_patches: int | None,
        target_samples: int | None,
        target_supervised_tokens: int | None,
        planning_window: int,
        seed: int,
        rank_balance: bool,
    ) -> ShaftDynamicBatchPlanningSpec:
        contract = ShaftDynamicBatchPlanningContract(
            optimizer_step_count=int(optimizer_step_count),
            data_world_size=int(data_world_size),
            gradient_accumulation_steps=int(gradient_accumulation_steps),
            max_samples_per_microbatch=int(max_samples_per_microbatch),
            max_padded_tokens=int(max_padded_tokens),
            max_vision_patches=(
                None if max_vision_patches is None else int(max_vision_patches)
            ),
            target_samples=(None if target_samples is None else int(target_samples)),
            target_supervised_tokens=(
                None
                if target_supervised_tokens is None
                else int(target_supervised_tokens)
            ),
            planning_window=int(planning_window),
            seed=int(seed),
            rank_balance=bool(rank_balance),
        )
        return contract.build_spec(plan)


@dataclass(frozen=True, slots=True)
class ShaftDynamicBatchPlanStats:
    sample_count: int
    useful_llm_tokens: int
    baseline_padded_llm_tokens: int
    planned_padded_llm_tokens: int
    supervised_tokens: int
    loss_weight_sum: float | None
    vision_patches: int
    min_local_batch_size: int
    max_local_batch_size: int
    max_local_padded_tokens: int
    max_local_vision_patches: int
    average_rank_cost_skew: float
    max_rank_cost_skew: float

    @property
    def baseline_padding_ratio(self) -> float:
        if self.baseline_padded_llm_tokens <= 0:
            return 0.0
        return 1.0 - self.useful_llm_tokens / self.baseline_padded_llm_tokens

    @property
    def padding_ratio(self) -> float:
        if self.planned_padded_llm_tokens <= 0:
            return 0.0
        return 1.0 - self.useful_llm_tokens / self.planned_padded_llm_tokens


@dataclass(frozen=True, slots=True)
class ShaftDynamicBatchPlanSummary:
    optimizer_step_count: int
    optimizer_step_sample_counts: tuple[int, ...]
    selected_sample_count: int
    useful_llm_tokens: int
    padded_llm_tokens: int
    supervised_tokens: int
    loss_weight_sum: float | None
    vision_patches: int
    min_local_batch_size: int
    max_local_batch_size: int
    max_local_padded_tokens: int
    max_local_vision_patches: int
    average_rank_cost_skew: float
    max_rank_cost_skew: float

    @property
    def padding_ratio(self) -> float:
        if self.padded_llm_tokens <= 0:
            return 0.0
        return 1.0 - self.useful_llm_tokens / self.padded_llm_tokens


@dataclass(frozen=True, slots=True)
class ShaftOptimizerBatchPlan:
    optimizer_step: int
    draw_start: int
    draw_stop: int
    microsteps: tuple[ShaftGlobalMicroBatchPlan, ...]
    stats: ShaftDynamicBatchPlanStats
    fingerprint: str

    @property
    def sample_refs(self) -> tuple[ShaftSampleRef, ...]:
        return tuple(
            sample_ref
            for microstep in self.microsteps
            for local_batch in microstep.rank_microbatches
            for sample_ref in local_batch.sample_refs
        )


class ShaftDynamicBatchPlanner:
    """Consume a draw prefix into bounded variable-cardinality optimizer batches."""

    def __init__(
        self,
        *,
        plan: ShaftSamplePlan,
        cost_provider: ShaftSampleCostProvider,
        spec: ShaftDynamicBatchPlanningSpec,
    ) -> None:
        if spec.sample_plan_fingerprint != str(plan.fingerprint):
            raise ValueError("Dynamic planning spec does not belong to the SamplePlan.")
        if spec.source_sample_count != len(plan):
            raise ValueError("Dynamic planning spec size does not match the SamplePlan.")
        if (
            spec.target_supervised_tokens is not None
            and plan.strategy == "weighted"
            and not plan.shuffle
        ):
            raise ValueError(
                "Dynamic token targets cannot consume a weighted, unshuffled "
                "SamplePlan because its source map depends on the maximum horizon."
            )
        self.plan = plan
        self.cost_provider = cost_provider
        self.spec = spec

    def build_signature(self) -> ShaftBatchPlanningSignature:
        return ShaftBatchPlanningSignature.from_spec(
            self.spec,
            cost_fingerprint=str(getattr(self.cost_provider, "fingerprint", "")),
        )

    def iter_optimizer_steps(
        self,
        *,
        plan_cycle: int = 0,
    ) -> Iterator[ShaftOptimizerBatchPlan]:
        cursor = 0
        for optimizer_step in range(self.spec.optimizer_step_count):
            entries, draw_stop = self._select_optimizer_entries(
                cursor,
                plan_cycle=plan_cycle,
            )
            yield self._build_optimizer_batch(
                entries,
                optimizer_step=optimizer_step,
                draw_start=cursor,
                draw_stop=draw_stop,
                plan_cycle=plan_cycle,
            )
            cursor = draw_stop

    def count_selected_samples(self, *, plan_cycle: int = 0) -> int:
        """Resolve token-bounded draw prefixes without materializing batch tuples."""

        cursor = 0
        selected_sample_count = 0
        for _ in range(self.spec.optimizer_step_count):
            entries, cursor = self._select_optimizer_entries(
                cursor,
                plan_cycle=plan_cycle,
            )
            selected_sample_count += len(entries)
        return selected_sample_count

    def summarize(self, *, plan_cycle: int = 0) -> ShaftDynamicBatchPlanSummary:
        optimizer_step_count = 0
        optimizer_step_sample_counts: list[int] = []
        selected_sample_count = 0
        useful_llm_tokens = 0
        padded_llm_tokens = 0
        supervised_tokens = 0
        loss_weight_sum = 0.0
        loss_weight_sum_known = True
        vision_patches = 0
        min_local_batch_size: int | None = None
        max_local_batch_size = 0
        max_local_padded_tokens = 0
        max_local_vision_patches = 0
        rank_skew_sum = 0.0
        max_rank_cost_skew = 0.0
        for optimizer_batch in self.iter_optimizer_steps(plan_cycle=plan_cycle):
            stats = optimizer_batch.stats
            optimizer_step_count += 1
            optimizer_step_sample_counts.append(stats.sample_count)
            selected_sample_count += stats.sample_count
            useful_llm_tokens += stats.useful_llm_tokens
            padded_llm_tokens += stats.planned_padded_llm_tokens
            supervised_tokens += stats.supervised_tokens
            if stats.loss_weight_sum is None:
                loss_weight_sum_known = False
            else:
                loss_weight_sum += stats.loss_weight_sum
            vision_patches += stats.vision_patches
            min_local_batch_size = (
                stats.min_local_batch_size
                if min_local_batch_size is None
                else min(min_local_batch_size, stats.min_local_batch_size)
            )
            max_local_batch_size = max(
                max_local_batch_size,
                stats.max_local_batch_size,
            )
            max_local_padded_tokens = max(
                max_local_padded_tokens,
                stats.max_local_padded_tokens,
            )
            max_local_vision_patches = max(
                max_local_vision_patches,
                stats.max_local_vision_patches,
            )
            rank_skew_sum += stats.average_rank_cost_skew
            max_rank_cost_skew = max(
                max_rank_cost_skew,
                stats.max_rank_cost_skew,
            )
        return ShaftDynamicBatchPlanSummary(
            optimizer_step_count=optimizer_step_count,
            optimizer_step_sample_counts=tuple(optimizer_step_sample_counts),
            selected_sample_count=selected_sample_count,
            useful_llm_tokens=useful_llm_tokens,
            padded_llm_tokens=padded_llm_tokens,
            supervised_tokens=supervised_tokens,
            loss_weight_sum=(loss_weight_sum if loss_weight_sum_known else None),
            vision_patches=vision_patches,
            min_local_batch_size=(min_local_batch_size or 0),
            max_local_batch_size=max_local_batch_size,
            max_local_padded_tokens=max_local_padded_tokens,
            max_local_vision_patches=max_local_vision_patches,
            average_rank_cost_skew=(
                rank_skew_sum / max(optimizer_step_count, 1)
            ),
            max_rank_cost_skew=max_rank_cost_skew,
        )

    def _select_optimizer_entries(
        self,
        draw_start: int,
        *,
        plan_cycle: int,
    ) -> tuple[tuple[tuple[ShaftSampleRef, ShaftSampleCost], ...], int]:
        entries: list[tuple[ShaftSampleRef, ShaftSampleCost]] = []
        supervised_tokens = 0
        target_samples = self.spec.target_samples
        capacity = self.spec.draw_capacity_per_optimizer_step
        while len(entries) < capacity:
            position = draw_start + len(entries)
            if position >= len(self.plan):
                break
            sample_ref = self.plan.ref_at(position, plan_cycle=plan_cycle)
            cost = self.cost_provider(sample_ref)
            self._validate_sample_cost(sample_ref, cost)
            entries.append((sample_ref, cost))
            supervised_tokens += int(cost.supervised_tokens)
            if target_samples is not None and len(entries) >= int(target_samples):
                break
            if (
                self.spec.target_supervised_tokens is not None
                and len(entries) >= self.spec.local_microbatch_slots
                and supervised_tokens >= int(self.spec.target_supervised_tokens)
            ):
                break

        if len(entries) < self.spec.local_microbatch_slots:
            raise ValueError(
                "Dynamic optimizer batch cannot provide one sample per local microbatch."
            )
        if target_samples is not None and len(entries) != int(target_samples):
            raise ValueError("Dynamic SamplePlan ended before target_samples was reached.")
        if (
            self.spec.target_supervised_tokens is not None
            and supervised_tokens < int(self.spec.target_supervised_tokens)
        ):
            raise ValueError(
                "Dynamic optimizer batch reached its sample capacity before the "
                "target_supervised_tokens budget."
            )
        return tuple(entries), draw_start + len(entries)

    def _validate_sample_cost(
        self,
        sample_ref: ShaftSampleRef,
        cost: ShaftSampleCost,
    ) -> None:
        if not cost.exact:
            raise ValueError(
                "Dynamic hard budgets require exact sample costs; "
                f"draw_id={sample_ref.context.draw_id} is inexact."
            )
        if int(cost.llm_tokens) > int(self.spec.max_padded_tokens):
            raise ValueError(
                "Dynamic planner encountered an oversize sample: "
                f"draw_id={sample_ref.context.draw_id}, llm_tokens={cost.llm_tokens}, "
                f"max_padded_tokens={self.spec.max_padded_tokens}."
            )
        if (
            self.spec.max_vision_patches is not None
            and int(cost.vision_patches) > int(self.spec.max_vision_patches)
        ):
            raise ValueError(
                "Dynamic planner encountered an oversize sample: "
                f"draw_id={sample_ref.context.draw_id}, "
                f"vision_patches={cost.vision_patches}, "
                f"max_vision_patches={self.spec.max_vision_patches}."
            )

    def _build_optimizer_batch(
        self,
        entries: tuple[tuple[ShaftSampleRef, ShaftSampleCost], ...],
        *,
        optimizer_step: int,
        draw_start: int,
        draw_stop: int,
        plan_cycle: int,
    ) -> ShaftOptimizerBatchPlan:
        local_batches = self._partition_local_batches(
            entries,
            optimizer_step=optimizer_step,
            plan_cycle=plan_cycle,
        )
        microsteps = self._build_microsteps(
            local_batches,
            optimizer_step=optimizer_step,
            plan_cycle=plan_cycle,
        )
        stats = self._build_stats(entries, local_batches, microsteps)
        fingerprint_payload = (
            SHAFT_DYNAMIC_BATCH_PLAN_VERSION,
            self.spec.fingerprint,
            str(getattr(self.cost_provider, "fingerprint", "")),
            optimizer_step,
            draw_start,
            draw_stop,
            plan_cycle,
            tuple(
                tuple(
                    tuple(ref.context.draw_id for ref in batch.sample_refs)
                    for batch in microstep.rank_microbatches
                )
                for microstep in microsteps
            ),
        )
        return ShaftOptimizerBatchPlan(
            optimizer_step=optimizer_step,
            draw_start=draw_start,
            draw_stop=draw_stop,
            microsteps=microsteps,
            stats=stats,
            fingerprint=hashlib.sha256(
                repr(fingerprint_payload).encode("utf-8")
            ).hexdigest(),
        )

    def _partition_local_batches(
        self,
        entries: tuple[tuple[ShaftSampleRef, ShaftSampleCost], ...],
        *,
        optimizer_step: int,
        plan_cycle: int,
    ) -> tuple[ShaftLocalMicroBatchPlan, ...]:
        slot_count = self.spec.local_microbatch_slots
        vision_denominator = self._vision_balance_denominator(entries)
        ordered = sorted(
            entries,
            key=lambda item: (
                item[1].llm_tokens,
                item[1].vision_patches,
                _splitmix64(
                    self.spec.seed
                    ^ optimizer_step
                    ^ plan_cycle
                    ^ item[0].context.draw_id
                ),
            ),
            reverse=True,
        )
        bins: list[list[tuple[ShaftSampleRef, ShaftSampleCost]]] = [
            [ordered[index]] for index in range(slot_count)
        ]
        for entry in ordered[slot_count:]:
            candidates: list[tuple[float, int, int]] = []
            for bin_index, candidate in enumerate(bins):
                projected = [*candidate, entry]
                if not self._entries_fit(projected):
                    continue
                candidates.append(
                    (
                        self._entries_balance_score(
                            projected,
                            vision_denominator=vision_denominator,
                        ),
                        len(candidate),
                        bin_index,
                    )
                )
            if not candidates:
                fallback_bins = self._search_feasible_partition(
                    entries,
                    vision_denominator=vision_denominator,
                )
                if fallback_bins is None:
                    raise ValueError(
                        "Dynamic planner could not partition the selected draw prefix "
                        "within the configured hard budgets."
                    )
                bins = fallback_bins
                break
            _, _, selected_bin = min(candidates)
            bins[selected_bin].append(entry)

        local_batches = [
            ShaftLocalMicroBatchPlan(
                sample_refs=tuple(item[0] for item in entries_in_bin),
                sample_costs=tuple(item[1] for item in entries_in_bin),
            )
            for entries_in_bin in bins
        ]
        if self.spec.rank_balance:
            local_batches.sort(
                key=lambda batch: (
                    self._batch_balance_score(
                        batch,
                        vision_denominator=vision_denominator,
                    ),
                    batch.padded_llm_tokens,
                    batch.vision_patches,
                    tuple(ref.context.draw_id for ref in batch.sample_refs),
                )
            )
        return tuple(local_batches)

    def _search_feasible_partition(
        self,
        entries: tuple[tuple[ShaftSampleRef, ShaftSampleCost], ...],
        *,
        vision_denominator: int,
    ) -> list[list[tuple[ShaftSampleRef, ShaftSampleCost]]] | None:
        """Deterministic bounded fallback for feasible plans missed by greedy seeding."""

        slot_count = self.spec.local_microbatch_slots
        total_vision_patches = sum(cost.vision_patches for _, cost in entries)
        has_vision_limit = (
            self.spec.max_vision_patches is not None
            and int(self.spec.max_vision_patches) < total_vision_patches
        )
        ordered = sorted(
            entries,
            key=lambda item: (
                max(
                    item[1].llm_tokens / self.spec.max_padded_tokens,
                    (
                        item[1].vision_patches / max(vision_denominator, 1)
                        if has_vision_limit
                        else 0.0
                    ),
                ),
                item[1].llm_tokens / self.spec.max_padded_tokens
                + (
                    item[1].vision_patches / max(vision_denominator, 1)
                    if has_vision_limit
                    else 0.0
                ),
                item[1].llm_tokens,
                item[1].vision_patches,
                _splitmix64(self.spec.seed ^ item[0].context.draw_id),
            ),
            reverse=True,
        )
        bins: list[list[tuple[ShaftSampleRef, ShaftSampleCost]]] = [
            [] for _ in range(slot_count)
        ]
        failed_states: set[tuple[object, ...]] = set()
        visited_nodes = 0

        def remaining_sample_capacity(
            candidate: list[tuple[ShaftSampleRef, ShaftSampleCost]],
        ) -> int:
            if not candidate:
                return int(self.spec.max_samples_per_microbatch)
            max_llm_tokens = max(item[1].llm_tokens for item in candidate)
            total_cardinality_capacity = min(
                int(self.spec.max_samples_per_microbatch),
                int(self.spec.max_padded_tokens) // max_llm_tokens,
            )
            return total_cardinality_capacity - len(candidate)

        stack: list[_DynamicPartitionSearchFrame] = []
        entry_index = 0
        while True:
            visited_nodes += 1
            if visited_nodes > _DYNAMIC_PARTITION_SEARCH_NODE_LIMIT:
                raise ShaftDynamicPartitionSearchBudgetExceeded(
                    "Dynamic partition feasibility search exceeded its deterministic "
                    f"node limit ({_DYNAMIC_PARTITION_SEARCH_NODE_LIMIT}). Reduce the "
                    "optimizer target or tighten planning_window."
                )
            empty_bin_count = sum(not candidate for candidate in bins)
            remaining_entry_count = len(ordered) - entry_index
            pruned = remaining_entry_count < empty_bin_count
            if not pruned and remaining_entry_count > sum(
                remaining_sample_capacity(candidate) for candidate in bins
            ):
                pruned = True
            if not pruned and has_vision_limit:
                assert self.spec.max_vision_patches is not None
                remaining_vision_patches = sum(
                    item[1].vision_patches for item in ordered[entry_index:]
                )
                available_vision_capacity = sum(
                    int(self.spec.max_vision_patches)
                    - sum(item[1].vision_patches for item in candidate)
                    for candidate in bins
                )
                if remaining_vision_patches > available_vision_capacity:
                    pruned = True
            if not pruned and entry_index >= len(ordered):
                if empty_bin_count == 0:
                    return bins
                pruned = True

            if not pruned:
                state = (
                    entry_index,
                    tuple(
                        sorted(
                            (
                                len(candidate),
                                max(
                                    (item[1].llm_tokens for item in candidate),
                                    default=0,
                                ),
                                (
                                    sum(
                                        item[1].vision_patches for item in candidate
                                    )
                                    if has_vision_limit
                                    else 0
                                ),
                            )
                            for candidate in bins
                        )
                    ),
                )
                if state not in failed_states:
                    entry = ordered[entry_index]
                    candidates: list[tuple[int, float, int, int]] = []
                    seen_bin_states: set[tuple[int, int, int]] = set()
                    for bin_index, candidate in enumerate(bins):
                        bin_state = (
                            len(candidate),
                            max(
                                (item[1].llm_tokens for item in candidate),
                                default=0,
                            ),
                            (
                                sum(
                                    item[1].vision_patches for item in candidate
                                )
                                if has_vision_limit
                                else 0
                            ),
                        )
                        if bin_state in seen_bin_states:
                            continue
                        seen_bin_states.add(bin_state)
                        projected = [*candidate, entry]
                        if not self._entries_fit(projected):
                            continue
                        capacity_loss = remaining_sample_capacity(
                            candidate
                        ) - remaining_sample_capacity(projected)
                        candidates.append(
                            (
                                capacity_loss,
                                (
                                    len(projected)
                                    * max(item[1].llm_tokens for item in projected)
                                    / self.spec.max_padded_tokens
                                    + (
                                        sum(
                                            item[1].vision_patches
                                            for item in projected
                                        )
                                        / max(vision_denominator, 1)
                                        if has_vision_limit
                                        else 0.0
                                    )
                                ),
                                len(candidate),
                                bin_index,
                            )
                        )
                    candidate_bins = tuple(
                        bin_index for _, _, _, bin_index in sorted(candidates)
                    )
                    if candidate_bins:
                        stack.append(
                            _DynamicPartitionSearchFrame(
                                entry_index=entry_index,
                                state=state,
                                candidate_bins=candidate_bins,
                            )
                        )
                    else:
                        failed_states.add(state)

            while stack:
                frame = stack[-1]
                if frame.assigned_bin is not None:
                    bins[frame.assigned_bin].pop()
                    frame.assigned_bin = None
                if frame.next_candidate < len(frame.candidate_bins):
                    bin_index = frame.candidate_bins[frame.next_candidate]
                    frame.next_candidate += 1
                    bins[bin_index].append(ordered[frame.entry_index])
                    frame.assigned_bin = bin_index
                    entry_index = frame.entry_index + 1
                    break
                failed_states.add(frame.state)
                stack.pop()
            else:
                return None

    def _entries_fit(
        self,
        entries: list[tuple[ShaftSampleRef, ShaftSampleCost]],
    ) -> bool:
        if len(entries) > int(self.spec.max_samples_per_microbatch):
            return False
        padded_tokens = len(entries) * max(item[1].llm_tokens for item in entries)
        if padded_tokens > int(self.spec.max_padded_tokens):
            return False
        vision_patches = sum(item[1].vision_patches for item in entries)
        return not (
            self.spec.max_vision_patches is not None
            and vision_patches > int(self.spec.max_vision_patches)
        )

    def _entries_balance_score(
        self,
        entries: list[tuple[ShaftSampleRef, ShaftSampleCost]],
        *,
        vision_denominator: int | None = None,
    ) -> float:
        padded_tokens = len(entries) * max(item[1].llm_tokens for item in entries)
        vision_patches = sum(item[1].vision_patches for item in entries)
        resolved_vision_denominator = (
            self.spec.max_vision_patches
            or vision_denominator
            or max(vision_patches, 1)
        )
        return float(
            padded_tokens / self.spec.max_padded_tokens
            + vision_patches / resolved_vision_denominator
        )

    def _batch_balance_score(
        self,
        batch: ShaftLocalMicroBatchPlan,
        *,
        vision_denominator: int,
    ) -> float:
        resolved_vision_denominator = self.spec.max_vision_patches or max(
            vision_denominator,
            1,
        )
        return float(
            batch.padded_llm_tokens / self.spec.max_padded_tokens
            + batch.vision_patches / resolved_vision_denominator
        )

    def _vision_balance_denominator(
        self,
        entries: tuple[tuple[ShaftSampleRef, ShaftSampleCost], ...],
    ) -> int:
        if self.spec.max_vision_patches is not None:
            return int(self.spec.max_vision_patches)
        total_patches = sum(cost.vision_patches for _, cost in entries)
        average_patches = (
            total_patches + self.spec.local_microbatch_slots - 1
        ) // self.spec.local_microbatch_slots
        return max(
            average_patches,
            max((cost.vision_patches for _, cost in entries), default=0),
            1,
        )

    def _build_microsteps(
        self,
        local_batches: tuple[ShaftLocalMicroBatchPlan, ...],
        *,
        optimizer_step: int,
        plan_cycle: int,
    ) -> tuple[ShaftGlobalMicroBatchPlan, ...]:
        microsteps: list[ShaftGlobalMicroBatchPlan] = []
        for start in range(0, len(local_batches), self.spec.data_world_size):
            group = list(local_batches[start : start + self.spec.data_world_size])
            rotation = int(
                _splitmix64(
                    self.spec.seed
                    ^ optimizer_step
                    ^ plan_cycle
                    ^ start
                    ^ 0xA0761D6478BD642F
                )
                % self.spec.data_world_size
            )
            group = group[rotation:] + group[:rotation]
            microsteps.append(
                ShaftGlobalMicroBatchPlan(rank_microbatches=tuple(group))
            )

        permutation_seed = _splitmix64(
            self.spec.seed ^ optimizer_step ^ plan_cycle ^ 0xE7037ED1A0B428DB
        )
        return tuple(
            microsteps[
                _affine_permute(
                    position,
                    len(microsteps),
                    seed=permutation_seed,
                )
            ]
            for position in range(len(microsteps))
        )

    def _build_stats(
        self,
        entries: tuple[tuple[ShaftSampleRef, ShaftSampleCost], ...],
        local_batches: tuple[ShaftLocalMicroBatchPlan, ...],
        microsteps: tuple[ShaftGlobalMicroBatchPlan, ...],
    ) -> ShaftDynamicBatchPlanStats:
        useful_tokens = sum(cost.llm_tokens for _, cost in entries)
        baseline_batches = self._build_even_cardinality_baseline(entries)
        vision_denominator = self._vision_balance_denominator(entries)
        rank_skews: list[float] = []
        for microstep in microsteps:
            scores = [
                batch.padded_llm_tokens / self.spec.max_padded_tokens
                + batch.vision_patches
                / vision_denominator
                for batch in microstep.rank_microbatches
            ]
            rank_skews.append(max(scores) - min(scores))
        loss_weight_sum = (
            sum(float(cost.loss_weight_sum) for _, cost in entries)
            if all(cost.loss_weight_sum is not None for _, cost in entries)
            else None
        )
        return ShaftDynamicBatchPlanStats(
            sample_count=len(entries),
            useful_llm_tokens=useful_tokens,
            baseline_padded_llm_tokens=sum(
                batch.padded_llm_tokens for batch in baseline_batches
            ),
            planned_padded_llm_tokens=sum(
                batch.padded_llm_tokens for batch in local_batches
            ),
            supervised_tokens=sum(cost.supervised_tokens for _, cost in entries),
            loss_weight_sum=loss_weight_sum,
            vision_patches=sum(cost.vision_patches for _, cost in entries),
            min_local_batch_size=min(len(batch.sample_refs) for batch in local_batches),
            max_local_batch_size=max(len(batch.sample_refs) for batch in local_batches),
            max_local_padded_tokens=max(
                batch.padded_llm_tokens for batch in local_batches
            ),
            max_local_vision_patches=max(
                batch.vision_patches for batch in local_batches
            ),
            average_rank_cost_skew=(sum(rank_skews) / len(rank_skews)),
            max_rank_cost_skew=max(rank_skews),
        )

    def _build_even_cardinality_baseline(
        self,
        entries: tuple[tuple[ShaftSampleRef, ShaftSampleCost], ...],
    ) -> tuple[ShaftLocalMicroBatchPlan, ...]:
        slot_count = self.spec.local_microbatch_slots
        base_size, remainder = divmod(len(entries), slot_count)
        batches: list[ShaftLocalMicroBatchPlan] = []
        cursor = 0
        for slot in range(slot_count):
            size = base_size + int(slot < remainder)
            chunk = entries[cursor : cursor + size]
            cursor += size
            batches.append(
                ShaftLocalMicroBatchPlan(
                    sample_refs=tuple(item[0] for item in chunk),
                    sample_costs=tuple(item[1] for item in chunk),
                )
            )
        return tuple(batches)
