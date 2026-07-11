from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Iterator

from .cost import ShaftSampleCost, ShaftSampleCostProvider
from .mixing import ShaftSamplePlan, ShaftSampleRef, _affine_permute, _splitmix64


_BATCH_PLAN_VERSION = "shaft-fixed-cost-batch-plan-v2"


def _log2_bucket(value: int) -> int:
    return (max(int(value), 1) - 1).bit_length()


def _optional_int(payload: dict[str, object], field_name: str) -> int | None:
    value = payload.get(field_name)
    return None if value is None else int(value)


def _resolve_fixed_batch_planning_geometry(
    *,
    sample_count: int,
    per_device_batch_size: int,
    data_world_size: int,
    planning_window: int,
    drop_last: bool,
) -> tuple[int, int, int]:
    sample_count = int(sample_count)
    per_device_batch_size = int(per_device_batch_size)
    data_world_size = int(data_world_size)
    planning_window = int(planning_window)
    if per_device_batch_size <= 0:
        raise ValueError("per_device_batch_size must be > 0.")
    if data_world_size <= 0:
        raise ValueError("data_world_size must be > 0.")
    if planning_window <= 0:
        raise ValueError("planning_window must be > 0.")
    global_microstep_samples = per_device_batch_size * data_world_size
    if planning_window < global_microstep_samples:
        raise ValueError(
            "planning_window must contain at least one complete global microstep: "
            f"planning_window={planning_window}, required={global_microstep_samples}."
        )
    remainder = sample_count % global_microstep_samples
    if remainder and not drop_last:
        raise ValueError(
            "Cost-aware fixed batching requires complete global microsteps; "
            f"plan size {sample_count} is not divisible by "
            f"per_device_batch_size({per_device_batch_size}) * "
            f"data_world_size({data_world_size}). Set drop_last=True or use a "
            "step sample budget that is globally divisible."
        )
    usable_sample_count = sample_count - remainder
    if usable_sample_count <= 0:
        raise ValueError(
            "Cost-aware fixed batching requires at least one complete global microstep."
        )
    effective_planning_window = (
        planning_window // global_microstep_samples
    ) * global_microstep_samples
    return global_microstep_samples, usable_sample_count, effective_planning_window


@dataclass(frozen=True, slots=True)
class ShaftFixedBatchPlanningSpec:
    """Single immutable source of fixed-cardinality planning geometry."""

    sample_plan_fingerprint: str
    source_sample_count: int
    usable_sample_count: int
    per_device_batch_size: int
    data_world_size: int
    gradient_accumulation_steps: int
    global_microstep_samples: int
    planning_window: int
    effective_planning_window: int
    seed: int
    drop_last: bool

    def __post_init__(self) -> None:
        if not str(self.sample_plan_fingerprint).strip():
            raise ValueError("sample_plan_fingerprint must not be empty.")
        if int(self.gradient_accumulation_steps) <= 0:
            raise ValueError("gradient_accumulation_steps must be > 0.")
        expected = _resolve_fixed_batch_planning_geometry(
            sample_count=self.source_sample_count,
            per_device_batch_size=self.per_device_batch_size,
            data_world_size=self.data_world_size,
            planning_window=self.planning_window,
            drop_last=self.drop_last,
        )
        actual = (
            int(self.global_microstep_samples),
            int(self.usable_sample_count),
            int(self.effective_planning_window),
        )
        if actual != expected:
            raise ValueError(
                "Fixed batch planning spec contains inconsistent derived geometry: "
                f"actual={actual}, expected={expected}."
            )

    @classmethod
    def from_plan(
        cls,
        plan: ShaftSamplePlan,
        *,
        per_device_batch_size: int,
        data_world_size: int,
        gradient_accumulation_steps: int,
        planning_window: int,
        seed: int = 42,
        drop_last: bool = False,
    ) -> ShaftFixedBatchPlanningSpec:
        source_sample_count = len(plan)
        (
            global_microstep_samples,
            usable_sample_count,
            effective_planning_window,
        ) = _resolve_fixed_batch_planning_geometry(
            sample_count=source_sample_count,
            per_device_batch_size=per_device_batch_size,
            data_world_size=data_world_size,
            planning_window=planning_window,
            drop_last=drop_last,
        )
        return cls(
            sample_plan_fingerprint=str(plan.fingerprint),
            source_sample_count=source_sample_count,
            usable_sample_count=usable_sample_count,
            per_device_batch_size=int(per_device_batch_size),
            data_world_size=int(data_world_size),
            gradient_accumulation_steps=int(gradient_accumulation_steps),
            global_microstep_samples=global_microstep_samples,
            planning_window=int(planning_window),
            effective_planning_window=effective_planning_window,
            seed=int(seed),
            drop_last=bool(drop_last),
        )


@dataclass(frozen=True, slots=True)
class ShaftLocalMicroBatchPlan:
    sample_refs: tuple[ShaftSampleRef, ...]
    sample_costs: tuple[ShaftSampleCost, ...]

    def __post_init__(self) -> None:
        if not self.sample_refs:
            raise ValueError("A local microbatch plan cannot be empty.")
        if len(self.sample_refs) != len(self.sample_costs):
            raise ValueError("Local microbatch refs and costs must have the same length.")

    @property
    def useful_llm_tokens(self) -> int:
        return sum(cost.llm_tokens for cost in self.sample_costs)

    @property
    def max_llm_tokens(self) -> int:
        return max(cost.llm_tokens for cost in self.sample_costs)

    @property
    def padded_llm_tokens(self) -> int:
        return len(self.sample_costs) * self.max_llm_tokens

    @property
    def supervised_tokens(self) -> int:
        return sum(cost.supervised_tokens for cost in self.sample_costs)

    @property
    def vision_patches(self) -> int:
        return sum(cost.vision_patches for cost in self.sample_costs)


@dataclass(frozen=True, slots=True)
class ShaftGlobalMicroBatchPlan:
    rank_microbatches: tuple[ShaftLocalMicroBatchPlan, ...]

    def __post_init__(self) -> None:
        if not self.rank_microbatches:
            raise ValueError("A global microbatch plan requires at least one data rank.")


@dataclass(frozen=True, slots=True)
class ShaftBatchPlanStats:
    sample_count: int
    local_batch_count: int
    global_microstep_count: int
    useful_llm_tokens: int
    baseline_padded_llm_tokens: int
    planned_padded_llm_tokens: int
    supervised_tokens: int
    loss_weight_sum: float | None
    vision_patches: int
    inexact_sample_count: int
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
class ShaftBatchPlan:
    window_start: int
    window_stop: int
    plan_cycle: int
    microsteps: tuple[ShaftGlobalMicroBatchPlan, ...]
    stats: ShaftBatchPlanStats
    fingerprint: str

    @property
    def sample_refs(self) -> tuple[ShaftSampleRef, ...]:
        return tuple(
            sample_ref
            for microstep in self.microsteps
            for local_batch in microstep.rank_microbatches
            for sample_ref in local_batch.sample_refs
        )


@dataclass(frozen=True, slots=True)
class ShaftBatchPlanningSignature:
    planner_version: str
    sample_plan_fingerprint: str
    cost_fingerprint: str
    source_sample_count: int
    sample_count: int | None
    per_device_batch_size: int | None
    data_world_size: int
    gradient_accumulation_steps: int
    planning_window: int
    effective_planning_window: int | None
    seed: int
    drop_last: bool | None
    strategy: str = "cost_aware"
    optimizer_step_count: int | None = None
    max_samples_per_microbatch: int | None = None
    max_padded_tokens: int | None = None
    max_vision_patches: int | None = None
    target_samples: int | None = None
    target_supervised_tokens: int | None = None
    rank_balance: bool | None = None
    planning_spec_fingerprint: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "planner_version",
            "sample_plan_fingerprint",
            "cost_fingerprint",
        ):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} must not be empty.")
        for field_name in (
            "source_sample_count",
            "data_world_size",
            "gradient_accumulation_steps",
            "planning_window",
        ):
            if int(getattr(self, field_name)) <= 0:
                raise ValueError(f"{field_name} must be > 0.")
        if self.strategy not in {"cost_aware", "dynamic_cost_aware"}:
            raise ValueError(f"Unsupported batch planning strategy={self.strategy!r}.")
        dynamic_values = (
            self.optimizer_step_count,
            self.max_samples_per_microbatch,
            self.max_padded_tokens,
            self.max_vision_patches,
            self.target_samples,
            self.target_supervised_tokens,
            self.rank_balance,
            self.planning_spec_fingerprint,
        )
        if self.strategy == "cost_aware":
            for field_name in (
                "sample_count",
                "per_device_batch_size",
                "effective_planning_window",
            ):
                value = getattr(self, field_name)
                if value is None or int(value) <= 0:
                    raise ValueError(f"{field_name} must be > 0 for fixed batching.")
            if self.drop_last is None:
                raise ValueError("drop_last must be set for fixed batching.")
            assert self.sample_count is not None
            if int(self.sample_count) > int(self.source_sample_count):
                raise ValueError("sample_count cannot exceed source_sample_count.")
            if not self.drop_last and int(self.sample_count) != int(
                self.source_sample_count
            ):
                raise ValueError(
                    "sample_count must equal source_sample_count when drop_last=False."
                )
            if any(value is not None for value in dynamic_values):
                raise ValueError(
                    "Fixed cost-aware signatures cannot contain dynamic batch fields."
                )
            return
        if any(
            value is not None
            for value in (
                self.sample_count,
                self.per_device_batch_size,
                self.effective_planning_window,
                self.drop_last,
            )
        ):
            raise ValueError(
                "Dynamic signatures cannot contain fixed-cardinality fields."
            )
        for field_name in (
            "optimizer_step_count",
            "max_samples_per_microbatch",
            "max_padded_tokens",
        ):
            value = getattr(self, field_name)
            if value is None or int(value) <= 0:
                raise ValueError(f"{field_name} must be > 0 for dynamic batching.")
        if self.max_vision_patches is not None and int(self.max_vision_patches) <= 0:
            raise ValueError("max_vision_patches must be > 0 when set.")
        if (self.target_samples is None) == (self.target_supervised_tokens is None):
            raise ValueError(
                "Dynamic batch signatures require exactly one optimizer target."
            )
        for field_name in ("target_samples", "target_supervised_tokens"):
            value = getattr(self, field_name)
            if value is not None and int(value) <= 0:
                raise ValueError(f"{field_name} must be > 0 when set.")
        if self.rank_balance is None:
            raise ValueError("rank_balance must be set for dynamic batching.")
        if not str(self.planning_spec_fingerprint or "").strip():
            raise ValueError(
                "planning_spec_fingerprint must be set for dynamic batching."
            )

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(repr(self._payload()).encode("utf-8")).hexdigest()

    def _payload(self) -> tuple[object, ...]:
        fixed_payload = (
            self.planner_version,
            self.sample_plan_fingerprint,
            self.cost_fingerprint,
            self.source_sample_count,
            self.sample_count,
            self.per_device_batch_size,
            self.data_world_size,
            self.gradient_accumulation_steps,
            self.planning_window,
            self.effective_planning_window,
            self.seed,
            self.drop_last,
        )
        if self.strategy == "cost_aware":
            # Preserve the Phase-1 fingerprint so existing fixed-plan checkpoints
            # remain resumable after dynamic batching is added.
            return fixed_payload
        return (
            *fixed_payload,
            self.strategy,
            self.optimizer_step_count,
            self.max_samples_per_microbatch,
            self.max_padded_tokens,
            self.max_vision_patches,
            self.target_samples,
            self.target_supervised_tokens,
            self.rank_balance,
            self.planning_spec_fingerprint,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "planner_version": self.planner_version,
            "sample_plan_fingerprint": self.sample_plan_fingerprint,
            "cost_fingerprint": self.cost_fingerprint,
            "source_sample_count": self.source_sample_count,
            "sample_count": self.sample_count,
            "per_device_batch_size": self.per_device_batch_size,
            "data_world_size": self.data_world_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "planning_window": self.planning_window,
            "effective_planning_window": self.effective_planning_window,
            "seed": self.seed,
            "drop_last": self.drop_last,
            "strategy": self.strategy,
            "optimizer_step_count": self.optimizer_step_count,
            "max_samples_per_microbatch": self.max_samples_per_microbatch,
            "max_padded_tokens": self.max_padded_tokens,
            "max_vision_patches": self.max_vision_patches,
            "target_samples": self.target_samples,
            "target_supervised_tokens": self.target_supervised_tokens,
            "rank_balance": self.rank_balance,
            "planning_spec_fingerprint": self.planning_spec_fingerprint,
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> ShaftBatchPlanningSignature:
        expected_fingerprint = str(payload.get("fingerprint", "")).strip()
        strategy = str(payload.get("strategy", "cost_aware"))
        signature = cls(
            planner_version=str(payload.get("planner_version", "")),
            sample_plan_fingerprint=str(payload.get("sample_plan_fingerprint", "")),
            cost_fingerprint=str(payload.get("cost_fingerprint", "")),
            source_sample_count=int(
                payload.get("source_sample_count", payload.get("sample_count", 0))
            ),
            sample_count=_optional_int(payload, "sample_count"),
            per_device_batch_size=_optional_int(
                payload,
                "per_device_batch_size",
            ),
            data_world_size=int(payload.get("data_world_size", 0)),
            gradient_accumulation_steps=int(
                payload.get("gradient_accumulation_steps", 0)
            ),
            planning_window=int(payload.get("planning_window", 0)),
            effective_planning_window=_optional_int(
                payload,
                "effective_planning_window",
            ),
            seed=int(payload.get("seed", 0)),
            drop_last=(
                None
                if strategy == "dynamic_cost_aware"
                and payload.get("drop_last") is None
                else bool(payload.get("drop_last", False))
            ),
            strategy=strategy,
            optimizer_step_count=_optional_int(payload, "optimizer_step_count"),
            max_samples_per_microbatch=_optional_int(
                payload,
                "max_samples_per_microbatch",
            ),
            max_padded_tokens=_optional_int(payload, "max_padded_tokens"),
            max_vision_patches=_optional_int(payload, "max_vision_patches"),
            target_samples=_optional_int(payload, "target_samples"),
            target_supervised_tokens=_optional_int(
                payload,
                "target_supervised_tokens",
            ),
            rank_balance=(
                None
                if payload.get("rank_balance") is None
                else bool(payload["rank_balance"])
            ),
            planning_spec_fingerprint=(
                None
                if payload.get("planning_spec_fingerprint") is None
                else str(payload["planning_spec_fingerprint"])
            ),
        )
        if expected_fingerprint and expected_fingerprint != signature.fingerprint:
            raise ValueError("Batch planning signature fingerprint is corrupt or stale.")
        return signature

    @classmethod
    def from_spec(
        cls,
        spec: object,
        *,
        cost_fingerprint: str,
    ) -> ShaftBatchPlanningSignature:
        if isinstance(spec, ShaftFixedBatchPlanningSpec):
            return cls(
                planner_version=_BATCH_PLAN_VERSION,
                sample_plan_fingerprint=spec.sample_plan_fingerprint,
                cost_fingerprint=str(cost_fingerprint),
                source_sample_count=spec.source_sample_count,
                sample_count=spec.usable_sample_count,
                per_device_batch_size=spec.per_device_batch_size,
                data_world_size=spec.data_world_size,
                gradient_accumulation_steps=spec.gradient_accumulation_steps,
                planning_window=spec.planning_window,
                effective_planning_window=spec.effective_planning_window,
                seed=spec.seed,
                drop_last=spec.drop_last,
            )

        from .dynamic_batching import (
            SHAFT_DYNAMIC_BATCH_PLAN_VERSION,
            ShaftDynamicBatchPlanningSpec,
        )

        if isinstance(spec, ShaftDynamicBatchPlanningSpec):
            return cls(
                planner_version=SHAFT_DYNAMIC_BATCH_PLAN_VERSION,
                sample_plan_fingerprint=spec.sample_plan_fingerprint,
                cost_fingerprint=str(cost_fingerprint),
                source_sample_count=spec.source_sample_count,
                sample_count=None,
                per_device_batch_size=None,
                data_world_size=spec.data_world_size,
                gradient_accumulation_steps=spec.gradient_accumulation_steps,
                planning_window=spec.planning_window,
                effective_planning_window=None,
                seed=spec.seed,
                drop_last=None,
                strategy="dynamic_cost_aware",
                optimizer_step_count=spec.optimizer_step_count,
                max_samples_per_microbatch=spec.max_samples_per_microbatch,
                max_padded_tokens=spec.max_padded_tokens,
                max_vision_patches=spec.max_vision_patches,
                target_samples=spec.target_samples,
                target_supervised_tokens=spec.target_supervised_tokens,
                rank_balance=spec.rank_balance,
                planning_spec_fingerprint=spec.fingerprint,
            )
        raise TypeError(f"Unsupported batch planning spec: {type(spec).__name__}.")


class ShaftFixedBatchPlanner:
    """Build bounded, fixed-cardinality cost-aware batch plans.

    The planner changes only sample execution order. Local batch size and optimizer
    accumulation remain owned by the existing HF Trainer path.
    """

    def __init__(
        self,
        *,
        plan: ShaftSamplePlan,
        cost_provider: ShaftSampleCostProvider,
        spec: ShaftFixedBatchPlanningSpec,
    ) -> None:
        if spec.sample_plan_fingerprint != str(plan.fingerprint):
            raise ValueError(
                "Fixed batch planning spec does not belong to the supplied SamplePlan."
            )
        if spec.source_sample_count != len(plan):
            raise ValueError(
                "Fixed batch planning spec sample count does not match the SamplePlan."
            )
        self.plan = plan
        self.cost_provider = cost_provider
        self.spec = spec
        self.per_device_batch_size = spec.per_device_batch_size
        self.data_world_size = spec.data_world_size
        self.planning_window = spec.planning_window
        self.seed = spec.seed
        self.drop_last = spec.drop_last
        self.global_microstep_samples = spec.global_microstep_samples
        self.usable_sample_count = spec.usable_sample_count
        self.effective_planning_window = spec.effective_planning_window

    def __len__(self) -> int:
        return self.usable_sample_count

    def build_signature(self) -> ShaftBatchPlanningSignature:
        return ShaftBatchPlanningSignature.from_spec(
            self.spec,
            cost_fingerprint=str(self.cost_provider.fingerprint),
        )

    def iter_window_plans(self, *, plan_cycle: int = 0) -> Iterator[ShaftBatchPlan]:
        plan_cycle = int(plan_cycle)
        for window_start in range(
            0,
            self.usable_sample_count,
            self.effective_planning_window,
        ):
            window_stop = min(
                window_start + self.effective_planning_window,
                self.usable_sample_count,
            )
            refs = tuple(
                self.plan.ref_at(position, plan_cycle=plan_cycle)
                for position in range(window_start, window_stop)
            )
            yield self._build_window_plan(
                refs,
                window_start=window_start,
                window_stop=window_stop,
                plan_cycle=plan_cycle,
            )

    def _build_window_plan(
        self,
        refs: tuple[ShaftSampleRef, ...],
        *,
        window_start: int,
        window_stop: int,
        plan_cycle: int,
    ) -> ShaftBatchPlan:
        entries = tuple((sample_ref, self.cost_provider(sample_ref)) for sample_ref in refs)
        baseline_batches = self._chunk_entries(entries)
        ordered_entries = tuple(
            sorted(
                entries,
                key=lambda item: self._sample_sort_key(
                    item[0],
                    item[1],
                    plan_cycle=plan_cycle,
                    window_start=window_start,
                ),
            )
        )
        local_batches = list(self._chunk_entries(ordered_entries))
        max_padded_tokens = max(batch.padded_llm_tokens for batch in local_batches)
        max_vision_patches = max(batch.vision_patches for batch in local_batches)
        local_batches.sort(
            key=lambda batch: (
                self._balance_score(
                    batch,
                    max_padded_tokens=max_padded_tokens,
                    max_vision_patches=max_vision_patches,
                ),
                batch.padded_llm_tokens,
                batch.vision_patches,
                tuple(ref.context.draw_id for ref in batch.sample_refs),
            )
        )

        microsteps: list[ShaftGlobalMicroBatchPlan] = []
        for group_start in range(0, len(local_batches), self.data_world_size):
            group = local_batches[group_start : group_start + self.data_world_size]
            rotation = int(
                _splitmix64(
                    self.seed
                    ^ plan_cycle
                    ^ window_start
                    ^ group_start
                    ^ 0xA0761D6478BD642F
                )
                % self.data_world_size
            )
            rotated = group[rotation:] + group[:rotation]
            microsteps.append(
                ShaftGlobalMicroBatchPlan(rank_microbatches=tuple(rotated))
            )

        permutation_seed = _splitmix64(
            self.seed ^ plan_cycle ^ window_start ^ 0xE7037ED1A0B428DB
        )
        microsteps = [
            microsteps[
                _affine_permute(
                    position,
                    len(microsteps),
                    seed=permutation_seed,
                )
            ]
            for position in range(len(microsteps))
        ]

        rank_skews = [
            self._microstep_cost_skew(
                microstep,
                max_padded_tokens=max_padded_tokens,
                max_vision_patches=max_vision_patches,
            )
            for microstep in microsteps
        ]
        useful_llm_tokens = sum(cost.llm_tokens for _, cost in entries)
        loss_weight_sum = (
            sum(float(cost.loss_weight_sum) for _, cost in entries)
            if all(cost.loss_weight_sum is not None for _, cost in entries)
            else None
        )
        stats = ShaftBatchPlanStats(
            sample_count=len(entries),
            local_batch_count=len(local_batches),
            global_microstep_count=len(microsteps),
            useful_llm_tokens=useful_llm_tokens,
            baseline_padded_llm_tokens=sum(
                batch.padded_llm_tokens for batch in baseline_batches
            ),
            planned_padded_llm_tokens=sum(
                batch.padded_llm_tokens for batch in local_batches
            ),
            supervised_tokens=sum(cost.supervised_tokens for _, cost in entries),
            loss_weight_sum=loss_weight_sum,
            vision_patches=sum(cost.vision_patches for _, cost in entries),
            inexact_sample_count=sum(not cost.exact for _, cost in entries),
            average_rank_cost_skew=(sum(rank_skews) / len(rank_skews)),
            max_rank_cost_skew=max(rank_skews),
        )
        fingerprint_payload = (
            _BATCH_PLAN_VERSION,
            str(getattr(self.cost_provider, "fingerprint", "")),
            self.per_device_batch_size,
            self.data_world_size,
            self.effective_planning_window,
            self.seed,
            plan_cycle,
            window_start,
            window_stop,
            tuple(
                (
                    ref.dataset_name,
                    ref.row_index,
                    ref.context.draw_id,
                    cost,
                )
                for ref, cost in entries
            ),
            tuple(
                tuple(
                    tuple(ref.context.draw_id for ref in batch.sample_refs)
                    for batch in microstep.rank_microbatches
                )
                for microstep in microsteps
            ),
        )
        fingerprint = hashlib.sha256(
            repr(fingerprint_payload).encode("utf-8")
        ).hexdigest()
        return ShaftBatchPlan(
            window_start=window_start,
            window_stop=window_stop,
            plan_cycle=plan_cycle,
            microsteps=tuple(microsteps),
            stats=stats,
            fingerprint=fingerprint,
        )

    def _chunk_entries(
        self,
        entries: tuple[tuple[ShaftSampleRef, ShaftSampleCost], ...],
    ) -> tuple[ShaftLocalMicroBatchPlan, ...]:
        batches: list[ShaftLocalMicroBatchPlan] = []
        for start in range(0, len(entries), self.per_device_batch_size):
            chunk = entries[start : start + self.per_device_batch_size]
            if len(chunk) != self.per_device_batch_size:
                raise ValueError("A planning window produced an incomplete local microbatch.")
            batches.append(
                ShaftLocalMicroBatchPlan(
                    sample_refs=tuple(item[0] for item in chunk),
                    sample_costs=tuple(item[1] for item in chunk),
                )
            )
        return tuple(batches)

    def _sample_sort_key(
        self,
        sample_ref: ShaftSampleRef,
        cost: ShaftSampleCost,
        *,
        plan_cycle: int,
        window_start: int,
    ) -> tuple[int, int, int, int, int]:
        tie_breaker = int(
            _splitmix64(
                self.seed
                ^ plan_cycle
                ^ window_start
                ^ sample_ref.context.draw_id
            )
        )
        return (
            _log2_bucket(cost.llm_tokens),
            _log2_bucket(cost.vision_patches),
            cost.llm_tokens,
            cost.vision_patches,
            tie_breaker,
        )

    @staticmethod
    def _balance_score(
        batch: ShaftLocalMicroBatchPlan,
        *,
        max_padded_tokens: int,
        max_vision_patches: int,
    ) -> float:
        llm_score = batch.padded_llm_tokens / max(max_padded_tokens, 1)
        vision_score = batch.vision_patches / max(max_vision_patches, 1)
        return float(llm_score + vision_score)

    def _microstep_cost_skew(
        self,
        microstep: ShaftGlobalMicroBatchPlan,
        *,
        max_padded_tokens: int,
        max_vision_patches: int,
    ) -> float:
        scores = [
            self._balance_score(
                batch,
                max_padded_tokens=max_padded_tokens,
                max_vision_patches=max_vision_patches,
            )
            for batch in microstep.rank_microbatches
        ]
        if not scores:
            return 0.0
        skew = max(scores) - min(scores)
        if not math.isfinite(skew):
            raise ValueError("Batch planner produced a non-finite rank cost skew.")
        return float(skew)
