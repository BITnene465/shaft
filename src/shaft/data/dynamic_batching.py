from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import combinations
import json
from typing import Any, Iterator

from shaft.config.data import SHAFT_BATCH_RESOURCE_NAMES
from shaft.utils.contract_schema import (
    json_int,
    json_int_value,
    json_list,
    json_mapping,
    json_optional_int,
    json_string,
    require_exact_keys,
    require_json_mapping,
)

from .batching import (
    ShaftBatchContext,
    ShaftGreedySequencePacker,
    ShaftLengthBatchGrouping,
    ShaftLocalMicroBatchPlan,
    ShaftLogicalSegmentPlan,
    ShaftPhysicalPackPlan,
    resolve_local_pack_count_bounds,
)
from .cost import ShaftSampleCost, ShaftSampleCostProvider
from .mixing import ShaftSampleRef, ShaftSampleSchedule, _splitmix64
from .planned import ShaftPlannedSampleRef


SHAFT_BATCH_PLANNING_VERSION = "shaft-batch-planning-v4"
_FULL_PARTITION_SEARCH_NODE_LIMIT = 100_000


class ShaftPartitionSearchBudgetExceeded(RuntimeError):
    """A full-cap partition could not be decided within the safety budget."""


def _mapping_fingerprint(payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ShaftBatchPlanningSpec:
    """Immutable, duration-independent planned-batching contract."""

    data_world_size: int
    buffer_size: int
    per_device_microbatch_size: int
    max_tokens_per_microbatch: int
    resource_budgets: tuple[tuple[str, int], ...]
    seed: int
    sample_schedule_fingerprint: str
    cost_fingerprint: str
    cardinality: str = "fixed"
    grouping: str = "bounded_cost"
    packing: str = "none"
    layout: str = "padded"
    max_sequence_length: int | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "data_world_size",
            "buffer_size",
            "per_device_microbatch_size",
            "max_tokens_per_microbatch",
        ):
            if int(getattr(self, field_name)) <= 0:
                raise ValueError(f"{field_name} must be > 0.")
        if self.grouping not in {"length", "bounded_cost"}:
            raise ValueError(
                f"Unsupported planned batching grouping: {self.grouping!r}."
            )
        if self.cardinality not in {"fixed", "token_budget"}:
            raise ValueError(
                f"Unsupported planned batching cardinality: {self.cardinality!r}."
            )
        if self.packing not in {"none", "greedy"}:
            raise ValueError(f"Unsupported planned batching packing: {self.packing!r}.")
        if self.layout not in {"padded", "varlen"}:
            raise ValueError(f"Unsupported planned batching layout: {self.layout!r}.")
        if self.grouping == "bounded_cost" and (
            self.packing != "none" or self.layout != "padded"
        ):
            raise ValueError(
                "bounded_cost grouping currently requires packing='none' and "
                "layout='padded'."
            )
        if self.grouping == "length" and self.cardinality != "fixed":
            raise ValueError("length grouping currently requires fixed cardinality.")
        if self.packing == "greedy" and self.layout != "varlen":
            raise ValueError("greedy packing requires layout='varlen'.")
        if self.packing == "greedy" and (
            self.max_sequence_length is None or int(self.max_sequence_length) <= 0
        ):
            raise ValueError("greedy packing requires max_sequence_length > 0.")
        if self.cardinality == "token_budget" and self.grouping != "bounded_cost":
            raise ValueError("token_budget cardinality requires bounded_cost grouping.")
        required_samples = int(self.data_world_size) * int(
            self.local_pack_count_bounds[0]
        )
        if int(self.buffer_size) < required_samples:
            raise ValueError(
                "buffer_size must hold one complete minimum global microbatch: "
                f"buffer_size={self.buffer_size}, required_samples={required_samples}."
            )
        names: list[str] = []
        for raw_name, raw_value in self.resource_budgets:
            name = str(raw_name).strip().lower()
            if not name:
                raise ValueError("resource_budgets contains an empty resource name.")
            if name != raw_name:
                raise ValueError("resource_budgets names must be normalized lowercase strings.")
            if name not in SHAFT_BATCH_RESOURCE_NAMES:
                raise ValueError(
                    f"Unsupported planned batching resource {name!r}; "
                    f"expected one of {SHAFT_BATCH_RESOURCE_NAMES}."
                )
            if int(raw_value) <= 0:
                raise ValueError(f"resource_budgets.{name} must be > 0.")
            names.append(name)
        if names != sorted(names) or len(names) != len(set(names)):
            raise ValueError("resource_budgets must be sorted and contain unique names.")
        if not str(self.sample_schedule_fingerprint).strip():
            raise ValueError("sample_schedule_fingerprint must not be empty.")
        if not str(self.cost_fingerprint).strip():
            raise ValueError("cost_fingerprint must not be empty.")

    @property
    def fingerprint(self) -> str:
        payload = (
            SHAFT_BATCH_PLANNING_VERSION,
            str(self.grouping),
            str(self.packing),
            str(self.layout),
            self.max_sequence_length,
            int(self.data_world_size),
            int(self.buffer_size),
            str(self.cardinality),
            int(self.per_device_microbatch_size),
            int(self.max_tokens_per_microbatch),
            tuple((str(name), int(value)) for name, value in self.resource_budgets),
            int(self.seed),
            str(self.sample_schedule_fingerprint),
            str(self.cost_fingerprint),
        )
        return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": SHAFT_BATCH_PLANNING_VERSION,
            "grouping": str(self.grouping),
            "packing": str(self.packing),
            "layout": str(self.layout),
            "max_sequence_length": self.max_sequence_length,
            "data_world_size": int(self.data_world_size),
            "buffer_size": int(self.buffer_size),
            "cardinality": str(self.cardinality),
            "per_device_microbatch_size": int(self.per_device_microbatch_size),
            "max_tokens_per_microbatch": int(self.max_tokens_per_microbatch),
            "resource_budgets": {
                str(name): int(value) for name, value in self.resource_budgets
            },
            "seed": int(self.seed),
            "sample_schedule_fingerprint": str(self.sample_schedule_fingerprint),
            "cost_fingerprint": str(self.cost_fingerprint),
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ShaftBatchPlanningSpec":
        role = "Planned batching spec"
        payload = require_json_mapping(payload, role=role)
        version = payload.get("version", "")
        if type(version) is not str:
            raise TypeError(f"{role}.version must be a JSON string.")
        if version != SHAFT_BATCH_PLANNING_VERSION:
            raise ValueError(
                "Unsupported planned batching spec version: "
                f"{version!r}; expected {SHAFT_BATCH_PLANNING_VERSION!r}."
            )
        require_exact_keys(
            payload,
            role=role,
            expected=frozenset(
                {
                    "version",
                    "grouping",
                    "packing",
                    "layout",
                    "max_sequence_length",
                    "data_world_size",
                    "buffer_size",
                    "cardinality",
                    "per_device_microbatch_size",
                    "max_tokens_per_microbatch",
                    "resource_budgets",
                    "seed",
                    "sample_schedule_fingerprint",
                    "cost_fingerprint",
                    "fingerprint",
                }
            ),
        )
        raw_resource_budgets = json_mapping(payload, "resource_budgets", role=role)
        spec = cls(
            data_world_size=json_int(payload, "data_world_size", role=role),
            buffer_size=json_int(payload, "buffer_size", role=role),
            cardinality=json_string(payload, "cardinality", role=role),
            per_device_microbatch_size=json_int(
                payload,
                "per_device_microbatch_size",
                role=role,
            ),
            max_tokens_per_microbatch=json_int(
                payload,
                "max_tokens_per_microbatch",
                role=role,
            ),
            resource_budgets=tuple(
                sorted(
                    (
                        name,
                        json_int_value(
                            value,
                            role=f"{role}.resource_budgets.{name}",
                        ),
                    )
                    for name, value in raw_resource_budgets.items()
                )
            ),
            seed=json_int(payload, "seed", role=role),
            sample_schedule_fingerprint=json_string(
                payload,
                "sample_schedule_fingerprint",
                role=role,
            ),
            cost_fingerprint=json_string(payload, "cost_fingerprint", role=role),
            grouping=json_string(payload, "grouping", role=role),
            packing=json_string(payload, "packing", role=role),
            layout=json_string(payload, "layout", role=role),
            max_sequence_length=json_optional_int(
                payload,
                "max_sequence_length",
                role=role,
            ),
        )
        serialized = json_string(payload, "fingerprint", role=role)
        if serialized != spec.fingerprint:
            raise ValueError("Planned batching spec fingerprint does not match its payload.")
        return spec

    def to_init_dict(self) -> dict[str, Any]:
        """Return canonical constructor fields without serialized metadata."""

        payload = self.to_dict()
        payload.pop("version", None)
        payload.pop("fingerprint", None)
        payload["resource_budgets"] = tuple(
            sorted(
                (str(name), int(value))
                for name, value in payload["resource_budgets"].items()
            )
        )
        return payload

    def resource_budget(self, name: str) -> int | None:
        normalized = str(name).strip().lower()
        return dict(self.resource_budgets).get(normalized)

    @property
    def local_pack_count_bounds(self) -> tuple[int, int]:
        return resolve_local_pack_count_bounds(
            self.cardinality,
            self.per_device_microbatch_size,
        )


@dataclass(frozen=True, slots=True)
class ShaftBufferedSample:
    sample_ref: ShaftSampleRef
    cost: ShaftSampleCost

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.sample_ref.dataset_name,
            "row_index": int(self.sample_ref.row_index),
            "context": self.sample_ref.context.to_dict(),
            "cost": {
                "llm_tokens": int(self.cost.llm_tokens),
                "supervised_tokens": int(self.cost.supervised_tokens),
                "vision_patches": int(self.cost.vision_patches),
                "loss_weight_sum": self.cost.loss_weight_sum,
                "exact": bool(self.cost.exact),
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ShaftBufferedSample":
        segment = ShaftLogicalSegmentPlan.from_dict(payload)
        return cls(
            sample_ref=segment.sample_ref,
            cost=segment.cost,
        )


@dataclass(frozen=True, slots=True)
class ShaftBatchPlanningState:
    """Checkpointable state at a completed global-microstep boundary."""

    contract_fingerprint: str
    global_microstep: int = 0
    next_draw_id: int = 0
    buffer: tuple[ShaftBufferedSample, ...] = ()
    emitted_samples: int = 0
    emitted_physical_packs: int | None = None
    emitted_llm_tokens: int = 0
    emitted_supervised_tokens: int = 0
    emitted_vision_patches: int = 0

    def __post_init__(self) -> None:
        if not str(self.contract_fingerprint).strip():
            raise ValueError("contract_fingerprint must not be empty.")
        if self.emitted_physical_packs is None:
            object.__setattr__(
                self,
                "emitted_physical_packs",
                int(self.emitted_samples),
            )
        for field_name in (
            "global_microstep",
            "next_draw_id",
            "emitted_samples",
            "emitted_physical_packs",
            "emitted_llm_tokens",
            "emitted_supervised_tokens",
            "emitted_vision_patches",
        ):
            if int(getattr(self, field_name)) < 0:
                raise ValueError(f"{field_name} must be >= 0.")
        draw_ids = [entry.sample_ref.context.draw_id for entry in self.buffer]
        if len(draw_ids) != len(set(draw_ids)):
            raise ValueError("Planned batching state buffer contains duplicate draws.")
        if draw_ids != sorted(draw_ids):
            raise ValueError("Planned batching state buffer must preserve FIFO draw order.")
        if draw_ids and max(draw_ids) >= int(self.next_draw_id):
            raise ValueError("Buffered draw ids must be lower than next_draw_id.")
        if int(self.next_draw_id) != int(self.emitted_samples) + len(self.buffer):
            raise ValueError(
                "Planned batching state violates draw conservation: "
                "next_draw_id must equal emitted_samples + len(buffer)."
            )

    def validate_against_spec(self, spec: ShaftBatchPlanningSpec) -> None:
        if self.contract_fingerprint != spec.fingerprint:
            raise ValueError(
                "Planned batching checkpoint contract differs from the active runtime."
            )
        if len(self.buffer) > int(spec.buffer_size):
            raise ValueError("Checkpoint buffer exceeds the configured buffer_size.")
        minimum_local, maximum_local = spec.local_pack_count_bounds
        completed_slots = int(self.global_microstep) * int(spec.data_world_size)
        minimum_emitted = completed_slots * minimum_local
        maximum_emitted = completed_slots * maximum_local
        physical_packs = int(self.emitted_physical_packs or 0)
        if not minimum_emitted <= physical_packs <= maximum_emitted:
            raise ValueError(
            "Planned batching state emitted logical segments/physical packs are "
                "outside its cardinality bounds: "
                f"actual={physical_packs}, "
                f"expected_range=({minimum_emitted}, {maximum_emitted})."
            )
        if spec.packing == "none" and int(self.emitted_samples) != physical_packs:
            raise ValueError(
                "Packing-none state requires emitted logical segments to equal "
                "emitted physical packs."
            )
        if spec.packing == "greedy" and int(self.emitted_samples) < physical_packs:
            raise ValueError(
                "Greedy packing state cannot emit fewer logical segments than "
                "physical packs."
            )
        if (
            self.buffer
            and int(self.buffer[0].sample_ref.context.draw_id)
            < int(self.global_microstep)
        ):
            raise ValueError(
                "Planned batching state retains an oldest draw that must already "
                "have been emitted."
            )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "version": SHAFT_BATCH_PLANNING_VERSION,
            "contract_fingerprint": str(self.contract_fingerprint),
            "global_microstep": int(self.global_microstep),
            "next_draw_id": int(self.next_draw_id),
            "buffer": [entry.to_dict() for entry in self.buffer],
            "emitted_samples": int(self.emitted_samples),
            "emitted_physical_packs": int(self.emitted_physical_packs or 0),
            "emitted_llm_tokens": int(self.emitted_llm_tokens),
            "emitted_supervised_tokens": int(self.emitted_supervised_tokens),
            "emitted_vision_patches": int(self.emitted_vision_patches),
        }
        payload["fingerprint"] = _mapping_fingerprint(payload)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ShaftBatchPlanningState":
        role = "Planned batching state"
        payload = require_json_mapping(payload, role=role)
        version = payload.get("version", "")
        if type(version) is not str:
            raise TypeError(f"{role}.version must be a JSON string.")
        if version != SHAFT_BATCH_PLANNING_VERSION:
            raise ValueError(
                "Unsupported planned batching state version: "
                f"{version!r}; expected {SHAFT_BATCH_PLANNING_VERSION!r}."
            )
        require_exact_keys(
            payload,
            role=role,
            expected=frozenset(
                {
                    "version",
                    "contract_fingerprint",
                    "global_microstep",
                    "next_draw_id",
                    "buffer",
                    "emitted_samples",
                    "emitted_physical_packs",
                    "emitted_llm_tokens",
                    "emitted_supervised_tokens",
                    "emitted_vision_patches",
                    "fingerprint",
                }
            ),
        )
        fingerprint = json_string(payload, "fingerprint", role=role)
        buffer_payload = json_list(payload, "buffer", role=role)
        buffer = tuple(ShaftBufferedSample.from_dict(item) for item in buffer_payload)
        unsigned = dict(payload)
        unsigned.pop("fingerprint", None)
        expected = _mapping_fingerprint(unsigned)
        if fingerprint != expected:
            raise ValueError("Planned batching state fingerprint does not match its payload.")
        return cls(
            contract_fingerprint=json_string(payload, "contract_fingerprint", role=role),
            global_microstep=json_int(payload, "global_microstep", role=role),
            next_draw_id=json_int(payload, "next_draw_id", role=role),
            buffer=buffer,
            emitted_samples=json_int(payload, "emitted_samples", role=role),
            emitted_physical_packs=json_int(
                payload,
                "emitted_physical_packs",
                role=role,
            ),
            emitted_llm_tokens=json_int(payload, "emitted_llm_tokens", role=role),
            emitted_supervised_tokens=json_int(
                payload,
                "emitted_supervised_tokens",
                role=role,
            ),
            emitted_vision_patches=json_int(
                payload,
                "emitted_vision_patches",
                role=role,
            ),
        )

    @property
    def emitted_logical_segments(self) -> int:
        return int(self.emitted_samples)


@dataclass(frozen=True, slots=True)
class ShaftBatchMicrobatchStats:
    sample_count: int
    physical_pack_count: int
    useful_llm_tokens: int
    padded_llm_tokens: int
    supervised_tokens: int
    vision_patches: int
    min_local_pack_count: int
    max_local_pack_count: int
    max_local_padded_tokens: int
    max_local_vision_patches: int
    max_rank_cost_skew: float

    @property
    def logical_segment_count(self) -> int:
        return int(self.sample_count)

    @property
    def padding_ratio(self) -> float:
        if self.padded_llm_tokens <= 0:
            return 0.0
        return 1.0 - self.useful_llm_tokens / self.padded_llm_tokens


@dataclass(frozen=True, slots=True)
class ShaftBatchMicrobatchPlan:
    global_microstep: int
    rank_microbatches: tuple[ShaftLocalMicroBatchPlan, ...]
    stats: ShaftBatchMicrobatchStats
    state_after: ShaftBatchPlanningState

    @property
    def fingerprint(self) -> str:
        payload = (
            SHAFT_BATCH_PLANNING_VERSION,
            int(self.global_microstep),
            tuple(
                tuple(
                    tuple(
                        (
                            segment.sample_ref.dataset_name,
                            int(segment.sample_ref.row_index),
                            segment.sample_ref.context.to_dict(),
                            segment.cost,
                        )
                        for segment in pack.segments
                    )
                    for pack in batch.packs
                )
                for batch in self.rank_microbatches
            ),
            self.state_after.to_dict()["fingerprint"],
        )
        return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()

    def planned_refs_for_rank(
        self,
        rank: int,
    ) -> tuple[ShaftPlannedSampleRef, ...]:
        rank_index = int(rank)
        if rank_index < 0 or rank_index >= len(self.rank_microbatches):
            raise IndexError(rank_index)
        local_batch = self.rank_microbatches[rank_index]
        plan_fingerprint = self.fingerprint
        planned: list[ShaftPlannedSampleRef] = []
        for pack_index, pack in enumerate(local_batch.packs):
            segment_count = len(pack.segments)
            for segment_index, segment in enumerate(pack.segments):
                planned.append(
                    ShaftPlannedSampleRef(
                        sample_ref=segment.sample_ref,
                        batch_context=ShaftBatchContext(
                            global_microstep=int(self.global_microstep),
                            plan_fingerprint=plan_fingerprint,
                            local_batch_id=rank_index,
                            pack_index=pack_index,
                            segment_index=segment_index,
                            pack_segment_count=segment_count,
                        ),
                    )
                )
        return tuple(planned)


class ShaftBatchPlanner:
    """Group one fixed or budget-cardinality microstep from bounded lookahead."""

    def __init__(
        self,
        *,
        schedule: ShaftSampleSchedule,
        cost_provider: ShaftSampleCostProvider,
        spec: ShaftBatchPlanningSpec,
        state: ShaftBatchPlanningState | None = None,
    ) -> None:
        if str(schedule.fingerprint) != str(spec.sample_schedule_fingerprint):
            raise ValueError("Planned batching spec does not belong to this sample schedule.")
        provider_fingerprint = str(getattr(cost_provider, "fingerprint", ""))
        if provider_fingerprint != str(spec.cost_fingerprint):
            raise ValueError("Planned batching spec cost fingerprint does not match provider.")
        if state is None:
            state = ShaftBatchPlanningState(
                contract_fingerprint=spec.fingerprint,
            )
        state.validate_against_spec(spec)
        self.schedule = schedule
        self.cost_provider = cost_provider
        self.spec = spec
        for entry in state.buffer:
            expected_ref = schedule.ref_at(entry.sample_ref.context.draw_id)
            if expected_ref != entry.sample_ref:
                raise ValueError(
                    "Planned batching checkpoint buffer does not match the sample schedule."
                )
            self._validate_sample_cost(entry.sample_ref, entry.cost)
        self.state = state

    def next_global_microbatch(self) -> ShaftBatchMicrobatchPlan:
        buffer = list(self.state.buffer)
        next_draw_id = int(self.state.next_draw_id)
        while len(buffer) < int(self.spec.buffer_size):
            sample_ref = self.schedule.ref_at(next_draw_id)
            cost = self.cost_provider(sample_ref)
            self._validate_sample_cost(sample_ref, cost)
            buffer.append(ShaftBufferedSample(sample_ref=sample_ref, cost=cost))
            next_draw_id += 1

        if self.spec.grouping == "length":
            return self._next_length_microbatch(
                buffer=buffer,
                next_draw_id=next_draw_id,
            )

        world_size = int(self.spec.data_world_size)
        local_limit = int(self.spec.per_device_microbatch_size)
        minimum_count = world_size * int(self.spec.local_pack_count_bounds[0])
        target_count = world_size * local_limit
        if len(buffer) < minimum_count:
            raise RuntimeError(
                "Planned batching buffer cannot hold one minimum global microbatch."
            )

        # The oldest draw is mandatory, which bounds waiting independently of cost.
        # The remaining rank anchors are the closest-cost entries in the lookahead,
        # so batch-size-one DDP still groups similar work into the same microstep.
        oldest = buffer[0]
        seed_candidates = sorted(
            buffer[1:],
            key=lambda entry: (
                self._sample_distance(oldest, entry),
                int(entry.sample_ref.context.draw_id),
            ),
        )
        anchors = [oldest, *seed_candidates[: world_size - 1]]
        bins: list[list[ShaftBufferedSample]] = [[entry] for entry in anchors]
        selected_draw_ids = {
            entry.sample_ref.context.draw_id for entry in anchors
        }
        candidates = sorted(
            (
                entry
                for entry in buffer
                if entry.sample_ref.context.draw_id not in selected_draw_ids
            ),
            key=lambda entry: (
                self._sample_load(entry.cost),
                _splitmix64(
                    int(self.spec.seed)
                    ^ int(entry.sample_ref.context.draw_id)
                    ^ int(self.state.global_microstep)
                ),
            ),
            reverse=True,
        )
        for entry in candidates:
            choices: list[tuple[tuple[float, float, int, int], int]] = []
            for bin_index, batch in enumerate(bins):
                if not self._fits(batch, entry):
                    continue
                projected = [*batch, entry]
                choices.append((self._placement_score(projected, bin_index), bin_index))
            if not choices:
                continue
            _, selected_bin = min(choices)
            bins[selected_bin].append(entry)
            selected_draw_ids.add(entry.sample_ref.context.draw_id)
            if len(selected_draw_ids) == target_count:
                break

        incomplete = [
            index for index, batch in enumerate(bins) if len(batch) != local_limit
        ]
        fallback: list[list[ShaftBufferedSample]] | None = None
        try_full_partition = bool(
            incomplete
            and len(buffer) >= target_count
            and int(oldest.cost.llm_tokens) * local_limit
            <= int(self.spec.max_tokens_per_microbatch)
        )
        if try_full_partition:
            for resource_name, budget in self.spec.resource_budgets:
                companion_values = sorted(
                    self._sample_resource(entry.cost, resource_name)
                    for entry in buffer[1:]
                )
                minimum_total = self._sample_resource(
                    oldest.cost,
                    resource_name,
                ) + sum(companion_values[: local_limit - 1])
                if minimum_total > int(budget):
                    try_full_partition = False
                    break
        if try_full_partition:
            try:
                fallback = self._find_full_partition(buffer)
            except ShaftPartitionSearchBudgetExceeded:
                if self.spec.cardinality == "fixed":
                    raise
        if fallback is not None:
            bins = fallback
            selected_draw_ids = {
                entry.sample_ref.context.draw_id
                for batch in bins
                for entry in batch
            }
        elif self.spec.cardinality == "fixed" and incomplete:
            raise ValueError(
                "Bounded fixed-cardinality planner could not fill every rank "
                "batch within the configured token/resource budgets; "
                f"incomplete_bins={incomplete}, local_batch={local_limit}, "
                f"buffer_size={self.spec.buffer_size}. Increase the budget/buffer "
                "or reduce train.per_device_train_batch_size."
            )

        local_batches = tuple(self._build_local_batch(batch) for batch in bins)
        remaining = tuple(
            entry
            for entry in buffer
            if entry.sample_ref.context.draw_id not in selected_draw_ids
        )
        stats = self._build_stats(local_batches)
        state_after = ShaftBatchPlanningState(
            contract_fingerprint=self.spec.fingerprint,
            global_microstep=int(self.state.global_microstep) + 1,
            next_draw_id=next_draw_id,
            buffer=remaining,
            emitted_samples=int(self.state.emitted_samples) + stats.sample_count,
            emitted_physical_packs=(
                int(self.state.emitted_physical_packs or 0)
                + stats.physical_pack_count
            ),
            emitted_llm_tokens=(
                int(self.state.emitted_llm_tokens) + stats.useful_llm_tokens
            ),
            emitted_supervised_tokens=(
                int(self.state.emitted_supervised_tokens) + stats.supervised_tokens
            ),
            emitted_vision_patches=(
                int(self.state.emitted_vision_patches) + stats.vision_patches
            ),
        )
        plan = ShaftBatchMicrobatchPlan(
            global_microstep=int(self.state.global_microstep),
            rank_microbatches=local_batches,
            stats=stats,
            state_after=state_after,
        )
        self.state = state_after
        return plan

    def _next_length_microbatch(
        self,
        *,
        buffer: list[ShaftBufferedSample],
        next_draw_id: int,
    ) -> ShaftBatchMicrobatchPlan:
        world_size = int(self.spec.data_world_size)
        local_pack_count = int(self.spec.per_device_microbatch_size)
        global_pack_count = world_size * local_pack_count
        if len(buffer) < global_pack_count:
            raise RuntimeError(
                "Length-grouping buffer cannot seed one complete global microbatch."
            )

        logical_segments = tuple(
            ShaftLogicalSegmentPlan(sample_ref=entry.sample_ref, cost=entry.cost)
            for entry in buffer
        )
        ordered = ShaftLengthBatchGrouping.build(
            logical_segments,
            seed=int(self.spec.seed),
            global_microstep=int(self.state.global_microstep),
        )
        oldest_draw_id = int(buffer[0].sample_ref.context.draw_id)
        if self.spec.packing == "none":
            oldest_position = next(
                index
                for index, segment in enumerate(ordered)
                if segment.draw_id == oldest_draw_id
            )
            start = min(
                (oldest_position // global_pack_count) * global_pack_count,
                len(ordered) - global_pack_count,
            )
            selected = ordered[start : start + global_pack_count]
            packs = tuple(
                ShaftPhysicalPackPlan(segments=(segment,)) for segment in selected
            )
        else:
            max_length = self.spec.max_sequence_length
            if max_length is None:
                raise RuntimeError("Greedy packing has no resolved max_sequence_length.")
            packs = ShaftGreedySequencePacker.build(
                ordered,
                physical_pack_count=global_pack_count,
                max_length=int(max_length),
                required_draw_id=oldest_draw_id,
                placement_feasible=self._packs_can_be_assigned,
            )

        assigned = self._assign_packs_to_ranks(packs)
        if assigned is None:
            raise ValueError(
                "Length grouping could not assign physical packs to all ranks within "
                "the local token/resource guards."
            )
        selected_draw_ids = {
            segment.draw_id for pack in packs for segment in pack.segments
        }
        remaining = tuple(
            entry
            for entry in buffer
            if int(entry.sample_ref.context.draw_id) not in selected_draw_ids
        )
        stats = self._build_stats(assigned)
        state_after = ShaftBatchPlanningState(
            contract_fingerprint=self.spec.fingerprint,
            global_microstep=int(self.state.global_microstep) + 1,
            next_draw_id=int(next_draw_id),
            buffer=remaining,
            emitted_samples=int(self.state.emitted_samples) + stats.sample_count,
            emitted_physical_packs=(
                int(self.state.emitted_physical_packs or 0)
                + stats.physical_pack_count
            ),
            emitted_llm_tokens=(
                int(self.state.emitted_llm_tokens) + stats.useful_llm_tokens
            ),
            emitted_supervised_tokens=(
                int(self.state.emitted_supervised_tokens) + stats.supervised_tokens
            ),
            emitted_vision_patches=(
                int(self.state.emitted_vision_patches) + stats.vision_patches
            ),
        )
        plan = ShaftBatchMicrobatchPlan(
            global_microstep=int(self.state.global_microstep),
            rank_microbatches=assigned,
            stats=stats,
            state_after=state_after,
        )
        self.state = state_after
        return plan

    def _packs_can_be_assigned(
        self,
        packs: tuple[ShaftPhysicalPackPlan, ...],
    ) -> bool:
        return self._assign_packs_to_ranks(packs, allow_partial=True) is not None

    def _assign_packs_to_ranks(
        self,
        packs: tuple[ShaftPhysicalPackPlan, ...],
        *,
        allow_partial: bool = False,
    ) -> tuple[ShaftLocalMicroBatchPlan, ...] | None:
        world_size = int(self.spec.data_world_size)
        local_limit = int(self.spec.per_device_microbatch_size)
        total_capacity = world_size * local_limit
        if len(packs) > total_capacity:
            return None
        if not allow_partial and len(packs) != total_capacity:
            return None

        ordered = sorted(
            packs,
            key=lambda pack: (
                -self._pack_load(pack),
                min(segment.draw_id for segment in pack.segments),
            ),
        )
        rank_packs: list[list[ShaftPhysicalPackPlan]] = [
            [] for _ in range(world_size)
        ]
        rank_tokens = [0] * world_size
        rank_resources = [
            {name: 0 for name, _ in self.spec.resource_budgets}
            for _ in range(world_size)
        ]

        def place(index: int) -> bool:
            if index == len(ordered):
                return allow_partial or all(
                    len(items) == local_limit for items in rank_packs
                )
            pack = ordered[index]
            candidate_ranks = sorted(
                range(world_size),
                key=lambda rank: (
                    rank_tokens[rank],
                    len(rank_packs[rank]),
                    rank,
                ),
            )
            seen_states: set[tuple[Any, ...]] = set()
            for rank in candidate_ranks:
                if len(rank_packs[rank]) >= local_limit:
                    continue
                state_key = (
                    len(rank_packs[rank]),
                    rank_tokens[rank],
                    tuple(
                        rank_resources[rank][name]
                        for name, _ in self.spec.resource_budgets
                    ),
                )
                if state_key in seen_states:
                    continue
                seen_states.add(state_key)
                projected_tokens = rank_tokens[rank] + pack.useful_llm_tokens
                if projected_tokens > int(self.spec.max_tokens_per_microbatch):
                    continue
                projected_resources = {
                    name: rank_resources[rank][name] + pack.resource_total(name)
                    for name, _ in self.spec.resource_budgets
                }
                if any(
                    projected_resources[name] > int(budget)
                    for name, budget in self.spec.resource_budgets
                ):
                    continue
                rank_packs[rank].append(pack)
                rank_tokens[rank] = projected_tokens
                previous_resources = rank_resources[rank]
                rank_resources[rank] = projected_resources
                if place(index + 1):
                    return True
                rank_packs[rank].pop()
                rank_tokens[rank] -= pack.useful_llm_tokens
                rank_resources[rank] = previous_resources
            return False

        if not place(0):
            return None
        return tuple(
            ShaftLocalMicroBatchPlan(packs=tuple(items))
            for items in rank_packs
            if items
        )

    def _pack_load(self, pack: ShaftPhysicalPackPlan) -> float:
        load = pack.useful_llm_tokens / int(self.spec.max_tokens_per_microbatch)
        for name, budget in self.spec.resource_budgets:
            load += pack.resource_total(name) / int(budget)
        return load

    def iter_global_microbatches(self, count: int) -> Iterator[ShaftBatchMicrobatchPlan]:
        for _ in range(int(count)):
            yield self.next_global_microbatch()

    def _validate_sample_cost(
        self,
        sample_ref: ShaftSampleRef,
        cost: ShaftSampleCost,
    ) -> None:
        draw_id = int(sample_ref.context.draw_id)
        if not bool(cost.exact):
            raise ValueError(
                "Planned batching requires exact sample costs; "
                f"draw_id={draw_id} is inexact."
            )
        if self.spec.grouping == "length":
            if (
                self.spec.max_sequence_length is not None
                and int(cost.llm_tokens) > int(self.spec.max_sequence_length)
            ):
                raise ValueError(
                    "Length grouping encountered an oversize logical segment: "
                    f"draw_id={draw_id}, llm_tokens={cost.llm_tokens}, "
                    f"max_sequence_length={self.spec.max_sequence_length}."
                )
        else:
            required_cardinality = int(self.spec.local_pack_count_bounds[0])
            padded_single_cost = int(cost.llm_tokens) * required_cardinality
            if padded_single_cost > int(self.spec.max_tokens_per_microbatch):
                raise ValueError(
                    "Planned batching encountered an oversize sample: "
                    f"draw_id={draw_id}, llm_tokens={cost.llm_tokens}, "
                    f"minimum_local_batch={required_cardinality}, "
                    "required_padded_tokens="
                    f"{padded_single_cost}, max_tokens_per_microbatch="
                    f"{self.spec.max_tokens_per_microbatch}."
                )
        for resource_name, budget in self.spec.resource_budgets:
            value = self._sample_resource(cost, resource_name)
            if value > int(budget):
                raise ValueError(
                    "Planned batching encountered an oversize sample: "
                    f"draw_id={draw_id}, {resource_name}={value}, "
                    f"resource_budget={budget}."
                )

    def _fits(
        self,
        batch: list[ShaftBufferedSample],
        entry: ShaftBufferedSample,
    ) -> bool:
        projected = [*batch, entry]
        if len(projected) > int(self.spec.per_device_microbatch_size):
            return False
        padded_tokens = len(projected) * max(
            int(item.cost.llm_tokens) for item in projected
        )
        if padded_tokens > int(self.spec.max_tokens_per_microbatch):
            return False
        for resource_name, budget in self.spec.resource_budgets:
            resource_value = sum(
                self._sample_resource(item.cost, resource_name) for item in projected
            )
            if resource_value > int(budget):
                return False
        return True

    def _find_full_partition(
        self,
        buffer: list[ShaftBufferedSample],
    ) -> list[list[ShaftBufferedSample]] | None:
        """Find a full-cap partition when the fast grouping heuristic dead-ends.

        The oldest draw remains mandatory. Subsequent groups are ordered by their
        smallest draw id, which removes rank-permutation duplicates while retaining
        every feasible selection from the lookahead buffer.
        """

        world_size = int(self.spec.data_world_size)
        local_size = int(self.spec.per_device_microbatch_size)
        if local_size == 1:
            # Every observed sample already passed the per-sample hard guards.
            return [[entry] for entry in buffer[:world_size]]

        ordered = sorted(
            buffer,
            key=lambda entry: int(entry.sample_ref.context.draw_id),
        )
        visited_nodes = 0

        def search(
            remaining: tuple[ShaftBufferedSample, ...],
            groups: tuple[tuple[ShaftBufferedSample, ...], ...],
            *,
            force_first: bool,
        ) -> tuple[tuple[ShaftBufferedSample, ...], ...] | None:
            nonlocal visited_nodes
            if len(groups) == world_size:
                return groups
            groups_left = world_size - len(groups)
            required_entries = groups_left * local_size
            if len(remaining) < required_entries:
                return None

            anchor_indices = (0,) if force_first else range(
                0,
                len(remaining) - required_entries + 1,
            )
            for anchor_index in anchor_indices:
                anchor = remaining[anchor_index]
                tail = remaining[anchor_index + 1 :]
                groups_after = groups_left - 1
                minimum_tail = groups_after * local_size
                if len(tail) < (local_size - 1) + minimum_tail:
                    continue
                for companions in combinations(tail, local_size - 1):
                    visited_nodes += 1
                    if visited_nodes > _FULL_PARTITION_SEARCH_NODE_LIMIT:
                        raise ShaftPartitionSearchBudgetExceeded(
                            "Bounded full-partition search exceeded "
                            f"its deterministic node limit "
                            f"({_FULL_PARTITION_SEARCH_NODE_LIMIT}). Increase the "
                            "budget/buffer only after inspecting the cost distribution."
                        )
                    candidate = [anchor, *companions]
                    if not self._batch_fits(candidate):
                        continue
                    selected = {
                        entry.sample_ref.context.draw_id for entry in candidate
                    }
                    next_remaining = tuple(
                        entry
                        for entry in tail
                        if entry.sample_ref.context.draw_id not in selected
                    )
                    resolved = search(
                        next_remaining,
                        (*groups, tuple(candidate)),
                        force_first=False,
                    )
                    if resolved is not None:
                        return resolved
            return None

        resolved = search(tuple(ordered), (), force_first=True)
        if resolved is None:
            return None
        return [list(group) for group in resolved]

    def _batch_fits(self, batch: list[ShaftBufferedSample]) -> bool:
        if len(batch) != int(self.spec.per_device_microbatch_size):
            return False
        padded_tokens = len(batch) * max(
            int(item.cost.llm_tokens) for item in batch
        )
        if padded_tokens > int(self.spec.max_tokens_per_microbatch):
            return False
        return all(
            sum(
                self._sample_resource(item.cost, resource_name)
                for item in batch
            )
            <= int(budget)
            for resource_name, budget in self.spec.resource_budgets
        )

    def _placement_score(
        self,
        batch: list[ShaftBufferedSample],
        bin_index: int,
    ) -> tuple[float, float, int, int]:
        useful = sum(int(item.cost.llm_tokens) for item in batch)
        padded = len(batch) * max(int(item.cost.llm_tokens) for item in batch)
        text_fill = padded / int(self.spec.max_tokens_per_microbatch)
        resource_fill = sum(
            sum(self._sample_resource(item.cost, name) for item in batch) / budget
            for name, budget in self.spec.resource_budgets
        )
        padding_waste = (padded - useful) / int(self.spec.max_tokens_per_microbatch)
        # Rank stragglers dominate DDP step time. Prefer the least-loaded feasible
        # projected bin first; padding waste and stable index break ties.
        projected_load = text_fill + resource_fill
        return (projected_load, padding_waste, len(batch), int(bin_index))

    def _sample_distance(
        self,
        first: ShaftBufferedSample,
        second: ShaftBufferedSample,
    ) -> float:
        distance = abs(int(first.cost.llm_tokens) - int(second.cost.llm_tokens)) / int(
            self.spec.max_tokens_per_microbatch
        )
        for name, budget in self.spec.resource_budgets:
            distance += abs(
                self._sample_resource(first.cost, name)
                - self._sample_resource(second.cost, name)
            ) / int(budget)
        return distance

    def _sample_load(self, cost: ShaftSampleCost) -> float:
        load = int(cost.llm_tokens) / int(self.spec.max_tokens_per_microbatch)
        for name, budget in self.spec.resource_budgets:
            load += self._sample_resource(cost, name) / int(budget)
        return load

    @staticmethod
    def _sample_resource(cost: ShaftSampleCost, name: str) -> int:
        return cost.resource_value(name)

    @staticmethod
    def _build_local_batch(
        batch: list[ShaftBufferedSample],
    ) -> ShaftLocalMicroBatchPlan:
        ordered = sorted(
            batch,
            key=lambda entry: int(entry.sample_ref.context.draw_id),
        )
        return ShaftLocalMicroBatchPlan.from_segments(
            tuple(
                ShaftLogicalSegmentPlan(
                    sample_ref=entry.sample_ref,
                    cost=entry.cost,
                )
                for entry in ordered
            )
        )

    def _batch_load(self, batch: ShaftLocalMicroBatchPlan) -> float:
        materialized_tokens = (
            batch.useful_llm_tokens
            if self.spec.layout == "varlen"
            else batch.padded_llm_tokens
        )
        text = materialized_tokens / int(self.spec.max_tokens_per_microbatch)
        resource = sum(
            sum(self._sample_resource(cost, name) for cost in batch.sample_costs)
            / int(budget)
            for name, budget in self.spec.resource_budgets
        )
        return text + resource

    def _build_stats(
        self,
        batches: tuple[ShaftLocalMicroBatchPlan, ...],
    ) -> ShaftBatchMicrobatchStats:
        loads = [self._batch_load(batch) for batch in batches]
        mean_load = sum(loads) / len(loads)
        max_rank_cost_skew = (
            0.0
            if mean_load <= 0
            else max(abs(load - mean_load) / mean_load for load in loads)
        )
        return ShaftBatchMicrobatchStats(
            sample_count=sum(batch.logical_segment_count for batch in batches),
            physical_pack_count=sum(batch.physical_pack_count for batch in batches),
            useful_llm_tokens=sum(batch.useful_llm_tokens for batch in batches),
            padded_llm_tokens=sum(
                (
                    batch.useful_llm_tokens
                    if self.spec.layout == "varlen"
                    else batch.padded_llm_tokens
                )
                for batch in batches
            ),
            supervised_tokens=sum(batch.supervised_tokens for batch in batches),
            vision_patches=sum(batch.vision_patches for batch in batches),
            min_local_pack_count=min(batch.physical_pack_count for batch in batches),
            max_local_pack_count=max(batch.physical_pack_count for batch in batches),
            max_local_padded_tokens=max(
                (
                    batch.useful_llm_tokens
                    if self.spec.layout == "varlen"
                    else batch.padded_llm_tokens
                )
                for batch in batches
            ),
            max_local_vision_patches=max(batch.vision_patches for batch in batches),
            max_rank_cost_skew=max_rank_cost_skew,
        )
