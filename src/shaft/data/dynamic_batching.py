from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterator

from .batching import ShaftLocalMicroBatchPlan
from .cost import ShaftSampleCost, ShaftSampleCostProvider
from .mixing import ShaftSampleContext, ShaftSampleRef, ShaftSampleSchedule, _splitmix64


SHAFT_BOUNDED_BATCHING_VERSION = "shaft-bounded-cost-batching-v1"


def _mapping_fingerprint(payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _optional_int(payload: dict[str, Any], field_name: str) -> int | None:
    value = payload.get(field_name)
    return None if value is None else int(value)


@dataclass(frozen=True, slots=True)
class ShaftBoundedBatchingSpec:
    """Immutable, duration-independent bounded batching contract."""

    data_world_size: int
    buffer_size: int
    max_samples_per_microbatch: int
    max_padded_tokens: int
    max_vision_patches: int | None
    seed: int
    sample_schedule_fingerprint: str
    cost_fingerprint: str

    def __post_init__(self) -> None:
        for field_name in (
            "data_world_size",
            "buffer_size",
            "max_samples_per_microbatch",
            "max_padded_tokens",
        ):
            if int(getattr(self, field_name)) <= 0:
                raise ValueError(f"{field_name} must be > 0.")
        if int(self.buffer_size) < int(self.data_world_size):
            raise ValueError(
                "buffer_size must hold at least one anchor per data rank: "
                f"buffer_size={self.buffer_size}, data_world_size={self.data_world_size}."
            )
        if self.max_vision_patches is not None and int(self.max_vision_patches) <= 0:
            raise ValueError("max_vision_patches must be > 0 when set.")
        if not str(self.sample_schedule_fingerprint).strip():
            raise ValueError("sample_schedule_fingerprint must not be empty.")
        if not str(self.cost_fingerprint).strip():
            raise ValueError("cost_fingerprint must not be empty.")

    @property
    def fingerprint(self) -> str:
        payload = (
            SHAFT_BOUNDED_BATCHING_VERSION,
            int(self.data_world_size),
            int(self.buffer_size),
            int(self.max_samples_per_microbatch),
            int(self.max_padded_tokens),
            self.max_vision_patches,
            int(self.seed),
            str(self.sample_schedule_fingerprint),
            str(self.cost_fingerprint),
        )
        return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": SHAFT_BOUNDED_BATCHING_VERSION,
            "data_world_size": int(self.data_world_size),
            "buffer_size": int(self.buffer_size),
            "max_samples_per_microbatch": int(self.max_samples_per_microbatch),
            "max_padded_tokens": int(self.max_padded_tokens),
            "max_vision_patches": self.max_vision_patches,
            "seed": int(self.seed),
            "sample_schedule_fingerprint": str(self.sample_schedule_fingerprint),
            "cost_fingerprint": str(self.cost_fingerprint),
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ShaftBoundedBatchingSpec":
        version = str(payload.get("version", ""))
        if version != SHAFT_BOUNDED_BATCHING_VERSION:
            raise ValueError(
                "Unsupported bounded batching state version: "
                f"{version!r}; expected {SHAFT_BOUNDED_BATCHING_VERSION!r}."
            )
        spec = cls(
            data_world_size=int(payload["data_world_size"]),
            buffer_size=int(payload["buffer_size"]),
            max_samples_per_microbatch=int(payload["max_samples_per_microbatch"]),
            max_padded_tokens=int(payload["max_padded_tokens"]),
            max_vision_patches=_optional_int(payload, "max_vision_patches"),
            seed=int(payload["seed"]),
            sample_schedule_fingerprint=str(
                payload["sample_schedule_fingerprint"]
            ),
            cost_fingerprint=str(payload["cost_fingerprint"]),
        )
        serialized = str(payload.get("fingerprint", spec.fingerprint))
        if serialized != spec.fingerprint:
            raise ValueError("Bounded batching spec fingerprint does not match its payload.")
        return spec


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
        context_payload = payload["context"]
        cost_payload = payload["cost"]
        if not isinstance(context_payload, dict) or not isinstance(cost_payload, dict):
            raise TypeError("Buffered sample context and cost must be mappings.")
        return cls(
            sample_ref=ShaftSampleRef(
                dataset_name=str(payload["dataset_name"]),
                row_index=int(payload["row_index"]),
                context=ShaftSampleContext(
                    draw_id=int(context_payload["draw_id"]),
                    plan_cycle=int(context_payload["plan_cycle"]),
                    transform_seed=int(context_payload["transform_seed"]),
                ),
            ),
            cost=ShaftSampleCost(
                llm_tokens=int(cost_payload["llm_tokens"]),
                supervised_tokens=int(cost_payload.get("supervised_tokens", 0)),
                vision_patches=int(cost_payload.get("vision_patches", 0)),
                loss_weight_sum=(
                    None
                    if cost_payload.get("loss_weight_sum") is None
                    else float(cost_payload["loss_weight_sum"])
                ),
                exact=bool(cost_payload.get("exact", False)),
            ),
        )


@dataclass(frozen=True, slots=True)
class ShaftBoundedBatchingState:
    """Checkpointable state at a completed global-microstep boundary."""

    contract_fingerprint: str
    global_microstep: int = 0
    next_draw_id: int = 0
    buffer: tuple[ShaftBufferedSample, ...] = ()
    emitted_samples: int = 0
    emitted_llm_tokens: int = 0
    emitted_supervised_tokens: int = 0
    emitted_vision_patches: int = 0

    def __post_init__(self) -> None:
        if not str(self.contract_fingerprint).strip():
            raise ValueError("contract_fingerprint must not be empty.")
        for field_name in (
            "global_microstep",
            "next_draw_id",
            "emitted_samples",
            "emitted_llm_tokens",
            "emitted_supervised_tokens",
            "emitted_vision_patches",
        ):
            if int(getattr(self, field_name)) < 0:
                raise ValueError(f"{field_name} must be >= 0.")
        draw_ids = [entry.sample_ref.context.draw_id for entry in self.buffer]
        if len(draw_ids) != len(set(draw_ids)):
            raise ValueError("Bounded batching state buffer contains duplicate draws.")
        if draw_ids != sorted(draw_ids):
            raise ValueError("Bounded batching state buffer must preserve FIFO draw order.")
        if draw_ids and max(draw_ids) >= int(self.next_draw_id):
            raise ValueError("Buffered draw ids must be lower than next_draw_id.")
        if int(self.next_draw_id) != int(self.emitted_samples) + len(self.buffer):
            raise ValueError(
                "Bounded batching state violates draw conservation: "
                "next_draw_id must equal emitted_samples + len(buffer)."
            )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "version": SHAFT_BOUNDED_BATCHING_VERSION,
            "contract_fingerprint": str(self.contract_fingerprint),
            "global_microstep": int(self.global_microstep),
            "next_draw_id": int(self.next_draw_id),
            "buffer": [entry.to_dict() for entry in self.buffer],
            "emitted_samples": int(self.emitted_samples),
            "emitted_llm_tokens": int(self.emitted_llm_tokens),
            "emitted_supervised_tokens": int(self.emitted_supervised_tokens),
            "emitted_vision_patches": int(self.emitted_vision_patches),
        }
        payload["fingerprint"] = _mapping_fingerprint(payload)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ShaftBoundedBatchingState":
        version = str(payload.get("version", ""))
        if version != SHAFT_BOUNDED_BATCHING_VERSION:
            raise ValueError(
                "Unsupported bounded batching state version: "
                f"{version!r}; expected {SHAFT_BOUNDED_BATCHING_VERSION!r}."
            )
        fingerprint = str(payload.get("fingerprint", ""))
        unsigned = dict(payload)
        unsigned.pop("fingerprint", None)
        expected = _mapping_fingerprint(unsigned)
        if fingerprint != expected:
            raise ValueError("Bounded batching state fingerprint does not match its payload.")
        buffer_payload = payload.get("buffer", [])
        if not isinstance(buffer_payload, list):
            raise TypeError("Bounded batching state buffer must be a list.")
        return cls(
            contract_fingerprint=str(payload["contract_fingerprint"]),
            global_microstep=int(payload.get("global_microstep", 0)),
            next_draw_id=int(payload.get("next_draw_id", 0)),
            buffer=tuple(
                ShaftBufferedSample.from_dict(item) for item in buffer_payload
            ),
            emitted_samples=int(payload.get("emitted_samples", 0)),
            emitted_llm_tokens=int(payload.get("emitted_llm_tokens", 0)),
            emitted_supervised_tokens=int(
                payload.get("emitted_supervised_tokens", 0)
            ),
            emitted_vision_patches=int(payload.get("emitted_vision_patches", 0)),
        )


@dataclass(frozen=True, slots=True)
class ShaftBoundedMicrobatchStats:
    sample_count: int
    useful_llm_tokens: int
    padded_llm_tokens: int
    supervised_tokens: int
    vision_patches: int
    min_local_batch_size: int
    max_local_batch_size: int
    max_local_padded_tokens: int
    max_local_vision_patches: int
    max_rank_cost_skew: float

    @property
    def padding_ratio(self) -> float:
        if self.padded_llm_tokens <= 0:
            return 0.0
        return 1.0 - self.useful_llm_tokens / self.padded_llm_tokens


@dataclass(frozen=True, slots=True)
class ShaftBoundedMicrobatchPlan:
    global_microstep: int
    rank_microbatches: tuple[ShaftLocalMicroBatchPlan, ...]
    stats: ShaftBoundedMicrobatchStats
    state_after: ShaftBoundedBatchingState

    @property
    def fingerprint(self) -> str:
        payload = (
            SHAFT_BOUNDED_BATCHING_VERSION,
            int(self.global_microstep),
            tuple(
                tuple(
                    (
                        entry.dataset_name,
                        int(entry.row_index),
                        entry.context.to_dict(),
                        cost,
                    )
                    for entry, cost in zip(
                        batch.sample_refs,
                        batch.sample_costs,
                        strict=True,
                    )
                )
                for batch in self.rank_microbatches
            ),
            self.state_after.to_dict()["fingerprint"],
        )
        return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


class ShaftBoundedBatchPlanner:
    """Greedily pack one global microstep from a bounded FIFO lookahead."""

    def __init__(
        self,
        *,
        schedule: ShaftSampleSchedule,
        cost_provider: ShaftSampleCostProvider,
        spec: ShaftBoundedBatchingSpec,
        state: ShaftBoundedBatchingState | None = None,
    ) -> None:
        if str(schedule.fingerprint) != str(spec.sample_schedule_fingerprint):
            raise ValueError("Bounded batching spec does not belong to this sample schedule.")
        provider_fingerprint = str(getattr(cost_provider, "fingerprint", ""))
        if provider_fingerprint != str(spec.cost_fingerprint):
            raise ValueError("Bounded batching spec cost fingerprint does not match provider.")
        if state is None:
            state = ShaftBoundedBatchingState(
                contract_fingerprint=spec.fingerprint,
            )
        if state.contract_fingerprint != spec.fingerprint:
            raise ValueError(
                "Bounded batching checkpoint contract differs from the active runtime."
            )
        if len(state.buffer) > int(spec.buffer_size):
            raise ValueError("Checkpoint buffer exceeds the configured buffer_size.")
        self.schedule = schedule
        self.cost_provider = cost_provider
        self.spec = spec
        for entry in state.buffer:
            expected_ref = schedule.ref_at(entry.sample_ref.context.draw_id)
            if expected_ref != entry.sample_ref:
                raise ValueError(
                    "Bounded batching checkpoint buffer does not match the sample schedule."
                )
            self._validate_sample_cost(entry.sample_ref, entry.cost)
        self.state = state

    def next_global_microbatch(self) -> ShaftBoundedMicrobatchPlan:
        buffer = list(self.state.buffer)
        next_draw_id = int(self.state.next_draw_id)
        while len(buffer) < int(self.spec.buffer_size):
            sample_ref = self.schedule.ref_at(next_draw_id)
            cost = self.cost_provider(sample_ref)
            self._validate_sample_cost(sample_ref, cost)
            buffer.append(ShaftBufferedSample(sample_ref=sample_ref, cost=cost))
            next_draw_id += 1

        world_size = int(self.spec.data_world_size)
        if len(buffer) < world_size:
            raise RuntimeError("Bounded batching buffer cannot seed every data rank.")

        anchors = buffer[:world_size]
        bins: list[list[ShaftBufferedSample]] = [[entry] for entry in anchors]
        selected_draw_ids = {
            entry.sample_ref.context.draw_id for entry in anchors
        }
        candidates = sorted(
            buffer[world_size:],
            key=lambda entry: (
                int(entry.cost.llm_tokens),
                int(entry.cost.vision_patches),
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

        local_batches = tuple(self._build_local_batch(batch) for batch in bins)
        remaining = tuple(
            entry
            for entry in buffer
            if entry.sample_ref.context.draw_id not in selected_draw_ids
        )
        stats = self._build_stats(local_batches)
        state_after = ShaftBoundedBatchingState(
            contract_fingerprint=self.spec.fingerprint,
            global_microstep=int(self.state.global_microstep) + 1,
            next_draw_id=next_draw_id,
            buffer=remaining,
            emitted_samples=int(self.state.emitted_samples) + stats.sample_count,
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
        plan = ShaftBoundedMicrobatchPlan(
            global_microstep=int(self.state.global_microstep),
            rank_microbatches=local_batches,
            stats=stats,
            state_after=state_after,
        )
        self.state = state_after
        return plan

    def iter_global_microbatches(self, count: int) -> Iterator[ShaftBoundedMicrobatchPlan]:
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
                "Bounded batching requires exact sample costs; "
                f"draw_id={draw_id} is inexact."
            )
        if int(cost.llm_tokens) > int(self.spec.max_padded_tokens):
            raise ValueError(
                "Bounded batching encountered an oversize sample: "
                f"draw_id={draw_id}, llm_tokens={cost.llm_tokens}, "
                f"max_padded_tokens={self.spec.max_padded_tokens}."
            )
        if (
            self.spec.max_vision_patches is not None
            and int(cost.vision_patches) > int(self.spec.max_vision_patches)
        ):
            raise ValueError(
                "Bounded batching encountered an oversize sample: "
                f"draw_id={draw_id}, vision_patches={cost.vision_patches}, "
                f"max_vision_patches={self.spec.max_vision_patches}."
            )

    def _fits(
        self,
        batch: list[ShaftBufferedSample],
        entry: ShaftBufferedSample,
    ) -> bool:
        projected = [*batch, entry]
        if len(projected) > int(self.spec.max_samples_per_microbatch):
            return False
        padded_tokens = len(projected) * max(
            int(item.cost.llm_tokens) for item in projected
        )
        if padded_tokens > int(self.spec.max_padded_tokens):
            return False
        if self.spec.max_vision_patches is not None:
            vision_patches = sum(int(item.cost.vision_patches) for item in projected)
            if vision_patches > int(self.spec.max_vision_patches):
                return False
        return True

    def _placement_score(
        self,
        batch: list[ShaftBufferedSample],
        bin_index: int,
    ) -> tuple[float, float, int, int]:
        useful = sum(int(item.cost.llm_tokens) for item in batch)
        padded = len(batch) * max(int(item.cost.llm_tokens) for item in batch)
        text_fill = padded / int(self.spec.max_padded_tokens)
        vision = sum(int(item.cost.vision_patches) for item in batch)
        vision_fill = (
            0.0
            if self.spec.max_vision_patches is None
            else vision / int(self.spec.max_vision_patches)
        )
        padding_waste = (padded - useful) / int(self.spec.max_padded_tokens)
        # Rank stragglers dominate DDP step time. Prefer the least-loaded feasible
        # projected bin first; padding waste and stable index break ties.
        projected_load = text_fill + vision_fill
        return (projected_load, padding_waste, len(batch), int(bin_index))

    @staticmethod
    def _build_local_batch(
        batch: list[ShaftBufferedSample],
    ) -> ShaftLocalMicroBatchPlan:
        ordered = sorted(
            batch,
            key=lambda entry: int(entry.sample_ref.context.draw_id),
        )
        return ShaftLocalMicroBatchPlan(
            sample_refs=tuple(entry.sample_ref for entry in ordered),
            sample_costs=tuple(entry.cost for entry in ordered),
        )

    def _batch_load(self, batch: ShaftLocalMicroBatchPlan) -> float:
        text = batch.padded_llm_tokens / int(self.spec.max_padded_tokens)
        vision = (
            0.0
            if self.spec.max_vision_patches is None
            else batch.vision_patches / int(self.spec.max_vision_patches)
        )
        return text + vision

    def _build_stats(
        self,
        batches: tuple[ShaftLocalMicroBatchPlan, ...],
    ) -> ShaftBoundedMicrobatchStats:
        loads = [self._batch_load(batch) for batch in batches]
        mean_load = sum(loads) / len(loads)
        max_rank_cost_skew = (
            0.0
            if mean_load <= 0
            else max(abs(load - mean_load) / mean_load for load in loads)
        )
        return ShaftBoundedMicrobatchStats(
            sample_count=sum(len(batch.sample_refs) for batch in batches),
            useful_llm_tokens=sum(batch.useful_llm_tokens for batch in batches),
            padded_llm_tokens=sum(batch.padded_llm_tokens for batch in batches),
            supervised_tokens=sum(batch.supervised_tokens for batch in batches),
            vision_patches=sum(batch.vision_patches for batch in batches),
            min_local_batch_size=min(len(batch.sample_refs) for batch in batches),
            max_local_batch_size=max(len(batch.sample_refs) for batch in batches),
            max_local_padded_tokens=max(batch.padded_llm_tokens for batch in batches),
            max_local_vision_patches=max(batch.vision_patches for batch in batches),
            max_rank_cost_skew=max_rank_cost_skew,
        )
