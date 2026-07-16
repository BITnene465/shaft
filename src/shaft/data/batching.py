from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from typing import Any, Sequence

import torch

from shaft.utils.contract_schema import (
    json_bool,
    json_int,
    json_mapping,
    json_optional_number,
    json_string,
    require_exact_keys,
    require_json_mapping,
)

from .cost import ShaftSampleCost
from .mixing import ShaftSampleContext, ShaftSampleRef
from .planned import ShaftBatchContext


@dataclass(frozen=True, slots=True)
class ShaftCollatedBatchStats:
    """Actual, post-processor resources materialized for one local microbatch.

    This is deliberately independent from planner estimates.  It travels as a
    private host-side collator field and is consumed by the Trainer before model
    forward.
    """

    logical_segments: int
    physical_packs: int
    useful_tokens: int
    materialized_tokens: int
    supervised_tokens: int
    weighted_supervision_mass: float | None
    sequence_length_sum: int
    sequence_length_square_sum: int
    vision_patches: int | None
    global_microstep: int | None = None
    plan_fingerprint: str | None = None
    local_batch_id: int | None = None

    def __post_init__(self) -> None:
        integer_fields = (
            "logical_segments",
            "physical_packs",
            "useful_tokens",
            "materialized_tokens",
            "supervised_tokens",
            "sequence_length_sum",
            "sequence_length_square_sum",
        )
        for name in integer_fields:
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be >= 0.")
        if self.logical_segments <= 0 or self.physical_packs <= 0:
            raise ValueError("A collated training batch cannot be empty.")
        if self.useful_tokens > self.materialized_tokens:
            raise ValueError("useful_tokens cannot exceed materialized_tokens.")
        if self.supervised_tokens > self.useful_tokens:
            raise ValueError("supervised_tokens cannot exceed useful_tokens.")
        if self.vision_patches is not None and int(self.vision_patches) < 0:
            raise ValueError("vision_patches must be >= 0 when known.")

    @classmethod
    def from_training_inputs(
        cls,
        *,
        sequence_inputs: dict[str, Any],
        varlen_plan: Any = None,
        vision_patches: int | None = None,
        ignore_index: int = -100,
    ) -> "ShaftCollatedBatchStats":
        input_ids = sequence_inputs.get("input_ids")
        labels = sequence_inputs.get("labels")
        if not torch.is_tensor(input_ids) or input_ids.ndim != 2:
            raise ValueError("Collated batch statistics require rank-2 input_ids.")
        if not torch.is_tensor(labels) or tuple(labels.shape) != tuple(input_ids.shape):
            raise ValueError("Collated batch statistics require labels aligned to input_ids.")

        attention_mask = sequence_inputs.get("attention_mask")
        if attention_mask is None:
            lengths = torch.full(
                (int(input_ids.shape[0]),),
                int(input_ids.shape[1]),
                dtype=torch.long,
            )
        else:
            if not torch.is_tensor(attention_mask) or tuple(attention_mask.shape) != tuple(
                input_ids.shape
            ):
                raise ValueError("attention_mask must align with input_ids.")
            lengths = attention_mask.to(dtype=torch.long, device="cpu").sum(dim=-1)

        shifted_valid = labels[..., 1:].ne(int(ignore_index))
        supervised_tokens = int(shifted_valid.sum().item())
        loss_scale = sequence_inputs.get("loss_scale")
        weighted_mass = None
        if loss_scale is not None:
            if not torch.is_tensor(loss_scale) or tuple(loss_scale.shape) != tuple(
                labels.shape
            ):
                raise ValueError("loss_scale must align with labels.")
            weighted_mass = float(
                (
                    loss_scale[..., 1:].to(dtype=torch.float32)
                    * shifted_valid.to(dtype=torch.float32)
                )
                .sum()
                .item()
            )

        if varlen_plan is None:
            physical_packs = int(input_ids.shape[0])
            logical_segments = physical_packs
            logical_lengths = lengths
            global_microstep = None
            plan_fingerprint = None
            local_batch_id = None
        else:
            physical_packs = int(varlen_plan.physical_pack_count)
            logical_segments = int(varlen_plan.logical_segment_count)
            logical_lengths = torch.tensor(
                [
                    int(segment.stop) - int(segment.start)
                    for segment in varlen_plan.segments
                ],
                dtype=torch.long,
            )
            if int(logical_lengths.sum().item()) != int(lengths.sum().item()):
                raise ValueError(
                    "Varlen logical segment lengths do not cover actual useful tokens."
                )
            global_microstep = int(varlen_plan.global_microstep)
            plan_fingerprint = str(varlen_plan.plan_fingerprint)
            local_batch_id = int(varlen_plan.local_batch_id)

        return cls(
            logical_segments=logical_segments,
            physical_packs=physical_packs,
            useful_tokens=int(lengths.sum().item()),
            materialized_tokens=int(input_ids.numel()),
            supervised_tokens=supervised_tokens,
            weighted_supervision_mass=weighted_mass,
            sequence_length_sum=int(logical_lengths.sum().item()),
            sequence_length_square_sum=int(
                (logical_lengths * logical_lengths).sum().item()
            ),
            vision_patches=None if vision_patches is None else int(vision_patches),
            global_microstep=global_microstep,
            plan_fingerprint=plan_fingerprint,
            local_batch_id=local_batch_id,
        )


@dataclass(frozen=True, slots=True)
class ShaftLogicalSegmentPlan:
    """One indivisible logical training sequence in a physical batch plan."""

    sample_ref: ShaftSampleRef
    cost: ShaftSampleCost

    @property
    def draw_id(self) -> int:
        return int(self.sample_ref.context.draw_id)

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
    def from_dict(cls, payload: dict[str, Any]) -> "ShaftLogicalSegmentPlan":
        role = "Logical segment"
        payload = require_json_mapping(payload, role=role)
        require_exact_keys(
            payload,
            role=role,
            expected=frozenset({"dataset_name", "row_index", "context", "cost"}),
        )
        context_payload = json_mapping(payload, "context", role=role)
        require_exact_keys(
            context_payload,
            role=f"{role}.context",
            expected=frozenset({"draw_id", "plan_cycle", "transform_seed"}),
        )
        cost_payload = json_mapping(payload, "cost", role=role)
        require_exact_keys(
            cost_payload,
            role=f"{role}.cost",
            expected=frozenset(
                {
                    "llm_tokens",
                    "supervised_tokens",
                    "vision_patches",
                    "loss_weight_sum",
                    "exact",
                }
            ),
        )
        return cls(
            sample_ref=ShaftSampleRef(
                dataset_name=json_string(payload, "dataset_name", role=role),
                row_index=json_int(payload, "row_index", role=role),
                context=ShaftSampleContext(
                    draw_id=json_int(context_payload, "draw_id", role=f"{role}.context"),
                    plan_cycle=json_int(
                        context_payload,
                        "plan_cycle",
                        role=f"{role}.context",
                    ),
                    transform_seed=json_int(
                        context_payload,
                        "transform_seed",
                        role=f"{role}.context",
                    ),
                ),
            ),
            cost=ShaftSampleCost(
                llm_tokens=json_int(cost_payload, "llm_tokens", role=f"{role}.cost"),
                supervised_tokens=json_int(
                    cost_payload,
                    "supervised_tokens",
                    role=f"{role}.cost",
                ),
                vision_patches=json_int(
                    cost_payload,
                    "vision_patches",
                    role=f"{role}.cost",
                ),
                loss_weight_sum=json_optional_number(
                    cost_payload,
                    "loss_weight_sum",
                    role=f"{role}.cost",
                ),
                exact=json_bool(cost_payload, "exact", role=f"{role}.cost"),
            ),
        )


@dataclass(frozen=True, slots=True)
class ShaftPhysicalPackPlan:
    """One physical sequence containing one or more whole logical segments."""

    segments: tuple[ShaftLogicalSegmentPlan, ...]

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError("A physical pack cannot be empty.")
        draw_ids = [segment.draw_id for segment in self.segments]
        if len(draw_ids) != len(set(draw_ids)):
            raise ValueError("A physical pack cannot contain duplicate logical draws.")

    @property
    def useful_llm_tokens(self) -> int:
        return sum(int(segment.cost.llm_tokens) for segment in self.segments)

    @property
    def supervised_tokens(self) -> int:
        return sum(int(segment.cost.supervised_tokens) for segment in self.segments)

    @property
    def vision_patches(self) -> int:
        return sum(int(segment.cost.vision_patches) for segment in self.segments)

    def resource_total(self, name: str) -> int:
        return sum(segment.cost.resource_value(name) for segment in self.segments)


@dataclass(frozen=True, slots=True)
class ShaftLocalMicroBatchPlan:
    """One rank-local microbatch with physical packs as its structural truth."""

    packs: tuple[ShaftPhysicalPackPlan, ...]

    def __post_init__(self) -> None:
        if not self.packs:
            raise ValueError("A local microbatch plan cannot be empty.")
        draw_ids = [segment.draw_id for pack in self.packs for segment in pack.segments]
        if len(draw_ids) != len(set(draw_ids)):
            raise ValueError("A local microbatch cannot contain duplicate logical draws.")

    @classmethod
    def from_segments(
        cls,
        segments: Sequence[ShaftLogicalSegmentPlan],
    ) -> "ShaftLocalMicroBatchPlan":
        """Build the packing-none representation: one singleton pack per segment."""

        return cls(
            packs=tuple(
                ShaftPhysicalPackPlan(segments=(segment,)) for segment in segments
            )
        )

    @property
    def physical_pack_count(self) -> int:
        return len(self.packs)

    @property
    def logical_segment_count(self) -> int:
        return sum(len(pack.segments) for pack in self.packs)

    @property
    def segments(self) -> tuple[ShaftLogicalSegmentPlan, ...]:
        return tuple(segment for pack in self.packs for segment in pack.segments)

    @property
    def sample_refs(self) -> tuple[ShaftSampleRef, ...]:
        return tuple(segment.sample_ref for segment in self.segments)

    @property
    def sample_costs(self) -> tuple[ShaftSampleCost, ...]:
        return tuple(segment.cost for segment in self.segments)

    @property
    def useful_llm_tokens(self) -> int:
        return sum(pack.useful_llm_tokens for pack in self.packs)

    @property
    def max_llm_tokens(self) -> int:
        return max(pack.useful_llm_tokens for pack in self.packs)

    @property
    def padded_llm_tokens(self) -> int:
        return self.physical_pack_count * self.max_llm_tokens

    @property
    def supervised_tokens(self) -> int:
        return sum(pack.supervised_tokens for pack in self.packs)

    @property
    def vision_patches(self) -> int:
        return sum(pack.vision_patches for pack in self.packs)

    def resource_total(self, name: str) -> int:
        return sum(pack.resource_total(name) for pack in self.packs)


class ShaftLengthBatchGrouping:
    """Deterministic length priority over one bounded candidate window."""

    name = "length"

    @classmethod
    def build(
        cls,
        segments: Sequence[ShaftLogicalSegmentPlan],
        *,
        seed: int,
        global_microstep: int,
    ) -> tuple[ShaftLogicalSegmentPlan, ...]:
        _ = seed, global_microstep
        draw_ids = [segment.draw_id for segment in segments]
        if len(draw_ids) != len(set(draw_ids)):
            raise ValueError("Length grouping candidates contain duplicate logical draws.")
        return tuple(
            sorted(
                segments,
                key=lambda segment: (
                    -int(segment.cost.llm_tokens),
                    int(segment.draw_id),
                ),
            )
        )


class ShaftGreedySequencePacker:
    """Stable whole-sample best-fit fill over a bounded candidate window."""

    name = "greedy"

    @classmethod
    def build(
        cls,
        segments: Sequence[ShaftLogicalSegmentPlan],
        *,
        physical_pack_count: int,
        max_length: int,
        required_draw_id: int,
        placement_feasible: Callable[
            [tuple[ShaftPhysicalPackPlan, ...]], bool
        ]
        | None = None,
    ) -> tuple[ShaftPhysicalPackPlan, ...]:
        pack_count = int(physical_pack_count)
        capacity = int(max_length)
        required = int(required_draw_id)
        if pack_count <= 0:
            raise ValueError("physical_pack_count must be > 0.")
        if capacity <= 0:
            raise ValueError("max_length must be > 0.")
        candidates = tuple(segments)
        if len(candidates) < pack_count:
            raise ValueError(
                "Greedy packing requires at least one logical segment per physical pack."
            )
        draw_ids = [segment.draw_id for segment in candidates]
        if len(draw_ids) != len(set(draw_ids)):
            raise ValueError("Greedy packing candidates contain duplicate logical draws.")
        for segment in candidates:
            tokens = int(segment.cost.llm_tokens)
            if tokens > capacity:
                raise ValueError(
                    "Greedy packing encountered an oversize logical segment: "
                    f"draw_id={segment.draw_id}, llm_tokens={tokens}, "
                    f"max_length={capacity}."
                )
        by_draw = {segment.draw_id: segment for segment in candidates}
        if required not in by_draw:
            raise ValueError(f"required_draw_id={required} is not in the candidate window.")

        required_segment = by_draw[required]
        remaining_for_seeds = sorted(
            (segment for segment in candidates if segment.draw_id != required),
            key=lambda segment: (
                -int(segment.cost.llm_tokens),
                int(segment.draw_id),
            ),
        )
        bins: list[list[ShaftLogicalSegmentPlan]] = [[required_segment]]
        loads = [int(required_segment.cost.llm_tokens)]
        if not cls._placement_is_feasible(bins, placement_feasible):
            raise ValueError(
                "The required oldest draw cannot satisfy physical placement guards: "
                f"draw_id={required}."
            )
        seeded_draws = {required_segment.draw_id}
        for segment in remaining_for_seeds:
            tentative = [*bins, [segment]]
            if not cls._placement_is_feasible(tentative, placement_feasible):
                continue
            bins.append([segment])
            loads.append(int(segment.cost.llm_tokens))
            seeded_draws.add(segment.draw_id)
            if len(bins) == pack_count:
                break
        if len(bins) != pack_count:
            raise ValueError(
                "Greedy packing cannot seed every physical pack within the active "
                "resource guards."
            )

        remaining = sorted(
            (segment for segment in candidates if segment.draw_id not in seeded_draws),
            key=lambda segment: (
                -int(segment.cost.llm_tokens),
                int(segment.draw_id),
            ),
        )
        for segment in remaining:
            tokens = int(segment.cost.llm_tokens)
            feasible: list[tuple[int, int]] = []
            for index in range(pack_count):
                if loads[index] + tokens > capacity:
                    continue
                tentative = [list(pack) for pack in bins]
                tentative[index].append(segment)
                if not cls._placement_is_feasible(tentative, placement_feasible):
                    continue
                feasible.append((capacity - (loads[index] + tokens), index))
            if not feasible:
                continue
            _, bin_index = min(feasible)
            bins[bin_index].append(segment)
            loads[bin_index] += tokens

        return tuple(
            ShaftPhysicalPackPlan(segments=tuple(pack)) for pack in bins
        )

    @staticmethod
    def _placement_is_feasible(
        bins: list[list[ShaftLogicalSegmentPlan]],
        callback: Callable[[tuple[ShaftPhysicalPackPlan, ...]], bool] | None,
    ) -> bool:
        if callback is None:
            return True
        packs = tuple(
            ShaftPhysicalPackPlan(segments=tuple(segments)) for segments in bins
        )
        return bool(callback(packs))


@dataclass(frozen=True, slots=True)
class ShaftVarlenSegmentLayout:
    """One logical segment's position in a flattened varlen tensor."""

    processor_row_index: int
    pack_index: int
    segment_index: int
    start: int
    stop: int

    def __post_init__(self) -> None:
        for name in ("processor_row_index", "pack_index", "segment_index", "start"):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"ShaftVarlenSegmentLayout.{name} must be >= 0.")
        if int(self.stop) <= int(self.start):
            raise ValueError("A varlen logical segment must contain at least one token.")

    @property
    def length(self) -> int:
        return int(self.stop) - int(self.start)


@dataclass(frozen=True, slots=True)
class ShaftVarlenLayoutPlan:
    """Host-side boundary truth consumed by a model execution policy."""

    global_microstep: int
    plan_fingerprint: str
    local_batch_id: int
    pack_lengths: tuple[int, ...]
    segments: tuple[ShaftVarlenSegmentLayout, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "pack_lengths",
            tuple(int(length) for length in self.pack_lengths),
        )
        object.__setattr__(self, "segments", tuple(self.segments))
        if int(self.global_microstep) < 0 or int(self.local_batch_id) < 0:
            raise ValueError("Varlen layout ids must be >= 0.")
        if not str(self.plan_fingerprint).strip():
            raise ValueError("Varlen layout plan_fingerprint must not be empty.")
        if not self.pack_lengths or any(int(length) <= 0 for length in self.pack_lengths):
            raise ValueError("Varlen layout pack_lengths must be positive.")
        if not self.segments:
            raise ValueError("Varlen layout must contain at least one segment.")

        expected_start = 0
        observed_order: list[tuple[int, int]] = []
        for segment in self.segments:
            if int(segment.start) != expected_start:
                raise ValueError("Varlen segments must be contiguous and ordered.")
            expected_start = int(segment.stop)
            observed_order.append((int(segment.pack_index), int(segment.segment_index)))
        if observed_order != sorted(observed_order):
            raise ValueError("Varlen segments must be ordered by pack and segment index.")

        processor_rows = tuple(
            int(segment.processor_row_index) for segment in self.segments
        )
        if processor_rows != tuple(range(len(self.segments))):
            raise ValueError("Varlen processor rows must be unique, contiguous, and ordered.")

        pack_count = len(self.pack_lengths)
        for pack_index in range(pack_count):
            pack_segments = tuple(
                segment
                for segment in self.segments
                if int(segment.pack_index) == pack_index
            )
            if not pack_segments:
                raise ValueError("Every varlen physical pack must contain a segment.")
            if tuple(int(segment.segment_index) for segment in pack_segments) != tuple(
                range(len(pack_segments))
            ):
                raise ValueError("Varlen segment indices must be contiguous within each pack.")
            if sum(segment.length for segment in pack_segments) != self.pack_lengths[
                pack_index
            ]:
                raise ValueError("Varlen pack lengths do not match their logical segments.")
        if any(
            int(segment.pack_index) >= pack_count for segment in self.segments
        ):
            raise ValueError("Varlen segment pack index exceeds pack_lengths.")
        if sum(self.pack_lengths) != expected_start:
            raise ValueError("Varlen pack lengths do not cover all flattened tokens.")

    @property
    def physical_pack_count(self) -> int:
        return len(self.pack_lengths)

    @property
    def logical_segment_count(self) -> int:
        return len(self.segments)

    @property
    def total_tokens(self) -> int:
        return int(self.segments[-1].stop)


class ShaftVarlenBatchLayout:
    """Assemble planned whole-sample packs into one padding-free tensor row."""

    name = "varlen"

    @classmethod
    def build(
        cls,
        *,
        contexts: Sequence[ShaftBatchContext | dict[str, Any] | None],
        input_ids: Sequence[torch.Tensor],
        labels: Sequence[torch.Tensor],
        mm_token_type_ids: Sequence[torch.Tensor | None],
        loss_scales: Sequence[torch.Tensor | None],
        ignore_index: int,
        max_sequence_length: int | None,
    ) -> tuple[dict[str, torch.Tensor], ShaftVarlenLayoutPlan]:
        row_count = len(input_ids)
        if row_count <= 0:
            raise ValueError("A varlen batch cannot be empty.")
        fields = {
            "contexts": contexts,
            "labels": labels,
            "mm_token_type_ids": mm_token_type_ids,
            "loss_scales": loss_scales,
        }
        for name, values in fields.items():
            if len(values) != row_count:
                raise ValueError(f"Varlen {name} must align with input_ids rows.")

        resolved_contexts: list[ShaftBatchContext] = []
        for payload in contexts:
            if payload is None:
                raise ValueError("varlen layout requires plan context for every row.")
            try:
                context = (
                    payload
                    if isinstance(payload, ShaftBatchContext)
                    else ShaftBatchContext.from_dict(payload)
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("varlen layout received an invalid plan context.") from exc
            resolved_contexts.append(context)

        identity = {
            (
                context.global_microstep,
                context.plan_fingerprint,
                context.local_batch_id,
            )
            for context in resolved_contexts
        }
        if len(identity) != 1:
            raise ValueError("varlen rows must belong to one local batch plan.")
        observed_order = [
            (context.pack_index, context.segment_index)
            for context in resolved_contexts
        ]
        if observed_order != sorted(observed_order) or len(observed_order) != len(
            set(observed_order)
        ):
            raise ValueError("varlen plan rows must be unique and ordered by pack/segment.")

        pack_indices = sorted({context.pack_index for context in resolved_contexts})
        if pack_indices != list(range(len(pack_indices))):
            raise ValueError("varlen plan pack indices must be contiguous from zero.")
        for pack_index in pack_indices:
            pack_contexts = [
                context
                for context in resolved_contexts
                if context.pack_index == pack_index
            ]
            declared_counts = {context.pack_segment_count for context in pack_contexts}
            expected_count = len(pack_contexts)
            if declared_counts != {expected_count} or [
                context.segment_index for context in pack_contexts
            ] != list(range(expected_count)):
                raise ValueError("varlen plan segment indices/counts are inconsistent.")

        normalized_inputs: list[torch.Tensor] = []
        normalized_labels: list[torch.Tensor] = []
        normalized_mm: list[torch.Tensor | None] = []
        normalized_scales: list[torch.Tensor | None] = []
        segments: list[ShaftVarlenSegmentLayout] = []
        cursor = 0
        for row_index, (context, token_row, label_row, mm_row, scale_row) in enumerate(
            zip(
                resolved_contexts,
                input_ids,
                labels,
                mm_token_type_ids,
                loss_scales,
                strict=True,
            )
        ):
            if not torch.is_tensor(token_row) or token_row.ndim != 1 or token_row.numel() == 0:
                raise ValueError("Varlen input rows must be non-empty 1D tensors.")
            length = int(token_row.shape[0])
            if not torch.is_tensor(label_row) or tuple(label_row.shape) != (length,):
                raise ValueError("Varlen labels must align with their input row.")
            if mm_row is not None and tuple(mm_row.shape) != (length,):
                raise ValueError("Varlen mm_token_type_ids must align with input rows.")
            if scale_row is not None and tuple(scale_row.shape) != (length,):
                raise ValueError("Varlen loss_scale must align with input rows.")
            labels_copy = label_row.clone()
            labels_copy[0] = int(ignore_index)
            scale_copy = None if scale_row is None else scale_row.clone()
            if scale_copy is not None:
                scale_copy[0] = 0
            normalized_inputs.append(token_row)
            normalized_labels.append(labels_copy)
            normalized_mm.append(mm_row)
            normalized_scales.append(scale_copy)
            segments.append(
                ShaftVarlenSegmentLayout(
                    processor_row_index=row_index,
                    pack_index=int(context.pack_index),
                    segment_index=int(context.segment_index),
                    start=cursor,
                    stop=cursor + length,
                )
            )
            cursor += length

        pack_lengths = tuple(
            sum(
                segment.length
                for segment in segments
                if segment.pack_index == pack_index
            )
            for pack_index in pack_indices
        )
        if max_sequence_length is not None and any(
            length > int(max_sequence_length) for length in pack_lengths
        ):
            raise ValueError(
                "varlen physical pack exceeds data.max_length after collation."
            )

        sequence_inputs: dict[str, torch.Tensor] = {
            "input_ids": torch.cat(normalized_inputs, dim=0).unsqueeze(0),
            "labels": torch.cat(normalized_labels, dim=0).unsqueeze(0),
        }
        if any(row is not None for row in normalized_mm):
            if not all(row is not None for row in normalized_mm):
                raise ValueError("Varlen mm_token_type_ids must be present for every row.")
            sequence_inputs["mm_token_type_ids"] = torch.cat(
                [row for row in normalized_mm if row is not None],
                dim=0,
            ).unsqueeze(0)
        if any(row is not None for row in normalized_scales):
            if not all(row is not None for row in normalized_scales):
                raise ValueError("Varlen loss_scale must be present for every row.")
            sequence_inputs["loss_scale"] = torch.cat(
                [row for row in normalized_scales if row is not None],
                dim=0,
            ).unsqueeze(0).to(dtype=torch.float32)

        first = resolved_contexts[0]
        plan = ShaftVarlenLayoutPlan(
            global_microstep=int(first.global_microstep),
            plan_fingerprint=str(first.plan_fingerprint),
            local_batch_id=int(first.local_batch_id),
            pack_lengths=pack_lengths,
            segments=tuple(segments),
        )
        return sequence_inputs, plan


def resolve_local_pack_count_bounds(
    cardinality: str,
    per_device_train_batch_size: int,
) -> tuple[int, int]:
    """Resolve the rank-local physical-pack count without another batch knob."""

    normalized = str(cardinality).strip().lower()
    maximum = int(per_device_train_batch_size)
    if maximum <= 0:
        raise ValueError("per_device_train_batch_size must be > 0.")
    if normalized == "fixed":
        return maximum, maximum
    if normalized == "token_budget":
        return 1, maximum
    raise ValueError(f"Unsupported batch cardinality: {cardinality!r}.")
