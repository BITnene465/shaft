from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import importlib.util
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Mapping

from packaging.version import InvalidVersion, Version
import torch

from .inference import ShaftInferenceContract, ShaftInferencePolicy
from .sharding import ModelShardingPolicy

if TYPE_CHECKING:
    from .descriptor import ResolvedModelDescriptor


def _dedupe_non_empty(values: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(item).strip() for item in values if str(item).strip()))


_CORE_SEQUENCE_INPUT_NAMES = (
    "input_ids",
    "attention_mask",
    "labels",
    "loss_scale",
    "completion_mask",
)


def _matches_group_prefix(name: str, prefix: str) -> bool:
    normalized_name = str(name).strip()
    normalized_prefix = str(prefix).strip()
    return bool(normalized_prefix) and (
        normalized_name == normalized_prefix or normalized_name.startswith(f"{normalized_prefix}.")
    )


def _missing_requires(requires: tuple[str, ...]) -> list[str]:
    missing: list[str] = []
    for requirement in requires:
        if requirement.startswith("module:"):
            module_name = requirement.removeprefix("module:").strip()
            if not module_name or importlib.util.find_spec(module_name) is None:
                missing.append(requirement)
            continue
        operator = ""
        package = requirement
        expected_version = ""
        for candidate in (">=", "=="):
            if candidate in requirement:
                package, expected_version = requirement.split(candidate, 1)
                operator = candidate
                break
        package = package.strip()
        if package and importlib.util.find_spec(package) is None:
            missing.append(requirement)
            continue
        if package and operator and expected_version:
            try:
                installed = Version(version(package))
                expected = Version(expected_version.strip())
            except (InvalidVersion, PackageNotFoundError):
                missing.append(requirement)
                continue
            if operator == ">=" and installed < expected:
                missing.append(requirement)
            elif operator == "==" and installed != expected:
                missing.append(requirement)
    return missing


@contextmanager
def _temporary_processor_padding_side(
    *,
    tokenizer: Any | None,
    processor: Any,
    padding_side: str | None,
):
    normalized = str(padding_side).strip().lower() if padding_side is not None else ""
    if not normalized:
        yield
        return
    if normalized not in {"left", "right"}:
        raise ValueError("padding_side must be 'left' or 'right'.")

    previous: list[tuple[Any, Any]] = []
    seen: set[int] = set()
    candidates = [tokenizer, getattr(processor, "tokenizer", None)]
    try:
        for candidate in candidates:
            if candidate is None or not hasattr(candidate, "padding_side"):
                continue
            candidate_id = id(candidate)
            if candidate_id in seen:
                continue
            seen.add(candidate_id)
            previous.append((candidate, getattr(candidate, "padding_side")))
            setattr(candidate, "padding_side", normalized)
        yield
    finally:
        for candidate, value in reversed(previous):
            setattr(candidate, "padding_side", value)


@dataclass(frozen=True)
class ModelCapabilities:
    is_multimodal: bool = True


@dataclass(frozen=True, slots=True)
class ProcessorInputPolicy:
    """Declare processor padding semantics for training and generation inputs."""

    training_padding_side: str = "right"
    generation_padding_side: str = "left"

    def __post_init__(self) -> None:
        for field_name in ("training_padding_side", "generation_padding_side"):
            value = str(getattr(self, field_name)).strip().lower()
            if value not in {"left", "right"}:
                raise ValueError(f"{field_name} must be 'left' or 'right'.")
            object.__setattr__(self, field_name, value)

    def resolve_padding_side(self, input_mode: str) -> str:
        normalized = str(input_mode).strip().lower()
        if normalized == "training":
            return self.training_padding_side
        if normalized == "generation":
            return self.generation_padding_side
        raise ValueError(
            f"Unsupported processor input_mode={input_mode!r}; expected 'training' or 'generation'."
        )


@dataclass(frozen=True, slots=True)
class ShaftProcessorCostEstimate:
    processed_image_tokens: int = 0
    vision_patches: int = 0
    exact: bool = False

    def __post_init__(self) -> None:
        if int(self.processed_image_tokens) < 0:
            raise ValueError("processed_image_tokens must be >= 0.")
        if int(self.vision_patches) < 0:
            raise ValueError("vision_patches must be >= 0.")


@dataclass(frozen=True)
class ModelModuleGroups:
    STRUCTURAL_GROUP_NAMES: ClassVar[tuple[str, ...]] = (
        "language_model",
        "vision_tower",
        "aligner",
        "generator",
    )

    language_model: tuple[str, ...] = ()
    vision_tower: tuple[str, ...] = ()
    aligner: tuple[str, ...] = ()
    generator: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for key in self.group_names():
            value = getattr(self, key)
            if isinstance(value, str):
                coerced = (value,) if value.strip() else ()
            else:
                coerced = _dedupe_non_empty(tuple(value))
            object.__setattr__(self, key, coerced)

    @classmethod
    def group_names(cls) -> tuple[str, ...]:
        """Return the public structural-group contract in stable order."""

        return cls.STRUCTURAL_GROUP_NAMES

    @property
    def has_structural_metadata(self) -> bool:
        return any(self.prefixes_for_group(name) for name in self.group_names())

    def prefixes_for_group(self, group_name: str) -> tuple[str, ...]:
        normalized = str(group_name).strip().lower()
        if normalized not in self.group_names():
            raise KeyError(f"Unknown model module group: {group_name!r}")
        return getattr(self, normalized)

    def resolve_group_for_name(self, name: str) -> str | None:
        normalized_name = str(name).strip()
        if not normalized_name:
            return None

        best_group: str | None = None
        best_prefix_len = -1
        for group_name in self.group_names():
            for prefix in self.prefixes_for_group(group_name):
                if not _matches_group_prefix(normalized_name, prefix):
                    continue
                prefix_len = len(prefix)
                if prefix_len > best_prefix_len:
                    best_group = group_name
                    best_prefix_len = prefix_len
        return best_group


@dataclass(frozen=True)
class ShaftProcessorTokenLayout:
    processed_boundaries: tuple[int, ...]
    protected_processed_spans: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        if not self.processed_boundaries or self.processed_boundaries[0] != 0:
            raise ValueError("processed_boundaries must start at 0.")
        if any(
            current <= previous
            for previous, current in zip(
                self.processed_boundaries,
                self.processed_boundaries[1:],
            )
        ):
            raise ValueError("processed_boundaries must be strictly increasing.")
        normalized_spans: list[tuple[int, int]] = []
        previous_stop = 0
        for raw_start, raw_stop in self.protected_processed_spans:
            start = int(raw_start)
            stop = int(raw_stop)
            if start < 0 or stop <= start or stop > self.processed_token_count:
                raise ValueError("Protected processor spans must lie inside the token layout.")
            if normalized_spans and start < previous_stop:
                raise ValueError("Protected processor spans must be ordered and non-overlapping.")
            normalized_spans.append((start, stop))
            previous_stop = stop
        object.__setattr__(self, "protected_processed_spans", tuple(normalized_spans))

    @property
    def rendered_token_count(self) -> int:
        return len(self.processed_boundaries) - 1

    @property
    def processed_token_count(self) -> int:
        return self.processed_boundaries[-1]

    def project_span(self, start: int, end: int) -> tuple[int, int]:
        if start < 0 or end <= start or end > self.rendered_token_count:
            raise ValueError(f"Invalid rendered token span: {(start, end)!r}.")
        return self.processed_boundaries[start], self.processed_boundaries[end]

    def intersects_protected_span(self, start: int, stop: int) -> bool:
        if start < 0 or stop <= start or stop > self.processed_token_count:
            raise ValueError(f"Invalid processed token span: {(start, stop)!r}.")
        return any(start < protected_stop and stop > protected_start for protected_start, protected_stop in self.protected_processed_spans)


@dataclass(frozen=True, slots=True)
class ShaftMediaSlice:
    start: int
    stop: int

    def __post_init__(self) -> None:
        start = int(self.start)
        stop = int(self.stop)
        if start < 0 or stop < start:
            raise ValueError("A media slice must satisfy 0 <= start <= stop.")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "stop", stop)

    @property
    def length(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True, slots=True)
class ShaftMediaSegmentManifest:
    processor_row_index: int
    image_grids: ShaftMediaSlice
    image_patches: ShaftMediaSlice

    def __post_init__(self) -> None:
        row_index = int(self.processor_row_index)
        if row_index < 0:
            raise ValueError("processor_row_index must be >= 0.")
        object.__setattr__(self, "processor_row_index", row_index)


@dataclass(frozen=True, slots=True)
class ShaftProcessorMediaManifest:
    segments: tuple[ShaftMediaSegmentManifest, ...]
    image_grid_count: int
    image_patch_count: int

    def __post_init__(self) -> None:
        image_grid_count = int(self.image_grid_count)
        image_patch_count = int(self.image_patch_count)
        if image_grid_count < 0 or image_patch_count < 0:
            raise ValueError("Processor media counts must be >= 0.")
        object.__setattr__(self, "segments", tuple(self.segments))
        object.__setattr__(self, "image_grid_count", image_grid_count)
        object.__setattr__(self, "image_patch_count", image_patch_count)

        expected_rows = tuple(range(len(self.segments)))
        actual_rows = tuple(segment.processor_row_index for segment in self.segments)
        if actual_rows != expected_rows:
            raise ValueError("Processor media manifest rows must be contiguous and ordered.")

        grid_cursor = 0
        patch_cursor = 0
        for segment in self.segments:
            if segment.image_grids.start != grid_cursor:
                raise ValueError("Processor image-grid slices must be contiguous.")
            if segment.image_patches.start != patch_cursor:
                raise ValueError("Processor image-patch slices must be contiguous.")
            grid_cursor = segment.image_grids.stop
            patch_cursor = segment.image_patches.stop
        if grid_cursor != image_grid_count:
            raise ValueError("Processor image-grid slices do not cover image_grid_count.")
        if patch_cursor != image_patch_count:
            raise ValueError("Processor image-patch slices do not cover image_patch_count.")


@dataclass(frozen=True, slots=True)
class ShaftProcessorSequenceField:
    """Model-owned layout for one processor output aligned to prompt tokens.

    ``sequence_axis`` is expressed on the batched processor tensor.  The common
    collation path never needs to know the field name or tensor rank.
    """

    name: str
    sequence_axis: int = 1
    padding_value: int | float | bool = 0
    continuation_value: int | float | bool | None = None

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("Processor sequence field names must not be empty.")
        if name in {
            "input_ids",
            "attention_mask",
            "labels",
            "loss_scale",
            "completion_mask",
        }:
            raise ValueError(f"Processor sequence field {name!r} is managed by the core.")
        sequence_axis = int(self.sequence_axis)
        if sequence_axis < 1:
            raise ValueError("Processor sequence_axis must follow the batch axis.")
        for field_name in ("padding_value", "continuation_value"):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, (bool, int, float)):
                raise TypeError(f"{field_name} must be a scalar bool/int/float when set.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "sequence_axis", sequence_axis)

    @property
    def row_sequence_axis(self) -> int:
        return self.sequence_axis - 1

    def extract_prompt_row(
        self,
        *,
        value: Any,
        row_index: int,
        attention_mask: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        if not torch.is_tensor(value) or value.ndim <= self.sequence_axis:
            raise ValueError(
                f"Processor sequence field {self.name!r} must be a batched tensor with "
                f"sequence_axis={self.sequence_axis}."
            )
        if int(value.shape[0]) != int(batch_size):
            raise ValueError(
                f"Processor sequence field {self.name!r} changed batch cardinality."
            )
        if int(value.shape[self.sequence_axis]) != int(attention_mask.shape[0]):
            raise ValueError(
                f"Processor sequence field {self.name!r} must align with attention_mask."
            )
        valid = torch.nonzero(attention_mask.bool(), as_tuple=False).flatten()
        row = value[int(row_index)]
        return row.index_select(self.row_sequence_axis, valid.to(device=row.device))

    def select_and_extend(
        self,
        row: torch.Tensor,
        *,
        prefix_indices: tuple[int, ...],
        continuation_length: int,
    ) -> torch.Tensor:
        continuation_length = int(continuation_length)
        if continuation_length < 0:
            raise ValueError("Processor sequence continuation_length must be >= 0.")
        indices = torch.tensor(prefix_indices, dtype=torch.long, device=row.device)
        selected = row.index_select(self.row_sequence_axis, indices)
        if continuation_length == 0:
            return selected
        if self.continuation_value is None:
            raise ValueError(
                f"Processor sequence field {self.name!r} requires an explicit continuation rule."
            )
        extension_shape = list(selected.shape)
        extension_shape[self.row_sequence_axis] = continuation_length
        extension = torch.full(
            extension_shape,
            self.continuation_value,
            dtype=selected.dtype,
            device=selected.device,
        )
        return torch.cat((selected, extension), dim=self.row_sequence_axis)

    def extend_batch(self, value: torch.Tensor, *, continuation_length: int) -> torch.Tensor:
        if not torch.is_tensor(value) or value.ndim <= self.sequence_axis:
            raise ValueError(
                f"Processor sequence field {self.name!r} has an invalid batched tensor."
            )
        continuation_length = int(continuation_length)
        if continuation_length <= 0:
            raise ValueError("Processor sequence continuation_length must be positive.")
        if self.continuation_value is None:
            raise ValueError(
                f"Processor sequence field {self.name!r} requires an explicit continuation rule."
            )
        extension_shape = list(value.shape)
        extension_shape[self.sequence_axis] = continuation_length
        extension = torch.full(
            extension_shape,
            self.continuation_value,
            dtype=value.dtype,
            device=value.device,
        )
        return torch.cat((value, extension), dim=self.sequence_axis)

    def collate_rows(
        self,
        rows: list[torch.Tensor],
        *,
        layout: str,
        padding_side: str,
    ) -> torch.Tensor:
        if not rows:
            raise ValueError(f"Processor sequence field {self.name!r} has no rows.")
        reference_shape = list(rows[0].shape)
        axis = self.row_sequence_axis
        if len(reference_shape) <= axis:
            raise ValueError(f"Processor sequence field {self.name!r} row rank is invalid.")
        for row in rows:
            if not torch.is_tensor(row) or row.ndim != len(reference_shape):
                raise ValueError(
                    f"Processor sequence field {self.name!r} rows must share one tensor rank."
                )
            shape = list(row.shape)
            if any(
                shape[index] != reference_shape[index]
                for index in range(len(shape))
                if index != axis
            ):
                raise ValueError(
                    f"Processor sequence field {self.name!r} rows changed non-sequence axes."
                )
        normalized_layout = str(layout).strip().lower()
        if normalized_layout == "varlen":
            return torch.cat(rows, dim=axis).unsqueeze(0)
        if normalized_layout != "padded":
            raise ValueError(f"Unsupported processor sequence layout: {layout!r}.")
        normalized_padding_side = str(padding_side).strip().lower()
        if normalized_padding_side not in {"left", "right"}:
            raise ValueError("padding_side must be 'left' or 'right'.")
        max_length = max(int(row.shape[axis]) for row in rows)
        padded: list[torch.Tensor] = []
        for row in rows:
            missing = max_length - int(row.shape[axis])
            if missing <= 0:
                padded.append(row)
                continue
            pad_shape = list(row.shape)
            pad_shape[axis] = missing
            pad = torch.full(
                pad_shape,
                self.padding_value,
                dtype=row.dtype,
                device=row.device,
            )
            parts = (pad, row) if normalized_padding_side == "left" else (row, pad)
            padded.append(torch.cat(parts, dim=axis))
        return torch.stack(padded, dim=0)


@dataclass(frozen=True)
class ShaftProcessedBatch:
    model_inputs: dict[str, Any]
    batch_size: int
    media_manifest: ShaftProcessorMediaManifest | None = None
    processor_sequence_fields: tuple[ShaftProcessorSequenceField, ...] = ()

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("ShaftProcessedBatch.batch_size must be positive.")
        if "input_ids" not in self.model_inputs or "attention_mask" not in self.model_inputs:
            raise ValueError("ShaftProcessedBatch requires processor input_ids and attention_mask.")
        for key in ("input_ids", "attention_mask"):
            value = self.model_inputs[key]
            if not torch.is_tensor(value) or value.ndim < 2:
                raise ValueError(f"Processor output {key!r} must be a batched tensor.")
            if int(value.shape[0]) != self.batch_size:
                raise ValueError(f"Processor output {key!r} batch axis does not match batch_size.")
        if self.media_manifest is not None:
            if len(self.media_manifest.segments) != self.batch_size:
                raise ValueError(
                    "Processor media manifest must contain one segment per processor row."
                )
        fields = tuple(self.processor_sequence_fields)
        names = tuple(field.name for field in fields)
        if len(names) != len(set(names)):
            raise ValueError("Processor sequence field names must be unique.")
        attention_mask = self.model_inputs["attention_mask"]
        for sequence_field in fields:
            if sequence_field.name not in self.model_inputs:
                raise ValueError(
                    f"Declared processor sequence field {sequence_field.name!r} is missing."
                )
            for row_index in range(self.batch_size):
                sequence_field.extract_prompt_row(
                    value=self.model_inputs[sequence_field.name],
                    row_index=row_index,
                    attention_mask=attention_mask[row_index],
                    batch_size=self.batch_size,
                )
        object.__setattr__(self, "processor_sequence_fields", fields)

    def build_processor_sequence_row(
        self,
        *,
        row_index: int,
        prefix_indices: tuple[int, ...],
        continuation_length: int,
    ) -> dict[str, torch.Tensor]:
        if row_index < 0 or row_index >= self.batch_size:
            raise ValueError("Processor sequence row_index is out of range.")
        attention_mask = self.model_inputs["attention_mask"][row_index]
        prompt_length = int(attention_mask.sum().item())
        if any(index < 0 or index >= prompt_length for index in prefix_indices):
            raise ValueError("Processor sequence prefix_indices are out of range.")
        output: dict[str, torch.Tensor] = {}
        for sequence_field in self.processor_sequence_fields:
            prompt_row = sequence_field.extract_prompt_row(
                value=self.model_inputs[sequence_field.name],
                row_index=row_index,
                attention_mask=attention_mask,
                batch_size=self.batch_size,
            )
            output[sequence_field.name] = sequence_field.select_and_extend(
                prompt_row,
                prefix_indices=prefix_indices,
                continuation_length=continuation_length,
            )
        return output

    def collate_processor_sequence_rows(
        self,
        rows: list[dict[str, torch.Tensor]],
        *,
        layout: str,
        padding_side: str,
    ) -> dict[str, torch.Tensor]:
        if not self.processor_sequence_fields:
            if any(rows):
                raise ValueError("Processor sequence rows were provided without field declarations.")
            return {}
        expected_names = {field.name for field in self.processor_sequence_fields}
        for row in rows:
            if set(row) != expected_names:
                raise ValueError(
                    "Processor sequence rows must contain every declared field exactly once."
                )
        return {
            field.name: field.collate_rows(
                [row[field.name] for row in rows],
                layout=layout,
                padding_side=padding_side,
            )
            for field in self.processor_sequence_fields
        }


@dataclass(frozen=True)
class ShaftRolloutScoringPlan:
    """Model inputs plus the exact full-sequence span represented by output logits."""

    model_inputs: dict[str, Any]
    sequence_length: int
    logit_start: int
    logit_count: int

    def __post_init__(self) -> None:
        sequence_length = int(self.sequence_length)
        logit_start = int(self.logit_start)
        logit_count = int(self.logit_count)
        if sequence_length <= 1:
            raise ValueError("Rollout scoring requires a sequence with at least two tokens.")
        if logit_start < 0 or logit_count <= 1:
            raise ValueError(
                "Rollout scoring logit span must be non-empty and causally shiftable."
            )
        if logit_start + logit_count != sequence_length:
            raise ValueError("Rollout scoring logits must represent an exact sequence tail.")
        object.__setattr__(self, "sequence_length", sequence_length)
        object.__setattr__(self, "logit_start", logit_start)
        object.__setattr__(self, "logit_count", logit_count)

    def align_completion_mask(
        self,
        completion_mask: torch.Tensor,
        *,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        if not torch.is_tensor(completion_mask) or completion_mask.ndim != 2:
            raise TypeError("Rollout completion_mask must be a 2-D tensor.")
        if int(completion_mask.shape[1]) != self.sequence_length:
            raise ValueError("Rollout completion_mask does not match the full scoring sequence.")
        if not torch.is_tensor(logits) or logits.ndim != 3:
            raise TypeError(
                "Rollout model logits must have shape [batch, sequence, vocabulary]."
            )
        if int(logits.shape[0]) != int(completion_mask.shape[0]):
            raise ValueError("Rollout model logits changed batch cardinality.")
        if int(logits.shape[1]) != self.logit_count:
            raise ValueError(
                "Model did not honor its declared rollout logit-tail contract: "
                f"expected={self.logit_count}, actual={int(logits.shape[1])}."
            )
        return completion_mask[:, self.logit_start :]


@dataclass(frozen=True)
class ProcessorPolicy:
    supports_pixel_budget: bool = False
    supports_exact_image_cost: bool = False
    input_policy: ProcessorInputPolicy = field(default_factory=ProcessorInputPolicy)
    sample_aligned_model_input_names: tuple[str, ...] = ()
    whole_batch_model_input_names: tuple[str, ...] = ()
    static_model_input_names: tuple[str, ...] = ()
    processor_sequence_fields: tuple[ShaftProcessorSequenceField, ...] = ()
    rollout_tail_logits_input_name: str | None = None

    def prepare_rollout_image(
        self,
        image: Any,
        *,
        min_pixels: int | None,
        max_pixels: int | None,
    ) -> Any:
        """Prepare one rollout image when an upstream trainer cannot forward budgets."""

        _ = min_pixels, max_pixels
        return image

    def __post_init__(self) -> None:
        field_names = (
            "sample_aligned_model_input_names",
            "whole_batch_model_input_names",
            "static_model_input_names",
        )
        for field_name in field_names:
            object.__setattr__(self, field_name, _dedupe_non_empty(getattr(self, field_name)))
        processor_sequence_fields = tuple(self.processor_sequence_fields)
        sequence_names = tuple(field.name for field in processor_sequence_fields)
        if len(sequence_names) != len(set(sequence_names)):
            raise ValueError("Processor sequence field names must be unique.")
        if not all(
            isinstance(field, ShaftProcessorSequenceField)
            for field in processor_sequence_fields
        ):
            raise TypeError(
                "processor_sequence_fields must contain ShaftProcessorSequenceField values."
            )
        object.__setattr__(
            self,
            "processor_sequence_fields",
            processor_sequence_fields,
        )
        tail_logits_input_name = (
            None
            if self.rollout_tail_logits_input_name is None
            else str(self.rollout_tail_logits_input_name).strip()
        )
        if tail_logits_input_name == "":
            raise ValueError("rollout_tail_logits_input_name must be non-empty when set.")
        object.__setattr__(
            self,
            "rollout_tail_logits_input_name",
            tail_logits_input_name,
        )

        declared_layouts: dict[str, str] = {}
        for field_name in field_names:
            for input_name in getattr(self, field_name):
                previous = declared_layouts.setdefault(input_name, field_name)
                if previous != field_name:
                    raise ValueError(
                        f"Processor model input {input_name!r} is declared by both "
                        f"{previous!r} and {field_name!r}."
                    )
        overlap = sorted(set(sequence_names) & declared_layouts.keys())
        if overlap:
            raise ValueError(
                "Processor model inputs cannot be both sequence-aligned and non-sequence: "
                f"{overlap}."
            )

    @property
    def assembled_sequence_input_names(self) -> tuple[str, ...]:
        return (*_CORE_SEQUENCE_INPUT_NAMES, *(field.name for field in self.processor_sequence_fields))

    def build_rollout_scoring_plan(
        self,
        *,
        prompt_inputs: dict[str, Any],
        sequences: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> ShaftRolloutScoringPlan:
        """Extend model-family sequence fields from a prompt to its rollout."""

        prompt_ids = prompt_inputs.get("input_ids")
        prompt_mask = prompt_inputs.get("attention_mask")
        for name, value in (
            ("input_ids", prompt_ids),
            ("attention_mask", prompt_mask),
            ("sequences", sequences),
        ):
            if not torch.is_tensor(value) or value.ndim != 2:
                raise TypeError(f"OPD rollout {name} must be a 2-D tensor.")
        assert torch.is_tensor(prompt_ids)
        assert torch.is_tensor(prompt_mask)
        if tuple(prompt_ids.shape) != tuple(prompt_mask.shape):
            raise ValueError("OPD prompt input_ids and attention_mask shapes must match.")
        if int(sequences.shape[0]) != int(prompt_ids.shape[0]):
            raise ValueError("OPD scoring sequences changed prompt batch cardinality.")
        completion_width = int(sequences.shape[1]) - int(prompt_ids.shape[1])
        if completion_width <= 0:
            raise ValueError("OPD scoring inputs require a non-empty completion.")
        if tuple(attention_mask.shape) != tuple(sequences.shape):
            raise ValueError("OPD full attention_mask must align with scoring sequences.")

        sequence_fields = {field.name: field for field in self.processor_sequence_fields}
        declared_non_sequence = {
            *self.sample_aligned_model_input_names,
            *self.whole_batch_model_input_names,
            *self.static_model_input_names,
        }
        output: dict[str, Any] = {}
        for name, value in prompt_inputs.items():
            if name in _CORE_SEQUENCE_INPUT_NAMES:
                continue
            sequence_field = sequence_fields.get(name)
            if sequence_field is None:
                if name not in declared_non_sequence and name not in {"use_cache"}:
                    raise ValueError(
                        f"Processor policy does not declare the layout of model input {name!r}."
                    )
                output[name] = value
                continue
            if not torch.is_tensor(value) or value.ndim <= sequence_field.sequence_axis:
                raise TypeError(f"OPD rollout sequence input {name!r} has an invalid tensor rank.")
            if int(value.shape[0]) != int(prompt_ids.shape[0]) or int(
                value.shape[sequence_field.sequence_axis]
            ) != int(prompt_ids.shape[1]):
                raise ValueError(f"OPD rollout sequence input {name!r} must align with the prompt.")
            output[name] = sequence_field.extend_batch(
                value,
                continuation_length=completion_width,
            )
        output.update(
            {
                "input_ids": sequences,
                "attention_mask": attention_mask,
                "use_cache": False,
            }
        )
        sequence_length = int(sequences.shape[1])
        logit_count = sequence_length
        if self.rollout_tail_logits_input_name is not None:
            logit_count = completion_width + 1
            output[self.rollout_tail_logits_input_name] = logit_count
        return ShaftRolloutScoringPlan(
            model_inputs=output,
            sequence_length=sequence_length,
            logit_start=sequence_length - logit_count,
            logit_count=logit_count,
        )

    def build_batch(
        self,
        *,
        processor: Any,
        tokenizer: Any | None,
        prompt_texts: list[str],
        images: list[Any],
        min_pixels: int | None,
        max_pixels: int | None,
        input_mode: str = "training",
    ) -> ShaftProcessedBatch:
        if not prompt_texts:
            raise ValueError("Processor batches must contain at least one prompt row.")
        if len(images) != len(prompt_texts):
            raise ValueError(
                "Processor images must use outer=batch-row semantics: "
                f"prompts={len(prompt_texts)}, image_rows={len(images)}."
            )
        for row_index, row_images in enumerate(images):
            if isinstance(row_images, (list, tuple)) and not row_images:
                raise ValueError(
                    f"Processor image row {row_index} must contain at least one image."
                )
        kwargs: dict[str, Any] = {
            "text": prompt_texts,
            "images": images,
            "padding": True,
            "return_tensors": "pt",
        }
        if self.supports_pixel_budget:
            images_kwargs: dict[str, Any] = {}
            if min_pixels is not None:
                images_kwargs["min_pixels"] = int(min_pixels)
            if max_pixels is not None:
                images_kwargs["max_pixels"] = int(max_pixels)
            if images_kwargs:
                kwargs["images_kwargs"] = images_kwargs
        with _temporary_processor_padding_side(
            tokenizer=tokenizer,
            processor=processor,
            padding_side=self.input_policy.resolve_padding_side(input_mode),
        ):
            outputs = processor(**kwargs)
        model_inputs = dict(outputs)
        return ShaftProcessedBatch(
            model_inputs=model_inputs,
            batch_size=len(prompt_texts),
            processor_sequence_fields=tuple(
                field
                for field in self.processor_sequence_fields
                if field.name in model_inputs
            ),
        )

    def estimate_image_cost(
        self,
        *,
        processor: Any,
        image_sizes: tuple[tuple[int, int], ...],
        min_pixels: int | None,
        max_pixels: int | None,
    ) -> ShaftProcessorCostEstimate:
        _ = processor, image_sizes, min_pixels, max_pixels
        raise ValueError(
            f"Processor policy {type(self).__name__!r} does not provide an exact image-cost "
            "estimator; register a model-specific processor policy before enabling "
            "cost-aware batching."
        )

    def cost_semantics_signature(
        self,
        *,
        processor: Any,
        min_pixels: int | None,
        max_pixels: int | None,
    ) -> tuple[object, ...]:
        _ = processor, min_pixels, max_pixels
        raise ValueError(
            f"Processor policy {type(self).__name__!r} must provide a versioned "
            "cost_semantics_signature before enabling cost-aware batching. The "
            "signature must bind every processor field used by its exact estimator."
        )

    def estimate_token_layout(
        self,
        *,
        processor: Any,
        tokenizer: Any,
        rendered_token_ids: tuple[int, ...],
        image_costs: tuple[ShaftProcessorCostEstimate, ...],
    ) -> ShaftProcessorTokenLayout:
        _ = processor, tokenizer
        if image_costs:
            raise ValueError(
                f"Processor policy {type(self).__name__!r} does not provide an exact "
                "multimodal token-layout estimator."
            )
        return ShaftProcessorTokenLayout(
            processed_boundaries=tuple(range(len(rendered_token_ids) + 1))
        )

    def build_token_layout(
        self,
        *,
        rendered_token_ids: tuple[int, ...],
        processed_batch: ShaftProcessedBatch,
        row_index: int,
    ) -> ShaftProcessorTokenLayout:
        processed_token_ids, _ = self._extract_token_row(
            processed_batch=processed_batch,
            row_index=row_index,
        )
        token_ids = [int(value) for value in processed_token_ids.tolist()]
        return self._finalize_token_layout(
            rendered_token_ids=rendered_token_ids,
            canonical_token_ids=token_ids,
            processed_boundaries=tuple(range(len(token_ids) + 1)),
            processed_token_count=len(token_ids),
        )

    @staticmethod
    def _extract_token_row(
        *,
        processed_batch: ShaftProcessedBatch,
        row_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            attention_mask = processed_batch.model_inputs["attention_mask"][row_index].bool()
            token_ids = processed_batch.model_inputs["input_ids"][row_index][attention_mask]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(
                "Processor outputs must provide aligned batched input_ids and attention_mask."
            ) from exc
        return token_ids, attention_mask

    def assemble_training_inputs(
        self,
        *,
        processed_batch: ShaftProcessedBatch,
        sequence_inputs: dict[str, Any],
        row_indices: tuple[int, ...],
    ) -> dict[str, Any]:
        if not row_indices:
            raise ValueError("row_indices must not be empty.")
        if any(index < 0 or index >= processed_batch.batch_size for index in row_indices):
            raise ValueError("row_indices contains an out-of-range processor batch row.")
        policy_sequence_names = {field.name for field in self.processor_sequence_fields}
        batch_sequence_names = {
            field.name for field in processed_batch.processor_sequence_fields
        }
        raw_sequence_names = policy_sequence_names & processed_batch.model_inputs.keys()
        if raw_sequence_names != batch_sequence_names:
            raise ValueError(
                "Processor sequence outputs must be carried by the processed-batch sequence "
                "contract."
            )
        missing_sequence_inputs = sorted(batch_sequence_names - sequence_inputs.keys())
        if missing_sequence_inputs:
            raise ValueError(
                "Collation omitted processor sequence fields declared by the processed batch: "
                f"{missing_sequence_inputs}."
            )
        unknown_sequence_inputs = sorted(
            set(sequence_inputs) - set(_CORE_SEQUENCE_INPUT_NAMES) - batch_sequence_names
        )
        if unknown_sequence_inputs:
            raise ValueError(
                "Collation produced sequence fields outside the processor policy: "
                f"{unknown_sequence_inputs}."
            )
        assembled: dict[str, Any] = {}
        for key, value in processed_batch.model_inputs.items():
            if key in self.assembled_sequence_input_names or key in sequence_inputs:
                continue
            assembled[key] = self._select_model_input_rows(
                name=key,
                value=value,
                batch_size=processed_batch.batch_size,
                row_indices=row_indices,
            )
        assembled.update(sequence_inputs)
        return assembled

    def _select_model_input_rows(
        self,
        *,
        name: str,
        value: Any,
        batch_size: int,
        row_indices: tuple[int, ...],
    ) -> Any:
        identity_rows = tuple(range(batch_size))
        if name in self.static_model_input_names:
            return value

        if name in self.whole_batch_model_input_names:
            if row_indices == identity_rows:
                self._validate_whole_batch_model_input(name=name, value=value)
                return value
            if len(row_indices) % batch_size == 0:
                repeats = len(row_indices) // batch_size
                if row_indices == identity_rows * repeats:
                    return self._repeat_whole_batch_model_input(
                        name=name,
                        value=value,
                        repeats=repeats,
                    )
            raise ValueError(
                f"Processor policy cannot select rows from whole-batch model input {name!r}."
            )

        if name not in self.sample_aligned_model_input_names:
            raise ValueError(
                f"Processor policy does not declare the layout of model input {name!r}."
            )
        if torch.is_tensor(value) and value.ndim > 0 and int(value.shape[0]) == batch_size:
            index = torch.tensor(row_indices, dtype=torch.long, device=value.device)
            return value.index_select(0, index)
        if isinstance(value, list) and len(value) == batch_size:
            return [value[index] for index in row_indices]
        if isinstance(value, tuple) and len(value) == batch_size:
            return tuple(value[index] for index in row_indices)
        raise ValueError(
            f"Processor policy cannot select rows from model input {name!r}; register a "
            "model-specific policy."
        )

    @staticmethod
    def _validate_whole_batch_model_input(*, name: str, value: Any) -> None:
        if value is None or isinstance(value, (list, tuple)):
            return
        if torch.is_tensor(value) and value.ndim > 0:
            return
        raise ValueError(
            f"Processor whole-batch model input {name!r} has unsupported type or shape."
        )

    @staticmethod
    def _repeat_whole_batch_model_input(
        *,
        name: str,
        value: Any,
        repeats: int,
    ) -> Any:
        ProcessorPolicy._validate_whole_batch_model_input(name=name, value=value)
        if torch.is_tensor(value) and value.ndim > 0:
            return torch.cat([value] * repeats, dim=0)
        if isinstance(value, list):
            return value * repeats
        if isinstance(value, tuple):
            return value * repeats
        if value is None:
            return None
        raise ValueError(
            f"Processor policy cannot repeat whole-batch model input {name!r} of type "
            f"{type(value).__name__}."
        )

    def _finalize_token_layout(
        self,
        *,
        rendered_token_ids: tuple[int, ...],
        canonical_token_ids: list[int],
        processed_boundaries: tuple[int, ...],
        processed_token_count: int,
        protected_processed_spans: tuple[tuple[int, int], ...] = (),
    ) -> ShaftProcessorTokenLayout:
        if tuple(canonical_token_ids) != rendered_token_ids:
            raise ValueError(
                "Processor token layout cannot align its output exactly with the rendered prompt "
                f"tokens under policy {type(self).__name__!r}; register a model-specific "
                "processor policy."
            )
        if len(processed_boundaries) != len(canonical_token_ids) + 1:
            raise ValueError(
                "Processor token layout must contain one boundary per canonical token plus the "
                "initial boundary."
            )
        if processed_boundaries[-1] != int(processed_token_count):
            raise ValueError("Processor token layout does not cover the full processed token row.")
        return ShaftProcessorTokenLayout(
            processed_boundaries,
            protected_processed_spans=protected_processed_spans,
        )


@dataclass(frozen=True, slots=True)
class ShaftSequenceExecutionContract:
    """Immutable model-owned sequence execution request and environment signature."""

    layout: str
    device_type: str
    attention_implementation: str | None
    torch_dtype: str
    distributed_strategy: str
    torch_compile: bool
    capability_signature: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "layout", str(self.layout).strip().lower())
        object.__setattr__(self, "device_type", str(self.device_type).strip().lower())
        attention = str(self.attention_implementation or "").strip().lower()
        object.__setattr__(
            self,
            "attention_implementation",
            attention or None,
        )
        object.__setattr__(self, "torch_dtype", str(self.torch_dtype).strip().lower())
        object.__setattr__(
            self,
            "distributed_strategy",
            str(self.distributed_strategy).strip().lower(),
        )
        object.__setattr__(self, "torch_compile", bool(self.torch_compile))
        object.__setattr__(
            self,
            "capability_signature",
            _dedupe_non_empty(self.capability_signature),
        )
        if self.layout not in {"padded", "varlen"}:
            raise ValueError(f"Unsupported sequence layout: {self.layout!r}.")
        if not self.device_type or not self.torch_dtype or not self.distributed_strategy:
            raise ValueError("Sequence execution contract fields must not be empty.")
        if not self.capability_signature:
            raise ValueError("Sequence execution capability_signature must not be empty.")

    @property
    def fingerprint(self) -> str:
        payload = (
            "shaft-sequence-execution-contract-v1",
            self.layout,
            self.device_type,
            self.attention_implementation,
            self.torch_dtype,
            self.distributed_strategy,
            self.torch_compile,
            self.capability_signature,
        )
        return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


class SequenceExecutionPolicy:
    """Model-owned conversion from a collated layout into forward inputs."""

    def build_contract(
        self,
        *,
        layout: str,
        device_type: str,
        attention_implementation: str | None,
        torch_dtype: str,
        distributed_strategy: str,
        torch_compile: bool = False,
    ) -> ShaftSequenceExecutionContract:
        normalized_layout = str(layout).strip().lower()
        if normalized_layout == "varlen":
            raise ValueError(
                f"Sequence policy {type(self).__name__!r} does not support varlen layout."
            )
        return ShaftSequenceExecutionContract(
            layout=normalized_layout,
            device_type=device_type,
            attention_implementation=attention_implementation,
            torch_dtype=torch_dtype,
            distributed_strategy=distributed_strategy,
            torch_compile=torch_compile,
            capability_signature=(f"{type(self).__module__}.{type(self).__qualname__}",),
        )

    def validate_runtime(
        self,
        *,
        model: Any,
        contract: ShaftSequenceExecutionContract,
    ) -> None:
        _ = model
        expected_signature = (f"{type(self).__module__}.{type(self).__qualname__}",)
        if contract.capability_signature != expected_signature:
            raise ValueError("Sequence execution contract belongs to another policy.")
        if contract.layout == "varlen":
            raise ValueError(
                f"Sequence policy {type(self).__name__!r} does not support varlen layout."
            )

    def configure_runtime(
        self,
        *,
        model: Any,
        contract: ShaftSequenceExecutionContract,
    ) -> None:
        """Install model-owned runtime adapters before validation.

        Most families need no mutation.  A family may use this hook for a
        narrowly versioned upstream compatibility adapter, while collators and
        trainers remain model-agnostic.
        """

        _ = model, contract

    def prepare_training_inputs(
        self,
        *,
        model: Any,
        inputs: dict[str, Any],
    ) -> dict[str, Any]:
        _ = model
        if "_shaft_varlen_layout" in inputs:
            raise ValueError(
                f"Sequence policy {type(self).__name__!r} cannot execute varlen inputs."
            )
        return dict(inputs)


class PeftPolicy(ABC):
    @abstractmethod
    def default_target_modules(self) -> list[str]:
        raise NotImplementedError

    def resolve_target_modules(self, target_modules: list[str]) -> list[str]:
        normalized = [str(item).strip() for item in target_modules if str(item).strip()]
        resolved: list[str] = []
        for value in normalized:
            if value == "auto":
                resolved.extend(self.default_target_modules())
            else:
                resolved.append(value)
        return list(dict.fromkeys(resolved))

    def default_target_parameters(self) -> list[str]:
        return []

    def resolve_target_parameters(self, target_parameters: list[str]) -> list[str]:
        normalized = [str(item).strip() for item in target_parameters if str(item).strip()]
        resolved: list[str] = []
        for value in normalized:
            if value == "auto":
                defaults = self.default_target_parameters()
                if not defaults:
                    raise ValueError(
                        "model.finetune.target_parameters=['auto'] is not available "
                        "for this model profile; configure explicit parameter suffixes "
                        "or leave target_parameters empty."
                    )
                resolved.extend(defaults)
            else:
                resolved.append(value)
        return list(dict.fromkeys(resolved))

    def validate_finetune_config(
        self,
        finetune: Any,
        *,
        model_descriptor: Any | None = None,
        model_name_or_path: str | None = None,
    ) -> None:
        _ = model_descriptor, model_name_or_path
        requested_parameters = [
            str(value).strip()
            for value in getattr(finetune, "target_parameters", ())
            if str(value).strip()
        ]
        if not requested_parameters:
            return
        mode = str(finetune.mode).strip().lower()
        if mode == "full":
            raise ValueError(
                "model.finetune.target_parameters applies only to adapter finetuning; "
                "remove it when mode='full'."
            )
        resolved_parameters = self.resolve_target_parameters(requested_parameters)
        if not resolved_parameters:
            raise ValueError(
                "model.finetune.target_parameters resolved to no trainable parameters."
            )
        if mode == "dora":
            raise ValueError("PEFT target_parameters do not support DoRA.")
        if float(finetune.lora_dropout) != 0.0:
            raise ValueError("PEFT target_parameters require lora_dropout=0.")

    def validate_training_finetune_config(
        self,
        finetune: Any,
        *,
        model_descriptor: Any | None = None,
        model_name_or_path: str | None = None,
    ) -> None:
        self.validate_finetune_config(
            finetune,
            model_descriptor=model_descriptor,
            model_name_or_path=model_name_or_path,
        )


@dataclass(frozen=True)
class DefaultPeftPolicy(PeftPolicy):
    target_modules: list[str]
    target_parameters: list[str] = field(default_factory=list)

    def default_target_modules(self) -> list[str]:
        return list(self.target_modules)

    def default_target_parameters(self) -> list[str]:
        return list(self.target_parameters)


@dataclass(frozen=True, slots=True)
class ShaftAuxiliaryLossTerm:
    """One model-owned objective term added beside Shaft's normalized SFT CE."""

    name: str
    value: torch.Tensor
    coefficient: float
    reduction: str = "optimizer_frame_mean"

    def __post_init__(self) -> None:
        import math

        normalized_name = str(self.name).strip()
        normalized_reduction = str(self.reduction).strip().lower()
        coefficient = float(self.coefficient)
        if not normalized_name:
            raise ValueError("Auxiliary loss term name must not be empty.")
        if (
            not isinstance(self.name, str)
            or self.name != normalized_name
            or self.name != self.name.lower()
        ):
            raise ValueError("Auxiliary loss term name must be a canonical lowercase string.")
        if not torch.is_tensor(self.value) or self.value.numel() != 1:
            raise ValueError("Auxiliary loss term value must be a scalar tensor.")
        if not math.isfinite(coefficient) or coefficient < 0.0:
            raise ValueError("Auxiliary loss term coefficient must be finite and >= 0.")
        if normalized_reduction != "optimizer_frame_mean":
            raise ValueError(
                "Unsupported auxiliary loss reduction; expected 'optimizer_frame_mean'."
            )
        object.__setattr__(self, "name", normalized_name)
        object.__setattr__(self, "coefficient", coefficient)
        object.__setattr__(self, "reduction", normalized_reduction)


@dataclass(frozen=True, slots=True)
class ShaftEvalAuxiliaryStatistic:
    """Additive, batch-first statistics for a model-owned eval metric."""

    name: str
    coefficient_key: str
    coefficient: float
    components: Mapping[str, torch.Tensor]

    def __post_init__(self) -> None:
        import math

        normalized_name = str(self.name).strip()
        normalized_coefficient_key = str(self.coefficient_key).strip()
        coefficient = float(self.coefficient)
        if not normalized_name:
            raise ValueError("Eval auxiliary statistic name must not be empty.")
        if not normalized_coefficient_key:
            raise ValueError("Eval auxiliary statistic coefficient_key must not be empty.")
        if (
            not isinstance(self.coefficient_key, str)
            or self.coefficient_key != normalized_coefficient_key
            or self.coefficient_key != self.coefficient_key.lower()
        ):
            raise ValueError(
                "Eval auxiliary statistic coefficient_key must be a canonical lowercase string."
            )
        if not math.isfinite(coefficient) or coefficient < 0.0:
            raise ValueError("Eval auxiliary statistic coefficient must be finite and >= 0.")
        normalized_components: dict[str, torch.Tensor] = {}
        batch_size: int | None = None
        for component_name, value in self.components.items():
            normalized_component_name = str(component_name).strip()
            if not normalized_component_name:
                raise ValueError("Eval auxiliary statistic component names must not be empty.")
            if not torch.is_tensor(value) or value.ndim < 1:
                raise ValueError("Eval auxiliary statistic components must be batch-first tensors.")
            current_batch_size = int(value.shape[0])
            if batch_size is None:
                batch_size = current_batch_size
            elif current_batch_size != batch_size:
                raise ValueError("Eval auxiliary statistic components must share one batch size.")
            normalized_components[normalized_component_name] = value
        if not normalized_components:
            raise ValueError("Eval auxiliary statistic must contain at least one component.")
        object.__setattr__(self, "name", normalized_name)
        object.__setattr__(self, "coefficient_key", normalized_coefficient_key)
        object.__setattr__(self, "coefficient", coefficient)
        object.__setattr__(self, "components", normalized_components)


@dataclass(frozen=True, slots=True)
class ShaftEvalAuxiliaryMetric:
    """A dataset-global auxiliary metric finalized from additive statistics."""

    name: str
    value: torch.Tensor
    coefficient_key: str
    coefficient: float

    def __post_init__(self) -> None:
        import math

        normalized_name = str(self.name).strip()
        normalized_coefficient_key = str(self.coefficient_key).strip()
        coefficient = float(self.coefficient)
        if not normalized_name:
            raise ValueError("Eval auxiliary metric name must not be empty.")
        if not normalized_coefficient_key:
            raise ValueError("Eval auxiliary metric coefficient_key must not be empty.")
        if (
            not isinstance(self.coefficient_key, str)
            or self.coefficient_key != normalized_coefficient_key
            or self.coefficient_key != self.coefficient_key.lower()
        ):
            raise ValueError(
                "Eval auxiliary metric coefficient_key must be a canonical lowercase string."
            )
        if not torch.is_tensor(self.value) or self.value.numel() != 1:
            raise ValueError("Eval auxiliary metric value must be a scalar tensor.")
        if not math.isfinite(coefficient) or coefficient < 0.0:
            raise ValueError("Eval auxiliary metric coefficient must be finite and >= 0.")
        object.__setattr__(self, "name", normalized_name)
        object.__setattr__(self, "coefficient_key", normalized_coefficient_key)
        object.__setattr__(self, "coefficient", coefficient)


class TrainingObjectivePolicy:
    """Model-owned additions to the shared SFT next-token objective."""

    def auxiliary_loss_names(self) -> tuple[str, ...]:
        """Return stable names that may be overridden by algorithm config."""

        return ()

    def prepare_sft_forward_inputs(
        self,
        *,
        model: Any,
        inputs: dict[str, Any],
    ) -> dict[str, Any]:
        _ = model
        return dict(inputs)

    def resolve_sft_auxiliary_loss_terms(
        self,
        *,
        model: Any,
        outputs: Any,
        inputs: dict[str, Any],
    ) -> tuple[ShaftAuxiliaryLossTerm, ...]:
        _ = model, outputs, inputs
        return ()

    def resolve_sft_eval_auxiliary_statistics(
        self,
        *,
        model: Any,
        outputs: Any,
        inputs: dict[str, Any],
    ) -> tuple[ShaftEvalAuxiliaryStatistic, ...]:
        _ = model, outputs, inputs
        return ()

    def finalize_sft_eval_auxiliary_statistics(
        self,
        statistics: tuple[ShaftEvalAuxiliaryStatistic, ...],
    ) -> tuple[ShaftEvalAuxiliaryMetric, ...]:
        if statistics:
            raise ValueError(
                f"{type(self).__name__} does not implement eval auxiliary statistic finalization."
            )
        return ()


def validate_auxiliary_weight_names(
    model_adapter: Any,
    weights: Mapping[str, float],
) -> tuple[str, ...]:
    """Validate run overrides against one model policy's declared term names."""

    if model_adapter is None:
        if weights:
            raise ValueError("SFT auxiliary loss weights require a resolved model adapter.")
        return ()
    names_resolver = getattr(model_adapter, "auxiliary_loss_names", None)
    if not callable(names_resolver):
        if weights:
            raise TypeError("The resolved model adapter does not declare auxiliary_loss_names().")
        return ()
    raw_names = tuple(names_resolver())
    if any(
        not isinstance(name, str) or not name or name != name.strip() or name != name.lower()
        for name in raw_names
    ) or len(set(raw_names)) != len(raw_names):
        raise ValueError(
            "The resolved model adapter returned invalid canonical auxiliary loss names."
        )
    unknown = sorted(set(weights) - set(raw_names))
    if unknown:
        raise ValueError(
            "Unknown SFT auxiliary loss weight names for the resolved model "
            f"profile: {unknown}. Supported names: {sorted(raw_names)}."
        )
    return raw_names


class ModelLoader(ABC):
    @abstractmethod
    def build(
        self,
        config: Any,
        *,
        model_meta: "ModelMeta",
        model_adapter: "ShaftModelAdapter",
        sequence_execution_contract: ShaftSequenceExecutionContract | None = None,
    ) -> "ModelArtifacts":
        raise NotImplementedError


@dataclass(frozen=True)
class ModelGroup:
    name: str
    model_ids: tuple[str, ...] = ()
    hf_model_types: tuple[str, ...] = ()
    template: str | None = None
    capabilities: ModelCapabilities | None = None
    module_groups: ModelModuleGroups | None = None
    processor_policy: ProcessorPolicy | None = None
    inference_policy: ShaftInferencePolicy | None = None
    sequence_execution_policy: SequenceExecutionPolicy | None = None
    training_objective_policy: TrainingObjectivePolicy | None = None
    peft_policy: PeftPolicy | None = None
    sharding_policy: ModelShardingPolicy | None = None
    default_experts_implementation: str | None = None
    requires: tuple[str, ...] = ()
    additional_saved_files: tuple[str, ...] = ()
    descriptor_matcher: Callable[[ResolvedModelDescriptor], bool] | None = None

    def matches(
        self,
        model_name_or_path: str,
        descriptor: ResolvedModelDescriptor | None = None,
    ) -> bool:
        if descriptor is not None:
            if self.descriptor_matcher is not None:
                return bool(self.descriptor_matcher(descriptor))
            if self.hf_model_types:
                return str(descriptor.hf_model_type).strip().lower() in {
                    str(value).strip().lower()
                    for value in self.hf_model_types
                    if str(value).strip()
                }
        normalized = str(model_name_or_path).strip().rstrip("/").lower()
        if not normalized:
            return False
        basename = normalized.rsplit("/", 1)[-1]
        return any(
            candidate == basename or candidate == normalized
            for candidate in (
                str(item).strip().lower() for item in self.model_ids if str(item).strip()
            )
        )


@dataclass(frozen=True)
class ModelMeta:
    model_type: str
    family: str
    default_template: str
    hf_model_types: tuple[str, ...] = ()
    model_groups: tuple[ModelGroup, ...] = ()
    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)
    module_groups: ModelModuleGroups = field(default_factory=ModelModuleGroups)
    processor_policy: ProcessorPolicy = field(default_factory=ProcessorPolicy)
    inference_policy: ShaftInferencePolicy = field(default_factory=ShaftInferencePolicy)
    sequence_execution_policy: SequenceExecutionPolicy = field(
        default_factory=SequenceExecutionPolicy
    )
    training_objective_policy: TrainingObjectivePolicy = field(
        default_factory=TrainingObjectivePolicy
    )
    peft_policy: PeftPolicy = field(
        default_factory=lambda: DefaultPeftPolicy(target_modules=["all-linear"])
    )
    sharding_policy: ModelShardingPolicy = field(default_factory=ModelShardingPolicy)
    requires: tuple[str, ...] = ()
    additional_saved_files: tuple[str, ...] = ()
    uses_hf_artifacts: bool = True
    loader: ModelLoader | None = None

    def with_loader(self, loader: ModelLoader) -> "ModelMeta":
        return ModelMeta(
            model_type=self.model_type,
            family=self.family,
            default_template=self.default_template,
            hf_model_types=self.hf_model_types,
            model_groups=self.model_groups,
            capabilities=self.capabilities,
            module_groups=self.module_groups,
            processor_policy=self.processor_policy,
            inference_policy=self.inference_policy,
            sequence_execution_policy=self.sequence_execution_policy,
            training_objective_policy=self.training_objective_policy,
            peft_policy=self.peft_policy,
            sharding_policy=self.sharding_policy,
            requires=self.requires,
            additional_saved_files=self.additional_saved_files,
            uses_hf_artifacts=self.uses_hf_artifacts,
            loader=loader,
        )

    def resolve_adapter(
        self,
        *,
        model_name_or_path: str,
        template_type: str | None = None,
        descriptor: ResolvedModelDescriptor | None = None,
    ) -> "ShaftModelAdapter":
        variant_types = {
            str(value).strip().lower()
            for value in (
                *self.hf_model_types,
                *(item for group in self.model_groups for item in group.hf_model_types),
            )
            if str(value).strip()
        }
        if (
            descriptor is not None
            and variant_types
            and descriptor.hf_model_type not in variant_types
        ):
            raise ValueError(
                f"HF model_type {descriptor.hf_model_type!r} is not a registered "
                f"variant of Shaft model family {self.model_type!r}; expected one "
                f"of {tuple(sorted(variant_types))}."
            )
        matched = self.get_matched_model_group(
            model_name_or_path,
            descriptor=descriptor,
        )
        if matched is None and descriptor is None and len(variant_types) > 1:
            raise ValueError(
                f"Model family {self.model_type!r} has multiple HF architecture "
                "variants, but the model could not be matched by catalog name and "
                "no local config.json descriptor is available. Resolve/download the "
                "HF config before selecting execution or sharding policies."
            )
        if matched is None and descriptor is not None and len(variant_types) > 1:
            raise ValueError(
                f"HF descriptor from {descriptor.source!r} does not match any "
                f"registered model group in Shaft family {self.model_type!r}. "
                "The model_type, architectures, or family-owned config facts are "
                "inconsistent with the declared model family."
            )
        resolved_template = str(template_type).strip().lower() if template_type else None
        if not resolved_template:
            resolved_template = (
                matched.template
                if matched is not None and matched.template
                else self.default_template
            )
        capabilities = (
            matched.capabilities
            if matched is not None and matched.capabilities is not None
            else self.capabilities
        )
        module_groups = (
            matched.module_groups
            if matched is not None and matched.module_groups is not None
            else self.module_groups
        )
        processor_policy = (
            matched.processor_policy
            if matched is not None and matched.processor_policy is not None
            else self.processor_policy
        )
        inference_policy = (
            matched.inference_policy
            if matched is not None and matched.inference_policy is not None
            else self.inference_policy
        )
        sequence_execution_policy = (
            matched.sequence_execution_policy
            if matched is not None and matched.sequence_execution_policy is not None
            else self.sequence_execution_policy
        )
        training_objective_policy = (
            matched.training_objective_policy
            if matched is not None and matched.training_objective_policy is not None
            else self.training_objective_policy
        )
        peft_policy = (
            matched.peft_policy
            if matched is not None and matched.peft_policy is not None
            else self.peft_policy
        )
        sharding_policy = (
            matched.sharding_policy
            if matched is not None and matched.sharding_policy is not None
            else self.sharding_policy
        )
        requires = list(self.requires)
        if matched is not None:
            requires.extend(matched.requires)
        additional_saved_files = list(self.additional_saved_files)
        if matched is not None:
            additional_saved_files.extend(matched.additional_saved_files)
        return ShaftModelAdapter(
            model_type=self.model_type,
            family=self.family,
            model_name_or_path=str(model_name_or_path),
            template_type=str(resolved_template).strip(),
            capabilities=capabilities,
            module_groups=module_groups,
            processor_policy=processor_policy,
            inference_policy=inference_policy,
            sequence_execution_policy=sequence_execution_policy,
            training_objective_policy=training_objective_policy,
            peft_policy=peft_policy,
            sharding_policy=sharding_policy,
            default_experts_implementation=(
                None if matched is None else matched.default_experts_implementation
            ),
            requires=_dedupe_non_empty(tuple(requires)),
            additional_saved_files=_dedupe_non_empty(tuple(additional_saved_files)),
            group_name=matched.name if matched is not None else None,
            model_meta=self,
            model_descriptor=descriptor,
        )

    def resolve_inference_contract(
        self,
        *,
        model_name_or_path: str,
        template_type: str | None = None,
        descriptor: ResolvedModelDescriptor | None = None,
    ) -> ShaftInferenceContract:
        """Resolve the template and policy required by a remote backend.

        Remote served-model aliases need not expose the underlying HF artifact.
        An unresolved variant is safe only when every registered variant has the
        same effective inference contract. Full model adapters remain strict.
        """

        matched = self.get_matched_model_group(
            model_name_or_path,
            descriptor=descriptor,
        )
        if matched is not None or descriptor is not None:
            adapter = self.resolve_adapter(
                model_name_or_path=model_name_or_path,
                template_type=template_type,
                descriptor=descriptor,
            )
            return ShaftInferenceContract(
                template_type=adapter.template_type,
                policy=adapter.inference_policy,
            )

        requested_template = str(template_type).strip().lower() if template_type else None
        candidate_groups: tuple[ModelGroup | None, ...] = (
            tuple(self.model_groups) if self.model_groups else (None,)
        )
        contracts = tuple(
            ShaftInferenceContract(
                template_type=(
                    requested_template
                    or (group.template if group is not None else None)
                    or self.default_template
                ),
                policy=(
                    group.inference_policy
                    if group is not None and group.inference_policy is not None
                    else self.inference_policy
                ),
            )
            for group in candidate_groups
        )
        resolved = contracts[0]
        incompatible_groups = tuple(
            group.name
            for group, contract in zip(candidate_groups, contracts, strict=True)
            if group is not None and contract != resolved
        )
        if incompatible_groups:
            raise ValueError(
                f"Remote model {model_name_or_path!r} does not identify a variant "
                f"of model family {self.model_type!r}, whose registered variants "
                "have different inference contracts. Use a registered model name "
                "or provide a resolvable HF descriptor. Incompatible groups: "
                f"{incompatible_groups}."
            )
        return resolved

    def default_target_modules(self) -> list[str]:
        return self.peft_policy.default_target_modules()

    def resolve_target_modules(self, target_modules: list[str]) -> list[str]:
        return self.peft_policy.resolve_target_modules(target_modules)

    def resolve_target_parameters(self, target_parameters: list[str]) -> list[str]:
        return self.peft_policy.resolve_target_parameters(target_parameters)

    def default_target_parameters(self) -> list[str]:
        return self.peft_policy.default_target_parameters()

    @property
    def candidate_templates(self) -> tuple[str, ...]:
        candidates = [self.default_template]
        candidates.extend(group.template for group in self.model_groups if group.template)
        return _dedupe_non_empty(tuple(candidates))

    def get_matched_model_group(
        self,
        model_name_or_path: str,
        *,
        descriptor: ResolvedModelDescriptor | None = None,
    ) -> ModelGroup | None:
        matches = tuple(
            group
            for group in self.model_groups
            if group.matches(model_name_or_path, descriptor=descriptor)
        )
        if len(matches) > 1:
            raise ValueError(
                f"Model {model_name_or_path!r} ambiguously matches groups "
                f"{tuple(group.name for group in matches)} in family "
                f"{self.model_type!r}."
            )
        return matches[0] if matches else None

    def resolve_template_type(self, model_name_or_path: str | None = None) -> str:
        if model_name_or_path:
            matched = self.get_matched_model_group(model_name_or_path)
            if matched is not None and matched.template:
                return matched.template
        return self.default_template

    def all_requires(self, model_name_or_path: str | None = None) -> list[str]:
        merged = list(self.requires)
        if model_name_or_path:
            matched = self.get_matched_model_group(model_name_or_path)
            if matched is not None:
                merged.extend(matched.requires)
        else:
            for group in self.model_groups:
                merged.extend(group.requires)
        return list(_dedupe_non_empty(tuple(merged)))

    def required_saved_files(self, model_name_or_path: str | None = None) -> tuple[str, ...]:
        merged = list(self.additional_saved_files)
        if model_name_or_path:
            matched = self.get_matched_model_group(model_name_or_path)
            if matched is not None:
                merged.extend(matched.additional_saved_files)
        else:
            for group in self.model_groups:
                merged.extend(group.additional_saved_files)
        return _dedupe_non_empty(tuple(merged))

    def check_requires(self, model_name_or_path: str | None = None) -> None:
        missing = _missing_requires(tuple(self.all_requires(model_name_or_path)))
        if missing:
            raise ImportError(
                f"Missing required packages for model_type={self.model_type!r}: {missing}"
            )


@dataclass(frozen=True)
class ShaftModelAdapter:
    model_type: str
    family: str
    model_name_or_path: str
    template_type: str
    capabilities: ModelCapabilities
    module_groups: ModelModuleGroups
    processor_policy: ProcessorPolicy
    peft_policy: PeftPolicy
    inference_policy: ShaftInferencePolicy = field(default_factory=ShaftInferencePolicy)
    sequence_execution_policy: SequenceExecutionPolicy = field(
        default_factory=SequenceExecutionPolicy
    )
    training_objective_policy: TrainingObjectivePolicy = field(
        default_factory=TrainingObjectivePolicy
    )
    sharding_policy: ModelShardingPolicy = field(default_factory=ModelShardingPolicy)
    default_experts_implementation: str | None = None
    requires: tuple[str, ...] = ()
    additional_saved_files: tuple[str, ...] = ()
    group_name: str | None = None
    model_meta: ModelMeta | None = None
    model_descriptor: ResolvedModelDescriptor | None = None

    def default_target_modules(self) -> list[str]:
        return self.peft_policy.default_target_modules()

    def resolve_target_modules(self, target_modules: list[str]) -> list[str]:
        return self.peft_policy.resolve_target_modules(target_modules)

    def resolve_target_parameters(self, target_parameters: list[str]) -> list[str]:
        return self.peft_policy.resolve_target_parameters(target_parameters)

    def default_target_parameters(self) -> list[str]:
        return self.peft_policy.default_target_parameters()

    def resolve_experts_implementation(self, requested: str | None) -> str | None:
        normalized = str(requested).strip().lower() if requested is not None else ""
        requested_value = normalized or None
        default_value = (
            str(self.default_experts_implementation).strip().lower()
            if self.default_experts_implementation is not None
            else ""
        ) or None
        if requested_value is not None and default_value is None:
            raise ValueError(
                "model.experts_implementation is only valid for a model profile "
                "that declares an expert execution backend."
            )
        return requested_value or default_value

    def validate_finetune_config(self, finetune: Any) -> None:
        self.peft_policy.validate_finetune_config(
            finetune,
            model_descriptor=self.model_descriptor,
            model_name_or_path=self.model_name_or_path,
        )

    def validate_training_finetune_config(self, finetune: Any) -> None:
        self.peft_policy.validate_training_finetune_config(
            finetune,
            model_descriptor=self.model_descriptor,
            model_name_or_path=self.model_name_or_path,
        )

    def validate_distributed_config(
        self,
        train: Any,
        *,
        finetune: Any | None = None,
    ) -> None:
        self.sharding_policy.validate_distributed_config(
            train,
            finetune=finetune,
        )

    def resolve_fsdp_transformer_layer_cls_to_wrap(self, values: list[str]) -> list[str]:
        try:
            return self.sharding_policy.resolve_fsdp_transformer_layer_cls_to_wrap(values)
        except ValueError as exc:
            raise ValueError(
                "train.distributed.fsdp.transformer_layer_cls_to_wrap=['auto'] is not available "
                f"for model.model_type={self.model_type!r}. Configure explicit transformer layer class names."
            ) from exc

    def prepare_sft_forward_inputs(
        self,
        *,
        model: Any,
        inputs: dict[str, Any],
    ) -> dict[str, Any]:
        return self.training_objective_policy.prepare_sft_forward_inputs(
            model=model,
            inputs=inputs,
        )

    def auxiliary_loss_names(self) -> tuple[str, ...]:
        return self.training_objective_policy.auxiliary_loss_names()

    def resolve_sft_auxiliary_loss_terms(
        self,
        *,
        model: Any,
        outputs: Any,
        inputs: dict[str, Any],
    ) -> tuple[ShaftAuxiliaryLossTerm, ...]:
        return self.training_objective_policy.resolve_sft_auxiliary_loss_terms(
            model=model,
            outputs=outputs,
            inputs=inputs,
        )

    def resolve_sft_eval_auxiliary_statistics(
        self,
        *,
        model: Any,
        outputs: Any,
        inputs: dict[str, Any],
    ) -> tuple[ShaftEvalAuxiliaryStatistic, ...]:
        return self.training_objective_policy.resolve_sft_eval_auxiliary_statistics(
            model=model,
            outputs=outputs,
            inputs=inputs,
        )

    def finalize_sft_eval_auxiliary_statistics(
        self,
        statistics: tuple[ShaftEvalAuxiliaryStatistic, ...],
    ) -> tuple[ShaftEvalAuxiliaryMetric, ...]:
        return self.training_objective_policy.finalize_sft_eval_auxiliary_statistics(statistics)

    def build_processor_batch(
        self,
        *,
        processor: Any,
        tokenizer: Any | None = None,
        prompt_texts: list[str],
        images: list[Any],
        min_pixels: int | None,
        max_pixels: int | None,
        input_mode: str = "training",
    ) -> ShaftProcessedBatch:
        return self.processor_policy.build_batch(
            processor=processor,
            tokenizer=tokenizer,
            prompt_texts=prompt_texts,
            images=images,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            input_mode=input_mode,
        )

    def resolve_processor_padding_side(self, input_mode: str) -> str:
        return self.processor_policy.input_policy.resolve_padding_side(input_mode)

    def prepare_rollout_image(
        self,
        image: Any,
        *,
        min_pixels: int | None,
        max_pixels: int | None,
    ) -> Any:
        return self.processor_policy.prepare_rollout_image(
            image,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

    def build_rollout_scoring_plan(
        self,
        *,
        prompt_inputs: dict[str, Any],
        sequences: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> ShaftRolloutScoringPlan:
        return self.processor_policy.build_rollout_scoring_plan(
            prompt_inputs=prompt_inputs,
            sequences=sequences,
            attention_mask=attention_mask,
        )

    def estimate_processor_image_cost(
        self,
        *,
        processor: Any,
        image_sizes: tuple[tuple[int, int], ...],
        min_pixels: int | None,
        max_pixels: int | None,
    ) -> ShaftProcessorCostEstimate:
        return self.processor_policy.estimate_image_cost(
            processor=processor,
            image_sizes=image_sizes,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

    def processor_cost_semantics_signature(
        self,
        *,
        processor: Any,
        min_pixels: int | None,
        max_pixels: int | None,
    ) -> tuple[object, ...]:
        return self.processor_policy.cost_semantics_signature(
            processor=processor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

    def estimate_processor_token_layout(
        self,
        *,
        processor: Any,
        tokenizer: Any,
        rendered_token_ids: tuple[int, ...],
        image_costs: tuple[ShaftProcessorCostEstimate, ...],
    ) -> ShaftProcessorTokenLayout:
        return self.processor_policy.estimate_token_layout(
            processor=processor,
            tokenizer=tokenizer,
            rendered_token_ids=rendered_token_ids,
            image_costs=image_costs,
        )

    def build_processor_token_layout(
        self,
        *,
        rendered_token_ids: tuple[int, ...],
        processed_batch: ShaftProcessedBatch,
        row_index: int,
    ) -> ShaftProcessorTokenLayout:
        return self.processor_policy.build_token_layout(
            rendered_token_ids=rendered_token_ids,
            processed_batch=processed_batch,
            row_index=row_index,
        )

    def assemble_processor_training_inputs(
        self,
        *,
        processed_batch: ShaftProcessedBatch,
        sequence_inputs: dict[str, Any],
        row_indices: tuple[int, ...],
    ) -> dict[str, Any]:
        return self.processor_policy.assemble_training_inputs(
            processed_batch=processed_batch,
            sequence_inputs=sequence_inputs,
            row_indices=row_indices,
        )

    def validate_sequence_execution(
        self,
        *,
        model: Any,
        contract: ShaftSequenceExecutionContract,
    ) -> None:
        self.sequence_execution_policy.validate_runtime(
            model=model,
            contract=contract,
        )

    def configure_sequence_execution(
        self,
        *,
        model: Any,
        contract: ShaftSequenceExecutionContract,
    ) -> None:
        self.sequence_execution_policy.configure_runtime(
            model=model,
            contract=contract,
        )

    def build_sequence_execution_contract(
        self,
        *,
        layout: str,
        device_type: str,
        attention_implementation: str | None,
        torch_dtype: str,
        distributed_strategy: str,
        torch_compile: bool = False,
    ) -> ShaftSequenceExecutionContract:
        return self.sequence_execution_policy.build_contract(
            layout=layout,
            device_type=device_type,
            attention_implementation=attention_implementation,
            torch_dtype=torch_dtype,
            distributed_strategy=distributed_strategy,
            torch_compile=torch_compile,
        )

    def prepare_sequence_training_inputs(
        self,
        *,
        model: Any,
        inputs: dict[str, Any],
    ) -> dict[str, Any]:
        return self.sequence_execution_policy.prepare_training_inputs(
            model=model,
            inputs=inputs,
        )

    def required_saved_files(self) -> tuple[str, ...]:
        return _dedupe_non_empty(self.additional_saved_files)

    def check_requires(self) -> None:
        missing = _missing_requires(self.requires)
        if missing:
            raise ImportError(
                f"Missing required packages for model_type={self.model_type!r}: {missing}"
            )

    def build_model_info(
        self,
        *,
        torch_dtype: torch.dtype | str,
        max_model_len: int | None = None,
        quant_method: str | None = None,
        quant_bits: int | None = None,
    ) -> "ModelInfo":
        return ModelInfo(
            model_type=self.model_type,
            model_dir=self.model_name_or_path,
            torch_dtype=torch_dtype,
            max_model_len=max_model_len,
            quant_method=quant_method,
            quant_bits=quant_bits,
            is_multimodal=self.capabilities.is_multimodal,
            family=self.family,
        )

    def build_template(self):
        from shaft.template import build_template_from_meta, resolve_template_meta

        template_meta = resolve_template_meta(template_type=self.template_type, model_adapter=self)
        return build_template_from_meta(template_meta)


@dataclass(frozen=True)
class ModelInfo:
    model_type: str
    model_dir: str
    torch_dtype: torch.dtype | str
    max_model_len: int | None = None
    quant_method: str | None = None
    quant_bits: int | None = None
    is_multimodal: bool = False
    family: str | None = None


@dataclass
class ModelArtifacts:
    model: torch.nn.Module
    tokenizer: object
    processor: object
    model_meta: ModelMeta
    model_adapter: ShaftModelAdapter
    model_info: ModelInfo
    template: object
    finetune_plan: object | None = None


@dataclass
class LoadedAdapterArtifacts:
    """Adapter model plus HF assets, without a synthetic training finetune plan."""

    model: torch.nn.Module
    tokenizer: object
    processor: object
    model_adapter: ShaftModelAdapter
