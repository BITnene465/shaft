from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import importlib.util
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from packaging.version import InvalidVersion, Version
import torch

from .sharding import ModelShardingPolicy


def _dedupe_non_empty(values: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(item).strip() for item in values if str(item).strip()))


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

    previous: list[tuple[Any, str]] = []
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
            previous.append((candidate, str(getattr(candidate, "padding_side"))))
            setattr(candidate, "padding_side", normalized)
        yield
    finally:
        for candidate, value in reversed(previous):
            setattr(candidate, "padding_side", value)


@dataclass(frozen=True)
class ModelCapabilities:
    is_multimodal: bool = True


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
    language_model: tuple[str, ...] = ()
    vision_tower: tuple[str, ...] = ()
    aligner: tuple[str, ...] = ()
    generator: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for key in ("language_model", "vision_tower", "aligner", "generator"):
            value = getattr(self, key)
            if isinstance(value, str):
                coerced = (value,) if value.strip() else ()
            else:
                coerced = _dedupe_non_empty(tuple(value))
            object.__setattr__(self, key, coerced)

    def prefixes_for_group(self, group_name: str) -> tuple[str, ...]:
        normalized = str(group_name).strip().lower()
        if normalized not in {"language_model", "vision_tower", "aligner", "generator"}:
            raise KeyError(f"Unknown model module group: {group_name!r}")
        return getattr(self, normalized)

    def resolve_group_for_name(self, name: str) -> str | None:
        normalized_name = str(name).strip()
        if not normalized_name:
            return None

        best_group: str | None = None
        best_prefix_len = -1
        for group_name in ("language_model", "vision_tower", "aligner", "generator"):
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


@dataclass(frozen=True)
class ShaftProcessedBatch:
    model_inputs: dict[str, Any]
    batch_size: int
    media_manifest: ShaftProcessorMediaManifest | None = None

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("ShaftProcessedBatch.batch_size must be positive.")
        if "input_ids" not in self.model_inputs or "attention_mask" not in self.model_inputs:
            raise ValueError(
                "ShaftProcessedBatch requires processor input_ids and attention_mask."
            )
        for key in ("input_ids", "attention_mask"):
            value = self.model_inputs[key]
            if not torch.is_tensor(value) or value.ndim < 2:
                raise ValueError(f"Processor output {key!r} must be a batched tensor.")
            if int(value.shape[0]) != self.batch_size:
                raise ValueError(
                    f"Processor output {key!r} batch axis does not match batch_size."
                )
        if self.media_manifest is not None:
            if len(self.media_manifest.segments) != self.batch_size:
                raise ValueError(
                    "Processor media manifest must contain one segment per processor row."
                )


@dataclass(frozen=True)
class ProcessorPolicy:
    supports_pixel_budget: bool = False
    supports_exact_image_cost: bool = False
    sample_aligned_model_input_names: tuple[str, ...] = ()
    whole_batch_model_input_names: tuple[str, ...] = ()
    static_model_input_names: tuple[str, ...] = ()
    assembled_sequence_input_names: tuple[str, ...] = (
        "input_ids",
        "attention_mask",
        "mm_token_type_ids",
        "labels",
        "loss_scale",
        "completion_mask",
    )
    unsupported_sequence_input_names: tuple[str, ...] = (
        "position_ids",
        "token_type_ids",
    )

    def __post_init__(self) -> None:
        field_names = (
            "sample_aligned_model_input_names",
            "whole_batch_model_input_names",
            "static_model_input_names",
            "assembled_sequence_input_names",
            "unsupported_sequence_input_names",
        )
        for field_name in field_names:
            object.__setattr__(self, field_name, _dedupe_non_empty(getattr(self, field_name)))

        declared_layouts: dict[str, str] = {}
        for field_name in field_names[:3]:
            for input_name in getattr(self, field_name):
                previous = declared_layouts.setdefault(input_name, field_name)
                if previous != field_name:
                    raise ValueError(
                        f"Processor model input {input_name!r} is declared by both "
                        f"{previous!r} and {field_name!r}."
                    )
        sequence_names = set(self.assembled_sequence_input_names) | set(
            self.unsupported_sequence_input_names
        )
        overlap = sorted(sequence_names & declared_layouts.keys())
        if overlap:
            raise ValueError(
                "Processor model inputs cannot be both sequence-aligned and non-sequence: "
                f"{overlap}."
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
        padding_side: str | None = None,
    ) -> ShaftProcessedBatch:
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
            padding_side=padding_side,
        ):
            outputs = processor(**kwargs)
        return ShaftProcessedBatch(
            model_inputs=dict(outputs),
            batch_size=len(prompt_texts),
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
        missing_sequence_inputs = [
            key
            for key in self.unsupported_sequence_input_names
            if key in processed_batch.model_inputs and key not in sequence_inputs
        ]
        if missing_sequence_inputs:
            raise ValueError(
                "Processor policy must explicitly assemble sequence-aligned model inputs: "
                f"{missing_sequence_inputs}."
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
        return ShaftProcessorTokenLayout(processed_boundaries)


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
            capability_signature=(
                f"{type(self).__module__}.{type(self).__qualname__}",
            ),
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
        if not normalized or normalized == ["auto"]:
            return self.default_target_modules()
        return normalized


@dataclass(frozen=True)
class DefaultPeftPolicy(PeftPolicy):
    target_modules: list[str]

    def default_target_modules(self) -> list[str]:
        return list(self.target_modules)


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
    template: str | None = None
    capabilities: ModelCapabilities | None = None
    module_groups: ModelModuleGroups | None = None
    processor_policy: ProcessorPolicy | None = None
    sequence_execution_policy: SequenceExecutionPolicy | None = None
    peft_policy: PeftPolicy | None = None
    sharding_policy: ModelShardingPolicy | None = None
    requires: tuple[str, ...] = ()
    additional_saved_files: tuple[str, ...] = ()

    def matches(self, model_name_or_path: str) -> bool:
        normalized = str(model_name_or_path).strip().rstrip("/").lower()
        if not normalized:
            return False
        basename = normalized.rsplit("/", 1)[-1]
        return any(
            candidate == basename or candidate == normalized
            for candidate in (str(item).strip().lower() for item in self.model_ids if str(item).strip())
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
    sequence_execution_policy: SequenceExecutionPolicy = field(
        default_factory=SequenceExecutionPolicy
    )
    peft_policy: PeftPolicy = field(default_factory=lambda: DefaultPeftPolicy(target_modules=["all-linear"]))
    sharding_policy: ModelShardingPolicy = field(default_factory=ModelShardingPolicy)
    requires: tuple[str, ...] = ()
    additional_saved_files: tuple[str, ...] = ()
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
            sequence_execution_policy=self.sequence_execution_policy,
            peft_policy=self.peft_policy,
            sharding_policy=self.sharding_policy,
            requires=self.requires,
            additional_saved_files=self.additional_saved_files,
            loader=loader,
        )

    def resolve_adapter(
        self,
        *,
        model_name_or_path: str,
        template_type: str | None = None,
    ) -> "ShaftModelAdapter":
        matched = self.get_matched_model_group(model_name_or_path)
        resolved_template = str(template_type).strip().lower() if template_type else None
        if not resolved_template:
            resolved_template = (
                matched.template if matched is not None and matched.template else self.default_template
            )
        capabilities = (
            matched.capabilities if matched is not None and matched.capabilities is not None else self.capabilities
        )
        module_groups = (
            matched.module_groups if matched is not None and matched.module_groups is not None else self.module_groups
        )
        processor_policy = (
            matched.processor_policy
            if matched is not None and matched.processor_policy is not None
            else self.processor_policy
        )
        sequence_execution_policy = (
            matched.sequence_execution_policy
            if matched is not None and matched.sequence_execution_policy is not None
            else self.sequence_execution_policy
        )
        peft_policy = (
            matched.peft_policy if matched is not None and matched.peft_policy is not None else self.peft_policy
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
            sequence_execution_policy=sequence_execution_policy,
            peft_policy=peft_policy,
            sharding_policy=sharding_policy,
            requires=_dedupe_non_empty(tuple(requires)),
            additional_saved_files=_dedupe_non_empty(tuple(additional_saved_files)),
            group_name=matched.name if matched is not None else None,
            model_meta=self,
        )

    def default_target_modules(self) -> list[str]:
        return self.peft_policy.default_target_modules()

    def resolve_target_modules(self, target_modules: list[str]) -> list[str]:
        return self.peft_policy.resolve_target_modules(target_modules)

    @property
    def candidate_templates(self) -> tuple[str, ...]:
        candidates = [self.default_template]
        candidates.extend(group.template for group in self.model_groups if group.template)
        return _dedupe_non_empty(tuple(candidates))

    def get_matched_model_group(self, model_name_or_path: str) -> ModelGroup | None:
        for group in self.model_groups:
            if group.matches(model_name_or_path):
                return group
        return None

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
    sequence_execution_policy: SequenceExecutionPolicy = field(
        default_factory=SequenceExecutionPolicy
    )
    sharding_policy: ModelShardingPolicy = field(default_factory=ModelShardingPolicy)
    requires: tuple[str, ...] = ()
    additional_saved_files: tuple[str, ...] = ()
    group_name: str | None = None
    model_meta: ModelMeta | None = None

    def default_target_modules(self) -> list[str]:
        return self.peft_policy.default_target_modules()

    def resolve_target_modules(self, target_modules: list[str]) -> list[str]:
        return self.peft_policy.resolve_target_modules(target_modules)

    def resolve_fsdp_transformer_layer_cls_to_wrap(self, values: list[str]) -> list[str]:
        try:
            return self.sharding_policy.resolve_fsdp_transformer_layer_cls_to_wrap(values)
        except ValueError as exc:
            raise ValueError(
                "train.distributed.fsdp.transformer_layer_cls_to_wrap=['auto'] is not available "
                f"for model.model_type={self.model_type!r}. Configure explicit transformer layer class names."
            ) from exc

    def build_processor_batch(
        self,
        *,
        processor: Any,
        tokenizer: Any | None = None,
        prompt_texts: list[str],
        images: list[Any],
        min_pixels: int | None,
        max_pixels: int | None,
        padding_side: str | None = None,
    ) -> ShaftProcessedBatch:
        return self.processor_policy.build_batch(
            processor=processor,
            tokenizer=tokenizer,
            prompt_texts=prompt_texts,
            images=images,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            padding_side=padding_side,
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
