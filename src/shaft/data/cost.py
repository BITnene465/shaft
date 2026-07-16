from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import logging
import math
from pathlib import Path
from typing import Any, Protocol
import warnings

from PIL import Image
from transformers import __version__ as transformers_version

from shaft.config.data import SHAFT_BATCH_RESOURCE_NAMES
from shaft.model.input_identity import tokenizer_artifact_fingerprint
from shaft.template import ShaftChatRenderer
from shaft.utils.distributed import is_rank_zero

from .mixing import ShaftSampleRef
from .transforms import (
    is_planning_safe_online_transform,
    planning_online_transform_fingerprint,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ShaftSFTSourceIdentity:
    fingerprint: str
    complete: bool
    incomplete_reasons: tuple[str, ...] = ()


def _sequence_fingerprint(records: Any) -> str:
    declared = str(getattr(records, "fingerprint", "")).strip()
    if declared:
        return declared
    digest = hashlib.sha256()
    digest.update(f"python-sequence:{len(records)}".encode("utf-8"))
    for record in records:
        digest.update(b"\0")
        digest.update(repr(record).encode("utf-8"))
    return digest.hexdigest()


def _dataset_records_fingerprint(records: Any) -> str:
    if isinstance(records, Mapping):
        payload = tuple(
            (str(dataset_name), _sequence_fingerprint(dataset_records))
            for dataset_name, dataset_records in sorted(records.items())
        )
    else:
        payload = (("__flat__", _sequence_fingerprint(records)),)
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


def sft_cost_source_fingerprint(dataset: Any) -> str:
    """Return a media-header-free fingerprint suitable for cross-rank preflight."""

    transform_fingerprints = tuple(
        planning_online_transform_fingerprint(transform)
        for transform in getattr(dataset, "online_transforms", ())
    )
    payload = (
        "shaft-sft-cost-source-v3-immutable-media-snapshot",
        _dataset_records_fingerprint(getattr(dataset, "records", ())),
        transform_fingerprints,
        str(getattr(dataset, "media_snapshot_id", "")).strip(),
    )
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


def sft_runtime_source_identity(dataset: Any) -> ShaftSFTSourceIdentity:
    """Build a non-blocking source identity for observability and A/B checks.

    Planned batching still requires the stricter cost fingerprint. Fixed paths may
    use unversioned transforms, so telemetry records an incomplete identity instead
    of changing whether training can start.
    """

    transform_identities: list[tuple[str, str]] = []
    incomplete_reasons: list[str] = []
    for transform in getattr(dataset, "online_transforms", ()):
        if is_planning_safe_online_transform(transform):
            transform_identities.append(
                ("versioned", planning_online_transform_fingerprint(transform))
            )
            continue
        qualified_name = (
            f"{getattr(transform, '__module__', type(transform).__module__)}."
            f"{getattr(transform, '__qualname__', type(transform).__qualname__)}"
        )
        transform_identities.append(("unversioned", qualified_name))
        incomplete_reasons.append(f"unversioned_transform:{qualified_name}")

    media_snapshot_id = str(getattr(dataset, "media_snapshot_id", "")).strip()
    if not media_snapshot_id:
        incomplete_reasons.append("missing_media_snapshot_id")
    payload = (
        "shaft-sft-runtime-source-identity-v1",
        _dataset_records_fingerprint(getattr(dataset, "records", ())),
        tuple(transform_identities),
        media_snapshot_id,
    )
    return ShaftSFTSourceIdentity(
        fingerprint=hashlib.sha256(repr(payload).encode("utf-8")).hexdigest(),
        complete=not incomplete_reasons,
        incomplete_reasons=tuple(incomplete_reasons),
    )


def validate_sft_cost_dataset(dataset: Any) -> None:
    if not hasattr(dataset, "get_planning_item"):
        raise TypeError("SFT cost-aware batching requires a dataset with get_planning_item().")
    if not str(getattr(dataset, "media_snapshot_id", "")).strip():
        raise ValueError("SFT cost-aware batching requires an immutable media_snapshot_id.")
    unsafe_transforms = [
        transform
        for transform in getattr(dataset, "online_transforms", ())
        if not is_planning_safe_online_transform(transform)
    ]
    if unsafe_transforms:
        names = [
            getattr(transform, "__name__", type(transform).__name__)
            for transform in unsafe_transforms
        ]
        raise ValueError(
            "SFT cost-aware batching requires deterministic, image-identity/geometry and "
            "media-placeholder preserving online transforms; unsafe transforms: "
            f"{names}."
        )


def validate_sft_cost_model_adapter(model_adapter: Any) -> None:
    processor_policy = getattr(model_adapter, "processor_policy", None)
    if not bool(getattr(processor_policy, "supports_exact_image_cost", False)):
        raise ValueError(
            "SFT cost-aware batching requires a model ProcessorPolicy that declares "
            "supports_exact_image_cost=True."
        )


@dataclass(frozen=True, slots=True)
class ShaftSampleCost:
    """Immutable planning cost for one logical sample draw.

    The values describe processed training work, not source-file byte size. `exact=False`
    marks an explicit estimator/fallback; hard dynamic-batch limits must not rely on an
    inexact cost without an additional safety margin.
    """

    llm_tokens: int
    supervised_tokens: int = 0
    vision_patches: int = 0
    loss_weight_sum: float | None = None
    exact: bool = False

    def __post_init__(self) -> None:
        if int(self.llm_tokens) <= 0:
            raise ValueError("ShaftSampleCost.llm_tokens must be > 0.")
        if int(self.supervised_tokens) < 0:
            raise ValueError("ShaftSampleCost.supervised_tokens must be >= 0.")
        if int(self.supervised_tokens) > int(self.llm_tokens):
            raise ValueError("ShaftSampleCost.supervised_tokens cannot exceed llm_tokens.")
        if int(self.vision_patches) < 0:
            raise ValueError("ShaftSampleCost.vision_patches must be >= 0.")
        if self.loss_weight_sum is not None:
            value = float(self.loss_weight_sum)
            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    "ShaftSampleCost.loss_weight_sum must be finite and >= 0 when set."
                )

    def resource_value(self, name: str) -> int:
        normalized = str(name).strip().lower()
        if normalized not in SHAFT_BATCH_RESOURCE_NAMES:
            raise ValueError(f"ShaftSampleCost does not expose configured resource {normalized!r}.")
        if normalized == "vision_patches":
            return int(self.vision_patches)
        raise AssertionError(
            "Configured batch-resource schema has no ShaftSampleCost field dispatch: "
            f"{normalized!r}."
        )


class ShaftSampleCostProvider(Protocol):
    fingerprint: str

    def __call__(self, sample_ref: ShaftSampleRef) -> ShaftSampleCost: ...


class ShaftRowInvariantCostProvider:
    """Small immutable mapping for costs that cannot vary across logical draws.

    The key deliberately excludes draw context. Do not use this adapter for prompt
    rotation or any transform whose cost can change for the same source row. Production
    SFT uses ``ShaftSFTSampleCostProvider`` so costs are resolved lazily per logical draw.
    """

    def __init__(
        self,
        costs: Mapping[tuple[str, int], ShaftSampleCost],
        *,
        fingerprint: str | None = None,
    ) -> None:
        self._costs = dict(costs)
        if not self._costs:
            raise ValueError("ShaftRowInvariantCostProvider requires at least one sample cost.")
        if fingerprint is None:
            payload = tuple(sorted(self._costs.items()))
            fingerprint = hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()
        self.fingerprint = str(fingerprint).strip()
        if not self.fingerprint:
            raise ValueError("ShaftRowInvariantCostProvider fingerprint must not be empty.")

    def __call__(self, sample_ref: ShaftSampleRef) -> ShaftSampleCost:
        key = (str(sample_ref.dataset_name), int(sample_ref.row_index))
        try:
            return self._costs[key]
        except KeyError as exc:
            raise KeyError(
                "Missing sample cost for "
                f"dataset={sample_ref.dataset_name!r}, row_index={sample_ref.row_index}."
            ) from exc


class ShaftSFTSampleCostProvider:
    """Exact policy-driven SFT runtime estimator without image decode/processor execution.

    Text/prompt transforms are resolved through `SFTDataset.get_planning_item`, so a
    logical draw uses the same prompt variant as the worker dataset. Image dimensions are
    read from the file header and delegated to the model's ProcessorPolicy.
    """

    def __init__(
        self,
        *,
        dataset: Any,
        model_adapter: Any,
        template: Any,
        processor: Any,
        tokenizer: Any,
        min_pixels: int | None,
        max_pixels: int | None,
        max_length: int | None,
        add_eos_token: bool,
        loss_scale_name: str,
        cache_size: int = 65536,
    ) -> None:
        validate_sft_cost_dataset(dataset)
        validate_sft_cost_model_adapter(model_adapter)
        self.dataset = dataset
        self.model_adapter = model_adapter
        self.template = template
        self.processor = processor
        self.tokenizer = tokenizer
        self.renderer = ShaftChatRenderer.from_components(
            processor=processor,
            tokenizer=tokenizer,
        )
        self.min_pixels = int(min_pixels) if min_pixels is not None else None
        self.max_pixels = int(max_pixels) if max_pixels is not None else None
        self.max_length = int(max_length) if max_length is not None else None
        self.add_eos_token = bool(add_eos_token)
        self.loss_scale_name = str(loss_scale_name).strip().lower() or "default"
        self.cache_size = max(int(cache_size), 0)
        self._image_sizes: OrderedDict[str, tuple[int, int]] = OrderedDict()
        self._sample_costs: OrderedDict[tuple[str, int, int, int], ShaftSampleCost] = OrderedDict()
        self._large_image_warning_count = 0
        processor_cost_semantics = model_adapter.processor_cost_semantics_signature(
            processor=processor,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        fingerprint_payload = (
            "shaft-sft-runtime-cost-v8-bounded",
            sft_cost_source_fingerprint(dataset),
            str(getattr(model_adapter, "model_type", "")),
            str(getattr(model_adapter, "template_type", "")),
            processor_cost_semantics,
            f"{type(processor).__module__}.{type(processor).__qualname__}",
            str(transformers_version),
            repr(getattr(processor, "chat_template", None)),
            tokenizer_artifact_fingerprint(tokenizer),
            f"{type(template).__module__}.{type(template).__qualname__}",
            repr(getattr(template, "template_meta", None)),
            self.min_pixels,
            self.max_pixels,
            self.max_length,
            self.add_eos_token,
            self.loss_scale_name,
        )
        self.fingerprint = hashlib.sha256(repr(fingerprint_payload).encode("utf-8")).hexdigest()

    def __call__(self, sample_ref: ShaftSampleRef) -> ShaftSampleCost:
        item = self.dataset.get_planning_item(sample_ref)
        target_text = str(item["target_text"])
        image_path = str(item.get("image_path", "")).strip()
        if not image_path:
            raise ValueError(
                "SFT cost estimation requires image_path for "
                f"dataset={sample_ref.dataset_name!r}, row={sample_ref.row_index}."
            )
        cache_key = (
            str(sample_ref.dataset_name),
            int(sample_ref.row_index),
            int(sample_ref.context.draw_id),
            int(sample_ref.context.transform_seed),
        )
        cached = self._sample_costs.get(cache_key)
        if cached is not None:
            self._sample_costs.move_to_end(cache_key)
            return cached
        supervision_plan = self.template.build_supervision_plan(
            item=item,
            target_text=target_text,
            renderer=self.renderer,
            loss_scale_name=self.loss_scale_name,
        )
        rendered_ids = supervision_plan.rendered_prefix_token_ids or self.renderer.tokenize(
            supervision_plan.prompt_text
        )
        image_size = self._get_image_size(image_path)
        image_estimate = self.model_adapter.estimate_processor_image_cost(
            processor=self.processor,
            image_sizes=(image_size,),
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        prefix_token_layout = self.model_adapter.estimate_processor_token_layout(
            processor=self.processor,
            tokenizer=self.tokenizer,
            rendered_token_ids=rendered_ids,
            image_costs=(image_estimate,),
        )
        supervision_cost = self.template.estimate_supervision_cost(
            plan=supervision_plan,
            tokenizer=self.tokenizer,
            prefix_token_layout=prefix_token_layout,
            add_eos_token=self.add_eos_token,
            max_length=self.max_length,
        )
        cost = ShaftSampleCost(
            llm_tokens=supervision_cost.llm_tokens,
            supervised_tokens=supervision_cost.supervised_tokens,
            vision_patches=image_estimate.vision_patches,
            loss_weight_sum=supervision_cost.loss_weight_sum,
            exact=bool(image_estimate.exact),
        )
        self._remember_sample_cost(cache_key, cost)
        return cost

    def _get_image_size(self, image_path: str) -> tuple[int, int]:
        path = Path(image_path).expanduser().resolve(strict=True)
        cache_key = str(path)
        cached = self._image_sizes.get(cache_key)
        if cached is not None:
            self._image_sizes.move_to_end(cache_key)
            return cached
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                width, height = image.size
        bomb_warnings = []
        for caught_warning in caught:
            if issubclass(
                caught_warning.category,
                Image.DecompressionBombWarning,
            ):
                bomb_warnings.append(caught_warning)
                continue
            warnings.warn_explicit(
                caught_warning.message,
                caught_warning.category,
                caught_warning.filename,
                caught_warning.lineno,
                source=getattr(caught_warning, "source", None),
            )
        if bomb_warnings:
            self._large_image_warning_count += 1
            if self._large_image_warning_count == 1 and is_rank_zero():
                logger.warning(
                    "[bounded-cost] large image header observed; path=%s size=%sx%s "
                    "further PIL DecompressionBombWarning messages are aggregated",
                    path,
                    width,
                    height,
                )
        resolved = (int(width), int(height))
        self._remember_image_size(cache_key, resolved)
        return resolved

    def _remember_image_size(self, image_path: str, image_size: tuple[int, int]) -> None:
        if self.cache_size > 0:
            self._image_sizes[image_path] = image_size
            self._image_sizes.move_to_end(image_path)
            while len(self._image_sizes) > self.cache_size:
                self._image_sizes.popitem(last=False)

    def _remember_sample_cost(
        self,
        key: tuple[str, int, int, int],
        cost: ShaftSampleCost,
    ) -> None:
        if self.cache_size <= 0:
            return
        self._sample_costs[key] = cost
        self._sample_costs.move_to_end(key)
        while len(self._sample_costs) > self.cache_size:
            self._sample_costs.popitem(last=False)

    @property
    def cache_entry_counts(self) -> tuple[int, int]:
        """Return ``(sample_costs, image_headers)`` for bounded diagnostics/tests."""

        return len(self._sample_costs), len(self._image_sizes)
