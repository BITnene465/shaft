from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Protocol

from PIL import Image
from transformers import __version__ as transformers_version

from shaft.template import ShaftChatRenderer

from .mixing import ShaftSampleRef
from .transforms import (
    is_planning_safe_online_transform,
    planning_online_transform_fingerprint,
)


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


def _stable_artifact_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return tuple(
            (str(key), _stable_artifact_value(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
    if isinstance(value, (list, tuple)):
        return tuple(_stable_artifact_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        resolved = [_stable_artifact_value(item) for item in value]
        return tuple(sorted(resolved, key=repr))
    if hasattr(value, "content"):
        return (
            f"{type(value).__module__}.{type(value).__qualname__}",
            str(getattr(value, "content", "")),
            bool(getattr(value, "single_word", False)),
            bool(getattr(value, "lstrip", False)),
            bool(getattr(value, "rstrip", False)),
            bool(getattr(value, "normalized", False)),
            bool(getattr(value, "special", False)),
        )
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _tokenizer_artifact_fingerprint(tokenizer: Any) -> str:
    """Bind token costs to tokenizer implementation and serialized vocabulary assets."""

    backend = getattr(tokenizer, "backend_tokenizer", None)
    backend_to_str = getattr(backend, "to_str", None)
    if callable(backend_to_str):
        artifact_kind = "backend-tokenizer-json"
        artifact_payload = str(backend_to_str())
    else:
        declared = getattr(tokenizer, "shaft_cost_fingerprint", None)
        declared = declared() if callable(declared) else declared
        artifact_payload = str(declared or "").strip()
        if not artifact_payload:
            raise ValueError(
                "Exact SFT CostPlan requires tokenizer.backend_tokenizer.to_str() "
                "or an explicit tokenizer.shaft_cost_fingerprint covering the full "
                "vocabulary and tokenization model (including merges/unigram state)."
            )
        artifact_kind = "declared-shaft-cost-fingerprint"

    metadata = (
        "shaft-tokenizer-artifact-v1",
        artifact_kind,
        hashlib.sha256(artifact_payload.encode("utf-8")).hexdigest(),
        f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
        str(getattr(tokenizer, "name_or_path", "")),
        getattr(tokenizer, "vocab_size", None),
        getattr(tokenizer, "eos_token_id", None),
        getattr(tokenizer, "bos_token_id", None),
        getattr(tokenizer, "pad_token_id", None),
        getattr(tokenizer, "model_max_length", None),
        getattr(tokenizer, "padding_side", None),
        getattr(tokenizer, "truncation_side", None),
        _stable_artifact_value(getattr(tokenizer, "special_tokens_map", {})),
        _stable_artifact_value(getattr(tokenizer, "init_kwargs", {})),
        _stable_artifact_value(getattr(tokenizer, "added_tokens_encoder", {})),
    )
    return hashlib.sha256(repr(metadata).encode("utf-8")).hexdigest()


def sft_cost_planning_source_fingerprint(dataset: Any) -> str:
    """Return a media-header-free fingerprint suitable for cross-rank preflight."""

    transform_fingerprints = tuple(
        planning_online_transform_fingerprint(transform)
        for transform in getattr(dataset, "online_transforms", ())
    )
    payload = (
        "shaft-sft-cost-planning-source-v1",
        str(getattr(getattr(dataset, "sample_plan", None), "fingerprint", "")),
        _dataset_records_fingerprint(getattr(dataset, "records", ())),
        transform_fingerprints,
    )
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


def _iter_planned_image_paths(dataset: Any):
    records = getattr(dataset, "records", ())
    sample_plan = getattr(dataset, "sample_plan", None)
    if sample_plan is not None:
        if not isinstance(records, Mapping):
            raise TypeError(
                "SFT cost planning with a SamplePlan requires dataset-keyed records."
            )
        for position in range(len(sample_plan)):
            sample_ref = sample_plan.ref_at(position)
            record = records[sample_ref.dataset_name][sample_ref.row_index]
            yield str(getattr(record, "image_path", "")).strip()
        return
    if not isinstance(records, Sequence):
        raise TypeError("SFT cost planning requires indexable dataset records.")
    for row_index in range(len(records)):
        yield str(getattr(records[row_index], "image_path", "")).strip()


def _image_asset_identity(image_path: str) -> tuple[str, tuple[int, ...], tuple[int, int]]:
    """Read a stable, header-only identity for one image asset.

    Content bytes are deliberately not hashed at runtime: doing so would turn planner
    startup into a full image-store scan on every data rank. Canonical path, inode/stat
    metadata and dimensions bind normal immutable-snapshot changes and, importantly,
    every field that can change the image cost. A mutation racing this scan is rejected.
    """

    path = Path(image_path).expanduser().resolve(strict=True)
    before = path.stat()
    with Image.open(path) as image:
        width, height = image.size
    after = path.stat()
    before_stat = (
        int(before.st_dev),
        int(before.st_ino),
        int(before.st_size),
        int(before.st_mtime_ns),
        int(before.st_ctime_ns),
    )
    after_stat = (
        int(after.st_dev),
        int(after.st_ino),
        int(after.st_size),
        int(after.st_mtime_ns),
        int(after.st_ctime_ns),
    )
    if before_stat != after_stat:
        raise RuntimeError(f"Image asset changed while building cost manifest: {path}")
    return str(path), after_stat, (int(width), int(height))


def validate_sft_cost_planning_dataset(dataset: Any) -> None:
    if not hasattr(dataset, "get_planning_item"):
        raise TypeError(
            "SFT cost-aware batching requires a dataset with get_planning_item()."
        )
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
            raise ValueError(
                "ShaftSampleCost.supervised_tokens cannot exceed llm_tokens."
            )
        if int(self.vision_patches) < 0:
            raise ValueError("ShaftSampleCost.vision_patches must be >= 0.")
        if self.loss_weight_sum is not None:
            value = float(self.loss_weight_sum)
            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    "ShaftSampleCost.loss_weight_sum must be finite and >= 0 when set."
                )


class ShaftSampleCostProvider(Protocol):
    fingerprint: str

    def __call__(self, sample_ref: ShaftSampleRef) -> ShaftSampleCost: ...


class ShaftRowInvariantCostProvider:
    """Small immutable mapping for costs that cannot vary across logical draws.

    The key deliberately excludes draw context. Do not use this adapter for prompt
    rotation or any transform whose cost can change for the same source row. Large
    production plans should use a draw-indexed memory-mapped provider.
    """

    def __init__(
        self,
        costs: Mapping[tuple[str, int], ShaftSampleCost],
        *,
        fingerprint: str | None = None,
    ) -> None:
        self._costs = dict(costs)
        if not self._costs:
            raise ValueError(
                "ShaftRowInvariantCostProvider requires at least one sample cost."
            )
        if fingerprint is None:
            payload = tuple(sorted(self._costs.items()))
            fingerprint = hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()
        self.fingerprint = str(fingerprint).strip()
        if not self.fingerprint:
            raise ValueError(
                "ShaftRowInvariantCostProvider fingerprint must not be empty."
            )

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
        image_size_cache_size: int = 8192,
    ) -> None:
        validate_sft_cost_planning_dataset(dataset)
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
        self.image_size_cache_size = max(int(image_size_cache_size), 0)
        self._image_sizes: OrderedDict[str, tuple[int, int]] = OrderedDict()
        self._manifest_image_sizes: dict[str, tuple[int, int]] = {}
        image_asset_manifest = self._build_image_asset_manifest()
        processor_cost_semantics = model_adapter.processor_cost_semantics_signature(
            processor=processor,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        fingerprint_payload = (
            "shaft-sft-runtime-cost-v6",
            sft_cost_planning_source_fingerprint(dataset),
            image_asset_manifest,
            str(getattr(model_adapter, "model_type", "")),
            str(getattr(model_adapter, "model_name_or_path", "")),
            str(getattr(model_adapter, "template_type", "")),
            processor_cost_semantics,
            f"{type(processor).__module__}.{type(processor).__qualname__}",
            str(transformers_version),
            repr(getattr(processor, "chat_template", None)),
            _tokenizer_artifact_fingerprint(tokenizer),
            f"{type(template).__module__}.{type(template).__qualname__}",
            repr(getattr(template, "template_meta", None)),
            self.min_pixels,
            self.max_pixels,
            self.max_length,
            self.add_eos_token,
            self.loss_scale_name,
        )
        self.fingerprint = hashlib.sha256(
            repr(fingerprint_payload).encode("utf-8")
        ).hexdigest()

    def _build_image_asset_manifest(self) -> str:
        digest = hashlib.sha256(b"shaft-image-asset-manifest-v2")
        canonical_paths: set[str] = set()
        for image_path in _iter_planned_image_paths(self.dataset):
            if not image_path:
                raise ValueError("SFT cost estimation requires image_path for every row.")
            canonical_paths.add(
                str(Path(image_path).expanduser().resolve(strict=True))
            )
        for canonical_path in sorted(canonical_paths):
            canonical_path, stat_identity, image_size = _image_asset_identity(
                canonical_path
            )
            digest.update(b"\0")
            digest.update(
                repr(
                    (
                        canonical_path,
                        stat_identity,
                        image_size,
                    )
                ).encode("utf-8")
            )
            self._manifest_image_sizes[canonical_path] = image_size
        digest.update(b"\0")
        digest.update(str(len(canonical_paths)).encode("utf-8"))
        return digest.hexdigest()

    def __call__(self, sample_ref: ShaftSampleRef) -> ShaftSampleCost:
        item = self.dataset.get_planning_item(sample_ref)
        target_text = str(item["target_text"])
        supervision_plan = self.template.build_supervision_plan(
            item=item,
            target_text=target_text,
            renderer=self.renderer,
            loss_scale_name=self.loss_scale_name,
        )
        rendered_ids = (
            supervision_plan.rendered_prefix_token_ids
            or self.renderer.tokenize(supervision_plan.prompt_text)
        )
        image_path = str(item.get("image_path", "")).strip()
        if not image_path:
            raise ValueError(
                "SFT cost estimation requires image_path for "
                f"dataset={sample_ref.dataset_name!r}, row={sample_ref.row_index}."
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
        return ShaftSampleCost(
            llm_tokens=supervision_cost.llm_tokens,
            supervised_tokens=supervision_cost.supervised_tokens,
            vision_patches=image_estimate.vision_patches,
            loss_weight_sum=supervision_cost.loss_weight_sum,
            exact=bool(image_estimate.exact),
        )

    def _get_image_size(self, image_path: str) -> tuple[int, int]:
        cached = self._image_sizes.get(image_path)
        if cached is not None:
            self._image_sizes.move_to_end(image_path)
            return cached
        path = Path(image_path).expanduser().resolve(strict=True)
        manifest_size = self._manifest_image_sizes.get(str(path))
        if manifest_size is not None:
            self._remember_image_size(image_path, manifest_size)
            return manifest_size
        with Image.open(path) as image:
            width, height = image.size
        resolved = (int(width), int(height))
        self._remember_image_size(image_path, resolved)
        return resolved

    def _remember_image_size(self, image_path: str, image_size: tuple[int, int]) -> None:
        if self.image_size_cache_size > 0:
            self._image_sizes[image_path] = image_size
            self._image_sizes.move_to_end(image_path)
            while len(self._image_sizes) > self.image_size_cache_size:
                self._image_sizes.popitem(last=False)
