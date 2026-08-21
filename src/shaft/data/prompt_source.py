from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
import hashlib
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Sequence

import yaml

from shaft.config import (
    PromptSourceConfig,
    PromptSourceFormulationSourceConfig,
    PromptSourceScheduleConfig,
)
from shaft.prompting import (
    ShaftPromptSchema,
    ShaftPromptTemplate,
    canonical_json,
    compile_prompt_variants,
)

from .dataset import SFTRecord
from .meta import ShaftDatasetMeta
from .sources import build_data_source
from .transforms import planning_safe_online_transform


PROMPT_SOURCE_VERSION = "shaft-prompt-source-v2"
PROMPT_SOURCE_POOL_VERSION = "shaft-prompt-source-pool-v2"
PROMPT_SOURCE_SCHEDULE_VERSION = "shaft-prompt-source-schedule-v1"
PROMPT_SOURCE_SELECTION_VERSION = "shaft-prompt-source-selection-v2"
FORMULATION_RECORD_STORE_VERSION = "shaft-formulation-record-store-v1"
_PROMPT_SOURCE_TARGETS_KEY = "_shaft_prompt_source_targets"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _finite_nonnegative_weight(value: Any, *, source: str) -> float:
    try:
        weight = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} must be a finite number >= 0.") from exc
    if not math.isfinite(weight) or weight < 0:
        raise ValueError(f"{source} must be a finite number >= 0.")
    return weight


def _nonnegative_integer(value: Any, *, source: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{source} must be an integer >= 0.")
    if value < 0:
        raise ValueError(f"{source} must be an integer >= 0.")
    return value


@dataclass(frozen=True, slots=True)
class ShaftTaskFormulation:
    formulation_id: str
    sampling_weight: float
    prompt_variants: tuple[ShaftPromptTemplate, ...]


@dataclass(frozen=True, slots=True)
class ShaftPromptSourcePool:
    pool_id: str
    version: str
    source_path: str
    schema: ShaftPromptSchema
    formulations: tuple[ShaftTaskFormulation, ...]
    explicit_formulations: bool
    fingerprint: str


class _ShaftPromptSourceRecord(SFTRecord):
    """PromptSource-private bridge from aligned stores to online selection."""

    def __init__(
        self,
        *,
        canonical: SFTRecord,
        targets: Mapping[str, str],
    ) -> None:
        super().__init__(
            image_paths=canonical.image_paths,
            target_text="",
            dataset_name=canonical.dataset_name,
            sample_id=canonical.sample_id,
            messages=canonical.messages,
            system_prompt=canonical.system_prompt,
            user_prompt=canonical.user_prompt,
            prompt_args=canonical.prompt_args,
            extra=canonical.extra,
        )
        self._targets = MappingProxyType(
            {
                str(formulation_id): str(target_text)
                for formulation_id, target_text in targets.items()
            }
        )

    def runtime_sample_fields(self) -> Mapping[str, Any]:
        return {_PROMPT_SOURCE_TARGETS_KEY: dict(self._targets)}


class ShaftFormulationRecordStore(Sequence[SFTRecord]):
    """Align materialized v5.7-format SFT rows by formulation and row identity."""

    _IDENTITY_FIELDS = (
        "image_paths",
        "dataset_name",
        "sample_id",
        "messages",
        "system_prompt",
        "user_prompt",
        "prompt_args",
        "extra",
    )

    def __init__(self, stores: Mapping[str, Sequence[SFTRecord]]) -> None:
        normalized: dict[str, Sequence[SFTRecord]] = {}
        for raw_formulation_id, store in stores.items():
            formulation_id = str(raw_formulation_id).strip()
            if not formulation_id:
                raise ValueError("Formulation record stores need non-empty formulation ids.")
            if formulation_id in normalized:
                raise ValueError(
                    f"Duplicate normalized formulation record store id {formulation_id!r}."
                )
            normalized[formulation_id] = store
        if not normalized:
            raise ValueError("Formulation record stores need non-empty formulation ids.")
        self.formulation_ids = tuple(sorted(normalized))
        self.stores = MappingProxyType(
            {formulation_id: normalized[formulation_id] for formulation_id in self.formulation_ids}
        )
        lengths = {formulation_id: len(store) for formulation_id, store in self.stores.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(
                "Materialized formulation sources must have identical row counts: "
                f"{lengths}."
            )
        self._length = next(iter(lengths.values()))
        if self._length <= 0:
            raise ValueError("Materialized formulation sources cannot be empty.")
        self._validate_alignment()
        self.fingerprint = _sha256_text(
            canonical_json(
                {
                    "version": FORMULATION_RECORD_STORE_VERSION,
                    "stores": {
                        formulation_id: self._store_fingerprint(store)
                        for formulation_id, store in self.stores.items()
                    },
                }
            )
        )

    @classmethod
    def _store_fingerprint(cls, store: Sequence[SFTRecord]) -> str:
        explicit = str(getattr(store, "fingerprint", "")).strip()
        if explicit:
            return explicit
        digest = hashlib.sha256(b"shaft-inline-formulation-store-v1\0")
        for record in store:
            payload = {
                field_name: getattr(record, field_name)
                for field_name in cls._IDENTITY_FIELDS
            }
            payload["image_paths"] = list(record.image_paths)
            payload["target_text"] = record.target_text
            encoded = canonical_json(payload).encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        return digest.hexdigest()

    def _validate_alignment(self) -> None:
        canonical_id = self.formulation_ids[0]
        canonical_store = self.stores[canonical_id]
        for row_index in range(self._length):
            canonical = canonical_store[row_index]
            if not str(canonical.target_text).strip():
                raise ValueError(
                    f"Materialized formulation {canonical_id!r} has empty target_text "
                    f"at row {row_index}."
                )
            if getattr(canonical, "formulation_targets", None):
                raise ValueError("Nested formulation_targets are not supported.")
            for formulation_id in self.formulation_ids[1:]:
                candidate = self.stores[formulation_id][row_index]
                if not str(candidate.target_text).strip():
                    raise ValueError(
                        f"Materialized formulation {formulation_id!r} has empty target_text "
                        f"at row {row_index}."
                    )
                if getattr(candidate, "formulation_targets", None):
                    raise ValueError("Nested formulation_targets are not supported.")
                mismatched = [
                    field_name
                    for field_name in self._IDENTITY_FIELDS
                    if getattr(candidate, field_name) != getattr(canonical, field_name)
                ]
                if mismatched:
                    raise ValueError(
                        "Materialized formulation sources must align exactly by row; "
                        f"row={row_index}, canonical={canonical_id!r}, "
                        f"candidate={formulation_id!r}, mismatched={mismatched}."
                    )

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int | slice) -> SFTRecord | list[SFTRecord]:
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]
        position = int(index)
        if position < 0:
            position += len(self)
        if position < 0 or position >= len(self):
            raise IndexError(position)
        canonical = self.stores[self.formulation_ids[0]][position]
        return _ShaftPromptSourceRecord(
            canonical=canonical,
            targets={
                formulation_id: self.stores[formulation_id][position].target_text
                for formulation_id in self.formulation_ids
            },
        )


class ShaftPromptSourceSchedule:
    """Resolve formulation weights from an absolute dataset-local draw index."""

    def __init__(
        self,
        *,
        formulation_ids: tuple[str, ...],
        static_weights: tuple[float, ...],
        config: PromptSourceScheduleConfig,
    ) -> None:
        self.formulation_ids = tuple(str(value) for value in formulation_ids)
        self.static_weights = tuple(
            _finite_nonnegative_weight(value, source="formulation sampling_weight")
            for value in static_weights
        )
        if len(self.formulation_ids) != len(self.static_weights):
            raise ValueError("PromptSource formulation ids and weights must have equal length.")
        if not self.formulation_ids or len(set(self.formulation_ids)) != len(
            self.formulation_ids
        ):
            raise ValueError("PromptSource formulation ids must be non-empty and unique.")
        if not any(self.static_weights):
            raise ValueError("PromptSource needs at least one positive formulation weight.")
        self.interpolation = str(config.interpolation).strip().lower()
        if self.interpolation not in {"step", "linear"}:
            raise ValueError("PromptSource schedule interpolation must be 'step' or 'linear'.")

        expected_ids = set(self.formulation_ids)
        draws: list[int] = []
        weights: list[tuple[float, ...]] = []
        for index, point in enumerate(config.points):
            source_draw = _nonnegative_integer(
                point.source_draw,
                source=f"PromptSource schedule points[{index}].source_draw",
            )
            if index == 0 and source_draw != 0:
                raise ValueError("PromptSource schedule first source_draw must be 0.")
            if draws and source_draw <= draws[-1]:
                raise ValueError(
                    "PromptSource schedule source_draw values must be strictly increasing."
                )
            actual_ids = set(point.weights)
            if actual_ids != expected_ids:
                missing = sorted(expected_ids - actual_ids)
                extra = sorted(actual_ids - expected_ids)
                raise ValueError(
                    "PromptSource schedule point must define every formulation exactly once: "
                    f"missing={missing}, extra={extra}."
                )
            resolved = tuple(
                _finite_nonnegative_weight(
                    point.weights[formulation_id],
                    source=(
                        f"PromptSource schedule weight at source_draw={source_draw} for "
                        f"{formulation_id!r}"
                    ),
                )
                for formulation_id in self.formulation_ids
            )
            if not any(resolved):
                raise ValueError(
                    f"PromptSource schedule point at source_draw={source_draw} needs a "
                    "positive weight."
                )
            draws.append(source_draw)
            weights.append(resolved)
        self._draws = tuple(draws)
        self._weights = tuple(weights)
        self.fingerprint = _sha256_text(
            canonical_json(
                {
                    "version": PROMPT_SOURCE_SCHEDULE_VERSION,
                    "formulation_ids": list(self.formulation_ids),
                    "static_weights": list(self.static_weights),
                    "interpolation": self.interpolation,
                    "points": [
                        {"source_draw": draw, "weights": list(point_weights)}
                        for draw, point_weights in zip(
                            self._draws,
                            self._weights,
                            strict=True,
                        )
                    ],
                }
            )
        )

    def weights_at(self, source_draw_id: int) -> tuple[float, ...]:
        source_draw_id = _nonnegative_integer(
            source_draw_id,
            source="PromptSource source_draw_id",
        )
        if not self._draws:
            return self.static_weights
        left = bisect_right(self._draws, source_draw_id) - 1
        if left < 0:  # construction requires the first point at zero
            raise RuntimeError("PromptSource schedule has no point for this draw.")
        if self.interpolation == "step" or left == len(self._draws) - 1:
            return self._weights[left]
        left_draw = self._draws[left]
        right_draw = self._draws[left + 1]
        ratio = (source_draw_id - left_draw) / (right_draw - left_draw)
        return tuple(
            left_weight + ratio * (right_weight - left_weight)
            for left_weight, right_weight in zip(
                self._weights[left],
                self._weights[left + 1],
                strict=True,
            )
        )

@dataclass(frozen=True, slots=True)
class _DatasetPromptSource:
    dataset_name: str
    apply_to: str
    seed: int
    pool: ShaftPromptSourcePool
    schedule: ShaftPromptSourceSchedule
    formulation_sources: Mapping[str, PromptSourceFormulationSourceConfig]

    def active_for(self, split: str) -> bool:
        return self.apply_to == "all" or split == "train"


def load_prompt_source_pool(path: str | Path) -> ShaftPromptSourcePool:
    pool_path = Path(path)
    payload = yaml.safe_load(pool_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"PromptSource pool must contain a mapping: {pool_path}")
    unknown = sorted(set(payload) - {"metadata", "arguments", "prompts", "formulations"})
    if unknown:
        raise ValueError(f"Unknown PromptSource pool keys {unknown}: {pool_path}")
    metadata = payload.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise ValueError(f"PromptSource metadata must be a mapping: {pool_path}")
    pool_id = str(metadata.get("id") or "").strip()
    version = str(metadata.get("version") or "").strip()
    if not pool_id:
        raise ValueError(f"Missing prompt pool id in {pool_path}.")
    if not version:
        raise ValueError(f"Missing prompt pool version in {pool_path}.")
    has_prompts = payload.get("prompts") is not None
    has_formulations = payload.get("formulations") is not None
    if has_prompts == has_formulations:
        raise ValueError(
            f"PromptSource pool must define exactly one of prompts or formulations: {pool_path}"
        )
    raw_arguments = payload.get("arguments")
    schema = ShaftPromptSchema.from_mapping(raw_arguments, source=str(pool_path))

    if has_prompts:
        prompt_variants = tuple(
            compile_prompt_variants(
                payload["prompts"],
                arguments=schema,
                metadata=metadata,
                pool_id=pool_id,
                version=version,
                source=str(pool_path),
            )
        )
        formulations = (
            ShaftTaskFormulation(
                formulation_id="default",
                sampling_weight=1.0,
                prompt_variants=prompt_variants,
            ),
        )
    else:
        raw_formulations = payload["formulations"]
        if not isinstance(raw_formulations, list) or not raw_formulations:
            raise ValueError(
                f"PromptSource formulations must be a non-empty list: {pool_path}"
            )
        compiled: list[ShaftTaskFormulation] = []
        seen_ids: set[str] = set()
        for index, raw in enumerate(raw_formulations):
            source = f"{pool_path}:formulations[{index}]"
            if not isinstance(raw, dict):
                raise ValueError(f"PromptSource formulation must be a mapping: {source}")
            unknown_formulation_keys = sorted(
                set(raw) - {"id", "sampling_weight", "prompts"}
            )
            if unknown_formulation_keys:
                raise ValueError(
                    f"Unknown PromptSource formulation keys {unknown_formulation_keys}: {source}"
                )
            formulation_id = str(raw.get("id") or "").strip()
            if not formulation_id:
                raise ValueError(f"PromptSource formulation is missing id: {source}")
            if formulation_id in seen_ids:
                raise ValueError(
                    f"Duplicate PromptSource formulation id {formulation_id!r}: {pool_path}"
                )
            seen_ids.add(formulation_id)
            sampling_weight = _finite_nonnegative_weight(
                raw.get("sampling_weight", 1.0),
                source=f"{source}.sampling_weight",
            )
            prompt_variants = tuple(
                compile_prompt_variants(
                    raw.get("prompts"),
                    arguments=schema,
                    metadata=metadata,
                    pool_id=f"{pool_id}.{formulation_id}",
                    version=version,
                    source=f"{pool_path}#{formulation_id}",
                )
            )
            compiled.append(
                ShaftTaskFormulation(
                    formulation_id=formulation_id,
                    sampling_weight=sampling_weight,
                    prompt_variants=prompt_variants,
                )
            )
        if not any(formulation.sampling_weight > 0 for formulation in compiled):
            raise ValueError(
                f"PromptSource needs at least one positive formulation weight: {pool_path}"
            )
        formulations = tuple(compiled)

    fingerprint_payload = {
        "version": PROMPT_SOURCE_POOL_VERSION,
        "pool_id": pool_id,
        "pool_version": version,
        "explicit_formulations": has_formulations,
        "schema_sha256": schema.fingerprint,
        "formulations": [
            {
                "id": formulation.formulation_id,
                "sampling_weight": formulation.sampling_weight,
                "prompts": [
                    {
                        "id": prompt.variant_id,
                        "weight": prompt.sampling_weight,
                        "system_prompt": prompt.system_prompt,
                        "program_sha256": prompt.program.program_sha256,
                    }
                    for prompt in formulation.prompt_variants
                ],
            }
            for formulation in formulations
        ],
    }
    return ShaftPromptSourcePool(
        pool_id=pool_id,
        version=version,
        source_path=str(pool_path),
        schema=schema,
        formulations=formulations,
        explicit_formulations=has_formulations,
        fingerprint=_sha256_text(canonical_json(fingerprint_payload)),
    )


class ShaftPromptSource:
    """Select one materialized task formulation and one prompt variant."""

    def __init__(self, sources: Mapping[str, _DatasetPromptSource]) -> None:
        self._sources = MappingProxyType(dict(sources))
        fingerprint_payload = {
            "version": PROMPT_SOURCE_VERSION,
            "selection_version": PROMPT_SOURCE_SELECTION_VERSION,
            "sources": {
                dataset_name: {
                    "apply_to": source.apply_to,
                    "seed": source.seed,
                    "pool_sha256": source.pool.fingerprint,
                    "schedule_sha256": source.schedule.fingerprint,
                }
                for dataset_name, source in sorted(self._sources.items())
            },
        }
        self.fingerprint = _sha256_text(canonical_json(fingerprint_payload))
        planning_safe_online_transform(self, fingerprint=self.fingerprint)

    def record_validation_fingerprint(self, dataset_name: str, *, split: str) -> str:
        source = self._sources.get(str(dataset_name))
        active = source is not None and source.active_for(str(split))
        return _sha256_text(
            canonical_json(
                {
                    "version": PROMPT_SOURCE_VERSION,
                    "dataset_name": str(dataset_name),
                    "split": str(split),
                    "active": active,
                    "source_sha256": (
                        source.pool.fingerprint if active and source is not None else "materialized"
                    ),
                    "schedule_sha256": (
                        source.schedule.fingerprint if active and source is not None else ""
                    ),
                }
            )
        )

    def prepare_records(
        self,
        dataset_meta: ShaftDatasetMeta,
        *,
        split: str,
        cache_dir: str | None,
        offline_pipeline: Callable[[Sequence[Any]], Sequence[Any]],
    ) -> Sequence[Any] | None:
        """Load an aligned formulation store, or defer to the ordinary data source."""

        source = self._sources.get(str(dataset_meta.dataset_name))
        if source is None or not source.active_for(str(split)):
            return None
        if not source.pool.explicit_formulations:
            if source.formulation_sources:
                raise ValueError(
                    f"PromptSource dataset={dataset_meta.dataset_name!r} defines "
                    "formulation_sources, but its pool uses top-level prompts."
                )
            return None
        expected_ids = {
            formulation.formulation_id for formulation in source.pool.formulations
        }
        actual_ids = set(source.formulation_sources)
        if actual_ids != expected_ids:
            raise ValueError(
                "PromptSource formulation_sources must match the pool exactly for "
                f"dataset={dataset_meta.dataset_name!r}: "
                f"missing={sorted(expected_ids - actual_ids)}, "
                f"extra={sorted(actual_ids - expected_ids)}."
            )
        stores: dict[str, Sequence[SFTRecord]] = {}
        for formulation in source.pool.formulations:
            formulation_source = source.formulation_sources[formulation.formulation_id]
            if split == "train":
                paths = tuple(formulation_source.train_paths) or (
                    ()
                    if not formulation_source.train_path
                    else (str(formulation_source.train_path),)
                )
            else:
                paths = tuple(formulation_source.val_paths) or (
                    ()
                    if not formulation_source.val_path
                    else (str(formulation_source.val_path),)
                )
            if not paths:
                raise ValueError(
                    f"PromptSource formulation {formulation.formulation_id!r} has no "
                    f"materialized {split} source for dataset {dataset_meta.dataset_name!r}."
                )
            formulation_meta = replace(
                dataset_meta,
                train_paths=(tuple(paths) if split == "train" else ()),
                val_paths=(tuple(paths) if split == "val" else ()),
            )
            validation_fingerprint = _sha256_text(
                canonical_json(
                    {
                        "record_validation_sha256": self.record_validation_fingerprint(
                            dataset_meta.dataset_name,
                            split=split,
                        ),
                        "formulation_id": formulation.formulation_id,
                    }
                )
            )
            source_impl = build_data_source(
                formulation_meta,
                cache_dir=cache_dir,
                record_validator=(
                    lambda record, current_split, current_id=formulation.formulation_id: (
                        self.validate_formulation_record(
                            record,
                            dataset_name=dataset_meta.dataset_name,
                            split=current_split,
                            formulation_id=current_id,
                        )
                    )
                ),
                validation_fingerprint=validation_fingerprint,
            )
            stores[formulation.formulation_id] = offline_pipeline(
                source_impl.load_split(split)
            )
        return ShaftFormulationRecordStore(stores)

    def validate_formulation_record(
        self,
        record: Any,
        *,
        dataset_name: str,
        split: str,
        formulation_id: str,
    ) -> None:
        source = self._sources.get(str(dataset_name))
        if source is None or not source.active_for(str(split)):
            raise ValueError(
                f"Dataset {dataset_name!r} has materialized formulation sources but no active "
                "PromptSource formulation pool."
            )
        if not source.pool.explicit_formulations:
            raise ValueError(
                f"Dataset {dataset_name!r} uses formulation_sources but its PromptSource pool "
                "defines only top-level prompts."
            )
        formulations = {
            formulation.formulation_id: formulation
            for formulation in source.pool.formulations
        }
        formulation = formulations.get(str(formulation_id))
        if formulation is None:
            raise ValueError(
                f"Unknown materialized formulation {formulation_id!r} for dataset "
                f"{dataset_name!r}; expected {sorted(formulations)}."
            )
        context = self._record_context(record, dataset_name=source.dataset_name)
        self._validate_pool_envelope(record, context=context)
        if getattr(record, "formulation_targets", None):
            raise ValueError(f"Formulation source rows cannot nest formulation targets ({context}).")
        if not str(getattr(record, "target_text", "") or "").strip():
            raise ValueError(
                f"PromptSource formulation {formulation.formulation_id!r} requires a "
                f"materialized target_text ({context})."
            )
        prompt_args = getattr(record, "prompt_args", {})
        for prompt in formulation.prompt_variants:
            if prompt.sampling_weight > 0:
                prompt.render(prompt_args, context=context)

    def validate_record(self, record: Any, *, dataset_name: str, split: str) -> None:
        source = self._sources.get(str(dataset_name))
        if source is None or not source.active_for(str(split)):
            self._validate_materialized(record, dataset_name=dataset_name)
            return
        self._validate_pool_record(record, source=source)

    @staticmethod
    def _record_context(record: Any, *, dataset_name: str) -> str:
        image_paths = tuple(getattr(record, "image_paths", ()) or ())
        sample_id = str(
            getattr(record, "sample_id", None) or (image_paths[0] if image_paths else "")
        )
        return f"dataset={dataset_name!r}, sample={sample_id!r}"

    def _validate_materialized(self, record: Any, *, dataset_name: str) -> None:
        context = self._record_context(record, dataset_name=dataset_name)
        if getattr(record, "prompt_args", None):
            raise ValueError(
                f"Sample has prompt_args but no configured PromptSource is active ({context})."
            )
        target_text = str(getattr(record, "target_text", "") or "")
        if not target_text.strip():
            raise ValueError(f"Materialized SFT sample is missing target_text ({context}).")

    def _validate_pool_record(
        self,
        record: Any,
        *,
        source: _DatasetPromptSource,
    ) -> None:
        context = self._record_context(record, dataset_name=source.dataset_name)
        self._validate_pool_envelope(record, context=context)
        prompt_args = getattr(record, "prompt_args", {})
        if source.pool.explicit_formulations:
            target_text = str(getattr(record, "target_text", "") or "")
            if target_text.strip():
                raise ValueError(
                    f"Aligned formulation records must not expose an unselected target_text "
                    f"({context})."
                )
            formulation_targets = dict(getattr(record, "_targets", {}) or {})
            expected_ids = {
                formulation.formulation_id for formulation in source.pool.formulations
            }
            actual_ids = set(formulation_targets)
            if actual_ids != expected_ids:
                raise ValueError(
                    "Materialized formulation targets must match the PromptSource pool exactly: "
                    f"missing={sorted(expected_ids - actual_ids)}, "
                    f"extra={sorted(actual_ids - expected_ids)} ({context})."
                )
            if any(not str(value).strip() for value in formulation_targets.values()):
                raise ValueError(f"Materialized formulation targets cannot be empty ({context}).")
        else:
            if getattr(record, "_targets", None):
                raise ValueError(
                    f"Top-level prompt pools do not accept formulation targets ({context})."
                )
            if not str(getattr(record, "target_text", "") or "").strip():
                raise ValueError(f"Materialized SFT sample is missing target_text ({context}).")
        for formulation in source.pool.formulations:
            for prompt in formulation.prompt_variants:
                if prompt.sampling_weight > 0:
                    prompt.render(prompt_args, context=context)

    @staticmethod
    def _validate_pool_envelope(record: Any, *, context: str) -> None:
        if getattr(record, "messages", None):
            raise ValueError(f"PromptSource pool mode forbids materialized messages ({context}).")
        if str(getattr(record, "system_prompt", "") or "").strip():
            raise ValueError(
                f"PromptSource pool mode forbids materialized system_prompt ({context})."
            )
        if str(getattr(record, "user_prompt", "") or "").strip():
            raise ValueError(f"PromptSource pool mode forbids materialized user_prompt ({context}).")

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        dataset_name = str(sample.get("dataset_name", "")).strip()
        split = str(sample.get("_split", "train")).strip().lower()
        source = self._sources.get(dataset_name)
        if source is None or not source.active_for(split):
            if "target_text" not in sample and "prompt_args" not in sample:
                return sample
            self._validate_materialized_sample(sample, dataset_name=dataset_name)
            return sample
        return self._resolve_pool_sample(sample, source=source)

    @staticmethod
    def _validate_materialized_sample(sample: dict[str, Any], *, dataset_name: str) -> None:
        sample_id = str(sample.get("sample_id", ""))
        context = f"dataset={dataset_name!r}, sample={sample_id!r}"
        if sample.get("prompt_args"):
            raise ValueError(
                f"Sample has prompt_args but no configured PromptSource is active ({context})."
            )
        if not str(sample.get("target_text", "")).strip():
            raise ValueError(f"Materialized SFT sample is missing target_text ({context}).")

    def _resolve_pool_sample(
        self,
        sample: dict[str, Any],
        *,
        source: _DatasetPromptSource,
    ) -> dict[str, Any]:
        dataset_name = source.dataset_name
        sample_id = str(sample.get("sample_id", "")).strip()
        if sample.get("messages"):
            raise ValueError(
                f"PromptSource pool mode forbids materialized messages "
                f"(dataset={dataset_name!r}, sample={sample_id!r})."
            )
        if str(sample.get("system_prompt", "") or "").strip():
            raise ValueError(
                f"PromptSource pool mode forbids materialized system_prompt "
                f"(dataset={dataset_name!r}, sample={sample_id!r})."
            )
        if str(sample.get("user_prompt", "")).strip():
            raise ValueError(
                f"PromptSource pool mode forbids materialized user_prompt "
                f"(dataset={dataset_name!r}, sample={sample_id!r})."
            )
        prompt_args = sample.get("prompt_args") or {}
        if not isinstance(prompt_args, dict):
            raise ValueError(
                f"prompt_args must be a JSON object (dataset={dataset_name!r}, "
                f"sample={sample_id!r})."
            )
        context = sample.get("_sample_context") or {}
        draw_id = _nonnegative_integer(
            context.get("draw_id", 0),
            source="PromptSource draw_id",
        )
        source_draw_id = _nonnegative_integer(
            context.get("source_draw_id", 0),
            source="PromptSource source_draw_id",
        )
        formulation_weights = source.schedule.weights_at(source_draw_id)
        formulation = _weighted_choice(
            source.pool.formulations,
            formulation_weights,
            key=(
                f"{PROMPT_SOURCE_SELECTION_VERSION}\n{source.seed}\n{dataset_name}\n"
                f"{sample_id}\n{draw_id}\n{source_draw_id}\nformulation"
            ),
        )
        prompt = _weighted_choice(
            formulation.prompt_variants,
            tuple(variant.sampling_weight for variant in formulation.prompt_variants),
            key=(
                f"{PROMPT_SOURCE_SELECTION_VERSION}\n{source.seed}\n{dataset_name}\n"
                f"{sample_id}\n{draw_id}\n{source_draw_id}\n"
                f"{formulation.formulation_id}\nprompt_variant"
            ),
        )
        render_context = (
            f"dataset={dataset_name!r}, sample={sample_id!r}, draw_id={draw_id}, "
            f"source_draw_id={source_draw_id}, pool={source.pool.pool_id!r}, "
            f"version={source.pool.version!r}, formulation={formulation.formulation_id!r}, "
            f"variant={prompt.variant_id!r}"
        )
        user_prompt, prompt_audit = prompt.render_with_audit(
            prompt_args,
            context=render_context,
        )
        if source.pool.explicit_formulations:
            formulation_targets = sample.get(_PROMPT_SOURCE_TARGETS_KEY) or {}
            if not isinstance(formulation_targets, dict):
                raise ValueError(
                    f"PromptSource runtime targets must be a mapping ({render_context})."
                )
            target_text = str(formulation_targets.get(formulation.formulation_id, "") or "")
            if not target_text.strip():
                raise ValueError(
                    f"Missing materialized target_text for formulation "
                    f"{formulation.formulation_id!r} ({render_context})."
                )
        else:
            target_text = str(sample.get("target_text", "") or "")
            if not target_text.strip():
                raise ValueError(f"Materialized target_text is missing ({render_context}).")

        updated = dict(sample)
        updated.pop(_PROMPT_SOURCE_TARGETS_KEY, None)
        updated["system_prompt"] = prompt.system_prompt
        updated["user_prompt"] = user_prompt
        updated["target_text"] = target_text
        extra = dict(updated.get("extra") or {})
        extra["prompt_source"] = {
            "pool_id": source.pool.pool_id,
            "pool_version": source.pool.version,
            "formulation_id": formulation.formulation_id,
            "variant_id": str(prompt.variant_id or ""),
            "draw_id": draw_id,
            "source_draw_id": source_draw_id,
            "formulation_weight": formulation_weights[
                source.pool.formulations.index(formulation)
            ],
            "variant_weight": prompt.sampling_weight,
            "prompt_program_sha256": prompt_audit["program_sha256"],
            "arguments_sha256": prompt_audit["args_sha256"],
            "user_prompt_sha256": _sha256_text(user_prompt),
            "target_text_sha256": _sha256_text(target_text),
        }
        updated["extra"] = extra
        return updated


def _weighted_choice(
    candidates: Sequence[Any],
    weights: Sequence[float],
    *,
    key: str,
) -> Any:
    if len(candidates) != len(weights) or not candidates:
        raise ValueError("Weighted PromptSource candidates and weights must be non-empty and aligned.")
    resolved = tuple(
        _finite_nonnegative_weight(weight, source="PromptSource sampling weight")
        for weight in weights
    )
    max_weight = max(resolved)
    if max_weight <= 0:
        raise ValueError("PromptSource sampling needs at least one positive weight.")
    scaled = tuple(weight / max_weight for weight in resolved)
    total = sum(scaled)
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    random_bits = int.from_bytes(digest[:8], "big") >> 11
    threshold = (random_bits / float(1 << 53)) * total
    cumulative = 0.0
    fallback = next(
        candidate
        for candidate, weight in zip(reversed(candidates), reversed(resolved), strict=True)
        if weight > 0
    )
    for candidate, weight in zip(candidates, scaled, strict=True):
        cumulative += weight
        if threshold < cumulative:
            return candidate
    return fallback


def build_prompt_source_resolver(
    configs: dict[str, PromptSourceConfig],
    *,
    default_seed: int,
) -> ShaftPromptSource:
    sources: dict[str, _DatasetPromptSource] = {}
    for dataset_name, config in sorted(configs.items()):
        pool = load_prompt_source_pool(config.path)
        schedule = ShaftPromptSourceSchedule(
            formulation_ids=tuple(
                formulation.formulation_id for formulation in pool.formulations
            ),
            static_weights=tuple(
                formulation.sampling_weight for formulation in pool.formulations
            ),
            config=config.schedule,
        )
        sources[str(dataset_name)] = _DatasetPromptSource(
            dataset_name=str(dataset_name),
            apply_to=str(config.apply_to).strip().lower(),
            seed=int(default_seed) if config.seed is None else int(config.seed),
            pool=pool,
            schedule=schedule,
            formulation_sources=MappingProxyType(dict(config.formulation_sources)),
        )
    return ShaftPromptSource(sources)
