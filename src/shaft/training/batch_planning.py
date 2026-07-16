from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any
import uuid

from transformers import TrainerCallback
from transformers.trainer import (
    OPTIMIZER_NAME,
    OPTIMIZER_NAME_BIN,
    SCHEDULER_NAME,
    TRAINER_STATE_NAME,
)
from transformers.trainer_callback import ExportableState

from shaft.data import (
    ShaftBatchPlanningSpec,
    ShaftPlannedBatchSampler,
    ShaftBatchPlanningState,
    resolve_local_pack_count_bounds,
)

from .distributed import (
    broadcast_object_from_rank_zero,
    get_world_size,
    is_rank_zero,
)
from .input_contract import ShaftTrainInputContract
from .resume_contract import ShaftTrainingResumeContract
from shaft.utils.contract_schema import (
    json_bool,
    json_int,
    json_list,
    json_number,
    json_optional_int,
    json_optional_string,
    json_string,
    load_strict_json,
    require_exact_keys,
    require_json_mapping,
)


BATCHING_RUN_METADATA_FILENAME = "shaft_batching_run_metadata.json"
BATCH_PLANNING_CALLBACK_NAME = "ShaftBatchPlanningCallback"
BATCHING_METADATA_CALLBACK_NAME = "ShaftBatchingMetadataCallback"
BATCH_PLANNING_CHECKPOINT_COMMIT_EXTENSION = "batch_planning"
_BATCH_PLANNING_CHECKPOINT_COMMIT_VERSION = "shaft-batch-planning-commit-v1"
_BATCH_CONTRACT_VERSION = "shaft-batch-contract-v3"
_BATCHING_RUN_METADATA_VERSION = "shaft-batching-run-metadata-v1"

_BATCH_CONTRACT_KEYS = frozenset(
    {
        "version",
        "grouping",
        "cardinality",
        "packing",
        "layout",
        "per_device_microbatch_size",
        "data_world_size",
        "gradient_accumulation_steps",
        "buffer_size",
        "max_tokens_per_microbatch",
        "max_sequence_length",
        "resource_budgets",
    }
)
_BATCHING_RUN_METADATA_KEYS = frozenset(
    {
        "version",
        "grouping",
        "cardinality",
        "packing",
        "layout",
        "per_device_train_batch_size",
        "data_world_size",
        "gradient_accumulation_steps",
        "global_pack_count",
        "optimizer_pack_count",
        "local_pack_count_range",
        "global_pack_count_range",
        "optimizer_pack_count_range",
        "min_pixels",
        "max_pixels",
        "source_weights",
        "media_snapshot_id",
        "buffer_size",
        "cost_cache_size",
        "max_tokens_per_microbatch",
        "max_sequence_length",
        "resource_budgets",
        "batch_contract",
        "batch_contract_fingerprint",
        "planner_spec_fingerprint",
        "sample_execution_fingerprint",
        "train_input_contract",
        "train_input_contract_fingerprint",
        "training_resume_contract",
        "training_resume_contract_fingerprint",
    }
)

logger = logging.getLogger(__name__)


def _json_int_pair(
    payload: Mapping[str, Any],
    field_name: str,
    *,
    role: str,
) -> list[int]:
    values = json_list(payload, field_name, role=role)
    if len(values) != 2 or any(type(value) is not int for value in values):
        raise TypeError(f"{role}.{field_name} must be a two-item JSON integer list.")
    return values


@dataclass(frozen=True, slots=True)
class ShaftBatchContract:
    """Single resolved source of truth for physical training-batch semantics."""

    grouping: str
    cardinality: str
    packing: str
    layout: str
    per_device_microbatch_size: int
    data_world_size: int
    gradient_accumulation_steps: int
    buffer_size: int | None = None
    max_tokens_per_microbatch: int | None = None
    max_sequence_length: int | None = None
    resource_budgets: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        for field_name in (
            "grouping",
            "cardinality",
            "packing",
            "layout",
        ):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"ShaftBatchContract.{field_name} must not be empty.")
        if self.grouping not in {"none", "length", "bounded_cost"}:
            raise ValueError(f"Unsupported resolved batch grouping: {self.grouping!r}.")
        if self.cardinality not in {"fixed", "token_budget"}:
            raise ValueError(f"Unsupported resolved batch cardinality: {self.cardinality!r}.")
        if self.cardinality == "token_budget" and self.grouping != "bounded_cost":
            raise ValueError("Resolved token-budget cardinality requires bounded_cost grouping.")
        if self.packing not in {"none", "greedy"}:
            raise ValueError(f"Unsupported resolved batch packing: {self.packing!r}.")
        if self.layout not in {"padded", "varlen"}:
            raise ValueError(f"Unsupported resolved batch layout: {self.layout!r}.")
        for field_name in (
            "per_device_microbatch_size",
            "data_world_size",
            "gradient_accumulation_steps",
        ):
            if int(getattr(self, field_name)) <= 0:
                raise ValueError(f"ShaftBatchContract.{field_name} must be > 0.")
        names = [name for name, _ in self.resource_budgets]
        if names != sorted(names) or len(names) != len(set(names)):
            raise ValueError("ShaftBatchContract.resource_budgets must be sorted and unique.")
        if any(not str(name).strip() or int(value) <= 0 for name, value in self.resource_budgets):
            raise ValueError("ShaftBatchContract.resource_budgets names and values must be valid.")
        if self.is_planned:
            if self.buffer_size is None or int(self.buffer_size) <= 0:
                raise ValueError("Planned ShaftBatchContract requires buffer_size > 0.")
            required_buffer_size = int(self.data_world_size) * int(self.local_pack_count_bounds[0])
            if int(self.buffer_size) < required_buffer_size:
                raise ValueError(
                    "Planned ShaftBatchContract buffer must hold one complete global "
                    f"microbatch: buffer_size={self.buffer_size}, "
                    f"required={required_buffer_size}."
                )
        if self.is_bounded:
            if self.packing != "none" or self.layout != "padded":
                raise ValueError(
                    "bounded_cost grouping requires packing='none' and layout='padded'."
                )
            if self.max_tokens_per_microbatch is None or int(self.max_tokens_per_microbatch) <= 0:
                raise ValueError(
                    "Bounded ShaftBatchContract requires max_tokens_per_microbatch > 0."
                )
            if self.max_sequence_length is not None:
                raise ValueError("Bounded ShaftBatchContract cannot carry max_sequence_length.")
        elif self.grouping == "length":
            if self.cardinality != "fixed":
                raise ValueError("Length grouping requires fixed cardinality.")
            if self.max_sequence_length is None or int(self.max_sequence_length) <= 0:
                raise ValueError("Length ShaftBatchContract requires max_sequence_length > 0.")
            if self.max_tokens_per_microbatch is not None:
                raise ValueError(
                    "Length ShaftBatchContract derives its local token capacity from "
                    "per_device_microbatch_size * max_sequence_length."
                )
            if self.packing == "greedy" and self.layout != "varlen":
                raise ValueError("Greedy packing requires layout='varlen'.")
            if self.packing == "greedy" and not dict(self.resource_budgets).get("vision_patches"):
                raise ValueError("Greedy multimodal packing requires a vision_patches hard guard.")
        elif (
            any(
                value is not None
                for value in (
                    self.buffer_size,
                    self.max_tokens_per_microbatch,
                    self.max_sequence_length,
                )
            )
            or self.resource_budgets
        ):
            raise ValueError("Non-bounded ShaftBatchContract cannot carry bounded planner fields.")
        if self.grouping == "none" and (
            self.cardinality != "fixed" or self.packing != "none" or self.layout != "padded"
        ):
            raise ValueError(
                "Unplanned batching requires fixed cardinality, packing='none', "
                "and layout='padded'."
            )

    @property
    def global_pack_count(self) -> int:
        """Return the exact physical-pack count in one global microbatch."""

        if self.cardinality != "fixed":
            raise ValueError(
                "global_pack_count is not exact for token-budget cardinality; "
                "use global_pack_count_bounds."
            )
        return int(self.per_device_microbatch_size) * int(self.data_world_size)

    @property
    def optimizer_pack_count(self) -> int:
        """Return the exact physical-pack count per optimizer step."""

        if self.cardinality != "fixed":
            raise ValueError(
                "optimizer_pack_count is not exact for token-budget cardinality; "
                "use optimizer_pack_count_bounds."
            )
        return self.global_pack_count * int(self.gradient_accumulation_steps)

    @property
    def local_pack_count_bounds(self) -> tuple[int, int]:
        return resolve_local_pack_count_bounds(
            self.cardinality,
            self.per_device_microbatch_size,
        )

    @property
    def global_pack_count_bounds(self) -> tuple[int, int]:
        minimum, maximum = self.local_pack_count_bounds
        world_size = int(self.data_world_size)
        return minimum * world_size, maximum * world_size

    @property
    def optimizer_pack_count_bounds(self) -> tuple[int, int]:
        minimum, maximum = self.global_pack_count_bounds
        accumulation = int(self.gradient_accumulation_steps)
        return minimum * accumulation, maximum * accumulation

    @property
    def is_bounded(self) -> bool:
        return self.grouping == "bounded_cost"

    @property
    def is_planned(self) -> bool:
        return self.grouping in {"length", "bounded_cost"}

    @property
    def local_token_capacity(self) -> int | None:
        """Return the rank-local hard token cap used by the planner."""

        if self.is_bounded:
            return int(self.max_tokens_per_microbatch or 0)
        if self.grouping == "length":
            return int(self.per_device_microbatch_size) * int(self.max_sequence_length or 0)
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": _BATCH_CONTRACT_VERSION,
            "grouping": self.grouping,
            "cardinality": self.cardinality,
            "packing": self.packing,
            "layout": self.layout,
            "per_device_microbatch_size": int(self.per_device_microbatch_size),
            "data_world_size": int(self.data_world_size),
            "gradient_accumulation_steps": int(self.gradient_accumulation_steps),
            "buffer_size": self.buffer_size,
            "max_tokens_per_microbatch": self.max_tokens_per_microbatch,
            "max_sequence_length": self.max_sequence_length,
            "resource_budgets": {str(name): int(value) for name, value in self.resource_budgets},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ShaftBatchContract":
        role = "Batch contract"
        payload = require_json_mapping(payload, role=role)
        require_exact_keys(
            payload,
            expected=_BATCH_CONTRACT_KEYS,
            role=role,
        )
        version = json_string(payload, "version", role=role)
        if version != _BATCH_CONTRACT_VERSION:
            raise ValueError(f"Unsupported batch contract version: {version!r}.")
        resource_budgets = require_json_mapping(
            payload["resource_budgets"],
            role="Batch contract.resource_budgets",
        )
        normalized_resource_budgets: list[tuple[str, int]] = []
        for name, value in resource_budgets.items():
            if type(value) is not int:
                raise TypeError(f"Batch contract.resource_budgets.{name} must be a JSON integer.")
            normalized_resource_budgets.append((name, value))
        return cls(
            grouping=json_string(payload, "grouping", role=role),
            cardinality=json_string(payload, "cardinality", role=role),
            packing=json_string(payload, "packing", role=role),
            layout=json_string(payload, "layout", role=role),
            per_device_microbatch_size=json_int(
                payload,
                "per_device_microbatch_size",
                role=role,
            ),
            data_world_size=json_int(payload, "data_world_size", role=role),
            gradient_accumulation_steps=json_int(
                payload,
                "gradient_accumulation_steps",
                role=role,
            ),
            buffer_size=json_optional_int(payload, "buffer_size", role=role),
            max_tokens_per_microbatch=json_optional_int(
                payload,
                "max_tokens_per_microbatch",
                role=role,
            ),
            max_sequence_length=json_optional_int(
                payload,
                "max_sequence_length",
                role=role,
            ),
            resource_budgets=tuple(sorted(normalized_resource_budgets)),
        )

    @property
    def fingerprint(self) -> str:
        canonical = json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def finite_sample_plan_size(self, *, max_steps: int) -> int | None:
        """Resolve the finite draw count for ordinary step-bounded loaders."""

        if int(max_steps) < 0 or self.is_planned:
            return None
        return int(max_steps) * self.optimizer_pack_count


def build_batch_contract(*, config: Any, training_args: Any) -> ShaftBatchContract:
    batching = config.data.batching
    grouping = str(batching.grouping).strip().lower()
    bounded = grouping == "bounded_cost"
    planned = grouping in {"length", "bounded_cost"}
    return ShaftBatchContract(
        grouping=grouping,
        cardinality=str(batching.cardinality).strip().lower(),
        packing=str(batching.packing.mode).strip().lower(),
        layout=str(batching.layout).strip().lower(),
        per_device_microbatch_size=int(training_args.per_device_train_batch_size),
        data_world_size=max(int(training_args.world_size), 1),
        gradient_accumulation_steps=int(training_args.gradient_accumulation_steps),
        buffer_size=(int(batching.buffer_size) if planned else None),
        max_tokens_per_microbatch=(
            None
            if not bounded or batching.max_tokens_per_microbatch is None
            else int(batching.max_tokens_per_microbatch)
        ),
        max_sequence_length=(
            int(config.data.max_length)
            if grouping == "length" and config.data.max_length is not None
            else None
        ),
        resource_budgets=tuple(
            sorted(
                (str(name), int(value))
                for name, value in (batching.resource_budgets.items() if planned else ())
            )
        ),
    )


@dataclass(frozen=True, slots=True)
class ShaftBatchingRunMetadata:
    grouping: str
    cardinality: str
    packing: str
    layout: str
    per_device_train_batch_size: int
    data_world_size: int
    gradient_accumulation_steps: int
    min_pixels: int | None
    max_pixels: int | None
    source_weights: tuple[tuple[str, float], ...]
    media_snapshot_id: str | None = None
    buffer_size: int | None = None
    cost_cache_size: int | None = None
    max_tokens_per_microbatch: int | None = None
    max_sequence_length: int | None = None
    resource_budgets: tuple[tuple[str, int], ...] = ()
    batch_contract_fingerprint: str | None = None
    planner_spec_fingerprint: str | None = None
    sample_execution_fingerprint: str | None = None
    train_input_contract: ShaftTrainInputContract | None = None
    training_resume_contract: ShaftTrainingResumeContract | None = None

    def __post_init__(self) -> None:
        contract = self.batch_contract
        if contract.is_planned:
            if self.cost_cache_size is None or int(self.cost_cache_size) < 0:
                raise ValueError("Planned batching metadata requires cost_cache_size >= 0.")
        elif self.cost_cache_size is not None:
            raise ValueError("Unplanned batching metadata cannot carry cost_cache_size.")
        expected_fingerprint = contract.fingerprint
        if self.batch_contract_fingerprint is None:
            object.__setattr__(
                self,
                "batch_contract_fingerprint",
                expected_fingerprint,
            )
        elif self.batch_contract_fingerprint != expected_fingerprint:
            raise ValueError(
                "Batching metadata batch_contract_fingerprint differs from its "
                "canonical batch contract."
            )
        has_planner_fingerprint = bool(str(self.planner_spec_fingerprint or "").strip())
        if contract.is_planned != has_planner_fingerprint:
            raise ValueError(
                "Planned batching metadata requires exactly one planner spec "
                "fingerprint; unplanned metadata cannot carry one."
            )
        if self.train_input_contract is not None:
            sample_fingerprint = str(self.sample_execution_fingerprint or "").strip()
            if sample_fingerprint != self.train_input_contract.data_execution_fingerprint:
                raise ValueError(
                    "Batching metadata sample execution fingerprint differs from "
                    "its training input contract."
                )
        if self.training_resume_contract is not None:
            if self.training_resume_contract.batch_contract_fingerprint != expected_fingerprint:
                raise ValueError(
                    "Batching metadata training resume contract refers to a "
                    "different batch contract."
                )
            if (
                self.train_input_contract is not None
                and self.training_resume_contract.algorithm != self.train_input_contract.algorithm
            ):
                raise ValueError(
                    "Batching metadata training resume and input contracts refer "
                    "to different algorithms."
                )
            if self.train_input_contract is None:
                raise ValueError(
                    "Training resume metadata requires its composed training input contract."
                )
            if (
                self.training_resume_contract.train_input_contract_fingerprint
                != self.train_input_contract.fingerprint
            ):
                raise ValueError(
                    "Batching metadata training resume contract refers to a different "
                    "training input contract."
                )
            if (
                self.training_resume_contract.data_execution_fingerprint
                != self.train_input_contract.data_execution_fingerprint
            ):
                raise ValueError(
                    "Batching metadata training resume contract refers to a different "
                    "data execution contract."
                )

    @property
    def batch_contract(self) -> ShaftBatchContract:
        """Rebuild the executable contract instead of duplicating its validation."""

        return ShaftBatchContract(
            grouping=self.grouping,
            cardinality=self.cardinality,
            packing=self.packing,
            layout=self.layout,
            per_device_microbatch_size=self.per_device_train_batch_size,
            data_world_size=self.data_world_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            buffer_size=self.buffer_size,
            max_tokens_per_microbatch=self.max_tokens_per_microbatch,
            max_sequence_length=self.max_sequence_length,
            resource_budgets=self.resource_budgets,
        )

    @property
    def local_pack_count_bounds(self) -> tuple[int, int]:
        return self.batch_contract.local_pack_count_bounds

    @property
    def global_pack_count_bounds(self) -> tuple[int, int]:
        return self.batch_contract.global_pack_count_bounds

    @property
    def optimizer_pack_count_bounds(self) -> tuple[int, int]:
        return self.batch_contract.optimizer_pack_count_bounds

    def to_dict(self) -> dict[str, Any]:
        batch_contract = self.batch_contract
        local_min, local_max = self.local_pack_count_bounds
        global_bounds = list(self.global_pack_count_bounds)
        optimizer_bounds = list(self.optimizer_pack_count_bounds)
        return {
            "version": _BATCHING_RUN_METADATA_VERSION,
            "grouping": str(self.grouping),
            "cardinality": str(self.cardinality),
            "packing": str(self.packing),
            "layout": str(self.layout),
            "per_device_train_batch_size": int(self.per_device_train_batch_size),
            "data_world_size": int(self.data_world_size),
            "gradient_accumulation_steps": int(self.gradient_accumulation_steps),
            "global_pack_count": (
                global_bounds[0] if global_bounds[0] == global_bounds[1] else None
            ),
            "optimizer_pack_count": (
                optimizer_bounds[0] if optimizer_bounds[0] == optimizer_bounds[1] else None
            ),
            "local_pack_count_range": [local_min, local_max],
            "global_pack_count_range": global_bounds,
            "optimizer_pack_count_range": optimizer_bounds,
            "min_pixels": None if self.min_pixels is None else int(self.min_pixels),
            "max_pixels": None if self.max_pixels is None else int(self.max_pixels),
            "source_weights": {name: float(weight) for name, weight in self.source_weights},
            "media_snapshot_id": self.media_snapshot_id,
            "buffer_size": self.buffer_size,
            "cost_cache_size": self.cost_cache_size,
            "max_tokens_per_microbatch": self.max_tokens_per_microbatch,
            "max_sequence_length": self.max_sequence_length,
            "resource_budgets": {str(name): int(value) for name, value in self.resource_budgets},
            "batch_contract": batch_contract.to_dict(),
            "batch_contract_fingerprint": self.batch_contract_fingerprint,
            "planner_spec_fingerprint": self.planner_spec_fingerprint,
            "sample_execution_fingerprint": self.sample_execution_fingerprint,
            "train_input_contract": (
                None if self.train_input_contract is None else self.train_input_contract.to_dict()
            ),
            "train_input_contract_fingerprint": (
                None if self.train_input_contract is None else self.train_input_contract.fingerprint
            ),
            "training_resume_contract": (
                None
                if self.training_resume_contract is None
                else self.training_resume_contract.to_dict()
            ),
            "training_resume_contract_fingerprint": (
                None
                if self.training_resume_contract is None
                else self.training_resume_contract.fingerprint
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ShaftBatchingRunMetadata":
        role = "Batching metadata"
        payload = require_json_mapping(payload, role=role)
        require_exact_keys(
            payload,
            expected=_BATCHING_RUN_METADATA_KEYS,
            role=role,
        )
        version = json_string(payload, "version", role=role)
        if version != _BATCHING_RUN_METADATA_VERSION:
            raise ValueError(f"Unsupported batching metadata version: {version!r}.")

        serialized_contract_payload = require_json_mapping(
            payload["batch_contract"],
            role="Batching metadata.batch_contract",
        )
        serialized_contract = ShaftBatchContract.from_dict(serialized_contract_payload)
        serialized_batch_fingerprint = json_string(
            payload,
            "batch_contract_fingerprint",
            role=role,
        ).strip()
        if not serialized_batch_fingerprint:
            raise ValueError("Batching metadata batch_contract_fingerprint must not be empty.")

        source_weights = require_json_mapping(
            payload["source_weights"],
            role="Batching metadata.source_weights",
        )
        normalized_source_weights = tuple(
            sorted(
                (
                    name,
                    json_number(
                        weight,
                        role=f"Batching metadata.source_weights.{name}",
                    ),
                )
                for name, weight in source_weights.items()
            )
        )
        resource_budgets = require_json_mapping(
            payload["resource_budgets"],
            role="Batching metadata.resource_budgets",
        )
        normalized_resource_budgets: list[tuple[str, int]] = []
        for name, value in resource_budgets.items():
            if type(value) is not int:
                raise TypeError(
                    f"Batching metadata.resource_budgets.{name} must be a JSON integer."
                )
            normalized_resource_budgets.append((name, value))

        serialized_global_pack_count = json_optional_int(
            payload,
            "global_pack_count",
            role=role,
        )
        serialized_optimizer_pack_count = json_optional_int(
            payload,
            "optimizer_pack_count",
            role=role,
        )
        serialized_local_pack_count_range = _json_int_pair(
            payload,
            "local_pack_count_range",
            role=role,
        )
        serialized_global_pack_count_range = _json_int_pair(
            payload,
            "global_pack_count_range",
            role=role,
        )
        serialized_optimizer_pack_count_range = _json_int_pair(
            payload,
            "optimizer_pack_count_range",
            role=role,
        )

        raw_input_contract = payload["train_input_contract"]
        train_input_contract = None
        if raw_input_contract is not None:
            train_input_contract = ShaftTrainInputContract.from_dict(
                require_json_mapping(
                    raw_input_contract,
                    role="Batching metadata.train_input_contract",
                )
            )
        raw_resume_contract = payload["training_resume_contract"]
        training_resume_contract = None
        if raw_resume_contract is not None:
            training_resume_contract = ShaftTrainingResumeContract.from_dict(
                require_json_mapping(
                    raw_resume_contract,
                    role="Batching metadata.training_resume_contract",
                )
            )

        metadata = cls(
            grouping=json_string(payload, "grouping", role=role),
            cardinality=json_string(payload, "cardinality", role=role),
            packing=json_string(payload, "packing", role=role),
            layout=json_string(payload, "layout", role=role),
            per_device_train_batch_size=json_int(
                payload,
                "per_device_train_batch_size",
                role=role,
            ),
            data_world_size=json_int(payload, "data_world_size", role=role),
            gradient_accumulation_steps=json_int(
                payload,
                "gradient_accumulation_steps",
                role=role,
            ),
            min_pixels=json_optional_int(payload, "min_pixels", role=role),
            max_pixels=json_optional_int(payload, "max_pixels", role=role),
            source_weights=normalized_source_weights,
            media_snapshot_id=json_optional_string(
                payload,
                "media_snapshot_id",
                role=role,
            ),
            buffer_size=json_optional_int(payload, "buffer_size", role=role),
            cost_cache_size=json_optional_int(
                payload,
                "cost_cache_size",
                role=role,
            ),
            max_tokens_per_microbatch=json_optional_int(
                payload,
                "max_tokens_per_microbatch",
                role=role,
            ),
            max_sequence_length=json_optional_int(
                payload,
                "max_sequence_length",
                role=role,
            ),
            resource_budgets=tuple(sorted(normalized_resource_budgets)),
            batch_contract_fingerprint=serialized_batch_fingerprint,
            planner_spec_fingerprint=json_optional_string(
                payload,
                "planner_spec_fingerprint",
                role=role,
            ),
            sample_execution_fingerprint=json_optional_string(
                payload,
                "sample_execution_fingerprint",
                role=role,
            ),
            train_input_contract=train_input_contract,
            training_resume_contract=training_resume_contract,
        )
        serialized_input_fingerprint = json_optional_string(
            payload,
            "train_input_contract_fingerprint",
            role=role,
        )
        if metadata.train_input_contract is not None:
            if not str(serialized_input_fingerprint or "").strip():
                raise ValueError(
                    "Batching metadata training input contract is missing its fingerprint."
                )
            if str(serialized_input_fingerprint) != metadata.train_input_contract.fingerprint:
                raise ValueError(
                    "Batching metadata training input contract fingerprint differs "
                    "from its canonical payload."
                )
        elif serialized_input_fingerprint is not None:
            raise ValueError(
                "Batching metadata cannot carry a training input contract fingerprint "
                "without the contract payload."
            )
        serialized_resume_fingerprint = json_optional_string(
            payload,
            "training_resume_contract_fingerprint",
            role=role,
        )
        if metadata.training_resume_contract is not None:
            if not str(serialized_resume_fingerprint or "").strip():
                raise ValueError(
                    "Batching metadata training resume contract is missing its fingerprint."
                )
            if str(serialized_resume_fingerprint) != metadata.training_resume_contract.fingerprint:
                raise ValueError(
                    "Batching metadata training resume contract fingerprint differs "
                    "from its canonical payload."
                )
        elif serialized_resume_fingerprint is not None:
            raise ValueError(
                "Batching metadata cannot carry a training resume contract "
                "fingerprint without the contract payload."
            )
        if metadata.batch_contract != serialized_contract:
            expected = serialized_contract.to_dict()
            actual = metadata.batch_contract.to_dict()
            differences = [key for key, value in expected.items() if actual.get(key) != value]
            raise ValueError(
                "Batching metadata flat audit fields differ from its canonical "
                f"batch_contract: changed fields: {differences}."
            )
        local_min, local_max = metadata.local_pack_count_bounds
        global_bounds = list(metadata.global_pack_count_bounds)
        optimizer_bounds = list(metadata.optimizer_pack_count_bounds)
        derived_fields = {
            "global_pack_count": (
                serialized_global_pack_count,
                global_bounds[0] if global_bounds[0] == global_bounds[1] else None,
            ),
            "optimizer_pack_count": (
                serialized_optimizer_pack_count,
                (optimizer_bounds[0] if optimizer_bounds[0] == optimizer_bounds[1] else None),
            ),
            "local_pack_count_range": (
                serialized_local_pack_count_range,
                [local_min, local_max],
            ),
            "global_pack_count_range": (
                serialized_global_pack_count_range,
                global_bounds,
            ),
            "optimizer_pack_count_range": (
                serialized_optimizer_pack_count_range,
                optimizer_bounds,
            ),
        }
        for field_name, (serialized, expected) in derived_fields.items():
            if serialized != expected:
                raise ValueError(
                    f"Batching metadata {field_name} differs from its source fields: "
                    f"serialized={serialized}, expected={expected}."
                )
        return metadata


def build_batching_run_metadata(
    *,
    config: Any,
    training_args: Any,
    planning_spec: ShaftBatchPlanningSpec | None = None,
    batch_contract: ShaftBatchContract | None = None,
    sample_execution_fingerprint: str | None = None,
    train_input_contract: ShaftTrainInputContract | None = None,
    training_resume_contract: ShaftTrainingResumeContract | None = None,
) -> ShaftBatchingRunMetadata:
    contract = batch_contract or build_batch_contract(
        config=config,
        training_args=training_args,
    )
    if contract.is_planned != (planning_spec is not None):
        raise ValueError("Resolved batch-planning spec does not match data.batching.grouping.")
    if planning_spec is not None:
        expected_spec_fields = (
            contract.grouping,
            contract.cardinality,
            contract.packing,
            contract.layout,
            contract.max_sequence_length,
            contract.data_world_size,
            contract.buffer_size,
            contract.per_device_microbatch_size,
            contract.local_token_capacity,
            contract.resource_budgets,
        )
        actual_spec_fields = (
            planning_spec.grouping,
            planning_spec.cardinality,
            planning_spec.packing,
            planning_spec.layout,
            planning_spec.max_sequence_length,
            planning_spec.data_world_size,
            planning_spec.buffer_size,
            planning_spec.per_device_microbatch_size,
            planning_spec.max_tokens_per_microbatch,
            planning_spec.resource_budgets,
        )
        if actual_spec_fields != expected_spec_fields:
            raise ValueError(
                "Batch-planning spec differs from the resolved batch contract: "
                f"expected={expected_spec_fields!r}, actual={actual_spec_fields!r}."
            )
    return ShaftBatchingRunMetadata(
        grouping=contract.grouping,
        cardinality=contract.cardinality,
        packing=contract.packing,
        layout=contract.layout,
        per_device_train_batch_size=contract.per_device_microbatch_size,
        data_world_size=contract.data_world_size,
        gradient_accumulation_steps=contract.gradient_accumulation_steps,
        min_pixels=config.data.min_pixels,
        max_pixels=config.data.max_pixels,
        source_weights=tuple(
            sorted(
                (dataset.dataset_name, float(dataset.weight))
                for dataset in config.data.datasets
                if dataset.enabled and float(dataset.weight) > 0
            )
        ),
        media_snapshot_id=config.data.media_snapshot_id,
        buffer_size=contract.buffer_size,
        cost_cache_size=(
            int(config.data.batching.cost_cache_size) if contract.is_planned else None
        ),
        max_tokens_per_microbatch=contract.max_tokens_per_microbatch,
        max_sequence_length=contract.max_sequence_length,
        resource_budgets=contract.resource_budgets,
        batch_contract_fingerprint=contract.fingerprint,
        planner_spec_fingerprint=(None if planning_spec is None else planning_spec.fingerprint),
        sample_execution_fingerprint=(
            None if sample_execution_fingerprint is None else str(sample_execution_fingerprint)
        ),
        train_input_contract=train_input_contract,
        training_resume_contract=training_resume_contract,
    )


def batching_run_metadata_path(path: str | Path) -> Path:
    return Path(path) / BATCHING_RUN_METADATA_FILENAME


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, allow_nan=False, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def write_batching_run_metadata(
    path: str | Path,
    metadata: ShaftBatchingRunMetadata,
) -> Path:
    return _atomic_write_json(batching_run_metadata_path(path), metadata.to_dict())


def publish_batching_run_metadata(
    path: str | Path,
    metadata: ShaftBatchingRunMetadata,
) -> Path:
    publish_error: Exception | None = None
    status: dict[str, Any] | None = None
    if is_rank_zero():
        try:
            target = write_batching_run_metadata(path, metadata)
            status = {"ok": True, "path": str(target)}
        except Exception as exc:  # noqa: BLE001 - propagate to every rank
            publish_error = exc
            status = {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
    raw_status = broadcast_object_from_rank_zero(status)
    try:
        resolved_status = require_json_mapping(
            raw_status,
            role="rank-zero batching metadata publish status",
        )
        ok = json_bool(
            resolved_status,
            "ok",
            role="rank-zero batching metadata publish status",
        )
        if ok:
            require_exact_keys(
                resolved_status,
                expected=frozenset({"ok", "path"}),
                role="rank-zero batching metadata publish status",
            )
            published_path = json_string(
                resolved_status,
                "path",
                role="rank-zero batching metadata publish status",
            ).strip()
            if not published_path:
                raise ValueError("Published batching metadata path must not be empty.")
        else:
            require_exact_keys(
                resolved_status,
                expected=frozenset({"ok", "error_type", "error"}),
                role="rank-zero batching metadata publish status",
            )
            error_type = json_string(
                resolved_status,
                "error_type",
                role="rank-zero batching metadata publish status",
            ).strip()
            error = json_string(
                resolved_status,
                "error",
                role="rank-zero batching metadata publish status",
            ).strip()
            if not error_type or not error:
                raise ValueError("Failed batching metadata status requires non-empty error fields.")
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "Rank-zero batching metadata publish returned a malformed status "
            f"envelope: {raw_status!r}."
        ) from exc
    if not ok:
        if get_world_size() == 1 and publish_error is not None:
            raise publish_error
        raise RuntimeError(f"Rank-zero batching metadata publish failed: {error_type}: {error}.")
    return Path(published_path)


def load_batching_run_metadata(path: str | Path) -> ShaftBatchingRunMetadata:
    target = batching_run_metadata_path(path)
    payload = load_strict_json(target, role=f"batching run metadata {target}")
    payload = require_json_mapping(payload, role=f"batching run metadata {target}")
    return ShaftBatchingRunMetadata.from_dict(payload)


def build_batch_planning_resume_contract_fingerprint(
    *,
    training_resume_contract: ShaftTrainingResumeContract,
    sequence_execution_contract_fingerprint: str,
) -> str:
    """Bind planner cursor state to the unified training trajectory contract."""

    if not str(sequence_execution_contract_fingerprint).strip():
        raise ValueError("Batch-planning resume fingerprint requires a sequence contract.")
    payload = (
        "shaft-batch-planning-resume-contract-v3",
        training_resume_contract.fingerprint,
        str(sequence_execution_contract_fingerprint),
    )
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


def _load_trainer_state_payload(path: str | Path) -> dict[str, Any]:
    target = Path(path) / TRAINER_STATE_NAME
    payload = load_strict_json(target, role=f"trainer state {target}")
    return require_json_mapping(payload, role=f"trainer state {target}")


def _stateful_callback_payload(
    trainer_state: dict[str, Any],
    callback_name: str,
) -> dict[str, Any]:
    callbacks = trainer_state.get("stateful_callbacks")
    if not isinstance(callbacks, dict):
        raise ValueError("Trainer state has no stateful callback payloads.")
    callback_payload = callbacks.get(callback_name)
    if isinstance(callback_payload, list):
        callback_payload = callback_payload[-1] if callback_payload else None
    if not isinstance(callback_payload, dict):
        raise ValueError(f"Trainer checkpoint has no {callback_name} callback state.")
    return callback_payload


def _batch_planning_callback_payload(trainer_state: dict[str, Any]) -> dict[str, Any]:
    return _stateful_callback_payload(
        trainer_state,
        BATCH_PLANNING_CALLBACK_NAME,
    )


def load_checkpoint_batching_metadata(
    path: str | Path,
) -> ShaftBatchingRunMetadata:
    trainer_state = _load_trainer_state_payload(path)
    return _batching_metadata_from_trainer_state(trainer_state)


def _batching_metadata_from_trainer_state(
    trainer_state: dict[str, Any],
) -> ShaftBatchingRunMetadata:
    callback_payload = _stateful_callback_payload(
        trainer_state,
        BATCHING_METADATA_CALLBACK_NAME,
    )
    args_payload = callback_payload.get("args")
    if not isinstance(args_payload, dict):
        raise ValueError("Batching metadata callback args are missing.")
    metadata_payload = args_payload.get("metadata")
    if not isinstance(metadata_payload, dict):
        raise ValueError("Batching metadata callback has no metadata payload.")
    return ShaftBatchingRunMetadata.from_dict(metadata_payload)


def validate_batching_resume_contract(
    path: str | Path,
    *,
    expected_contract: ShaftBatchContract,
    expected_sample_execution_fingerprint: str | None = None,
    expected_train_input_contract: ShaftTrainInputContract | None = None,
    expected_training_resume_contract: ShaftTrainingResumeContract | None = None,
    require_train_input_contract_payload: bool = False,
    require_training_resume_contract_payload: bool = False,
) -> ShaftBatchingRunMetadata:
    metadata = load_checkpoint_batching_metadata(path)
    if require_training_resume_contract_payload:
        checkpoint_resume_contract = metadata.training_resume_contract
        if checkpoint_resume_contract is None:
            raise ValueError(
                "Checkpoint predates the unified training resume contract and "
                "cannot be used for exact resume. Start from model weights or "
                "restore the original framework version."
            )
    if require_train_input_contract_payload:
        checkpoint_input_contract = metadata.train_input_contract
        if checkpoint_input_contract is None:
            raise ValueError(
                "Checkpoint predates the complete training input contract and cannot "
                "be used for exact resume. Start from model weights or restore the "
                "original framework version."
            )
        if not checkpoint_input_contract.exact_resume_safe:
            raise ValueError(
                "Checkpoint training input contract is incomplete and is not "
                "exact-resume safe: "
                f"{list(checkpoint_input_contract.incomplete_reasons)}."
            )
    actual_contract = metadata.batch_contract
    if actual_contract.fingerprint != expected_contract.fingerprint:
        actual = actual_contract.to_dict()
        expected = expected_contract.to_dict()
        differences = [key for key, value in expected.items() if actual.get(key) != value]
        raise ValueError(
            "Training batch contract changed across exact resume; "
            f"changed fields: {differences}. Start a new training schedule from "
            "model weights or restore the original batching settings."
        )
    if expected_sample_execution_fingerprint is not None:
        actual_execution_fingerprint = str(metadata.sample_execution_fingerprint or "").strip()
        if actual_execution_fingerprint != str(expected_sample_execution_fingerprint):
            raise ValueError(
                "Training sample execution changed across exact resume. Restore the "
                "original data schedule/prompt transforms or start a new training "
                "schedule from model weights."
            )
    if expected_train_input_contract is not None:
        if not expected_train_input_contract.exact_resume_safe:
            raise ValueError(
                "Exact resume is unavailable because the training input identity "
                "is incomplete: "
                f"{list(expected_train_input_contract.incomplete_reasons)}. "
                "Version every active transform/input policy and provide complete "
                "tokenizer artifacts plus an immutable media snapshot, or start a "
                "new training schedule."
            )
        actual_input_contract = metadata.train_input_contract
        if actual_input_contract is None:
            raise ValueError(
                "Checkpoint predates the complete training input contract and cannot "
                "be used for exact resume. Start from model weights or restore the "
                "original framework version."
            )
        if not actual_input_contract.exact_resume_safe:
            raise ValueError(
                "Checkpoint training input contract is incomplete and is not "
                "exact-resume safe: "
                f"{list(actual_input_contract.incomplete_reasons)}."
            )
        if actual_input_contract.fingerprint != expected_train_input_contract.fingerprint:
            actual = actual_input_contract.to_dict()
            expected = expected_train_input_contract.to_dict()
            differences = [key for key, value in expected.items() if actual.get(key) != value]
            raise ValueError(
                "Training input contract changed across exact resume; changed fields: "
                f"{differences}. Restore the original data limits, model processor, "
                "tokenizer, template and collator semantics or start a new training "
                "schedule from model weights."
            )
    if expected_training_resume_contract is not None:
        actual_resume_contract = metadata.training_resume_contract
        if actual_resume_contract is None:
            raise ValueError(
                "Checkpoint predates the unified training resume contract and "
                "cannot be used for exact resume. Start from model weights or "
                "restore the original framework version."
            )
        if actual_resume_contract.fingerprint != expected_training_resume_contract.fingerprint:
            actual = actual_resume_contract.to_dict()
            expected = expected_training_resume_contract.to_dict()
            differences = [key for key, value in expected.items() if actual.get(key) != value]
            raise ValueError(
                "Training resume contract changed across exact resume; changed "
                f"fields: {differences}. Restore the original optimizer, scheduler, "
                "duration, randomness and algorithm objective settings, or start "
                "a new training schedule from model weights."
            )
    return metadata


def _batch_planning_resume_artifact_names(
    checkpoint: Path,
    *,
    spec: ShaftBatchPlanningSpec,
) -> tuple[str, ...]:
    def is_nonempty_file(name: str) -> bool:
        target = checkpoint / name
        return target.is_file() and target.stat().st_size > 0

    optimizer_name = next(
        (name for name in (OPTIMIZER_NAME, OPTIMIZER_NAME_BIN) if is_nonempty_file(name)),
        None,
    )
    if optimizer_name is None:
        raise ValueError("Planned exact-resume checkpoint has no optimizer state.")
    if not is_nonempty_file(SCHEDULER_NAME):
        raise ValueError("Planned exact-resume checkpoint has no scheduler state.")
    if int(spec.data_world_size) <= 1:
        rng_names = ("rng_state.pth",)
    else:
        rng_names = tuple(f"rng_state_{rank}.pth" for rank in range(int(spec.data_world_size)))
    missing_rng_names = [name for name in rng_names if not is_nonempty_file(name)]
    if missing_rng_names:
        raise ValueError(
            f"Planned exact-resume checkpoint is missing per-rank RNG state: {missing_rng_names}."
        )
    return tuple(sorted((TRAINER_STATE_NAME, optimizer_name, SCHEDULER_NAME, *rng_names)))


def _validate_batch_planning_checkpoint_payload(
    path: str | Path,
) -> tuple[
    ShaftBatchPlanningSpec,
    ShaftBatchPlanningState,
    int,
    int,
    str,
    str,
]:
    """Load a self-consistent exact-resume checkpoint payload.

    This validation is independent of the active run config, so run-root auto-resume
    can skip a newer torn checkpoint before comparing it with the requested contract.
    """

    checkpoint = Path(path)
    trainer_state = _load_trainer_state_payload(checkpoint)
    callback_payload = _batch_planning_callback_payload(trainer_state)
    args_payload = require_json_mapping(
        callback_payload.get("args"),
        role="batch-planning callback.args",
    )
    attributes_payload = require_json_mapping(
        callback_payload.get("attributes"),
        role="batch-planning callback.attributes",
    )
    spec_payload = require_json_mapping(
        args_payload.get("spec"),
        role="batch-planning callback.args.spec",
    )
    state_payload = require_json_mapping(
        attributes_payload.get("planning_state"),
        role="batch-planning callback.attributes.planning_state",
    )

    spec = ShaftBatchPlanningSpec.from_dict(spec_payload)
    state = ShaftBatchPlanningState.from_dict(state_payload)
    state.validate_against_spec(spec)

    global_step = json_int(
        trainer_state,
        "global_step",
        role="batch-planning trainer state",
    )
    if global_step < 0:
        raise ValueError("Trainer global_step must be >= 0.")
    gradient_accumulation_steps = json_int(
        args_payload,
        "gradient_accumulation_steps",
        role="batch-planning callback.args",
    )
    if gradient_accumulation_steps <= 0:
        raise ValueError("Stored gradient_accumulation_steps must be > 0.")
    resume_contract_fingerprint = json_string(
        args_payload,
        "resume_contract_fingerprint",
        role="batch-planning callback.args",
    ).strip()
    if not resume_contract_fingerprint:
        raise ValueError("Stored planning resume contract fingerprint must not be empty.")
    metadata = _batching_metadata_from_trainer_state(trainer_state)
    if metadata.planner_spec_fingerprint != spec.fingerprint:
        raise ValueError(
            "Batching metadata planner_spec_fingerprint differs from the "
            "batch-planning callback spec."
        )
    contract = metadata.batch_contract
    expected_spec_fields = (
        contract.grouping,
        contract.cardinality,
        contract.packing,
        contract.layout,
        contract.max_sequence_length,
        contract.data_world_size,
        contract.buffer_size,
        contract.per_device_microbatch_size,
        contract.local_token_capacity,
        contract.resource_budgets,
    )
    actual_spec_fields = (
        spec.grouping,
        spec.cardinality,
        spec.packing,
        spec.layout,
        spec.max_sequence_length,
        spec.data_world_size,
        spec.buffer_size,
        spec.per_device_microbatch_size,
        spec.max_tokens_per_microbatch,
        spec.resource_budgets,
    )
    if actual_spec_fields != expected_spec_fields:
        raise ValueError(
            "Batching metadata batch_contract differs from the planning callback spec."
        )
    if contract.gradient_accumulation_steps != gradient_accumulation_steps:
        raise ValueError("Batching metadata gradient accumulation differs from the planning state.")
    batch_contract_fingerprint = str(metadata.batch_contract_fingerprint)
    expected_microstep = global_step * gradient_accumulation_steps
    if int(state.global_microstep) != expected_microstep:
        raise ValueError(
            "Batch-planning checkpoint is not aligned with trainer global_step: "
            f"state_microstep={state.global_microstep}, expected={expected_microstep}."
        )

    return (
        spec,
        state,
        global_step,
        gradient_accumulation_steps,
        resume_contract_fingerprint,
        batch_contract_fingerprint,
    )


def build_batch_planning_checkpoint_commit_payload(
    path: str | Path,
) -> dict[str, Any] | None:
    """Build the planned-data extension for the shared training commit manifest."""

    checkpoint = Path(path)
    trainer_state = _load_trainer_state_payload(checkpoint)
    callbacks = trainer_state.get("stateful_callbacks")
    if not isinstance(callbacks, dict) or BATCH_PLANNING_CALLBACK_NAME not in callbacks:
        return None
    (
        spec,
        state,
        global_step,
        gradient_accumulation_steps,
        resume_contract_fingerprint,
        batch_contract_fingerprint,
    ) = _validate_batch_planning_checkpoint_payload(checkpoint)
    required_artifacts = _batch_planning_resume_artifact_names(checkpoint, spec=spec)
    return {
        "version": _BATCH_PLANNING_CHECKPOINT_COMMIT_VERSION,
        "contract_fingerprint": spec.fingerprint,
        "batch_contract_fingerprint": batch_contract_fingerprint,
        "planning_state_fingerprint": state.to_dict()["fingerprint"],
        "resume_contract_fingerprint": resume_contract_fingerprint,
        "global_step": global_step,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "data_world_size": int(spec.data_world_size),
        "required_artifacts": list(required_artifacts),
    }


def validate_batch_planning_checkpoint_commit_payload(
    path: str | Path,
    payload: Mapping[str, Any],
) -> None:
    expected = build_batch_planning_checkpoint_commit_payload(path)
    if expected is None:
        raise ValueError(
            "Training checkpoint commit declares batch planning but the trainer "
            "state has no planning callback."
        )
    if dict(payload) != expected:
        raise ValueError("Batch-planning commit payload differs from the trainer/callback state.")


def _validate_committed_batch_planning_checkpoint(
    path: str | Path,
    *,
    expected_commit_fingerprint: str | None = None,
) -> tuple[
    ShaftBatchPlanningSpec,
    ShaftBatchPlanningState,
    int,
    int,
    str,
    str,
]:
    from .checkpointing import (
        TRAINING_CHECKPOINT_COMMIT_FILENAME,
        validate_training_checkpoint_commit,
    )

    if expected_commit_fingerprint is None:
        validate_training_checkpoint_commit(path)
    else:
        marker_path = Path(path) / TRAINING_CHECKPOINT_COMMIT_FILENAME
        marker = load_strict_json(
            marker_path,
            role="training checkpoint commit manifest",
        )
        marker = require_json_mapping(
            marker,
            role="training checkpoint commit manifest",
        )
        canonical = json.dumps(
            marker,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        actual_fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if actual_fingerprint != str(expected_commit_fingerprint):
            raise ValueError(
                "Batch-planning checkpoint commit marker changed during startup."
            )
    return _validate_batch_planning_checkpoint_payload(path)


def checkpoint_has_batch_planning_state(path: str | Path) -> bool:
    try:
        _validate_committed_batch_planning_checkpoint(path)
        return True
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def validate_batch_planning_resume_contract(
    path: str | Path,
    *,
    expected_resume_contract_fingerprint: str,
    expected_commit_fingerprint: str | None = None,
) -> None:
    """Reject duration/optimizer/scheduler drift before loading data or a model."""

    (
        _spec,
        _state,
        _global_step,
        _gradient_accumulation_steps,
        actual_resume_fingerprint,
        _batch_contract_fingerprint,
    ) = _validate_committed_batch_planning_checkpoint(
        path,
        expected_commit_fingerprint=expected_commit_fingerprint,
    )
    if actual_resume_fingerprint != str(expected_resume_contract_fingerprint):
        raise ValueError(
            "Batch-planning exact-resume training contract changed; duration, "
            "gradient accumulation, optimizer, scheduler, or learning-rate semantics "
            "differ. Use init_from_checkpoint for a new training schedule."
        )


def load_batch_planning_state(
    path: str | Path,
    *,
    expected_spec: ShaftBatchPlanningSpec,
    expected_global_step: int,
    gradient_accumulation_steps: int,
    expected_resume_contract_fingerprint: str,
    expected_commit_fingerprint: str | None = None,
) -> ShaftBatchPlanningState:
    (
        actual_spec,
        state,
        actual_global_step,
        actual_gradient_accumulation_steps,
        actual_resume_fingerprint,
        _actual_batch_contract_fingerprint,
    ) = _validate_committed_batch_planning_checkpoint(
        path,
        expected_commit_fingerprint=expected_commit_fingerprint,
    )
    if actual_resume_fingerprint != str(expected_resume_contract_fingerprint):
        raise ValueError(
            "Batch-planning exact-resume training contract changed; duration, "
            "gradient accumulation, optimizer, scheduler, or learning-rate semantics "
            "differ. Use init_from_checkpoint for a new training schedule."
        )
    if actual_gradient_accumulation_steps != int(gradient_accumulation_steps):
        raise ValueError(
            "Batch-planning exact-resume gradient accumulation changed. "
            "Use init_from_checkpoint for a new training schedule."
        )
    if actual_global_step != int(expected_global_step):
        raise ValueError(
            "Batch-planning checkpoint trainer global_step differs from the "
            f"requested checkpoint step: actual={actual_global_step}, "
            f"expected={expected_global_step}."
        )
    if actual_spec.fingerprint != expected_spec.fingerprint:
        expected = expected_spec.to_dict()
        actual = actual_spec.to_dict()
        differences = [
            key
            for key, value in expected.items()
            if key != "fingerprint" and actual.get(key) != value
        ]
        raise ValueError(
            "Batch-planning resume contract changed; "
            f"changed fields: {differences}. Start a new run from weights or restore "
            "the original data/topology/buffer/budget settings."
        )
    if state.contract_fingerprint != expected_spec.fingerprint:
        raise ValueError("Batch-planning state points to a different contract.")
    return state


class ShaftBatchPlanningCallback(TrainerCallback, ExportableState):
    """Commit executed state and embed it in HF's atomic Trainer state."""

    def __init__(
        self,
        sampler: ShaftPlannedBatchSampler,
        spec: ShaftBatchPlanningSpec,
        *,
        gradient_accumulation_steps: int,
        resume_contract_fingerprint: str,
    ) -> None:
        self.sampler = sampler
        self.spec = spec
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)
        self.resume_contract_fingerprint = str(resume_contract_fingerprint)
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be > 0.")
        if not self.resume_contract_fingerprint:
            raise ValueError("resume_contract_fingerprint must not be empty.")

    @staticmethod
    def _normalize_step_progress(state: Any) -> None:
        """Expose one finite planned stream instead of resume-local HF epochs."""

        max_steps = int(getattr(state, "max_steps", 0) or 0)
        if max_steps <= 0:
            return
        state.num_train_epochs = 1
        state.epoch = min(max(int(state.global_step), 0) / max_steps, 1.0)

    def on_train_begin(self, args, state, control, **kwargs):
        _ = args, kwargs
        self._normalize_step_progress(state)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        _ = args, kwargs
        self.sampler.commit_global_microstep(
            int(state.global_step) * self.gradient_accumulation_steps
        )
        self._normalize_step_progress(state)
        return control

    def state(self) -> dict[str, Any]:
        return {
            "args": {
                "spec": self.spec.to_dict(),
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "resume_contract_fingerprint": self.resume_contract_fingerprint,
            },
            "attributes": {
                "planning_state": self.sampler.committed_state.to_dict(),
            },
        }


class ShaftBatchingMetadataCallback(TrainerCallback, ExportableState):
    """Publish resolved Shaft batching metadata into the active W&B run config."""

    def __init__(
        self,
        metadata: ShaftBatchingRunMetadata | Mapping[str, Any],
    ) -> None:
        self.metadata = (
            metadata
            if isinstance(metadata, ShaftBatchingRunMetadata)
            else ShaftBatchingRunMetadata.from_dict(dict(metadata))
        )
        self._wandb_published = False

    @staticmethod
    def _reports_to_wandb(args: Any) -> bool:
        report_to = getattr(args, "report_to", ())
        values = {report_to} if isinstance(report_to, str) else set(report_to or ())
        return "wandb" in {str(value).strip().lower() for value in values}

    def _publish_wandb(self, args: Any, state: Any) -> None:
        if self._wandb_published or not bool(state.is_world_process_zero):
            return
        if not self._reports_to_wandb(args):
            return
        try:
            import wandb
        except ImportError:
            logger.warning(
                "W&B reporting was requested but wandb cannot be imported; "
                "batching metadata remains in %s.",
                BATCHING_RUN_METADATA_FILENAME,
            )
            return
        run = getattr(wandb, "run", None)
        if run is None:
            return
        run.config.update(
            {"shaft_batching": self.metadata.to_dict()},
            allow_val_change=True,
        )
        self._wandb_published = True

    def on_train_begin(self, args, state, control, **kwargs):
        _ = kwargs
        self._publish_wandb(args, state)
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs, kwargs
        self._publish_wandb(args, state)
        return control

    def state(self) -> dict[str, Any]:
        return {
            "args": {"metadata": self.metadata.to_dict()},
            "attributes": {},
        }
