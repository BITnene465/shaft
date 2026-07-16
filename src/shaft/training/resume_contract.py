from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
import hashlib
from importlib import metadata as importlib_metadata
import inspect
import json
import math
from pathlib import Path
from typing import Any

from shaft.utils.contract_schema import (
    json_bool,
    json_int,
    json_list,
    json_number,
    json_optional_string,
    json_string,
    require_exact_keys,
    require_json_mapping,
    validate_json_value,
)
from .input_contract import callable_semantic_signature, component_semantic_signature
from .distributed import all_gather_objects


_TRAINING_RESUME_CONTRACT_VERSION = "shaft-training-resume-contract-v2"
_DISTRIBUTED_STAGE_STATUS_KEYS = frozenset({"ok", "error_type", "error", "fingerprints"})


def _canonical_json_value(value: Any) -> Any:
    """Return a strict, stable JSON value for trajectory-defining options."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Training resume contract floats must be finite.")
        return value
    if isinstance(value, Enum):
        return _canonical_json_value(value.value)
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _canonical_json_value(asdict(value))
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise TypeError(
                "Training resume contract mapping keys must be JSON strings; "
                "key coercion would make semantic identity ambiguous."
            )
        return {
            key: _canonical_json_value(item)
            for key, item in sorted(value.items())
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_canonical_json_value(item) for item in value]
        return sorted(normalized, key=lambda item: json.dumps(item, sort_keys=True))
    raise TypeError(
        "Training resume contract cannot canonically encode value of type "
        f"{type(value).__module__}.{type(value).__qualname__}."
    )


def _module_implementation_signature(value: Any) -> str:
    try:
        source = inspect.getsource(value)
    except (OSError, TypeError) as exc:
        raise ValueError(
            "Exact-resume training policy identity requires source-visible modules; "
            f"cannot inspect {value!r}."
        ) from exc
    module_name = str(getattr(value, "__name__", ""))
    payload = {
        "version": "shaft-module-semantic-identity-v3",
        "module": module_name,
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "runtime_functions": {
            name: callable_semantic_signature(
                item,
                role=f"module:{module_name}:{name}",
                include_dependencies=False,
            )
            for name, item in sorted(vars(value).items())
            if inspect.isfunction(item)
            and (
                str(getattr(item, "__module__", "")) != module_name
                or str(getattr(item, "__name__", "")) != name
            )
        },
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _owning_module_signature(value: Any) -> str:
    module = inspect.getmodule(value)
    if module is None:
        raise ValueError(f"Cannot resolve owning module for {value!r}.")
    return _module_implementation_signature(module)


_OPTIMIZER_KEYS = frozenset(
    {
        "name",
        "builder_signature",
        "policy_signature",
        "learning_rate",
        "param_group_lrs",
        "no_decay_name_patterns",
        "weight_decay",
        "adam_beta1",
        "adam_beta2",
        "adam_epsilon",
        "max_grad_norm",
    }
)
_SCHEDULER_KEYS = frozenset(
    {
        "name",
        "resolved_name",
        "builder_signature",
        "policy_signature",
        "lr_scheduler_type",
        "warmup_ratio",
        "num_cycles",
        "power",
    }
)
_DURATION_KEYS = frozenset({"unit", "value", "resolved_max_steps", "resolved_num_train_epochs"})
_TRAINING_RESUME_KEYS = frozenset(
    {
        "version",
        "algorithm",
        "batch_contract_fingerprint",
        "train_input_contract_fingerprint",
        "data_execution_fingerprint",
        "seed",
        "data_seed",
        "duration",
        "gradient_accumulation_steps",
        "full_determinism",
        "precision",
        "gradient_checkpointing",
        "distributed_strategy",
        "execution",
        "implementation",
        "optimizer",
        "scheduler",
        "objective",
    }
)


@dataclass(frozen=True, slots=True)
class ShaftOptimizerResumeContract:
    name: str
    builder_signature: str
    policy_signature: str
    learning_rate: float
    param_group_lrs: tuple[tuple[str, float], ...]
    no_decay_name_patterns: tuple[str, ...]
    weight_decay: float
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    max_grad_norm: float

    def __post_init__(self) -> None:
        if not all(
            str(getattr(self, field_name)).strip()
            for field_name in ("name", "builder_signature", "policy_signature")
        ):
            raise ValueError("Optimizer resume contract identities must not be empty.")
        group_names = [name for name, _ in self.param_group_lrs]
        if group_names != sorted(group_names) or len(group_names) != len(set(group_names)):
            raise ValueError("Optimizer param_group_lrs must be sorted and unique.")
        if any(not str(name).strip() for name in group_names):
            raise ValueError("Optimizer param-group names must not be empty.")
        if len(self.no_decay_name_patterns) != len(set(self.no_decay_name_patterns)):
            raise ValueError("Optimizer no-decay patterns must be unique.")
        for field_name in (
            "learning_rate",
            "weight_decay",
            "adam_beta1",
            "adam_beta2",
            "adam_epsilon",
            "max_grad_norm",
        ):
            if not math.isfinite(float(getattr(self, field_name))):
                raise ValueError(f"Optimizer {field_name} must be finite.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": str(self.name),
            "builder_signature": self.builder_signature,
            "policy_signature": self.policy_signature,
            "learning_rate": float(self.learning_rate),
            "param_group_lrs": {str(name): float(value) for name, value in self.param_group_lrs},
            "no_decay_name_patterns": list(self.no_decay_name_patterns),
            "weight_decay": float(self.weight_decay),
            "adam_beta1": float(self.adam_beta1),
            "adam_beta2": float(self.adam_beta2),
            "adam_epsilon": float(self.adam_epsilon),
            "max_grad_norm": float(self.max_grad_norm),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ShaftOptimizerResumeContract":
        payload = require_json_mapping(
            payload,
            role="optimizer resume contract",
        )
        require_exact_keys(
            payload,
            expected=_OPTIMIZER_KEYS,
            role="optimizer resume contract",
        )
        param_group_lrs = require_json_mapping(
            payload["param_group_lrs"],
            role="optimizer resume contract.param_group_lrs",
        )
        normalized_group_lrs: list[tuple[str, float]] = []
        for name, value in param_group_lrs.items():
            resolved = json_number(
                value,
                role=(f"optimizer resume contract.param_group_lrs[{name!r}]"),
            )
            normalized_group_lrs.append((name, resolved))
        patterns = json_list(
            payload,
            "no_decay_name_patterns",
            role="optimizer resume contract",
        )
        if any(type(item) is not str for item in patterns):
            raise TypeError("Optimizer no_decay_name_patterns entries must be JSON strings.")
        return cls(
            name=json_string(payload, "name", role="optimizer resume contract"),
            builder_signature=json_string(
                payload,
                "builder_signature",
                role="optimizer resume contract",
            ),
            policy_signature=json_string(
                payload,
                "policy_signature",
                role="optimizer resume contract",
            ),
            learning_rate=json_number(
                payload["learning_rate"],
                role="optimizer resume contract.learning_rate",
            ),
            param_group_lrs=tuple(sorted(normalized_group_lrs)),
            no_decay_name_patterns=tuple(patterns),
            weight_decay=json_number(
                payload["weight_decay"],
                role="optimizer resume contract.weight_decay",
            ),
            adam_beta1=json_number(
                payload["adam_beta1"],
                role="optimizer resume contract.adam_beta1",
            ),
            adam_beta2=json_number(
                payload["adam_beta2"],
                role="optimizer resume contract.adam_beta2",
            ),
            adam_epsilon=json_number(
                payload["adam_epsilon"],
                role="optimizer resume contract.adam_epsilon",
            ),
            max_grad_norm=json_number(
                payload["max_grad_norm"],
                role="optimizer resume contract.max_grad_norm",
            ),
        )


@dataclass(frozen=True, slots=True)
class ShaftSchedulerResumeContract:
    name: str
    resolved_name: str
    builder_signature: str
    policy_signature: str
    lr_scheduler_type: str
    warmup_ratio: float
    num_cycles: float
    power: float

    def __post_init__(self) -> None:
        if not all(
            str(getattr(self, field_name)).strip()
            for field_name in (
                "name",
                "resolved_name",
                "builder_signature",
                "policy_signature",
                "lr_scheduler_type",
            )
        ):
            raise ValueError("Scheduler names must not be empty.")
        for field_name in ("warmup_ratio", "num_cycles", "power"):
            if not math.isfinite(float(getattr(self, field_name))):
                raise ValueError(f"Scheduler {field_name} must be finite.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": str(self.name),
            "resolved_name": str(self.resolved_name),
            "builder_signature": self.builder_signature,
            "policy_signature": self.policy_signature,
            "lr_scheduler_type": str(self.lr_scheduler_type),
            "warmup_ratio": float(self.warmup_ratio),
            "num_cycles": float(self.num_cycles),
            "power": float(self.power),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ShaftSchedulerResumeContract":
        payload = require_json_mapping(
            payload,
            role="scheduler resume contract",
        )
        require_exact_keys(
            payload,
            expected=_SCHEDULER_KEYS,
            role="scheduler resume contract",
        )
        return cls(
            name=json_string(payload, "name", role="scheduler resume contract"),
            resolved_name=json_string(
                payload,
                "resolved_name",
                role="scheduler resume contract",
            ),
            builder_signature=json_string(
                payload,
                "builder_signature",
                role="scheduler resume contract",
            ),
            policy_signature=json_string(
                payload,
                "policy_signature",
                role="scheduler resume contract",
            ),
            lr_scheduler_type=json_string(
                payload,
                "lr_scheduler_type",
                role="scheduler resume contract",
            ),
            warmup_ratio=json_number(
                payload["warmup_ratio"],
                role="scheduler resume contract.warmup_ratio",
            ),
            num_cycles=json_number(
                payload["num_cycles"],
                role="scheduler resume contract.num_cycles",
            ),
            power=json_number(
                payload["power"],
                role="scheduler resume contract.power",
            ),
        )


@dataclass(frozen=True, slots=True)
class ShaftTrainingResumeContract:
    """Canonical identity of semantics that determine future optimizer updates."""

    algorithm: str
    batch_contract_fingerprint: str
    train_input_contract_fingerprint: str
    data_execution_fingerprint: str
    seed: int
    data_seed: int
    duration_unit: str
    duration_value: float
    resolved_max_steps: int
    resolved_num_train_epochs: float
    gradient_accumulation_steps: int
    full_determinism: bool
    precision: str
    gradient_checkpointing: bool
    distributed_strategy: str
    execution: tuple[tuple[str, Any], ...]
    implementation: tuple[tuple[str, Any], ...]
    optimizer: ShaftOptimizerResumeContract
    scheduler: ShaftSchedulerResumeContract
    objective: tuple[tuple[str, Any], ...]

    def __post_init__(self) -> None:
        if self.algorithm not in {"sft", "dpo", "grpo"}:
            raise ValueError(
                "Training resume contract only supports exact-resumable "
                f"algorithms, got {self.algorithm!r}."
            )
        for field_name in (
            "batch_contract_fingerprint",
            "train_input_contract_fingerprint",
            "data_execution_fingerprint",
            "duration_unit",
            "precision",
            "distributed_strategy",
        ):
            value = getattr(self, field_name)
            if type(value) is not str or not value.strip():
                raise ValueError(f"ShaftTrainingResumeContract.{field_name} must not be empty.")
        if self.duration_unit not in {"steps", "epochs"}:
            raise ValueError(f"Unsupported training duration unit: {self.duration_unit!r}.")
        if not math.isfinite(float(self.duration_value)) or float(self.duration_value) <= 0:
            raise ValueError("Training duration value must be finite and > 0.")
        if not math.isfinite(float(self.resolved_num_train_epochs)):
            raise ValueError("Resolved num_train_epochs must be finite.")
        if int(self.gradient_accumulation_steps) <= 0:
            raise ValueError("gradient_accumulation_steps must be > 0.")
        if not isinstance(self.full_determinism, bool):
            raise TypeError("full_determinism must be a boolean.")
        if not isinstance(self.gradient_checkpointing, bool):
            raise TypeError("gradient_checkpointing must be a boolean.")
        objective_names = [name for name, _ in self.objective]
        if objective_names != sorted(objective_names) or len(objective_names) != len(
            set(objective_names)
        ):
            raise ValueError("Training objective fields must be sorted and unique.")
        execution_names = [name for name, _ in self.execution]
        if execution_names != sorted(execution_names) or len(execution_names) != len(
            set(execution_names)
        ):
            raise ValueError("Training execution fields must be sorted and unique.")
        implementation_names = [name for name, _ in self.implementation]
        if implementation_names != sorted(implementation_names) or len(implementation_names) != len(
            set(implementation_names)
        ):
            raise ValueError("Training implementation fields must be sorted and unique.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": _TRAINING_RESUME_CONTRACT_VERSION,
            "algorithm": self.algorithm,
            "batch_contract_fingerprint": self.batch_contract_fingerprint,
            "train_input_contract_fingerprint": self.train_input_contract_fingerprint,
            "data_execution_fingerprint": self.data_execution_fingerprint,
            "seed": int(self.seed),
            "data_seed": int(self.data_seed),
            "duration": {
                "unit": self.duration_unit,
                "value": float(self.duration_value),
                "resolved_max_steps": int(self.resolved_max_steps),
                "resolved_num_train_epochs": float(self.resolved_num_train_epochs),
            },
            "gradient_accumulation_steps": int(self.gradient_accumulation_steps),
            "full_determinism": bool(self.full_determinism),
            "precision": self.precision,
            "gradient_checkpointing": bool(self.gradient_checkpointing),
            "distributed_strategy": self.distributed_strategy,
            "execution": {
                str(name): _canonical_json_value(value) for name, value in self.execution
            },
            "implementation": {
                str(name): _canonical_json_value(value) for name, value in self.implementation
            },
            "optimizer": self.optimizer.to_dict(),
            "scheduler": self.scheduler.to_dict(),
            "objective": {
                str(name): _canonical_json_value(value) for name, value in self.objective
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ShaftTrainingResumeContract":
        payload = require_json_mapping(
            payload,
            role="training resume contract",
        )
        require_exact_keys(
            payload,
            expected=_TRAINING_RESUME_KEYS,
            role="training resume contract",
        )
        version = json_string(
            payload,
            "version",
            role="training resume contract",
        )
        if version != _TRAINING_RESUME_CONTRACT_VERSION:
            raise ValueError(f"Unsupported training resume contract version: {version!r}.")
        duration = require_json_mapping(
            payload["duration"],
            role="training resume contract.duration",
        )
        require_exact_keys(
            duration,
            expected=_DURATION_KEYS,
            role="training resume contract duration",
        )
        objective = require_json_mapping(
            payload["objective"],
            role="training resume contract.objective",
        )
        execution = require_json_mapping(
            payload["execution"],
            role="training resume contract.execution",
        )
        implementation = require_json_mapping(
            payload["implementation"],
            role="training resume contract.implementation",
        )
        return cls(
            algorithm=json_string(
                payload,
                "algorithm",
                role="training resume contract",
            ),
            batch_contract_fingerprint=json_string(
                payload,
                "batch_contract_fingerprint",
                role="training resume contract",
            ),
            train_input_contract_fingerprint=json_string(
                payload,
                "train_input_contract_fingerprint",
                role="training resume contract",
            ),
            data_execution_fingerprint=json_string(
                payload,
                "data_execution_fingerprint",
                role="training resume contract",
            ),
            seed=json_int(payload, "seed", role="training resume contract"),
            data_seed=json_int(payload, "data_seed", role="training resume contract"),
            duration_unit=json_string(
                duration,
                "unit",
                role="training resume contract duration",
            ),
            duration_value=json_number(
                duration["value"],
                role="training resume contract duration.value",
            ),
            resolved_max_steps=json_int(
                duration,
                "resolved_max_steps",
                role="training resume contract duration",
            ),
            resolved_num_train_epochs=json_number(
                duration["resolved_num_train_epochs"],
                role=("training resume contract duration.resolved_num_train_epochs"),
            ),
            gradient_accumulation_steps=json_int(
                payload,
                "gradient_accumulation_steps",
                role="training resume contract",
            ),
            full_determinism=json_bool(
                payload,
                "full_determinism",
                role="training resume contract",
            ),
            precision=json_string(
                payload,
                "precision",
                role="training resume contract",
            ),
            gradient_checkpointing=json_bool(
                payload,
                "gradient_checkpointing",
                role="training resume contract",
            ),
            distributed_strategy=json_string(
                payload,
                "distributed_strategy",
                role="training resume contract",
            ),
            execution=tuple(
                sorted(
                    (
                        name,
                        validate_json_value(
                            value,
                            role=f"training resume contract.execution.{name}",
                        ),
                    )
                    for name, value in execution.items()
                )
            ),
            implementation=tuple(
                sorted(
                    (
                        name,
                        validate_json_value(
                            value,
                            role=(f"training resume contract.implementation.{name}"),
                        ),
                    )
                    for name, value in implementation.items()
                )
            ),
            optimizer=ShaftOptimizerResumeContract.from_dict(
                require_json_mapping(
                    payload["optimizer"],
                    role="training resume contract.optimizer",
                )
            ),
            scheduler=ShaftSchedulerResumeContract.from_dict(
                require_json_mapping(
                    payload["scheduler"],
                    role="training resume contract.scheduler",
                )
            ),
            objective=tuple(
                sorted(
                    (
                        name,
                        validate_json_value(
                            value,
                            role=f"training resume contract.objective.{name}",
                        ),
                    )
                    for name, value in objective.items()
                )
            ),
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

    def model_execution_fingerprints(self) -> tuple[str, str, str, str | None]:
        execution = dict(self.execution)
        model_execution = execution.get("model_execution")
        if type(model_execution) is not dict:
            raise ValueError("Training resume contract has no canonical model_execution mapping.")
        required_names = (
            "model_plan_fingerprint",
            "resolved_finetune_plan_fingerprint",
            "resolved_optimizer_plan_fingerprint",
        )
        required_values: list[str] = []
        for name in required_names:
            value = model_execution.get(name)
            if type(value) is not str or not value.strip():
                raise TypeError(
                    f"Training resume contract model_execution.{name} must be a "
                    "non-empty JSON string."
                )
            required_values.append(value)
        sequence_value = model_execution.get("sequence_execution_contract_fingerprint")
        if sequence_value is not None and (
            type(sequence_value) is not str or not sequence_value.strip()
        ):
            raise TypeError(
                "Training resume contract model_execution."
                "sequence_execution_contract_fingerprint must be null or a "
                "non-empty JSON string."
            )
        return (
            required_values[0],
            required_values[1],
            required_values[2],
            sequence_value,
        )


def training_contract_section_fingerprint(
    contract: ShaftTrainingResumeContract,
    *,
    section: str,
    key: str | None = None,
) -> str:
    payload = contract.to_dict().get(section)
    if key is not None:
        if not isinstance(payload, Mapping):
            raise ValueError(f"Training contract section {section!r} is not a mapping.")
        payload = payload.get(key)
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def converge_training_contract_fingerprints(
    *,
    stage: str,
    fingerprints: Mapping[str, str],
) -> None:
    """Fail every rank if a trajectory identity differs before a side effect."""

    with distributed_training_contract_stage(
        stage=stage,
        fingerprints=lambda: fingerprints,
    ):
        pass


@contextmanager
def distributed_training_contract_stage(
    *,
    stage: str,
    fingerprints: Callable[[], Mapping[str, str]],
) -> Iterator[None]:
    """Converge a fallible startup stage without stranding peer ranks.

    Every rank enters the same status-envelope collective even when its local
    builder or validator raises.  Fingerprints are evaluated only after the
    local stage succeeds, then travel in that same envelope.  Distributed
    failures are reported as one deterministic error on every rank; a
    non-distributed call preserves its original exception type and traceback.
    """

    local_error: Exception | None = None
    normalized: dict[str, str] | None = None
    try:
        yield
        raw_fingerprints = fingerprints()
        if not isinstance(raw_fingerprints, Mapping):
            raise TypeError(
                f"Distributed {stage} contract fingerprints must be a mapping."
            )
        if any(
            type(name) is not str
            or not name
            or type(value) is not str
            or not value
            for name, value in raw_fingerprints.items()
        ):
            raise TypeError(
                f"Distributed {stage} contract fingerprint names and values must "
                "be non-empty strings."
            )
        normalized = dict(sorted(raw_fingerprints.items()))
    except Exception as exc:  # noqa: BLE001 - peers must reach the collective
        local_error = exc

    status = {
        "ok": local_error is None,
        "error_type": (
            None
            if local_error is None
            else (f"{type(local_error).__module__}.{type(local_error).__qualname__}")
        ),
        "error": (
            None if local_error is None else str(local_error) or type(local_error).__qualname__
        ),
        "fingerprints": normalized,
    }
    statuses = all_gather_objects(status)
    normalized_statuses: list[dict[str, Any]] = []
    try:
        for rank, raw_status in enumerate(statuses):
            peer_status = require_json_mapping(
                raw_status,
                role=f"distributed {stage} status rank {rank}",
            )
            require_exact_keys(
                peer_status,
                expected=_DISTRIBUTED_STAGE_STATUS_KEYS,
                role=f"distributed {stage} status rank {rank}",
            )
            peer_ok = json_bool(
                peer_status,
                "ok",
                role=f"distributed {stage} status rank {rank}",
            )
            peer_error_type = json_optional_string(
                peer_status,
                "error_type",
                role=f"distributed {stage} status rank {rank}",
            )
            peer_error = json_optional_string(
                peer_status,
                "error",
                role=f"distributed {stage} status rank {rank}",
            )
            raw_fingerprints = peer_status["fingerprints"]
            if peer_ok:
                if peer_error_type is not None or peer_error is not None:
                    raise ValueError("successful status must not contain an error")
                peer_fingerprints = require_json_mapping(
                    raw_fingerprints,
                    role=(f"distributed {stage} status rank {rank}.fingerprints"),
                )
                if any(type(value) is not str or not value for value in peer_fingerprints.values()):
                    raise TypeError("fingerprint values must be non-empty JSON strings")
            else:
                if not peer_error_type or not peer_error or raw_fingerprints is not None:
                    raise ValueError(
                        "failed status requires error_type/error and null fingerprints"
                    )
                peer_fingerprints = None
            normalized_statuses.append(
                {
                    "ok": peer_ok,
                    "error_type": peer_error_type,
                    "error": peer_error,
                    "fingerprints": peer_fingerprints,
                }
            )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Distributed {stage} contract convergence returned a malformed "
            f"status envelope: {statuses!r}."
        ) from exc
    failures = [
        {
            "rank": rank,
            "error_type": item.get("error_type"),
            "error": item.get("error"),
        }
        for rank, item in enumerate(normalized_statuses)
        if item["ok"] is False
    ]
    if failures:
        if len(normalized_statuses) == 1 and local_error is not None:
            raise local_error
        raise RuntimeError(f"Distributed {stage} startup failed on peer ranks: {failures!r}.")

    peer_fingerprints = [item["fingerprints"] for item in normalized_statuses]
    normalized_fingerprints = [dict(item) for item in peer_fingerprints if item is not None]
    if any(item != normalized_fingerprints[0] for item in normalized_fingerprints[1:]):
        raise ValueError(
            f"Distributed {stage} contract differs across ranks: {normalized_fingerprints!r}."
        )


def _resolved_precision(training_args: Any) -> str:
    if bool(getattr(training_args, "bf16", False)):
        return "bf16"
    if bool(getattr(training_args, "fp16", False)):
        return "fp16"
    return "fp32"


def _dpo_reference_semantics(finetune_mode: object) -> str:
    mode = str(finetune_mode).strip().lower()
    if mode in {"lora", "dora", "qlora"}:
        return "policy_with_adapter_disabled"
    return "frozen_policy_copy"


def _sft_objective(config: Any) -> dict[str, Any]:
    return {
        "ignore_index": -100,
        "loss_name": str(config.train.loss_name),
        "loss_scale": str(config.train.loss_scale),
    }


def _resolved_dpo_objective(config: Any, resolved_dpo_args: Any) -> dict[str, Any]:
    finetune_mode = str(config.model.finetune.mode).strip().lower()
    fields = (
        "disable_dropout",
        "pad_token",
        "max_length",
        "truncation_mode",
        "padding_free",
        "pad_to_multiple_of",
        "precompute_ref_log_probs",
        "precompute_ref_batch_size",
        "loss_type",
        "loss_weights",
        "ld_alpha",
        "f_divergence_type",
        "f_alpha_divergence_coef",
        "label_smoothing",
        "beta",
        "use_weighting",
        "discopop_tau",
        "activation_offloading",
        "sync_ref_model",
        "ref_model_mixup_alpha",
        "ref_model_sync_steps",
    )
    objective = {
        field_name: _canonical_json_value(getattr(resolved_dpo_args, field_name, None))
        for field_name in fields
    }
    objective.update(
        {
            "finetune_mode": finetune_mode,
            "reference_model": _dpo_reference_semantics(finetune_mode),
        }
    )
    return objective


def _grpo_objective(
    config: Any,
    resolved_grpo_args: Any,
    training_args: Any,
) -> dict[str, Any]:
    from shaft.algorithms.grpo_rewards import GRPO_REWARD_REGISTRY
    from shaft.codec import CODEC_REGISTRY

    rewards = [
        {
            "name": str(reward.name),
            "codec": str(reward.codec),
            "weight": float(reward.weight),
            "params": _canonical_json_value(reward.params),
            "implementation_signature": callable_semantic_signature(
                GRPO_REWARD_REGISTRY.get(str(reward.name)),
                role=f"grpo_reward:{reward.name}",
            ),
            "implementation_module_signature": _owning_module_signature(
                GRPO_REWARD_REGISTRY.get(str(reward.name))
            ),
            "codec_implementation_signature": callable_semantic_signature(
                CODEC_REGISTRY.get(str(reward.codec)),
                role=f"grpo_codec:{reward.codec}",
            ),
            "codec_module_signature": _owning_module_signature(
                CODEC_REGISTRY.get(str(reward.codec))
            ),
        }
        for reward in config.rlhf.grpo.reward_functions
    ]
    fields = (
        "disable_dropout",
        "beta",
        "num_generations",
        "max_completion_length",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "repetition_penalty",
        "generation_kwargs",
        "chat_template_kwargs",
        "cache_implementation",
        "use_transformers_paged",
        "ds3_gather_for_generation",
        "steps_per_generation",
        "num_iterations",
        "generation_batch_size",
        "shuffle_dataset",
        "use_vllm",
        "vllm_mode",
        "vllm_model_impl",
        "vllm_structured_outputs_regex",
        "epsilon",
        "delta",
        "epsilon_high",
        "sapo_temperature_neg",
        "sapo_temperature_pos",
        "importance_sampling_level",
        "reward_weights",
        "multi_objective_aggregation",
        "scale_rewards",
        "loss_type",
        "mask_truncated_completions",
        "sync_ref_model",
        "ref_model_mixup_alpha",
        "ref_model_sync_steps",
        "top_entropy_quantile",
        "max_tool_calling_iterations",
        "vllm_importance_sampling_correction",
        "vllm_importance_sampling_mode",
        "vllm_importance_sampling_cap",
        "off_policy_mask_threshold",
        "use_bias_correction_kl",
    )
    objective = {
        field_name: _canonical_json_value(getattr(resolved_grpo_args, field_name, None))
        for field_name in fields
    }
    gradient_accumulation = int(training_args.gradient_accumulation_steps)
    generation_reuse_microsteps = int(getattr(resolved_grpo_args, "steps_per_generation")) * int(
        getattr(resolved_grpo_args, "num_iterations")
    )
    if gradient_accumulation <= 0 or generation_reuse_microsteps <= 0:
        raise ValueError("Resolved GRPO update cadence values must be > 0.")
    max_steps = int(training_args.max_steps)
    unique_prompts_per_group = int(getattr(resolved_grpo_args, "generation_batch_size")) // int(
        getattr(resolved_grpo_args, "num_generations")
    )
    objective.update(
        {
            "checkpoint_optimizer_step_cadence": (
                generation_reuse_microsteps
                // math.gcd(gradient_accumulation, generation_reuse_microsteps)
            ),
            "generation_reuse_microsteps": generation_reuse_microsteps,
            "step_horizon_unique_prompt_budget": (
                None
                if max_steps < 0
                else math.ceil(max_steps * gradient_accumulation / generation_reuse_microsteps)
                * unique_prompts_per_group
            ),
            "unique_prompts_per_generation_group": unique_prompts_per_group,
        }
    )
    objective["reward_functions"] = rewards
    return objective


def _deepspeed_config_identity(config: Any, training_args: Any) -> Any:
    configured = config.train.distributed.deepspeed
    if configured.config:
        return {"source": "inline", "config": _canonical_json_value(configured.config)}
    if configured.config_path:
        path = Path(str(configured.config_path))
        if not path.is_file():
            raise ValueError(f"DeepSpeed config file does not exist: {path}.")
        return {
            "source": "file",
            "content_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    resolved = getattr(training_args, "deepspeed", None)
    if resolved is None:
        return None
    if isinstance(resolved, Mapping):
        return {"source": "resolved", "config": _canonical_json_value(resolved)}
    resolved_config = getattr(resolved, "config", None)
    if isinstance(resolved_config, Mapping):
        return {
            "source": "resolved_object",
            "config": _canonical_json_value(resolved_config),
        }
    raise TypeError(
        "Cannot canonically identify the resolved DeepSpeed configuration of type "
        f"{type(resolved).__module__}.{type(resolved).__qualname__}."
    )


def _training_execution(
    config: Any,
    training_args: Any,
    *,
    model_plan_fingerprint: str,
    resolved_finetune_plan_fingerprint: str,
    resolved_optimizer_plan_fingerprint: str,
    sequence_execution_contract_fingerprint: str | None,
) -> dict[str, Any]:
    distributed = config.train.distributed
    strategy = str(distributed.strategy).strip().lower()
    resolved_fsdp = getattr(training_args, "fsdp", None)
    resolved_fsdp_config = getattr(training_args, "fsdp_config", None)
    requested_cpu = bool(getattr(training_args, "use_cpu", config.train.use_cpu))
    device = getattr(training_args, "device", None)
    load_best_model_at_end = bool(
        getattr(
            training_args,
            "load_best_model_at_end",
            config.train.load_best_model_at_end,
        )
    )
    final_selection: dict[str, Any] = {
        "load_best_model_at_end": load_best_model_at_end,
    }
    if load_best_model_at_end:
        final_selection.update(
            {
                "metric_for_best_model": str(config.eval.metric_for_best_model),
                "greater_is_better": bool(config.eval.greater_is_better),
                "eval_strategy": str(config.eval.eval_strategy),
                "eval_steps": int(config.eval.eval_steps),
                "eval_epoch_interval": int(config.eval.epoch_interval),
                "save_strategy": str(config.train.save_strategy),
                "save_steps": int(config.train.save_steps),
                "save_epoch_interval": int(config.train.save_epoch_interval),
                "save_total_limit": int(config.train.save_total_limit),
                "eval_config": _canonical_json_value(config.eval),
            }
        )
    return {
        "average_tokens_across_devices": bool(
            getattr(training_args, "average_tokens_across_devices", True)
        ),
        "algorithm_params": _canonical_json_value(config.algorithm.params),
        "ddp_find_unused_parameters": bool(config.train.ddp_find_unused_parameters),
        "ddp_static_graph": bool(getattr(training_args, "ddp_static_graph", False)),
        "device_type": (
            str(getattr(device, "type", device))
            if device is not None
            else "cpu"
            if requested_cpu
            else "auto"
        ),
        "distributed_config": _canonical_json_value(distributed),
        "finetune_config": _canonical_json_value(config.model.finetune),
        "final_selection": final_selection,
        "model_execution": {
            "model_plan_fingerprint": str(model_plan_fingerprint),
            "resolved_finetune_plan_fingerprint": str(resolved_finetune_plan_fingerprint),
            "resolved_optimizer_plan_fingerprint": str(resolved_optimizer_plan_fingerprint),
            "sequence_execution_contract_fingerprint": (
                None
                if sequence_execution_contract_fingerprint is None
                else str(sequence_execution_contract_fingerprint)
            ),
            "torch_dtype": str(config.model.torch_dtype),
            "attention_implementation": config.model.attn_implementation,
            "device_map": _canonical_json_value(config.model.device_map),
            "torch_compile": bool(getattr(training_args, "torch_compile", False)),
            "torch_compile_backend": getattr(
                training_args,
                "torch_compile_backend",
                None,
            ),
            "torch_compile_mode": getattr(
                training_args,
                "torch_compile_mode",
                None,
            ),
        },
        "effective_deepspeed_config": (
            _deepspeed_config_identity(config, training_args) if strategy == "deepspeed" else None
        ),
        "effective_fsdp": _canonical_json_value(resolved_fsdp),
        "effective_fsdp_config": _canonical_json_value(resolved_fsdp_config),
        "use_cpu": requested_cpu,
    }


def _optimizer_implementation(name: object) -> tuple[str, str]:
    import shaft.training.optimizer as optimizer_module
    import shaft.training.optimizer_plan as optimizer_plan_module

    normalized = str(name).strip().lower()
    builder = optimizer_module.OPTIMIZER_REGISTRY.get(normalized)
    dependency_signatures: dict[str, str] = {}
    target = inspect.unwrap(builder)
    target_module = inspect.getmodule(target)
    code = getattr(target, "__code__", None)
    if target_module is not None and code is not None:
        for dependency_name in sorted(set(code.co_names)):
            dependency = vars(target_module).get(dependency_name)
            dependency_module = str(getattr(dependency, "__module__", ""))
            if not dependency_module.startswith("shaft."):
                continue
            if inspect.isclass(dependency):
                dependency_signatures[dependency_name] = component_semantic_signature(
                    dependency,
                    role=f"optimizer_dependency:{normalized}:{dependency_name}",
                )
            elif callable(dependency):
                dependency_signatures[dependency_name] = callable_semantic_signature(
                    dependency,
                    role=f"optimizer_dependency:{normalized}:{dependency_name}",
                )
    policy_payload = (
        _module_implementation_signature(optimizer_module),
        _module_implementation_signature(optimizer_plan_module),
        dependency_signatures,
    )
    policy_signature = hashlib.sha256(repr(policy_payload).encode("utf-8")).hexdigest()
    return (
        callable_semantic_signature(builder, role=f"optimizer_builder:{normalized}"),
        policy_signature,
    )


def _scheduler_implementation(name: object) -> tuple[str, str, str]:
    import shaft.training.scheduler as scheduler_module

    requested = str(name).strip().lower()
    resolved = "cosine" if requested in {"", "auto"} else requested
    builder = scheduler_module.SCHEDULER_REGISTRY.get(resolved)
    return (
        resolved,
        callable_semantic_signature(builder, role=f"scheduler_builder:{resolved}"),
        _module_implementation_signature(scheduler_module),
    )


def _runtime_package_version(distribution_name: str) -> str | None:
    try:
        return importlib_metadata.version(distribution_name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _trajectory_implementation(
    config: Any,
    algorithm: str,
    *,
    hook_instances: Sequence[Any],
    interceptor_instances: Sequence[Any],
    sequence_execution_capabilities: Sequence[str],
) -> dict[str, Any]:
    import shaft.model.finetune as finetune_module
    import shaft.model.finetune_plan as finetune_plan_module
    import shaft.model.freeze as freeze_module
    import shaft.training.loss as loss_module
    import shaft.training.optimizer_mixin as optimizer_mixin_module
    import shaft.training.reproducibility as reproducibility_module
    import shaft.training.train_sampler_mixin as train_sampler_module
    from shaft.algorithms.registry import ALGORITHM_REGISTRY

    if algorithm == "sft":
        from shaft.algorithms import sft as _algorithm_module  # noqa: F401
    elif algorithm == "dpo":
        from shaft.algorithms import dpo as _algorithm_module  # noqa: F401
    else:
        from shaft.algorithms import grpo as _algorithm_module  # noqa: F401

    algorithm_impl = ALGORITHM_REGISTRY.get(algorithm)
    package_names = ["torch", "transformers", "accelerate"]
    if algorithm in {"dpo", "grpo"}:
        package_names.append("trl")
    finetune_mode = str(config.model.finetune.mode).strip().lower()
    if finetune_mode in {"lora", "dora", "qlora"}:
        package_names.append("peft")
    if finetune_mode == "qlora":
        package_names.append("bitsandbytes")
    if str(config.train.optimizer_name).strip().lower() in {
        "adam8bit",
        "paged_adamw_8bit",
    }:
        package_names.append("bitsandbytes")
    if str(config.train.distributed.strategy).strip().lower() == "deepspeed":
        package_names.append("deepspeed")
    attention_implementation = str(config.model.attn_implementation or "").strip().lower()
    if attention_implementation == "flash_attention_2":
        package_names.append("flash-attn")
    for capability in sequence_execution_capabilities:
        package_name = str(capability).split("=", 1)[0].strip()
        if package_name in {
            "flash-attn",
            "flash-linear-attention",
            "causal-conv1d",
        }:
            package_names.append(package_name)

    if algorithm == "sft":
        from .sft_trainer import ShaftSFTTrainer

        selected_loss = loss_module.LOSS_REGISTRY.get(str(config.train.loss_name).strip().lower())
        objective_impl: dict[str, Any] = {
            "loss_callable": callable_semantic_signature(
                selected_loss,
                role=f"sft_loss:{config.train.loss_name}",
            ),
            "loss_policy": _module_implementation_signature(loss_module),
        }
        trainer_impl = ShaftSFTTrainer
    else:
        import shaft.algorithms.rlhf_utils as rlhf_utils_module

        from .trl_trainers import ShaftDPOTrainer, ShaftGRPOTrainer

        config_builder = (
            rlhf_utils_module.build_trl_dpo_config
            if algorithm == "dpo"
            else rlhf_utils_module.build_trl_grpo_config
        )
        config_policy = [
            callable_semantic_signature(
                rlhf_utils_module._normalize_training_args_payload,
                role="trl_config:normalize_training_args",
            ),
            callable_semantic_signature(
                config_builder,
                role=f"trl_config:{algorithm}",
            ),
        ]
        if algorithm == "grpo":
            config_policy.extend(
                [
                    callable_semantic_signature(
                        rlhf_utils_module._precision_model_init_kwargs,
                        role="trl_config:precision_model_init",
                    ),
                    callable_semantic_signature(
                        rlhf_utils_module._set_default_model_init_kwargs,
                        role="trl_config:set_default_model_init",
                    ),
                ]
            )
        objective_impl = {
            "trl_config_policy": config_policy,
        }
        trainer_impl = ShaftDPOTrainer if algorithm == "dpo" else ShaftGRPOTrainer

    return {
        "algorithm": callable_semantic_signature(
            algorithm_impl,
            role=f"algorithm:{algorithm}",
        ),
        "algorithm_policy": _owning_module_signature(algorithm_impl),
        "finetune_policy": [
            _module_implementation_signature(finetune_module),
            _module_implementation_signature(finetune_plan_module),
            _module_implementation_signature(freeze_module),
        ],
        "objective": objective_impl,
        "optimizer_mixin_policy": _module_implementation_signature(optimizer_mixin_module),
        "plugins": _configured_plugin_identity(
            config,
            hook_instances=hook_instances,
            interceptor_instances=interceptor_instances,
        ),
        "reproducibility_policy": _module_implementation_signature(reproducibility_module),
        "runtime_packages": {
            name: _runtime_package_version(name) for name in sorted(set(package_names))
        },
        "sequence_execution_capabilities": list(sequence_execution_capabilities),
        "sampler_trainer_policy": _module_implementation_signature(train_sampler_module),
        "trainer": callable_semantic_signature(
            trainer_impl,
            role=f"trainer:{algorithm}",
        ),
    }


def _plugin_instance_identity(instance: Any, *, role: str) -> dict[str, Any]:
    # The shared component identity already binds declared implementation,
    # callable closure/default state and instance semantic state with cycle
    # guards.  Re-serializing dataclasses through ``asdict`` (or ``vars``)
    # created a second identity truth source, recursed forever on self-links,
    # and deep-copied large observer caches before hashing them.
    return {
        "implementation": component_semantic_signature(instance, role=role),
    }


def _configured_plugin_identity(
    config: Any,
    *,
    hook_instances: Sequence[Any],
    interceptor_instances: Sequence[Any],
) -> dict[str, Any]:
    import shaft.plugins.hooks as hooks_module
    import shaft.plugins.interceptors as interceptors_module

    hook_names = tuple(str(name) for name in config.plugins.hooks)
    interceptor_names = tuple(str(name) for name in config.plugins.interceptors)
    if len(hook_names) != len(hook_instances):
        raise ValueError("Resolved hook instances do not match config.plugins.hooks.")
    if len(interceptor_names) != len(interceptor_instances):
        raise ValueError("Resolved interceptor instances do not match config.plugins.interceptors.")
    hooks = [
        {
            "name": str(name),
            **_plugin_instance_identity(
                instance,
                role=f"training_hook:{name}",
            ),
        }
        for name, instance in zip(hook_names, hook_instances, strict=True)
    ]
    interceptors = [
        {
            "name": str(name),
            **_plugin_instance_identity(
                instance,
                role=f"training_interceptor:{name}",
            ),
        }
        for name, instance in zip(
            interceptor_names,
            interceptor_instances,
            strict=True,
        )
    ]
    checkpointable = (
        str(config.train.save_strategy).strip().lower() != "no"
        or config.train.resume_from_checkpoint is not None
    )
    for role, instances in (
        ("hook", hook_instances),
        ("interceptor", interceptor_instances),
    ):
        malformed = [
            type(instance).__qualname__
            for instance in instances
            if type(getattr(instance, "shaft_trajectory_neutral", False)) is not bool
        ]
        if malformed:
            raise TypeError(
                f"Plugin shaft_trajectory_neutral must be a boolean; malformed {role}s={malformed}."
            )
    if checkpointable:
        for role, instances in (
            ("hook", hook_instances),
            ("interceptor", interceptor_instances),
        ):
            unsupported = [
                type(instance).__qualname__
                for instance in instances
                if getattr(instance, "shaft_trajectory_neutral", False) is False
            ]
            if unsupported:
                raise ValueError(
                    "Checkpointable exact-resume plugins must explicitly declare "
                    "shaft_trajectory_neutral=True until versioned plugin state_dict "
                    f"support exists; non-neutral {role}s={unsupported}."
                )
    return {
        "hook_manager_policy": _module_implementation_signature(hooks_module),
        "hooks": hooks,
        "interceptor_manager_policy": _module_implementation_signature(interceptors_module),
        "interceptors": interceptors,
    }


def build_training_resume_contract(
    *,
    config: Any,
    training_args: Any,
    batch_contract_fingerprint: str,
    train_input_contract_fingerprint: str,
    data_execution_fingerprint: str,
    model_plan_fingerprint: str,
    resolved_finetune_plan_fingerprint: str,
    resolved_optimizer_plan_fingerprint: str,
    sequence_execution_contract_fingerprint: str | None = None,
    sequence_execution_capabilities: Sequence[str] = (),
    resolved_dpo_args: Any | None = None,
    resolved_grpo_args: Any | None = None,
    hook_instances: Sequence[Any] = (),
    interceptor_instances: Sequence[Any] = (),
) -> ShaftTrainingResumeContract:
    """Build the one trajectory contract shared by fixed and planned training."""

    if not str(model_plan_fingerprint).strip():
        raise ValueError("Training resume contract requires a model-plan fingerprint.")
    if not str(resolved_finetune_plan_fingerprint).strip():
        raise ValueError("Training resume contract requires a resolved finetune-plan fingerprint.")
    if not str(resolved_optimizer_plan_fingerprint).strip():
        raise ValueError("Training resume contract requires a resolved optimizer-plan fingerprint.")
    if type(train_input_contract_fingerprint) is not str or not train_input_contract_fingerprint.strip():
        raise ValueError("Training resume contract requires a train-input contract fingerprint.")
    if type(data_execution_fingerprint) is not str or not data_execution_fingerprint.strip():
        raise ValueError("Training resume contract requires a data-execution fingerprint.")
    algorithm = str(config.algorithm.name).strip().lower()
    if algorithm == "ppo":
        raise ValueError("PPO does not support exact-resume training contracts.")
    if algorithm == "sft":
        objective = _sft_objective(config)
    elif algorithm == "dpo":
        if resolved_dpo_args is None:
            raise ValueError("DPO training resume contract requires resolved TRL arguments.")
        objective = _resolved_dpo_objective(config, resolved_dpo_args)
    elif algorithm == "grpo":
        if resolved_grpo_args is None:
            raise ValueError("GRPO training resume contract requires resolved TRL arguments.")
        objective = _grpo_objective(config, resolved_grpo_args, training_args)
    else:
        raise ValueError(f"Unsupported exact-resume algorithm: {algorithm!r}.")

    train = config.train
    duration = train.duration
    optimizer_builder_signature, optimizer_policy_signature = _optimizer_implementation(
        train.optimizer_name
    )
    (
        resolved_scheduler_name,
        scheduler_builder_signature,
        scheduler_policy_signature,
    ) = _scheduler_implementation(train.scheduler_name)
    return ShaftTrainingResumeContract(
        algorithm=algorithm,
        batch_contract_fingerprint=str(batch_contract_fingerprint),
        train_input_contract_fingerprint=train_input_contract_fingerprint,
        data_execution_fingerprint=data_execution_fingerprint,
        seed=int(getattr(training_args, "seed", config.experiment.seed)),
        data_seed=int(
            getattr(training_args, "data_seed", None)
            if getattr(training_args, "data_seed", None) is not None
            else getattr(training_args, "seed", config.experiment.seed)
        ),
        duration_unit=str(duration.unit).strip().lower(),
        duration_value=float(duration.value),
        resolved_max_steps=int(training_args.max_steps),
        resolved_num_train_epochs=float(training_args.num_train_epochs),
        gradient_accumulation_steps=int(training_args.gradient_accumulation_steps),
        full_determinism=bool(training_args.full_determinism),
        precision=_resolved_precision(training_args),
        gradient_checkpointing=bool(training_args.gradient_checkpointing),
        distributed_strategy=str(train.distributed.strategy).strip().lower(),
        execution=tuple(
            sorted(
                (str(name), _canonical_json_value(value))
                for name, value in _training_execution(
                    config,
                    training_args,
                    model_plan_fingerprint=model_plan_fingerprint,
                    resolved_finetune_plan_fingerprint=(resolved_finetune_plan_fingerprint),
                    resolved_optimizer_plan_fingerprint=(resolved_optimizer_plan_fingerprint),
                    sequence_execution_contract_fingerprint=(
                        sequence_execution_contract_fingerprint
                    ),
                ).items()
            )
        ),
        implementation=tuple(
            sorted(
                (str(name), _canonical_json_value(value))
                for name, value in _trajectory_implementation(
                    config,
                    algorithm,
                    hook_instances=hook_instances,
                    interceptor_instances=interceptor_instances,
                    sequence_execution_capabilities=(sequence_execution_capabilities),
                ).items()
            )
        ),
        optimizer=ShaftOptimizerResumeContract(
            name=str(train.optimizer_name),
            builder_signature=optimizer_builder_signature,
            policy_signature=optimizer_policy_signature,
            learning_rate=float(train.learning_rate),
            param_group_lrs=tuple(
                sorted((str(name), float(value)) for name, value in train.param_group_lrs.items())
            ),
            no_decay_name_patterns=tuple(str(item) for item in train.no_decay_name_patterns),
            weight_decay=float(train.weight_decay),
            adam_beta1=float(train.adam_beta1),
            adam_beta2=float(train.adam_beta2),
            adam_epsilon=float(train.adam_epsilon),
            max_grad_norm=float(train.max_grad_norm),
        ),
        scheduler=ShaftSchedulerResumeContract(
            name=str(train.scheduler_name),
            resolved_name=resolved_scheduler_name,
            builder_signature=scheduler_builder_signature,
            policy_signature=scheduler_policy_signature,
            lr_scheduler_type=str(train.lr_scheduler_type),
            warmup_ratio=float(train.warmup_ratio),
            num_cycles=float(train.scheduler_num_cycles),
            power=float(train.scheduler_power),
        ),
        objective=tuple(
            sorted((str(name), _canonical_json_value(value)) for name, value in objective.items())
        ),
    )


def build_training_resume_preflight_contract(
    *,
    checkpoint_contract: ShaftTrainingResumeContract,
    config: Any,
    training_args: Any,
    batch_contract_fingerprint: str,
    resolved_dpo_args: Any | None = None,
    resolved_grpo_args: Any | None = None,
    hook_instances: Sequence[Any] = (),
    interceptor_instances: Sequence[Any] = (),
    sequence_execution_capabilities: Sequence[str] = (),
) -> ShaftTrainingResumeContract:
    """Project current cheap semantics through stored model-dependent identities.

    The returned value is still the canonical typed contract. Only the fields that
    cannot be known without immutable artifact resolution/model construction are
    taken from the checkpoint. Therefore an optimizer/objective/GA/plugin drift is
    rejected before model hashing, while the normal full contract validation later
    proves the model and resolved-plan identities independently.
    """

    (
        model_plan_fingerprint,
        finetune_plan_fingerprint,
        optimizer_plan_fingerprint,
        sequence_contract_fingerprint,
    ) = checkpoint_contract.model_execution_fingerprints()
    if not sequence_execution_capabilities:
        stored_implementation = dict(checkpoint_contract.implementation)
        stored_capabilities = stored_implementation.get(
            "sequence_execution_capabilities",
            (),
        )
        if not isinstance(stored_capabilities, list):
            raise TypeError("Checkpoint sequence_execution_capabilities must be a list.")
        sequence_execution_capabilities = tuple(str(item) for item in stored_capabilities)
    return build_training_resume_contract(
        config=config,
        training_args=training_args,
        batch_contract_fingerprint=batch_contract_fingerprint,
        train_input_contract_fingerprint=(
            checkpoint_contract.train_input_contract_fingerprint
        ),
        data_execution_fingerprint=checkpoint_contract.data_execution_fingerprint,
        model_plan_fingerprint=model_plan_fingerprint,
        resolved_finetune_plan_fingerprint=finetune_plan_fingerprint,
        resolved_optimizer_plan_fingerprint=optimizer_plan_fingerprint,
        sequence_execution_contract_fingerprint=sequence_contract_fingerprint,
        resolved_dpo_args=resolved_dpo_args,
        resolved_grpo_args=resolved_grpo_args,
        hook_instances=hook_instances,
        interceptor_instances=interceptor_instances,
        sequence_execution_capabilities=sequence_execution_capabilities,
    )
