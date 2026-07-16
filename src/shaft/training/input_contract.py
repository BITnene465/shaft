from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from functools import lru_cache
import hashlib
from importlib import metadata as importlib_metadata
import json
import math
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np
import torch

from shaft.model.input_identity import (
    added_token_semantic_value,
    artifact_semantic_value,
    tokenizer_artifact_fingerprint,
)
from shaft.model.training_identity import model_training_semantic_fingerprint
from shaft.utils.semantic_identity import (
    callable_semantic_fingerprint as shared_callable_semantic_fingerprint,
    component_semantic_fingerprint as shared_component_semantic_fingerprint,
)

from shaft.utils.contract_schema import (
    json_bool,
    json_list,
    json_string,
    require_exact_keys,
    require_json_mapping,
    validate_json_value,
)


_TRAIN_INPUT_CONTRACT_VERSION = "shaft-train-input-contract-v2"
_TRAIN_INPUT_CONTRACT_KEYS = frozenset(
    {
        "version",
        "algorithm",
        "data_execution_fingerprint",
        "data_execution_contract_complete",
        "incomplete_reasons",
        "train_dataset_signature",
        "model_plan_fingerprint",
        "model_adapter_signature",
        "processor_signature",
        "tokenizer_signature",
        "template_signature",
        "input_builder_signature",
        "input_policy_version",
        "input_options",
    }
)


def _qualified_name(value: Any) -> str:
    cls = value if isinstance(value, type) else type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


@lru_cache(maxsize=1)
def _package_distributions() -> Mapping[str, list[str]]:
    return importlib_metadata.packages_distributions()


@lru_cache(maxsize=None)
def _package_version(module_name: str) -> str | None:
    package_name = str(module_name).split(".", 1)[0]
    if not package_name:
        return None
    if package_name == "builtins" or package_name in sys.stdlib_module_names:
        return f"python-{platform.python_version()}"
    distributions = _package_distributions().get(
        package_name,
        [package_name],
    )
    for distribution_name in distributions:
        try:
            return importlib_metadata.version(distribution_name)
        except importlib_metadata.PackageNotFoundError:
            continue
    return None


def _canonical_value(
    value: Any,
    *,
    unresolved_types: set[str] | None = None,
) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return {"float": "nan"}
        if math.isinf(value):
            return {"float": "inf" if value > 0 else "-inf"}
        return value
    if isinstance(value, Enum):
        return {
            "enum_type": _qualified_name(value),
            "value": _canonical_value(
                value.value,
                unresolved_types=unresolved_types,
            ),
        }
    if isinstance(value, torch.dtype):
        return {"torch_dtype": str(value)}
    if isinstance(value, torch.device):
        return {"torch_device": str(value)}
    if isinstance(value, np.dtype):
        return {"numpy_dtype": str(value)}
    if isinstance(value, np.generic):
        return _canonical_value(
            value.item(),
            unresolved_types=unresolved_types,
        )
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, type):
        return {"class": _qualified_name(value)}
    added_token = added_token_semantic_value(value)
    if added_token is not None:
        return added_token
    if is_dataclass(value) and not isinstance(value, type):
        return _canonical_value(
            asdict(value),
            unresolved_types=unresolved_types,
        )
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(
                item,
                unresolved_types=unresolved_types,
            )
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item, unresolved_types=unresolved_types) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_canonical_value(item, unresolved_types=unresolved_types) for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(item, sort_keys=True, default=str),
        )
    qualified_type = _qualified_name(value)
    if unresolved_types is None:
        raise TypeError(
            f"Training input identity cannot canonically encode value of type {qualified_type}."
        )
    unresolved_types.add(qualified_type)
    return {"unresolved_type": qualified_type}


def _component_state(
    component: Any,
    *,
    role: str,
    unresolved_types: set[str],
) -> dict[str, Any]:
    cls = component if isinstance(component, type) else type(component)
    state: dict[str, Any] = {
        "runtime_package_versions": {
            f"{base.__module__}.{base.__qualname__}": _package_version(str(base.__module__))
            for base in cls.__mro__
            if base is not object
        }
    }
    if role == "train_dataset":
        state["runtime_dependencies"] = {
            "pillow": _package_version("PIL"),
        }
    if role == "model_adapter":
        model_meta = getattr(component, "model_meta", None)
        if model_meta is not None and getattr(model_meta, "loader", None) is not None:
            state["model_training_semantic_fingerprint"] = model_training_semantic_fingerprint(
                component
            )
        for field_name in (
            "model_type",
            "family",
            "template_type",
            "group_name",
            "capabilities",
        ):
            if hasattr(component, field_name):
                state[field_name] = _canonical_value(
                    getattr(component, field_name),
                    unresolved_types=unresolved_types,
                )
        for policy_name in (
            "processor_policy",
            "sequence_execution_policy",
        ):
            policy = getattr(component, policy_name, None)
            if policy is None:
                continue
            semantic_state_provider = getattr(
                policy,
                "shaft_semantic_state",
                None,
            )
            if callable(semantic_state_provider):
                policy_state = _canonical_value(
                    semantic_state_provider(),
                    unresolved_types=unresolved_types,
                )
            elif is_dataclass(policy) and not isinstance(policy, type):
                policy_state: Any = _canonical_value(
                    policy,
                    unresolved_types=unresolved_types,
                )
            else:
                # Non-dataclass policies are implementation-defined and often carry
                # mutable telemetry/cache fields. Their class source/runtime methods
                # are the identity unless they explicitly publish semantic state.
                policy_state = {}
            state[policy_name] = {
                "implementation": shared_component_semantic_fingerprint(
                    policy,
                    role=f"model_adapter.{policy_name}",
                    include_state=False,
                ),
                "state": policy_state,
            }
    if role == "template":
        template_meta = getattr(component, "template_meta", None)
        if template_meta is not None:
            state["template_meta"] = _canonical_value(
                template_meta,
                unresolved_types=unresolved_types,
            )

    if role in {"processor", "tokenizer"}:
        to_dict = getattr(component, "to_dict", None)
        if callable(to_dict):
            try:
                payload = to_dict()
            except (AttributeError, TypeError, ValueError):
                payload = None
            if isinstance(payload, Mapping):
                state["config"] = _canonical_value(
                    (artifact_semantic_value(payload) if role == "tokenizer" else payload),
                    unresolved_types=unresolved_types,
                )

        for field_name in (
            "model_max_length",
            "padding_side",
            "truncation_side",
            "chat_template",
            "special_tokens_map",
            "eos_token_id",
            "pad_token_id",
            "bos_token_id",
        ):
            if hasattr(component, field_name):
                state[field_name] = _canonical_value(
                    getattr(component, field_name),
                    unresolved_types=unresolved_types,
                )
        if hasattr(component, "init_kwargs"):
            state["init_kwargs"] = _canonical_value(
                (
                    artifact_semantic_value(getattr(component, "init_kwargs"))
                    if role == "tokenizer"
                    else getattr(component, "init_kwargs")
                ),
                unresolved_types=unresolved_types,
            )
        get_added_vocab = getattr(component, "get_added_vocab", None)
        if callable(get_added_vocab):
            state["added_vocab"] = _canonical_value(
                get_added_vocab(),
                unresolved_types=unresolved_types,
            )

        if role == "tokenizer":
            backend_tokenizer = getattr(component, "backend_tokenizer", None)
            if backend_tokenizer is not None:
                state["backend_tokenizer"] = {
                    "implementation": shared_component_semantic_fingerprint(
                        backend_tokenizer,
                        role="tokenizer.backend_tokenizer",
                        include_state=False,
                    ),
                }

        for nested_name in ("image_processor", "video_processor"):
            nested = getattr(component, nested_name, None)
            if nested is None:
                continue
            nested_to_dict = getattr(nested, "to_dict", None)
            nested_payload = nested_to_dict() if callable(nested_to_dict) else None
            state[nested_name] = {
                "implementation": shared_component_semantic_fingerprint(
                    nested,
                    role=f"processor.{nested_name}",
                    include_state=False,
                ),
                "config": (
                    _canonical_value(
                        nested_payload,
                        unresolved_types=unresolved_types,
                    )
                    if isinstance(nested_payload, Mapping)
                    else None
                ),
            }
    return state


def _component_semantic_identity(
    component: Any,
    *,
    role: str,
) -> tuple[str, tuple[str, ...]]:
    unresolved_types: set[str] = set()
    payload = {
        "role": str(role),
        "implementation": shared_component_semantic_fingerprint(
            component,
            role=role,
            include_state=False,
        ),
        "state": _component_state(
            component,
            role=role,
            unresolved_types=unresolved_types,
        ),
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    incomplete_reasons = {
        f"unresolved_{role}_state_type:{qualified_type}" for qualified_type in unresolved_types
    }
    if role == "train_dataset" and _package_version("PIL") is None:
        incomplete_reasons.add("missing_train_dataset_runtime_version:Pillow")
    return (
        hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        tuple(sorted(incomplete_reasons)),
    )


def component_semantic_signature(component: Any, *, role: str) -> str:
    return shared_component_semantic_fingerprint(component, role=role)


def callable_semantic_signature(
    value: Any,
    *,
    role: str,
    include_dependencies: bool = True,
) -> str:
    """Fingerprint the active callable, including code, defaults, closure and state.

    This is intentionally stricter than a source-only hash: registry entries and
    plugins can replace a function without changing its owning source file, and
    closures/defaults can change behavior while bytecode stays identical.
    """

    return shared_callable_semantic_fingerprint(
        value,
        role=role,
        include_dependencies=include_dependencies,
    )


@dataclass(frozen=True, slots=True)
class ShaftTrainInputContract:
    """Versioned identity of the full path that produces model training inputs."""

    algorithm: str
    data_execution_fingerprint: str
    data_execution_contract_complete: bool
    incomplete_reasons: tuple[str, ...]
    train_dataset_signature: str
    model_plan_fingerprint: str
    model_adapter_signature: str
    processor_signature: str
    tokenizer_signature: str
    template_signature: str
    input_builder_signature: str
    input_policy_version: str
    input_options: tuple[tuple[str, Any], ...]

    def __post_init__(self) -> None:
        for field_name in (
            "algorithm",
            "data_execution_fingerprint",
            "train_dataset_signature",
            "model_plan_fingerprint",
            "model_adapter_signature",
            "processor_signature",
            "tokenizer_signature",
            "template_signature",
            "input_builder_signature",
            "input_policy_version",
        ):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"ShaftTrainInputContract.{field_name} must not be empty.")
        if not isinstance(self.data_execution_contract_complete, bool):
            raise TypeError(
                "ShaftTrainInputContract.data_execution_contract_complete must be a boolean."
            )
        option_names = [name for name, _ in self.input_options]
        if option_names != sorted(option_names) or len(option_names) != len(set(option_names)):
            raise ValueError("ShaftTrainInputContract.input_options must be sorted and unique.")
        if not self.data_execution_contract_complete and not self.incomplete_reasons:
            raise ValueError("An incomplete data execution contract requires an explicit reason.")

    @property
    def exact_resume_safe(self) -> bool:
        return bool(self.data_execution_contract_complete and not self.incomplete_reasons)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": _TRAIN_INPUT_CONTRACT_VERSION,
            "algorithm": self.algorithm,
            "data_execution_fingerprint": self.data_execution_fingerprint,
            "data_execution_contract_complete": bool(self.data_execution_contract_complete),
            "incomplete_reasons": list(self.incomplete_reasons),
            "train_dataset_signature": self.train_dataset_signature,
            "model_plan_fingerprint": self.model_plan_fingerprint,
            "model_adapter_signature": self.model_adapter_signature,
            "processor_signature": self.processor_signature,
            "tokenizer_signature": self.tokenizer_signature,
            "template_signature": self.template_signature,
            "input_builder_signature": self.input_builder_signature,
            "input_policy_version": self.input_policy_version,
            "input_options": {
                str(name): _canonical_value(value) for name, value in self.input_options
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ShaftTrainInputContract":
        role = "Training input contract"
        payload = require_json_mapping(payload, role=role)
        require_exact_keys(
            payload,
            expected=_TRAIN_INPUT_CONTRACT_KEYS,
            role=role,
        )
        version = json_string(payload, "version", role=role)
        if version != _TRAIN_INPUT_CONTRACT_VERSION:
            raise ValueError(f"Unsupported training input contract version: {version!r}.")
        input_options = require_json_mapping(
            payload["input_options"],
            role="Training input contract.input_options",
        )
        incomplete_reasons = json_list(
            payload,
            "incomplete_reasons",
            role=role,
        )
        if any(type(item) is not str for item in incomplete_reasons):
            raise TypeError(
                "Training input contract.incomplete_reasons entries must be JSON strings."
            )
        return cls(
            algorithm=json_string(payload, "algorithm", role=role),
            data_execution_fingerprint=json_string(
                payload,
                "data_execution_fingerprint",
                role=role,
            ),
            data_execution_contract_complete=json_bool(
                payload,
                "data_execution_contract_complete",
                role=role,
            ),
            incomplete_reasons=tuple(incomplete_reasons),
            train_dataset_signature=json_string(
                payload,
                "train_dataset_signature",
                role=role,
            ),
            model_plan_fingerprint=json_string(
                payload,
                "model_plan_fingerprint",
                role=role,
            ),
            model_adapter_signature=json_string(
                payload,
                "model_adapter_signature",
                role=role,
            ),
            processor_signature=json_string(
                payload,
                "processor_signature",
                role=role,
            ),
            tokenizer_signature=json_string(
                payload,
                "tokenizer_signature",
                role=role,
            ),
            template_signature=json_string(
                payload,
                "template_signature",
                role=role,
            ),
            input_builder_signature=json_string(
                payload,
                "input_builder_signature",
                role=role,
            ),
            input_policy_version=json_string(
                payload,
                "input_policy_version",
                role=role,
            ),
            input_options=tuple(
                sorted(
                    (
                        name,
                        validate_json_value(
                            value,
                            role=f"Training input contract.input_options.{name}",
                        ),
                    )
                    for name, value in input_options.items()
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


def build_train_input_contract(
    *,
    algorithm: str,
    data_execution_fingerprint: str,
    data_execution_contract_complete: bool,
    data_execution_incomplete_reasons: tuple[str, ...] = (),
    train_dataset_type: type[Any],
    model_plan_fingerprint: str,
    model_adapter: Any,
    processor: Any,
    tokenizer: Any,
    template: Any,
    input_builder: type[Any],
    input_options: Mapping[str, Any],
) -> ShaftTrainInputContract:
    if not isinstance(data_execution_contract_complete, bool):
        raise TypeError("data_execution_contract_complete must be a boolean value.")
    if data_execution_contract_complete and data_execution_incomplete_reasons:
        raise ValueError("A complete data execution contract cannot declare incomplete reasons.")
    incomplete_reasons: list[str] = []
    if not data_execution_contract_complete:
        incomplete_reasons.extend(
            str(reason)
            for reason in (
                data_execution_incomplete_reasons or ("incomplete_data_execution_identity",)
            )
        )

    resolved_options = dict(input_options)
    processor_cost_signature = getattr(
        model_adapter,
        "processor_cost_semantics_signature",
        None,
    )
    if callable(processor_cost_signature):
        try:
            resolved_options["processor_cost_semantics"] = processor_cost_signature(
                processor=processor,
                min_pixels=resolved_options.get("min_pixels"),
                max_pixels=resolved_options.get("max_pixels"),
            )
        except (AttributeError, TypeError, ValueError):
            # Exact cost support is optional for fixed batching. The processor
            # artifact/config and policy implementation signatures still bind the
            # runtime path; model families with exact cost support contribute their
            # stronger model-owned semantic signature here.
            pass
    unresolved_option_types: set[str] = set()
    normalized_options = tuple(
        sorted(
            (
                str(name),
                _canonical_value(
                    value,
                    unresolved_types=unresolved_option_types,
                ),
            )
            for name, value in resolved_options.items()
        )
    )
    incomplete_reasons.extend(
        f"unresolved_input_option_type:{qualified_type}"
        for qualified_type in sorted(unresolved_option_types)
    )

    train_dataset_signature, dataset_incomplete_reasons = _component_semantic_identity(
        train_dataset_type,
        role="train_dataset",
    )
    incomplete_reasons.extend(dataset_incomplete_reasons)

    tokenizer_component_signature, tokenizer_component_reasons = _component_semantic_identity(
        tokenizer,
        role="tokenizer",
    )
    incomplete_reasons.extend(tokenizer_component_reasons)
    try:
        tokenizer_artifact_signature = tokenizer_artifact_fingerprint(tokenizer)
    except ValueError:
        tokenizer_artifact_signature = "incomplete-tokenizer-artifact"
        incomplete_reasons.append("incomplete_tokenizer_artifact_identity")
    tokenizer_signature = hashlib.sha256(
        json.dumps(
            {
                "version": "shaft-tokenizer-semantic-identity-v2",
                "artifact": tokenizer_artifact_signature,
                "implementation": tokenizer_component_signature,
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    input_policy_version = str(getattr(input_builder, "SHAFT_INPUT_POLICY_VERSION", "")).strip()
    if not input_policy_version:
        incomplete_reasons.append("unversioned_input_policy")
        input_policy_version = "unversioned"

    component_signatures: dict[str, str] = {}
    for role, component in (
        ("model_adapter", model_adapter),
        ("processor", processor),
        ("template", template),
        ("input_builder", input_builder),
    ):
        signature, component_incomplete_reasons = _component_semantic_identity(
            component,
            role=role,
        )
        component_signatures[role] = signature
        incomplete_reasons.extend(component_incomplete_reasons)

    return ShaftTrainInputContract(
        algorithm=str(algorithm).strip().lower(),
        data_execution_fingerprint=str(data_execution_fingerprint),
        data_execution_contract_complete=data_execution_contract_complete,
        incomplete_reasons=tuple(sorted(set(incomplete_reasons))),
        train_dataset_signature=train_dataset_signature,
        model_plan_fingerprint=str(model_plan_fingerprint),
        model_adapter_signature=component_signatures["model_adapter"],
        processor_signature=component_signatures["processor"],
        tokenizer_signature=tokenizer_signature,
        template_signature=component_signatures["template"],
        input_builder_signature=component_signatures["input_builder"],
        input_policy_version=input_policy_version,
        input_options=normalized_options,
    )


def _normalize_save_strategy(save_strategy: object) -> str:
    strategy_value = getattr(save_strategy, "value", save_strategy)
    normalized_strategy = str(strategy_value).strip().lower()
    if normalized_strategy not in {"no", "steps", "epoch"}:
        raise ValueError(f"Unsupported checkpoint save strategy: {normalized_strategy!r}.")
    return normalized_strategy


def validate_train_data_identity_checkpointability(
    *,
    data_execution_contract_complete: bool,
    incomplete_reasons: tuple[str, ...],
    train_dataset_type: type[Any],
    save_strategy: object,
    resume_requested: bool,
) -> None:
    """Fail before model loading when data identity cannot support exact resume."""

    if not isinstance(data_execution_contract_complete, bool):
        raise TypeError("data_execution_contract_complete must be a boolean value.")
    if not isinstance(resume_requested, bool):
        raise TypeError("resume_requested must be a boolean value.")
    if data_execution_contract_complete and incomplete_reasons:
        raise ValueError("A complete data execution contract cannot declare incomplete reasons.")
    _, dataset_incomplete_reasons = _component_semantic_identity(
        train_dataset_type,
        role="train_dataset",
    )
    all_incomplete_reasons = tuple(
        sorted(
            {
                *(str(reason) for reason in incomplete_reasons),
                *dataset_incomplete_reasons,
            }
        )
    )
    complete_identity = data_execution_contract_complete and not dataset_incomplete_reasons
    normalized_strategy = _normalize_save_strategy(save_strategy)
    exact_identity_required = resume_requested or normalized_strategy != "no"
    if exact_identity_required and not complete_identity:
        action = "Exact resume" if resume_requested else "Checkpointing"
        raise ValueError(
            f"{action} requires a complete training data identity before model "
            f"loading, but identity is incomplete: {list(all_incomplete_reasons)}. "
            "Version every active online transform and declare an immutable media "
            "snapshot, or start a fresh run with train.save_strategy='no'."
        )


def validate_train_input_checkpointability(
    contract: ShaftTrainInputContract,
    *,
    save_strategy: object,
) -> None:
    normalized_strategy = _normalize_save_strategy(save_strategy)
    if normalized_strategy != "no" and not contract.exact_resume_safe:
        raise ValueError(
            "Checkpointing requires a complete exact-resume training input "
            "contract, but identity is incomplete: "
            f"{list(contract.incomplete_reasons)}. Version every active online "
            "transform/input policy, provide complete tokenizer artifacts and "
            "declare an immutable media snapshot, or set "
            "train.save_strategy='no'."
        )
