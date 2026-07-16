from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import inspect
import json
from typing import Any, Generic, Protocol, TypeVar

import torch

from shaft.utils.contract_schema import validate_json_value
from shaft.utils.semantic_identity import callable_semantic_fingerprint


@dataclass
class AlgorithmContext:
    params: dict[str, Any]


TrainerT = TypeVar("TrainerT")

_COMMON_TRAINER_ARG_FIELDS = (
    "adam_beta1",
    "adam_beta2",
    "adam_epsilon",
    "average_tokens_across_devices",
    "bf16",
    "data_seed",
    "dataloader_drop_last",
    "dataloader_num_workers",
    "dataloader_persistent_workers",
    "dataloader_pin_memory",
    "dataloader_prefetch_factor",
    "ddp_backend",
    "ddp_broadcast_buffers",
    "ddp_bucket_cap_mb",
    "ddp_find_unused_parameters",
    "ddp_static_graph",
    "ddp_timeout",
    "deepspeed",
    "fp16",
    "fsdp",
    "fsdp_config",
    "full_determinism",
    "gradient_accumulation_steps",
    "gradient_checkpointing",
    "gradient_checkpointing_kwargs",
    "learning_rate",
    "lr_scheduler_kwargs",
    "lr_scheduler_type",
    "max_grad_norm",
    "max_steps",
    "num_train_epochs",
    "optim",
    "optim_args",
    "optim_target_modules",
    "per_device_eval_batch_size",
    "per_device_train_batch_size",
    "remove_unused_columns",
    "seed",
    "tf32",
    "torch_compile",
    "torch_compile_backend",
    "torch_compile_mode",
    "use_cpu",
    "warmup_ratio",
    "warmup_steps",
    "weight_decay",
)
_MAX_CONSTRUCTOR_ITEMS = 256
_MAX_MODEL_MODULES = 100_000
_MAX_MODEL_TENSORS = 100_000
_MAX_TENSOR_NAME_BYTES = 4_096


def _qualified_name(value: Any) -> str:
    value_type = value if isinstance(value, type) else type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def model_topology_signature(model: Any) -> dict[str, Any]:
    """Hash bounded module structure without reading parameter or buffer values."""

    if model is None:
        return {"type": None, "topology_sha256": None, "tensor_count": 0}
    if not isinstance(model, torch.nn.Module):
        raise TypeError(
            "Prepared trainer model roles must be torch modules or None, got "
            f"{_qualified_name(model)}."
        )
    digest = hashlib.sha256()
    tensor_count = 0
    module_count = 0
    parameter_count = 0
    buffer_count = 0
    for name, module in model.named_modules():
        module_count += 1
        if module_count > _MAX_MODEL_MODULES:
            raise ValueError(
                "Prepared trainer model topology exceeds the bounded module limit "
                f"of {_MAX_MODEL_MODULES}."
            )
        encoded_name = str(name).encode("utf-8")
        if len(encoded_name) > _MAX_TENSOR_NAME_BYTES:
            raise ValueError("Prepared trainer model topology contains an oversized name.")
        digest.update(
            json.dumps(
                {
                    "kind": "module",
                    "name": str(name),
                    "type": _qualified_name(module),
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    for kind, tensors in (
        ("parameter", model.named_parameters()),
        ("buffer", model.named_buffers()),
    ):
        for name, tensor in tensors:
            tensor_count += 1
            if tensor_count > _MAX_MODEL_TENSORS:
                raise ValueError(
                    "Prepared trainer model topology exceeds the bounded tensor limit "
                    f"of {_MAX_MODEL_TENSORS}."
                )
            encoded_name = str(name).encode("utf-8")
            if len(encoded_name) > _MAX_TENSOR_NAME_BYTES:
                raise ValueError("Prepared trainer model topology contains an oversized name.")
            record = {
                "kind": kind,
                "name": str(name),
                "shape": [int(size) for size in tensor.shape],
                "dtype": str(tensor.dtype),
                "requires_grad": bool(tensor.requires_grad) if kind == "parameter" else False,
            }
            digest.update(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            digest.update(b"\n")
            if kind == "parameter":
                parameter_count += 1
            else:
                buffer_count += 1
    return {
        "type": _qualified_name(model),
        "topology_sha256": digest.hexdigest(),
        "module_count": module_count,
        "tensor_count": tensor_count,
        "parameter_count": parameter_count,
        "buffer_count": buffer_count,
    }


def _prepared_value(value: Any, *, depth: int = 0) -> Any:
    if depth > 3:
        return {"type": _qualified_name(value), "depth_limited": True}
    if value is None or type(value) in {bool, int, float, str}:
        return validate_json_value(value, role="prepared trainer constructor value")
    if isinstance(value, Enum):
        return {
            "enum": _qualified_name(value),
            "value": _prepared_value(value.value, depth=depth + 1),
        }
    if isinstance(value, torch.nn.Module):
        return {"model_topology": model_topology_signature(value)}
    if isinstance(value, type):
        return {"class": _qualified_name(value)}
    if type(value) is dict:
        if len(value) > _MAX_CONSTRUCTOR_ITEMS:
            raise ValueError("Prepared trainer constructor mapping exceeds the bounded item limit.")
        if any(type(key) is not str for key in value):
            raise TypeError("Prepared trainer constructor mappings require string keys.")
        return {
            key: _prepared_value(item, depth=depth + 1)
            for key, item in sorted(value.items())
        }
    if type(value) in {list, tuple}:
        if len(value) > _MAX_CONSTRUCTOR_ITEMS:
            raise ValueError("Prepared trainer constructor sequence exceeds the bounded item limit.")
        return {
            "sequence_type": _qualified_name(value),
            "items": [_prepared_value(item, depth=depth + 1) for item in value],
        }
    fingerprint = getattr(value, "fingerprint", None)
    if type(fingerprint) is str and fingerprint:
        return {"type": _qualified_name(value), "fingerprint": fingerprint}
    if inspect.isfunction(value) or inspect.ismethod(value):
        module = getattr(value, "__module__", None)
        qualname = getattr(value, "__qualname__", None)
        if type(module) is str and type(qualname) is str:
            return {
                "callable": f"{module}.{qualname}",
                "semantic_sha256": callable_semantic_fingerprint(
                    value,
                    role="prepared_trainer_callable",
                ),
            }
    return {"type": _qualified_name(value)}


def _prepared_kwargs_contract(kwargs: dict[str, Any]) -> dict[str, Any]:
    if len(kwargs) > _MAX_CONSTRUCTOR_ITEMS:
        raise ValueError("Prepared trainer kwargs exceed the bounded item limit.")
    return {
        key: _prepared_value(value)
        for key, value in sorted(kwargs.items())
    }


def trainer_spec_contract(
    *,
    algorithm: str,
    args: Any,
    train_config: Any,
    arg_fields: tuple[str, ...] = (),
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Capture constructor-shaping values without live model/dataset state."""

    args_to_dict = getattr(args, "to_dict", None)
    if not callable(args_to_dict):
        raise TypeError("Prepared trainer args must provide to_dict().")
    args_payload = args_to_dict()
    selected_fields = tuple(dict.fromkeys((*_COMMON_TRAINER_ARG_FIELDS, *arg_fields)))
    return validate_json_value(
        {
            "version": 1,
            "algorithm": algorithm,
            "args": {
                field_name: args_payload.get(field_name)
                for field_name in selected_fields
                if field_name in args_payload
            },
            "optimizer": {
                "name": train_config.optimizer_name,
                "scheduler": train_config.scheduler_name,
                "scheduler_num_cycles": train_config.scheduler_num_cycles,
                "scheduler_power": train_config.scheduler_power,
                "adam_beta1": train_config.adam_beta1,
                "adam_beta2": train_config.adam_beta2,
                "adam_epsilon": train_config.adam_epsilon,
                "param_group_lrs": dict(train_config.param_group_lrs),
                "no_decay_name_patterns": list(train_config.no_decay_name_patterns),
            },
            "extra": dict(extra or {}),
        },
        role=f"{algorithm} trainer spec contract",
    )


@dataclass(frozen=True, slots=True)
class ShaftTrainerSpec(Generic[TrainerT]):
    """Pure-local trainer preparation result.

    Creating the spec must not initialize distributed runtime state. ``build`` is
    intentionally the only constructor boundary so pipelines can run preparation
    under rank-status consensus and invoke Trainer/Accelerator construction only
    after every rank is ready.
    """

    trainer_cls: type[TrainerT]
    kwargs: dict[str, Any]
    contract: dict[str, Any]

    @property
    def implementation(self) -> dict[str, str]:
        constructor = inspect.getattr_static(self.trainer_cls, "__init__")
        if isinstance(constructor, (classmethod, staticmethod)):
            constructor = constructor.__func__
        return {
            "type": f"{self.trainer_cls.__module__}.{self.trainer_cls.__qualname__}",
            "semantic_sha256": callable_semantic_fingerprint(
                self.trainer_cls,
                role="trainer_spec.trainer_cls",
                # The class digest still covers its live declared MRO/method
                # surface, including constructor replacements. Trainer modules
                # and their runtime packages are bound by the resume contract;
                # do not recursively traverse mutable HF/TRL registries here.
                include_dependencies=False,
            ),
            "constructor_semantic_sha256": callable_semantic_fingerprint(
                constructor,
                role="trainer_spec.trainer_constructor",
                # Constructor globals directly shape Trainer/Accelerator setup.
                # Bind their live implementations without recursively walking
                # every method and mutable registry on the whole Trainer class.
                include_dependencies=True,
            ),
        }

    @property
    def fingerprint(self) -> str:
        payload = validate_json_value(
            {
                "version": 3,
                "implementation": self.implementation,
                "prepared_kwargs": _prepared_kwargs_contract(self.kwargs),
                "contract": self.contract,
            },
            role="trainer spec fingerprint",
        )
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def build(self) -> TrainerT:
        return self.trainer_cls(**self.kwargs)


class Algorithm(Protocol):
    name: str

    def prepare_trainer(
        self,
        *,
        context: AlgorithmContext,
        **kwargs: Any,
    ) -> ShaftTrainerSpec[Any]: ...

    def build_trainer(self, *, context: AlgorithmContext, **kwargs: Any) -> Any: ...
