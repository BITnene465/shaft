from __future__ import annotations

import json
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin, get_type_hints

import yaml

from .dataset_catalog import resolve_dataset_catalog
from .data import DataBatchingConfig
from .normalize import normalize_runtime_config
from .runtime import RuntimeConfig

T = TypeVar("T")

_DEEPSPEED_SHAFT_MANAGED_KEYS = {"optimizer", "scheduler"}


def to_resolved_payload(config: RuntimeConfig) -> dict[str, Any]:
    """Build a canonical launch snapshot, not a lossless source-YAML roundtrip."""

    payload = asdict(config)
    data = payload["data"]
    if data["datasets"]:
        data.pop("catalog_path", None)
        data.pop("catalog_names", None)
    batching = data["batching"]
    if batching["grouping"] == "none":
        for field_name in (
            "buffer_size",
            "cost_cache_size",
            "max_tokens_per_microbatch",
            "resource_budgets",
        ):
            batching.pop(field_name, None)
    elif batching["grouping"] == "length":
        batching.pop("max_tokens_per_microbatch", None)

    for dataset in data["datasets"]:
        if dataset["train_paths"]:
            dataset.pop("train_path", None)
        if dataset["val_paths"]:
            dataset.pop("val_path", None)
    return payload


def _is_optional(annotation: Any) -> bool:
    origin = get_origin(annotation)
    if origin is None:
        return False
    return type(None) in get_args(annotation)


def _is_dict(annotation: Any) -> bool:
    return get_origin(annotation) is dict


def _is_list(annotation: Any) -> bool:
    return get_origin(annotation) is list


def _unwrap_optional(annotation: Any) -> Any:
    if not _is_optional(annotation):
        return annotation
    return next(arg for arg in get_args(annotation) if arg is not type(None))


def _build_dataclass(cls: type[T], payload: dict[str, Any], *, path: str = "") -> T:
    if not isinstance(payload, dict):
        raise TypeError(f"Config node must be a mapping at {path or '<root>'}.")

    field_map = {f.name: f for f in fields(cls)}
    type_hints = get_type_hints(cls)
    unknown = sorted(set(payload.keys()) - set(field_map.keys()))
    if unknown:
        raise ValueError(f"Unknown config keys at {path or '<root>'}: {unknown}")

    kwargs: dict[str, Any] = {}
    for name, field_obj in field_map.items():
        if name not in payload:
            continue
        value = payload[name]
        ann = _unwrap_optional(type_hints.get(name, field_obj.type))
        subpath = f"{path}.{name}" if path else name
        if is_dataclass(ann):
            kwargs[name] = _build_dataclass(ann, value, path=subpath)
        elif _is_list(ann):
            (item_type,) = get_args(ann)
            item_type = _unwrap_optional(item_type)
            if is_dataclass(item_type):
                if not isinstance(value, list):
                    raise TypeError(f"Expected list at {subpath}.")
                kwargs[name] = [
                    _build_dataclass(item_type, item, path=f"{subpath}[{idx}]")
                    for idx, item in enumerate(value)
                ]
            else:
                kwargs[name] = value
        elif _is_dict(ann):
            if not isinstance(value, dict):
                raise TypeError(f"Expected dict at {subpath}.")
            key_type, item_type = get_args(ann)
            if key_type is not str:
                raise TypeError(f"Only str-key dict is supported at {subpath}.")
            item_type = _unwrap_optional(item_type)
            if is_dataclass(item_type):
                kwargs[name] = {
                    str(k): _build_dataclass(item_type, v, path=f"{subpath}.{k}")
                    for k, v in value.items()
                }
            else:
                kwargs[name] = value
        else:
            kwargs[name] = value
    return cls(**kwargs)


def _resolve_config_relative_path(value: Any, *, config_path: Path) -> str:
    text = str(value).strip()
    path = Path(text)
    if path.is_absolute():
        return str(path)
    return str((config_path.parent / path).resolve())


def _validate_deepspeed_runtime_config(config: dict[str, Any], *, source: str) -> None:
    managed_keys = sorted(set(config) & _DEEPSPEED_SHAFT_MANAGED_KEYS)
    if managed_keys:
        raise ValueError(
            f"{source} contains DeepSpeed-managed keys {managed_keys!r}. "
            "Shaft owns optimizer/scheduler construction so param_group_lrs and scheduler settings "
            "stay consistent; remove those keys from train.distributed.deepspeed."
        )


def _validate_deepspeed_config_path(path: str) -> None:
    config_path = Path(path)
    if not config_path.exists():
        return
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"DeepSpeed config_path is not valid JSON: {config_path}") from exc
    if not isinstance(config, dict):
        raise TypeError(f"DeepSpeed config root must be a mapping: {config_path}")
    _validate_deepspeed_runtime_config(
        config,
        source=f"DeepSpeed config_path {config_path}",
    )


def _validate_explicit_batching_policy(payload: dict[str, Any]) -> None:
    data_payload = payload.get("data")
    if not isinstance(data_payload, dict):
        raise ValueError(
            "Training YAML must explicitly set data.batching.grouping, cardinality, "
            "packing.mode, and layout; "
            "the data mapping is missing."
        )
    batching_payload = data_payload.get("batching")
    if not isinstance(batching_payload, dict):
        raise ValueError(
            "data.batching must be explicit in every training YAML."
        )
    batching_fields = {field.name for field in fields(DataBatchingConfig)}
    unknown_batching_fields = sorted(set(batching_payload) - batching_fields)
    if unknown_batching_fields:
        raise ValueError(
            "Unknown config keys at data.batching: "
            f"{unknown_batching_fields}"
        )
    for field_name in ("grouping", "cardinality", "layout"):
        value = str(batching_payload.get(field_name, "")).strip().lower()
        if not value:
            raise ValueError(
                f"data.batching.{field_name} must be explicit and non-empty "
                "in every training YAML."
            )
    packing_payload = batching_payload.get("packing")
    if not isinstance(packing_payload, dict):
        raise ValueError(
            "data.batching.packing must be explicit in every training YAML."
        )
    packing_mode = str(packing_payload.get("mode", "")).strip().lower()
    if not packing_mode:
        raise ValueError(
            "data.batching.packing.mode must be explicit and non-empty in every "
            "training YAML."
        )
    grouping = str(batching_payload["grouping"]).strip().lower()
    planned_only_fields = {
        "buffer_size",
        "cost_cache_size",
        "resource_budgets",
    }
    ignored_planned_fields = sorted(
        planned_only_fields & batching_payload.keys()
        if grouping == "none"
        else ()
    )
    if ignored_planned_fields:
        raise ValueError(
            "data.batching planning fields require grouping='length' or "
            f"grouping='bounded_cost': {ignored_planned_fields}."
        )
    if grouping != "bounded_cost" and "max_tokens_per_microbatch" in batching_payload:
        raise ValueError(
            "data.batching.max_tokens_per_microbatch is only valid when "
            "grouping='bounded_cost'."
        )

def _resolve_deepspeed_config_path(payload: dict[str, Any], *, config_path: Path) -> dict[str, Any]:
    train_payload = payload.get("train")
    if train_payload is None:
        return payload
    if not isinstance(train_payload, dict):
        raise TypeError("Config key `train` must be a mapping.")
    distributed_payload = train_payload.get("distributed")
    if distributed_payload is None:
        return payload
    if not isinstance(distributed_payload, dict):
        raise TypeError("Config key `train.distributed` must be a mapping.")
    deepspeed_payload = distributed_payload.get("deepspeed")
    if deepspeed_payload is None:
        return payload
    if not isinstance(deepspeed_payload, dict):
        raise TypeError("Config key `train.distributed.deepspeed` must be a mapping.")

    inline_config = deepspeed_payload.get("config")
    if inline_config is not None:
        if not isinstance(inline_config, dict):
            raise TypeError("Config key `train.distributed.deepspeed.config` must be a mapping.")
        _validate_deepspeed_runtime_config(
            inline_config,
            source="train.distributed.deepspeed.config",
        )

    config_path_value = deepspeed_payload.get("config_path")
    if config_path_value is None:
        return payload
    config_path_text = str(config_path_value).strip()
    if not config_path_text:
        return payload
    deepspeed_payload["config_path"] = _resolve_config_relative_path(
        config_path_text,
        config_path=config_path,
    )
    _validate_deepspeed_config_path(str(deepspeed_payload["config_path"]))
    return payload


def _resolve_data_cache_dirs(payload: dict[str, Any], *, config_path: Path) -> dict[str, Any]:
    data_payload = payload.get("data")
    if data_payload is None:
        return payload
    if not isinstance(data_payload, dict):
        raise TypeError("Config key `data` must be a mapping.")
    cache_locations: list[tuple[dict[str, Any], str]] = [
        (data_payload, "record_cache_dir"),
    ]
    for container, key in cache_locations:
        cache_dir = container.get(key)
        if cache_dir is None or not str(cache_dir).strip():
            continue
        path = Path(str(cache_dir).strip()).expanduser()
        if not path.is_absolute():
            path = (config_path.parent / path).resolve()
        container[key] = str(path)
    return payload


def load_config(path: str | Path) -> RuntimeConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return load_config_from_payload(payload, config_path=config_path)


def load_config_from_text(text: str, *, config_path: str | Path) -> RuntimeConfig:
    payload = yaml.safe_load(text) or {}
    return load_config_from_payload(payload, config_path=config_path)


def load_config_from_payload(payload: dict[str, Any], *, config_path: str | Path) -> RuntimeConfig:
    config_path = Path(config_path)
    if not isinstance(payload, dict):
        raise TypeError("Config root must be a mapping.")
    _validate_explicit_batching_policy(payload)
    payload = resolve_dataset_catalog(payload, config_path=config_path.resolve())
    payload = _resolve_data_cache_dirs(payload, config_path=config_path.resolve())
    payload = _resolve_deepspeed_config_path(payload, config_path=config_path.resolve())
    config = _build_dataclass(RuntimeConfig, payload)
    return normalize_runtime_config(config)
