from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin, get_type_hints

import yaml

from .dataset_catalog import resolve_dataset_catalog
from .normalize import normalize_runtime_config
from .runtime import RuntimeConfig

T = TypeVar("T")

_DEEPSPEED_SHAFT_MANAGED_KEYS = {"optimizer", "scheduler"}


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


def _resolve_record_cache_dir(payload: dict[str, Any], *, config_path: Path) -> dict[str, Any]:
    data_payload = payload.get("data")
    if data_payload is None:
        return payload
    if not isinstance(data_payload, dict):
        raise TypeError("Config key `data` must be a mapping.")
    cache_dir = data_payload.get("record_cache_dir")
    if cache_dir is None or not str(cache_dir).strip():
        return payload
    path = Path(str(cache_dir)).expanduser()
    if not path.is_absolute():
        path = (config_path.parent / path).resolve()
    data_payload["record_cache_dir"] = str(path)
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
    payload = resolve_dataset_catalog(payload, config_path=config_path.resolve())
    payload = _resolve_record_cache_dir(payload, config_path=config_path.resolve())
    payload = _resolve_deepspeed_config_path(payload, config_path=config_path.resolve())
    config = _build_dataclass(RuntimeConfig, payload)
    return normalize_runtime_config(config)
