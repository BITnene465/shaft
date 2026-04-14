from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin, get_type_hints

import yaml

from .schema import InferPipelineConfig

T = TypeVar("T")


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
            continue
        if _is_list(ann):
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
            continue
        if _is_dict(ann):
            key_type, item_type = get_args(ann)
            if key_type is not str:
                raise TypeError(f"Only str-key dict is supported at {subpath}.")
            if not isinstance(value, dict):
                raise TypeError(f"Expected dict at {subpath}.")
            item_type = _unwrap_optional(item_type)
            if is_dataclass(item_type):
                kwargs[name] = {
                    str(k): _build_dataclass(item_type, v, path=f"{subpath}.{k}")
                    for k, v in value.items()
                }
            else:
                kwargs[name] = value
            continue
        kwargs[name] = value
    return cls(**kwargs)


def load_infer_config(path: str | Path) -> InferPipelineConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise TypeError("Infer config root must be a mapping.")
    return _build_dataclass(InferPipelineConfig, payload)

