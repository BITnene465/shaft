from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin, get_type_hints

import yaml

from .codec import CODEC_REGISTRY
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
    config = _build_dataclass(InferPipelineConfig, payload)
    _validate_infer_config(config)
    return config


def _validate_infer_config(config: InferPipelineConfig) -> None:
    if not config.engines:
        raise ValueError("Infer config requires at least one engine.")
    if not config.stages:
        raise ValueError("Infer config requires at least one stage.")
    known_engines = set(config.engines.keys())
    for name, engine in config.engines.items():
        backend = str(engine.backend).strip().lower()
        if backend not in {"hf_local", "vllm_openai"}:
            raise ValueError(
                f"engines.{name}.backend={engine.backend!r} is unsupported. "
                "Expected one of {'hf_local', 'vllm_openai'}."
            )
        if backend == "vllm_openai":
            endpoint = str(engine.endpoint or "").strip()
            if not endpoint:
                raise ValueError(f"engines.{name}.endpoint is required when backend='vllm_openai'.")
            if float(engine.request_timeout_seconds) <= 0:
                raise ValueError(f"engines.{name}.request_timeout_seconds must be > 0.")
            model_name = str(engine.served_model_name or engine.model_name_or_path).strip()
            if not model_name:
                raise ValueError(
                    f"engines.{name}.served_model_name or model_name_or_path must be non-empty."
                )
    for index, stage in enumerate(config.stages):
        if stage.engine not in known_engines:
            raise ValueError(
                f"stages[{index}].engine={stage.engine!r} not found in engines: {sorted(known_engines)}."
            )
        if int(stage.max_retries) < 0:
            raise ValueError(f"stages[{index}].max_retries must be >= 0.")
        if float(stage.retry_backoff_seconds) < 0:
            raise ValueError(f"stages[{index}].retry_backoff_seconds must be >= 0.")
        if stage.timeout_seconds is not None and float(stage.timeout_seconds) <= 0:
            raise ValueError(f"stages[{index}].timeout_seconds must be > 0.")
        if stage.min_pixels is not None and int(stage.min_pixels) <= 0:
            raise ValueError(f"stages[{index}].min_pixels must be > 0.")
        if stage.max_pixels is not None and int(stage.max_pixels) <= 0:
            raise ValueError(f"stages[{index}].max_pixels must be > 0.")
        if stage.min_pixels is not None and stage.max_pixels is not None:
            if int(stage.min_pixels) > int(stage.max_pixels):
                raise ValueError(
                    f"stages[{index}].min_pixels must be <= max_pixels."
                )
        if not isinstance(stage.backend_options, dict):
            raise ValueError(f"stages[{index}].backend_options must be a mapping.")
        codec_name = str(stage.codec).strip().lower()
        if not CODEC_REGISTRY.has(codec_name):
            raise ValueError(
                f"stages[{index}].codec={stage.codec!r} is unregistered. "
                f"Registered codecs: {sorted(CODEC_REGISTRY.keys())}."
            )
