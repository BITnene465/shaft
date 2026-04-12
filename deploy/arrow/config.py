from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")

__all__ = [
    "ArrowProtocolSpec",
    "ArrowRuntimeConfig",
    "ArrowStageSpec",
    "load_arrow_config",
]


@dataclass(frozen=True)
class ArrowProtocolSpec:
    labels: tuple[str, ...]
    num_bins: int


@dataclass(frozen=True)
class ArrowStageSpec:
    route: str
    prompt: str
    max_tokens: int
    do_sample: bool
    temperature: float
    top_p: float
    min_pixels: int | None
    max_pixels: int | None


@dataclass(frozen=True)
class ArrowRuntimeConfig:
    protocol: ArrowProtocolSpec
    stage1: ArrowStageSpec
    stage2: ArrowStageSpec
    padding_ratio: float


def load_arrow_config(config_path: str | Path | None = None) -> ArrowRuntimeConfig:
    path = Path(config_path) if config_path is not None else _DEFAULT_CONFIG_PATH
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Arrow deploy config must be a mapping: {path}")

    protocol = _load_protocol_spec(raw.get("protocol", {}))
    stage1 = _load_stage_spec(raw.get("stage1", {}), "stage1")
    stage2 = _load_stage_spec(raw.get("stage2", {}), "stage2")

    pipeline = raw.get("pipeline", {})
    if not isinstance(pipeline, dict):
        raise ValueError("Arrow deploy config field 'pipeline' must be a mapping.")
    padding_ratio = float(pipeline.get("padding_ratio", 0.3))
    if padding_ratio < 0:
        raise ValueError("Arrow deploy config field 'padding_ratio' must be non-negative.")

    return ArrowRuntimeConfig(
        protocol=protocol,
        stage1=stage1,
        stage2=stage2,
        padding_ratio=padding_ratio,
    )


def _load_protocol_spec(raw: Any) -> ArrowProtocolSpec:
    if not isinstance(raw, dict):
        raise ValueError("Arrow deploy config field 'protocol' must be a mapping.")

    labels = raw.get("labels", [])
    if not isinstance(labels, list) or not labels:
        raise ValueError("Arrow deploy config field 'protocol.labels' must be a non-empty list.")
    label_tuple = tuple(str(label) for label in labels)
    num_bins = int(raw.get("num_bins", 1000))
    if num_bins <= 1:
        raise ValueError("Arrow deploy config field 'protocol.num_bins' must be greater than 1.")
    return ArrowProtocolSpec(labels=label_tuple, num_bins=num_bins)


def _load_stage_spec(raw: Any, stage_name: str) -> ArrowStageSpec:
    if not isinstance(raw, dict):
        raise ValueError(f"Arrow deploy config field '{stage_name}' must be a mapping.")
    route = str(raw.get("route", "")).strip()
    prompt = str(raw.get("prompt", "")).strip()
    max_tokens = int(raw.get("max_tokens", 0))
    do_sample = bool(raw.get("do_sample", False))
    temperature = float(raw.get("temperature", 0.0))
    top_p = float(raw.get("top_p", 1.0))
    min_pixels_raw = raw.get("min_pixels")
    max_pixels_raw = raw.get("max_pixels")
    min_pixels = int(min_pixels_raw) if min_pixels_raw is not None else None
    max_pixels = int(max_pixels_raw) if max_pixels_raw is not None else None
    if not route:
        raise ValueError(f"Arrow deploy config field '{stage_name}.route' must be provided.")
    if not prompt:
        raise ValueError(f"Arrow deploy config field '{stage_name}.prompt' must be provided.")
    if max_tokens <= 0:
        raise ValueError(f"Arrow deploy config field '{stage_name}.max_tokens' must be positive.")
    if not do_sample:
        temperature = 0.0
        top_p = 1.0
    if top_p <= 0 or top_p > 1:
        raise ValueError(f"Arrow deploy config field '{stage_name}.top_p' must be in (0, 1].")
    if temperature < 0:
        raise ValueError(f"Arrow deploy config field '{stage_name}.temperature' must be non-negative.")
    if min_pixels is not None and min_pixels <= 0:
        raise ValueError(f"Arrow deploy config field '{stage_name}.min_pixels' must be positive when provided.")
    if max_pixels is not None and max_pixels <= 0:
        raise ValueError(f"Arrow deploy config field '{stage_name}.max_pixels' must be positive when provided.")
    if min_pixels is not None and max_pixels is not None and min_pixels > max_pixels:
        raise ValueError(f"Arrow deploy config field '{stage_name}.min_pixels' cannot exceed max_pixels.")
    return ArrowStageSpec(
        route=route,
        prompt=prompt,
        max_tokens=max_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
