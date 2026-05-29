from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
from typing import Any


@dataclass(frozen=True)
class GpuInfo:
    index: str
    total_mb: int | None = None
    free_mb: int | None = None


@dataclass(frozen=True)
class RuntimePlacement:
    cuda_visible_devices: str | None
    tensor_parallel_size: int


def parse_cuda_visible_devices(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    devices = []
    for item in str(value).split(","):
        normalized = item.strip()
        if normalized and normalized != "-1":
            devices.append(normalized)
    return devices


def detect_cuda_devices() -> list[GpuInfo]:
    env_devices = parse_cuda_visible_devices(os.environ.get("CUDA_VISIBLE_DEVICES"))
    if env_devices:
        return [GpuInfo(index=item) for item in env_devices]
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    devices: list[GpuInfo] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if not parts or not parts[0]:
            continue
        devices.append(
            GpuInfo(
                index=parts[0],
                total_mb=_optional_int(parts[1]) if len(parts) > 1 else None,
                free_mb=_optional_int(parts[2]) if len(parts) > 2 else None,
            )
        )
    return devices


def resolve_vllm_runtime_placement(
    *,
    model_path: str | Path | None,
    cuda_visible_devices: Any,
    tensor_parallel_size: Any,
) -> RuntimePlacement:
    requested_devices = parse_cuda_visible_devices(cuda_visible_devices)
    detected_devices = detect_cuda_devices()
    detected_ids = [item.index for item in detected_devices]
    candidate_devices = requested_devices or detected_ids
    requested_tp = _resolve_requested_tp(tensor_parallel_size)

    if requested_tp is not None:
        selected_devices = candidate_devices[:requested_tp] if candidate_devices else requested_devices
        return RuntimePlacement(
            cuda_visible_devices=",".join(selected_devices) if selected_devices else None,
            tensor_parallel_size=requested_tp,
        )

    if candidate_devices:
        selected_tp = _largest_attention_head_divisor(
            model_path=model_path,
            max_devices=len(candidate_devices),
        )
        selected_devices = candidate_devices[:selected_tp]
        return RuntimePlacement(
            cuda_visible_devices=",".join(selected_devices),
            tensor_parallel_size=selected_tp,
        )

    return RuntimePlacement(cuda_visible_devices=None, tensor_parallel_size=1)


def _largest_attention_head_divisor(
    *,
    model_path: str | Path | None,
    max_devices: int,
) -> int:
    if max_devices <= 1:
        return 1
    attention_heads = _model_attention_heads(model_path)
    if attention_heads is None:
        return max_devices
    for value in range(max_devices, 0, -1):
        if attention_heads % value == 0:
            return value
    return 1


def _model_attention_heads(model_path: str | Path | None) -> int | None:
    if not model_path:
        return None
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return None
    try:
        import json

        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    text_config = payload.get("text_config")
    values: list[Any] = []
    if isinstance(text_config, dict):
        values.append(text_config.get("num_attention_heads"))
    values.append(payload.get("num_attention_heads"))
    for value in values:
        parsed = _optional_int(value)
        if parsed is not None:
            return parsed
    return None


def _resolve_requested_tp(value: Any) -> int | None:
    if value in (None, "", "auto"):
        return None
    parsed = _optional_int(value)
    if parsed is None or parsed <= 0:
        raise ValueError("tensor_parallel_size must be a positive integer or 'auto'.")
    return parsed


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
