from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable


@dataclass
class VLLMGenerationBackendConfig:
    """Shared vLLM runtime topology; domain configs own whether it is selected."""

    mode: str = "server"  # server | colocate
    model_impl: str = "vllm"  # auto | vllm | transformers
    trust_remote_code: bool = False
    enable_sleep_mode: bool = False
    structured_outputs_regex: str | None = None
    server_base_url: str | None = None
    server_host: str = "0.0.0.0"
    server_port: int = 8000
    server_timeout: float = 240.0
    group_port: int = 51216
    gpu_memory_utilization: float = 0.3
    max_model_length: int | None = None
    max_num_seqs: int | None = None
    tensor_parallel_size: int = 1


@dataclass
class GRPOVLLMConfig(VLLMGenerationBackendConfig):
    enabled: bool = False


@dataclass
class OPDVLLMConfig(VLLMGenerationBackendConfig):
    pass


def normalize_vllm_generation_backend(
    config: VLLMGenerationBackendConfig,
    *,
    field_prefix: str,
    normalize_bool: Callable[[Any, str], bool] | None = None,
) -> None:
    """Normalize one shared vLLM topology without domain-specific selection rules."""

    bool_parser = normalize_bool or _require_bool
    config.mode = str(config.mode).strip().lower()
    if config.mode not in {"server", "colocate"}:
        raise ValueError(f"{field_prefix}.mode must be 'server' or 'colocate'.")
    config.model_impl = str(config.model_impl).strip().lower()
    if config.model_impl not in {"auto", "vllm", "transformers"}:
        raise ValueError(
            f"{field_prefix}.model_impl must be 'auto', 'vllm', or 'transformers'."
        )
    config.enable_sleep_mode = bool_parser(
        config.enable_sleep_mode,
        f"{field_prefix}.enable_sleep_mode",
    )
    config.trust_remote_code = bool_parser(
        config.trust_remote_code,
        f"{field_prefix}.trust_remote_code",
    )
    config.structured_outputs_regex = (
        str(config.structured_outputs_regex)
        if config.structured_outputs_regex is not None
        else None
    )
    config.server_base_url = (
        str(config.server_base_url).strip() if config.server_base_url is not None else None
    )
    config.server_host = str(config.server_host).strip() or "0.0.0.0"
    config.server_port = int(config.server_port)
    config.server_timeout = float(config.server_timeout)
    config.group_port = int(config.group_port)
    config.gpu_memory_utilization = float(config.gpu_memory_utilization)
    config.tensor_parallel_size = int(config.tensor_parallel_size)
    if config.max_model_length is not None:
        config.max_model_length = int(config.max_model_length)
    if config.max_num_seqs is not None:
        config.max_num_seqs = int(config.max_num_seqs)
    if config.server_port <= 0:
        raise ValueError(f"{field_prefix}.server_port must be > 0.")
    if not math.isfinite(config.server_timeout) or config.server_timeout <= 0:
        raise ValueError(f"{field_prefix}.server_timeout must be finite and > 0.")
    if config.group_port <= 0:
        raise ValueError(f"{field_prefix}.group_port must be > 0.")
    if not math.isfinite(config.gpu_memory_utilization) or not (
        0.0 < config.gpu_memory_utilization <= 1.0
    ):
        raise ValueError(f"{field_prefix}.gpu_memory_utilization must be in (0, 1].")
    if config.max_model_length is not None and config.max_model_length <= 0:
        raise ValueError(f"{field_prefix}.max_model_length must be > 0 when configured.")
    if config.max_num_seqs is not None and config.max_num_seqs <= 0:
        raise ValueError(f"{field_prefix}.max_num_seqs must be > 0 when configured.")
    if config.tensor_parallel_size <= 0:
        raise ValueError(f"{field_prefix}.tensor_parallel_size must be > 0.")


def _require_bool(value: Any, field_name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{field_name} must be a boolean.")
    return value
