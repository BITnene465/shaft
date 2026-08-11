from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

from .generation_backend import OPDVLLMConfig, normalize_vllm_generation_backend


_TORCH_DTYPES = {
    "auto",
    "bf16",
    "bfloat16",
    "fp16",
    "float16",
    "half",
    "fp32",
    "float32",
}
_DIVERGENCES = {"forward_kl", "reverse_kl", "jsd"}
_OBJECTIVE_MODES = {"full_vocab", "topk_tail"}


@dataclass
class OPDRemoteTeacherConfig:
    endpoint: str = ""
    artifact_fingerprint: str = ""
    api_key_env: str | None = None
    request_timeout_seconds: float = 120.0
    max_request_bytes: int = 512 * 1024 * 1024
    max_response_bytes: int = 512 * 1024 * 1024


@dataclass
class OPDTeacherConfig:
    provider: str = "hf_local"
    model_type: str = ""
    model_name_or_path: str = ""
    revision: str | None = None
    cache_dir: str | None = None
    local_files_only: bool = False
    template: str | None = None
    trust_remote_code: bool = True
    attn_implementation: str | None = None
    torch_dtype: str = "bfloat16"
    remote: OPDRemoteTeacherConfig = field(default_factory=OPDRemoteTeacherConfig)


@dataclass
class OPDRolloutConfig:
    backend: str = "hf_local"
    max_new_tokens: int = 256
    do_sample: bool = True
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: float = 1.0
    min_p: float = 0.0
    vllm: OPDVLLMConfig = field(default_factory=OPDVLLMConfig)


@dataclass
class OPDObjectiveConfig:
    mode: str = "full_vocab"
    divergence: str = ""
    temperature: float = 1.0
    top_k: int | None = None
    token_chunk_size: int | None = None


@dataclass
class OPDConfig:
    teacher: OPDTeacherConfig = field(default_factory=OPDTeacherConfig)
    rollout: OPDRolloutConfig = field(default_factory=OPDRolloutConfig)
    objective: OPDObjectiveConfig = field(default_factory=OPDObjectiveConfig)


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    return str(value).strip() or None


def normalize_opd_runtime_config(config: Any) -> None:
    """Validate OPD-only semantics after generic runtime normalization."""

    opd = config.opd
    teacher = opd.teacher
    rollout = opd.rollout
    objective = opd.objective

    teacher.provider = str(teacher.provider).strip().lower()
    if not teacher.provider:
        raise ValueError("opd.teacher.provider must not be empty.")
    if teacher.provider not in {"hf_local", "http"}:
        raise ValueError("opd.teacher.provider must be 'hf_local' or 'http'.")
    teacher.model_type = str(teacher.model_type).strip().lower()
    teacher.model_name_or_path = str(teacher.model_name_or_path).strip()
    if not teacher.model_type:
        raise ValueError("opd.teacher.model_type must not be empty.")
    if teacher.provider == "hf_local" and not teacher.model_name_or_path:
        raise ValueError("opd.teacher.model_name_or_path must not be empty.")
    if teacher.model_type != config.model.model_type:
        raise ValueError(
            "OPD requires teacher/student model_type equality so processor and "
            "multimodal token contracts are identical."
        )
    teacher.revision = _optional_text(teacher.revision)
    teacher.cache_dir = _optional_text(teacher.cache_dir)
    teacher.template = _optional_text(teacher.template)
    teacher.attn_implementation = _optional_text(teacher.attn_implementation)
    teacher.torch_dtype = str(teacher.torch_dtype).strip().lower()
    if teacher.torch_dtype not in _TORCH_DTYPES:
        raise ValueError(
            f"Unsupported opd.teacher.torch_dtype={teacher.torch_dtype!r}."
        )
    remote = teacher.remote
    remote.endpoint = str(remote.endpoint).strip().rstrip("/")
    remote.artifact_fingerprint = str(remote.artifact_fingerprint).strip().lower()
    remote.api_key_env = _optional_text(remote.api_key_env)
    remote.request_timeout_seconds = float(remote.request_timeout_seconds)
    remote.max_request_bytes = int(remote.max_request_bytes)
    remote.max_response_bytes = int(remote.max_response_bytes)
    if teacher.provider == "http":
        if not remote.endpoint.startswith(("http://", "https://")):
            raise ValueError("opd.teacher.remote.endpoint must be an HTTP(S) base URL.")
        if len(remote.artifact_fingerprint) != 64 or any(
            character not in "0123456789abcdef"
            for character in remote.artifact_fingerprint
        ):
            raise ValueError(
                "opd.teacher.remote.artifact_fingerprint must be a SHA-256 digest."
            )
    for field_name in ("request_timeout_seconds",):
        value = getattr(remote, field_name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"opd.teacher.remote.{field_name} must be finite and > 0.")
    for field_name in ("max_request_bytes", "max_response_bytes"):
        if getattr(remote, field_name) <= 0:
            raise ValueError(f"opd.teacher.remote.{field_name} must be > 0.")

    rollout.backend = str(rollout.backend).strip().lower()
    if not rollout.backend:
        raise ValueError("opd.rollout.backend must not be empty.")
    if rollout.backend not in {"hf_local", "vllm"}:
        raise ValueError("opd.rollout.backend must be 'hf_local' or 'vllm'.")
    rollout.max_new_tokens = int(rollout.max_new_tokens)
    rollout.top_k = int(rollout.top_k)
    for field_name in ("temperature", "top_p", "min_p", "repetition_penalty"):
        value = float(getattr(rollout, field_name))
        if not math.isfinite(value):
            raise ValueError(f"opd.rollout.{field_name} must be finite.")
        setattr(rollout, field_name, value)
    if rollout.max_new_tokens <= 0:
        raise ValueError("opd.rollout.max_new_tokens must be > 0.")
    if rollout.temperature <= 0:
        raise ValueError("opd.rollout.temperature must be > 0.")
    if not 0 < rollout.top_p <= 1:
        raise ValueError("opd.rollout.top_p must be in (0, 1].")
    if rollout.top_k < 0:
        raise ValueError("opd.rollout.top_k must be >= 0.")
    if not 0 <= rollout.min_p <= 1:
        raise ValueError("opd.rollout.min_p must be in [0, 1].")
    if rollout.repetition_penalty <= 0:
        raise ValueError("opd.rollout.repetition_penalty must be > 0.")
    normalize_vllm_generation_backend(
        rollout.vllm,
        field_prefix="opd.rollout.vllm",
    )

    objective.mode = str(objective.mode).strip().lower()
    if objective.mode not in _OBJECTIVE_MODES:
        raise ValueError(
            f"opd.objective.mode must be one of {sorted(_OBJECTIVE_MODES)}."
        )
    objective.divergence = str(objective.divergence).strip().lower()
    if objective.divergence not in _DIVERGENCES:
        raise ValueError(
            "opd.objective.divergence must be explicitly set to one of "
            f"{sorted(_DIVERGENCES)}."
        )
    objective.temperature = float(objective.temperature)
    if not math.isfinite(objective.temperature) or objective.temperature <= 0:
        raise ValueError("opd.objective.temperature must be finite and > 0.")
    if objective.top_k is not None:
        objective.top_k = int(objective.top_k)
        if objective.top_k <= 0:
            raise ValueError("opd.objective.top_k must be > 0 when configured.")
    if objective.mode == "topk_tail" and objective.top_k is None:
        raise ValueError("opd.objective.mode='topk_tail' requires opd.objective.top_k.")
    if objective.mode == "full_vocab" and objective.top_k is not None:
        raise ValueError("opd.objective.mode='full_vocab' does not accept top_k.")
    if objective.token_chunk_size is not None:
        objective.token_chunk_size = int(objective.token_chunk_size)
        if objective.token_chunk_size <= 0:
            raise ValueError("opd.objective.token_chunk_size must be > 0 when configured.")

    if config.data.max_length is None:
        raise ValueError("OPD requires data.max_length as a strict prompt+completion limit.")
    if int(config.data.max_length) <= rollout.max_new_tokens:
        raise ValueError(
            "data.max_length must exceed opd.rollout.max_new_tokens and leave room "
            "for at least one prompt token."
        )
    batching = config.data.batching
    if (
        batching.grouping != "none"
        or batching.cardinality != "fixed"
        or batching.packing.mode != "none"
        or batching.layout != "padded"
    ):
        raise ValueError(
            "OPD supports only grouping='none', cardinality='fixed', "
            "packing.mode='none', layout='padded'."
        )
    if config.eval.enabled:
        raise ValueError("OPD evaluation is not implemented; set eval.enabled=false.")
    if config.train.loss_name != "auto" or config.train.loss_scale != "default":
        raise ValueError(
            "OPD owns its distribution objective; train.loss_name must be 'auto' and "
            "train.loss_scale must be 'default'."
        )
