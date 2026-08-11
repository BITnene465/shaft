from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
import struct
from types import SimpleNamespace
from typing import Any, Mapping, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors
import torch

from .loss import OPDTeacherDistribution, resolve_opd_objective_plan
from .teacher import OPDTeacherScoreRequest


PROTOCOL_VERSION = "shaft-opd-teacher-v1"
CONTENT_TYPE = "application/vnd.shaft.opd-teacher-v1+safetensors"
_MAGIC = b"SOPD1"
_HEADER_LIMIT = 1024 * 1024


@dataclass(frozen=True, slots=True)
class OPDTeacherIdentity:
    protocol_version: str
    artifact_fingerprint: str
    model_type: str
    tokenizer_fingerprint: str
    processor_fingerprint: str
    vocab_size: int

    def __post_init__(self) -> None:
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError(
                f"Unsupported OPD teacher protocol {self.protocol_version!r}; "
                f"expected {PROTOCOL_VERSION!r}."
            )
        for field_name in (
            "artifact_fingerprint",
            "tokenizer_fingerprint",
            "processor_fingerprint",
        ):
            value = str(getattr(self, field_name)).strip().lower()
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(
                    f"OPD teacher identity {field_name} must be a SHA-256 digest."
                )
        if not str(self.model_type).strip():
            raise ValueError("OPD teacher identity model_type must not be empty.")
        if type(self.vocab_size) is not int or self.vocab_size <= 0:
            raise ValueError("OPD teacher identity vocab_size must be > 0.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "artifact_fingerprint": self.artifact_fingerprint,
            "model_type": self.model_type,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "processor_fingerprint": self.processor_fingerprint,
            "vocab_size": self.vocab_size,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "OPDTeacherIdentity":
        expected = {
            "protocol_version",
            "artifact_fingerprint",
            "model_type",
            "tokenizer_fingerprint",
            "processor_fingerprint",
            "vocab_size",
        }
        if set(payload) != expected:
            raise ValueError(
                "OPD teacher identity fields differ from the protocol; "
                f"missing={sorted(expected - set(payload))} "
                f"extra={sorted(set(payload) - expected)}."
            )
        return cls(
            protocol_version=str(payload["protocol_version"]),
            artifact_fingerprint=str(payload["artifact_fingerprint"]),
            model_type=str(payload["model_type"]),
            tokenizer_fingerprint=str(payload["tokenizer_fingerprint"]),
            processor_fingerprint=str(payload["processor_fingerprint"]),
            vocab_size=int(payload["vocab_size"]),
        )


def _encode_envelope(metadata: Mapping[str, Any], tensors: Mapping[str, torch.Tensor]) -> bytes:
    header = json.dumps(
        dict(metadata),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    if len(header) > _HEADER_LIMIT:
        raise ValueError("OPD teacher protocol metadata exceeds the 1 MiB header limit.")
    normalized_tensors: dict[str, torch.Tensor] = {}
    for name, tensor in tensors.items():
        if not torch.is_tensor(tensor):
            raise TypeError(f"OPD teacher protocol tensor {name!r} is not a tensor.")
        normalized_tensors[str(name)] = tensor.detach().to(device="cpu").contiguous()
    tensor_payload = save_safetensors(normalized_tensors)
    return _MAGIC + struct.pack(">I", len(header)) + header + tensor_payload


def _decode_envelope(payload: bytes, *, max_bytes: int) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    if len(payload) > int(max_bytes):
        raise ValueError(
            f"OPD teacher protocol payload has {len(payload)} bytes; limit={int(max_bytes)}."
        )
    prefix_length = len(_MAGIC) + 4
    if len(payload) < prefix_length or payload[: len(_MAGIC)] != _MAGIC:
        raise ValueError("OPD teacher protocol payload has an invalid magic header.")
    header_length = struct.unpack(">I", payload[len(_MAGIC) : prefix_length])[0]
    if header_length > _HEADER_LIMIT:
        raise ValueError("OPD teacher protocol header exceeds the 1 MiB limit.")
    header_stop = prefix_length + header_length
    if header_stop > len(payload):
        raise ValueError("OPD teacher protocol payload has a truncated metadata header.")
    try:
        metadata = json.loads(
            payload[prefix_length:header_stop].decode("utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("OPD teacher protocol metadata is not valid JSON.") from exc
    if not isinstance(metadata, dict):
        raise ValueError("OPD teacher protocol metadata root must be an object.")
    try:
        tensors = load_safetensors(payload[header_stop:])
    except Exception as exc:  # noqa: BLE001 - normalize third-party parser failures
        raise ValueError("OPD teacher protocol tensor payload is invalid.") from exc
    return metadata, tensors


def encode_teacher_score_request(request: OPDTeacherScoreRequest) -> bytes:
    tensors: dict[str, torch.Tensor] = {
        "causal_position_mask": request.causal_position_mask,
    }
    model_kwargs: dict[str, Any] = {}
    for name, value in request.model_inputs.items():
        if torch.is_tensor(value):
            tensors[f"model_input.{name}"] = value
            continue
        if value is None or type(value) in {bool, int, float, str}:
            if type(value) is float and not math.isfinite(value):
                raise ValueError(
                    "External OPD teacher model inputs cannot contain non-finite floats; "
                    f"field={name!r}."
                )
            model_kwargs[str(name)] = value
            continue
        raise TypeError(
            "External OPD teacher supports tensor or JSON-scalar model inputs only; "
            f"field={name!r}."
        )
    objective = request.objective_plan
    return _encode_envelope(
        {
            "protocol_version": PROTOCOL_VERSION,
            "request_ids": list(request.request_ids),
            "model_kwargs": model_kwargs,
            "objective": {
                "mode": objective.mode,
                "divergence": objective.divergence,
                "temperature": objective.temperature,
                "top_k": objective.top_k,
                "token_chunk_size": objective.token_chunk_size,
            },
        },
        tensors,
    )


def decode_teacher_score_request(payload: bytes, *, max_bytes: int) -> OPDTeacherScoreRequest:
    metadata, tensors = _decode_envelope(payload, max_bytes=max_bytes)
    if metadata.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("OPD teacher score request protocol version mismatch.")
    objective_payload = metadata.get("objective")
    if not isinstance(objective_payload, dict):
        raise ValueError("OPD teacher score request has no objective object.")
    objective = resolve_opd_objective_plan(SimpleNamespace(**objective_payload))
    request_ids = metadata.get("request_ids")
    if not isinstance(request_ids, list) or not all(
        isinstance(value, str) and value for value in request_ids
    ):
        raise ValueError("OPD teacher score request IDs must be non-empty strings.")
    model_kwargs = metadata.get("model_kwargs")
    if not isinstance(model_kwargs, dict) or not all(
        isinstance(name, str)
        and (value is None or type(value) in {bool, int, float, str})
        for name, value in model_kwargs.items()
    ):
        raise ValueError("OPD teacher score request model_kwargs must be JSON scalars.")
    causal_position_mask = tensors.pop("causal_position_mask", None)
    if causal_position_mask is None:
        raise ValueError("OPD teacher score request has no causal_position_mask.")
    model_inputs = {
        name.removeprefix("model_input."): tensor
        for name, tensor in tensors.items()
        if name.startswith("model_input.")
    }
    if len(model_inputs) != len(tensors):
        unknown = sorted(name for name in tensors if not name.startswith("model_input."))
        raise ValueError(f"OPD teacher score request has unknown tensors: {unknown}.")
    overlap = set(model_inputs) & set(model_kwargs)
    if overlap:
        raise ValueError(f"OPD teacher score request duplicates model inputs: {sorted(overlap)}.")
    model_inputs.update(model_kwargs)
    return OPDTeacherScoreRequest(
        model_inputs=model_inputs,
        causal_position_mask=causal_position_mask.to(dtype=torch.bool),
        request_ids=tuple(request_ids),
        objective_plan=objective,
    )


def encode_teacher_distribution(distribution: OPDTeacherDistribution) -> bytes:
    tensors: dict[str, torch.Tensor] = {}
    for name in ("dense_logits", "topk_token_ids", "topk_log_probs", "tail_log_probs"):
        value = getattr(distribution, name)
        if value is not None:
            tensors[name] = value
    return _encode_envelope(
        {
            "protocol_version": PROTOCOL_VERSION,
            "kind": distribution.kind,
            "vocab_size": distribution.vocab_size,
            "temperature": distribution.temperature,
        },
        tensors,
    )


def decode_teacher_distribution(payload: bytes, *, max_bytes: int) -> OPDTeacherDistribution:
    metadata, tensors = _decode_envelope(payload, max_bytes=max_bytes)
    if metadata.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("OPD teacher score response protocol version mismatch.")
    allowed = {"dense_logits", "topk_token_ids", "topk_log_probs", "tail_log_probs"}
    if not set(tensors).issubset(allowed):
        raise ValueError(
            f"OPD teacher score response has unknown tensors: {sorted(set(tensors) - allowed)}."
        )
    return OPDTeacherDistribution(
        kind=str(metadata.get("kind", "")),
        vocab_size=int(metadata.get("vocab_size", 0)),
        dense_logits=tensors.get("dense_logits"),
        topk_token_ids=tensors.get("topk_token_ids"),
        topk_log_probs=tensors.get("topk_log_probs"),
        tail_log_probs=tensors.get("tail_log_probs"),
        temperature=(
            None if metadata.get("temperature") is None else float(metadata["temperature"])
        ),
    )


class OPDTeacherHTTPTransport(Protocol):
    def get_identity(self) -> OPDTeacherIdentity: ...

    def score(self, payload: bytes, *, idempotency_key: str) -> bytes: ...


class UrllibOPDTeacherHTTPTransport:
    """Dependency-free HTTP transport with bounded response reads and no secret persistence."""

    def __init__(self, config: Any) -> None:
        self.endpoint = str(config.endpoint).rstrip("/")
        self.timeout = float(config.request_timeout_seconds)
        self.max_request_bytes = int(config.max_request_bytes)
        self.max_response_bytes = int(config.max_response_bytes)
        self._api_key = None
        if config.api_key_env:
            self._api_key = os.environ.get(str(config.api_key_env))
            if not self._api_key:
                raise ValueError(
                    f"OPD teacher API key environment variable {config.api_key_env!r} is empty."
                )

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": CONTENT_TYPE}
        if self._api_key is not None:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _read(self, request: Request, *, max_bytes: int) -> bytes:
        try:
            with urlopen(request, timeout=self.timeout) as response:  # noqa: S310 - configured URL
                payload = response.read(max_bytes + 1)
        except HTTPError as exc:
            detail = exc.read(4096).decode("utf-8", errors="replace")
            raise RuntimeError(
                f"OPD teacher HTTP request failed with status={exc.code}: {detail}"
            ) from exc
        except (URLError, TimeoutError) as exc:
            raise RuntimeError(f"OPD teacher HTTP request failed: {exc}") from exc
        if len(payload) > max_bytes:
            raise ValueError(f"OPD teacher HTTP response exceeds {max_bytes} bytes.")
        return payload

    def get_identity(self) -> OPDTeacherIdentity:
        request = Request(
            f"{self.endpoint}/v1/identity",
            headers={**self._headers(), "Accept": "application/json"},
            method="GET",
        )
        payload = self._read(request, max_bytes=min(self.max_response_bytes, _HEADER_LIMIT))
        try:
            decoded = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("OPD teacher identity response is not valid JSON.") from exc
        if not isinstance(decoded, dict):
            raise ValueError("OPD teacher identity response root must be an object.")
        return OPDTeacherIdentity.from_mapping(decoded)

    def score(self, payload: bytes, *, idempotency_key: str) -> bytes:
        if len(payload) > self.max_request_bytes:
            raise ValueError(f"OPD teacher HTTP request exceeds {self.max_request_bytes} bytes.")
        request = Request(
            f"{self.endpoint}/v1/score",
            data=payload,
            headers={
                **self._headers(),
                "Content-Type": CONTENT_TYPE,
                "Idempotency-Key": str(idempotency_key),
            },
            method="POST",
        )
        return self._read(request, max_bytes=self.max_response_bytes)


def teacher_request_idempotency_key(payload: bytes) -> str:
    return hashlib.sha256(PROTOCOL_VERSION.encode("utf-8") + b"\0" + payload).hexdigest()


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant {value!r} is not allowed.")
