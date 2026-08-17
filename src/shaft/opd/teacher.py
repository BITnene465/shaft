from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch

from shaft.config import OPDTeacherConfig

from .loss import OPDObjectivePlan, OPDTeacherDistribution
from .input_abi import ShaftOPDInputABI, validate_opd_input_abi_compatibility


def _prepare_local_teacher(model: torch.nn.Module, accelerator: Any) -> torch.nn.Module:
    state = accelerator.state
    modes = [
        name
        for name, active in (
            ("deepspeed", getattr(state, "deepspeed_plugin", None) is not None),
            ("fsdp", getattr(state, "fsdp_plugin", None) is not None),
        )
        if active
    ]
    if len(modes) > 1:
        raise RuntimeError(f"OPD teacher observed competing sharded backends: {modes}.")
    if not modes:
        return model.to(accelerator.device)
    try:
        from trl.models.utils import prepare_deepspeed, prepare_fsdp
    except ImportError as exc:
        raise ImportError(
            'Sharded OPD teacher preparation requires `uv pip install -e ".[rlhf]"`.'
        ) from exc
    preparers = {
        "deepspeed": prepare_deepspeed,
        "fsdp": prepare_fsdp,
    }
    return preparers[modes[0]](model, accelerator)


@dataclass(frozen=True, slots=True)
class OPDTeacherScoreRequest:
    """One canonical teacher forward and distribution projection request."""

    model_inputs: dict[str, Any]
    causal_position_mask: torch.Tensor
    request_ids: tuple[str, ...]
    objective_plan: OPDObjectivePlan

    def __post_init__(self) -> None:
        if self.causal_position_mask.ndim != 2:
            raise ValueError("OPD teacher causal_position_mask must be 2-D.")
        if self.causal_position_mask.dtype != torch.bool:
            raise TypeError("OPD teacher causal_position_mask must use torch.bool.")
        input_ids = self.model_inputs.get("input_ids")
        if not torch.is_tensor(input_ids) or input_ids.ndim != 2:
            raise ValueError("OPD teacher request requires 2-D input_ids.")
        expected_batch_size = int(input_ids.shape[0])
        shifted_input_width = max(int(input_ids.shape[1]) - 1, 0)
        actual_batch_size, causal_width = map(int, self.causal_position_mask.shape)
        if actual_batch_size != expected_batch_size or causal_width > shifted_input_width:
            raise ValueError(
                "OPD teacher causal_position_mask must describe a causally shiftable "
                "logit tail within input_ids; "
                f"actual={tuple(self.causal_position_mask.shape)} "
                f"input_ids={tuple(input_ids.shape)}."
            )
        if len(self.request_ids) != int(input_ids.shape[0]):
            raise ValueError("OPD teacher request IDs must match batch cardinality.")
        if any(not str(value).strip() for value in self.request_ids):
            raise ValueError("OPD teacher request IDs must not be empty.")
        if len(set(self.request_ids)) != len(self.request_ids):
            raise ValueError("OPD teacher request IDs must be unique within a local batch.")
        if not bool(self.causal_position_mask.any().item()):
            raise ValueError("OPD teacher request contains no completion positions.")


class OPDTeacherProvider(ABC):
    """Provide frozen teacher scores without exposing provider details to the trainer."""

    name: str
    exact_resume_supported: bool = False

    @classmethod
    @abstractmethod
    def resolve_artifact_plan(
        cls,
        config: Any,
        *,
        checkpointing_requested: bool,
    ) -> Any:
        raise NotImplementedError

    def __init__(
        self,
        config: OPDTeacherConfig,
        *,
        model: torch.nn.Module | None,
    ) -> None:
        self.config = config
        self._model = model
        self.telemetry = None

    def bind_telemetry(self, telemetry: Any) -> None:
        self.telemetry = telemetry

    @abstractmethod
    def validate_student_model(self, student_model: torch.nn.Module) -> None:
        raise NotImplementedError

    @abstractmethod
    def prepare(self, accelerator: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def validate_input_abi(self, input_abi: ShaftOPDInputABI) -> str:
        raise NotImplementedError

    @abstractmethod
    def score(self, request: OPDTeacherScoreRequest) -> OPDTeacherDistribution:
        raise NotImplementedError


class LocalHFOPDTeacherProvider(OPDTeacherProvider):
    """Independent local HF teacher; immutable and stateless across optimizer steps."""

    name = "hf_local"
    exact_resume_supported = True

    @classmethod
    def resolve_artifact_plan(
        cls,
        config: Any,
        *,
        checkpointing_requested: bool,
    ) -> Any:
        from .teacher_assembly import resolve_local_teacher_artifact_plan

        return resolve_local_teacher_artifact_plan(
            config,
            checkpointing_requested=checkpointing_requested,
        )

    def __init__(
        self,
        config: OPDTeacherConfig,
        *,
        model: torch.nn.Module | None,
    ) -> None:
        if model is None:
            raise ValueError("Local HF OPD teacher provider requires a model.")
        super().__init__(config, model=model)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval()

    @property
    def model(self) -> torch.nn.Module:
        assert self._model is not None
        return self._model

    def validate_student_model(self, student_model: torch.nn.Module) -> None:
        if self.model is student_model:
            raise ValueError("OPD teacher and student must be independent module objects.")

    def prepare(self, accelerator: Any) -> None:
        self._model = _prepare_local_teacher(self.model, accelerator)
        self.model.eval()
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise RuntimeError("OPD teacher parameters must all be frozen.")

    def validate_input_abi(self, input_abi: ShaftOPDInputABI) -> str:
        return input_abi.fingerprint

    def score(self, request: OPDTeacherScoreRequest) -> OPDTeacherDistribution:
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**request.model_inputs)
        logits = getattr(outputs, "logits", None)
        if not torch.is_tensor(logits) or logits.ndim != 3:
            raise TypeError("OPD local teacher must return 3-D logits.")
        if tuple(logits.shape[:2]) != (
            int(request.causal_position_mask.shape[0]),
            int(request.causal_position_mask.shape[1]) + 1,
        ):
            raise ValueError("OPD local teacher logits do not match score request positions.")
        flattened = logits[:, :-1, :][request.causal_position_mask.to(logits.device)]
        return request.objective_plan.build_teacher_distribution(flattened)


class HTTPRemoteOPDTeacherProvider(OPDTeacherProvider):
    """Immutable HTTP teacher using the versioned Shaft safetensors protocol."""

    name = "http"
    exact_resume_supported = True

    @classmethod
    def resolve_artifact_plan(
        cls,
        config: Any,
        *,
        checkpointing_requested: bool,
    ) -> Any:
        from .teacher_assembly import resolve_http_teacher_artifact_plan

        return resolve_http_teacher_artifact_plan(
            config,
            checkpointing_requested=checkpointing_requested,
        )

    def __init__(
        self,
        config: OPDTeacherConfig,
        *,
        model: torch.nn.Module | None,
        transport: Any | None = None,
    ) -> None:
        if model is not None:
            raise ValueError("HTTP OPD teacher provider must not receive a local model.")
        super().__init__(config, model=None)
        if transport is None:
            from .remote_teacher import UrllibOPDTeacherHTTPTransport

            transport = UrllibOPDTeacherHTTPTransport(config.remote)
        self.transport = transport
        self._identity = None

    def _resolve_identity(self):
        if self._identity is None:
            identity = self.transport.get_identity()
            expected_artifact = str(self.config.remote.artifact_fingerprint).strip().lower()
            if identity.artifact_fingerprint.strip().lower() != expected_artifact:
                raise ValueError(
                    "Remote OPD teacher artifact identity differs from configuration; "
                    f"actual={identity.artifact_fingerprint!r} expected={expected_artifact!r}."
                )
            if identity.model_type != str(self.config.model_type).strip().lower():
                raise ValueError(
                    "Remote OPD teacher model_type differs from configuration; "
                    f"actual={identity.model_type!r} expected={self.config.model_type!r}."
                )
            self._identity = identity
        return self._identity

    def validate_student_model(self, student_model: torch.nn.Module) -> None:
        _ = student_model

    def prepare(self, accelerator: Any) -> None:
        _ = accelerator
        self._resolve_identity()

    def validate_input_abi(self, input_abi: ShaftOPDInputABI) -> str:
        identity = self._resolve_identity()
        return validate_opd_input_abi_compatibility(
            student=input_abi,
            teacher=identity.input_abi,
        )

    def score(self, request: OPDTeacherScoreRequest) -> OPDTeacherDistribution:
        from .remote_teacher import (
            decode_teacher_distribution,
            encode_teacher_score_request,
            teacher_request_idempotency_key,
        )

        self._resolve_identity()
        payload = encode_teacher_score_request(request)
        response = self.transport.score(
            payload,
            idempotency_key=teacher_request_idempotency_key(payload),
        )
        if self.telemetry is not None:
            self.telemetry.record_teacher_transfer(
                request_bytes=len(payload),
                response_bytes=len(response),
            )
        distribution = decode_teacher_distribution(
            response,
            max_bytes=int(self.config.remote.max_response_bytes),
        )
        if distribution.num_positions != int(request.causal_position_mask.sum().item()):
            raise ValueError("Remote OPD teacher returned the wrong completion-position count.")
        request.objective_plan.validate_teacher_distribution(distribution)
        return distribution
