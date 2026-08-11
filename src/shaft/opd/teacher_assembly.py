from __future__ import annotations

from abc import ABC, abstractmethod
import copy
import hashlib
import json
from typing import Any, Callable

from shaft.config import FinetuneConfig, ModelConfig, RuntimeConfig
from shaft.model import (
    ModelArtifacts,
    materialize_resolved_model_artifact_identity,
    resolve_model_plan,
    validate_model_artifact_checkpointability,
    validate_resolved_model_descriptor,
)
from shaft.model.input_identity import tokenizer_artifact_fingerprint
from shaft.training.input_contract import (
    component_semantic_signature,
    input_component_semantic_signature,
)


class OPDTeacherArtifactPlan(ABC):
    """Provider-owned model/artifact assembly without provider branches in the pipeline."""

    fingerprint: str

    @abstractmethod
    def validate(self, *, save_strategy: str, resume_requested: bool) -> None:
        raise NotImplementedError

    @abstractmethod
    def build_sequence_contract(
        self,
        *,
        training_args: Any,
        device_type: str,
        distributed_strategy: str,
        torch_dtype: Any,
    ) -> Any | None:
        raise NotImplementedError

    @abstractmethod
    def build_artifacts(self, builder: Callable[..., ModelArtifacts]) -> ModelArtifacts | None:
        raise NotImplementedError

    @abstractmethod
    def provider_model(self, artifacts: ModelArtifacts | None) -> Any | None:
        raise NotImplementedError

    @abstractmethod
    def configure_and_validate(
        self,
        *,
        student: ModelArtifacts,
        teacher: ModelArtifacts | None,
        provider: Any,
        sequence_contract: Any | None,
    ) -> str:
        raise NotImplementedError


def _teacher_runtime_config(config: RuntimeConfig) -> RuntimeConfig:
    teacher = config.opd.teacher
    resolved = copy.deepcopy(config)
    resolved.model = ModelConfig(
        model_type=teacher.model_type,
        model_name_or_path=teacher.model_name_or_path,
        revision=teacher.revision,
        cache_dir=teacher.cache_dir,
        local_files_only=teacher.local_files_only,
        template=teacher.template,
        trust_remote_code=teacher.trust_remote_code,
        attn_implementation=teacher.attn_implementation,
        torch_dtype=teacher.torch_dtype,
        device_map=None,
        finetune=FinetuneConfig(mode="full"),
    )
    resolved.train.gradient_checkpointing = False
    resolved.train.init_from_checkpoint = None
    resolved.train.resume_from_checkpoint = None
    return resolved


def model_artifact_input_identity(student: ModelArtifacts) -> tuple[str, str, str]:
    model_type = str(student.model_adapter.model_type)
    tokenizer = tokenizer_artifact_fingerprint(student.tokenizer)
    payload = {
        "processor": input_component_semantic_signature(student.processor, role="processor"),
        "processor_policy": component_semantic_signature(
            student.model_adapter.processor_policy,
            role="opd_processor_policy",
        ),
        "template": input_component_semantic_signature(student.template, role="template"),
    }
    processor = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return model_type, tokenizer, processor


class LocalHFOPDTeacherArtifactPlan(OPDTeacherArtifactPlan):
    def __init__(self, config: RuntimeConfig, *, checkpointing_requested: bool) -> None:
        self.runtime_config = _teacher_runtime_config(config)
        model_plan = resolve_model_plan(self.runtime_config, require_immutable_artifact=False)
        if checkpointing_requested:
            model_plan = materialize_resolved_model_artifact_identity(model_plan)
        self.model_plan = model_plan
        self.fingerprint = model_plan.fingerprint
        self._sequence_contract = None

    def validate(self, *, save_strategy: str, resume_requested: bool) -> None:
        validate_resolved_model_descriptor(self.model_plan)
        self.model_plan.model_adapter.validate_distributed_config(
            self.runtime_config.train,
            finetune=self.runtime_config.model.finetune,
        )
        validate_model_artifact_checkpointability(
            self.model_plan,
            save_strategy=save_strategy,
            resume_requested=resume_requested,
        )

    def build_sequence_contract(
        self,
        *,
        training_args: Any,
        device_type: str,
        distributed_strategy: str,
        torch_dtype: Any,
    ) -> Any:
        self._sequence_contract = self.model_plan.build_sequence_execution_contract(
            layout="padded",
            device_type=device_type,
            attention_implementation=self.runtime_config.model.attn_implementation,
            torch_dtype=torch_dtype,
            distributed_strategy=distributed_strategy,
            torch_compile=bool(getattr(training_args, "torch_compile", False)),
        )
        return self._sequence_contract

    def build_artifacts(self, builder: Callable[..., ModelArtifacts]) -> ModelArtifacts:
        if self._sequence_contract is None:
            raise RuntimeError("Local OPD teacher sequence contract was not resolved.")
        return builder(
            self.runtime_config,
            self.model_plan,
            self._sequence_contract,
            role="teacher",
            init_from_checkpoint=None,
        )

    def provider_model(self, artifacts: ModelArtifacts | None) -> Any:
        if artifacts is None:
            raise RuntimeError("Local OPD teacher artifacts were not built.")
        return artifacts.model

    def configure_and_validate(
        self,
        *,
        student: ModelArtifacts,
        teacher: ModelArtifacts | None,
        provider: Any,
        sequence_contract: Any | None,
    ) -> str:
        if teacher is None or sequence_contract is None:
            raise RuntimeError("Local OPD teacher assembly is incomplete.")
        teacher.model_adapter.configure_sequence_execution(
            model=teacher.model,
            contract=sequence_contract,
        )
        teacher.model_adapter.validate_sequence_execution(
            model=teacher.model,
            contract=sequence_contract,
        )
        student_identity = model_artifact_input_identity(student)
        teacher_identity = model_artifact_input_identity(teacher)
        if student_identity != teacher_identity:
            raise ValueError(
                "OPD local teacher/student input identities differ; "
                f"student={student_identity} teacher={teacher_identity}."
            )
        return provider.validate_input_identity(
            model_type=student_identity[0],
            tokenizer_fingerprint=student_identity[1],
            processor_fingerprint=student_identity[2],
        )


class HTTPRemoteOPDTeacherArtifactPlan(OPDTeacherArtifactPlan):
    def __init__(self, config: RuntimeConfig, *, checkpointing_requested: bool) -> None:
        _ = checkpointing_requested
        self.fingerprint = str(config.opd.teacher.remote.artifact_fingerprint).strip().lower()

    def validate(self, *, save_strategy: str, resume_requested: bool) -> None:
        _ = save_strategy, resume_requested
        if not self.fingerprint:
            raise ValueError("Remote OPD teacher artifact fingerprint must not be empty.")

    def build_sequence_contract(
        self,
        *,
        training_args: Any,
        device_type: str,
        distributed_strategy: str,
        torch_dtype: Any,
    ) -> None:
        _ = training_args, device_type, distributed_strategy, torch_dtype
        return None

    def build_artifacts(self, builder: Callable[..., ModelArtifacts]) -> None:
        _ = builder
        return None

    def provider_model(self, artifacts: ModelArtifacts | None) -> None:
        if artifacts is not None:
            raise RuntimeError("Remote OPD teacher must not build local artifacts.")
        return None

    def configure_and_validate(
        self,
        *,
        student: ModelArtifacts,
        teacher: ModelArtifacts | None,
        provider: Any,
        sequence_contract: Any | None,
    ) -> str:
        if teacher is not None or sequence_contract is not None:
            raise RuntimeError("Remote OPD teacher unexpectedly owns local model state.")
        model_type, tokenizer, processor = model_artifact_input_identity(student)
        return provider.validate_input_identity(
            model_type=model_type,
            tokenizer_fingerprint=tokenizer,
            processor_fingerprint=processor,
        )


def resolve_local_teacher_artifact_plan(
    config: RuntimeConfig,
    *,
    checkpointing_requested: bool,
) -> OPDTeacherArtifactPlan:
    return LocalHFOPDTeacherArtifactPlan(
        config,
        checkpointing_requested=checkpointing_requested,
    )


def resolve_http_teacher_artifact_plan(
    config: RuntimeConfig,
    *,
    checkpointing_requested: bool,
) -> OPDTeacherArtifactPlan:
    return HTTPRemoteOPDTeacherArtifactPlan(
        config,
        checkpointing_requested=checkpointing_requested,
    )
