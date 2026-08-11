from __future__ import annotations

from dataclasses import dataclass

import torch

from shaft.config import OPDConfig, OPDRolloutConfig, OPDTeacherConfig
from shaft.plugins import Registry

from .rollout import HFLocalOPDRolloutBackend, OPDRolloutBackend, VLLMOPDRolloutBackend
from .teacher import (
    HTTPRemoteOPDTeacherProvider,
    LocalHFOPDTeacherProvider,
    OPDTeacherProvider,
)


@dataclass(frozen=True, slots=True)
class OPDExecutionPlan:
    """Resolved OPD execution implementations before heavyweight model loading."""

    rollout_backend_name: str
    rollout_backend_type: type[OPDRolloutBackend]
    teacher_provider_name: str
    teacher_provider_type: type[OPDTeacherProvider]

    def validate_checkpointing(self, *, checkpointing_requested: bool) -> None:
        if not checkpointing_requested:
            return
        for name, implementation in (
            (self.rollout_backend_name, self.rollout_backend_type),
            (self.teacher_provider_name, self.teacher_provider_type),
        ):
            if type(implementation.exact_resume_supported) is not bool:
                raise TypeError(
                    f"OPD execution component {name!r} must declare "
                    "exact_resume_supported as a boolean."
                )
        unsupported = [
            name
            for name, implementation in (
                (self.rollout_backend_name, self.rollout_backend_type),
                (self.teacher_provider_name, self.teacher_provider_type),
            )
            if implementation.exact_resume_supported is False
        ]
        if unsupported:
            raise ValueError(
                "OPD execution component does not support exact resume: "
                f"{sorted(unsupported)}. Disable checkpointing or select an exact-resumable "
                "implementation."
            )


@dataclass(frozen=True, slots=True)
class OPDExecutionRuntime:
    rollout_backend: OPDRolloutBackend
    teacher_provider: OPDTeacherProvider


class OPDExecutionRegistry:
    """Own the two independently extensible OPD runtime axes."""

    def __init__(self) -> None:
        self._rollout_backends: Registry[type[OPDRolloutBackend]] = Registry(
            "opd_rollout_backend"
        )
        self._teacher_providers: Registry[type[OPDTeacherProvider]] = Registry(
            "opd_teacher_provider"
        )

    def register_rollout_backend(
        self,
        name: str,
        implementation: type[OPDRolloutBackend],
    ) -> type[OPDRolloutBackend]:
        if not issubclass(implementation, OPDRolloutBackend):
            raise TypeError("OPD rollout implementations must subclass OPDRolloutBackend.")
        self._validate_implementation_name(name, implementation, role="rollout backend")
        return self._rollout_backends.register(name, implementation)

    def register_teacher_provider(
        self,
        name: str,
        implementation: type[OPDTeacherProvider],
    ) -> type[OPDTeacherProvider]:
        if not issubclass(implementation, OPDTeacherProvider):
            raise TypeError("OPD teacher implementations must subclass OPDTeacherProvider.")
        self._validate_implementation_name(name, implementation, role="teacher provider")
        return self._teacher_providers.register(name, implementation)

    @staticmethod
    def _validate_implementation_name(
        name: str,
        implementation: type[OPDRolloutBackend] | type[OPDTeacherProvider],
        *,
        role: str,
    ) -> None:
        normalized = str(name).strip().lower()
        declared = str(implementation.name).strip().lower()
        if not normalized or declared != normalized:
            raise ValueError(
                f"OPD {role} registration name {normalized!r} does not match "
                f"implementation name {declared!r}."
            )
        if type(implementation.exact_resume_supported) is not bool:
            raise TypeError(
                f"OPD {role} {normalized!r} must declare exact_resume_supported "
                "as a boolean."
            )

    def resolve(self, config: OPDConfig) -> OPDExecutionPlan:
        rollout_name = str(config.rollout.backend).strip().lower()
        teacher_name = str(config.teacher.provider).strip().lower()
        try:
            rollout_type = self._rollout_backends.get(rollout_name)
        except KeyError as exc:
            raise ValueError(
                f"Unknown OPD rollout backend {rollout_name!r}; registered="
                f"{self._rollout_backends.keys()}."
            ) from exc
        try:
            teacher_type = self._teacher_providers.get(teacher_name)
        except KeyError as exc:
            raise ValueError(
                f"Unknown OPD teacher provider {teacher_name!r}; registered="
                f"{self._teacher_providers.keys()}."
            ) from exc
        return OPDExecutionPlan(
            rollout_backend_name=rollout_name,
            rollout_backend_type=rollout_type,
            teacher_provider_name=teacher_name,
            teacher_provider_type=teacher_type,
        )


OPD_EXECUTION_REGISTRY = OPDExecutionRegistry()
OPD_EXECUTION_REGISTRY.register_rollout_backend("hf_local", HFLocalOPDRolloutBackend)
OPD_EXECUTION_REGISTRY.register_rollout_backend("vllm", VLLMOPDRolloutBackend)
OPD_EXECUTION_REGISTRY.register_teacher_provider("hf_local", LocalHFOPDTeacherProvider)
OPD_EXECUTION_REGISTRY.register_teacher_provider("http", HTTPRemoteOPDTeacherProvider)


def resolve_opd_execution_plan(config: OPDConfig) -> OPDExecutionPlan:
    return OPD_EXECUTION_REGISTRY.resolve(config)


def build_opd_execution_runtime(
    plan: OPDExecutionPlan,
    *,
    rollout_config: OPDRolloutConfig,
    teacher_config: OPDTeacherConfig,
    teacher_model: torch.nn.Module | None,
    seed: int = 0,
) -> OPDExecutionRuntime:
    rollout_backend = plan.rollout_backend_type(rollout_config, seed=seed)
    teacher_provider = plan.teacher_provider_type(
        teacher_config,
        model=teacher_model,
    )
    if rollout_backend.name != plan.rollout_backend_name:
        raise ValueError("Resolved OPD rollout backend published a mismatched name.")
    if teacher_provider.name != plan.teacher_provider_name:
        raise ValueError("Resolved OPD teacher provider published a mismatched name.")
    return OPDExecutionRuntime(
        rollout_backend=rollout_backend,
        teacher_provider=teacher_provider,
    )
