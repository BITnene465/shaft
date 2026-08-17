from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import shaft.opd.loss as loss_module
import shaft.opd.rollout as rollout_module
import shaft.opd.teacher as teacher_module
from shaft.training.input_contract import callable_semantic_signature
from shaft.training.resume_contract import (
    canonical_training_resume_value,
    register_training_resume_policy,
    training_module_implementation_signature,
)

from .trainer import ShaftOPDTrainer
from .execution import resolve_opd_execution_plan
from .loss import resolve_opd_objective_plan


def _require_context(context: Mapping[str, Any], key: str) -> Any:
    value = context.get(key)
    if value is None:
        raise ValueError(f"OPD training resume contract requires {key}.")
    return value


def _resume_objective(
    config: Any,
    training_args: Any,
    context: Mapping[str, Any],
) -> dict[str, Any]:
    _ = training_args
    return {
        "teacher_model_plan_fingerprint": str(
            _require_context(context, "teacher_model_plan_fingerprint")
        ),
        "teacher_student_input_abi_fingerprint": str(
            _require_context(context, "teacher_student_input_abi_fingerprint")
        ),
        "rollout": canonical_training_resume_value(config.opd.rollout),
        "objective": canonical_training_resume_value(config.opd.objective),
    }


def _resume_implementation(config: Any) -> dict[str, Any]:
    execution_plan = resolve_opd_execution_plan(config.opd)
    objective_plan = resolve_opd_objective_plan(config.opd.objective)
    return {
        "algorithm_impl": ShaftOPDTrainer,
        "trainer_impl": ShaftOPDTrainer,
        "objective_impl": {
            "objective_mode": objective_plan.mode,
            "objective_compute": callable_semantic_signature(
                objective_plan.backend_type.compute,
                role=f"opd_objective:{objective_plan.mode}",
            ),
            "teacher_projection": callable_semantic_signature(
                objective_plan.backend_type.build_teacher_distribution,
                role=f"opd_teacher_projection:{objective_plan.mode}",
            ),
            "loss_policy": training_module_implementation_signature(loss_module),
            "rollout_backend": {
                "name": execution_plan.rollout_backend_name,
                "generate": callable_semantic_signature(
                    execution_plan.rollout_backend_type.generate,
                    role=(f"opd_rollout:{execution_plan.rollout_backend_name}"),
                ),
                "policy": training_module_implementation_signature(rollout_module),
            },
            "teacher_provider": {
                "name": execution_plan.teacher_provider_name,
                "score": callable_semantic_signature(
                    execution_plan.teacher_provider_type.score,
                    role=(f"opd_teacher:{execution_plan.teacher_provider_name}"),
                ),
                "policy": training_module_implementation_signature(teacher_module),
            },
        },
        "package_names": (),
    }


register_training_resume_policy(
    "opd",
    objective_builder=_resume_objective,
    implementation_builder=_resume_implementation,
)
