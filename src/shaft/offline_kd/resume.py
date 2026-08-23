from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import shaft.training.distribution_loss as distribution_loss_module
from shaft.algorithms import offline_kd as _algorithm_module  # noqa: F401
from shaft.algorithms.registry import ALGORITHM_REGISTRY
from shaft.training.input_contract import callable_semantic_signature
from shaft.training.resume_contract import (
    canonical_training_resume_value,
    register_training_resume_policy,
    training_module_implementation_signature,
)

from .trainer import ShaftOfflineKDTrainer


def _resume_objective(
    config: Any, training_args: Any, context: Mapping[str, Any]
) -> dict[str, Any]:
    _ = training_args, context
    return {
        "offline_kd": canonical_training_resume_value(config.offline_kd),
    }


def _resume_implementation(config: Any) -> dict[str, Any]:
    mode = str(config.offline_kd.objective.mode).strip().lower()
    plan = distribution_loss_module.resolve_distribution_objective_plan(
        config.offline_kd.objective
    )
    backend = plan.backend_type
    return {
        "algorithm_impl": ALGORITHM_REGISTRY.get("offline_kd"),
        "trainer_impl": ShaftOfflineKDTrainer,
        "objective_impl": {
            "objective_compute": callable_semantic_signature(
                backend.compute,
                role=f"offline_kd_objective:{mode}",
            ),
            "loss_policy": training_module_implementation_signature(
                distribution_loss_module
            ),
        },
        "package_names": ("safetensors",),
    }


register_training_resume_policy(
    "offline_kd",
    objective_builder=_resume_objective,
    implementation_builder=_resume_implementation,
)
