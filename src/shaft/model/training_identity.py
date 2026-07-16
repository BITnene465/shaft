from __future__ import annotations

import hashlib
import json
from typing import Any

from shaft.utils.semantic_identity import component_semantic_fingerprint


def _component_fingerprint(value: Any, *, role: str) -> str:
    """Delegate implementation, MRO, callable, state, and cycle semantics."""

    return component_semantic_fingerprint(
        value,
        role=f"model_training:{role}",
    )


def model_training_semantic_identity(model_adapter: Any) -> dict[str, Any]:
    """Return the model-owned identity that can alter training execution.

    The model layer owns only the selection of the active loader and policies.
    Encoding their implementations and semantic state is deliberately delegated
    to the shared semantic-identity utility, so model plans, input contracts,
    and resume contracts cannot drift into parallel callable/MRO algorithms.
    """

    model_meta = getattr(model_adapter, "model_meta", None)
    loader = None if model_meta is None else getattr(model_meta, "loader", None)
    if loader is None:
        raise ValueError("Resolved model adapter has no active model loader identity.")
    policies = {
        name: _component_fingerprint(
            getattr(model_adapter, name),
            role=name,
        )
        for name in (
            "module_groups",
            "processor_policy",
            "sequence_execution_policy",
            "peft_policy",
            "sharding_policy",
        )
    }
    return {
        "version": "shaft-model-training-semantic-identity-v2",
        "model_type": str(model_adapter.model_type),
        "family": str(model_adapter.family),
        "group_name": model_adapter.group_name,
        "template_type": str(model_adapter.template_type),
        "loader": _component_fingerprint(loader, role="loader"),
        "policies": policies,
    }


def model_training_semantic_fingerprint(model_adapter: Any) -> str:
    payload = model_training_semantic_identity(model_adapter)
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
