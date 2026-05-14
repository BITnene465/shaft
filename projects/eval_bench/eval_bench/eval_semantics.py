from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .label_policy import resolve_target_label_policy
from .metric_profiles import MetricProfile, resolve_metric_profile


@dataclass(frozen=True)
class EvalSemantics:
    task: str
    metric_profile: MetricProfile
    target_labels: list[str]
    target_labels_source: str


def resolve_eval_semantics(spec: Mapping[str, Any]) -> EvalSemantics:
    task = str(spec.get("task") or "")
    prompt = spec.get("prompt")
    if not isinstance(prompt, Mapping):
        prompt = {}
    prompt_metadata = prompt.get("metadata")
    if not isinstance(prompt_metadata, Mapping):
        prompt_metadata = {}

    target_policy = resolve_target_label_policy(
        explicit=spec.get("target_labels"),
        prompt_id=str(prompt.get("prompt_id") or ""),
        task=task,
        prompt_metadata=prompt_metadata,
    )
    metric_profile = resolve_metric_profile(str(spec.get("metric_profile") or ""), task=task)
    return EvalSemantics(
        task=task,
        metric_profile=metric_profile,
        target_labels=target_policy.labels,
        target_labels_source=target_policy.source,
    )
