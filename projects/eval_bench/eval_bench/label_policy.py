from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


LAYOUT_TARGET_LABELS = ("icon", "image", "shape")
ARROW_TARGET_LABELS = ("arrow",)


@dataclass(frozen=True)
class TargetLabelPolicy:
    labels: list[str]
    source: str


def normalize_target_labels(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        values = value.replace(",", " ").split()
    elif isinstance(value, list | tuple):
        values = [str(item) for item in value]
    else:
        raise ValueError("target_labels must be a list or a comma/space separated string.")
    labels: list[str] = []
    for item in values:
        label = str(item).strip()
        if label and label not in labels:
            labels.append(label)
    return labels


def resolve_target_label_policy(
    *,
    explicit: Any = None,
    prompt_id: str | None = None,
    task: str | None = None,
    prompt_metadata: Mapping[str, Any] | None = None,
) -> TargetLabelPolicy:
    labels = normalize_target_labels(explicit)
    if labels:
        return TargetLabelPolicy(labels=labels, source="explicit")
    metadata = dict(prompt_metadata or {})
    labels = normalize_target_labels(metadata.get("target_labels"))
    if labels:
        return TargetLabelPolicy(labels=labels, source="prompt_metadata")
    lower_prompt_id = (prompt_id or "").lower()
    if "layout" in lower_prompt_id:
        return TargetLabelPolicy(labels=list(LAYOUT_TARGET_LABELS), source="legacy_prompt_id")
    if "keypoint" in lower_prompt_id or "arrow" in lower_prompt_id:
        return TargetLabelPolicy(labels=list(ARROW_TARGET_LABELS), source="legacy_prompt_id")
    if task == "keypoint":
        return TargetLabelPolicy(labels=list(ARROW_TARGET_LABELS), source="task_default")
    return TargetLabelPolicy(labels=[], source="unscoped")


def resolve_target_labels(
    *,
    explicit: Any = None,
    prompt_id: str | None = None,
    task: str | None = None,
    prompt_metadata: Mapping[str, Any] | None = None,
) -> list[str]:
    return resolve_target_label_policy(
        explicit=explicit,
        prompt_id=prompt_id,
        task=task,
        prompt_metadata=prompt_metadata,
    ).labels
