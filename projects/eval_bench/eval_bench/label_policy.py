from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


LAYOUT_TARGET_LABELS = ("icon", "image", "shape")
ARROW_TARGET_LABELS = ("arrow",)
KEYPOINT_TARGET_LABELS = frozenset(ARROW_TARGET_LABELS)
TARGET_LABEL_SOURCES = frozenset(
    {"explicit", "prompt_metadata", "suite_default", "task_default", "unscoped"}
)


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
    if "grounding_layout" in lower_prompt_id:
        return TargetLabelPolicy(labels=list(LAYOUT_TARGET_LABELS), source="suite_default")
    if "grounding_icon_image" in lower_prompt_id:
        return TargetLabelPolicy(labels=["icon", "image"], source="suite_default")
    if "grounding_shape" in lower_prompt_id:
        return TargetLabelPolicy(labels=["shape"], source="suite_default")
    if (
        "point_arrow" in lower_prompt_id
        or "keypoint_arrow" in lower_prompt_id
        or "arrow_keypoint" in lower_prompt_id
        or "grounding_arrow" in lower_prompt_id
    ):
        return TargetLabelPolicy(labels=list(ARROW_TARGET_LABELS), source="suite_default")
    if task == "keypoint":
        return TargetLabelPolicy(labels=list(ARROW_TARGET_LABELS), source="task_default")
    return TargetLabelPolicy(labels=[], source="unscoped")


def target_label_task_errors(*, task: str | None, labels: list[str]) -> list[str]:
    if task != "keypoint" or not labels:
        return []
    unsupported = [label for label in labels if label not in KEYPOINT_TARGET_LABELS]
    if not unsupported:
        return []
    return [
        "keypoint target_labels only support arrow; "
        f"unsupported labels: {', '.join(unsupported)}"
    ]


def validate_target_labels_for_task(*, task: str | None, labels: list[str]) -> None:
    errors = target_label_task_errors(task=task, labels=labels)
    if errors:
        raise ValueError(errors[0])


def target_label_benchmark_messages(
    *,
    labels: Any,
    benchmark_labels: Any,
    benchmark_id: str | None = None,
    missing_index_action: str = "preflight-validated",
) -> tuple[list[str], list[str]]:
    target_labels = normalize_target_labels(labels)
    if not target_labels:
        return [], []
    available_labels = normalize_target_labels(benchmark_labels)
    if not available_labels:
        if not benchmark_id:
            return [], []
        return [], [
            f"benchmark {benchmark_id} has no label index; "
            f"target_labels could not be {missing_index_action}."
        ]
    available = set(available_labels)
    missing = [label for label in target_labels if label not in available]
    if not missing:
        return [], []
    return [
        "target_labels not found in benchmark label index: "
        f"{', '.join(missing)}. Available labels: {', '.join(available_labels)}"
    ], []


def validate_target_labels_for_benchmark(
    *,
    labels: Any,
    benchmark_labels: Any,
    benchmark_id: str | None = None,
) -> None:
    errors, _warnings = target_label_benchmark_messages(
        labels=labels,
        benchmark_labels=benchmark_labels,
        benchmark_id=benchmark_id,
    )
    if errors:
        raise ValueError(errors[0])


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


def target_label_resolution_payload(
    *,
    task: str | None = None,
    prompt_id: str | None = None,
    explicit: Any = None,
    prompt_metadata: Mapping[str, Any] | None = None,
    benchmark_id: str | None = None,
    benchmark_labels: Any = None,
) -> dict[str, Any]:
    prompt_labels = normalize_target_labels((prompt_metadata or {}).get("target_labels"))
    explicit_labels = normalize_target_labels(explicit)
    available_labels = normalize_target_labels(benchmark_labels)
    policy = resolve_target_label_policy(
        explicit=explicit_labels,
        prompt_id=prompt_id,
        task=task,
        prompt_metadata=prompt_metadata,
    )
    candidate_labels = _unique_labels(
        [
            *available_labels,
            *prompt_labels,
            *explicit_labels,
            *policy.labels,
        ]
    )
    errors: list[str] = []
    warnings: list[str] = []
    errors.extend(target_label_task_errors(task=task, labels=policy.labels))
    benchmark_errors, benchmark_warnings = target_label_benchmark_messages(
        labels=policy.labels,
        benchmark_labels=available_labels,
        benchmark_id=benchmark_id,
        missing_index_action="validated",
    )
    errors.extend(benchmark_errors)
    warnings.extend(benchmark_warnings)
    return {
        "task": task or "",
        "benchmark_id": benchmark_id or "",
        "prompt_id": prompt_id or "",
        "target_labels": policy.labels,
        "target_labels_source": policy.source,
        "candidate_labels": candidate_labels,
        "benchmark_labels": available_labels,
        "prompt_target_labels": prompt_labels,
        "explicit_target_labels": explicit_labels,
        "label_subtasks_supported": task == "detection",
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
    }


def _unique_labels(labels: list[str]) -> list[str]:
    values: list[str] = []
    for label in labels:
        if label and label not in values:
            values.append(label)
    return values
