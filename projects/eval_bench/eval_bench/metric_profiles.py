from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricProfile:
    profile_id: str
    task: str
    matcher: str
    primary_score: str
    description: str
    iou_threshold: float = 0.5
    endpoint_threshold_px: float | None = None


METRIC_PROFILES: dict[str, MetricProfile] = {
    "detection_iou_v1": MetricProfile(
        profile_id="detection_iou_v1",
        task="detection",
        matcher="bbox_iou",
        primary_score="precision_recall_iou50",
        description="Box detection profile using per-label greedy IoU matching.",
    ),
    "keypoint_endpoint_v1": MetricProfile(
        profile_id="keypoint_endpoint_v1",
        task="keypoint",
        matcher="ordered_endpoint_distance",
        primary_score="mean_keypoint_distance",
        endpoint_threshold_px=20.0,
        description="Arrow endpoint profile using ordered start/end point distance matching.",
    ),
}

DEFAULT_METRIC_PROFILE_BY_TASK = {
    "detection": "detection_iou_v1",
    "keypoint": "keypoint_endpoint_v1",
}


def resolve_metric_profile(profile_id: str | None, *, task: str) -> MetricProfile:
    normalized = str(profile_id or "").strip()
    if normalized in {"", "default"}:
        normalized = DEFAULT_METRIC_PROFILE_BY_TASK.get(task, "")
    profile = METRIC_PROFILES.get(normalized)
    if profile is None:
        raise ValueError(f"unsupported metric_profile: {profile_id!r}")
    if profile.task != task:
        raise ValueError(
            f"metric_profile={profile.profile_id!r} is for task={profile.task!r}, "
            f"but eval spec task is {task!r}."
        )
    return profile
