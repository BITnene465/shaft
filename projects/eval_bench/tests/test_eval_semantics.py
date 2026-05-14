from __future__ import annotations

import pytest

from eval_bench.eval_semantics import resolve_eval_semantics
from eval_bench.label_policy import resolve_target_label_policy
from eval_bench.metric_profiles import resolve_metric_profile


def test_eval_semantics_prefers_explicit_target_labels() -> None:
    semantics = resolve_eval_semantics(
        {
            "task": "detection",
            "metric_profile": "detection_iou_v1",
            "target_labels": ["arrow"],
            "prompt": {
                "prompt_id": "grounding_layout.latest",
                "metadata": {"target_labels": ["icon", "image", "shape"]},
            },
        }
    )

    assert semantics.target_labels == ["arrow"]
    assert semantics.target_labels_source == "explicit"
    assert semantics.metric_profile.profile_id == "detection_iou_v1"


def test_target_label_policy_records_legacy_prompt_id_source() -> None:
    policy = resolve_target_label_policy(prompt_id="grounding_layout.latest", task="detection")

    assert policy.labels == ["icon", "image", "shape"]
    assert policy.source == "legacy_prompt_id"


def test_metric_profile_default_follows_task() -> None:
    assert resolve_metric_profile("default", task="detection").profile_id == "detection_iou_v1"
    keypoint = resolve_metric_profile("default", task="keypoint")

    assert keypoint.profile_id == "keypoint_endpoint_v1"
    assert keypoint.matcher == "ordered_endpoint_distance"
    assert keypoint.endpoint_threshold_px == 20.0


def test_metric_profile_rejects_cross_task_profile() -> None:
    with pytest.raises(ValueError, match="is for task='detection'"):
        resolve_metric_profile("detection_iou_v1", task="keypoint")
