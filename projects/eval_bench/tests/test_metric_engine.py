from __future__ import annotations

from pathlib import Path

from eval_bench.metric_profiles import resolve_metric_profile
from eval_bench.metrics.engine import MetricSample, evaluate_metric_samples


def test_detection_iou_maxmatch_v2_can_recover_non_greedy_assignment() -> None:
    sample = MetricSample(
        sample_index=0,
        json_relative=Path("part1/json/a.json"),
        image=Path("part1/images/a.png"),
        gt_instances=[
            {"label": "shape", "bbox": [0, 0, 100, 100]},
            {"label": "shape", "bbox": [40, 0, 140, 100]},
        ],
        pred_instances=[
            {"label": "shape", "bbox": [10, 0, 110, 100]},
            {"label": "shape", "bbox": [-20, 0, 80, 100]},
        ],
        has_prediction=True,
    )

    greedy = evaluate_metric_samples(
        [sample],
        profile=resolve_metric_profile("detection_iou_v1", task="detection"),
    )
    maxmatch = evaluate_metric_samples(
        [sample],
        profile=resolve_metric_profile("detection_iou_maxmatch_v2", task="detection"),
    )

    assert greedy.matched_count == 1
    assert greedy.recall_iou50 == 0.5
    assert maxmatch.matched_count == 2
    assert maxmatch.precision_iou50 == 1.0
    assert maxmatch.recall_iou50 == 1.0
    assert maxmatch.samples[0]["false_negative_count"] == 0
    assert maxmatch.samples[0]["false_positive_count"] == 0
