from __future__ import annotations

import pytest

from shaft.codec import ShaftCodecResult
from shaft.metrics import build_eval_metric


def test_parse_metrics_distinguish_complete_and_partial_json() -> None:
    parse_success = build_eval_metric("parse_success")
    parse_partial_rate = build_eval_metric("parse_partial_rate")

    complete = ShaftCodecResult(
        raw_text='{"ok": 1}',
        parsed={"ok": 1},
        valid=True,
        partial=False,
        error_type=None,
        error=None,
    )
    partial = ShaftCodecResult(
        raw_text='{"ok": 1',
        parsed={"ok": 1},
        valid=True,
        partial=True,
        error_type=None,
        error=None,
    )
    invalid = ShaftCodecResult(
        raw_text="oops",
        parsed=None,
        valid=False,
        partial=False,
        error_type="json_decode_error",
        error="bad json",
    )

    for prediction in (complete, partial, invalid):
        parse_success.update(prediction=prediction, target=None, sample_meta={})
        parse_partial_rate.update(prediction=prediction, target=None, sample_meta={})

    assert parse_success.compute() == pytest.approx(1.0 / 3.0)
    assert parse_partial_rate.compute() == pytest.approx(1.0 / 3.0)


def test_keypoint_pck_uses_normalized_coordinate_scale_by_default() -> None:
    metric = build_eval_metric("keypoint_pck")
    prediction = ShaftCodecResult(
        raw_text="",
        parsed={
            "keypoints_2d": [[530, 500], [500, 470]],
            "stroke_pattern": "solid",
            "geometry_style": "straight",
        },
        valid=True,
        partial=False,
        error_type=None,
        error=None,
    )
    target = {
        "keypoints_2d": [[500, 500], [500, 500]],
        "stroke_pattern": "solid",
        "geometry_style": "straight",
    }

    metric.update(
        prediction=prediction,
        target=target,
        sample_meta={"extra": {"image_width": 100, "image_height": 100}},
    )

    assert metric.compute() == pytest.approx(1.0)


def test_keypoint_pck_accepts_points_2d_alias() -> None:
    metric = build_eval_metric("keypoint_pck", params={"coordinate_space": "points_2d"})
    prediction = ShaftCodecResult(
        raw_text="",
        parsed={"label": "line", "points_2d": [[228, 492], [810, 492]]},
        valid=True,
        partial=False,
        error_type=None,
        error=None,
    )
    target = {"label": "line", "points_2d": [[228, 492], [810, 492]]}

    metric.update(prediction=prediction, target=target, sample_meta={})

    assert metric.compute() == pytest.approx(1.0)


def test_keypoint_pck_accepts_line_parameters_points_segments() -> None:
    metric = build_eval_metric("keypoint_pck", params={"coordinate_space": "points"})
    prediction = ShaftCodecResult(
        raw_text="",
        parsed={
            "type": "line",
            "parameters": {
                "is_single": False,
                "points": [
                    [[100, 100], [500, 100]],
                    [[500, 100], [900, 300]],
                ],
            },
        },
        valid=True,
        partial=False,
        error_type=None,
        error=None,
    )
    target = {
        "type": "line",
        "parameters": {
            "is_single": False,
            "points": [
                [[100, 100], [500, 100]],
                [[500, 100], [900, 300]],
            ],
        },
    }

    metric.update(prediction=prediction, target=target, sample_meta={})

    assert metric.compute() == pytest.approx(1.0)
