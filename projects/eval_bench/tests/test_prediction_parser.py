from __future__ import annotations

import json

import pytest

from eval_bench.prediction_parser import parse_prediction_text


def test_detection_parser_normalizes_shape_combination_and_dedupes() -> None:
    doc = parse_prediction_text(
        text=json.dumps(
            {
                "detections": [
                    {"label": "shape combination", "bbox_2d": [100, 100, 300, 300]},
                    {"label": "shape_combination", "bbox_2d": [101, 101, 301, 301]},
                    {"label": "image", "bbox_2d": [500, 500, 750, 750]},
                    {"label": "unknown", "bbox_2d": [0, 0, 10, 10]},
                ]
            }
        ),
        task="detection",
        image="part2/images/a.png",
        image_width=2000,
        image_height=1000,
        metadata={"model_id": "unit"},
    )

    assert doc.image == "part2/images/a.png"
    assert [instance.label for instance in doc.instances] == ["icon", "image"]
    assert doc.instances[0].bbox == pytest.approx([200.1001, 100.0, 600.3003, 300.0])
    assert doc.instances[1].bbox == pytest.approx([1000.5005, 500.0, 1500.7508, 750.0])


def test_detection_parser_normalizes_arrow_subclasses_to_arrow() -> None:
    doc = parse_prediction_text(
        text=json.dumps(
            [
                {"label": "single_arrow", "bbox_2d": [100, 100, 300, 300]},
                {"label": "double arrow", "bbox_2d": [400, 100, 600, 300]},
            ]
        ),
        task="detection",
        image="part1/images/arrow.png",
        image_width=1000,
        image_height=500,
        metadata={"model_id": "unit"},
    )

    assert [instance.label for instance in doc.instances] == ["arrow", "arrow"]
    assert doc.instances[0].extra["source_label_before_normalization"] == "single_arrow"
    assert doc.instances[1].extra["source_label_before_normalization"] == "double_arrow"


def test_detection_parser_accepts_grouped_bbox_payload_and_returns_instances() -> None:
    doc = parse_prediction_text(
        text=json.dumps(
            {
                "shape": [[100, 100, 300, 300]],
                "icon": [],
                "image": [[500, 500, 750, 750]],
                "line": [[0, 100, 1000, 120]],
            }
        ),
        task="detection",
        image="part3/images/grouped.png",
        image_width=2000,
        image_height=1000,
        metadata={"model_id": "unit"},
    )

    assert [instance.label for instance in doc.instances] == ["shape", "image", "line"]
    assert doc.instances[0].bbox == pytest.approx([200.1001, 100.0, 600.3003, 300.0])
    assert doc.instances[1].bbox == pytest.approx([1000.5005, 500.0, 1500.7508, 750.0])
    assert doc.instances[2].bbox == pytest.approx([0.0, 100.0, 1999.0, 120.0])
    assert doc.to_dict(task="detection")["instances"][0]["label"] == "shape"


def test_keypoint_parser_accepts_top_level_keypoints_and_clips_bbox() -> None:
    doc = parse_prediction_text(
        text=json.dumps({"keypoints_2d": [[0, 0], [1000, 1000]]}),
        task="keypoint",
        image="part1/images/arrow.png",
        image_width=1200,
        image_height=800,
        metadata={"model_id": "unit"},
    )

    assert len(doc.instances) == 1
    instance = doc.instances[0]
    assert instance.label == "arrow"
    assert instance.keypoints[0] == pytest.approx([0.0, 0.0])
    assert instance.keypoints[1] == pytest.approx([1199.0, 799.0])
    assert instance.bbox == pytest.approx([0.0, 0.0, 1199.0, 799.0])


def test_keypoint_parser_accepts_top_level_points_2d_for_line() -> None:
    doc = parse_prediction_text(
        text=json.dumps({"label": "line", "points_2d": [[228, 492], [810, 492]]}),
        task="keypoint",
        image="part1/images/line.png",
        image_width=1000,
        image_height=500,
        metadata={"model_id": "unit"},
    )

    assert len(doc.instances) == 1
    instance = doc.instances[0]
    assert instance.label == "line"
    assert instance.keypoints[0] == pytest.approx([228.0, 245.7538])
    assert instance.keypoints[1] == pytest.approx([810.0, 245.7538])
    assert instance.bbox == pytest.approx([228.0, 244.7538, 810.0, 246.7538])


def test_keypoint_parser_accepts_line_parameters_points_segments() -> None:
    doc = parse_prediction_text(
        text=json.dumps(
            {
                "type": "line",
                "parameters": {
                    "is_single": False,
                    "points": [
                        [[100, 100], [500, 100]],
                        [[500, 100], [900, 300]],
                    ],
                },
            }
        ),
        task="keypoint",
        image="part1/images/line.png",
        image_width=1000,
        image_height=500,
        metadata={"model_id": "unit"},
    )

    assert len(doc.instances) == 1
    instance = doc.instances[0]
    assert instance.label == "line"
    assert len(instance.keypoints) == 4
    assert instance.keypoints[0] == pytest.approx([100.0, 49.9499499])
    assert instance.keypoints[-1] == pytest.approx([900.0, 149.8498498])


def test_keypoint_parser_derives_valid_bbox_for_single_edge_point() -> None:
    doc = parse_prediction_text(
        text=json.dumps({"instances": [{"label": "arrow", "linestrip": [[0, 0]]}]}),
        task="keypoint",
        image="part1/images/arrow.png",
        image_width=1200,
        image_height=800,
        metadata={"model_id": "unit"},
    )

    assert len(doc.instances) == 1
    assert doc.instances[0].bbox == [0.0, 0.0, 1.0, 1.0]
    assert doc.instances[0].keypoints == [[0.0, 0.0]]


def test_parser_returns_empty_document_for_malformed_text() -> None:
    doc = parse_prediction_text(
        text="not json",
        task="detection",
        image="part1/images/a.png",
        image_width=100,
        image_height=100,
        metadata={"model_id": "unit"},
    )

    assert doc.instances == []
    assert doc.metadata["parser"]["decode_valid"] is False
    assert doc.metadata["parser"]["decode_partial"] is False


def test_parser_records_partial_empty_repair_for_malformed_json_array() -> None:
    doc = parse_prediction_text(
        text='[{"label":"arrow","bbox_2d":[0,0,100,100]},{"label":"bbox_2d":[200,200,300,300]}]',
        task="detection",
        image="part1/images/a.png",
        image_width=100,
        image_height=100,
        metadata={"model_id": "unit"},
    )

    assert doc.instances == []
    assert doc.metadata["parser"]["decode_valid"] is True
    assert doc.metadata["parser"]["decode_partial"] is True
    assert doc.metadata["parser"]["decode_empty_repair"] is True
