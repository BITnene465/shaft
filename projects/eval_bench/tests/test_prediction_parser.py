from __future__ import annotations

import json

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
    assert doc.instances[0].bbox == [200.0, 100.0, 600.0, 300.0]
    assert doc.instances[1].bbox == [1000.0, 500.0, 1500.0, 750.0]


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
    assert instance.keypoints == [[0.0, 0.0], [1200.0, 800.0]]
    assert instance.bbox == [0.0, 0.0, 1200.0, 800.0]


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
