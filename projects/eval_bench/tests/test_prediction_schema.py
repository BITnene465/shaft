from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval_bench.artifacts import RunArtifacts, load_prediction
from eval_bench.schema import PredictionDocument, PredictionInstance


def test_prediction_document_allows_detection_without_keypoints(tmp_path: Path) -> None:
    doc = PredictionDocument(
        image="part1/images/a.png",
        instances=[PredictionInstance(label="icon", bbox=[1, 2, 10, 20])],
        metadata={"producer": "eval_bench"},
    )

    artifacts = RunArtifacts(tmp_path, "run1")
    path = artifacts.write_prediction(doc, task="detection")

    loaded = load_prediction(path, task="detection")
    assert loaded.image == "part1/images/a.png"
    assert loaded.instances[0].label == "icon"
    assert "keypoints" not in json.loads(path.read_text())["instances"][0]


def test_keypoint_document_allows_missing_keypoints_for_metric_failure() -> None:
    doc = PredictionDocument(
        image="part1/images/a.png",
        instances=[PredictionInstance(label="arrow", bbox=[1, 2, 10, 20])],
        metadata={"producer": "eval_bench"},
    )

    payload = doc.to_dict(task="keypoint")

    assert payload["instances"][0]["label"] == "arrow"
    assert "keypoints" not in payload["instances"][0]


def test_prediction_document_rejects_malformed_instance_payload(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            {
                "image": "part1/images/a.png",
                "status": "predicted",
                "instances": ["not-an-object"],
                "metadata": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"instances\[0\] must be an object"):
        load_prediction(path, task="detection")
