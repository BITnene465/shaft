from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval_bench.database import EvalBenchDatabase
from eval_bench.cli import (
    _build_parser,
    _cmd_import_predictions,
)
from support.cli_contracts import assert_cli_json_payload as _assert_cli_json_payload
from support.files import write_json as _write_json


pytestmark = pytest.mark.contract


def test_cli_import_predictions_accepts_target_label_subset(tmp_path: Path, capsys) -> None:
    split_path = tmp_path / "benchmarks" / "bench1" / "splits" / "val.txt"
    data_root = tmp_path / "benchmarks" / "bench1" / "data"
    split_path.parent.mkdir(parents=True)
    split_path.write_text("part1/json/a.json\n", encoding="utf-8")
    _write_json(
        tmp_path / "benchmarks" / "bench1" / "benchmark.json",
        {
            "benchmark_id": "bench1",
            "tasks": ["detection"],
            "split": "val",
            "sample_count": 1,
            "root": str(data_root),
            "manifest_path": str(split_path),
        },
    )
    _write_json(
        data_root / "part1" / "json" / "a.json",
        {
            "image_path": "part1/images/a.png",
            "instances": [
                {"label": "icon", "bbox": [0, 0, 100, 100]},
                {"label": "arrow", "bbox": [200, 200, 260, 260]},
            ],
        },
    )
    prediction_root = tmp_path / "predictions"
    _write_json(
        prediction_root / "part1" / "json" / "a.json",
        {
            "image": "part1/images/a.png",
            "instances": [
                {"label": "icon", "bbox": [0, 0, 100, 100]},
                {"label": "arrow", "bbox": [200, 200, 260, 260]},
            ],
        },
    )
    args = _build_parser().parse_args(
        [
            "import-predictions",
            "--output-root",
            str(tmp_path),
            "--run-id",
            "imported_arrow",
            "--benchmark-id",
            "bench1",
            "--prediction-root",
            str(prediction_root),
            "--task",
            "detection",
            "--model-id",
            "external-model",
            "--prompt-id",
            "grounding_layout.v2.4.main",
            "--target-label",
            "arrow",
        ]
    )

    _cmd_import_predictions(args)
    payload = json.loads(capsys.readouterr().out)
    _assert_cli_json_payload("import-predictions", payload)
    report = json.loads(Path(payload["report_path"]).read_text(encoding="utf-8"))

    assert payload["run_id"] == "imported_arrow"
    assert payload["summary_path"].endswith("summary.json")
    assert Path(payload["summary_path"]).exists()
    assert report["target_labels"] == ["arrow"]
    assert report["target_labels_source"] == "explicit"
    assert [item["label"] for item in report["labels"]] == ["arrow"]


def test_cli_import_predictions_uses_prompt_template_target_labels(tmp_path: Path, capsys) -> None:
    split_path = tmp_path / "benchmarks" / "bench1" / "splits" / "val.txt"
    data_root = tmp_path / "benchmarks" / "bench1" / "data"
    split_path.parent.mkdir(parents=True)
    split_path.write_text("part1/json/a.json\n", encoding="utf-8")
    _write_json(
        tmp_path / "benchmarks" / "bench1" / "benchmark.json",
        {
            "benchmark_id": "bench1",
            "tasks": ["detection"],
            "labels": ["custom_arrow", "icon"],
            "split": "val",
            "sample_count": 1,
            "root": str(data_root),
            "manifest_path": str(split_path),
        },
    )
    _write_json(
        data_root / "part1" / "json" / "a.json",
        {
            "image_path": "part1/images/a.png",
            "instances": [
                {"label": "icon", "bbox": [0, 0, 100, 100]},
                {"label": "custom_arrow", "bbox": [200, 200, 260, 260]},
            ],
        },
    )
    prediction_root = tmp_path / "predictions"
    _write_json(
        prediction_root / "part1" / "json" / "a.json",
        {
            "image": "part1/images/a.png",
            "instances": [
                {"label": "icon", "bbox": [0, 0, 100, 100]},
                {"label": "custom_arrow", "bbox": [200, 200, 260, 260]},
            ],
        },
    )
    EvalBenchDatabase(tmp_path).upsert_prompt_template(
        {
            "prompt_id": "custom.arrow.import",
            "label": "Custom arrow import",
            "task": "detection",
            "system_prompt": "Inspect diagrams.",
            "user_prompt": "Find custom arrows.",
            "metadata": {"target_labels": ["custom_arrow"], "owner": "bench-team"},
        }
    )
    args = _build_parser().parse_args(
        [
            "import-predictions",
            "--output-root",
            str(tmp_path),
            "--run-id",
            "imported_custom_arrow",
            "--benchmark-id",
            "bench1",
            "--prediction-root",
            str(prediction_root),
            "--task",
            "detection",
            "--model-id",
            "external-model",
            "--prompt-id",
            "custom.arrow.import",
        ]
    )

    _cmd_import_predictions(args)
    payload = json.loads(capsys.readouterr().out)
    _assert_cli_json_payload("import-predictions", payload)
    report = json.loads(Path(payload["report_path"]).read_text(encoding="utf-8"))
    run_payload = json.loads(
        (tmp_path / "runs" / "imported_custom_arrow" / "run.json").read_text(encoding="utf-8")
    )

    assert report["target_labels"] == ["custom_arrow"]
    assert report["target_labels_source"] == "prompt_metadata"
    assert [item["label"] for item in report["labels"]] == ["custom_arrow"]
    assert run_payload["spec"]["prompt"]["metadata"]["target_labels"] == ["custom_arrow"]
    assert run_payload["spec"]["prompt"]["metadata"]["owner"] == "bench-team"


def test_cli_import_predictions_rejects_unknown_target_label(tmp_path: Path) -> None:
    split_path = tmp_path / "benchmarks" / "bench1" / "splits" / "val.txt"
    data_root = tmp_path / "benchmarks" / "bench1" / "data"
    split_path.parent.mkdir(parents=True)
    split_path.write_text("part1/json/a.json\n", encoding="utf-8")
    _write_json(
        tmp_path / "benchmarks" / "bench1" / "benchmark.json",
        {
            "benchmark_id": "bench1",
            "tasks": ["detection"],
            "labels": ["arrow", "icon"],
            "split": "val",
            "sample_count": 1,
            "root": str(data_root),
            "manifest_path": str(split_path),
        },
    )
    prediction_root = tmp_path / "predictions"
    prediction_root.mkdir()
    args = _build_parser().parse_args(
        [
            "import-predictions",
            "--output-root",
            str(tmp_path),
            "--run-id",
            "bad_import",
            "--benchmark-id",
            "bench1",
            "--prediction-root",
            str(prediction_root),
            "--task",
            "detection",
            "--model-id",
            "external-model",
            "--target-label",
            "arrwo",
        ]
    )

    with pytest.raises(
        ValueError,
        match="target_labels not found in benchmark label index: arrwo",
    ):
        _cmd_import_predictions(args)
    assert not (tmp_path / "runs" / "bad_import").exists()
