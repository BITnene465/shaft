from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval_bench.cli import _build_parser, _cmd_rank_board
from support.cli_contracts import assert_cli_json_payload as _assert_cli_json_payload
from support.files import write_json as _write_json


pytestmark = pytest.mark.contract


def test_cli_rank_board_filters_by_label_split_and_min_score(tmp_path: Path, capsys) -> None:
    _write_rank_board_store(tmp_path)

    args = _build_parser().parse_args(
        [
            "rank-board",
            "--output-root",
            str(tmp_path),
            "--label",
            "icon",
            "--metric-profile",
            "detection_iou_v1",
            "--benchmark-split",
            "grounding_layout",
            "--min-score",
            "0.7",
            "--sort-by",
            "run_id",
            "--sort-order",
            "desc",
        ]
    )
    _cmd_rank_board(args)
    payload = json.loads(capsys.readouterr().out)

    _assert_cli_json_payload("rank-board", payload)
    assert payload["total"] == 1
    assert payload["primary_metric"] == "f1_iou50"
    assert payload["primary_metric_label"] == "F1@.50"
    assert payload["sort_by"] == "run_id"
    assert payload["sort_order"] == "desc"
    assert payload["filters"]["benchmark_split"] == "grounding_layout"
    assert payload["filters"]["min_score"] == "0.7"
    assert payload["facets"]["splits"] == [{"value": "grounding_layout", "count": 1}]
    assert payload["facets"]["metric_profiles"] == [{"value": "detection_iou_v1", "count": 1}]
    assert payload["entries"][0]["run_id"] == "run_a"
    assert payload["entries"][0]["benchmark_split"] == "grounding_layout"
    assert payload["entries"][0]["rank"] == 1
    assert payload["entries"][0]["score"] == pytest.approx(0.9)
    assert payload["entries"][0]["score_delta"] == pytest.approx(0.0)


def test_cli_rank_board_uses_requested_metric_as_primary_score(tmp_path: Path, capsys) -> None:
    _write_rank_board_store(tmp_path)

    args = _build_parser().parse_args(
        [
            "rank-board",
            "--output-root",
            str(tmp_path),
            "--sort-by",
            "recall_iou50",
        ]
    )
    _cmd_rank_board(args)
    payload = json.loads(capsys.readouterr().out)

    _assert_cli_json_payload("rank-board", payload)
    assert payload["primary_metric"] == "recall_iou50"
    assert payload["primary_metric_label"] == "R@.50"
    assert payload["score_label"] == "R@.50"
    assert payload["entries"][0]["run_id"] == "run_a"
    assert payload["entries"][0]["score"] == pytest.approx(0.9)
    assert payload["entries"][0]["score_delta"] == pytest.approx(0.0)
    assert payload["entries"][1]["score_delta"] < 0


def _write_rank_board_store(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "benchmarks" / "bench1" / "benchmark.json",
        {
            "benchmark_id": "bench1",
            "benchmark_type": "official",
            "tasks": ["detection"],
            "layers": ["layout", "arrow"],
            "labels": ["icon", "arrow"],
            "split": "suite",
            "sample_count": 0,
            "sample_counts": {"grounding_layout": 0, "grounding_arrow": 0},
            "split_manifests": {
                "grounding_layout": str(
                    tmp_path / "benchmarks" / "bench1" / "splits" / "grounding_layout.txt"
                ),
                "grounding_arrow": str(
                    tmp_path / "benchmarks" / "bench1" / "splits" / "grounding_arrow.txt"
                ),
            },
            "root": str(tmp_path / "benchmarks" / "bench1" / "data"),
            "manifest_path": str(tmp_path / "benchmarks" / "bench1" / "splits" / "suite.txt"),
        },
    )
    for run_id, label, split, precision in (
        ("run_a", "icon", "grounding_layout", 0.9),
        ("run_b", "arrow", "grounding_arrow", 0.5),
    ):
        _write_json(
            tmp_path / "runs" / run_id / "run.json",
            {
                "run_id": run_id,
                "status": "succeeded",
                "created_at": "2026-05-09T00:10:00Z",
                "model": {"model_id": run_id, "path": "outputs/model/best"},
                "benchmark": {
                    "benchmark_id": "bench1",
                    "root": str(tmp_path / "benchmarks" / "bench1" / "data"),
                    "split": split,
                    "tasks": ["detection"],
                },
                "spec": {
                    "task": "detection",
                    "metric_profile": "detection_iou_v1",
                    "target_labels": [label],
                },
            },
        )
        _write_json(
            tmp_path / "runs" / run_id / "reports" / "summary.json",
            {
                "precision_iou50": precision,
                "recall_iou50": precision,
                "mean_iou": precision,
                "prediction_file_count": 1,
            },
        )
