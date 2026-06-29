from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval_bench.cli import (
    _build_parser,
    _cmd_list_benchmarks,
    _cmd_list_comparisons,
    _cmd_list_runs,
    _cmd_show_benchmark,
)
from support.cli_contracts import assert_cli_json_payload as _assert_cli_json_payload
from support.files import write_json as _write_json


pytestmark = pytest.mark.contract


def test_cli_lists_benchmarks_with_agent_filters(tmp_path: Path, capsys) -> None:
    _write_listing_store(tmp_path)

    args = _build_parser().parse_args(
        [
            "list-benchmarks",
            "--output-root",
            str(tmp_path),
            "--task",
            "detection",
            "--layer",
            "layout",
            "--split",
            "val",
            "--query",
            "bench1",
        ]
    )
    _cmd_list_benchmarks(args)
    payload = json.loads(capsys.readouterr().out)

    _assert_cli_json_payload("list-benchmarks", payload)
    assert payload["total"] == 1
    assert payload["filters"]["task"] == "detection"
    assert payload["filters"]["split"] == "val"
    assert payload["facets"]["tasks"] == [
        {"value": "detection", "count": 1},
        {"value": "keypoint", "count": 1},
    ]
    assert payload["benchmarks"][0]["benchmark_id"] == "bench1"
    assert payload["benchmarks"][0]["labels"] == ["arrow", "icon"]

    detail_args = _build_parser().parse_args(
        ["show-benchmark", "--output-root", str(tmp_path), "--benchmark-id", "bench1"]
    )
    _cmd_show_benchmark(detail_args)
    detail = json.loads(capsys.readouterr().out)
    _assert_cli_json_payload("show-benchmark", detail)
    assert detail["benchmark"]["benchmark_id"] == "bench1"
    assert detail["benchmark"]["tasks"] == ["detection"]
    assert detail["benchmark"]["labels"] == ["arrow", "icon"]
    assert detail["benchmark"]["sample_count"] == 2


def test_cli_lists_runs_with_agent_filters(tmp_path: Path, capsys) -> None:
    _write_listing_store(tmp_path)

    args = _build_parser().parse_args(
        [
            "list-runs",
            "--output-root",
            str(tmp_path),
            "--task",
            "detection",
            "--benchmark-id",
            "bench1",
            "--benchmark-split",
            "val",
            "--label",
            "arrow",
            "--model-id",
            "model-a",
            "--metric-profile",
            "detection_iou_v1",
            "--query",
            "grounding",
        ]
    )
    _cmd_list_runs(args)
    payload = json.loads(capsys.readouterr().out)

    _assert_cli_json_payload("list-runs", payload)
    assert payload["total"] == 1
    assert payload["filters"]["benchmark_split"] == "val"
    assert payload["filters"]["label"] == "arrow"
    assert payload["facets"]["labels"] == [{"value": "arrow", "count": 2}]
    assert payload["runs"][0]["run_id"] == "run_a"
    assert payload["runs"][0]["target_labels"] == ["arrow"]


def test_cli_lists_comparisons_with_agent_filters(tmp_path: Path, capsys) -> None:
    _write_listing_store(tmp_path)

    args = _build_parser().parse_args(
        [
            "list-comparisons",
            "--output-root",
            str(tmp_path),
            "--task",
            "detection",
            "--benchmark-id",
            "bench1",
            "--benchmark-split",
            "val",
            "--baseline-run-id",
            "run_base",
            "--label",
            "arrow",
            "--query",
            "run_a",
        ]
    )
    _cmd_list_comparisons(args)
    payload = json.loads(capsys.readouterr().out)

    _assert_cli_json_payload("list-comparisons", payload)
    assert payload["total"] == 1
    assert payload["filters"]["benchmark_id"] == "bench1"
    assert payload["filters"]["benchmark_split"] == "val"
    assert payload["filters"]["baseline_run_id"] == "run_base"
    assert payload["comparisons"][0]["comparison_id"] == "run_base__vs__run_a"
    assert payload["comparisons"][0]["metric_profile"] == "detection_iou_v1"
    assert payload["comparisons"][0]["warnings"] == [
        "baseline and candidate benchmark splits differ"
    ]


def _write_listing_store(tmp_path: Path) -> None:
    _write_benchmark(tmp_path, "bench1", task="detection", labels=["arrow", "icon"])
    _write_benchmark(tmp_path, "bench2", task="keypoint", layers=["arrow"], sample_count=1)
    _write_run(tmp_path, "run_a", task="detection", benchmark_id="bench1", model_id="model-a")
    _write_run(
        tmp_path,
        "run_b",
        task="keypoint",
        benchmark_id="bench2",
        model_id="model-b",
        status="failed",
        metric_profile="keypoint_endpoint_v1",
        prompt_id="keypoint_arrow.test.main",
    )
    _write_json(
        tmp_path / "exports" / "comparisons" / "run_base__vs__run_a.json",
        {
            "comparison_id": "run_base__vs__run_a",
            "baseline_run_id": "run_base",
            "candidate_run_id": "run_a",
            "benchmark_id": "bench1",
            "benchmark_split": "val",
            "task": "detection",
            "metric_profile": "detection_iou_v1",
            "target_labels": ["arrow"],
            "warnings": ["baseline and candidate benchmark splits differ"],
            "sample_count": 2,
            "created_at": "2026-05-09T00:30:00Z",
            "delta": {"precision_iou50": 0.2},
            "summary": {"improved_samples": 1},
        },
    )


def _write_benchmark(
    tmp_path: Path,
    benchmark_id: str,
    *,
    task: str,
    labels: list[str] | None = None,
    layers: list[str] | None = None,
    sample_count: int = 2,
) -> None:
    _write_json(
        tmp_path / "benchmarks" / benchmark_id / "benchmark.json",
        {
            "benchmark_id": benchmark_id,
            "tasks": [task],
            "labels": labels or [],
            "layers": layers or ["layout"],
            "split": "val",
            "sample_count": sample_count,
            "root": str(tmp_path / "benchmarks" / benchmark_id / "data"),
            "manifest_path": str(tmp_path / "benchmarks" / benchmark_id / "splits" / "val.txt"),
            "created_at": "2026-05-09T00:00:00Z",
        },
    )


def _write_run(
    tmp_path: Path,
    run_id: str,
    *,
    task: str,
    benchmark_id: str,
    model_id: str,
    status: str = "succeeded",
    metric_profile: str = "detection_iou_v1",
    prompt_id: str = "grounding_arrow.v2.4.main",
) -> None:
    _write_json(
        tmp_path / "runs" / run_id / "run.json",
        {
            "run_id": run_id,
            "status": status,
            "created_at": "2026-05-09T00:10:00Z",
            "model": {"model_id": model_id, "path": f"outputs/{model_id}/best"},
            "benchmark": {
                "benchmark_id": benchmark_id,
                "root": str(tmp_path / "benchmarks" / benchmark_id / "data"),
                "split": "val",
                "tasks": [task],
            },
            "spec": {
                "task": task,
                "metric_profile": metric_profile,
                "target_labels": ["arrow"],
                "prompt": {"prompt_id": prompt_id},
            },
        },
    )
