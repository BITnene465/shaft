from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval_bench.database import EvalBenchDatabase
from eval_bench.cli import _build_parser, _cmd_ops_summary
from support.cli_contracts import assert_cli_json_payload as _assert_cli_json_payload
from support.files import write_json as _write_json


pytestmark = pytest.mark.contract


def test_cli_prints_agent_ops_summary(tmp_path: Path, capsys) -> None:
    _write_json(
        tmp_path / "benchmarks" / "bench1" / "benchmark.json",
        {
            "benchmark_id": "bench1",
            "benchmark_type": "official",
            "tasks": ["detection"],
            "split": "val",
            "sample_count": 0,
            "root": str(tmp_path / "benchmarks" / "bench1" / "data"),
            "manifest_path": str(tmp_path / "benchmarks" / "bench1" / "splits" / "val.txt"),
        },
    )
    _write_json(
        tmp_path / "runs" / "run-best" / "run.json",
        {
            "run_id": "run-best",
            "status": "succeeded",
            "created_at": "2026-05-09T00:10:00Z",
            "model": {"model_id": "model-a", "path": "outputs/model-a/best"},
            "benchmark": {
                "benchmark_id": "bench1",
                "root": str(tmp_path / "benchmarks" / "bench1" / "data"),
                "split": "val",
                "tasks": ["detection"],
            },
            "spec": {
                "task": "detection",
                "target_labels": ["arrow"],
                "prompt": {"prompt_id": "grounding_arrow.v2.4.main"},
                "metric_profile": "detection",
            },
        },
    )
    _write_json(
        tmp_path / "runs" / "run-best" / "reports" / "summary.json",
        {
            "precision_iou50": 0.75,
            "recall_iou50": 0.75,
            "mean_iou": 0.7,
            "prediction_file_count": 3,
        },
    )
    _write_json(
        tmp_path / "runs" / "run-best" / "reports" / "metrics.json",
        {
            "precision_iou50": 0.75,
            "recall_iou50": 0.75,
            "mean_iou": 0.7,
            "prediction_file_count": 3,
        },
    )
    _write_json(
        tmp_path / "runs" / "run-waiting" / "run.json",
        {
            "run_id": "run-waiting",
            "status": "created",
            "created_at": "2026-05-09T00:20:00Z",
            "model": {"model_id": "model-b", "path": "outputs/model-b/best"},
            "benchmark": {
                "benchmark_id": "bench1",
                "root": str(tmp_path / "benchmarks" / "bench1" / "data"),
                "split": "val",
                "tasks": ["detection"],
            },
            "spec": {"task": "detection", "target_labels": ["arrow"]},
        },
    )
    _write_json(
        tmp_path / "runs" / "run-waiting" / "predictions" / "sample.json",
        {"image": "sample.png", "instances": []},
    )
    database = EvalBenchDatabase(tmp_path)
    database.create_job(kind="eval", job_id="queued-job", payload={}, status="queued")
    database.create_job(kind="eval", job_id="running-job", payload={}, status="running")
    database.create_job(kind="eval", job_id="failed-job", payload={}, status="failed")
    database.upsert_service(
        kind="external_vllm",
        service_id="svc1",
        status="registered",
        config={"endpoint": "http://127.0.0.1:8000/v1"},
    )

    args = _build_parser().parse_args(["ops-summary", "--output-root", str(tmp_path)])
    _cmd_ops_summary(args)
    payload = json.loads(capsys.readouterr().out)

    _assert_cli_json_payload("ops-summary", payload)
    assert payload["source"] == "ops_summary"
    assert payload["runs"]["total"] == 2
    assert payload["runs"]["evaluated"] == 1
    assert payload["runs"]["waiting_evaluation"] == 1
    assert payload["runs"]["best_f1"] == 0.75
    assert payload["runs"]["best_f1_run"]["run_id"] == "run-best"
    assert payload["runs"]["best_f1_run"]["target_labels"] == ["arrow"]
    assert payload["benchmarks"] == {
        "total": 1,
        "sample_count": 0,
        "prediction_count": 4,
    }
    assert payload["jobs"] == {
        "total": 3,
        "queued": 1,
        "running": 1,
        "failed": 1,
        "active": 2,
    }
    assert payload["services"]["total"] == 1
    assert payload["scheduler"]["source"] == "cli_snapshot"
