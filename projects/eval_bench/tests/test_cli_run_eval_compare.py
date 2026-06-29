from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from eval_bench.cli import _build_parser, _cmd_compare_runs, _cmd_evaluate_run
from support.cli_contracts import assert_cli_json_payload as _assert_cli_json_payload
from support.files import write_json as _write_json
from support.store import write_sample_store as _write_sample_store


pytestmark = pytest.mark.contract


def test_cli_evaluate_run_writes_report_payload(tmp_path: Path, capsys) -> None:
    _write_sample_store(tmp_path)

    args = _build_parser().parse_args(
        ["evaluate-run", "--output-root", str(tmp_path), "--run-id", "run_arrow"]
    )
    _cmd_evaluate_run(args)
    payload = json.loads(capsys.readouterr().out)

    _assert_cli_json_payload("evaluate-run", payload)
    assert payload == {
        "run_id": "run_arrow",
        "report_path": str(tmp_path / "runs" / "run_arrow" / "reports" / "metrics.json"),
        "summary_path": str(tmp_path / "runs" / "run_arrow" / "reports" / "summary.json"),
    }
    assert Path(payload["report_path"]).exists()
    assert Path(payload["summary_path"]).exists()


def test_cli_compare_runs_writes_comparison_report(tmp_path: Path, capsys) -> None:
    _write_sample_store(tmp_path)
    for run_id in ("run_base", "run_a"):
        _clone_sample_run(tmp_path, run_id)

    args = _build_parser().parse_args(
        [
            "compare-runs",
            "--output-root",
            str(tmp_path),
            "--baseline-run-id",
            "run_base",
            "--candidate-run-id",
            "run_a",
        ]
    )
    _cmd_compare_runs(args)
    payload = json.loads(capsys.readouterr().out)

    _assert_cli_json_payload("compare-runs", payload)
    assert payload == {
        "comparison_id": "run_base__vs__run_a",
        "baseline_run_id": "run_base",
        "candidate_run_id": "run_a",
        "benchmark_id": "bench1",
        "benchmark_split": "val",
        "warnings": [],
        "report_path": str(tmp_path / "exports" / "comparisons" / "run_base__vs__run_a.json"),
    }
    assert Path(payload["report_path"]).exists()


def _clone_sample_run(tmp_path: Path, run_id: str) -> None:
    source_run = tmp_path / "runs" / "run_arrow"
    target = tmp_path / "runs" / run_id
    shutil.copytree(source_run, target)
    run_manifest = json.loads((target / "run.json").read_text(encoding="utf-8"))
    run_manifest["run_id"] = run_id
    _write_json(target / "run.json", run_manifest)
    for report_name in ("metrics.json", "summary.json"):
        report_path = target / "reports" / report_name
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["run_id"] = run_id
        report["benchmark_id"] = "bench1"
        report["benchmark_split"] = "val"
        _write_json(report_path, report)
