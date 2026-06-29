from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from eval_bench.cli import (
    _build_parser,
    _cmd_list_benchmark_samples,
    _cmd_list_run_samples,
    _cmd_show_benchmark_sample,
    _cmd_show_comparison,
    _cmd_show_comparison_sample,
    _cmd_show_run,
    _cmd_show_run_report,
    _cmd_show_run_sample,
)
from support.cli_contracts import assert_cli_json_payload as _assert_cli_json_payload
from support.files import write_json as _write_json
from support.store import write_sample_store as _write_sample_store


pytestmark = pytest.mark.contract


def test_cli_shows_saved_comparison_and_sample_detail_for_agents(
    tmp_path: Path,
    capsys,
) -> None:
    _write_sample_store(tmp_path)
    source_run = tmp_path / "runs" / "run_arrow"
    for run_id in ("run_base", "run_a"):
        target = tmp_path / "runs" / run_id
        shutil.copytree(source_run, target)
        run_manifest = json.loads((target / "run.json").read_text(encoding="utf-8"))
        run_manifest["run_id"] = run_id
        _write_json(target / "run.json", run_manifest)
    _write_json(
        tmp_path / "exports" / "comparisons" / "run_base__vs__run_a.json",
        {
            "comparison_id": "run_base__vs__run_a",
            "baseline_run_id": "run_base",
            "candidate_run_id": "run_a",
            "task": "detection",
            "metric_profile": "detection_iou_v1",
            "target_labels": ["arrow"],
            "sample_count": 2,
            "created_at": "2026-05-09T00:30:00Z",
            "delta": {"recall_iou50": 0.0},
            "summary": {"improved_samples": 0, "regressed_samples": 0},
        },
    )

    show_args = _build_parser().parse_args(
        [
            "show-comparison",
            "--output-root",
            str(tmp_path),
            "--baseline-run-id",
            "run_base",
            "--candidate-run-id",
            "run_a",
        ]
    )
    _cmd_show_comparison(show_args)
    comparison = json.loads(capsys.readouterr().out)
    _assert_cli_json_payload("show-comparison", comparison)
    assert comparison["comparison_id"] == "run_base__vs__run_a"
    assert comparison["target_labels"] == ["arrow"]
    assert comparison["delta"]["recall_iou50"] == 0.0
    assert comparison["summary"]["improved_samples"] == 0

    by_id_args = _build_parser().parse_args(
        [
            "show-comparison",
            "--output-root",
            str(tmp_path),
            "--comparison-id",
            "run_base__vs__run_a",
        ]
    )
    _cmd_show_comparison(by_id_args)
    comparison_by_id = json.loads(capsys.readouterr().out)
    assert comparison_by_id["baseline_run_id"] == "run_base"

    sample_args = _build_parser().parse_args(
        [
            "show-comparison-sample",
            "--output-root",
            str(tmp_path),
            "--baseline-run-id",
            "run_base",
            "--candidate-run-id",
            "run_a",
            "--sample-index",
            "0",
        ]
    )
    _cmd_show_comparison_sample(sample_args)
    sample = json.loads(capsys.readouterr().out)
    _assert_cli_json_payload("show-comparison-sample", sample)
    assert sample["baseline_run_id"] == "run_base"
    assert sample["candidate_run_id"] == "run_a"
    assert sample["sample_index"] == 0
    assert sample["baseline"]["sample"]["index"] == 0
    assert sample["candidate"]["sample"]["index"] == 0
    assert [item["label"] for item in sample["baseline"]["gt_instances"]] == ["arrow"]
    assert [item["label"] for item in sample["baseline"]["raw_payload"]["instances"]] == ["arrow"]


def test_cli_reads_run_reports_and_scoped_samples_for_agents(tmp_path: Path, capsys) -> None:
    _write_sample_store(tmp_path)

    show_args = _build_parser().parse_args(
        ["show-run", "--output-root", str(tmp_path), "--run-id", "run_arrow"]
    )
    _cmd_show_run(show_args)
    run_payload = json.loads(capsys.readouterr().out)
    assert run_payload["run"]["run_id"] == "run_arrow"
    assert run_payload["run"]["target_labels"] == ["arrow"]
    assert run_payload["run"]["f1_iou50"] == 1.0
    assert run_payload["run"]["precision_iou50"] == 1.0

    report_args = _build_parser().parse_args(
        ["show-run-report", "--output-root", str(tmp_path), "--run-id", "run_arrow", "--summary"]
    )
    _cmd_show_run_report(report_args)
    report = json.loads(capsys.readouterr().out)
    _assert_cli_json_payload("show-run-report", report)
    assert report["target_labels_source"] == "explicit"
    assert report["labels"] == ["arrow"]

    samples_args = _build_parser().parse_args(
        [
            "list-run-samples",
            "--output-root",
            str(tmp_path),
            "--run-id",
            "run_arrow",
            "--label",
            "arrow",
        ]
    )
    _cmd_list_run_samples(samples_args)
    samples = json.loads(capsys.readouterr().out)
    assert samples["filters"] == {
        "run_id": "run_arrow",
        "label": "arrow",
        "error_filter": "all",
    }
    assert samples["labels"] == ["arrow"]
    assert samples["total"] == 1
    assert samples["samples"][0]["labels"] == ["arrow"]
    assert samples["samples"][0]["gt_instance_count"] == 1
    assert samples["samples"][0]["diagnostics"]["matched_count"] == 1

    detail_args = _build_parser().parse_args(
        [
            "show-run-sample",
            "--output-root",
            str(tmp_path),
            "--run-id",
            "run_arrow",
            "--sample-index",
            "0",
        ]
    )
    _cmd_show_run_sample(detail_args)
    detail = json.loads(capsys.readouterr().out)
    _assert_cli_json_payload("show-run-sample", detail)
    assert [item["label"] for item in detail["gt_instances"]] == ["arrow"]
    assert [item["label"] for item in detail["pred_instances"]] == ["arrow"]
    assert [item["label"] for item in detail["raw_payload"]["instances"]] == ["arrow"]
    assert [item["label"] for item in detail["prediction_payload"]["instances"]] == ["arrow"]


def test_cli_reads_benchmark_samples_for_agents(tmp_path: Path, capsys) -> None:
    _write_sample_store(tmp_path)

    samples_args = _build_parser().parse_args(
        [
            "list-benchmark-samples",
            "--output-root",
            str(tmp_path),
            "--benchmark-id",
            "bench1",
            "--label",
            "arrow",
        ]
    )
    _cmd_list_benchmark_samples(samples_args)
    samples = json.loads(capsys.readouterr().out)
    assert samples["benchmark_id"] == "bench1"
    assert samples["filters"] == {"benchmark_id": "bench1", "label": "arrow"}
    assert samples["labels"] == ["arrow", "icon"]
    assert samples["total"] == 1
    assert samples["samples"][0]["labels"] == ["arrow", "icon"]

    detail_args = _build_parser().parse_args(
        [
            "show-benchmark-sample",
            "--output-root",
            str(tmp_path),
            "--benchmark-id",
            "bench1",
            "--sample-index",
            "0",
        ]
    )
    _cmd_show_benchmark_sample(detail_args)
    detail = json.loads(capsys.readouterr().out)
    _assert_cli_json_payload("show-benchmark-sample", detail)
    assert detail["sample"]["instance_count"] == 2
    assert [item["label"] for item in detail["gt_instances"]] == ["icon", "arrow"]
