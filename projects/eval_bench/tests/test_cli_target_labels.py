from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval_bench.database import EvalBenchDatabase
from eval_bench.cli import _build_parser, _cmd_resolve_target_labels
from support.cli_contracts import assert_cli_json_payload as _assert_cli_json_payload
from support.files import write_json as _write_json


pytestmark = pytest.mark.contract


def test_cli_resolves_target_labels_from_prompt_metadata(tmp_path: Path, capsys) -> None:
    _write_target_label_store(tmp_path)

    args = _build_parser().parse_args(
        [
            "resolve-target-labels",
            "--output-root",
            str(tmp_path),
            "--benchmark-id",
            "bench1",
            "--prompt-id",
            "grounding_arrow.v2.4.main",
        ]
    )
    _cmd_resolve_target_labels(args)
    payload = json.loads(capsys.readouterr().out)

    _assert_cli_json_payload("resolve-target-labels", payload)
    assert payload["task"] == "detection"
    assert payload["target_labels"] == ["arrow"]
    assert payload["target_labels_source"] == "prompt_metadata"
    assert payload["candidate_labels"] == ["arrow", "icon"]
    assert payload["label_subtasks_supported"] is True
    assert payload["valid"] is True


def test_cli_resolves_legacy_keypoint_arrow_target(tmp_path: Path, capsys) -> None:
    _write_target_label_store(tmp_path)

    args = _build_parser().parse_args(
        [
            "resolve-target-labels",
            "--output-root",
            str(tmp_path),
            "--benchmark-id",
            "bench1",
            "--task",
            "keypoint",
            "--prompt-id",
            "keypoint_arrow.test.main",
        ]
    )
    _cmd_resolve_target_labels(args)
    payload = json.loads(capsys.readouterr().out)

    _assert_cli_json_payload("resolve-target-labels", payload)
    assert payload["task"] == "keypoint"
    assert payload["target_labels"] == ["arrow"]
    assert payload["label_subtasks_supported"] is False
    assert payload["valid"] is True


def test_cli_rejects_keypoint_non_arrow_target_label(tmp_path: Path, capsys) -> None:
    _write_target_label_store(tmp_path)

    args = _build_parser().parse_args(
        [
            "resolve-target-labels",
            "--output-root",
            str(tmp_path),
            "--benchmark-id",
            "bench1",
            "--task",
            "keypoint",
            "--target-label",
            "icon",
        ]
    )
    _cmd_resolve_target_labels(args)
    payload = json.loads(capsys.readouterr().out)

    _assert_cli_json_payload("resolve-target-labels", payload)
    assert payload["label_subtasks_supported"] is False
    assert payload["valid"] is False
    assert any("keypoint target_labels only support arrow" in item for item in payload["errors"])


def test_cli_rejects_unknown_target_label(tmp_path: Path, capsys) -> None:
    _write_target_label_store(tmp_path)

    args = _build_parser().parse_args(
        [
            "resolve-target-labels",
            "--output-root",
            str(tmp_path),
            "--benchmark-id",
            "bench1",
            "--task",
            "detection",
            "--target-label",
            "arrwo",
        ]
    )
    _cmd_resolve_target_labels(args)
    payload = json.loads(capsys.readouterr().out)

    _assert_cli_json_payload("resolve-target-labels", payload)
    assert payload["valid"] is False
    assert any("arrwo" in item for item in payload["errors"])


def _write_target_label_store(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "benchmarks" / "bench1" / "benchmark.json",
        {
            "benchmark_id": "bench1",
            "tasks": ["detection"],
            "labels": ["arrow", "icon"],
            "split": "val",
            "sample_count": 1,
            "root": str(tmp_path / "benchmarks" / "bench1" / "data"),
            "manifest_path": str(tmp_path / "benchmarks" / "bench1" / "splits" / "val.txt"),
        },
    )
    EvalBenchDatabase(tmp_path).upsert_prompt_template(
        {
            "prompt_id": "grounding_arrow.v2.4.main",
            "label": "Arrow grounding",
            "task": "detection",
            "system_prompt": "You inspect diagrams.",
            "user_prompt": "Find arrows.",
            "metadata": {"target_labels": ["arrow"]},
        }
    )
