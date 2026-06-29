from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval_bench.database import EvalBenchDatabase
from eval_bench.cli import (
    _build_parser,
    _cmd_init_run,
)
from support.cli_contracts import assert_cli_json_payload as _assert_cli_json_payload
from support.files import write_json as _write_json


pytestmark = pytest.mark.contract


def test_init_run_cli_accepts_target_label_subset(tmp_path: Path, capsys) -> None:
    args = _build_parser().parse_args(
        [
            "init-run",
            "--output-root",
            str(tmp_path),
            "--run-id",
            "run1",
            "--task",
            "detection",
            "--model-id",
            "model-a",
            "--model-path",
            "outputs/model-a/best",
            "--benchmark-id",
            "bench1",
            "--benchmark-root",
            str(tmp_path / "benchmarks" / "bench1" / "data"),
            "--split",
            "val",
            "--spec-id",
            "layout.icons",
            "--prompt-id",
            "grounding_layout.v2.4.main",
            "--target-label",
            "icon",
            "--target-label",
            "image",
        ]
    )

    _cmd_init_run(args)
    output = json.loads(capsys.readouterr().out)
    _assert_cli_json_payload("init-run", output)

    payload = json.loads((tmp_path / "runs" / "run1" / "run.json").read_text(encoding="utf-8"))
    assert output == {
        "run_id": "run1",
        "manifest_path": str(tmp_path / "runs" / "run1" / "run.json"),
        "artifact_root": str(tmp_path / "runs" / "run1"),
        "task": "detection",
        "benchmark_id": "bench1",
        "target_labels": ["icon", "image"],
        "target_labels_source": "explicit",
    }
    assert payload["spec"]["target_labels"] == ["icon", "image"]
    assert payload["spec"]["metadata"]["target_labels_source"] == "explicit"


def test_init_run_cli_rejects_unknown_target_label_when_benchmark_has_index(
    tmp_path: Path,
) -> None:
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
    args = _build_parser().parse_args(
        [
            "init-run",
            "--output-root",
            str(tmp_path),
            "--run-id",
            "bad_run",
            "--task",
            "detection",
            "--model-id",
            "model-a",
            "--model-path",
            "outputs/model-a/best",
            "--benchmark-id",
            "bench1",
            "--benchmark-root",
            str(tmp_path / "benchmarks" / "bench1" / "data"),
            "--split",
            "val",
            "--spec-id",
            "layout.typo",
            "--prompt-id",
            "grounding_layout.v2.4.main",
            "--target-label",
            "arrwo",
        ]
    )

    with pytest.raises(
        ValueError,
        match="target_labels not found in benchmark label index: arrwo",
    ):
        _cmd_init_run(args)
    assert not (tmp_path / "runs" / "bad_run").exists()


def test_init_run_cli_infers_target_labels_from_prompt_policy(tmp_path: Path) -> None:
    args = _build_parser().parse_args(
        [
            "init-run",
            "--output-root",
            str(tmp_path),
            "--run-id",
            "run1",
            "--task",
            "detection",
            "--model-id",
            "model-a",
            "--model-path",
            "outputs/model-a/best",
            "--benchmark-id",
            "bench1",
            "--benchmark-root",
            str(tmp_path / "benchmarks" / "bench1" / "data"),
            "--split",
            "val",
            "--spec-id",
            "layout.default",
            "--prompt-id",
            "grounding_layout.v2.4.main",
        ]
    )

    _cmd_init_run(args)

    payload = json.loads((tmp_path / "runs" / "run1" / "run.json").read_text(encoding="utf-8"))
    assert payload["spec"]["target_labels"] == ["icon", "image", "shape"]
    assert payload["spec"]["metadata"]["target_labels_source"] == "prompt_metadata"
    assert payload["spec"]["prompt"]["metadata"]["target_labels"] == ["icon", "image", "shape"]
    assert payload["spec"]["parser"] == "raw_data_detection_v1"
    assert payload["spec"]["metric_profile"] == "detection_iou_v1"
    assert payload["spec"]["visualization_profile"] == "default"
    assert payload["spec"]["inference"]["max_tokens"] == 4096
    assert payload["spec"]["inference"]["temperature"] == 0.0
    assert payload["spec"]["inference"]["top_p"] == 1.0
    assert payload["spec"]["inference"]["max_pixels"] == 2_000_000
    assert payload["spec"]["inference"]["batch_size"] == 1


def test_init_run_cli_uses_custom_prompt_template_defaults(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "benchmarks" / "bench1" / "benchmark.json",
        {
            "benchmark_id": "bench1",
            "tasks": ["detection"],
            "labels": ["custom_arrow", "icon"],
            "split": "val",
            "sample_count": 1,
            "root": str(tmp_path / "benchmarks" / "bench1" / "data"),
            "manifest_path": str(tmp_path / "benchmarks" / "bench1" / "splits" / "val.txt"),
        },
    )
    EvalBenchDatabase(tmp_path).upsert_prompt_template(
        {
            "prompt_id": "custom.diagram.arrow",
            "label": "Custom diagram arrow",
            "task": "detection",
            "system_prompt": "Inspect the diagram.",
            "user_prompt": "Find custom arrows.",
            "parser": "custom_parser_v2",
            "metric_profile": "custom_metric_v2",
            "visualization_profile": "custom_visual_v2",
            "generation": {"max_tokens": 123, "temperature": 0.2, "top_p": 0.7},
            "data": {"max_pixels": 654321, "batch_size": 3},
            "metadata": {"target_labels": ["custom_arrow"]},
        }
    )
    args = _build_parser().parse_args(
        [
            "init-run",
            "--output-root",
            str(tmp_path),
            "--run-id",
            "custom_prompt_run",
            "--task",
            "detection",
            "--model-id",
            "model-a",
            "--model-path",
            "outputs/model-a/best",
            "--benchmark-id",
            "bench1",
            "--benchmark-root",
            str(tmp_path / "benchmarks" / "bench1" / "data"),
            "--split",
            "val",
            "--spec-id",
            "custom.prompt",
            "--prompt-id",
            "custom.diagram.arrow",
        ]
    )

    _cmd_init_run(args)

    payload = json.loads(
        (tmp_path / "runs" / "custom_prompt_run" / "run.json").read_text(encoding="utf-8")
    )
    assert payload["spec"]["target_labels"] == ["custom_arrow"]
    assert payload["spec"]["metadata"]["target_labels_source"] == "prompt_metadata"
    assert payload["spec"]["prompt"]["metadata"]["target_labels"] == ["custom_arrow"]
    assert payload["spec"]["parser"] == "custom_parser_v2"
    assert payload["spec"]["metric_profile"] == "custom_metric_v2"
    assert payload["spec"]["visualization_profile"] == "custom_visual_v2"
    assert payload["spec"]["inference"]["max_tokens"] == 123
    assert payload["spec"]["inference"]["temperature"] == 0.2
    assert payload["spec"]["inference"]["top_p"] == 0.7
    assert payload["spec"]["inference"]["max_pixels"] == 654321
    assert payload["spec"]["inference"]["batch_size"] == 3


def test_init_run_cli_generation_args_override_prompt_template_defaults(tmp_path: Path) -> None:
    EvalBenchDatabase(tmp_path).upsert_prompt_template(
        {
            "prompt_id": "custom.diagram.arrow",
            "label": "Custom diagram arrow",
            "task": "detection",
            "system_prompt": "Inspect the diagram.",
            "user_prompt": "Find custom arrows.",
            "generation": {"max_tokens": 123, "temperature": 0.2, "top_p": 0.7},
            "data": {"max_pixels": 654321, "batch_size": 3},
            "metadata": {"target_labels": ["custom_arrow"]},
        }
    )
    args = _build_parser().parse_args(
        [
            "init-run",
            "--output-root",
            str(tmp_path),
            "--run-id",
            "custom_prompt_run",
            "--task",
            "detection",
            "--model-id",
            "model-a",
            "--model-path",
            "outputs/model-a/best",
            "--benchmark-id",
            "bench1",
            "--benchmark-root",
            str(tmp_path / "benchmarks" / "bench1" / "data"),
            "--split",
            "val",
            "--spec-id",
            "custom.prompt",
            "--prompt-id",
            "custom.diagram.arrow",
            "--max-tokens",
            "777",
            "--temperature",
            "0",
            "--top-p",
            "1",
            "--max-pixels",
            "42",
            "--batch-size",
            "2",
        ]
    )

    _cmd_init_run(args)

    payload = json.loads(
        (tmp_path / "runs" / "custom_prompt_run" / "run.json").read_text(encoding="utf-8")
    )
    assert payload["spec"]["inference"]["max_tokens"] == 777
    assert payload["spec"]["inference"]["temperature"] == 0.0
    assert payload["spec"]["inference"]["top_p"] == 1.0
    assert payload["spec"]["inference"]["max_pixels"] == 42
    assert payload["spec"]["inference"]["batch_size"] == 2
