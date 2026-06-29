from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval_bench.cli import (
    _build_parser,
    _cmd_create_job,
    _cmd_list_jobs,
    _cmd_preflight_job,
    _cmd_show_job,
)
from support.cli_contracts import assert_cli_json_payload as _assert_cli_json_payload
from support.files import write_json as _write_json


pytestmark = pytest.mark.contract


def _write_benchmark_manifest(
    tmp_path: Path,
    *,
    tasks: list[str],
    labels: list[str] | None = None,
    create_split: bool = False,
) -> None:
    if create_split:
        (tmp_path / "benchmarks" / "bench1" / "splits").mkdir(parents=True)
        (tmp_path / "benchmarks" / "bench1" / "splits" / "val.txt").write_text(
            "part1/json/a.json\n",
            encoding="utf-8",
        )
    payload = {
        "benchmark_id": "bench1",
        "tasks": tasks,
        "split": "val",
        "sample_count": 1,
        "root": str(tmp_path / "benchmarks" / "bench1" / "data"),
        "manifest_path": str(tmp_path / "benchmarks" / "bench1" / "splits" / "val.txt"),
    }
    if labels is not None:
        payload["labels"] = labels
    _write_json(tmp_path / "benchmarks" / "bench1" / "benchmark.json", payload)


def _write_eval_job_payload(
    tmp_path: Path,
    *,
    task: str = "detection",
    prompt_id: str = "grounding_arrow.v2.4.main",
    target_labels: list[str] | None = None,
    metric_profile: str | None = None,
) -> Path:
    model_path = tmp_path / "models" / "model-a"
    _write_json(model_path / "config.json", {"num_attention_heads": 4})
    payload_path = tmp_path / "job.json"
    eval_payload: dict[str, object] = {
        "model_id": "model-a",
        "benchmark_id": "bench1",
        "task": task,
        "prompt_id": prompt_id,
    }
    if target_labels is not None:
        eval_payload["target_labels"] = target_labels
    if metric_profile is not None:
        eval_payload["metric_profile"] = metric_profile
    _write_json(
        payload_path,
        {
            "manifest": {
                "kind": "eval_job",
                "runtime": {
                    "mode": "ephemeral",
                    "engine": "vllm_openai",
                    "env": {"CUDA_VISIBLE_DEVICES": "0"},
                    "args": {
                        "model": str(model_path),
                        "served-model-name": "model-a",
                        "host": "127.0.0.1",
                        "tensor-parallel-size": 1,
                        "trust-remote-code": True,
                    },
                },
                "eval": eval_payload,
            }
        },
    )
    return payload_path


def test_cli_preflights_and_creates_manifest_first_job(tmp_path: Path, capsys) -> None:
    _write_benchmark_manifest(
        tmp_path,
        tasks=["detection"],
        labels=["arrow", "icon"],
        create_split=True,
    )
    payload_path = _write_eval_job_payload(tmp_path, target_labels=["arrow"])

    preflight_args = _build_parser().parse_args(
        [
            "preflight-job",
            "--output-root",
            str(tmp_path),
            "--payload-file",
            str(payload_path),
        ]
    )
    _cmd_preflight_job(preflight_args)
    preflight = json.loads(capsys.readouterr().out)
    assert preflight["ok"] is True
    assert preflight["kind"] == "eval_job"
    assert preflight["resolved_payload"]["prompt_text"]
    assert preflight["resolved_payload"]["target_labels"] == ["arrow"]
    assert preflight["runtime_command"][0]

    create_args = _build_parser().parse_args(
        [
            "create-job",
            "--output-root",
            str(tmp_path),
            "--payload-file",
            str(payload_path),
        ]
    )
    _cmd_create_job(create_args)
    job = json.loads(capsys.readouterr().out)
    assert job["kind"] == "eval"
    assert job["status"] == "queued"
    assert job["payload"]["benchmark_id"] == "bench1"
    assert job["payload"]["target_labels"] == ["arrow"]
    assert job["payload"]["job_manifest"]["kind"] == "eval_job"
    assert "manifest" not in job["payload"]

    list_args = _build_parser().parse_args(
        [
            "list-jobs",
            "--output-root",
            str(tmp_path),
            "--kind",
            "eval",
            "--status",
            "queued",
            "--query",
            "grounding_arrow",
        ]
    )
    _cmd_list_jobs(list_args)
    jobs = json.loads(capsys.readouterr().out)
    assert jobs["total"] == 1
    assert jobs["filters"]["kind"] == "eval"
    assert jobs["facets"]["kinds"] == [{"value": "eval", "count": 1}]
    assert jobs["jobs"][0]["job_id"] == job["job_id"]

    show_job_args = _build_parser().parse_args(
        ["show-job", "--output-root", str(tmp_path), "--job-id", job["job_id"]]
    )
    _cmd_show_job(show_job_args)
    job_detail = json.loads(capsys.readouterr().out)
    assert job_detail["job"]["job_id"] == job["job_id"]
    assert job_detail["job"]["payload"]["target_labels"] == ["arrow"]
    assert job_detail["job"]["payload"]["job_manifest"]["kind"] == "eval_job"
    assert "manifest" not in job_detail["job"]["payload"]


def test_cli_preflight_failure_uses_stable_agent_shape(tmp_path: Path, capsys) -> None:
    payload_path = tmp_path / "bad-job.json"
    _write_json(payload_path, {"manifest": {"kind": "unsupported_job"}})

    args = _build_parser().parse_args(
        [
            "preflight-job",
            "--output-root",
            str(tmp_path),
            "--payload-file",
            str(payload_path),
        ]
    )
    _cmd_preflight_job(args)
    payload = json.loads(capsys.readouterr().out)

    _assert_cli_json_payload("preflight-job", payload)
    assert payload["ok"] is False
    assert payload["kind"] == ""
    assert payload["resolved_manifest"] is None
    assert payload["resolved_payload"] is None
    assert payload["runtime_command"] is None
    assert payload["errors"] == ["unsupported job kind: unsupported_job"]


def test_cli_preflight_rejects_unknown_target_label(tmp_path: Path, capsys) -> None:
    _write_benchmark_manifest(tmp_path, tasks=["detection"], labels=["arrow", "icon"])
    payload_path = _write_eval_job_payload(tmp_path, target_labels=["arrwo"])

    preflight_args = _build_parser().parse_args(
        [
            "preflight-job",
            "--output-root",
            str(tmp_path),
            "--payload-file",
            str(payload_path),
        ]
    )
    _cmd_preflight_job(preflight_args)
    preflight = json.loads(capsys.readouterr().out)

    assert preflight["ok"] is False
    assert any(
        "target_labels not found in benchmark label index: arrwo" in item
        for item in preflight["errors"]
    )


def test_cli_preflight_rejects_keypoint_label_subtasks(tmp_path: Path, capsys) -> None:
    _write_benchmark_manifest(tmp_path, tasks=["keypoint"], labels=["arrow", "icon"])
    payload_path = _write_eval_job_payload(
        tmp_path,
        task="keypoint",
        prompt_id="keypoint_arrow.test.main",
        metric_profile="keypoint_endpoint_v1",
        target_labels=["icon"],
    )

    preflight_args = _build_parser().parse_args(
        [
            "preflight-job",
            "--output-root",
            str(tmp_path),
            "--payload-file",
            str(payload_path),
        ]
    )
    _cmd_preflight_job(preflight_args)
    preflight = json.loads(capsys.readouterr().out)

    assert preflight["ok"] is False
    assert any("keypoint target_labels only support arrow" in item for item in preflight["errors"])


def test_cli_create_job_persists_preflight_warnings(tmp_path: Path, capsys) -> None:
    _write_benchmark_manifest(tmp_path, tasks=["detection"], labels=None, create_split=True)
    payload_path = _write_eval_job_payload(tmp_path, target_labels=["arrow"])

    create_args = _build_parser().parse_args(
        [
            "create-job",
            "--output-root",
            str(tmp_path),
            "--payload-file",
            str(payload_path),
        ]
    )
    _cmd_create_job(create_args)
    job = json.loads(capsys.readouterr().out)

    assert job["status"] == "queued"
    assert job["payload"]["target_labels"] == ["arrow"]
    assert any(
        "target_labels could not be preflight-validated" in item
        for item in job["metadata"]["preflight_warnings"]
    )
