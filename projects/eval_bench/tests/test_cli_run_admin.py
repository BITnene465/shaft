from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval_bench.database import EvalBenchDatabase
from eval_bench.cli import (
    _build_parser,
    _cmd_archive_run,
    _cmd_backend_logs,
    _cmd_cancel_job,
    _cmd_dashboard_state,
    _cmd_delete_job,
    _cmd_delete_run,
    _cmd_job_logs,
    _cmd_scheduler_status,
)
from support.cli_contracts import assert_cli_json_payload as _assert_cli_json_payload
from support.store import write_basic_run as _write_basic_run


pytestmark = pytest.mark.contract


def test_cli_dashboard_state_and_archive_run_emit_json(tmp_path: Path, capsys) -> None:
    _write_basic_run(tmp_path, run_id="run1")

    state_args = _build_parser().parse_args(["dashboard-state", "--output-root", str(tmp_path)])
    _cmd_dashboard_state(state_args)
    state_payload = json.loads(capsys.readouterr().out)
    _assert_cli_json_payload("dashboard-state", state_payload)
    assert state_payload["run_count"] == 1

    archive_args = _build_parser().parse_args(
        ["archive-run", "--output-root", str(tmp_path), "--run-id", "run1"]
    )
    _cmd_archive_run(archive_args)
    archived = json.loads(capsys.readouterr().out)
    _assert_cli_json_payload("archive-run", archived)
    assert archived["status"] == "archived"
    assert (
        json.loads((tmp_path / "runs" / "run1" / "run.json").read_text(encoding="utf-8"))[
            "status"
        ]
        == "archived"
    )


def test_cli_log_commands_tail_backend_and_job_logs(tmp_path: Path, capsys) -> None:
    _write_basic_run(tmp_path, run_id="run1")

    backend_log = tmp_path / "logs" / "backend.log"
    backend_log.parent.mkdir(parents=True)
    backend_log.write_text("alpha\nbeta\n", encoding="utf-8")
    backend_log_args = _build_parser().parse_args(
        ["backend-logs", "--output-root", str(tmp_path), "--max-lines", "1"]
    )
    _cmd_backend_logs(backend_log_args)
    backend_payload = json.loads(capsys.readouterr().out)
    _assert_cli_json_payload("backend-logs", backend_payload)
    assert backend_payload["lines"] == ["beta\n"]

    runtime_log = tmp_path / "runs" / "run1" / "logs" / "runtime.log"
    runtime_log.parent.mkdir(parents=True)
    runtime_log.write_text("step1\nstep2\nstep3\n", encoding="utf-8")
    EvalBenchDatabase(tmp_path).create_job(
        kind="eval",
        job_id="job1",
        payload={"run_id": "run1"},
        status="queued",
        metadata={"runtime_log_path": str(runtime_log)},
    )
    job_log_args = _build_parser().parse_args(
        ["job-logs", "--output-root", str(tmp_path), "--job-id", "job1", "--max-lines", "2"]
    )
    _cmd_job_logs(job_log_args)
    job_payload = json.loads(capsys.readouterr().out)
    _assert_cli_json_payload("job-logs", job_payload)
    assert job_payload["lines"] == ["step2\n", "step3\n"]


def test_cli_job_admin_commands_cancel_and_delete_jobs(tmp_path: Path, capsys) -> None:
    database = EvalBenchDatabase(tmp_path)
    database.create_job(kind="eval", job_id="job1", payload={"run_id": "run1"}, status="queued")

    cancel_args = _build_parser().parse_args(
        ["cancel-job", "--output-root", str(tmp_path), "--job-id", "job1"]
    )
    _cmd_cancel_job(cancel_args)
    cancelled_job = json.loads(capsys.readouterr().out)
    _assert_cli_json_payload("cancel-job", cancelled_job)
    assert cancelled_job["status"] == "cancelled"

    delete_args = _build_parser().parse_args(
        ["delete-job", "--output-root", str(tmp_path), "--job-id", "job1"]
    )
    _cmd_delete_job(delete_args)
    deleted_job = json.loads(capsys.readouterr().out)
    _assert_cli_json_payload("delete-job", deleted_job)
    assert deleted_job["job_id"] == "job1"
    assert deleted_job["deleted"] is True
    assert Path(deleted_job["trash_path"]).exists()
    assert database.get_job("job1") is None


def test_cli_scheduler_status_uses_cli_snapshot(tmp_path: Path, capsys) -> None:
    args = _build_parser().parse_args(["scheduler-status", "--output-root", str(tmp_path)])
    _cmd_scheduler_status(args)
    payload = json.loads(capsys.readouterr().out)

    _assert_cli_json_payload("scheduler-status", payload)
    assert payload["source"] == "cli_snapshot"
    assert payload["enabled"] is False


def test_cli_delete_run_moves_artifacts_to_trash(tmp_path: Path, capsys) -> None:
    _write_basic_run(tmp_path, run_id="run1")

    args = _build_parser().parse_args(
        ["delete-run", "--output-root", str(tmp_path), "--run-id", "run1"]
    )
    _cmd_delete_run(args)
    payload = json.loads(capsys.readouterr().out)

    _assert_cli_json_payload("delete-run", payload)
    assert payload["deleted"] is True
    assert not (tmp_path / "runs" / "run1").exists()
    assert Path(payload["trash_path"]).exists()
