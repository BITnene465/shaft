from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval_bench.cli import (
    _build_parser,
    _cmd_append_run_note,
    _cmd_get_run_note,
    _cmd_set_run_note,
)
from eval_bench.store import RunNoteConflictError
from support.cli_contracts import assert_cli_json_payload as _assert_cli_json_payload
from support.store import write_basic_run as _write_basic_run


pytestmark = pytest.mark.contract


def test_cli_gets_sets_and_appends_run_note(tmp_path: Path, capsys) -> None:
    _write_basic_run(tmp_path, run_id="run1")

    note_file = tmp_path / "note.md"
    note_file.write_text("repro: ckpt epoch_3\nidea: prompt v2", encoding="utf-8")
    set_args = _build_parser().parse_args(
        [
            "set-run-note",
            "--output-root",
            str(tmp_path),
            "--run-id",
            "run1",
            "--note-file",
            str(note_file),
        ]
    )
    _cmd_set_run_note(set_args)
    set_payload = json.loads(capsys.readouterr().out)
    _assert_cli_json_payload("set-run-note", set_payload)

    assert set_payload["note"] == "repro: ckpt epoch_3\nidea: prompt v2"
    assert set_payload["max_length"] == 20_000

    get_args = _build_parser().parse_args(
        ["get-run-note", "--output-root", str(tmp_path), "--run-id", "run1"]
    )
    _cmd_get_run_note(get_args)
    get_payload = json.loads(capsys.readouterr().out)
    _assert_cli_json_payload("get-run-note", get_payload)
    assert get_payload["note"] == set_payload["note"]
    assert get_payload["path"].endswith("runs/run1/note.json")
    assert get_payload["max_length"] == 20_000

    append_args = _build_parser().parse_args(
        [
            "append-run-note",
            "--output-root",
            str(tmp_path),
            "--run-id",
            "run1",
            "--heading",
            "follow-up",
            "--note",
            "next: inspect false positives",
        ]
    )
    _cmd_append_run_note(append_args)
    append_payload = json.loads(capsys.readouterr().out)
    _assert_cli_json_payload("append-run-note", append_payload)

    assert append_payload["note"].startswith("repro: ckpt epoch_3\nidea: prompt v2\n\n")
    assert "## follow-up\nnext: inspect false positives" in append_payload["note"]


def test_cli_run_note_uses_updated_at_guards(tmp_path: Path, capsys) -> None:
    _write_basic_run(tmp_path, run_id="run1")

    set_args = _build_parser().parse_args(
        [
            "set-run-note",
            "--output-root",
            str(tmp_path),
            "--run-id",
            "run1",
            "--note",
            "initial",
        ]
    )
    _cmd_set_run_note(set_args)
    set_payload = json.loads(capsys.readouterr().out)

    stale_append_args = _build_parser().parse_args(
        [
            "append-run-note",
            "--output-root",
            str(tmp_path),
            "--run-id",
            "run1",
            "--heading",
            "agent",
            "--note",
            "stale append",
            "--expected-updated-at",
            "2026-01-01T00:00:00Z",
        ]
    )
    with pytest.raises(RunNoteConflictError):
        _cmd_append_run_note(stale_append_args)

    guarded_append_args = _build_parser().parse_args(
        [
            "append-run-note",
            "--output-root",
            str(tmp_path),
            "--run-id",
            "run1",
            "--heading",
            "agent",
            "--note",
            "guarded append",
            "--expected-updated-at",
            set_payload["updated_at"],
        ]
    )
    _cmd_append_run_note(guarded_append_args)
    append_payload = json.loads(capsys.readouterr().out)
    _assert_cli_json_payload("append-run-note", append_payload)
    assert "## agent\nguarded append" in append_payload["note"]

    stale_set_args = _build_parser().parse_args(
        [
            "set-run-note",
            "--output-root",
            str(tmp_path),
            "--run-id",
            "run1",
            "--note",
            "stale set",
            "--expected-updated-at",
            "2026-01-01T00:00:00Z",
        ]
    )
    with pytest.raises(RunNoteConflictError):
        _cmd_set_run_note(stale_set_args)

    guarded_set_args = _build_parser().parse_args(
        [
            "set-run-note",
            "--output-root",
            str(tmp_path),
            "--run-id",
            "run1",
            "--note",
            "curated guarded note",
            "--expected-updated-at",
            append_payload["updated_at"],
        ]
    )
    _cmd_set_run_note(guarded_set_args)
    guarded_payload = json.loads(capsys.readouterr().out)
    _assert_cli_json_payload("set-run-note", guarded_payload)
    assert guarded_payload["note"] == "curated guarded note"
