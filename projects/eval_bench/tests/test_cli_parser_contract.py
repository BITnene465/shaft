from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from eval_bench.cli import (
    CLI_JSON_OUTPUT_SCHEMAS,
    CLI_DESTRUCTIVE_COMMANDS,
    CLI_JSON_COMMANDS,
    _command_handlers,
)
from support.cli_contracts import parser_command_names as _parser_command_names


pytestmark = pytest.mark.contract


def test_cli_parser_commands_have_handlers() -> None:
    command_names = _parser_command_names()
    handler_names = set(_command_handlers())

    assert command_names == handler_names
    assert CLI_JSON_COMMANDS <= command_names
    assert CLI_DESTRUCTIVE_COMMANDS <= CLI_JSON_COMMANDS
    assert set(CLI_JSON_OUTPUT_SCHEMAS) == CLI_JSON_COMMANDS
    assert "list-agent-commands" not in command_names
    assert "show-agent-command" not in command_names


def test_cli_help_is_the_public_discovery_surface() -> None:
    root = Path(__file__).resolve().parents[3]
    top_help = subprocess.run(
        [sys.executable, "scripts/eval_bench.py", "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    rank_help = subprocess.run(
        [sys.executable, "scripts/eval_bench.py", "rank-board", "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert top_help.returncode == 0
    assert "rank-board" in top_help.stdout
    assert "show-run" in top_help.stdout
    assert "list-agent-commands" not in top_help.stdout
    assert "show-agent-command" not in top_help.stdout
    assert "output_schema" not in top_help.stdout
    assert "contract" not in top_help.stdout.lower()

    architecture_doc = (root / "docs" / "eval_bench_architecture.md").read_text()
    assert "AGENT_COMMAND_METADATA" not in architecture_doc
    assert "AGENT_STABLE_COMMANDS" not in architecture_doc
    assert "AGENT_COMMAND_OUTPUT_SCHEMAS" not in architecture_doc
    assert "show-agent-command" not in architecture_doc
    assert "list-agent-commands" not in architecture_doc

    assert rank_help.returncode == 0
    assert "--sort-by" in rank_help.stdout
    assert "f1_iou50" in rank_help.stdout
    assert "contract" not in rank_help.stdout.lower()


def test_cli_suppresses_broken_pipe_traceback_for_json_stdout() -> None:
    result = subprocess.run(
        (f"{sys.executable} scripts/eval_bench.py rank-board | head -c 16 >/dev/null"),
        cwd=Path(__file__).resolve().parents[3],
        shell=True,
        executable="/bin/bash",
        capture_output=True,
        text=True,
        check=False,
    )
    assert "BrokenPipeError" not in result.stderr
    assert "Traceback" not in result.stderr


def test_cli_can_emit_json_errors(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/eval_bench.py",
            "--json-errors",
            "show-run",
            "--output-root",
            str(tmp_path),
            "--run-id",
            "missing-run",
        ],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert result.stdout == ""
    payload = json.loads(result.stderr)
    assert payload["ok"] is False
    assert payload["command"] == "show-run"
    assert payload["error_type"] == "FileNotFoundError"
    assert "Traceback" not in result.stderr


def test_cli_can_emit_json_parse_errors() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/eval_bench.py",
            "--json-errors",
            "show-run",
            "--run-id",
        ],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert result.stdout == ""
    payload = json.loads(result.stderr)
    assert payload == {
        "ok": False,
        "command": "show-run",
        "error_type": "ArgumentError",
        "message": "argument --run-id: expected one argument",
    }
