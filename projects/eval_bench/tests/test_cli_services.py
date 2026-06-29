from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval_bench.database import EvalBenchDatabase
from eval_bench.cli import (
    _build_parser,
    _cmd_delete_service,
    _cmd_list_services,
    _cmd_register_service,
    _cmd_show_service,
)
from support.cli_contracts import assert_cli_json_payload as _assert_cli_json_payload


pytestmark = pytest.mark.contract


def test_cli_lists_services_with_agent_filters(tmp_path: Path, capsys) -> None:
    register_args = _build_parser().parse_args(
        [
            "register-service",
            "--output-root",
            str(tmp_path),
            "--kind",
            "external_vllm",
            "--service-id",
            "external-qwen3vl",
            "--endpoint",
            "http://127.0.0.1:8000/v1",
            "--served-model-name",
            "qwen3vl-best",
        ]
    )
    _cmd_register_service(register_args)
    registered = json.loads(capsys.readouterr().out)
    _assert_cli_json_payload("register-service", registered)
    assert registered["config"]["endpoint"] == "http://127.0.0.1:8000/v1"
    assert registered["runtime"] == {}
    assert registered["metadata"] == {}

    list_args = _build_parser().parse_args(
        [
            "list-services",
            "--output-root",
            str(tmp_path),
            "--kind",
            "external_vllm",
            "--status",
            "registered",
            "--query",
            "qwen3vl",
        ]
    )
    _cmd_list_services(list_args)
    payload = json.loads(capsys.readouterr().out)
    _assert_cli_json_payload("list-services", payload)

    assert payload["total"] == 1
    assert payload["filters"]["kind"] == "external_vllm"
    assert payload["facets"]["kinds"] == [{"value": "external_vllm", "count": 1}]
    assert payload["services"][0]["service_id"] == "external-qwen3vl"
    assert payload["services"][0]["config"]["endpoint"] == "http://127.0.0.1:8000/v1"
    assert payload["services"][0]["runtime"] == {}
    assert payload["services"][0]["metadata"] == {}

    show_args = _build_parser().parse_args(
        [
            "show-service",
            "--output-root",
            str(tmp_path),
            "--service-id",
            "external-qwen3vl",
        ]
    )
    _cmd_show_service(show_args)
    detail = json.loads(capsys.readouterr().out)
    _assert_cli_json_payload("show-service", detail)
    assert detail["service"]["service_id"] == "external-qwen3vl"
    assert detail["service"]["kind"] == "external_vllm"
    assert detail["service"]["config"]["endpoint"] == "http://127.0.0.1:8000/v1"
    assert detail["service"]["runtime"] == {}
    assert detail["service"]["metadata"] == {}


def test_cli_deletes_registered_service(tmp_path: Path, capsys) -> None:
    register_args = _build_parser().parse_args(
        [
            "register-service",
            "--output-root",
            str(tmp_path),
            "--kind",
            "external_vllm",
            "--service-id",
            "external-qwen3vl",
            "--endpoint",
            "http://127.0.0.1:8000/v1",
        ]
    )
    _cmd_register_service(register_args)
    capsys.readouterr()

    delete_args = _build_parser().parse_args(
        ["delete-service", "--output-root", str(tmp_path), "--service-id", "external-qwen3vl"]
    )
    _cmd_delete_service(delete_args)
    payload = json.loads(capsys.readouterr().out)

    _assert_cli_json_payload("delete-service", payload)
    assert payload["service"]["service_id"] == "external-qwen3vl"
    assert EvalBenchDatabase(tmp_path).get_service("external-qwen3vl") is None
