from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from eval_bench.database import EvalBenchDatabase
from eval_bench.cli import (
    AGENT_COMMAND_METADATA,
    AGENT_COMMAND_OUTPUT_SCHEMAS,
    AGENT_DESTRUCTIVE_COMMANDS,
    AGENT_STABLE_COMMANDS,
    _build_parser,
    _command_handlers,
    _cmd_archive_run,
    _cmd_backend_logs,
    _cmd_cancel_job,
    _cmd_create_job,
    _cmd_dashboard_state,
    _cmd_delete_job,
    _cmd_delete_run,
    _cmd_delete_service,
    _cmd_evaluate_run,
    _cmd_get_run_note,
    _cmd_init_run,
    _cmd_import_predictions,
    _cmd_list_agent_commands,
    _cmd_job_logs,
    _cmd_list_benchmarks,
    _cmd_list_benchmark_samples,
    _cmd_list_comparisons,
    _cmd_list_job_templates,
    _cmd_list_jobs,
    _cmd_list_prompt_templates,
    _cmd_list_run_samples,
    _cmd_list_runs,
    _cmd_list_services,
    _cmd_preflight_job,
    _cmd_rank_board,
    _cmd_register_service,
    _cmd_resolve_target_labels,
    _cmd_scheduler_status,
    _cmd_delete_prompt_template,
    _cmd_set_run_note,
    _cmd_append_run_note,
    _cmd_compare_runs,
    _cmd_show_benchmark,
    _cmd_show_benchmark_sample,
    _cmd_show_agent_command,
    _cmd_show_comparison,
    _cmd_show_comparison_sample,
    _cmd_show_job,
    _cmd_show_job_template,
    _cmd_show_prompt_template,
    _cmd_show_run,
    _cmd_show_run_report,
    _cmd_show_run_sample,
    _cmd_show_service,
    _cmd_upsert_prompt_template,
)
from eval_bench.store import RunNoteConflictError


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _parser_command_names() -> set[str]:
    subparsers_action = next(
        action for action in _build_parser()._actions if action.dest == "command"
    )
    return set(subparsers_action.choices)


def _assert_agent_output_payload(command_name: str, payload: object) -> None:
    schema = AGENT_COMMAND_OUTPUT_SCHEMAS[command_name]
    _assert_schema_node(schema, payload, command_name)


def _assert_schema_node(schema: object, value: object, path: str) -> None:
    if isinstance(schema, str):
        _assert_schema_type(schema, value, path)
        return
    assert isinstance(schema, dict), f"{path}: schema must be a string or object"
    schema_type = schema.get("type")
    if isinstance(schema_type, str):
        _assert_schema_type(schema_type, value, path)
    if schema.get("required"):
        assert isinstance(value, dict), f"{path}: required fields need object payload"
        for key in schema["required"]:
            assert key in value, f"{path}: missing required field {key}"
    properties = schema.get("properties")
    if isinstance(properties, dict) and isinstance(value, dict):
        for key, child_schema in properties.items():
            if key in value:
                _assert_schema_node(child_schema, value[key], f"{path}.{key}")
    item_shape = schema.get("item_shape")
    if item_shape is None:
        return
    if schema_type == "array":
        assert isinstance(value, list), f"{path}: expected array payload"
        if value:
            _assert_schema_node(
                {"type": "object", "properties": item_shape},
                value[0],
                f"{path}[0]",
            )
    elif schema_type in {"object", "object|null"} and value is not None:
        assert isinstance(value, dict), f"{path}: expected object payload"
        _assert_schema_node({"type": "object", "properties": item_shape}, value, path)


def _assert_schema_type(schema_type: str, value: object, path: str) -> None:
    if schema_type.endswith("|null") and value is None:
        return
    if schema_type.startswith("list["):
        assert isinstance(value, list), f"{path}: expected {schema_type}"
        return
    if schema_type == "array":
        assert isinstance(value, list), f"{path}: expected array"
        return
    if schema_type in {"object", "dict"}:
        assert isinstance(value, dict), f"{path}: expected object"
        return
    if schema_type == "object|null":
        assert value is None or isinstance(value, dict), f"{path}: expected object|null"
        return
    if schema_type == "str":
        assert isinstance(value, str), f"{path}: expected str"
        return
    if schema_type == "str|null":
        assert value is None or isinstance(value, str), f"{path}: expected str|null"
        return
    if schema_type == "int":
        assert isinstance(value, int) and not isinstance(value, bool), f"{path}: expected int"
        return
    if schema_type == "float":
        assert isinstance(value, (int, float)) and not isinstance(
            value, bool
        ), f"{path}: expected float"
        return
    if schema_type == "float|null":
        assert value is None or (
            isinstance(value, (int, float)) and not isinstance(value, bool)
        ), f"{path}: expected float|null"
        return
    if schema_type == "bool":
        assert isinstance(value, bool), f"{path}: expected bool"


def test_cli_parser_commands_have_handlers_for_agent_contract() -> None:
    command_names = _parser_command_names()
    handler_names = set(_command_handlers())

    assert command_names == handler_names
    assert AGENT_STABLE_COMMANDS <= command_names
    assert set(AGENT_COMMAND_METADATA) == AGENT_STABLE_COMMANDS
    assert AGENT_DESTRUCTIVE_COMMANDS <= AGENT_STABLE_COMMANDS
    assert all(
        isinstance(item["domain"], str) and item["domain"]
        for item in AGENT_COMMAND_METADATA.values()
    )
    assert all(
        isinstance(item["mutates_state"], bool) for item in AGENT_COMMAND_METADATA.values()
    )
    assert all(
        bool(AGENT_COMMAND_METADATA[name]["mutates_state"])
        for name in AGENT_DESTRUCTIVE_COMMANDS
    )


def test_cli_lists_agent_stable_commands(capsys) -> None:
    args = _build_parser().parse_args(["list-agent-commands"])
    _cmd_list_agent_commands(args)
    payload = json.loads(capsys.readouterr().out)
    _assert_agent_output_payload("list-agent-commands", payload)

    command_names = [item["name"] for item in payload["commands"]]
    commands_by_name = {item["name"]: item for item in payload["commands"]}
    assert payload["total"] == len(AGENT_STABLE_COMMANDS)
    assert payload["mutating_count"] == sum(
        1 for item in AGENT_COMMAND_METADATA.values() if item["mutates_state"]
    )
    assert payload["read_only_count"] == sum(
        1 for item in AGENT_COMMAND_METADATA.values() if not item["mutates_state"]
    )
    assert payload["destructive_count"] == len(AGENT_DESTRUCTIVE_COMMANDS)
    assert set(payload["domains"]) == {
        item["domain"] for item in AGENT_COMMAND_METADATA.values()
    }
    assert payload["recommended_runner"] == [".venv/bin/python", "scripts/eval_bench.py"]
    assert command_names == sorted(AGENT_STABLE_COMMANDS)
    assert "rank-board" in command_names
    assert "show-agent-command" in command_names
    assert "init-run" in command_names
    assert "validate-prediction" in command_names
    assert "process-next-job" in command_names
    assert "register-service" in command_names
    assert "start-service" in command_names
    assert "stop-service" in command_names
    assert "compare-runs" in command_names
    assert "import-predictions" in command_names
    assert "upsert-prompt-template" in command_names
    assert "show-comparison-sample" in command_names
    assert "serve-dashboard" not in command_names
    assert "write-demo-prediction" not in command_names
    assert all(item["help"] for item in payload["commands"])
    assert all(item["usage"].startswith("usage: eval_bench.py ") for item in payload["commands"])
    assert all(
        item["argv_prefix"] == ["scripts/eval_bench.py", item["name"]]
        for item in payload["commands"]
    )
    assert all(item["domain"] for item in payload["commands"])
    assert all(isinstance(item["mutates_state"], bool) for item in payload["commands"])
    assert all(isinstance(item["destructive"], bool) for item in payload["commands"])
    assert all(isinstance(item["arguments"], list) for item in payload["commands"])
    assert all(isinstance(item["argument_semantics"], dict) for item in payload["commands"])
    assert all(isinstance(item["mutually_exclusive_groups"], list) for item in payload["commands"])
    assert all(isinstance(item["output_schema"], dict) for item in payload["commands"])
    assert set(AGENT_COMMAND_OUTPUT_SCHEMAS) == AGENT_STABLE_COMMANDS
    assert all(item["output_schema"] for item in payload["commands"])
    assert commands_by_name["dashboard-state"]["output_schema"]["properties"]["runs"]["item_shape"][
        "target_labels"
    ] == "list[str]"
    assert (
        commands_by_name["scheduler-status"]["output_schema"]["properties"]["enabled"]
        == "bool"
    )
    assert commands_by_name["backend-logs"]["output_schema"]["properties"]["lines"] == "list[str]"
    assert commands_by_name["job-logs"]["output_schema"]["properties"]["job_id"] == "str"
    assert commands_by_name["service-logs"]["output_schema"]["properties"]["service_id"] == "str"
    assert commands_by_name["list-agent-commands"]["output_schema"]["properties"]["commands"][
        "item_shape"
    ]["output_schema"] == "object"
    assert commands_by_name["show-agent-command"]["output_schema"]["properties"]["command"][
        "item_shape"
    ]["arguments"] == "list[object]"
    assert commands_by_name["show-agent-command"]["output_schema"]["properties"]["command"][
        "item_shape"
    ]["argument_semantics"] == "object"
    assert commands_by_name["create-benchmark"]["output_schema"]["properties"]["labels"] == "list[str]"
    init_output_schema = commands_by_name["init-run"]["output_schema"]
    assert init_output_schema["required"] == [
        "run_id",
        "manifest_path",
        "artifact_root",
        "task",
        "benchmark_id",
        "target_labels",
        "target_labels_source",
    ]
    assert init_output_schema["properties"]["target_labels"]["type"] == "list[str]"
    assert commands_by_name["validate-prediction"]["output_schema"]["properties"]["instances"] == "int"
    assert commands_by_name["rank-board"]["domain"] == "rank"
    assert commands_by_name["rank-board"]["mutates_state"] is False
    assert "rank-board" in commands_by_name["rank-board"]["usage"]
    assert "pytest" not in commands_by_name["rank-board"]["usage"]
    assert "rank-scheme-json" in commands_by_name["rank-board"]["usage"]
    rank_output_schema = commands_by_name["rank-board"]["output_schema"]
    assert rank_output_schema["required"] == [
        "offset",
        "limit",
        "total",
        "evaluated_count",
        "filters",
        "primary_metric",
        "primary_metric_label",
        "sort_by",
        "sort_order",
        "score_formula",
        "rank_scheme",
        "facets",
        "entries",
    ]
    assert rank_output_schema["properties"]["facets"]["keys"] == [
        "tasks",
        "benchmarks",
        "statuses",
        "labels",
        "models",
        "prompts",
        "metric_profiles",
    ]
    assert "score_delta" in rank_output_schema["properties"]["entries"]["item_shape"]
    assert commands_by_name["create-job"]["domain"] == "job"
    assert commands_by_name["create-job"]["mutates_state"] is True
    assert commands_by_name["init-run"]["domain"] == "run"
    assert commands_by_name["init-run"]["mutates_state"] is True
    assert commands_by_name["validate-prediction"]["domain"] == "prediction"
    assert commands_by_name["validate-prediction"]["mutates_state"] is False
    assert commands_by_name["process-next-job"]["domain"] == "job"
    assert commands_by_name["process-next-job"]["mutates_state"] is True
    assert commands_by_name["start-service"]["domain"] == "service"
    assert commands_by_name["start-service"]["mutates_state"] is True
    assert commands_by_name["start-service"]["destructive"] is False
    assert commands_by_name["stop-service"]["destructive"] is True
    assert commands_by_name["delete-run"]["destructive"] is True
    assert commands_by_name["set-run-note"]["destructive"] is False
    assert commands_by_name["append-run-note"]["domain"] == "note"
    assert commands_by_name["append-run-note"]["mutates_state"] is True
    assert commands_by_name["append-run-note"]["destructive"] is False
    note_output_schema = commands_by_name["get-run-note"]["output_schema"]
    assert note_output_schema["required"] == [
        "run_id",
        "note",
        "updated_at",
        "path",
        "max_length",
    ]
    assert commands_by_name["set-run-note"]["output_schema"] == note_output_schema
    assert commands_by_name["append-run-note"]["output_schema"] == note_output_schema
    resolve_output_schema = commands_by_name["resolve-target-labels"]["output_schema"]
    assert resolve_output_schema["required"] == [
        "task",
        "benchmark_id",
        "prompt_id",
        "target_labels",
        "target_labels_source",
        "candidate_labels",
        "benchmark_labels",
        "prompt_target_labels",
        "explicit_target_labels",
        "label_subtasks_supported",
        "valid",
        "errors",
        "warnings",
    ]
    assert "detection" in resolve_output_schema["properties"]["label_subtasks_supported"]["description"]
    assert "keypoint is fixed to arrow" in resolve_output_schema["properties"]["label_subtasks_supported"]["description"]
    runs_output_schema = commands_by_name["list-runs"]["output_schema"]
    assert runs_output_schema["required"] == ["offset", "limit", "total", "filters", "runs"]
    assert runs_output_schema["properties"]["runs"]["item_shape"]["target_labels"] == "list[str]"
    assert runs_output_schema["properties"]["runs"]["item_shape"]["note_updated_at"] == "str|null"
    assert runs_output_schema["properties"]["runs"]["item_shape"]["f1_iou50"] == "float|null"
    assert commands_by_name["show-run"]["output_schema"]["properties"]["run"]["item_shape"] == (
        runs_output_schema["properties"]["runs"]["item_shape"]
    )
    run_samples_output_schema = commands_by_name["list-run-samples"]["output_schema"]
    assert run_samples_output_schema["required"] == [
        "run_id",
        "offset",
        "limit",
        "total",
        "labels",
        "samples",
    ]
    assert run_samples_output_schema["properties"]["samples"]["item_shape"]["gt_instance_count"] == "int"
    assert run_samples_output_schema["properties"]["samples"]["item_shape"]["diagnostics"] == "object|null"
    assert (
        commands_by_name["show-run-sample"]["output_schema"]["properties"]["sample"]["item_shape"]
        == run_samples_output_schema["properties"]["samples"]["item_shape"]
    )
    benchmark_samples_output_schema = commands_by_name["list-benchmark-samples"]["output_schema"]
    assert benchmark_samples_output_schema["properties"]["samples"]["item_shape"]["instance_count"] == "int"
    assert (
        commands_by_name["show-benchmark-sample"]["output_schema"]["properties"]["sample"][
            "item_shape"
        ]
        == benchmark_samples_output_schema["properties"]["samples"]["item_shape"]
    )
    jobs_output_schema = commands_by_name["list-jobs"]["output_schema"]
    assert jobs_output_schema["required"] == ["offset", "limit", "total", "filters", "jobs"]
    assert jobs_output_schema["properties"]["jobs"]["item_shape"]["payload"] == "object"
    assert jobs_output_schema["properties"]["jobs"]["item_shape"]["metadata"] == "object"
    assert commands_by_name["show-job"]["output_schema"]["properties"]["job"]["item_shape"] == (
        jobs_output_schema["properties"]["jobs"]["item_shape"]
    )
    assert (
        commands_by_name["list-job-templates"]["output_schema"]["properties"]["templates"][
            "item_shape"
        ]["manifest"]
        == "object"
    )
    assert (
        commands_by_name["show-job-template"]["output_schema"]["properties"]["template"][
            "item_shape"
        ]["description"]
        == "str"
    )
    prompt_templates_output_schema = commands_by_name["list-prompt-templates"]["output_schema"]
    assert prompt_templates_output_schema["properties"]["templates"]["item_shape"]["metadata"] == "object"
    assert prompt_templates_output_schema["properties"]["by_id"]["item_shape"]["prompt_id"] == "str"
    assert (
        commands_by_name["show-prompt-template"]["output_schema"]["properties"]["template"][
            "item_shape"
        ]
        == prompt_templates_output_schema["properties"]["templates"]["item_shape"]
    )
    assert commands_by_name["upsert-prompt-template"]["output_schema"]["properties"]["prompt_id"] == "str"
    assert (
        commands_by_name["delete-prompt-template"]["output_schema"]["properties"]["deleted"][
            "type"
        ]
        == "bool"
    )
    assert commands_by_name["preflight-job"]["output_schema"]["properties"]["runtime_command"] == "list[str]"
    assert commands_by_name["create-job"]["output_schema"]["properties"]["payload"] == "object"
    assert commands_by_name["cancel-job"]["output_schema"]["properties"]["status"] == "str"
    assert commands_by_name["delete-job"]["output_schema"]["properties"]["deleted"]["type"] == "bool"
    assert commands_by_name["process-next-job"]["output_schema"]["properties"]["job"][
        "item_shape"
    ]["job_id"] == "str"
    services_output_schema = commands_by_name["list-services"]["output_schema"]
    assert services_output_schema["properties"]["services"]["item_shape"]["config"] == "object"
    assert services_output_schema["properties"]["services"]["item_shape"]["runtime"] == "object"
    assert (
        commands_by_name["show-service"]["output_schema"]["properties"]["service"]["item_shape"]
        == services_output_schema["properties"]["services"]["item_shape"]
    )
    assert commands_by_name["register-service"]["output_schema"]["properties"]["runtime"] == "object"
    assert commands_by_name["service-command"]["output_schema"]["properties"]["command"]["type"] == "list[str]"
    assert commands_by_name["start-service"]["output_schema"]["properties"]["service_id"] == "str"
    assert commands_by_name["service-health"]["output_schema"]["properties"]["error"] == "str|null"
    assert commands_by_name["stop-service"]["output_schema"]["properties"]["status"] == "str"
    assert (
        commands_by_name["delete-service"]["output_schema"]["properties"]["service"][
            "item_shape"
        ]["service_id"]
        == "str"
    )
    assert commands_by_name["archive-run"]["output_schema"]["properties"]["manifest_path"] == "str"
    assert commands_by_name["delete-run"]["output_schema"]["properties"]["trash_path"] == "str|null"
    assert commands_by_name["evaluate-run"]["output_schema"]["required"] == [
        "run_id",
        "report_path",
        "summary_path",
    ]
    assert (
        commands_by_name["import-predictions"]["output_schema"]["properties"][
            "missing_prediction_count"
        ]
        == "int"
    )
    assert commands_by_name["compare-runs"]["output_schema"]["required"] == [
        "comparison_id",
        "baseline_run_id",
        "candidate_run_id",
        "report_path",
    ]
    assert commands_by_name["show-run-report"]["output_schema"]["type"] == "object"
    comparisons_output_schema = commands_by_name["list-comparisons"]["output_schema"]
    assert comparisons_output_schema["properties"]["comparisons"]["item_shape"]["target_labels"] == "list[str]"
    assert comparisons_output_schema["properties"]["comparisons"]["item_shape"]["delta"] == "object"
    assert commands_by_name["show-comparison"]["output_schema"]["properties"]["summary"] == "object"
    comparison_sample_output_schema = commands_by_name["show-comparison-sample"]["output_schema"]
    assert comparison_sample_output_schema["properties"]["baseline"]["item_shape"]["pred_instances"] == "list[object]"
    assert comparison_sample_output_schema["properties"]["candidate"]["item_shape"]["diagnostics"] == "object|null"
    assert commands_by_name["service-health"]["mutates_state"] is True
    assert commands_by_name["compare-runs"]["domain"] == "comparison"
    assert commands_by_name["compare-runs"]["mutates_state"] is True
    assert commands_by_name["list-agent-commands"]["domain"] == "meta"
    assert commands_by_name["list-agent-commands"]["arguments"] == []
    assert commands_by_name["show-agent-command"]["domain"] == "meta"
    assert commands_by_name["show-agent-command"]["mutates_state"] is False
    show_agent_args = {
        item["dest"]: item for item in commands_by_name["show-agent-command"]["arguments"]
    }
    assert show_agent_args["name"]["required"] is True
    assert "rank-board" in show_agent_args["name"]["choices"]
    set_note_args = {
        item["dest"]: item for item in commands_by_name["set-run-note"]["arguments"]
    }
    assert set_note_args["expected_updated_at"]["flags"] == ["--expected-updated-at"]

    create_benchmark_args = {
        item["dest"]: item for item in commands_by_name["create-benchmark"]["arguments"]
    }
    assert create_benchmark_args["benchmark_id"]["flags"] == ["--benchmark-id"]
    assert create_benchmark_args["benchmark_id"]["required"] is True
    assert create_benchmark_args["task"]["action"] == "append"
    assert create_benchmark_args["task"]["repeatable"] is True
    assert create_benchmark_args["task"]["choices"] == ["detection", "keypoint"]
    assert create_benchmark_args["overwrite"]["type"] == "bool"
    assert create_benchmark_args["overwrite"]["action"] == "store_true"

    init_run_args = {item["dest"]: item for item in commands_by_name["init-run"]["arguments"]}
    assert init_run_args["task"]["choices"] == ["detection", "keypoint"]
    assert init_run_args["benchmark_id"]["required"] is True
    assert init_run_args["target_labels"]["action"] == "append"
    assert init_run_args["target_labels"]["repeatable"] is True
    assert "Detection label subtask scope" in init_run_args["target_labels"]["help"]
    assert "Keypoint runs are fixed to arrow" in init_run_args["target_labels"]["help"]
    target_label_semantics = commands_by_name["resolve-target-labels"]["argument_semantics"][
        "target_labels"
    ]
    assert target_label_semantics["task_scope"]["detection"][
        "label_subtasks_supported"
    ] is True
    assert target_label_semantics["task_scope"]["detection"]["repeatable"] is True
    assert target_label_semantics["task_scope"]["detection"]["empty_uses_label_policy"] is True
    assert (
        target_label_semantics["task_scope"]["keypoint"]["fixed_target_labels"]
        == ["arrow"]
    )
    assert target_label_semantics["task_scope"]["keypoint"]["rejects_non_arrow"] is True
    assert (
        target_label_semantics["source_priority"][0]
        == "explicit_target_labels"
    )
    assert target_label_semantics["recommended_discovery_command"] == "resolve-target-labels"
    for command_name in ["init-run", "import-predictions", "resolve-target-labels"]:
        assert (
            commands_by_name[command_name]["argument_semantics"]["target_labels"]
            == target_label_semantics
        )
    assert init_run_args["max_tokens"]["type"] == "int"
    assert init_run_args["batch_size"]["default"] == 1

    validate_args = {
        item["dest"]: item for item in commands_by_name["validate-prediction"]["arguments"]
    }
    assert validate_args["path"]["positional"] is True
    assert validate_args["path"]["required"] is True
    assert validate_args["task"]["choices"] == ["detection", "keypoint"]

    process_args = {
        item["dest"]: item for item in commands_by_name["process-next-job"]["arguments"]
    }
    assert process_args["kind"]["default"] == "eval"

    preflight_args = {item["dest"]: item for item in commands_by_name["preflight-job"]["arguments"]}
    assert preflight_args["payload_json"]["required"] is False
    assert preflight_args["payload_file"]["required"] is False
    assert {
        tuple(group["arguments"]): group["required"]
        for group in commands_by_name["preflight-job"]["mutually_exclusive_groups"]
    } == {("payload_json", "payload_file"): True}

    rank_args = {item["dest"]: item for item in commands_by_name["rank-board"]["arguments"]}
    assert rank_args["sort_by"]["default"] == "f1_iou50"
    assert "weighted_score" in rank_args["sort_by"]["choices"]
    rank_sort_semantics = commands_by_name["rank-board"]["argument_semantics"]["sort_by"]
    assert rank_sort_semantics["primary_metrics"] == [
        "f1_iou50",
        "precision_iou50",
        "recall_iou50",
        "mean_iou",
        "prediction_count",
    ]
    assert rank_sort_semantics["auxiliary_sorts"] == ["created_at", "run_id"]
    assert rank_sort_semantics["weighted_sort"] == "weighted_score"
    assert rank_sort_semantics["auxiliary_sort_keeps_primary_metric"] == "f1_iou50"
    assert rank_args["min_score"]["type"] == "float"
    assert {
        tuple(group["arguments"]): group["required"]
        for group in commands_by_name["rank-board"]["mutually_exclusive_groups"]
    } == {("rank_scheme_json", "rank_scheme_file"): False}

    import_args = {item["dest"]: item for item in commands_by_name["import-predictions"]["arguments"]}
    assert import_args["target_labels"]["action"] == "append"
    assert import_args["target_labels"]["repeatable"] is True
    assert "Detection label subtask scope" in import_args["target_labels"]["help"]
    assert "Keypoint runs are fixed to arrow" in import_args["target_labels"]["help"]
    assert import_args["skip_evaluate"]["action"] == "store_true"

    resolve_args = {
        item["dest"]: item for item in commands_by_name["resolve-target-labels"]["arguments"]
    }
    assert "Detection label subtask scope" in resolve_args["target_labels"]["help"]
    assert "Keypoint runs are fixed to arrow" in resolve_args["target_labels"]["help"]


def test_cli_shows_single_agent_command_contract(capsys) -> None:
    args = _build_parser().parse_args(["show-agent-command", "--name", "rank-board"])
    _cmd_show_agent_command(args)
    payload = json.loads(capsys.readouterr().out)
    _assert_agent_output_payload("show-agent-command", payload)

    assert payload["recommended_runner"] == [".venv/bin/python", "scripts/eval_bench.py"]
    command = payload["command"]
    assert command["name"] == "rank-board"
    assert command["domain"] == "rank"
    assert command["mutates_state"] is False
    assert command["destructive"] is False
    assert command["argv_prefix"] == ["scripts/eval_bench.py", "rank-board"]
    assert "rank-board" in command["usage"]
    assert command["argument_semantics"]["sort_by"]["primary_metrics"][0] == "f1_iou50"
    assert command["argument_semantics"]["sort_by"]["auxiliary_sorts"] == [
        "created_at",
        "run_id",
    ]
    assert (
        command["argument_semantics"]["sort_by"]["weighted_sort_requires"]
        == ["--rank-scheme-json", "--rank-scheme-file"]
    )
    assert command["output_schema"]["properties"]["facets"]["item_shape"] == {
        "value": "str",
        "count": "int",
    }
    assert command["output_schema"]["properties"]["entries"]["item_shape"]["score"] == "float|null"
    args_by_dest = {item["dest"]: item for item in command["arguments"]}
    assert args_by_dest["sort_by"]["default"] == "f1_iou50"
    assert "weighted_score" in args_by_dest["sort_by"]["choices"]
    assert {
        tuple(group["arguments"]): group["required"]
        for group in command["mutually_exclusive_groups"]
    } == {("rank_scheme_json", "rank_scheme_file"): False}

    label_args = _build_parser().parse_args(["show-agent-command", "--name", "resolve-target-labels"])
    _cmd_show_agent_command(label_args)
    label_command = json.loads(capsys.readouterr().out)["command"]
    label_semantics = label_command["argument_semantics"]["target_labels"]
    assert label_semantics["task_scope"]["detection"]["label_subtasks_supported"] is True
    assert label_semantics["task_scope"]["keypoint"]["fixed_target_labels"] == ["arrow"]
    assert label_semantics["recommended_discovery_command"] == "resolve-target-labels"
    assert label_command["output_schema"]["properties"]["target_labels"]["type"] == "list[str]"
    assert (
        label_command["output_schema"]["properties"]["label_subtasks_supported"]["description"]
        == "true only for detection; keypoint is fixed to arrow."
    )

    note_args = _build_parser().parse_args(["show-agent-command", "--name", "get-run-note"])
    _cmd_show_agent_command(note_args)
    note_command = json.loads(capsys.readouterr().out)["command"]
    assert note_command["output_schema"]["properties"]["updated_at"]["type"] == "str|null"
    assert note_command["output_schema"]["properties"]["max_length"]["type"] == "int"

    list_runs_args = _build_parser().parse_args(["show-agent-command", "--name", "list-runs"])
    _cmd_show_agent_command(list_runs_args)
    list_runs_command = json.loads(capsys.readouterr().out)["command"]
    assert list_runs_command["output_schema"]["properties"]["runs"]["item_shape"]["run_id"] == "str"
    assert (
        list_runs_command["output_schema"]["properties"]["runs"]["item_shape"]["note_max_length"]
        == "int"
    )

    sample_args = _build_parser().parse_args(["show-agent-command", "--name", "show-run-sample"])
    _cmd_show_agent_command(sample_args)
    sample_command = json.loads(capsys.readouterr().out)["command"]
    assert sample_command["output_schema"]["properties"]["sample"]["item_shape"]["labels"] == "list[str]"
    assert sample_command["output_schema"]["properties"]["prediction_payload"]["type"] == "object|null"

    job_args = _build_parser().parse_args(["show-agent-command", "--name", "list-jobs"])
    _cmd_show_agent_command(job_args)
    job_command = json.loads(capsys.readouterr().out)["command"]
    assert job_command["output_schema"]["properties"]["jobs"]["item_shape"]["job_id"] == "str"
    assert job_command["output_schema"]["properties"]["jobs"]["item_shape"]["error"] == "str|null"

    job_template_args = _build_parser().parse_args(
        ["show-agent-command", "--name", "show-job-template"]
    )
    _cmd_show_agent_command(job_template_args)
    job_template_command = json.loads(capsys.readouterr().out)["command"]
    assert (
        job_template_command["output_schema"]["properties"]["template"]["item_shape"]["manifest"]
        == "object"
    )

    prompt_templates_args = _build_parser().parse_args(
        ["show-agent-command", "--name", "list-prompt-templates"]
    )
    _cmd_show_agent_command(prompt_templates_args)
    prompt_templates_command = json.loads(capsys.readouterr().out)["command"]
    assert (
        prompt_templates_command["output_schema"]["properties"]["by_id"]["item_shape"]["task"]
        == "str"
    )

    preflight_args = _build_parser().parse_args(["show-agent-command", "--name", "preflight-job"])
    _cmd_show_agent_command(preflight_args)
    preflight_command = json.loads(capsys.readouterr().out)["command"]
    assert preflight_command["output_schema"]["properties"]["runtime_command"] == "list[str]"

    create_job_args = _build_parser().parse_args(["show-agent-command", "--name", "create-job"])
    _cmd_show_agent_command(create_job_args)
    create_job_command = json.loads(capsys.readouterr().out)["command"]
    assert create_job_command["output_schema"]["properties"]["job_id"] == "str"

    service_args = _build_parser().parse_args(["show-agent-command", "--name", "show-service"])
    _cmd_show_agent_command(service_args)
    service_command = json.loads(capsys.readouterr().out)["command"]
    assert service_command["output_schema"]["properties"]["service"]["item_shape"]["service_id"] == "str"
    assert service_command["output_schema"]["properties"]["service"]["item_shape"]["runtime"] == "object"

    comparison_args = _build_parser().parse_args(["show-agent-command", "--name", "show-comparison-sample"])
    _cmd_show_agent_command(comparison_args)
    comparison_command = json.loads(capsys.readouterr().out)["command"]
    assert comparison_command["output_schema"]["properties"]["baseline"]["item_shape"]["run_id"] == "str"
    assert (
        comparison_command["output_schema"]["properties"]["candidate"]["item_shape"][
            "prediction_payload"
        ]
        == "object|null"
    )


def _write_sample_store(tmp_path: Path) -> None:
    split_path = tmp_path / "benchmarks" / "bench1" / "splits" / "val.txt"
    data_root = tmp_path / "benchmarks" / "bench1" / "data"
    split_path.parent.mkdir(parents=True)
    split_path.write_text("part1/json/a.json\npart1/json/b.json\n", encoding="utf-8")
    _write_json(
        tmp_path / "benchmarks" / "bench1" / "benchmark.json",
        {
            "benchmark_id": "bench1",
            "tasks": ["detection"],
            "layers": ["layout", "arrow"],
            "split": "val",
            "sample_count": 2,
            "root": str(data_root),
            "manifest_path": str(split_path),
            "labels": ["arrow", "icon"],
        },
    )
    _write_json(
        data_root / "part1" / "json" / "a.json",
        {
            "image_path": "part1/images/a.png",
            "image_width": 100,
            "image_height": 100,
            "instances": [
                {"label": "icon", "bbox": [0, 0, 40, 40]},
                {"label": "arrow", "bbox": [50, 50, 90, 90]},
            ],
        },
    )
    _write_json(
        data_root / "part1" / "json" / "b.json",
        {
            "image_path": "part1/images/b.png",
            "instances": [{"label": "icon", "bbox": [0, 0, 30, 30]}],
        },
    )
    run_dir = tmp_path / "runs" / "run_arrow"
    _write_json(
        run_dir / "run.json",
        {
            "run_id": "run_arrow",
            "status": "succeeded",
            "created_at": "2026-05-09T00:10:00Z",
            "model": {"model_id": "model-a", "path": "outputs/model-a/best"},
            "benchmark": {
                "benchmark_id": "bench1",
                "root": str(data_root),
                "split": "val",
                "tasks": ["detection"],
                "manifest_path": str(split_path),
            },
            "spec": {
                "task": "detection",
                "metric_profile": "detection_iou_v1",
                "target_labels": ["arrow"],
                "metadata": {"target_labels_source": "explicit"},
                "prompt": {"prompt_id": "grounding_arrow.latest"},
            },
        },
    )
    _write_json(
        run_dir / "predictions" / "part1" / "json" / "a.json",
        {
            "image": "part1/images/a.png",
            "instances": [
                {"label": "icon", "bbox": [0, 0, 40, 40]},
                {"label": "arrow", "bbox": [52, 52, 88, 88]},
            ],
        },
    )
    _write_json(
        run_dir / "reports" / "summary.json",
        {
            "run_id": "run_arrow",
            "task": "detection",
            "metric_profile": "detection_iou_v1",
            "target_labels": ["arrow"],
            "target_labels_source": "explicit",
            "prediction_file_count": 1,
            "precision_iou50": 1.0,
            "recall_iou50": 1.0,
            "mean_iou": 0.81,
            "labels": ["arrow"],
        },
    )
    _write_json(
        run_dir / "reports" / "metrics.json",
        {
            "run_id": "run_arrow",
            "task": "detection",
            "metric_profile": "detection_iou_v1",
            "target_labels": ["arrow"],
            "target_labels_source": "explicit",
            "sample_count": 2,
            "prediction_file_count": 1,
            "precision_iou50": 1.0,
            "recall_iou50": 1.0,
            "mean_iou": 0.81,
            "labels": [
                {
                    "label": "arrow",
                    "gt_count": 1,
                    "pred_count": 1,
                    "matched_count": 1,
                    "false_positive_count": 0,
                    "false_negative_count": 0,
                    "precision_iou50": 1.0,
                    "recall_iou50": 1.0,
                    "mean_iou": 0.81,
                }
            ],
            "samples": [
                {
                    "index": 0,
                    "image": "part1/images/a.png",
                    "gt_instance_count": 1,
                    "pred_instance_count": 1,
                    "matched_count": 1,
                    "false_negative_count": 0,
                    "false_positive_count": 0,
                    "mean_iou": 0.81,
                    "labels": {
                        "arrow": {
                            "gt_count": 1,
                            "pred_count": 1,
                            "matched_count": 1,
                            "false_negative_count": 0,
                            "false_positive_count": 0,
                            "mean_iou": 0.81,
                        }
                    },
                    "matches": [{"label": "arrow", "gt_index": 0, "pred_index": 0, "iou": 0.81}],
                    "false_negatives": [],
                    "false_positives": [],
                },
                {
                    "index": 1,
                    "image": "part1/images/b.png",
                    "gt_instance_count": 0,
                    "pred_instance_count": 0,
                    "matched_count": 0,
                    "false_negative_count": 0,
                    "false_positive_count": 0,
                    "mean_iou": 0.0,
                    "labels": {},
                    "matches": [],
                    "false_negatives": [],
                    "false_positives": [],
                },
            ],
        },
    )


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
            "grounding_layout.latest",
            "--target-label",
            "icon",
            "--target-label",
            "image",
        ]
    )

    _cmd_init_run(args)
    output = json.loads(capsys.readouterr().out)
    _assert_agent_output_payload("init-run", output)

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
            "grounding_layout.latest",
        ]
    )

    _cmd_init_run(args)

    payload = json.loads((tmp_path / "runs" / "run1" / "run.json").read_text(encoding="utf-8"))
    assert payload["spec"]["target_labels"] == ["icon", "image", "shape"]
    assert payload["spec"]["metadata"]["target_labels_source"] == "legacy_prompt_id"


def test_cli_lifecycle_commands_emit_agent_json_payloads(tmp_path: Path, capsys) -> None:
    _write_sample_store(tmp_path)

    eval_args = _build_parser().parse_args(
        ["evaluate-run", "--output-root", str(tmp_path), "--run-id", "run_arrow"]
    )
    _cmd_evaluate_run(eval_args)
    eval_payload = json.loads(capsys.readouterr().out)
    _assert_agent_output_payload("evaluate-run", eval_payload)
    assert eval_payload == {
        "run_id": "run_arrow",
        "report_path": str(tmp_path / "runs" / "run_arrow" / "reports" / "metrics.json"),
        "summary_path": str(tmp_path / "runs" / "run_arrow" / "reports" / "summary.json"),
    }
    assert Path(eval_payload["report_path"]).exists()
    assert Path(eval_payload["summary_path"]).exists()

    source_run = tmp_path / "runs" / "run_arrow"
    for run_id in ("run_base", "run_a"):
        target = tmp_path / "runs" / run_id
        shutil.copytree(source_run, target)
        run_manifest = json.loads((target / "run.json").read_text(encoding="utf-8"))
        run_manifest["run_id"] = run_id
        _write_json(target / "run.json", run_manifest)

    compare_args = _build_parser().parse_args(
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
    _cmd_compare_runs(compare_args)
    compare_payload = json.loads(capsys.readouterr().out)
    _assert_agent_output_payload("compare-runs", compare_payload)
    assert compare_payload == {
        "comparison_id": "run_base__vs__run_a",
        "baseline_run_id": "run_base",
        "candidate_run_id": "run_a",
        "report_path": str(tmp_path / "exports" / "comparisons" / "run_base__vs__run_a.json"),
    }
    assert Path(compare_payload["report_path"]).exists()


def test_cli_gets_and_sets_run_note(tmp_path: Path, capsys) -> None:
    init_args = _build_parser().parse_args(
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
            "grounding_layout.latest",
        ]
    )
    _cmd_init_run(init_args)
    capsys.readouterr()

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
    _assert_agent_output_payload("set-run-note", set_payload)

    assert set_payload["note"] == "repro: ckpt epoch_3\nidea: prompt v2"
    assert set_payload["max_length"] == 20_000

    get_args = _build_parser().parse_args(
        ["get-run-note", "--output-root", str(tmp_path), "--run-id", "run1"]
    )
    _cmd_get_run_note(get_args)
    get_payload = json.loads(capsys.readouterr().out)
    _assert_agent_output_payload("get-run-note", get_payload)
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
    _assert_agent_output_payload("append-run-note", append_payload)

    assert append_payload["note"].startswith("repro: ckpt epoch_3\nidea: prompt v2\n\n")
    assert "## follow-up\nnext: inspect false positives" in append_payload["note"]

    guarded_file = tmp_path / "guarded.md"
    guarded_file.write_text("curated guarded note", encoding="utf-8")
    stale_set_args = _build_parser().parse_args(
        [
            "set-run-note",
            "--output-root",
            str(tmp_path),
            "--run-id",
            "run1",
            "--note-file",
            str(guarded_file),
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
            "--note-file",
            str(guarded_file),
            "--expected-updated-at",
            append_payload["updated_at"],
        ]
    )
    _cmd_set_run_note(guarded_set_args)
    guarded_payload = json.loads(capsys.readouterr().out)
    _assert_agent_output_payload("set-run-note", guarded_payload)
    assert guarded_payload["note"] == "curated guarded note"

    get_args = _build_parser().parse_args(
        ["get-run-note", "--output-root", str(tmp_path), "--run-id", "run1"]
    )
    _cmd_get_run_note(get_args)
    get_payload = json.loads(capsys.readouterr().out)
    assert get_payload["note"] == guarded_payload["note"]


def test_cli_exposes_agent_lifecycle_and_log_commands(tmp_path: Path, capsys) -> None:
    _write_json(
        tmp_path / "runs" / "run1" / "run.json",
        {
            "run_id": "run1",
            "status": "succeeded",
            "created_at": "2026-05-09T00:10:00Z",
            "model": {"model_id": "model-a", "path": "outputs/model-a/best"},
            "benchmark": {
                "benchmark_id": "bench1",
                "root": str(tmp_path / "benchmarks" / "bench1" / "data"),
                "split": "val",
                "tasks": ["detection"],
            },
            "spec": {"task": "detection"},
        },
    )
    state_args = _build_parser().parse_args(["dashboard-state", "--output-root", str(tmp_path)])
    _cmd_dashboard_state(state_args)
    state_payload = json.loads(capsys.readouterr().out)
    _assert_agent_output_payload("dashboard-state", state_payload)
    assert state_payload["run_count"] == 1

    archive_args = _build_parser().parse_args(
        ["archive-run", "--output-root", str(tmp_path), "--run-id", "run1"]
    )
    _cmd_archive_run(archive_args)
    archived = json.loads(capsys.readouterr().out)
    _assert_agent_output_payload("archive-run", archived)
    assert archived["status"] == "archived"
    assert json.loads((tmp_path / "runs" / "run1" / "run.json").read_text(encoding="utf-8"))[
        "status"
    ] == "archived"

    backend_log = tmp_path / "logs" / "backend.log"
    backend_log.parent.mkdir(parents=True)
    backend_log.write_text("alpha\nbeta\n", encoding="utf-8")
    backend_log_args = _build_parser().parse_args(
        ["backend-logs", "--output-root", str(tmp_path), "--max-lines", "1"]
    )
    _cmd_backend_logs(backend_log_args)
    backend_logs_payload = json.loads(capsys.readouterr().out)
    _assert_agent_output_payload("backend-logs", backend_logs_payload)
    assert backend_logs_payload["lines"] == ["beta\n"]

    runtime_log = tmp_path / "runs" / "run1" / "logs" / "runtime.log"
    runtime_log.parent.mkdir(parents=True)
    runtime_log.write_text("step1\nstep2\nstep3\n", encoding="utf-8")
    database = EvalBenchDatabase(tmp_path)
    database.create_job(
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
    job_logs_payload = json.loads(capsys.readouterr().out)
    _assert_agent_output_payload("job-logs", job_logs_payload)
    assert job_logs_payload["lines"] == ["step2\n", "step3\n"]

    cancel_args = _build_parser().parse_args(
        ["cancel-job", "--output-root", str(tmp_path), "--job-id", "job1"]
    )
    _cmd_cancel_job(cancel_args)
    cancelled_job = json.loads(capsys.readouterr().out)
    _assert_agent_output_payload("cancel-job", cancelled_job)
    assert cancelled_job["status"] == "cancelled"

    delete_job_args = _build_parser().parse_args(
        ["delete-job", "--output-root", str(tmp_path), "--job-id", "job1"]
    )
    _cmd_delete_job(delete_job_args)
    deleted_job = json.loads(capsys.readouterr().out)
    _assert_agent_output_payload("delete-job", deleted_job)
    assert deleted_job["deleted"] is True
    assert database.get_job("job1") is None

    scheduler_args = _build_parser().parse_args(
        ["scheduler-status", "--output-root", str(tmp_path)]
    )
    _cmd_scheduler_status(scheduler_args)
    scheduler_payload = json.loads(capsys.readouterr().out)
    _assert_agent_output_payload("scheduler-status", scheduler_payload)
    assert scheduler_payload["source"] == "cli_snapshot"
    assert scheduler_payload["enabled"] is False

    register_service_args = _build_parser().parse_args(
        [
            "register-service",
            "--output-root",
            str(tmp_path),
            "--kind",
            "external_vllm",
            "--service-id",
            "svc1",
            "--endpoint",
            "http://127.0.0.1:8000/v1",
        ]
    )
    _cmd_register_service(register_service_args)
    registered_service = json.loads(capsys.readouterr().out)
    _assert_agent_output_payload("register-service", registered_service)
    assert registered_service["service_id"] == "svc1"

    delete_service_args = _build_parser().parse_args(
        ["delete-service", "--output-root", str(tmp_path), "--service-id", "svc1"]
    )
    _cmd_delete_service(delete_service_args)
    deleted_service = json.loads(capsys.readouterr().out)
    _assert_agent_output_payload("delete-service", deleted_service)
    assert deleted_service["service"]["service_id"] == "svc1"

    delete_run_args = _build_parser().parse_args(
        ["delete-run", "--output-root", str(tmp_path), "--run-id", "run1"]
    )
    _cmd_delete_run(delete_run_args)
    deleted_run = json.loads(capsys.readouterr().out)
    _assert_agent_output_payload("delete-run", deleted_run)
    assert deleted_run["deleted"] is True
    assert not (tmp_path / "runs" / "run1").exists()
    assert Path(deleted_run["trash_path"]).exists()


def test_cli_import_predictions_accepts_target_label_subset(tmp_path: Path, capsys) -> None:
    split_path = tmp_path / "benchmarks" / "bench1" / "splits" / "val.txt"
    data_root = tmp_path / "benchmarks" / "bench1" / "data"
    split_path.parent.mkdir(parents=True)
    split_path.write_text("part1/json/a.json\n", encoding="utf-8")
    _write_json(
        tmp_path / "benchmarks" / "bench1" / "benchmark.json",
        {
            "benchmark_id": "bench1",
            "tasks": ["detection"],
            "split": "val",
            "sample_count": 1,
            "root": str(data_root),
            "manifest_path": str(split_path),
        },
    )
    _write_json(
        data_root / "part1" / "json" / "a.json",
        {
            "image_path": "part1/images/a.png",
            "instances": [
                {"label": "icon", "bbox": [0, 0, 100, 100]},
                {"label": "arrow", "bbox": [200, 200, 260, 260]},
            ],
        },
    )
    prediction_root = tmp_path / "predictions"
    _write_json(
        prediction_root / "part1" / "json" / "a.json",
        {
            "image": "part1/images/a.png",
            "instances": [
                {"label": "icon", "bbox": [0, 0, 100, 100]},
                {"label": "arrow", "bbox": [200, 200, 260, 260]},
            ],
        },
    )
    args = _build_parser().parse_args(
        [
            "import-predictions",
            "--output-root",
            str(tmp_path),
            "--run-id",
            "imported_arrow",
            "--benchmark-id",
            "bench1",
            "--prediction-root",
            str(prediction_root),
            "--task",
            "detection",
            "--model-id",
            "external-model",
            "--prompt-id",
            "grounding_layout.latest",
            "--target-label",
            "arrow",
        ]
    )

    _cmd_import_predictions(args)
    payload = json.loads(capsys.readouterr().out)
    _assert_agent_output_payload("import-predictions", payload)
    report = json.loads(Path(payload["report_path"]).read_text(encoding="utf-8"))

    assert payload["run_id"] == "imported_arrow"
    assert report["target_labels"] == ["arrow"]
    assert report["target_labels_source"] == "explicit"
    assert [item["label"] for item in report["labels"]] == ["arrow"]


def test_cli_prints_filtered_rank_board(tmp_path: Path, capsys) -> None:
    for run_id, label, precision in (
        ("run_a", "icon", 0.9),
        ("run_b", "arrow", 0.5),
    ):
        run_dir = tmp_path / "runs" / run_id
        (run_dir / "reports").mkdir(parents=True)
        (run_dir / "run.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "status": "succeeded",
                    "created_at": "2026-05-09T00:10:00Z",
                    "model": {"model_id": run_id, "path": "outputs/model/best"},
                    "benchmark": {
                        "benchmark_id": "bench1",
                        "root": str(tmp_path / "benchmarks" / "bench1" / "data"),
                        "split": "val",
                        "tasks": ["detection"],
                    },
                    "spec": {
                        "task": "detection",
                        "metric_profile": "detection_iou_v1",
                        "target_labels": [label],
                    },
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "reports" / "summary.json").write_text(
            json.dumps(
                {
                    "precision_iou50": precision,
                    "recall_iou50": precision,
                    "mean_iou": precision,
                    "prediction_file_count": 1,
                }
            ),
            encoding="utf-8",
        )

    args = _build_parser().parse_args(
        [
            "rank-board",
            "--output-root",
            str(tmp_path),
            "--label",
            "icon",
            "--metric-profile",
            "detection_iou_v1",
            "--min-score",
            "0.7",
            "--sort-by",
            "run_id",
            "--sort-order",
            "desc",
            "--rank-scheme-json",
            json.dumps(
                {
                    "name": "agent_weighted_quality",
                    "terms": [
                        {
                            "benchmark_id": "bench1",
                            "metric": "precision_iou50",
                            "weight": 0.4,
                            "missing": "drop",
                        },
                        {
                            "benchmark_id": "bench1",
                            "metric": "mean_iou",
                            "weight": 0.6,
                            "missing": "zero",
                        },
                    ],
                }
            ),
        ]
    )
    _cmd_rank_board(args)
    payload = json.loads(capsys.readouterr().out)
    _assert_agent_output_payload("rank-board", payload)

    assert payload["total"] == 1
    assert payload["primary_metric"] == "weighted_score"
    assert payload["primary_metric_label"] == "agent_weighted_quality"
    assert payload["sort_by"] == "run_id"
    assert payload["sort_order"] == "desc"
    assert payload["filters"]["min_score"] == "0.7"
    assert payload["filters"]["rank_scheme"] == "agent_weighted_quality"
    assert payload["rank_scheme"]["name"] == "agent_weighted_quality"
    assert payload["facets"]["metric_profiles"] == [{"value": "detection_iou_v1", "count": 1}]
    assert payload["entries"][0]["run_id"] == "run_a"
    assert payload["entries"][0]["rank"] == 1
    assert payload["entries"][0]["score"] == pytest.approx(0.9)
    assert payload["entries"][0]["score_delta"] == pytest.approx(0.0)
    assert payload["entries"][0]["score_components"][0]["metric"] == "precision_iou50"

    metric_args = _build_parser().parse_args(
        [
            "rank-board",
            "--output-root",
            str(tmp_path),
            "--sort-by",
            "recall_iou50",
        ]
    )
    _cmd_rank_board(metric_args)
    metric_payload = json.loads(capsys.readouterr().out)
    _assert_agent_output_payload("rank-board", metric_payload)
    assert metric_payload["primary_metric"] == "recall_iou50"
    assert metric_payload["primary_metric_label"] == "R@.50"
    assert metric_payload["score_formula"] == "R@.50"
    assert metric_payload["entries"][0]["run_id"] == "run_a"
    assert metric_payload["entries"][0]["score"] == pytest.approx(0.9)
    assert metric_payload["entries"][0]["score_delta"] == pytest.approx(0.0)
    assert metric_payload["entries"][1]["score_delta"] < 0


def test_cli_lists_benchmarks_runs_and_comparisons_with_agent_filters(
    tmp_path: Path,
    capsys,
) -> None:
    _write_json(
        tmp_path / "benchmarks" / "bench1" / "benchmark.json",
        {
            "benchmark_id": "bench1",
            "tasks": ["detection"],
            "labels": ["arrow", "icon"],
            "layers": ["layout"],
            "split": "val",
            "sample_count": 2,
            "root": str(tmp_path / "benchmarks" / "bench1" / "data"),
            "manifest_path": str(tmp_path / "benchmarks" / "bench1" / "splits" / "val.txt"),
            "created_at": "2026-05-09T00:00:00Z",
        },
    )
    _write_json(
        tmp_path / "benchmarks" / "bench2" / "benchmark.json",
        {
            "benchmark_id": "bench2",
            "tasks": ["keypoint"],
            "layers": ["arrow"],
            "split": "val",
            "sample_count": 1,
            "root": str(tmp_path / "benchmarks" / "bench2" / "data"),
            "manifest_path": str(tmp_path / "benchmarks" / "bench2" / "splits" / "val.txt"),
        },
    )
    _write_json(
        tmp_path / "runs" / "run_a" / "run.json",
        {
            "run_id": "run_a",
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
                "metric_profile": "detection_iou_v1",
                "target_labels": ["arrow"],
                "prompt": {"prompt_id": "grounding_arrow.latest"},
            },
        },
    )
    _write_json(
        tmp_path / "runs" / "run_b" / "run.json",
        {
            "run_id": "run_b",
            "status": "failed",
            "model": {"model_id": "model-b", "path": "outputs/model-b/best"},
            "benchmark": {
                "benchmark_id": "bench2",
                "root": str(tmp_path / "benchmarks" / "bench2" / "data"),
                "split": "val",
                "tasks": ["keypoint"],
            },
            "spec": {
                "task": "keypoint",
                "metric_profile": "keypoint_endpoint_v1",
                "target_labels": ["arrow"],
                "prompt": {"prompt_id": "keypoint_arrow.latest"},
            },
        },
    )
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
            "delta": {"precision_iou50": 0.2},
            "summary": {"improved_samples": 1},
        },
    )

    benchmark_args = _build_parser().parse_args(
        [
            "list-benchmarks",
            "--output-root",
            str(tmp_path),
            "--task",
            "detection",
            "--layer",
            "layout",
            "--split",
            "val",
            "--query",
            "bench1",
        ]
    )
    _cmd_list_benchmarks(benchmark_args)
    benchmarks = json.loads(capsys.readouterr().out)
    assert benchmarks["total"] == 1
    assert benchmarks["filters"]["task"] == "detection"
    assert benchmarks["filters"]["split"] == "val"
    assert benchmarks["benchmarks"][0]["benchmark_id"] == "bench1"
    assert benchmarks["benchmarks"][0]["labels"] == ["arrow", "icon"]

    benchmark_detail_args = _build_parser().parse_args(
        ["show-benchmark", "--output-root", str(tmp_path), "--benchmark-id", "bench1"]
    )
    _cmd_show_benchmark(benchmark_detail_args)
    benchmark_detail = json.loads(capsys.readouterr().out)
    assert benchmark_detail["benchmark"]["benchmark_id"] == "bench1"
    assert benchmark_detail["benchmark"]["tasks"] == ["detection"]
    assert benchmark_detail["benchmark"]["labels"] == ["arrow", "icon"]
    assert benchmark_detail["benchmark"]["sample_count"] == 2

    run_args = _build_parser().parse_args(
        [
            "list-runs",
            "--output-root",
            str(tmp_path),
            "--task",
            "detection",
            "--benchmark-id",
            "bench1",
            "--label",
            "arrow",
            "--model-id",
            "model-a",
            "--metric-profile",
            "detection_iou_v1",
            "--query",
            "grounding",
        ]
    )
    _cmd_list_runs(run_args)
    runs = json.loads(capsys.readouterr().out)
    assert runs["total"] == 1
    assert runs["filters"]["label"] == "arrow"
    assert runs["runs"][0]["run_id"] == "run_a"
    assert runs["runs"][0]["target_labels"] == ["arrow"]

    comparison_args = _build_parser().parse_args(
        [
            "list-comparisons",
            "--output-root",
            str(tmp_path),
            "--task",
            "detection",
            "--baseline-run-id",
            "run_base",
            "--label",
            "arrow",
            "--query",
            "run_a",
        ]
    )
    _cmd_list_comparisons(comparison_args)
    comparisons = json.loads(capsys.readouterr().out)
    assert comparisons["total"] == 1
    assert comparisons["filters"]["baseline_run_id"] == "run_base"
    assert comparisons["comparisons"][0]["comparison_id"] == "run_base__vs__run_a"
    assert comparisons["comparisons"][0]["metric_profile"] == "detection_iou_v1"


def test_cli_resolves_target_labels_for_agent_label_subtasks(
    tmp_path: Path,
    capsys,
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
    EvalBenchDatabase(tmp_path).upsert_prompt_template(
        {
            "prompt_id": "grounding_arrow.latest",
            "label": "Arrow grounding",
            "task": "detection",
            "system_prompt": "You inspect diagrams.",
            "user_prompt": "Find arrows.",
            "metadata": {"target_labels": ["arrow"]},
        }
    )

    args = _build_parser().parse_args(
        [
            "resolve-target-labels",
            "--output-root",
            str(tmp_path),
            "--benchmark-id",
            "bench1",
            "--prompt-id",
            "grounding_arrow.latest",
        ]
    )
    _cmd_resolve_target_labels(args)
    payload = json.loads(capsys.readouterr().out)
    assert payload["task"] == "detection"
    assert payload["target_labels"] == ["arrow"]
    assert payload["target_labels_source"] == "prompt_metadata"
    assert payload["candidate_labels"] == ["arrow", "icon"]
    assert payload["label_subtasks_supported"] is True
    assert payload["valid"] is True

    keypoint_args = _build_parser().parse_args(
        [
            "resolve-target-labels",
            "--output-root",
            str(tmp_path),
            "--benchmark-id",
            "bench1",
            "--task",
            "keypoint",
            "--prompt-id",
            "keypoint_arrow.latest",
        ]
    )
    _cmd_resolve_target_labels(keypoint_args)
    keypoint_payload = json.loads(capsys.readouterr().out)
    assert keypoint_payload["task"] == "keypoint"
    assert keypoint_payload["target_labels"] == ["arrow"]
    assert keypoint_payload["label_subtasks_supported"] is False
    assert keypoint_payload["valid"] is True

    bad_keypoint_args = _build_parser().parse_args(
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
    _cmd_resolve_target_labels(bad_keypoint_args)
    bad_keypoint_payload = json.loads(capsys.readouterr().out)
    assert bad_keypoint_payload["label_subtasks_supported"] is False
    assert bad_keypoint_payload["valid"] is False
    assert any(
        "keypoint target_labels only support arrow" in item
        for item in bad_keypoint_payload["errors"]
    )

    bad_args = _build_parser().parse_args(
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
    _cmd_resolve_target_labels(bad_args)
    bad_payload = json.loads(capsys.readouterr().out)
    assert bad_payload["valid"] is False
    assert any("arrwo" in item for item in bad_payload["errors"])


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
    assert comparison["comparison_id"] == "run_base__vs__run_a"
    assert comparison["target_labels"] == ["arrow"]

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
    assert sample["baseline_run_id"] == "run_base"
    assert sample["candidate_run_id"] == "run_a"
    assert sample["sample_index"] == 0
    assert sample["baseline"]["sample"]["index"] == 0
    assert sample["candidate"]["sample"]["index"] == 0
    assert [item["label"] for item in sample["baseline"]["gt_instances"]] == ["arrow"]


def test_cli_manages_job_and_prompt_templates_for_agents(tmp_path: Path, capsys) -> None:
    job_template_args = _build_parser().parse_args(["list-job-templates", "--query", "keypoint"])
    _cmd_list_job_templates(job_template_args)
    job_templates = json.loads(capsys.readouterr().out)
    assert job_templates["total"] == 1
    assert "keypoint_eval_job" in job_templates["templates"]
    assert job_templates["templates"]["keypoint_eval_job"]["manifest"]["eval"]["task"] == "keypoint"

    show_job_template_args = _build_parser().parse_args(
        ["show-job-template", "--template-id", "keypoint_eval_job"]
    )
    _cmd_show_job_template(show_job_template_args)
    job_template = json.loads(capsys.readouterr().out)
    assert job_template["template_id"] == "keypoint_eval_job"
    assert job_template["template"]["manifest"]["eval"]["metric_profile"] == "keypoint_endpoint_v1"

    list_args = _build_parser().parse_args(
        ["list-prompt-templates", "--output-root", str(tmp_path), "--task", "detection"]
    )
    _cmd_list_prompt_templates(list_args)
    prompt_templates = json.loads(capsys.readouterr().out)
    assert prompt_templates["total"] >= 1
    assert "grounding_arrow.latest" in prompt_templates["by_id"]
    assert prompt_templates["by_id"]["grounding_arrow.latest"]["task"] == "detection"

    show_prompt_args = _build_parser().parse_args(
        [
            "show-prompt-template",
            "--output-root",
            str(tmp_path),
            "--prompt-id",
            "grounding_arrow.latest",
        ]
    )
    _cmd_show_prompt_template(show_prompt_args)
    prompt_template = json.loads(capsys.readouterr().out)
    assert prompt_template["template"]["prompt_id"] == "grounding_arrow.latest"
    assert prompt_template["template"]["metadata"]["target_labels"] == ["arrow"]

    custom_payload = {
        "prompt_id": "custom.arrow.v1",
        "label": "Custom Arrow",
        "task": "detection",
        "system_prompt": "You inspect visual structures.",
        "user_prompt": "Detect arrows only.",
        "parser": "raw_data_detection_v1",
        "metric_profile": "detection_iou_v1",
        "metadata": {"target_labels": ["arrow"], "source": "agent_cli_test"},
    }
    upsert_args = _build_parser().parse_args(
        [
            "upsert-prompt-template",
            "--output-root",
            str(tmp_path),
            "--payload-json",
            json.dumps(custom_payload),
        ]
    )
    _cmd_upsert_prompt_template(upsert_args)
    upserted = json.loads(capsys.readouterr().out)
    assert upserted["prompt_id"] == "custom.arrow.v1"
    assert upserted["metadata"]["target_labels"] == ["arrow"]

    custom_list_args = _build_parser().parse_args(
        [
            "list-prompt-templates",
            "--output-root",
            str(tmp_path),
            "--query",
            "agent_cli_test",
        ]
    )
    _cmd_list_prompt_templates(custom_list_args)
    custom_list = json.loads(capsys.readouterr().out)
    assert custom_list["total"] == 1
    assert custom_list["templates"][0]["prompt_id"] == "custom.arrow.v1"

    delete_args = _build_parser().parse_args(
        [
            "delete-prompt-template",
            "--output-root",
            str(tmp_path),
            "--prompt-id",
            "custom.arrow.v1",
        ]
    )
    _cmd_delete_prompt_template(delete_args)
    deleted = json.loads(capsys.readouterr().out)
    assert deleted == {"prompt_id": "custom.arrow.v1", "deleted": True}


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
    assert detail["sample"]["instance_count"] == 2
    assert [item["label"] for item in detail["gt_instances"]] == ["icon", "arrow"]


def test_cli_preflights_and_creates_manifest_first_job(tmp_path: Path, capsys) -> None:
    model_path = tmp_path / "models" / "model-a"
    _write_json(model_path / "config.json", {"num_attention_heads": 4})
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
    payload_path = tmp_path / "job.json"
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
                "eval": {
                    "model_id": "model-a",
                    "benchmark_id": "bench1",
                    "task": "detection",
                    "prompt_id": "grounding_arrow.latest",
                    "target_labels": ["arrow"],
                },
            }
        },
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
    assert job["payload"]["manifest"]["kind"] == "eval_job"

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
    assert jobs["jobs"][0]["job_id"] == job["job_id"]

    show_job_args = _build_parser().parse_args(
        ["show-job", "--output-root", str(tmp_path), "--job-id", job["job_id"]]
    )
    _cmd_show_job(show_job_args)
    job_detail = json.loads(capsys.readouterr().out)
    assert job_detail["job"]["job_id"] == job["job_id"]
    assert job_detail["job"]["payload"]["target_labels"] == ["arrow"]
    assert job_detail["job"]["payload"]["manifest"]["kind"] == "eval_job"


def test_cli_preflight_rejects_unknown_target_label(tmp_path: Path, capsys) -> None:
    model_path = tmp_path / "models" / "model-a"
    _write_json(model_path / "config.json", {"num_attention_heads": 4})
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
    payload_path = tmp_path / "job.json"
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
                "eval": {
                    "model_id": "model-a",
                    "benchmark_id": "bench1",
                    "task": "detection",
                    "prompt_id": "grounding_arrow.latest",
                    "target_labels": ["arrwo"],
                },
            }
        },
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
    assert any(
        "target_labels not found in benchmark label index: arrwo" in item
        for item in preflight["errors"]
    )


def test_cli_preflight_rejects_keypoint_label_subtasks(tmp_path: Path, capsys) -> None:
    model_path = tmp_path / "models" / "model-a"
    _write_json(model_path / "config.json", {"num_attention_heads": 4})
    _write_json(
        tmp_path / "benchmarks" / "bench1" / "benchmark.json",
        {
            "benchmark_id": "bench1",
            "tasks": ["keypoint"],
            "labels": ["arrow", "icon"],
            "split": "val",
            "sample_count": 1,
            "root": str(tmp_path / "benchmarks" / "bench1" / "data"),
            "manifest_path": str(tmp_path / "benchmarks" / "bench1" / "splits" / "val.txt"),
        },
    )
    payload_path = tmp_path / "job.json"
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
                "eval": {
                    "model_id": "model-a",
                    "benchmark_id": "bench1",
                    "task": "keypoint",
                    "prompt_id": "keypoint_arrow.latest",
                    "metric_profile": "keypoint_endpoint_v1",
                    "target_labels": ["icon"],
                },
            }
        },
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
    assert any(
        "keypoint target_labels only support arrow" in item for item in preflight["errors"]
    )


def test_cli_create_job_persists_preflight_warnings(tmp_path: Path, capsys) -> None:
    model_path = tmp_path / "models" / "model-a"
    _write_json(model_path / "config.json", {"num_attention_heads": 4})
    _write_json(
        tmp_path / "benchmarks" / "bench1" / "benchmark.json",
        {
            "benchmark_id": "bench1",
            "tasks": ["detection"],
            "split": "val",
            "sample_count": 1,
            "root": str(tmp_path / "benchmarks" / "bench1" / "data"),
            "manifest_path": str(tmp_path / "benchmarks" / "bench1" / "splits" / "val.txt"),
        },
    )
    payload_path = tmp_path / "job.json"
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
                "eval": {
                    "model_id": "model-a",
                    "benchmark_id": "bench1",
                    "task": "detection",
                    "prompt_id": "grounding_arrow.latest",
                    "target_labels": ["arrow"],
                },
            }
        },
    )

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
    capsys.readouterr()

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

    assert payload["total"] == 1
    assert payload["filters"]["kind"] == "external_vllm"
    assert payload["services"][0]["service_id"] == "external-qwen3vl"

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
    assert detail["service"]["service_id"] == "external-qwen3vl"
    assert detail["service"]["kind"] == "external_vllm"
    assert detail["service"]["config"]["endpoint"] == "http://127.0.0.1:8000/v1"
