from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from eval_bench.database import EvalBenchDatabase
from eval_bench.prompt_templates import DEFAULT_PROMPT_SPECS
from eval_bench.cli import (
    CLI_JSON_OUTPUT_SCHEMAS,
    CLI_DESTRUCTIVE_COMMANDS,
    CLI_JSON_COMMANDS,
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
    _cmd_ops_summary,
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


def _assert_cli_json_payload(command_name: str, payload: object) -> None:
    schema = CLI_JSON_OUTPUT_SCHEMAS[command_name]
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


def _assert_paged_schema(schema: dict[str, object]) -> None:
    properties = schema["properties"]
    assert properties["offset"]["type"] == "int"
    assert properties["limit"]["type"] == "int"
    assert properties["total"]["type"] == "int"
    assert properties["filters"]["type"] == "object"


def _assert_filter_schema(schema: dict[str, object], keys: list[str]) -> None:
    filters = schema["properties"]["filters"]
    assert filters["type"] == "object"
    assert filters["required"] == keys
    for key in keys:
        assert filters["properties"][key]["type"] == "str"


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
    assert "--rank-scheme-json" in rank_help.stdout
    assert "--rank-scheme-file" in rank_help.stdout
    assert "f1_iou50" in rank_help.stdout
    assert "weighted_score" in rank_help.stdout
    assert "contract" not in rank_help.stdout.lower()


def test_cli_json_output_schemas_cover_stable_commands() -> None:
    command_names = _parser_command_names()
    assert set(CLI_JSON_OUTPUT_SCHEMAS) == CLI_JSON_COMMANDS
    assert CLI_JSON_COMMANDS <= command_names
    assert "list-agent-commands" not in CLI_JSON_OUTPUT_SCHEMAS
    assert "show-agent-command" not in CLI_JSON_OUTPUT_SCHEMAS

    for command_name, schema in CLI_JSON_OUTPUT_SCHEMAS.items():
        for key in schema.get("required", []):
            assert key in schema.get("properties", {}), (
                f"{command_name}: required field {key!r} must declare a JSON schema property"
            )

    init_output_schema = CLI_JSON_OUTPUT_SCHEMAS["init-run"]
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
    assert CLI_JSON_OUTPUT_SCHEMAS["validate-prediction"]["properties"]["instances"] == "int"
    ops_summary_schema = CLI_JSON_OUTPUT_SCHEMAS["ops-summary"]
    assert ops_summary_schema["required"] == [
        "source",
        "store_root",
        "runs",
        "benchmarks",
        "jobs",
        "services",
        "scheduler",
    ]
    assert ops_summary_schema["properties"]["runs"]["properties"]["waiting_evaluation"][
        "type"
    ] == "int"
    assert ops_summary_schema["properties"]["runs"]["properties"]["best_f1"][
        "type"
    ] == "float|null"
    assert ops_summary_schema["properties"]["runs"]["properties"]["best_f1_run"][
        "type"
    ] == "object|null"
    assert ops_summary_schema["properties"]["runs"]["properties"]["best_f1_run"][
        "properties"
    ]["target_labels"] == "list[str]"
    assert ops_summary_schema["properties"]["benchmarks"]["properties"]["sample_count"][
        "type"
    ] == "int"
    assert ops_summary_schema["properties"]["jobs"]["properties"]["active"]["type"] == "int"
    assert ops_summary_schema["properties"]["services"]["properties"]["running"]["type"] == "int"
    assert ops_summary_schema["properties"]["scheduler"]["required"] == ["enabled"]
    assert ops_summary_schema["properties"]["scheduler"]["properties"]["enabled"] == "bool"
    assert (
        ops_summary_schema["properties"]["scheduler"]["properties"]["active_worker_threads"]
        == "list[str]"
    )
    rank_output_schema = CLI_JSON_OUTPUT_SCHEMAS["rank-board"]
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
    _assert_paged_schema(rank_output_schema)
    _assert_filter_schema(
        rank_output_schema,
        [
            "task",
            "benchmark_id",
            "benchmark_split",
            "status",
            "label",
            "model_id",
            "prompt_id",
            "metric_profile",
            "min_score",
            "query",
            "rank_scheme",
        ],
    )
    assert rank_output_schema["properties"]["evaluated_count"]["type"] == "int"
    assert rank_output_schema["properties"]["primary_metric"]["type"] == "str"
    assert rank_output_schema["properties"]["primary_metric_label"]["type"] == "str"
    assert rank_output_schema["properties"]["rank_scheme"]["type"] == "object|null"
    assert rank_output_schema["properties"]["facets"]["keys"] == [
        "tasks",
        "benchmarks",
        "splits",
        "statuses",
        "labels",
        "models",
        "prompts",
        "metric_profiles",
    ]
    assert rank_output_schema["properties"]["facets"]["item_shape"] == {
        "value": "str",
        "count": "int",
    }
    benchmark_output_schema = CLI_JSON_OUTPUT_SCHEMAS["list-benchmarks"]
    assert benchmark_output_schema["required"] == [
        "offset",
        "limit",
        "total",
        "filters",
        "facets",
        "benchmarks",
    ]
    _assert_paged_schema(benchmark_output_schema)
    _assert_filter_schema(benchmark_output_schema, ["task", "layer", "split", "query"])
    assert benchmark_output_schema["properties"]["facets"]["keys"] == [
        "tasks",
        "layers",
        "splits",
        "labels",
    ]
    assert benchmark_output_schema["properties"]["facets"]["item_shape"] == {
        "value": "str",
        "count": "int",
    }
    assert "score_delta" in rank_output_schema["properties"]["entries"]["item_shape"]
    note_output_schema = CLI_JSON_OUTPUT_SCHEMAS["get-run-note"]
    assert note_output_schema["required"] == [
        "run_id",
        "note",
        "updated_at",
        "path",
        "max_length",
    ]
    assert CLI_JSON_OUTPUT_SCHEMAS["set-run-note"] == note_output_schema
    assert CLI_JSON_OUTPUT_SCHEMAS["append-run-note"] == note_output_schema
    resolve_output_schema = CLI_JSON_OUTPUT_SCHEMAS["resolve-target-labels"]
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
    assert resolve_output_schema["properties"]["valid"]["type"] == "bool"
    assert resolve_output_schema["properties"]["benchmark_labels"]["type"] == "list[str]"
    runs_output_schema = CLI_JSON_OUTPUT_SCHEMAS["list-runs"]
    assert runs_output_schema["required"] == ["offset", "limit", "total", "filters", "facets", "runs"]
    _assert_paged_schema(runs_output_schema)
    _assert_filter_schema(
        runs_output_schema,
        [
            "task",
            "benchmark_id",
            "benchmark_split",
            "status",
            "label",
            "model_id",
            "prompt_id",
            "metric_profile",
            "query",
        ],
    )
    assert runs_output_schema["properties"]["facets"]["keys"] == [
        "tasks",
        "benchmarks",
        "splits",
        "statuses",
        "labels",
        "models",
        "prompts",
        "metric_profiles",
    ]
    assert runs_output_schema["properties"]["facets"] == rank_output_schema["properties"]["facets"]
    assert runs_output_schema["properties"]["runs"]["item_shape"]["target_labels"] == "list[str]"
    assert runs_output_schema["properties"]["runs"]["item_shape"]["benchmark_split"] == "str"
    assert runs_output_schema["properties"]["runs"]["item_shape"]["note_updated_at"] == "str|null"
    assert runs_output_schema["properties"]["runs"]["item_shape"]["f1_iou50"] == "float|null"
    assert CLI_JSON_OUTPUT_SCHEMAS["show-run"]["properties"]["run"]["item_shape"] == (
        runs_output_schema["properties"]["runs"]["item_shape"]
    )
    run_samples_output_schema = CLI_JSON_OUTPUT_SCHEMAS["list-run-samples"]
    assert run_samples_output_schema["required"] == [
        "run_id",
        "offset",
        "limit",
        "total",
        "filters",
        "labels",
        "samples",
    ]
    _assert_paged_schema(run_samples_output_schema)
    _assert_filter_schema(run_samples_output_schema, ["run_id", "label", "error_filter"])
    assert run_samples_output_schema["properties"]["samples"]["item_shape"]["gt_instance_count"] == "int"
    assert run_samples_output_schema["properties"]["samples"]["item_shape"]["diagnostics"] == "object|null"
    assert (
        CLI_JSON_OUTPUT_SCHEMAS["show-run-sample"]["properties"]["sample"]["item_shape"]
        == run_samples_output_schema["properties"]["samples"]["item_shape"]
    )
    assert (
        CLI_JSON_OUTPUT_SCHEMAS["show-run-sample"]["properties"]["raw_payload"][
            "properties"
        ]["instances"]
        == "list[object]"
    )
    assert (
        CLI_JSON_OUTPUT_SCHEMAS["show-run-sample"]["properties"]["prediction_payload"][
            "item_shape"
        ]["image"]
        == "str|null"
    )
    assert (
        CLI_JSON_OUTPUT_SCHEMAS["show-run-sample"]["properties"]["diagnostics"][
            "item_shape"
        ]["matched_count"]
        == "int"
    )
    benchmark_samples_output_schema = CLI_JSON_OUTPUT_SCHEMAS["list-benchmark-samples"]
    assert "filters" in benchmark_samples_output_schema["required"]
    _assert_paged_schema(benchmark_samples_output_schema)
    _assert_filter_schema(benchmark_samples_output_schema, ["benchmark_id", "label"])
    assert benchmark_samples_output_schema["properties"]["samples"]["item_shape"]["instance_count"] == "int"
    assert (
        CLI_JSON_OUTPUT_SCHEMAS["show-benchmark-sample"]["properties"]["sample"][
            "item_shape"
        ]
        == benchmark_samples_output_schema["properties"]["samples"]["item_shape"]
    )
    assert (
        CLI_JSON_OUTPUT_SCHEMAS["show-benchmark-sample"]["properties"]["raw_payload"][
            "properties"
        ]["image_width"]
        == "int|null"
    )
    jobs_output_schema = CLI_JSON_OUTPUT_SCHEMAS["list-jobs"]
    assert jobs_output_schema["required"] == ["offset", "limit", "total", "filters", "facets", "jobs"]
    _assert_paged_schema(jobs_output_schema)
    _assert_filter_schema(jobs_output_schema, ["kind", "status", "query"])
    assert jobs_output_schema["properties"]["facets"]["keys"] == ["kinds", "statuses"]
    assert jobs_output_schema["properties"]["facets"]["item_shape"] == {
        "value": "str",
        "count": "int",
    }
    job_payload_schema = jobs_output_schema["properties"]["jobs"]["item_shape"]["payload"]
    job_metadata_schema = jobs_output_schema["properties"]["jobs"]["item_shape"]["metadata"]
    assert job_payload_schema["properties"]["benchmark_id"] == "str"
    assert job_payload_schema["properties"]["target_labels"] == "list[str]"
    assert job_payload_schema["properties"]["runtime_mode"] == "str"
    assert job_payload_schema["properties"]["job_manifest"]["properties"]["runtime"][
        "properties"
    ]["args"]["properties"]["max-model-len"] == "int|null"
    assert job_payload_schema["properties"]["job_manifest"]["properties"]["eval"][
        "properties"
    ]["generation"]["properties"]["max_tokens"] == "int"
    assert job_metadata_schema["properties"]["preflight_warnings"] == "list[str]"
    assert job_metadata_schema["properties"]["progress_done"] == "int|null"
    assert job_metadata_schema["properties"]["resolved_manifest"] == job_payload_schema[
        "properties"
    ]["job_manifest"]
    assert CLI_JSON_OUTPUT_SCHEMAS["show-job"]["properties"]["job"]["item_shape"] == (
        jobs_output_schema["properties"]["jobs"]["item_shape"]
    )
    assert (
        CLI_JSON_OUTPUT_SCHEMAS["list-job-templates"]["properties"]["templates"][
            "item_shape"
        ]["manifest"]
        == job_payload_schema["properties"]["job_manifest"]
    )
    assert CLI_JSON_OUTPUT_SCHEMAS["list-job-templates"]["properties"]["total"]["type"] == "int"
    _assert_filter_schema(CLI_JSON_OUTPUT_SCHEMAS["list-job-templates"], ["query"])
    assert (
        CLI_JSON_OUTPUT_SCHEMAS["show-job-template"]["properties"]["template"][
            "item_shape"
        ]["description"]
        == "str"
    )
    prompt_templates_output_schema = CLI_JSON_OUTPUT_SCHEMAS["list-prompt-templates"]
    _assert_paged_schema(prompt_templates_output_schema)
    _assert_filter_schema(prompt_templates_output_schema, ["task", "query"])
    assert prompt_templates_output_schema["properties"]["templates"]["item_shape"]["metadata"] == "object"
    assert (
        prompt_templates_output_schema["properties"]["templates"]["item_shape"]["generation"][
            "properties"
        ]["temperature"]
        == "float"
    )
    assert (
        prompt_templates_output_schema["properties"]["templates"]["item_shape"]["data"][
            "properties"
        ]["batch_size"]
        == "int"
    )
    assert prompt_templates_output_schema["properties"]["by_id"]["item_shape"]["prompt_id"] == "str"
    assert (
        CLI_JSON_OUTPUT_SCHEMAS["show-prompt-template"]["properties"]["template"][
            "item_shape"
        ]
        == prompt_templates_output_schema["properties"]["templates"]["item_shape"]
    )
    assert CLI_JSON_OUTPUT_SCHEMAS["upsert-prompt-template"]["properties"]["prompt_id"] == "str"
    assert (
        CLI_JSON_OUTPUT_SCHEMAS["delete-prompt-template"]["properties"]["deleted"][
            "type"
        ]
        == "bool"
    )
    preflight_schema = CLI_JSON_OUTPUT_SCHEMAS["preflight-job"]
    assert "runtime_command" in preflight_schema["required"]
    assert preflight_schema["properties"]["runtime_command"] == "list[str]|null"
    assert preflight_schema["properties"]["resolved_manifest"]["properties"]["runtime"][
        "properties"
    ]["mode"] == "str"
    assert preflight_schema["properties"]["resolved_manifest"]["properties"]["runtime"][
        "properties"
    ]["env"]["properties"]["CUDA_VISIBLE_DEVICES"] == "str|null"
    assert preflight_schema["properties"]["resolved_manifest"]["properties"]["eval"][
        "properties"
    ]["target_labels"] == "list[str]"
    assert preflight_schema["properties"]["resolved_manifest"]["properties"]["eval"][
        "properties"
    ]["data"]["properties"]["max_pixels"] == "int|null"
    assert preflight_schema["properties"]["resolved_manifest"]["properties"]["preannotate"][
        "properties"
    ]["source_root"] == "str|null"
    assert preflight_schema["properties"]["resolved_payload"]["properties"]["runtime_mode"] == "str"
    assert preflight_schema["properties"]["resolved_payload"]["properties"]["target_labels"] == "list[str]"
    assert (
        preflight_schema["properties"]["resolved_payload"]["properties"]["job_manifest"]
        == job_payload_schema["properties"]["job_manifest"]
    )
    assert CLI_JSON_OUTPUT_SCHEMAS["create-job"]["properties"]["payload"] == job_payload_schema
    assert CLI_JSON_OUTPUT_SCHEMAS["cancel-job"]["properties"]["status"] == "str"
    assert CLI_JSON_OUTPUT_SCHEMAS["delete-job"]["properties"]["deleted"]["type"] == "bool"
    assert CLI_JSON_OUTPUT_SCHEMAS["process-next-job"]["properties"]["job"][
        "item_shape"
    ]["job_id"] == "str"
    services_output_schema = CLI_JSON_OUTPUT_SCHEMAS["list-services"]
    assert services_output_schema["required"] == [
        "offset",
        "limit",
        "total",
        "filters",
        "facets",
        "services",
    ]
    _assert_paged_schema(services_output_schema)
    _assert_filter_schema(services_output_schema, ["kind", "status", "query"])
    assert services_output_schema["properties"]["facets"]["keys"] == ["kinds", "statuses"]
    assert services_output_schema["properties"]["facets"]["item_shape"] == {
        "value": "str",
        "count": "int",
    }
    service_shape = services_output_schema["properties"]["services"]["item_shape"]
    service_config_schema = service_shape["config"]
    service_runtime_schema = service_shape["runtime"]
    service_metadata_schema = service_shape["metadata"]
    assert service_config_schema["properties"]["endpoint"] == "str|null"
    assert service_config_schema["properties"]["extra_args"] == "list[str]"
    assert service_runtime_schema["properties"]["command"] == "list[str]"
    assert service_runtime_schema["properties"]["health"]["properties"]["ok"] == "bool"
    assert service_runtime_schema["properties"]["health"]["properties"]["status_code"] == "int|null"
    assert service_metadata_schema["properties"] == {}
    assert (
        CLI_JSON_OUTPUT_SCHEMAS["show-service"]["properties"]["service"]["item_shape"]
        == service_shape
    )
    assert CLI_JSON_OUTPUT_SCHEMAS["register-service"]["properties"]["runtime"] == service_runtime_schema
    assert CLI_JSON_OUTPUT_SCHEMAS["service-command"]["properties"]["command"]["type"] == "list[str]"
    assert CLI_JSON_OUTPUT_SCHEMAS["start-service"]["properties"]["service_id"] == "str"
    assert CLI_JSON_OUTPUT_SCHEMAS["service-health"]["properties"]["error"] == "str|null"
    assert CLI_JSON_OUTPUT_SCHEMAS["stop-service"]["properties"]["status"] == "str"
    assert (
        CLI_JSON_OUTPUT_SCHEMAS["delete-service"]["properties"]["service"][
            "item_shape"
        ]["service_id"]
        == "str"
    )
    assert CLI_JSON_OUTPUT_SCHEMAS["archive-run"]["properties"]["manifest_path"] == "str"
    assert CLI_JSON_OUTPUT_SCHEMAS["delete-run"]["properties"]["trash_path"] == "str|null"
    assert CLI_JSON_OUTPUT_SCHEMAS["evaluate-run"]["required"] == [
        "run_id",
        "report_path",
        "summary_path",
    ]
    assert (
        CLI_JSON_OUTPUT_SCHEMAS["import-predictions"]["properties"][
            "missing_prediction_count"
        ]
        == "int"
    )
    assert CLI_JSON_OUTPUT_SCHEMAS["compare-runs"]["required"] == [
        "comparison_id",
        "baseline_run_id",
        "candidate_run_id",
        "report_path",
    ]
    run_report_schema = CLI_JSON_OUTPUT_SCHEMAS["show-run-report"]
    assert run_report_schema["type"] == "object"
    assert run_report_schema["properties"]["precision_iou50"] == "float"
    assert run_report_schema["properties"]["labels"] == "list[object|str]"
    assert run_report_schema["properties"]["samples"]["item_shape"]["matched_count"] == "int"
    comparisons_output_schema = CLI_JSON_OUTPUT_SCHEMAS["list-comparisons"]
    _assert_paged_schema(comparisons_output_schema)
    _assert_filter_schema(
        comparisons_output_schema,
        ["task", "baseline_run_id", "candidate_run_id", "label", "query"],
    )
    assert comparisons_output_schema["properties"]["comparisons"]["item_shape"]["target_labels"] == "list[str]"
    comparison_shape = comparisons_output_schema["properties"]["comparisons"]["item_shape"]
    assert comparison_shape["delta"]["properties"]["precision_iou50"] == "float"
    assert comparison_shape["delta"]["properties"]["false_negative_count"] == "int"
    assert comparison_shape["summary"]["properties"]["improved_samples"] == "int"
    assert comparison_shape["baseline"]["properties"]["matched_count"] == "int"
    assert comparison_shape["labels"]["item_shape"]["delta"]["properties"]["mean_iou"] == "float"
    assert (
        comparison_shape["top_regressions"]["item_shape"]["baseline"]["item_shape"][
            "false_positive_count"
        ]
        == "int"
    )
    assert (
        CLI_JSON_OUTPUT_SCHEMAS["show-comparison"]["properties"]["summary"]
        == comparison_shape["summary"]
    )
    comparison_sample_output_schema = CLI_JSON_OUTPUT_SCHEMAS["show-comparison-sample"]
    assert comparison_sample_output_schema["properties"]["baseline"]["item_shape"]["pred_instances"] == "list[object]"
    assert (
        comparison_sample_output_schema["properties"]["candidate"]["item_shape"][
            "diagnostics"
        ]["item_shape"]["false_positive_count"]
        == "int"
    )
    assert (
        comparison_sample_output_schema["properties"]["baseline"]["item_shape"][
            "sample"
        ]["item_shape"]["gt_instance_count"]
        == "int"
    )


def test_cli_suppresses_broken_pipe_traceback_for_json_stdout() -> None:
    result = subprocess.run(
        (
            f"{sys.executable} scripts/eval_bench.py rank-board "
            "| head -c 16 >/dev/null"
        ),
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
                "prompt": {"prompt_id": "grounding_arrow.v2.4.main"},
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


def test_cli_lifecycle_commands_emit_agent_json_payloads(tmp_path: Path, capsys) -> None:
    _write_sample_store(tmp_path)

    eval_args = _build_parser().parse_args(
        ["evaluate-run", "--output-root", str(tmp_path), "--run-id", "run_arrow"]
    )
    _cmd_evaluate_run(eval_args)
    eval_payload = json.loads(capsys.readouterr().out)
    _assert_cli_json_payload("evaluate-run", eval_payload)
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
    _assert_cli_json_payload("compare-runs", compare_payload)
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
            "grounding_layout.v2.4.main",
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
            append_payload["updated_at"],
        ]
    )
    _cmd_append_run_note(guarded_append_args)
    guarded_append_payload = json.loads(capsys.readouterr().out)
    _assert_cli_json_payload("append-run-note", guarded_append_payload)
    assert "## agent\nguarded append" in guarded_append_payload["note"]

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
            guarded_append_payload["updated_at"],
        ]
    )
    _cmd_set_run_note(guarded_set_args)
    guarded_payload = json.loads(capsys.readouterr().out)
    _assert_cli_json_payload("set-run-note", guarded_payload)
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
    _assert_cli_json_payload("dashboard-state", state_payload)
    assert state_payload["run_count"] == 1

    archive_args = _build_parser().parse_args(
        ["archive-run", "--output-root", str(tmp_path), "--run-id", "run1"]
    )
    _cmd_archive_run(archive_args)
    archived = json.loads(capsys.readouterr().out)
    _assert_cli_json_payload("archive-run", archived)
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
    _assert_cli_json_payload("backend-logs", backend_logs_payload)
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
    _assert_cli_json_payload("job-logs", job_logs_payload)
    assert job_logs_payload["lines"] == ["step2\n", "step3\n"]

    cancel_args = _build_parser().parse_args(
        ["cancel-job", "--output-root", str(tmp_path), "--job-id", "job1"]
    )
    _cmd_cancel_job(cancel_args)
    cancelled_job = json.loads(capsys.readouterr().out)
    _assert_cli_json_payload("cancel-job", cancelled_job)
    assert cancelled_job["status"] == "cancelled"

    delete_job_args = _build_parser().parse_args(
        ["delete-job", "--output-root", str(tmp_path), "--job-id", "job1"]
    )
    _cmd_delete_job(delete_job_args)
    deleted_job = json.loads(capsys.readouterr().out)
    _assert_cli_json_payload("delete-job", deleted_job)
    assert deleted_job["deleted"] is True
    assert database.get_job("job1") is None

    scheduler_args = _build_parser().parse_args(
        ["scheduler-status", "--output-root", str(tmp_path)]
    )
    _cmd_scheduler_status(scheduler_args)
    scheduler_payload = json.loads(capsys.readouterr().out)
    _assert_cli_json_payload("scheduler-status", scheduler_payload)
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
    _assert_cli_json_payload("register-service", registered_service)
    assert registered_service["service_id"] == "svc1"

    delete_service_args = _build_parser().parse_args(
        ["delete-service", "--output-root", str(tmp_path), "--service-id", "svc1"]
    )
    _cmd_delete_service(delete_service_args)
    deleted_service = json.loads(capsys.readouterr().out)
    _assert_cli_json_payload("delete-service", deleted_service)
    assert deleted_service["service"]["service_id"] == "svc1"

    delete_run_args = _build_parser().parse_args(
        ["delete-run", "--output-root", str(tmp_path), "--run-id", "run1"]
    )
    _cmd_delete_run(delete_run_args)
    deleted_run = json.loads(capsys.readouterr().out)
    _assert_cli_json_payload("delete-run", deleted_run)
    assert deleted_run["deleted"] is True
    assert not (tmp_path / "runs" / "run1").exists()
    assert Path(deleted_run["trash_path"]).exists()


def test_cli_prints_agent_ops_summary(tmp_path: Path, capsys) -> None:
    _write_json(
        tmp_path / "runs" / "run-best" / "run.json",
        {
            "run_id": "run-best",
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
                "target_labels": ["arrow"],
                "prompt": {"prompt_id": "grounding_arrow.v2.4.main"},
                "metric_profile": "detection",
            },
        },
    )
    _write_json(
        tmp_path / "runs" / "run-best" / "reports" / "summary.json",
        {
            "precision_iou50": 0.75,
            "recall_iou50": 0.75,
            "mean_iou": 0.7,
            "prediction_file_count": 3,
        },
    )
    _write_json(
        tmp_path / "runs" / "run-best" / "reports" / "metrics.json",
        {
            "precision_iou50": 0.75,
            "recall_iou50": 0.75,
            "mean_iou": 0.7,
            "prediction_file_count": 3,
        },
    )
    _write_json(
        tmp_path / "runs" / "run-waiting" / "run.json",
        {
            "run_id": "run-waiting",
            "status": "created",
            "created_at": "2026-05-09T00:20:00Z",
            "model": {"model_id": "model-b", "path": "outputs/model-b/best"},
            "benchmark": {
                "benchmark_id": "bench1",
                "root": str(tmp_path / "benchmarks" / "bench1" / "data"),
                "split": "val",
                "tasks": ["detection"],
            },
            "spec": {"task": "detection", "target_labels": ["arrow"]},
        },
    )
    _write_json(
        tmp_path / "runs" / "run-waiting" / "predictions" / "sample.json",
        {"image": "sample.png", "instances": []},
    )
    database = EvalBenchDatabase(tmp_path)
    database.create_job(kind="eval", job_id="queued-job", payload={}, status="queued")
    database.create_job(kind="eval", job_id="running-job", payload={}, status="running")
    database.create_job(kind="eval", job_id="failed-job", payload={}, status="failed")
    database.upsert_service(
        kind="external_vllm",
        service_id="svc1",
        status="registered",
        config={"endpoint": "http://127.0.0.1:8000/v1"},
    )

    args = _build_parser().parse_args(["ops-summary", "--output-root", str(tmp_path)])
    _cmd_ops_summary(args)
    payload = json.loads(capsys.readouterr().out)

    _assert_cli_json_payload("ops-summary", payload)
    assert payload["source"] == "ops_summary"
    assert payload["runs"]["total"] == 2
    assert payload["runs"]["evaluated"] == 1
    assert payload["runs"]["waiting_evaluation"] == 1
    assert payload["runs"]["best_f1"] == 0.75
    assert payload["runs"]["best_f1_run"]["run_id"] == "run-best"
    assert payload["runs"]["best_f1_run"]["target_labels"] == ["arrow"]
    assert payload["benchmarks"] == {
        "total": 0,
        "sample_count": 0,
        "prediction_count": 4,
    }
    assert payload["jobs"] == {
        "total": 3,
        "queued": 1,
        "running": 1,
        "failed": 1,
        "active": 2,
    }
    assert payload["services"]["total"] == 1
    assert payload["scheduler"]["source"] == "cli_snapshot"


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
            "grounding_layout.v2.4.main",
            "--target-label",
            "arrow",
        ]
    )

    _cmd_import_predictions(args)
    payload = json.loads(capsys.readouterr().out)
    _assert_cli_json_payload("import-predictions", payload)
    report = json.loads(Path(payload["report_path"]).read_text(encoding="utf-8"))

    assert payload["run_id"] == "imported_arrow"
    assert report["target_labels"] == ["arrow"]
    assert report["target_labels_source"] == "explicit"
    assert [item["label"] for item in report["labels"]] == ["arrow"]


def test_cli_import_predictions_uses_prompt_template_target_labels(
    tmp_path: Path, capsys
) -> None:
    split_path = tmp_path / "benchmarks" / "bench1" / "splits" / "val.txt"
    data_root = tmp_path / "benchmarks" / "bench1" / "data"
    split_path.parent.mkdir(parents=True)
    split_path.write_text("part1/json/a.json\n", encoding="utf-8")
    _write_json(
        tmp_path / "benchmarks" / "bench1" / "benchmark.json",
        {
            "benchmark_id": "bench1",
            "tasks": ["detection"],
            "labels": ["custom_arrow", "icon"],
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
                {"label": "custom_arrow", "bbox": [200, 200, 260, 260]},
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
                {"label": "custom_arrow", "bbox": [200, 200, 260, 260]},
            ],
        },
    )
    EvalBenchDatabase(tmp_path).upsert_prompt_template(
        {
            "prompt_id": "custom.arrow.import",
            "label": "Custom arrow import",
            "task": "detection",
            "system_prompt": "Inspect diagrams.",
            "user_prompt": "Find custom arrows.",
            "metadata": {"target_labels": ["custom_arrow"], "owner": "bench-team"},
        }
    )
    args = _build_parser().parse_args(
        [
            "import-predictions",
            "--output-root",
            str(tmp_path),
            "--run-id",
            "imported_custom_arrow",
            "--benchmark-id",
            "bench1",
            "--prediction-root",
            str(prediction_root),
            "--task",
            "detection",
            "--model-id",
            "external-model",
            "--prompt-id",
            "custom.arrow.import",
        ]
    )

    _cmd_import_predictions(args)
    payload = json.loads(capsys.readouterr().out)
    _assert_cli_json_payload("import-predictions", payload)
    report = json.loads(Path(payload["report_path"]).read_text(encoding="utf-8"))
    run_payload = json.loads(
        (tmp_path / "runs" / "imported_custom_arrow" / "run.json").read_text(encoding="utf-8")
    )

    assert report["target_labels"] == ["custom_arrow"]
    assert report["target_labels_source"] == "prompt_metadata"
    assert [item["label"] for item in report["labels"]] == ["custom_arrow"]
    assert run_payload["spec"]["prompt"]["metadata"]["target_labels"] == ["custom_arrow"]
    assert run_payload["spec"]["prompt"]["metadata"]["owner"] == "bench-team"


def test_cli_import_predictions_rejects_unknown_target_label(tmp_path: Path) -> None:
    split_path = tmp_path / "benchmarks" / "bench1" / "splits" / "val.txt"
    data_root = tmp_path / "benchmarks" / "bench1" / "data"
    split_path.parent.mkdir(parents=True)
    split_path.write_text("part1/json/a.json\n", encoding="utf-8")
    _write_json(
        tmp_path / "benchmarks" / "bench1" / "benchmark.json",
        {
            "benchmark_id": "bench1",
            "tasks": ["detection"],
            "labels": ["arrow", "icon"],
            "split": "val",
            "sample_count": 1,
            "root": str(data_root),
            "manifest_path": str(split_path),
        },
    )
    prediction_root = tmp_path / "predictions"
    prediction_root.mkdir()
    args = _build_parser().parse_args(
        [
            "import-predictions",
            "--output-root",
            str(tmp_path),
            "--run-id",
            "bad_import",
            "--benchmark-id",
            "bench1",
            "--prediction-root",
            str(prediction_root),
            "--task",
            "detection",
            "--model-id",
            "external-model",
            "--target-label",
            "arrwo",
        ]
    )

    with pytest.raises(
        ValueError,
        match="target_labels not found in benchmark label index: arrwo",
    ):
        _cmd_import_predictions(args)
    assert not (tmp_path / "runs" / "bad_import").exists()


def test_cli_prints_filtered_rank_board(tmp_path: Path, capsys) -> None:
    for run_id, label, split, precision in (
        ("run_a", "icon", "grounding_layout", 0.9),
        ("run_b", "arrow", "grounding_arrow", 0.5),
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
                        "split": split,
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
            "--benchmark-split",
            "grounding_layout",
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
    _assert_cli_json_payload("rank-board", payload)

    assert payload["total"] == 1
    assert payload["primary_metric"] == "weighted_score"
    assert payload["primary_metric_label"] == "agent_weighted_quality"
    assert payload["sort_by"] == "run_id"
    assert payload["sort_order"] == "desc"
    assert payload["filters"]["benchmark_split"] == "grounding_layout"
    assert payload["filters"]["min_score"] == "0.7"
    assert payload["filters"]["rank_scheme"] == "agent_weighted_quality"
    assert payload["rank_scheme"]["name"] == "agent_weighted_quality"
    assert payload["facets"]["splits"] == [{"value": "grounding_layout", "count": 1}]
    assert payload["facets"]["metric_profiles"] == [{"value": "detection_iou_v1", "count": 1}]
    assert payload["entries"][0]["run_id"] == "run_a"
    assert payload["entries"][0]["benchmark_split"] == "grounding_layout"
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
    _assert_cli_json_payload("rank-board", metric_payload)
    assert metric_payload["primary_metric"] == "recall_iou50"
    assert metric_payload["primary_metric_label"] == "R@.50"
    assert metric_payload["score_formula"] == "R@.50"
    assert metric_payload["entries"][0]["run_id"] == "run_a"
    assert metric_payload["entries"][0]["score"] == pytest.approx(0.9)
    assert metric_payload["entries"][0]["score_delta"] == pytest.approx(0.0)
    assert metric_payload["entries"][1]["score_delta"] < 0

    weighted_sort_without_scheme = _build_parser().parse_args(
        [
            "rank-board",
            "--output-root",
            str(tmp_path),
            "--sort-by",
            "weighted_score",
        ]
    )
    with pytest.raises(ValueError, match="requires rank_scheme"):
        _cmd_rank_board(weighted_sort_without_scheme)


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
                "prompt": {"prompt_id": "grounding_arrow.v2.4.main"},
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
                "prompt": {"prompt_id": "keypoint_arrow.test.main"},
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
    assert benchmarks["facets"]["tasks"] == [
        {"value": "detection", "count": 1},
        {"value": "keypoint", "count": 1},
    ]
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
            "--benchmark-split",
            "val",
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
    assert runs["filters"]["benchmark_split"] == "val"
    assert runs["filters"]["label"] == "arrow"
    assert runs["facets"]["labels"] == [{"value": "arrow", "count": 2}]
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
    _assert_cli_json_payload("list-comparisons", comparisons)
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
            "prompt_id": "grounding_arrow.v2.4.main",
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
            "grounding_arrow.v2.4.main",
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
            "keypoint_arrow.test.main",
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


def test_cli_manages_job_and_prompt_templates_for_agents(tmp_path: Path, capsys) -> None:
    job_template_args = _build_parser().parse_args(["list-job-templates", "--query", "arrow"])
    _cmd_list_job_templates(job_template_args)
    job_templates = json.loads(capsys.readouterr().out)
    _assert_cli_json_payload("list-job-templates", job_templates)
    assert job_templates["total"] == 1
    assert "eval_job" in job_templates["templates"]
    assert job_templates["templates"]["eval_job"]["manifest"]["eval"]["task"] == "detection"
    assert (
        job_templates["templates"]["eval_job"]["manifest"]["runtime"]["args"]["max-model-len"]
        == 32768
    )

    show_job_template_args = _build_parser().parse_args(
        ["show-job-template", "--template-id", "eval_job"]
    )
    _cmd_show_job_template(show_job_template_args)
    job_template = json.loads(capsys.readouterr().out)
    _assert_cli_json_payload("show-job-template", job_template)
    assert job_template["template_id"] == "eval_job"
    assert job_template["template"]["manifest"]["eval"]["metric_profile"] == "detection_iou_v1"

    list_args = _build_parser().parse_args(
        ["list-prompt-templates", "--output-root", str(tmp_path), "--task", "detection"]
    )
    _cmd_list_prompt_templates(list_args)
    prompt_templates = json.loads(capsys.readouterr().out)
    _assert_cli_json_payload("list-prompt-templates", prompt_templates)
    arrow_prompt_id = str(DEFAULT_PROMPT_SPECS[0]["prompt_id"])
    assert prompt_templates["total"] >= 1
    assert arrow_prompt_id in prompt_templates["by_id"]
    assert prompt_templates["by_id"][arrow_prompt_id]["task"] == "detection"
    assert prompt_templates["by_id"][arrow_prompt_id]["generation"]["max_tokens"] == 4096
    assert prompt_templates["by_id"][arrow_prompt_id]["data"]["max_pixels"] == 2_000_000
    point_prompt_id = "point_arrow.v2.4.main"
    keypoint_list_args = _build_parser().parse_args(
        ["list-prompt-templates", "--output-root", str(tmp_path), "--task", "keypoint"]
    )
    _cmd_list_prompt_templates(keypoint_list_args)
    keypoint_prompt_templates = json.loads(capsys.readouterr().out)
    _assert_cli_json_payload("list-prompt-templates", keypoint_prompt_templates)
    assert point_prompt_id in keypoint_prompt_templates["by_id"]
    assert keypoint_prompt_templates["by_id"][point_prompt_id]["parser"] == "raw_data_keypoint_v1"
    assert keypoint_prompt_templates["by_id"][point_prompt_id]["metric_profile"] == (
        "keypoint_endpoint_v1"
    )
    assert keypoint_prompt_templates["by_id"][point_prompt_id]["metadata"]["target_labels"] == [
        "arrow"
    ]

    show_prompt_args = _build_parser().parse_args(
        [
            "show-prompt-template",
            "--output-root",
            str(tmp_path),
            "--prompt-id",
            arrow_prompt_id,
        ]
    )
    _cmd_show_prompt_template(show_prompt_args)
    prompt_template = json.loads(capsys.readouterr().out)
    _assert_cli_json_payload("show-prompt-template", prompt_template)
    assert prompt_template["template"]["prompt_id"] == arrow_prompt_id
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
    _assert_cli_json_payload("upsert-prompt-template", upserted)
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


def test_cli_preflights_and_creates_manifest_first_job(tmp_path: Path, capsys) -> None:
    model_path = tmp_path / "models" / "model-a"
    _write_json(model_path / "config.json", {"num_attention_heads": 4})
    (tmp_path / "benchmarks" / "bench1" / "splits").mkdir(parents=True)
    (tmp_path / "benchmarks" / "bench1" / "splits" / "val.txt").write_text(
        "part1/json/a.json\n",
        encoding="utf-8",
    )
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
                    "prompt_id": "grounding_arrow.v2.4.main",
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
    assert jobs["facets"]["kinds"] == [{"value": "eval", "count": 1}]
    assert jobs["jobs"][0]["job_id"] == job["job_id"]

    show_job_args = _build_parser().parse_args(
        ["show-job", "--output-root", str(tmp_path), "--job-id", job["job_id"]]
    )
    _cmd_show_job(show_job_args)
    job_detail = json.loads(capsys.readouterr().out)
    assert job_detail["job"]["job_id"] == job["job_id"]
    assert job_detail["job"]["payload"]["target_labels"] == ["arrow"]
    assert job_detail["job"]["payload"]["manifest"]["kind"] == "eval_job"


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
                    "prompt_id": "grounding_arrow.v2.4.main",
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
                    "prompt_id": "keypoint_arrow.test.main",
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
    (tmp_path / "benchmarks" / "bench1" / "splits").mkdir(parents=True)
    (tmp_path / "benchmarks" / "bench1" / "splits" / "val.txt").write_text(
        "part1/json/a.json\n",
        encoding="utf-8",
    )
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
                    "prompt_id": "grounding_arrow.v2.4.main",
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
