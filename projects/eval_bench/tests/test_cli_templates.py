from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval_bench.cli import (
    _build_parser,
    _cmd_delete_prompt_template,
    _cmd_list_job_templates,
    _cmd_list_prompt_templates,
    _cmd_show_job_template,
    _cmd_show_prompt_template,
    _cmd_upsert_prompt_template,
)
from eval_bench.prompt_templates import DEFAULT_PROMPT_SPECS
from support.cli_contracts import assert_cli_json_payload as _assert_cli_json_payload


pytestmark = pytest.mark.contract


def test_cli_lists_and_shows_job_templates_for_agents(capsys) -> None:
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


def test_cli_lists_and_shows_repo_prompt_templates(tmp_path: Path, capsys) -> None:
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


def test_cli_lists_keypoint_prompt_templates(tmp_path: Path, capsys) -> None:
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


def test_cli_upserts_lists_and_deletes_custom_prompt_template(
    tmp_path: Path,
    capsys,
) -> None:
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
