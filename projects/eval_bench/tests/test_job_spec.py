from __future__ import annotations

import json
from pathlib import Path

from eval_bench.job_spec import job_templates, preflight_job_payload, resolve_job_payload


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_eval_job_manifest_resolves_to_worker_payload() -> None:
    manifest = job_templates()["eval_job"]["manifest"]
    assert manifest["runtime"]["args"]["model"] == ""
    assert manifest["runtime"]["args"]["served-model-name"] == ""
    assert manifest["runtime"]["args"]["port"] is None
    assert manifest["runtime"]["args"]["tensor-parallel-size"] is None
    assert manifest["eval"]["model_id"] == ""
    assert manifest["eval"]["benchmark_id"] == ""

    manifest["runtime"]["args"]["model"] = "outputs/qwen3vl-sft/4b/banana-v2.1/best"
    manifest["runtime"]["env"]["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    resolved = resolve_job_payload({"manifest": manifest})

    assert resolved.kind == "eval_job"
    assert resolved.payload["job_kind"] == "eval_job"
    assert resolved.payload["runtime_mode"] == "ephemeral"
    assert resolved.payload["backend"] == "vllm_openai"
    assert resolved.payload["model_id"] == "banana-v2.1-best"
    assert resolved.payload["model_path"] == "outputs/qwen3vl-sft/4b/banana-v2.1/best"
    assert resolved.payload["served_model_name"] == "banana-v2.1-best"
    assert resolved.payload["cuda_visible_devices"] == "0,1,2,3"
    assert resolved.payload["tensor_parallel_size"] == 4
    assert int(resolved.payload["port"]) >= 8000
    assert resolved.payload["max_model_len"] == 32768
    assert resolved.payload["max_tokens"] == 4096
    assert resolved.payload["max_pixels"] == 1048576
    assert resolved.payload["batch_size"] == 1
    assert resolved.payload["prompt_id"] == "grounding_arrow.latest"
    assert resolved.payload["target_labels"] == ["arrow"]
    assert "--trust-remote-code" in resolved.payload["extra_args"]


def test_layout_eval_job_template_remains_available() -> None:
    manifest = job_templates()["layout_eval_job"]["manifest"]
    resolved = resolve_job_payload({"manifest": manifest})

    assert resolved.payload["prompt_id"] == "grounding_layout.latest"
    assert resolved.payload["task"] == "detection"
    assert resolved.payload["target_labels"] == ["icon", "image", "shape"]


def test_eval_job_payload_resolves_runtime_target_label_policy() -> None:
    keypoint_manifest = job_templates()["keypoint_eval_job"]["manifest"]
    keypoint_manifest["eval"].pop("target_labels")
    keypoint_resolved = resolve_job_payload({"manifest": keypoint_manifest})

    assert keypoint_resolved.payload["task"] == "keypoint"
    assert keypoint_resolved.payload["target_labels"] == ["arrow"]
    assert keypoint_resolved.payload["target_labels_source"] == "legacy_prompt_id"

    keypoint_manifest["eval"]["prompt_id"] = "custom_eval"
    keypoint_custom_resolved = resolve_job_payload({"manifest": keypoint_manifest})
    assert keypoint_custom_resolved.payload["target_labels"] == ["arrow"]
    assert keypoint_custom_resolved.payload["target_labels_source"] == "task_default"

    layout_manifest = job_templates()["eval_job"]["manifest"]
    layout_manifest["eval"]["prompt_id"] = "grounding_layout.latest"
    layout_manifest["eval"].pop("target_labels")
    layout_resolved = resolve_job_payload({"manifest": layout_manifest})

    assert layout_resolved.payload["task"] == "detection"
    assert layout_resolved.payload["target_labels"] == ["icon", "image", "shape"]
    assert layout_resolved.payload["target_labels_source"] == "legacy_prompt_id"


def test_eval_job_manifest_preserves_unknown_runtime_args_as_cli_extra_args() -> None:
    manifest = job_templates()["eval_job"]["manifest"]
    manifest = {
        **manifest,
        "runtime": {
            **manifest["runtime"],
            "args": {
                **manifest["runtime"]["args"],
                "limit-mm-per-prompt": {"image": 1},
                "disable-log-requests": True,
            },
        },
    }

    resolved = resolve_job_payload({"manifest": manifest})

    assert "--limit-mm-per-prompt" in resolved.payload["extra_args"]
    assert '{"image": 1}' in resolved.payload["extra_args"]
    assert "--disable-log-requests" in resolved.payload["extra_args"]


def test_eval_job_manifest_resolves_prompt_template_defaults() -> None:
    manifest = job_templates()["eval_job"]["manifest"]
    manifest = {
        **manifest,
        "eval": {
            "model_id": "model-a",
            "benchmark_id": "bench1",
            "prompt_id": "custom.layout",
        },
    }

    resolved = resolve_job_payload(
        {"manifest": manifest},
        prompt_templates={
            "custom.layout": {
                "prompt_id": "custom.layout",
                "label": "Custom Layout",
                "task": "detection",
                "system_prompt": "JSON only.",
                "user_prompt": "Detect icons.",
                "parser": "raw_data_detection_v1",
                "metric_profile": "detection_iou_v1",
                "generation": {"max_tokens": 2048},
                "data": {"max_pixels": 123456, "batch_size": 2},
                "metadata": {"target_labels": ["icon"]},
            }
        },
    )

    assert resolved.manifest["eval"]["prompt_text"] == "Detect icons."
    assert resolved.payload["system_prompt"] == "JSON only."
    assert resolved.payload["prompt_text"] == "Detect icons."
    assert resolved.payload["task"] == "detection"
    assert resolved.payload["parser"] == "raw_data_detection_v1"
    assert resolved.payload["max_tokens"] == 2048
    assert resolved.payload["max_pixels"] == 123456
    assert resolved.payload["batch_size"] == 2
    assert resolved.manifest["eval"]["target_labels"] == ["icon"]
    assert resolved.manifest["eval"]["target_labels_source"] == "prompt_metadata"
    assert resolved.payload["target_labels"] == ["icon"]
    assert resolved.payload["target_labels_source"] == "prompt_metadata"


def test_prompt_template_target_labels_replace_empty_manifest_list() -> None:
    resolved = resolve_job_payload(
        {
            "manifest": {
                "kind": "eval_job",
                "runtime": {"mode": "existing_service", "engine": "vllm_openai"},
                "eval": {
                    "model_id": "model-a",
                    "benchmark_id": "bench1",
                    "prompt_id": "custom.arrow",
                    "target_labels": [],
                },
            }
        },
        prompt_templates={
            "custom.arrow": {
                "prompt_id": "custom.arrow",
                "label": "Custom Arrow",
                "task": "detection",
                "system_prompt": "JSON only.",
                "user_prompt": "Detect arrows.",
                "parser": "raw_data_detection_v1",
                "metric_profile": "detection_iou_v1",
                "metadata": {"target_labels": ["arrow"]},
            }
        },
    )

    assert resolved.manifest["eval"]["target_labels"] == ["arrow"]
    assert resolved.manifest["eval"]["target_labels_source"] == "prompt_metadata"
    assert resolved.payload["target_labels"] == ["arrow"]
    assert resolved.payload["target_labels_source"] == "prompt_metadata"


def test_legacy_eval_payload_preserves_target_labels_in_resolved_manifest() -> None:
    resolved = resolve_job_payload(
        {
            "backend": "dry_run",
            "model_id": "model-a",
            "model_path": "outputs/model-a/best",
            "benchmark_id": "bench1",
            "task": "detection",
            "prompt_id": "grounding_layout.latest",
            "target_labels": ["icon", "image"],
        }
    )

    assert resolved.manifest["eval"]["target_labels"] == ["icon", "image"]
    assert resolved.payload["target_labels"] == ["icon", "image"]


def test_preflight_checks_benchmark_model_and_command(tmp_path: Path) -> None:
    model_path = tmp_path / "outputs" / "model" / "best"
    model_path.mkdir(parents=True)
    benchmark_root = tmp_path / "benchmarks" / "bench1"
    _write_json(
        benchmark_root / "benchmark.json",
        {
            "benchmark_id": "bench1",
            "tasks": ["detection"],
            "layers": ["layout"],
            "split": "val",
            "sample_count": 1,
            "root": str(benchmark_root / "data"),
            "manifest_path": str(benchmark_root / "splits" / "val.txt"),
        },
    )
    manifest = job_templates()["eval_job"]["manifest"]
    manifest["runtime"]["args"]["model"] = str(model_path)
    manifest["runtime"]["args"]["port"] = 65530
    manifest["eval"]["benchmark_id"] = "bench1"

    result = preflight_job_payload({"manifest": manifest}, store_root=tmp_path)

    assert result["ok"] is True
    assert result["errors"] == []
    assert result["resolved_payload"]["benchmark_id"] == "bench1"
    assert "--model" in result["runtime_command"]


def test_preflight_reports_missing_required_eval_inputs(tmp_path: Path) -> None:
    result = preflight_job_payload({"manifest": {"kind": "eval_job", "runtime": {}, "eval": {}}}, store_root=tmp_path)

    assert result["ok"] is False
    assert any("model_id" in error for error in result["errors"])
    assert any("benchmark_id" in error for error in result["errors"])


def test_preflight_rejects_tensor_parallel_size_not_dividing_attention_heads(tmp_path: Path) -> None:
    model_path = tmp_path / "outputs" / "model" / "best"
    model_path.mkdir(parents=True)
    _write_json(model_path / "config.json", {"text_config": {"num_attention_heads": 32}})
    benchmark_root = tmp_path / "benchmarks" / "bench1"
    _write_json(
        benchmark_root / "benchmark.json",
        {
            "benchmark_id": "bench1",
            "tasks": ["detection"],
            "layers": ["arrow"],
            "split": "val",
            "sample_count": 1,
            "root": str(benchmark_root / "data"),
            "manifest_path": str(benchmark_root / "splits" / "val.txt"),
        },
    )
    manifest = job_templates()["eval_job"]["manifest"]
    manifest["runtime"]["args"]["model"] = str(model_path)
    manifest["runtime"]["args"]["tensor-parallel-size"] = 3
    manifest["eval"]["benchmark_id"] = "bench1"

    result = preflight_job_payload({"manifest": manifest}, store_root=tmp_path)

    assert result["ok"] is False
    assert any("attention heads" in error and "tp=3" in error for error in result["errors"])


def test_preflight_rejects_unwired_preannotate_job(tmp_path: Path) -> None:
    source_root = tmp_path / "raw_data"
    source_manifest = source_root / "splits" / "val.txt"
    source_manifest.parent.mkdir(parents=True)
    source_manifest.write_text("", encoding="utf-8")

    result = preflight_job_payload(
        {
            "manifest": {
                "kind": "preannotate_job",
                "runtime": {"mode": "existing_service", "engine": "vllm_openai"},
                "preannotate": {
                    "source_root": str(source_root),
                    "source_manifest": str(source_manifest),
                    "output_root": str(tmp_path / "preannotations"),
                    "task": "detection",
                    "prompt_id": "grounding_layout.latest",
                },
            }
        },
        store_root=tmp_path,
    )

    assert result["ok"] is False
    assert any("preannotate execution is not wired" in error for error in result["errors"])
