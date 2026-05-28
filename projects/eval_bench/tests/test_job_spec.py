from __future__ import annotations

import json
from pathlib import Path

import pytest

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
    assert resolved.payload["max_pixels"] == 2_000_000
    assert resolved.payload["batch_size"] == 1
    assert resolved.payload["prompt_id"] == manifest["eval"]["prompt_id"]
    assert resolved.payload["target_labels"] == ["arrow"]
    assert "--trust-remote-code" in resolved.payload["extra_args"]


def test_eval_job_manifest_top_level_run_id_is_worker_run_id() -> None:
    manifest = job_templates()["eval_job"]["manifest"]
    manifest["run_id"] = "banana_v2_4_best__grounding_layout"

    resolved = resolve_job_payload({"manifest": manifest})

    assert resolved.payload["run_id"] == "banana_v2_4_best__grounding_layout"


def test_layout_eval_job_template_remains_available() -> None:
    manifest = job_templates()["layout_eval_job"]["manifest"]
    resolved = resolve_job_payload({"manifest": manifest})

    assert resolved.payload["prompt_id"] == manifest["eval"]["prompt_id"]
    assert resolved.payload["task"] == "detection"
    assert resolved.payload["target_labels"] == ["icon", "image", "shape"]


def test_eval_job_payload_resolves_runtime_target_label_policy() -> None:
    keypoint_manifest = job_templates()["eval_job"]["manifest"]
    keypoint_manifest["eval"] = {
        "task": "keypoint",
        "prompt_id": "point_arrow.test.main",
        "parser": "raw_data_keypoint_v1",
        "metric_profile": "keypoint_endpoint_v1",
    }
    keypoint_resolved = resolve_job_payload({"manifest": keypoint_manifest})

    assert keypoint_resolved.payload["task"] == "keypoint"
    assert keypoint_resolved.payload["target_labels"] == ["arrow"]
    assert keypoint_resolved.payload["target_labels_source"] == "suite_default"

    keypoint_manifest["eval"]["prompt_id"] = "custom_eval"
    keypoint_custom_resolved = resolve_job_payload({"manifest": keypoint_manifest})
    assert keypoint_custom_resolved.payload["target_labels"] == ["arrow"]
    assert keypoint_custom_resolved.payload["target_labels_source"] == "task_default"

    layout_manifest = job_templates()["eval_job"]["manifest"]
    layout_manifest["eval"]["prompt_id"] = "grounding_layout.test.main"
    layout_manifest["eval"].pop("target_labels")
    layout_resolved = resolve_job_payload({"manifest": layout_manifest})

    assert layout_resolved.payload["task"] == "detection"
    assert layout_resolved.payload["target_labels"] == ["icon", "image", "shape"]
    assert layout_resolved.payload["target_labels_source"] == "suite_default"


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


def test_eval_job_manifest_is_only_semantic_source() -> None:
    manifest = {
        "kind": "eval_job",
        "metadata": {"notes": "manifest note"},
        "runtime": {
            "mode": "existing_service",
            "engine": "dry_run",
            "endpoint": "http://manifest-endpoint/v1",
            "service_id": "manifest-service",
            "env": {"CUDA_VISIBLE_DEVICES": "0,1"},
            "args": {
                "model": "outputs/manifest-model/best",
                "served-model-name": "manifest-served",
                "host": "127.0.0.2",
                "port": 8123,
                "tensor-parallel-size": 2,
            },
        },
        "eval": {
            "run_id": "manifest-run",
            "model_id": "manifest-model",
            "benchmark_id": "manifest-bench",
            "benchmark_split": "manifest-split",
            "task": "detection",
            "prompt_id": "manifest.prompt",
            "prompt_text": "manifest prompt",
            "parser": "raw_data_detection_v1",
            "metric_profile": "detection_iou_v1",
            "target_labels": ["icon"],
            "generation": {"max_tokens": 128, "temperature": 0.2, "top_p": 0.9},
            "data": {"max_pixels": 123456, "batch_size": 2},
        },
    }

    resolved = resolve_job_payload(
        {
            "manifest": manifest,
            "kind": "preannotate_job",
            "backend": "vllm_openai",
            "run_id": "top-run",
            "model_id": "top-model",
            "model_path": "outputs/top-model/best",
            "benchmark_id": "top-bench",
            "benchmark_split": "top-split",
            "task": "keypoint",
            "prompt_id": "top.prompt",
            "prompt_text": "top prompt",
            "parser": "top_parser",
            "metric_profile": "top_metric",
            "target_labels": ["arrow"],
            "endpoint": "http://top-endpoint/v1",
            "service_id": "top-service",
            "max_tokens": 999,
            "temperature": 1,
            "top_p": 0.1,
            "max_pixels": 999999,
            "batch_size": 9,
            "metadata": {"notes": "top note"},
            "stray": "must not be copied",
        },
        prompt_templates={
            "top.prompt": {
                "prompt_id": "top.prompt",
                "task": "keypoint",
                "prompt_text": "top template prompt",
                "metadata": {"target_labels": ["arrow"]},
            }
        },
    )

    assert resolved.payload["backend"] == "dry_run"
    assert resolved.payload["run_id"] == "manifest-run"
    assert resolved.payload["model_id"] == "manifest-model"
    assert resolved.payload["model_path"] == "outputs/manifest-model/best"
    assert resolved.payload["served_model_name"] == "manifest-served"
    assert resolved.payload["benchmark_id"] == "manifest-bench"
    assert resolved.payload["benchmark_split"] == "manifest-split"
    assert resolved.payload["task"] == "detection"
    assert resolved.payload["prompt_id"] == "manifest.prompt"
    assert resolved.payload["prompt_text"] == "manifest prompt"
    assert resolved.payload["parser"] == "raw_data_detection_v1"
    assert resolved.payload["metric_profile"] == "detection_iou_v1"
    assert resolved.payload["target_labels"] == ["icon"]
    assert resolved.payload["endpoint"] == "http://manifest-endpoint/v1"
    assert resolved.payload["service_id"] == "manifest-service"
    assert resolved.payload["cuda_visible_devices"] == "0,1"
    assert resolved.payload["tensor_parallel_size"] == 2
    assert resolved.payload["port"] == 8123
    assert resolved.payload["max_tokens"] == 128
    assert resolved.payload["temperature"] == 0.2
    assert resolved.payload["top_p"] == 0.9
    assert resolved.payload["max_pixels"] == 123456
    assert resolved.payload["batch_size"] == 2
    assert resolved.payload["metadata"] == {"notes": "manifest note"}
    assert "stray" not in resolved.payload


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


def test_legacy_eval_payload_is_rejected() -> None:
    with pytest.raises(ValueError, match="manifest-first suite schema"):
        resolve_job_payload(
            {
                "backend": "dry_run",
                "model_id": "model-a",
                "model_path": "outputs/model-a/best",
                "benchmark_id": "bench1",
                "task": "detection",
                "prompt_id": "grounding_layout.test.main",
                "target_labels": ["icon", "image"],
            }
        )


def test_legacy_job_kind_alias_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported job kind: eval"):
        resolve_job_payload(
            {
                "manifest": {
                    "kind": "eval",
                    "runtime": {"mode": "existing_service", "engine": "dry_run"},
                    "eval": {
                        "model_id": "model-a",
                        "model_path": "outputs/model-a/best",
                        "benchmark_id": "bench1",
                        "task": "detection",
                        "prompt_id": "grounding_layout.test.main",
                    },
                }
            }
        )


def test_legacy_preannotate_job_kind_alias_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported job kind: preannotate"):
        resolve_job_payload(
            {
                "manifest": {
                    "kind": "preannotate",
                    "runtime": {"mode": "existing_service", "engine": "dry_run"},
                    "preannotate": {
                        "model_id": "model-a",
                        "source_root": "raw",
                        "output_root": "out",
                    },
                }
            }
        )


def test_preflight_checks_benchmark_model_and_command(tmp_path: Path) -> None:
    model_path = tmp_path / "outputs" / "model" / "best"
    model_path.mkdir(parents=True)
    benchmark_root = tmp_path / "benchmarks" / "bench1"
    (benchmark_root / "splits").mkdir(parents=True)
    (benchmark_root / "splits" / "val.txt").write_text("part1/json/a.json\n", encoding="utf-8")
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


def test_preflight_rejects_missing_benchmark_split_manifest(tmp_path: Path) -> None:
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
    manifest["eval"]["benchmark_id"] = "bench1"

    result = preflight_job_payload({"manifest": manifest}, store_root=tmp_path)

    assert result["ok"] is False
    assert any("benchmark split manifest does not exist" in error for error in result["errors"])


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
                    "prompt_id": "grounding_layout.test.main",
                },
            }
        },
        store_root=tmp_path,
    )

    assert result["ok"] is False
    assert any("preannotate execution is not wired" in error for error in result["errors"])
