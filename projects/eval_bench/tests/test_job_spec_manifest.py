from __future__ import annotations

from eval_bench.job_spec import job_templates, resolve_job_payload


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
    assert resolved.payload["trust_remote_code"] is True
    assert "extra_args" not in resolved.payload


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


def test_resolved_job_payload_re_resolves_from_job_manifest() -> None:
    manifest = job_templates()["eval_job"]["manifest"]
    manifest["runtime"]["args"]["model"] = "outputs/model-a/best"
    manifest["runtime"]["engine"] = "dry_run"
    manifest["eval"]["model_id"] = "model-a"
    manifest["eval"]["benchmark_id"] = "bench1"

    first = resolve_job_payload({"manifest": manifest})
    second = resolve_job_payload(first.payload)

    assert second.kind == "eval_job"
    assert second.payload["job_manifest"] == first.payload["job_manifest"]
    assert second.payload["model_id"] == "model-a"
    assert "manifest" not in second.payload
