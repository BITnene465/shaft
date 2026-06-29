from __future__ import annotations

import pytest

from eval_bench import runtime_resources
from eval_bench.job_spec import job_templates, resolve_job_payload


def test_eval_job_manifest_auto_runtime_placement_uses_detected_gpus(monkeypatch) -> None:
    monkeypatch.setattr(
        runtime_resources,
        "detect_cuda_devices",
        lambda: [runtime_resources.GpuInfo(str(index)) for index in range(8)],
    )
    manifest = job_templates()["eval_job"]["manifest"]
    manifest["runtime"]["args"]["model"] = "outputs/qwen3vl-sft/32b/zero-shot"

    resolved = resolve_job_payload({"manifest": manifest})

    assert resolved.payload["cuda_visible_devices"] == "0,1,2,3,4,5,6,7"
    assert resolved.payload["tensor_parallel_size"] == 8


def test_eval_job_manifest_explicit_tp_selects_matching_detected_gpus(monkeypatch) -> None:
    monkeypatch.setattr(
        runtime_resources,
        "detect_cuda_devices",
        lambda: [runtime_resources.GpuInfo(str(index)) for index in range(8)],
    )
    manifest = job_templates()["eval_job"]["manifest"]
    manifest["runtime"]["args"]["model"] = "outputs/qwen3vl-sft/32b/zero-shot"
    manifest["runtime"]["args"]["tensor-parallel-size"] = 4

    resolved = resolve_job_payload({"manifest": manifest})

    assert resolved.payload["cuda_visible_devices"] == "0,1,2,3"
    assert resolved.payload["tensor_parallel_size"] == 4


def test_eval_job_manifest_rejects_unknown_runtime_args() -> None:
    manifest = job_templates()["eval_job"]["manifest"]
    manifest = {
        **manifest,
        "runtime": {
            **manifest["runtime"],
            "args": {
                **manifest["runtime"]["args"],
                "disable-log-requests": True,
            },
        },
    }

    with pytest.raises(ValueError, match="unknown runtime.args key"):
        resolve_job_payload({"manifest": manifest})


def test_eval_job_manifest_resolves_common_vllm_runtime_args_as_first_class_fields() -> None:
    manifest = job_templates()["eval_job"]["manifest"]
    manifest = {
        **manifest,
        "runtime": {
            **manifest["runtime"],
            "args": {
                **manifest["runtime"]["args"],
                "trust-remote-code": True,
                "generation-config": "vllm",
                "dtype": "bfloat16",
                "kv-cache-dtype": "auto",
                "quantization": "fp8",
                "load-format": "auto",
                "enforce-eager": True,
                "disable-custom-all-reduce": True,
                "max-num-batched-tokens": 8192,
                "limit-mm-per-prompt": {"image": 1},
            },
        },
    }

    resolved = resolve_job_payload({"manifest": manifest})

    assert resolved.payload["trust_remote_code"] is True
    assert resolved.payload["generation_config"] == "vllm"
    assert resolved.payload["dtype"] == "bfloat16"
    assert resolved.payload["kv_cache_dtype"] == "auto"
    assert resolved.payload["quantization"] == "fp8"
    assert resolved.payload["load_format"] == "auto"
    assert resolved.payload["enforce_eager"] is True
    assert resolved.payload["disable_custom_all_reduce"] is True
    assert resolved.payload["max_num_batched_tokens"] == 8192
    assert resolved.payload["limit_mm_per_prompt"] == {"image": 1}
    assert "extra_args" not in resolved.payload


@pytest.mark.parametrize(
    "runtime_args",
    [
        {"max-pixels": 1_000_000},
        {"mm-processor-kwargs": {"max_pixels": 1_000_000}},
    ],
)
def test_eval_job_manifest_rejects_runtime_pixel_budget_args(
    runtime_args: dict[str, object],
) -> None:
    manifest = job_templates()["eval_job"]["manifest"]
    manifest["runtime"]["args"].update(runtime_args)

    with pytest.raises(ValueError, match="pixel budget belongs to eval.data"):
        resolve_job_payload({"manifest": manifest})


def test_eval_job_manifest_rejects_runtime_extra_args() -> None:
    manifest = job_templates()["eval_job"]["manifest"]
    manifest["runtime"]["extra_args"] = ["--disable-log-requests"]

    with pytest.raises(ValueError, match="runtime.extra_args is not supported"):
        resolve_job_payload({"manifest": manifest})
