from __future__ import annotations


def eval_job_payload(
    *,
    model_id: str,
    model_path: str,
    benchmark_id: str,
    task: str,
    prompt_id: str,
    backend: str = "vllm_openai",
    runtime_mode: str = "existing_service",
    benchmark_split: str | None = None,
    target_labels: list[str] | None = None,
    served_model_name: str | None = None,
    endpoint: str | None = None,
    service_id: str | None = None,
    system_prompt: str | None = None,
    prompt_text: str | None = None,
    cuda_visible_devices: str | None = None,
    tensor_parallel_size: int | None = None,
    port: int | None = None,
    max_model_len: int | None = None,
    gpu_memory_utilization: float | None = None,
    max_num_seqs: int | None = None,
    trust_remote_code: bool | None = None,
    generation_config: str | None = None,
    dtype: str | None = None,
    kv_cache_dtype: str | None = None,
    load_format: str | None = None,
    max_num_batched_tokens: int | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_pixels: int | None = None,
    batch_size: int | None = None,
    metadata: dict | None = None,
) -> dict:
    runtime_args = {
        "model": model_path,
        "served-model-name": served_model_name or model_id,
        "host": "127.0.0.1",
        "port": port,
        "tensor-parallel-size": tensor_parallel_size,
        "max-model-len": max_model_len,
        "gpu-memory-utilization": gpu_memory_utilization,
        "max-num-seqs": max_num_seqs,
        "trust-remote-code": trust_remote_code,
        "generation-config": generation_config,
        "dtype": dtype,
        "kv-cache-dtype": kv_cache_dtype,
        "load-format": load_format,
        "max-num-batched-tokens": max_num_batched_tokens,
    }
    payload = {
        "manifest": {
            "kind": "eval_job",
            "runtime": {
                "mode": runtime_mode,
                "engine": backend,
                "endpoint": endpoint,
                "service_id": service_id,
                "env": {"CUDA_VISIBLE_DEVICES": cuda_visible_devices},
                "args": {
                    key: value for key, value in runtime_args.items() if value not in (None, "")
                },
            },
            "eval": {
                "model_id": model_id,
                "benchmark_id": benchmark_id,
                "benchmark_split": benchmark_split or "",
                "task": task,
                "prompt_id": prompt_id,
                "system_prompt": system_prompt,
                "prompt_text": prompt_text,
                "target_labels": list(target_labels or []),
                "generation": {
                    key: value
                    for key, value in {
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                    }.items()
                    if value is not None
                },
                "data": {
                    key: value
                    for key, value in {
                        "max_pixels": max_pixels,
                        "batch_size": batch_size,
                    }.items()
                    if value is not None
                },
            },
        }
    }
    if metadata is not None:
        payload["manifest"]["metadata"] = metadata
    return payload


def ephemeral_eval_job_payload() -> dict:
    return eval_job_payload(
        model_id="served-model",
        model_path="outputs/model-a/best",
        benchmark_id="bench1",
        task="detection",
        prompt_id="grounding_layout.test.main",
        runtime_mode="ephemeral",
        cuda_visible_devices="0",
        served_model_name="served-model",
        port=8000,
        tensor_parallel_size=1,
        max_model_len=32768,
        gpu_memory_utilization=0.9,
        max_num_seqs=8,
        prompt_text="detect icons",
        max_tokens=16,
        temperature=0,
        top_p=1,
        batch_size=1,
        max_pixels=1048576,
    )
