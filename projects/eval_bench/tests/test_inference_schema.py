from __future__ import annotations

import pytest

from eval_bench.schema import InferenceParams


def test_inference_params_validate_service_launcher_fields() -> None:
    params = InferenceParams(
        backend="vllm_openai",
        service_id="local-vllm-0",
        cuda_visible_devices="0,1",
        tensor_parallel_size=2,
        port=8000,
        max_model_len=65536,
        gpu_memory_utilization=0.9,
        max_num_seqs=16,
    )

    params.validate()

    with pytest.raises(ValueError, match="gpu_memory_utilization"):
        InferenceParams(gpu_memory_utilization=1.2).validate()
