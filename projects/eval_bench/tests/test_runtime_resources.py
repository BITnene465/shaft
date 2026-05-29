from __future__ import annotations

import json
from pathlib import Path

from eval_bench import runtime_resources


def test_runtime_placement_uses_largest_attention_head_divisor(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"text_config": {"num_attention_heads": 32}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        runtime_resources,
        "detect_cuda_devices",
        lambda: [runtime_resources.GpuInfo(str(index)) for index in range(6)],
    )

    placement = runtime_resources.resolve_vllm_runtime_placement(
        model_path=model_dir,
        cuda_visible_devices=None,
        tensor_parallel_size=None,
    )

    assert placement.cuda_visible_devices == "0,1,2,3"
    assert placement.tensor_parallel_size == 4


def test_runtime_placement_preserves_explicit_tensor_parallel_size(monkeypatch) -> None:
    monkeypatch.setattr(
        runtime_resources,
        "detect_cuda_devices",
        lambda: [runtime_resources.GpuInfo(str(index)) for index in range(8)],
    )

    placement = runtime_resources.resolve_vllm_runtime_placement(
        model_path=None,
        cuda_visible_devices=None,
        tensor_parallel_size=2,
    )

    assert placement.cuda_visible_devices == "0,1"
    assert placement.tensor_parallel_size == 2
