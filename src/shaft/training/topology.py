from __future__ import annotations

import os

import torch

from shaft.config import RuntimeConfig


def _get_int_env(name: str) -> int | None:
    raw_value = os.environ.get(name)
    if raw_value is None or str(raw_value).strip() == "":
        return None
    try:
        return int(str(raw_value).strip())
    except ValueError:
        return None


def is_distributed_launch() -> bool:
    world_size = _get_int_env("WORLD_SIZE")
    local_rank = _get_int_env("LOCAL_RANK")
    rank = _get_int_env("RANK")
    return bool((world_size is not None and world_size > 1) or local_rank is not None or rank is not None)


def validate_training_topology(config: RuntimeConfig) -> None:
    if bool(config.train.use_cpu) or not torch.cuda.is_available():
        return

    visible_cuda_count = int(torch.cuda.device_count())
    if visible_cuda_count <= 1 or is_distributed_launch():
        return

    raise RuntimeError(
        "Unsafe single-process multi-GPU training detected: "
        f"{visible_cuda_count} CUDA devices are visible but the process was not launched with torchrun/DDP. "
        "Hugging Face Trainer would use torch.nn.DataParallel, which is incompatible with multimodal "
        "variable-length visual tensors such as Qwen3VL pixel_values/image_grid_thw and can corrupt the "
        "alignment between image patches and grid metadata. "
        "Restrict the run to one GPU, for example `CUDA_VISIBLE_DEVICES=1 python scripts/train.py ...`, "
        "or launch distributed training with torchrun."
    )
