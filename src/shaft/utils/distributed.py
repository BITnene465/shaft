from __future__ import annotations

import logging
import os

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank())
    return int(os.environ.get("RANK", "0"))


def get_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_world_size())
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_distributed() -> bool:
    return get_world_size() > 1


def is_rank_zero() -> bool:
    return get_rank() == 0


def _nccl_barrier_kwargs() -> dict[str, list[int]]:
    try:
        backend = str(dist.get_backend()).lower()
    except Exception:  # noqa: BLE001
        return {}
    if backend != "nccl" or not torch.cuda.is_available():
        return {}
    return {"device_ids": [int(torch.cuda.current_device())]}


def barrier_if_distributed() -> None:
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.barrier(**_nccl_barrier_kwargs())


def destroy_process_group_if_initialized() -> None:
    if not (dist.is_available() and dist.is_initialized()):
        return
    try:
        dist.destroy_process_group()
    except Exception:  # noqa: BLE001
        logger.warning("failed to destroy distributed process group during shutdown", exc_info=True)
