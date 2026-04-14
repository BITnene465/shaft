from __future__ import annotations

import os

import torch.distributed as dist


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


def barrier_if_distributed() -> None:
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.barrier()

