from __future__ import annotations

import logging
import os
from typing import Any

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


def broadcast_object_from_rank_zero(value: Any) -> Any:
    if not (
        dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size() > 1
    ):
        return value
    payload = [value if dist.get_rank() == 0 else None]
    device = None
    try:
        if str(dist.get_backend()).lower() == "nccl" and torch.cuda.is_available():
            device = torch.device("cuda", torch.cuda.current_device())
    except Exception:  # noqa: BLE001 - backend probing must not mask the collective
        device = None
    dist.broadcast_object_list(payload, src=0, device=device)
    return payload[0]


def all_gather_objects(value: Any) -> list[Any]:
    if not (
        dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size() > 1
    ):
        return [value]
    gathered: list[Any] = [None] * int(dist.get_world_size())
    dist.all_gather_object(gathered, value)
    return gathered


def destroy_process_group_if_initialized() -> None:
    if not (dist.is_available() and dist.is_initialized()):
        return
    try:
        dist.destroy_process_group()
    except Exception:  # noqa: BLE001
        logger.warning("failed to destroy distributed process group during shutdown", exc_info=True)
