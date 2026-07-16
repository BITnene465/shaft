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


def initialize_process_group_if_needed(*, use_cpu: bool) -> None:
    """Initialize torchrun's default group before fallible rank-local setup.

    Hugging Face normally initializes the group while resolving
    ``TrainingArguments.device``.  Shaft needs an earlier status collective so
    plugin/config/TrainingArguments construction itself cannot strand peers.
    Under torchrun the rendezvous environment is already authoritative.  The
    resolved training intent is authoritative for backend selection: an
    explicit CPU run must use Gloo even when CUDA devices are visible.
    """

    if type(use_cpu) is not bool:
        raise TypeError("Distributed process-group use_cpu intent must be a boolean.")
    if not dist.is_available() or dist.is_initialized() or get_world_size() <= 1:
        return
    use_cuda = not use_cpu and torch.cuda.is_available()
    if use_cuda:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl" if use_cuda else "gloo")


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
