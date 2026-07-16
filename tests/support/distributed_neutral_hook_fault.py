from __future__ import annotations

import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
from transformers.trainer_callback import (
    CallbackHandler,
    TrainerControl,
    TrainerState,
)

from shaft.plugins.hooks import FunctionHook, HookManager, TrainerHookCallback


def _run_event(event: str) -> None:
    calls = 0

    def _rank_local_observer_failure(_: dict) -> None:
        nonlocal calls
        calls += 1
        if dist.get_rank() == 1:
            raise RuntimeError(f"synthetic rank-local neutral {event} failure")

    hook = FunctionHook(
        name=f"distributed_{event}_observer",
        shaft_trajectory_neutral=True,
        **{f"{event}_fn": _rank_local_observer_failure},
    )
    callback = TrainerHookCallback(HookManager(hooks=[hook]))
    handler = CallbackHandler(
        [callback],
        model=None,
        processing_class=None,
        optimizer=None,
        lr_scheduler=None,
    )
    state = TrainerState()
    control = TrainerControl()
    args = SimpleNamespace()
    if event == "before_step":
        dispatch = handler.on_step_begin
    elif event == "after_step":
        dispatch = handler.on_step_end
    elif event == "on_save":
        # Calling the unmodified HF callback handler models backend-native
        # checkpoint dispatch: ShaftCheckpointCommitMixin deliberately does not
        # replace CallbackHandler.on_save for FSDP/DeepSpeed checkpoints.
        dispatch = handler.on_save
    else:  # pragma: no cover - script-internal invariant
        raise AssertionError(f"unexpected event: {event}")

    dispatch(args, state, control)
    dispatch(args, state, control)

    peer_calls: list[int | None] = [None] * dist.get_world_size()
    dist.all_gather_object(peer_calls, calls)
    if peer_calls != [2, 1]:
        raise AssertionError(f"neutral observer was not disabled per rank: {peer_calls!r}")

    # If the rank-local telemetry exception escaped, rank zero would reach this
    # collective alone and the test would time out/fail.
    marker = torch.tensor([dist.get_rank() + 1], dtype=torch.int64)
    dist.all_reduce(marker)
    if int(marker.item()) != 3:
        raise AssertionError(f"unexpected collective result: {marker.item()}")


def main() -> None:
    dist.init_process_group(backend="gloo")
    try:
        for event in ("before_step", "after_step", "on_save"):
            _run_event(event)
        if dist.get_rank() == 0:
            print("neutral hook distributed isolation ok", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
