from __future__ import annotations

import time

from shaft.observability import emit_event
from .interceptors import interceptor


@interceptor(
    "pipeline.sft.run",
    phase="before",
    name="trace_sft_start",
    order=10,
    trajectory_neutral=True,
)
def trace_sft_start(state: dict) -> None:
    state["start_time"] = time.perf_counter()
    emit_event("pipeline.sft.start", point=state.get("point", "pipeline.sft.run"))


@interceptor(
    "pipeline.sft.run",
    phase="after",
    name="trace_sft_done",
    order=90,
    trajectory_neutral=True,
)
def trace_sft_done(state: dict) -> None:
    _emit_pipeline_done("pipeline.sft.done", state)


@interceptor(
    "pipeline.rl.run",
    phase="before",
    name="trace_rl_start",
    order=10,
    trajectory_neutral=True,
)
def trace_rl_start(state: dict) -> None:
    state["start_time"] = time.perf_counter()
    emit_event("pipeline.rl.start", point=state.get("point", "pipeline.rl.run"))


@interceptor(
    "pipeline.rl.run",
    phase="after",
    name="trace_rl_done",
    order=90,
    trajectory_neutral=True,
)
def trace_rl_done(state: dict) -> None:
    _emit_pipeline_done("pipeline.rl.done", state)


@interceptor(
    "pipeline.opd.run",
    phase="before",
    name="trace_opd_start",
    order=10,
    trajectory_neutral=True,
)
def trace_opd_start(state: dict) -> None:
    state["start_time"] = time.perf_counter()
    emit_event("pipeline.opd.start", point=state.get("point", "pipeline.opd.run"))


@interceptor(
    "pipeline.opd.run",
    phase="after",
    name="trace_opd_done",
    order=90,
    trajectory_neutral=True,
)
def trace_opd_done(state: dict) -> None:
    _emit_pipeline_done("pipeline.opd.done", state)


def _emit_pipeline_done(event_name: str, state: dict) -> None:
    start_time = state.get("start_time")
    if isinstance(start_time, float):
        elapsed = time.perf_counter() - start_time
        emit_event(event_name, elapsed_s=f"{elapsed:.3f}")
