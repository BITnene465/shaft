from __future__ import annotations

import time

from shaft.observability import emit_event
from .interceptors import interceptor


@interceptor("pipeline.train.run", phase="before", name="trace_train_start", order=10)
def trace_train_start(state: dict) -> None:
    state["start_time"] = time.perf_counter()
    emit_event("pipeline.train.start", point=state.get("point", "pipeline.train.run"))


@interceptor("pipeline.train.run", phase="after", name="trace_train_done", order=90)
def trace_train_done(state: dict) -> None:
    start_time = state.get("start_time")
    if isinstance(start_time, float):
        elapsed = time.perf_counter() - start_time
        emit_event("pipeline.train.done", elapsed_s=f"{elapsed:.3f}")
