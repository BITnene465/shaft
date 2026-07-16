from __future__ import annotations

from shaft.observability import emit_event
from .hooks import hook


@hook("before_step", name="log_before_step", trajectory_neutral=True)
def log_before_step(state: dict) -> None:
    trainer_state = state.get("trainer_state")
    global_step = getattr(trainer_state, "global_step", 0)
    if isinstance(global_step, int) and global_step % 100 == 0:
        emit_event("train.step.marker", global_step=global_step)


@hook("on_save", name="log_on_save", trajectory_neutral=True)
def log_on_save(state: dict) -> None:
    trainer_state = state.get("trainer_state")
    global_step = getattr(trainer_state, "global_step", "unknown")
    emit_event("train.checkpoint.saved", global_step=global_step)
