from __future__ import annotations

from uuid import uuid4

import pytest

from shaft.plugins import HOOK_REGISTRY, TrainerHookCallback, build_hook_manager, hook


def test_builtin_hook_registered() -> None:
    assert HOOK_REGISTRY.has("log_before_step")
    assert HOOK_REGISTRY.has("log_on_save")


def test_hook_decorator_and_callback_dispatch() -> None:
    event_values: list[int] = []
    hook_name = f"unit_before_step_{uuid4().hex}"

    @hook("before_step", name=hook_name)
    def _before_step(state: dict) -> None:
        event_values.append(int(state["marker"]))

    manager = build_hook_manager([hook_name])
    callback = TrainerHookCallback(manager)
    callback.on_step_begin(args=None, state=object(), control=object(), marker=3)
    assert event_values == [3]


def test_invalid_event_raises() -> None:
    with pytest.raises(ValueError):

        @hook("not_a_real_event", name=f"bad_{uuid4().hex}")
        def _bad(_: dict) -> None:
            return None
