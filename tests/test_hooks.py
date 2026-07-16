from __future__ import annotations

import logging
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


def test_hook_decorator_rejects_string_trajectory_neutral() -> None:
    with pytest.raises(TypeError, match="trajectory_neutral.*boolean"):
        hook(
            "after_step",
            name=f"bad_neutral_{uuid4().hex}",
            trajectory_neutral="false",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("event", "callback_method"),
    [
        ("before_step", "on_step_begin"),
        ("after_step", "on_step_end"),
        ("on_save", "on_save"),
    ],
)
def test_neutral_hook_failure_warns_once_and_disables_observer(
    caplog: pytest.LogCaptureFixture,
    event: str,
    callback_method: str,
) -> None:
    calls: list[str] = []
    healthy_calls: list[str] = []
    hook_name = f"neutral_failure_{uuid4().hex}"
    healthy_hook_name = f"neutral_healthy_{uuid4().hex}"

    @hook(event, name=hook_name, trajectory_neutral=True)
    def _failing_observer(_: dict) -> None:
        calls.append(event)
        raise RuntimeError("synthetic neutral telemetry failure")

    @hook(event, name=healthy_hook_name, trajectory_neutral=True)
    def _healthy_observer(_: dict) -> None:
        healthy_calls.append(event)

    callback = TrainerHookCallback(
        build_hook_manager([hook_name, healthy_hook_name])
    )
    dispatch = getattr(callback, callback_method)
    with caplog.at_level(logging.WARNING, logger="shaft.plugins.hooks"):
        dispatch(args=None, state=object(), control=object())
        dispatch(args=None, state=object(), control=object())

    assert calls == [event]
    assert healthy_calls == [event, event]
    warning_messages = [
        record.getMessage()
        for record in caplog.records
        if "synthetic neutral telemetry failure" in record.getMessage()
    ]
    assert len(warning_messages) == 1
    assert hook_name in warning_messages[0]


def test_non_neutral_hook_failure_is_not_silently_disabled() -> None:
    calls = 0
    hook_name = f"trajectory_failure_{uuid4().hex}"

    @hook("before_step", name=hook_name, trajectory_neutral=False)
    def _failing_hook(_: dict) -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("synthetic trajectory failure")

    callback = TrainerHookCallback(build_hook_manager([hook_name]))
    for _ in range(2):
        with pytest.raises(RuntimeError, match="synthetic trajectory failure"):
            callback.on_step_begin(args=None, state=object(), control=object())

    assert calls == 2
