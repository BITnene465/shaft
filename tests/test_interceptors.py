from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from unittest.mock import patch
from uuid import uuid4

import pytest

import shaft.pipeline.execution as execution_module
from shaft.plugins import (
    ExecutionProxy,
    build_interceptor_manager,
    interceptor,
    interceptable,
)
from shaft.plugins.interceptors import FunctionInterceptor, InterceptorManager
from shaft.pipeline.execution import prepare_pipeline_call


def test_interceptable_allows_pre_post_mutation() -> None:
    run_id = uuid4().hex
    before_name = f"{run_id}_before"
    after_name = f"{run_id}_after"
    events: list[str] = []

    @interceptor("unit.point", phase="before", name=before_name, order=10)
    def _before(state: dict) -> None:
        events.append("before")
        state["kwargs"]["x"] = int(state["kwargs"]["x"]) + 1

    @interceptor("unit.point", phase="after", name=after_name, order=90)
    def _after(state: dict) -> None:
        events.append("after")
        state["result"] = int(state["result"]) * 2

    manager = build_interceptor_manager([before_name, after_name])

    class _Runner:
        def __init__(self, local_manager):
            self.local_manager = local_manager

        @interceptable(
            "unit.point", manager_getter=lambda self, *args, **kwargs: self.local_manager
        )
        def run(self, *, x: int) -> int:
            return int(x)

    output = _Runner(manager).run(x=3)
    assert output == 8
    assert events == ["before", "after"]


def test_execution_proxy_wraps_callable() -> None:
    run_id = uuid4().hex
    before_name = f"{run_id}_before"
    points: list[str] = []

    @interceptor("unit.proxy.call", phase="before", name=before_name, order=5)
    def _capture_point(state: dict) -> None:
        points.append(str(state.get("point")))

    manager = build_interceptor_manager([before_name])
    proxy = ExecutionProxy(
        point="unit.proxy.call",
        target=lambda value: int(value) + 5,
        interceptor_manager=manager,
    )
    result = proxy(7)
    assert result == 12
    assert points == ["unit.proxy.call"]


def test_execution_proxy_can_converge_before_phase_before_invoking_target() -> None:
    run_id = uuid4().hex
    before_name = f"{run_id}_before"
    after_name = f"{run_id}_after"
    events: list[str] = []

    @interceptor("unit.proxy.split", phase="before", name=before_name)
    def _before(state: dict) -> None:
        events.append("before")
        state["kwargs"]["value"] += 1

    @interceptor("unit.proxy.split", phase="after", name=after_name)
    def _after(state: dict) -> None:
        events.append("after")
        state["result"] *= 2

    def _target(*, value: int) -> int:
        events.append("target")
        return value

    proxy = ExecutionProxy(
        point="unit.proxy.split",
        target=_target,
        interceptor_manager=build_interceptor_manager([before_name, after_name]),
    )
    invocation = proxy.prepare(value=3)

    assert events == ["before"]
    assert proxy.invoke(invocation) == 8
    assert events == ["before", "target", "after"]


def test_execution_proxy_does_not_run_after_interceptors_when_target_fails() -> None:
    run_id = uuid4().hex
    after_name = f"{run_id}_after"
    events: list[str] = []

    @interceptor("unit.proxy.error", phase="after", name=after_name)
    def _after(_state: dict) -> None:
        events.append("after")

    def _target() -> int:
        events.append("target")
        raise LookupError("intentional target failure")

    proxy = ExecutionProxy(
        point="unit.proxy.error",
        target=_target,
        interceptor_manager=build_interceptor_manager([after_name]),
    )

    with pytest.raises(LookupError, match="intentional target failure"):
        proxy()
    assert events == ["target"]


def test_execution_proxy_preserves_original_arguments_when_state_keys_are_removed() -> None:
    interceptor_name = f"remove_call_state_{uuid4().hex}"

    @interceptor("unit.proxy.removed-state", phase="before", name=interceptor_name)
    def _remove_call_state(state: dict) -> None:
        state.pop("args")
        state.pop("kwargs")

    proxy = ExecutionProxy(
        point="unit.proxy.removed-state",
        target=lambda value, *, offset: int(value) + int(offset),
        interceptor_manager=build_interceptor_manager([interceptor_name]),
    )

    assert proxy(5, offset=7) == 12


def test_pipeline_before_interceptor_cannot_change_zero_argument_contract() -> None:
    interceptor_name = f"mutate_pipeline_args_{uuid4().hex}"
    target_called = False

    @interceptor("unit.pipeline", phase="before", name=interceptor_name)
    def _mutate_arguments(state: dict) -> None:
        state["kwargs"]["unexpected"] = True

    def _target() -> None:
        nonlocal target_called
        target_called = True

    proxy = ExecutionProxy(
        point="unit.pipeline",
        target=_target,
        interceptor_manager=build_interceptor_manager([interceptor_name]),
    )

    with pytest.raises(ValueError, match="zero-argument call contract"):
        prepare_pipeline_call(
            proxy,
            stage="unit-before-interceptors",
        )
    assert target_called is False


def test_pipeline_target_runs_outside_before_status_envelope() -> None:
    original_stage = execution_module.distributed_training_contract_stage
    status_depth = 0

    @contextmanager
    def _tracked_stage(*args, **kwargs):
        nonlocal status_depth
        status_depth += 1
        try:
            with original_stage(*args, **kwargs):
                yield
        finally:
            status_depth -= 1

    def _target() -> int:
        assert status_depth == 0
        return 7

    proxy = ExecutionProxy(point="unit.pipeline", target=_target)
    with patch.object(
        execution_module,
        "distributed_training_contract_stage",
        _tracked_stage,
    ):
        invocation = prepare_pipeline_call(
            proxy,
            stage="unit-before-interceptors",
        )
        assert status_depth == 0
        assert proxy.invoke(invocation) == 7


def test_interceptor_schedule_fingerprint_binds_same_name_implementation() -> None:
    def _first(state: dict) -> None:
        state["implementation"] = "first"

    def _second(state: dict) -> None:
        state["implementation"] = "second"

    def _manager(fn) -> InterceptorManager:
        return InterceptorManager(
            interceptors=[
                FunctionInterceptor(
                    name="same-name",
                    point="unit.schedule",
                    phase="before",
                    order=10,
                    fn=fn,
                    shaft_trajectory_neutral=True,
                )
            ]
        )

    first = _manager(_first).semantic_schedule_fingerprint(point="unit.schedule")
    equivalent = _manager(_first).semantic_schedule_fingerprint(point="unit.schedule")
    second = _manager(_second).semantic_schedule_fingerprint(point="unit.schedule")

    assert first == equivalent
    assert first != second


def test_interceptor_schedule_fingerprint_binds_stable_equal_order_positions() -> None:
    def _observe(_state: dict) -> None:
        return None

    def _handler(name: str) -> FunctionInterceptor:
        return FunctionInterceptor(
            name=name,
            point="unit.schedule-order",
            phase="before",
            order=10,
            fn=_observe,
            shaft_trajectory_neutral=True,
        )

    forward = InterceptorManager(
        interceptors=[_handler("first"), _handler("second")]
    ).semantic_schedule_fingerprint(point="unit.schedule-order")
    reversed_order = InterceptorManager(
        interceptors=[_handler("second"), _handler("first")]
    ).semantic_schedule_fingerprint(point="unit.schedule-order")

    assert forward != reversed_order


def test_interceptor_schedule_fingerprint_binds_instance_configuration() -> None:
    @dataclass
    class _ConfiguredInterceptor:
        delta: int
        name: str = "same-name"
        point: str = "unit.schedule-state"
        phase: str = "before"
        order: int = 10
        shaft_trajectory_neutral: bool = True

        def __call__(self, state: dict) -> None:
            state["value"] = int(state.get("value", 0)) + self.delta

    first = InterceptorManager(
        interceptors=[_ConfiguredInterceptor(delta=1)]
    ).semantic_schedule_fingerprint(point="unit.schedule-state")
    equivalent = InterceptorManager(
        interceptors=[_ConfiguredInterceptor(delta=1)]
    ).semantic_schedule_fingerprint(point="unit.schedule-state")
    changed = InterceptorManager(
        interceptors=[_ConfiguredInterceptor(delta=2)]
    ).semantic_schedule_fingerprint(point="unit.schedule-state")

    assert first == equivalent
    assert first != changed


def test_interceptor_decorator_rejects_string_trajectory_neutral() -> None:
    with pytest.raises(TypeError, match="trajectory_neutral.*boolean"):
        interceptor(
            "unit.invalid-neutral",
            name=f"bad_neutral_{uuid4().hex}",
            trajectory_neutral="false",  # type: ignore[arg-type]
        )
