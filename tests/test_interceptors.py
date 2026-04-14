from __future__ import annotations

from uuid import uuid4

from shaft.plugins import ExecutionProxy, build_interceptor_manager, interceptor, interceptable


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

        @interceptable("unit.point", manager_getter=lambda self, *args, **kwargs: self.local_manager)
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
