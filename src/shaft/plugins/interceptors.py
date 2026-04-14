from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Literal, ParamSpec, TypeVar, cast

from .registry import Registry

HookPhase = Literal["before", "after"]
InterceptorFn = Callable[[dict[str, Any]], None]
INTERCEPTOR_PHASES: set[HookPhase] = {"before", "after"}
INTERCEPTOR_REGISTRY: Registry[Callable[[], "Interceptor"] | "Interceptor"] = Registry("interceptor")

P = ParamSpec("P")
R = TypeVar("R")


class Interceptor:
    name: str
    point: str
    phase: HookPhase
    order: int

    def __call__(self, state: dict[str, Any]) -> None: ...


@dataclass
class FunctionInterceptor:
    name: str
    point: str
    phase: HookPhase
    order: int
    fn: InterceptorFn

    def __call__(self, state: dict[str, Any]) -> None:
        self.fn(state)


def interceptor(
    point: str,
    *,
    phase: HookPhase = "before",
    name: str | None = None,
    order: int = 100,
):
    normalized_point = str(point).strip().lower()
    if not normalized_point:
        raise ValueError("Interceptor point cannot be empty.")
    normalized_phase = str(phase).strip().lower()
    if normalized_phase not in INTERCEPTOR_PHASES:
        allowed = ", ".join(sorted(INTERCEPTOR_PHASES))
        raise ValueError(f"Unsupported interceptor phase {phase!r}. Allowed: {allowed}")

    def _decorator(fn: InterceptorFn):
        interceptor_name = (name or fn.__name__).strip().lower()
        if not interceptor_name:
            raise ValueError("Interceptor name cannot be empty.")

        def _builder() -> Interceptor:
            return FunctionInterceptor(
                name=interceptor_name,
                point=normalized_point,
                phase=cast(HookPhase, normalized_phase),
                order=int(order),
                fn=fn,
            )

        INTERCEPTOR_REGISTRY.register(interceptor_name, _builder)
        return fn

    return _decorator


@dataclass
class InterceptorManager:
    interceptors: list[Interceptor]

    def __post_init__(self) -> None:
        self._index: dict[tuple[str, HookPhase], list[Interceptor]] = {}
        for interceptor_obj in self.interceptors:
            key = (interceptor_obj.point, interceptor_obj.phase)
            self._index.setdefault(key, []).append(interceptor_obj)
        for key in list(self._index.keys()):
            self._index[key] = sorted(self._index[key], key=lambda x: x.order)

    def emit(self, *, point: str, phase: HookPhase, state: dict[str, Any]) -> None:
        key = (str(point).strip().lower(), str(phase).strip().lower())
        handlers = self._index.get(cast(tuple[str, HookPhase], key), [])
        for handler in handlers:
            handler(state)


def build_interceptor_manager(enabled_interceptors: list[str] | None) -> InterceptorManager:
    if not enabled_interceptors:
        return InterceptorManager(interceptors=[])
    interceptors = [INTERCEPTOR_REGISTRY.create(name) for name in enabled_interceptors]
    return InterceptorManager(interceptors=interceptors)


def interceptable(
    point: str | Callable[P, str],
    *,
    manager_getter: Callable[P, InterceptorManager | None] | None = None,
):
    def _decorator(fn: Callable[P, R]) -> Callable[P, R]:
        @wraps(fn)
        def _wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            resolved_point = point(*args, **kwargs) if callable(point) else str(point)
            normalized_point = str(resolved_point).strip().lower()
            manager = manager_getter(*args, **kwargs) if manager_getter is not None else None
            if manager is None:
                return fn(*args, **kwargs)

            state: dict[str, Any] = {
                "point": normalized_point,
                "args": args,
                "kwargs": kwargs,
            }
            manager.emit(point=normalized_point, phase="before", state=state)
            call_args = state.get("args", args)
            call_kwargs = state.get("kwargs", kwargs)
            result = fn(*call_args, **call_kwargs)
            state["result"] = result
            manager.emit(point=normalized_point, phase="after", state=state)
            return cast(R, state.get("result", result))

        return _wrapped

    return _decorator
