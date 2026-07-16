from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
import hashlib
import json
from typing import Any, Callable, Literal, ParamSpec, TypeVar, cast

from shaft.utils.semantic_identity import (
    callable_semantic_fingerprint,
    component_semantic_fingerprint,
)

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
    shaft_trajectory_neutral: bool

    def __call__(self, state: dict[str, Any]) -> None: ...


@dataclass
class FunctionInterceptor:
    name: str
    point: str
    phase: HookPhase
    order: int
    fn: InterceptorFn
    shaft_trajectory_neutral: bool = False

    def __call__(self, state: dict[str, Any]) -> None:
        self.fn(state)


def interceptor(
    point: str,
    *,
    phase: HookPhase = "before",
    name: str | None = None,
    order: int = 100,
    trajectory_neutral: bool = False,
):
    if type(trajectory_neutral) is not bool:
        raise TypeError("interceptor trajectory_neutral must be a boolean.")
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
                shaft_trajectory_neutral=trajectory_neutral,
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

    def semantic_schedule_fingerprint(self, *, point: str) -> str:
        """Fingerprint the exact ordered schedule executed for one point."""

        normalized_point = str(point).strip().lower()
        schedule = []
        for phase in ("before", "after"):
            handlers = self._index.get((normalized_point, phase), [])
            schedule.extend(
                {
                    "phase": phase,
                    "position": position,
                    "name": str(handler.name),
                    "order": int(handler.order),
                    "component_implementation": component_semantic_fingerprint(
                        handler,
                        role="pipeline_interceptor_schedule",
                        # Instance configuration can change interceptor behavior
                        # even when type/name/order and __call__ code are equal.
                        include_state=True,
                    ),
                    "call_implementation": callable_semantic_fingerprint(
                        handler.fn
                        if isinstance(handler, FunctionInterceptor)
                        else handler.__call__,
                        role="pipeline_interceptor_schedule_call",
                    ),
                }
                for position, handler in enumerate(handlers)
            )
        payload = {
            "version": "shaft-interceptor-schedule-v2",
            "point": normalized_point,
            "schedule": schedule,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ShaftInterceptorInvocation:
    """A call whose local ``before`` interceptors have already completed."""

    point: str
    manager: InterceptorManager | None
    state: dict[str, Any]
    original_args: tuple[Any, ...]
    original_kwargs: dict[str, Any]


def prepare_interceptor_invocation(
    *,
    point: str,
    manager: InterceptorManager | None,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> ShaftInterceptorInvocation:
    """Run only the local ``before`` phase and retain its mutable call state."""

    normalized_point = str(point).strip().lower()
    state: dict[str, Any] = {
        "point": normalized_point,
        "args": args,
        "kwargs": kwargs,
    }
    if manager is not None:
        manager.emit(point=normalized_point, phase="before", state=state)
    return ShaftInterceptorInvocation(
        point=normalized_point,
        manager=manager,
        state=state,
        original_args=args,
        original_kwargs=kwargs,
    )


def invoke_interceptor_invocation(
    invocation: ShaftInterceptorInvocation,
    fn: Callable[..., R],
) -> R:
    """Invoke the target and then the successful-call ``after`` phase."""

    state = invocation.state
    call_args = state.get("args", invocation.original_args)
    call_kwargs = state.get("kwargs", invocation.original_kwargs)
    result = fn(*call_args, **call_kwargs)
    state["result"] = result
    if invocation.manager is not None:
        invocation.manager.emit(
            point=invocation.point,
            phase="after",
            state=state,
        )
    return cast(R, state.get("result", result))


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
            manager = manager_getter(*args, **kwargs) if manager_getter is not None else None
            invocation = prepare_interceptor_invocation(
                point=resolved_point,
                manager=manager,
                args=cast(tuple[Any, ...], args),
                kwargs=cast(dict[str, Any], kwargs),
            )
            return invoke_interceptor_invocation(invocation, fn)

        return _wrapped

    return _decorator
