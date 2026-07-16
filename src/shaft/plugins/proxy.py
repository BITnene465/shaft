from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Generic, ParamSpec, TypeVar, cast

from .interceptors import (
    InterceptorManager,
    ShaftInterceptorInvocation,
    invoke_interceptor_invocation,
    prepare_interceptor_invocation,
)

P = ParamSpec("P")
R = TypeVar("R")


@dataclass
class ExecutionProxy(Generic[P, R]):
    """Callable proxy with interceptor support."""

    point: str
    target: Callable[P, R]
    interceptor_manager: InterceptorManager | None = None

    def prepare(self, *args: P.args, **kwargs: P.kwargs) -> ShaftInterceptorInvocation:
        """Run local ``before`` interceptors without entering the target."""

        return prepare_interceptor_invocation(
            point=self.point,
            manager=self.interceptor_manager,
            args=cast(tuple[Any, ...], args),
            kwargs=cast(dict[str, Any], kwargs),
        )

    def invoke(self, invocation: ShaftInterceptorInvocation) -> R:
        """Invoke the target outside any caller-owned readiness envelope."""

        return invoke_interceptor_invocation(invocation, self.target)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return self.invoke(self.prepare(*args, **kwargs))
