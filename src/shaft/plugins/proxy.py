from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, ParamSpec, TypeVar

from .interceptors import InterceptorManager, interceptable

P = ParamSpec("P")
R = TypeVar("R")


@dataclass
class ExecutionProxy(Generic[P, R]):
    """Callable proxy with interceptor support."""

    point: str
    target: Callable[P, R]
    interceptor_manager: InterceptorManager | None = None

    @interceptable(lambda self, *args, **kwargs: self.point, manager_getter=lambda self, *args, **kwargs: self.interceptor_manager)
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return self.target(*args, **kwargs)
