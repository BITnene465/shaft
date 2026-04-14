from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Any, Iterator

_LOG_CONTEXT: ContextVar[dict[str, Any]] = ContextVar("shaft_log_context", default={})


def get_log_context() -> dict[str, Any]:
    return dict(_LOG_CONTEXT.get())


def set_log_context(**fields: Any) -> None:
    base = get_log_context()
    base.update(fields)
    _LOG_CONTEXT.set(base)


@contextmanager
def bind_log_context(**fields: Any) -> Iterator[None]:
    base = get_log_context()
    merged = dict(base)
    merged.update(fields)
    token: Token[dict[str, Any]] = _LOG_CONTEXT.set(merged)
    try:
        yield
    finally:
        _LOG_CONTEXT.reset(token)
