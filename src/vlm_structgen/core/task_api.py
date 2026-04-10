from __future__ import annotations

from .registry import (
    TaskAdapter,
    get_adapter,
    get_adapter_for_route,
    list_registered_task_types,
    list_registered_routes,
    normalize_domain_type,
    parse_route_key,
    register_route_binding,
    register_routes,
    normalize_task_type,
    resolve_route_binding,
)

__all__ = [
    "TaskAdapter",
    "get_adapter",
    "get_adapter_for_route",
    "list_registered_routes",
    "list_registered_task_types",
    "normalize_domain_type",
    "parse_route_key",
    "register_route_binding",
    "register_routes",
    "resolve_route_binding",
    "normalize_task_type",
]
