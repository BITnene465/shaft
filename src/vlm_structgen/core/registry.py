from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Protocol

from vlm_structgen.core.routing import (
    normalize_domain_type,
    normalize_route_key,
    normalize_task_type,
    parse_route_key,
)

_TASK_ADAPTER_BUILDERS: dict[str, Callable[..., "TaskAdapter"]] = {}
_ROUTE_BINDINGS: dict[str, "RouteBinding"] = {}


@dataclass(frozen=True)
class RouteBinding:
    route_key: str
    task_type: str
    domain_type: str


class TaskAdapter(Protocol):
    task_type: str
    domain_type: str
    num_bins: int

    def build_gt_struct_from_record(self, record: dict) -> dict:
        ...

    def encode_target_text(self, gt_struct: dict, *, image_width: int, image_height: int) -> str:
        ...

    def decode(self, text: str, *, image_width: int, image_height: int, strict: bool = False) -> dict:
        ...

    def decode_with_meta(
        self,
        text: str,
        *,
        image_width: int,
        image_height: int,
        strict: bool = False,
    ) -> tuple[dict, dict]:
        ...

    def empty_prediction(self) -> dict:
        ...

    def score_prediction(
        self,
        gt_struct: dict,
        pred_struct: dict,
        *,
        eval_options: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        ...

    def summarize_eval_counts(self, counts: dict[str, float]) -> dict[str, float]:
        ...

    def default_eval_primary_metric(self) -> str:
        ...


def register_task_adapter(task_type: str, builder: Callable[..., TaskAdapter]) -> None:
    normalized_task_type = normalize_task_type(task_type)
    if not callable(builder):
        raise ValueError(f"Task adapter builder for {normalized_task_type!r} must be callable.")
    existing_builder = _TASK_ADAPTER_BUILDERS.get(normalized_task_type)
    if existing_builder is not None and existing_builder is not builder:
        raise ValueError(
            f"Duplicate task adapter registration for task_type={normalized_task_type!r}."
        )
    _TASK_ADAPTER_BUILDERS[normalized_task_type] = builder
    get_adapter.cache_clear()
    get_adapter_for_route.cache_clear()


def list_registered_task_types() -> list[str]:
    return sorted(_TASK_ADAPTER_BUILDERS.keys())


def register_route_binding(
    *,
    route_key: str,
    task_type: str | None = None,
    domain_type: str | None = None,
) -> None:
    normalized_route_key = normalize_route_key(route_key)
    if task_type is None or domain_type is None:
        parsed_task_type, parsed_domain_type = parse_route_key(normalized_route_key)
        task_type = task_type or parsed_task_type
        domain_type = domain_type or parsed_domain_type
    binding = RouteBinding(
        route_key=normalized_route_key,
        task_type=normalize_task_type(task_type),
        domain_type=normalize_domain_type(domain_type),
    )
    existing = _ROUTE_BINDINGS.get(normalized_route_key)
    if existing is not None and existing != binding:
        raise ValueError(
            f"Duplicate route binding with conflicting payload for route={normalized_route_key!r}: "
            f"{existing} vs {binding}."
        )
    _ROUTE_BINDINGS[normalized_route_key] = binding
    resolve_route_binding.cache_clear()
    get_adapter_for_route.cache_clear()


def register_routes(route_keys: list[str] | tuple[str, ...] | set[str]) -> None:
    for route_key in route_keys:
        register_route_binding(route_key=str(route_key))


def list_registered_routes() -> list[str]:
    return sorted(_ROUTE_BINDINGS.keys())


@lru_cache(maxsize=256)
def resolve_route_binding(route_key: str | None) -> RouteBinding:
    normalized_route_key = normalize_route_key(route_key)
    existing = _ROUTE_BINDINGS.get(normalized_route_key)
    if existing is not None:
        return existing
    task_type, domain_type = parse_route_key(normalized_route_key)
    return RouteBinding(
        route_key=normalized_route_key,
        task_type=task_type,
        domain_type=domain_type,
    )


@lru_cache(maxsize=64)
def get_adapter(
    *,
    task_type: str | None,
    domain_type: str | None,
    num_bins: int,
    task_options_key: tuple[tuple[str, object], ...] = (),
) -> TaskAdapter:
    normalized_task_type = normalize_task_type(task_type)
    normalized_domain_type = normalize_domain_type(domain_type)
    task_options: dict[str, Any] = dict(task_options_key)
    builder = _TASK_ADAPTER_BUILDERS.get(normalized_task_type)
    if builder is None:
        raise ValueError(
            "No task adapter registered for "
            f"task_type={normalized_task_type!r}. Registered task types: {list_registered_task_types()}. "
            "Register adapters before dataset/eval/infer, e.g. via vlm_structgen.tasks.bootstrap."
        )
    return builder(
        domain_type=normalized_domain_type,
        num_bins=num_bins,
        task_options=task_options,
    )


@lru_cache(maxsize=256)
def get_adapter_for_route(
    *,
    route_key: str,
    num_bins: int,
    task_options_key: tuple[tuple[str, object], ...] = (),
) -> TaskAdapter:
    resolved_binding = resolve_route_binding(route_key)
    return get_adapter(
        task_type=resolved_binding.task_type,
        domain_type=resolved_binding.domain_type,
        num_bins=num_bins,
        task_options_key=task_options_key,
    )
