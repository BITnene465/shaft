from __future__ import annotations

import base64
import re
from collections.abc import Mapping


def normalize_task_type(task_type: str | None) -> str:
    normalized = str(task_type or "").strip().lower()
    if not normalized:
        raise ValueError("task_type is required.")
    return normalized


def normalize_domain_type(domain_type: str | None) -> str:
    normalized = str(domain_type or "").strip().lower()
    if not normalized:
        raise ValueError("domain_type is required.")
    return normalized


def normalize_route_key(route_key: str | None) -> str:
    normalized = str(route_key or "").strip().lower()
    if not normalized:
        raise ValueError("route is required.")
    return normalized


def compose_route_key(task_type: str | None, domain_type: str | None) -> str:
    return f"{normalize_task_type(task_type)}/{normalize_domain_type(domain_type)}"


def parse_route_key(route_key: str | None) -> tuple[str, str]:
    normalized_route_key = normalize_route_key(route_key)
    parts = normalized_route_key.split("/", 1)
    if len(parts) != 2:
        raise ValueError(
            f"Invalid route={normalized_route_key!r}. Expected '<task_type>/<domain_type>'."
        )
    task_type, domain_type = parts
    return normalize_task_type(task_type), normalize_domain_type(domain_type)


def resolve_record_route_key(record: Mapping[str, object]) -> str:
    route_key = record.get("route")
    if route_key is not None and str(route_key).strip():
        return normalize_route_key(str(route_key))
    task_type = record.get("task_type")
    domain_type = record.get("domain_type")
    if str(task_type or "").strip() and str(domain_type or "").strip():
        return compose_route_key(str(task_type), str(domain_type))
    raise ValueError(
        "Sample is missing route binding. Expected `route`, or both `task_type/domain_type` as legacy fallback."
    )


def encode_route_token(route_key: str) -> str:
    normalized_route = normalize_route_key(route_key)
    encoded = base64.urlsafe_b64encode(normalized_route.encode("utf-8")).decode("ascii")
    return encoded.rstrip("=")


def decode_route_token(route_token: str) -> str:
    token = str(route_token).strip()
    if not token:
        raise ValueError("route token is required.")
    padding = "=" * (-len(token) % 4)
    decoded = base64.urlsafe_b64decode((token + padding).encode("ascii")).decode("utf-8")
    return normalize_route_key(decoded)


def route_metric_label(route_key: str) -> str:
    normalized_route = normalize_route_key(route_key)
    rendered = re.sub(r"[^a-z0-9_]+", "__", normalized_route).strip("_")
    return rendered or "route"
