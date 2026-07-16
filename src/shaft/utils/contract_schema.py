from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def strict_json_loads(
    document: str | bytes | bytearray,
    *,
    role: str,
) -> Any:
    """Parse one canonical JSON document without lossy parser extensions.

    Python's default decoder silently keeps the last duplicate object key and
    accepts JavaScript-style ``NaN``/``Infinity`` constants.  Neither behavior
    is safe for versioned resume contracts because two byte streams can then
    acquire an ambiguous in-memory meaning.
    """

    def reject_constant(constant: str) -> Any:
        raise ValueError(f"{role} contains non-finite JSON constant {constant!r}.")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key, value in pairs:
            if key in payload:
                raise ValueError(f"{role} contains duplicate JSON object key {key!r}.")
            payload[key] = value
        return payload

    payload = json.loads(
        document,
        object_pairs_hook=reject_duplicate_keys,
        parse_constant=reject_constant,
    )
    return validate_json_value(payload, role=role)


def load_strict_json(path: str | Path, *, role: str) -> Any:
    """Read and strictly parse one UTF-8 JSON file."""

    return strict_json_loads(Path(path).read_text(encoding="utf-8"), role=role)


def require_json_mapping(value: Any, *, role: str) -> dict[str, Any]:
    """Require a canonical JSON object, not an arbitrary Python mapping."""

    if type(value) is not dict:
        raise TypeError(f"{role} must be a JSON mapping.")
    if any(type(key) is not str for key in value):
        raise TypeError(f"{role} keys must be JSON strings.")
    return value


def require_exact_keys(
    payload: dict[str, Any],
    *,
    expected: frozenset[str],
    role: str,
) -> None:
    mapping = require_json_mapping(payload, role=role)
    actual = frozenset(mapping)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        raise ValueError(
            f"{role} schema differs from the exact versioned schema; "
            f"missing required fields={missing}, unknown fields={unknown}."
        )


def json_string(payload: dict[str, Any], field_name: str, *, role: str) -> str:
    value = payload[field_name]
    if type(value) is not str:
        raise TypeError(f"{role}.{field_name} must be a JSON string.")
    return value


def json_optional_string(
    payload: dict[str, Any],
    field_name: str,
    *,
    role: str,
) -> str | None:
    value = payload[field_name]
    if value is None:
        return None
    if type(value) is not str:
        raise TypeError(f"{role}.{field_name} must be null or a JSON string.")
    return value


def json_bool(payload: dict[str, Any], field_name: str, *, role: str) -> bool:
    value = payload[field_name]
    if type(value) is not bool:
        raise TypeError(f"{role}.{field_name} must be a JSON boolean.")
    return value


def json_int(payload: dict[str, Any], field_name: str, *, role: str) -> int:
    value = payload[field_name]
    if type(value) is not int:
        raise TypeError(f"{role}.{field_name} must be a JSON integer.")
    return value


def json_int_value(value: Any, *, role: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{role} must be a JSON integer.")
    return value


def json_optional_int(
    payload: dict[str, Any],
    field_name: str,
    *,
    role: str,
) -> int | None:
    value = payload[field_name]
    if value is None:
        return None
    if type(value) is not int:
        raise TypeError(f"{role}.{field_name} must be null or a JSON integer.")
    return value


def json_number(value: Any, *, role: str) -> float:
    if type(value) not in {int, float}:
        raise TypeError(f"{role} must be a JSON number.")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{role} must be finite.")
    return resolved


def json_optional_number(
    payload: dict[str, Any],
    field_name: str,
    *,
    role: str,
) -> float | None:
    value = payload[field_name]
    if value is None:
        return None
    return json_number(value, role=f"{role}.{field_name}")


def json_list(payload: dict[str, Any], field_name: str, *, role: str) -> list[Any]:
    value = payload[field_name]
    if type(value) is not list:
        raise TypeError(f"{role}.{field_name} must be a JSON list.")
    return value


def json_mapping(
    payload: dict[str, Any],
    field_name: str,
    *,
    role: str,
) -> dict[str, Any]:
    return require_json_mapping(
        payload[field_name],
        role=f"{role}.{field_name}",
    )


def validate_json_value(value: Any, *, role: str) -> Any:
    """Validate a recursively canonical in-memory JSON value without coercion."""

    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{role} must contain only finite JSON numbers.")
        return value
    if type(value) is list:
        return [
            validate_json_value(item, role=f"{role}[{index}]") for index, item in enumerate(value)
        ]
    if type(value) is dict:
        mapping = require_json_mapping(value, role=role)
        return {
            key: validate_json_value(item, role=f"{role}.{key}") for key, item in mapping.items()
        }
    raise TypeError(
        f"{role} must contain only canonical JSON null, boolean, integer, finite "
        "number, string, list, or mapping values."
    )
