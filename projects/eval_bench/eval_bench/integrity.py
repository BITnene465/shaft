from __future__ import annotations

from typing import Any


OFFICIAL_BENCHMARK_TYPES = {"official"}


def benchmark_type(payload: dict[str, Any] | None) -> str:
    if not payload:
        return "missing"
    value = str(payload.get("benchmark_type") or "").strip()
    if value:
        return value
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    metadata_type = str((metadata or {}).get("benchmark_type") or "").strip()
    return metadata_type or "official"


def benchmark_is_official(payload: dict[str, Any] | None) -> bool:
    if not payload:
        return False
    kind = benchmark_type(payload)
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    official = (metadata or {}).get("official", True)
    return kind in OFFICIAL_BENCHMARK_TYPES and official is not False


def run_integrity(
    *,
    benchmark_id: str,
    benchmark_payload: dict[str, Any] | None,
    benchmark_official: bool,
) -> tuple[str, str]:
    if not benchmark_id:
        return "missing_benchmark", "run manifest does not declare benchmark_id"
    if benchmark_payload is None:
        return "missing_benchmark", "benchmark manifest is missing"
    if not benchmark_official:
        return "non_official_benchmark", "benchmark is excluded from official reporting"
    return "ok", ""
