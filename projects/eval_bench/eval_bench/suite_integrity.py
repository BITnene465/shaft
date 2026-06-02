from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schema import EvalSuiteManifest, SuiteTaskSplit


SUITE_INTEGRITY_OK = "ok"
SUITE_INTEGRITY_NON_OFFICIAL = "non_official_suite"
SUITE_INTEGRITY_INVALID_MANIFEST = "invalid_manifest"
SUITE_INTEGRITY_MISSING_SPLIT = "missing_split_manifest"
SUITE_INTEGRITY_SAMPLE_MISMATCH = "sample_count_mismatch"


@dataclass(frozen=True)
class SuiteIntegrityResult:
    status: str
    reason: str
    errors: list[str]

    @property
    def ok(self) -> bool:
        return self.status == SUITE_INTEGRITY_OK


def validate_suite_manifest_payload(
    payload: dict[str, Any],
    *,
    manifest_path: Path,
) -> SuiteIntegrityResult:
    """Validate an explicit suite manifest as an official reporting source."""
    errors: list[str] = []
    benchmark_type = str(payload.get("benchmark_type") or "official").strip() or "official"
    official = _official_flag(payload, default=benchmark_type == "official")
    if not official:
        return SuiteIntegrityResult(
            status=SUITE_INTEGRITY_NON_OFFICIAL,
            reason="suite is excluded from official reporting",
            errors=[],
        )

    suite_manifest = _suite_manifest_from_payload(payload, manifest_path=manifest_path, errors=errors)
    if suite_manifest is not None:
        try:
            suite_manifest.validate()
        except ValueError as exc:
            errors.append(str(exc))
        _validate_task_splits(
            suite_manifest,
            suite_manifest_path=manifest_path,
            errors=errors,
        )
        _validate_sample_universe(suite_manifest, errors=errors)

    if not errors:
        return SuiteIntegrityResult(status=SUITE_INTEGRITY_OK, reason="", errors=[])
    status = _status_from_errors(errors)
    return SuiteIntegrityResult(status=status, reason=errors[0], errors=errors)


def validate_derived_suite(
    *,
    official: bool,
    task_splits: list[dict[str, Any]],
    sample_universe: dict[str, Any],
) -> SuiteIntegrityResult:
    if not official:
        return SuiteIntegrityResult(
            status=SUITE_INTEGRITY_NON_OFFICIAL,
            reason="suite is excluded from official reporting",
            errors=[],
        )
    errors: list[str] = []
    split_names: set[str] = set()
    total_count = 0
    for item in task_splits:
        split = str(item.get("split") or "").strip()
        manifest_ref = str(item.get("manifest_path") or "").strip()
        sample_count = _safe_int(item.get("sample_count"), default=0)
        total_count += sample_count
        if not split:
            errors.append("task_splits[].split must be a non-empty string.")
        elif split in split_names:
            errors.append(f"duplicate task split: {split}")
        else:
            split_names.add(split)
        if not manifest_ref:
            errors.append(f"task split {split or '<empty>'} is missing manifest_path.")
            continue
        manifest_path = Path(manifest_ref)
        if not manifest_path.exists():
            errors.append(f"task split {split or '<empty>'} manifest does not exist: {manifest_ref}")
            continue
        actual_count = _line_count(manifest_path)
        if sample_count != actual_count:
            errors.append(
                f"task split {split or '<empty>'} sample_count={sample_count} "
                f"does not match manifest rows={actual_count}."
            )
    declared_total = _sample_universe_count(sample_universe)
    if declared_total is not None and declared_total != total_count:
        errors.append(
            f"sample_universe sample_count={declared_total} does not match "
            f"task split total={total_count}."
        )
    if not errors:
        return SuiteIntegrityResult(status=SUITE_INTEGRITY_OK, reason="", errors=[])
    status = _status_from_errors(errors)
    return SuiteIntegrityResult(status=status, reason=errors[0], errors=errors)


def _suite_manifest_from_payload(
    payload: dict[str, Any],
    *,
    manifest_path: Path,
    errors: list[str],
) -> EvalSuiteManifest | None:
    task_splits: list[SuiteTaskSplit] = []
    for index, item in enumerate(payload.get("task_splits") or []):
        if not isinstance(item, dict):
            errors.append(f"task_splits[{index}] must be an object.")
            continue
        try:
            sample_count = int(item.get("sample_count") or 0)
        except (TypeError, ValueError):
            sample_count = 0
            errors.append(f"task_splits[{index}].sample_count must be an integer.")
        task_splits.append(
            SuiteTaskSplit(
                split=str(item.get("split") or ""),
                benchmark_id=str(item.get("benchmark_id") or payload.get("benchmark_id") or ""),
                manifest_path=str(item.get("manifest_path") or ""),
                sample_count=sample_count,
                tasks=_string_list(item.get("tasks") or []),
                layers=_string_list(item.get("layers") or []),
                target_labels=_string_list(item.get("target_labels") or []),
                metadata=dict(item.get("metadata") or {}),
            )
        )
    try:
        return EvalSuiteManifest(
            suite_id=str(payload.get("suite_id") or manifest_path.parent.name),
            version=str(payload.get("version") or "unversioned"),
            benchmark_id=str(payload.get("benchmark_id") or "") or None,
            benchmark_type=str(payload.get("benchmark_type") or "official"),  # type: ignore[arg-type]
            official=_official_flag(payload, default=True),
            metric_profile=str(payload.get("metric_profile") or ""),
            sample_universe=dict(payload.get("sample_universe") or {}),
            task_splits=task_splits,
            created_at=str(payload.get("created_at") or ""),
            metadata=dict(payload.get("metadata") or {}),
        )
    except (TypeError, ValueError) as exc:
        errors.append(f"suite manifest cannot be parsed: {exc}")
        return None


def _validate_task_splits(
    suite: EvalSuiteManifest,
    *,
    suite_manifest_path: Path,
    errors: list[str],
) -> None:
    seen: set[str] = set()
    total_count = 0
    for task_split in suite.task_splits:
        split = task_split.split.strip()
        if split in seen:
            errors.append(f"duplicate task split: {split}")
        elif split:
            seen.add(split)
        total_count += int(task_split.sample_count)
        if not task_split.manifest_path.strip():
            continue
        split_manifest_path = _resolve_manifest_path(
            task_split.manifest_path,
            suite_manifest_path=suite_manifest_path,
        )
        if not split_manifest_path.exists():
            errors.append(
                f"task split {split or '<empty>'} manifest does not exist: "
                f"{task_split.manifest_path}"
            )
            continue
        actual_count = _line_count(split_manifest_path)
        if int(task_split.sample_count) != actual_count:
            errors.append(
                f"task split {split or '<empty>'} sample_count={task_split.sample_count} "
                f"does not match manifest rows={actual_count}."
            )
    declared_total = _sample_universe_count(suite.sample_universe)
    if declared_total is not None and declared_total != total_count:
        errors.append(
            f"sample_universe sample_count={declared_total} does not match "
            f"task split total={total_count}."
        )


def _validate_sample_universe(suite: EvalSuiteManifest, *, errors: list[str]) -> None:
    if not suite.sample_universe:
        errors.append("official suite must declare sample_universe.")
        return
    if _sample_universe_count(suite.sample_universe) is None:
        errors.append("official suite sample_universe must declare sample_count or sample_counts.")


def _sample_universe_count(sample_universe: dict[str, Any]) -> int | None:
    if "sample_count" in sample_universe:
        return _safe_int(sample_universe.get("sample_count"), default=-1)
    sample_counts = sample_universe.get("sample_counts")
    if isinstance(sample_counts, dict):
        return sum(max(0, _safe_int(value, default=0)) for value in sample_counts.values())
    return None


def _resolve_manifest_path(path_value: str, *, suite_manifest_path: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    candidates = [
        suite_manifest_path.parent / path,
        suite_manifest_path.parent.parent / path,
        Path.cwd() / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _line_count(path: Path) -> int:
    try:
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    except OSError:
        return 0


def _safe_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _official_flag(payload: dict[str, Any], *, default: bool) -> bool:
    value = payload.get("official", default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"false", "0", "no", "off"}:
            return False
        if normalized in {"true", "1", "yes", "on"}:
            return True
    return bool(value)


def _status_from_errors(errors: list[str]) -> str:
    joined = "\n".join(errors)
    if "manifest does not exist" in joined:
        return SUITE_INTEGRITY_MISSING_SPLIT
    if "does not match manifest rows" in joined or "does not match task split total" in joined:
        return SUITE_INTEGRITY_SAMPLE_MISMATCH
    return SUITE_INTEGRITY_INVALID_MANIFEST


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]
