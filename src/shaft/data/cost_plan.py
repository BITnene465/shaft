from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import mmap
import os
from pathlib import Path
import struct
import time
from typing import Any, Iterator
import uuid

from .cost import ShaftSampleCost, ShaftSampleCostProvider
from .mixing import ShaftSamplePlan, ShaftSampleRef


_COST_PLAN_FORMAT_VERSION = "shaft-mmap-cost-plan-v1"
_COST_PLAN_REFERENCE_VERSION = "shaft-cost-plan-reference-v1"
_COST_RECORD = struct.Struct("<QQQQdB7x")
_EXACT_FLAG = 1

COST_PLAN_REFERENCE_FILENAME = "shaft_cost_plan_reference.json"


class ShaftCostPlanCacheError(ValueError):
    """Raised when a shared CostPlan artifact is corrupt or incompatible."""


def _payload_fingerprint(payload: tuple[Any, ...]) -> str:
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


def _sample_ref_fingerprint(sample_ref: ShaftSampleRef) -> int:
    payload = (
        str(sample_ref.dataset_name),
        int(sample_ref.row_index),
        int(sample_ref.context.draw_id),
        int(sample_ref.context.plan_cycle),
        int(sample_ref.context.transform_seed),
    )
    digest = hashlib.blake2b(repr(payload).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False)


def _cache_key(
    plan: ShaftSamplePlan,
    *,
    cost_fingerprint: str,
) -> str:
    return _payload_fingerprint(
        (
            _COST_PLAN_FORMAT_VERSION,
            str(plan.fingerprint),
            str(cost_fingerprint),
            len(plan),
            0,
        )
    )


@dataclass(frozen=True, slots=True)
class ShaftCostPlanManifest:
    format_version: str
    cache_key: str
    sample_plan_fingerprint: str
    cost_fingerprint: str
    sample_count: int
    plan_cycle: int
    record_size: int
    data_filename: str
    data_bytes: int
    content_sha256: str

    def __post_init__(self) -> None:
        if self.format_version != _COST_PLAN_FORMAT_VERSION:
            raise ShaftCostPlanCacheError(
                f"Unsupported CostPlan format: {self.format_version!r}."
            )
        for field_name in (
            "cache_key",
            "sample_plan_fingerprint",
            "cost_fingerprint",
            "data_filename",
            "content_sha256",
        ):
            if not str(getattr(self, field_name)).strip():
                raise ShaftCostPlanCacheError(f"CostPlan {field_name} must not be empty.")
        if Path(self.data_filename).name != self.data_filename:
            raise ShaftCostPlanCacheError("CostPlan data_filename must be a basename.")
        if int(self.sample_count) <= 0:
            raise ShaftCostPlanCacheError("CostPlan sample_count must be > 0.")
        if int(self.plan_cycle) != 0:
            raise ShaftCostPlanCacheError("CostPlan currently supports plan_cycle=0 only.")
        if int(self.record_size) != _COST_RECORD.size:
            raise ShaftCostPlanCacheError(
                f"CostPlan record size mismatch: {self.record_size} != {_COST_RECORD.size}."
            )
        expected_bytes = int(self.sample_count) * int(self.record_size)
        if int(self.data_bytes) != expected_bytes:
            raise ShaftCostPlanCacheError(
                f"CostPlan byte size mismatch: {self.data_bytes} != {expected_bytes}."
            )

    @property
    def fingerprint(self) -> str:
        return _payload_fingerprint(self._payload())

    def _payload(self) -> tuple[Any, ...]:
        return (
            self.format_version,
            self.cache_key,
            self.sample_plan_fingerprint,
            self.cost_fingerprint,
            self.sample_count,
            self.plan_cycle,
            self.record_size,
            self.data_filename,
            self.data_bytes,
            self.content_sha256,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "cache_key": self.cache_key,
            "sample_plan_fingerprint": self.sample_plan_fingerprint,
            "cost_fingerprint": self.cost_fingerprint,
            "sample_count": self.sample_count,
            "plan_cycle": self.plan_cycle,
            "record_size": self.record_size,
            "data_filename": self.data_filename,
            "data_bytes": self.data_bytes,
            "content_sha256": self.content_sha256,
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ShaftCostPlanManifest:
        expected_fingerprint = str(payload.get("fingerprint", "")).strip()
        try:
            values = {
                "format_version": str(payload.get("format_version", "")),
                "cache_key": str(payload.get("cache_key", "")),
                "sample_plan_fingerprint": str(
                    payload.get("sample_plan_fingerprint", "")
                ),
                "cost_fingerprint": str(payload.get("cost_fingerprint", "")),
                "sample_count": int(payload.get("sample_count", 0)),
                "plan_cycle": int(payload.get("plan_cycle", -1)),
                "record_size": int(payload.get("record_size", 0)),
                "data_filename": str(payload.get("data_filename", "")),
                "data_bytes": int(payload.get("data_bytes", 0)),
                "content_sha256": str(payload.get("content_sha256", "")),
            }
        except (TypeError, ValueError, OverflowError) as exc:
            raise ShaftCostPlanCacheError(
                "CostPlan manifest contains malformed scalar fields."
            ) from exc
        manifest = cls(**values)
        if expected_fingerprint and expected_fingerprint != manifest.fingerprint:
            raise ShaftCostPlanCacheError("CostPlan manifest fingerprint is corrupt.")
        return manifest


class ShaftMMapCostPlanProvider:
    """Read-only, draw-indexed CostPlan backed by a shared memory map."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        manifest: ShaftCostPlanManifest | None = None,
        verify_checksum: bool = False,
    ) -> None:
        self._handle = None
        self._mmap = None
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.manifest = manifest or _load_manifest(self.manifest_path)
        self.data_path = self.manifest_path.parent / self.manifest.data_filename
        self.semantic_fingerprint = self.manifest.cost_fingerprint
        self.fingerprint = self.manifest.fingerprint
        self.sample_plan_fingerprint = self.manifest.sample_plan_fingerprint
        self.sample_count = int(self.manifest.sample_count)
        self._open(verify_checksum=verify_checksum)

    def _open(self, *, verify_checksum: bool) -> None:
        try:
            stat = self.data_path.stat()
        except FileNotFoundError as exc:
            raise ShaftCostPlanCacheError(
                f"Missing CostPlan data file: {self.data_path}"
            ) from exc
        if int(stat.st_size) != int(self.manifest.data_bytes):
            raise ShaftCostPlanCacheError(
                "CostPlan data size does not match manifest: "
                f"{stat.st_size} != {self.manifest.data_bytes}."
            )
        if verify_checksum:
            digest = hashlib.sha256()
            with self.data_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != self.manifest.content_sha256:
                raise ShaftCostPlanCacheError("CostPlan data checksum is corrupt.")
        self._handle = self.data_path.open("rb")
        self._mmap = mmap.mmap(self._handle.fileno(), 0, access=mmap.ACCESS_READ)

    def __call__(self, sample_ref: ShaftSampleRef) -> ShaftSampleCost:
        if int(sample_ref.context.plan_cycle) != 0:
            raise ValueError("Shared CostPlan currently supports plan_cycle=0 only.")
        draw_id = int(sample_ref.context.draw_id)
        if draw_id < 0 or draw_id >= self.sample_count:
            raise IndexError(
                f"CostPlan draw_id {draw_id} is outside [0, {self.sample_count})."
            )
        if self._mmap is None:
            raise RuntimeError("CostPlan provider is closed.")
        offset = draw_id * _COST_RECORD.size
        (
            expected_ref_fingerprint,
            llm_tokens,
            supervised_tokens,
            vision_patches,
            loss_weight_sum,
            flags,
        ) = _COST_RECORD.unpack_from(self._mmap, offset)
        actual_ref_fingerprint = _sample_ref_fingerprint(sample_ref)
        if int(expected_ref_fingerprint) != int(actual_ref_fingerprint):
            raise ValueError(
                "CostPlan sample ref does not match the materialized draw at "
                f"draw_id={draw_id}."
            )
        return ShaftSampleCost(
            llm_tokens=int(llm_tokens),
            supervised_tokens=int(supervised_tokens),
            vision_patches=int(vision_patches),
            loss_weight_sum=(
                None if math.isnan(float(loss_weight_sum)) else float(loss_weight_sum)
            ),
            exact=bool(int(flags) & _EXACT_FLAG),
        )

    def close(self) -> None:
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    @property
    def closed(self) -> bool:
        return self._mmap is None and self._handle is None

    def __enter__(self) -> ShaftMMapCostPlanProvider:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        _ = exc_type, exc_value, traceback
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best-effort process cleanup
        self.close()


@dataclass(frozen=True, slots=True)
class ShaftCostPlanMaterialization:
    provider: ShaftMMapCostPlanProvider
    manifest_path: Path
    cache_hit: bool
    elapsed_seconds: float
    data_bytes: int


def resolve_cost_plan_cache_dir(
    configured: str | Path | None,
    *,
    record_cache_dir: str | Path | None = None,
) -> Path:
    if configured is not None and str(configured).strip():
        return Path(configured).expanduser()
    if record_cache_dir is not None and str(record_cache_dir).strip():
        return Path(record_cache_dir).expanduser() / "cost_plans"
    from_environment = str(os.environ.get("SHAFT_COST_PLAN_CACHE_DIR", "")).strip()
    if from_environment:
        return Path(from_environment).expanduser()
    return Path.home() / ".cache" / "shaft" / "cost_plans"


def materialize_cost_plan(
    plan: ShaftSamplePlan,
    *,
    cost_provider: ShaftSampleCostProvider,
    cache_dir: str | Path | None = None,
    record_cache_dir: str | Path | None = None,
) -> ShaftCostPlanMaterialization:
    started = time.perf_counter()
    cost_fingerprint = str(getattr(cost_provider, "fingerprint", "")).strip()
    if not cost_fingerprint:
        raise ValueError("CostPlan materialization requires a provider fingerprint.")
    resolved_cache_dir = resolve_cost_plan_cache_dir(
        cache_dir,
        record_cache_dir=record_cache_dir,
    ).resolve()
    resolved_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = _cache_key(plan, cost_fingerprint=cost_fingerprint)
    manifest_path = resolved_cache_dir / f"{cache_key}.json"
    lock_path = resolved_cache_dir / f"{cache_key}.lock"

    with _exclusive_lock(lock_path):
        try:
            manifest = _load_manifest(manifest_path)
            _validate_manifest_for_plan(
                manifest,
                plan=plan,
                expected_cost_fingerprint=cost_fingerprint,
            )
            provider = ShaftMMapCostPlanProvider(
                manifest_path,
                manifest=manifest,
                verify_checksum=True,
            )
            return ShaftCostPlanMaterialization(
                provider=provider,
                manifest_path=manifest_path,
                cache_hit=True,
                elapsed_seconds=time.perf_counter() - started,
                data_bytes=manifest.data_bytes,
            )
        except (
            FileNotFoundError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            ShaftCostPlanCacheError,
        ):
            pass

        manifest = _build_cost_plan(
            plan,
            cost_provider=cost_provider,
            cost_fingerprint=cost_fingerprint,
            cache_key=cache_key,
            manifest_path=manifest_path,
        )
        provider = ShaftMMapCostPlanProvider(
            manifest_path,
            manifest=manifest,
            verify_checksum=False,
        )
        return ShaftCostPlanMaterialization(
            provider=provider,
            manifest_path=manifest_path,
            cache_hit=False,
            elapsed_seconds=time.perf_counter() - started,
            data_bytes=manifest.data_bytes,
        )


def cost_plan_reference_path(path: str | Path) -> Path:
    return Path(path) / COST_PLAN_REFERENCE_FILENAME


def write_cost_plan_reference(
    path: str | Path,
    materialization: ShaftCostPlanMaterialization,
) -> Path:
    manifest = materialization.provider.manifest
    payload = {
        "format_version": _COST_PLAN_REFERENCE_VERSION,
        "manifest_path": str(materialization.manifest_path.resolve()),
        "manifest_fingerprint": manifest.fingerprint,
        "sample_plan_fingerprint": manifest.sample_plan_fingerprint,
        "cost_fingerprint": manifest.cost_fingerprint,
        "sample_count": manifest.sample_count,
    }
    target = cost_plan_reference_path(path)
    _write_json_atomic(target, payload)
    return target


def load_cost_plan_reference(
    path: str | Path,
    *,
    plan: ShaftSamplePlan,
    verify_checksum: bool = False,
) -> ShaftMMapCostPlanProvider:
    reference_path = cost_plan_reference_path(path)
    try:
        payload = json.loads(reference_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing CostPlan reference: {reference_path}") from exc
    if not isinstance(payload, dict):
        raise ShaftCostPlanCacheError("CostPlan reference must be a JSON object.")
    if payload.get("format_version") != _COST_PLAN_REFERENCE_VERSION:
        raise ShaftCostPlanCacheError("Unsupported CostPlan reference format.")
    manifest_path = Path(str(payload.get("manifest_path", ""))).expanduser()
    manifest = _load_manifest(manifest_path)
    if str(payload.get("manifest_fingerprint", "")) != manifest.fingerprint:
        raise ShaftCostPlanCacheError("CostPlan reference manifest fingerprint changed.")
    for field_name in (
        "sample_plan_fingerprint",
        "cost_fingerprint",
        "sample_count",
    ):
        if payload.get(field_name) != manifest.to_dict()[field_name]:
            raise ShaftCostPlanCacheError(
                f"CostPlan reference {field_name} does not match its manifest."
            )
    _validate_manifest_for_plan(manifest, plan=plan)
    return ShaftMMapCostPlanProvider(
        manifest_path,
        manifest=manifest,
        verify_checksum=verify_checksum,
    )


def load_cost_plan_manifest(
    manifest_path: str | Path,
    *,
    plan: ShaftSamplePlan,
    expected_manifest_fingerprint: str | None = None,
    verify_checksum: bool = False,
) -> ShaftMMapCostPlanProvider:
    manifest = _load_manifest(manifest_path)
    if (
        expected_manifest_fingerprint is not None
        and manifest.fingerprint != str(expected_manifest_fingerprint)
    ):
        raise ShaftCostPlanCacheError(
            "CostPlan rendezvous manifest fingerprint changed."
        )
    _validate_manifest_for_plan(manifest, plan=plan)
    return ShaftMMapCostPlanProvider(
        manifest_path,
        manifest=manifest,
        verify_checksum=verify_checksum,
    )


def _build_cost_plan(
    plan: ShaftSamplePlan,
    *,
    cost_provider: ShaftSampleCostProvider,
    cost_fingerprint: str,
    cache_key: str,
    manifest_path: Path,
) -> ShaftCostPlanManifest:
    temp_data_path = manifest_path.parent / (
        f".{cache_key}.{uuid.uuid4().hex}.bin.tmp"
    )
    digest = hashlib.sha256()
    try:
        with temp_data_path.open("xb") as handle:
            for position in range(len(plan)):
                sample_ref = plan.ref_at(position, plan_cycle=0)
                cost = cost_provider(sample_ref)
                packed = _pack_cost(sample_ref, cost)
                handle.write(packed)
                digest.update(packed)
            handle.flush()
            os.fsync(handle.fileno())
        content_sha256 = digest.hexdigest()
        data_filename = f"{cache_key}.{content_sha256}.bin"
        data_path = manifest_path.parent / data_filename
        os.replace(temp_data_path, data_path)
        manifest = ShaftCostPlanManifest(
            format_version=_COST_PLAN_FORMAT_VERSION,
            cache_key=cache_key,
            sample_plan_fingerprint=str(plan.fingerprint),
            cost_fingerprint=cost_fingerprint,
            sample_count=len(plan),
            plan_cycle=0,
            record_size=_COST_RECORD.size,
            data_filename=data_filename,
            data_bytes=len(plan) * _COST_RECORD.size,
            content_sha256=content_sha256,
        )
        _write_json_atomic(manifest_path, manifest.to_dict())
        return manifest
    finally:
        temp_data_path.unlink(missing_ok=True)


def _pack_cost(sample_ref: ShaftSampleRef, cost: ShaftSampleCost) -> bytes:
    unsigned_values = (
        _sample_ref_fingerprint(sample_ref),
        int(cost.llm_tokens),
        int(cost.supervised_tokens),
        int(cost.vision_patches),
    )
    if any(value < 0 or value > (1 << 64) - 1 for value in unsigned_values):
        raise OverflowError("CostPlan integer field is outside uint64 range.")
    loss_weight_sum = (
        math.nan if cost.loss_weight_sum is None else float(cost.loss_weight_sum)
    )
    flags = _EXACT_FLAG if cost.exact else 0
    return _COST_RECORD.pack(*unsigned_values, loss_weight_sum, flags)


def _load_manifest(path: str | Path) -> ShaftCostPlanManifest:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ShaftCostPlanCacheError("CostPlan manifest must be a JSON object.")
    return ShaftCostPlanManifest.from_dict(payload)


def _validate_manifest_for_plan(
    manifest: ShaftCostPlanManifest,
    *,
    plan: ShaftSamplePlan,
    expected_cost_fingerprint: str | None = None,
) -> None:
    if manifest.sample_plan_fingerprint != str(plan.fingerprint):
        raise ShaftCostPlanCacheError(
            "CostPlan SamplePlan fingerprint does not match the runtime plan."
        )
    if manifest.sample_count != len(plan):
        raise ShaftCostPlanCacheError(
            f"CostPlan sample count {manifest.sample_count} != runtime {len(plan)}."
        )
    if (
        expected_cost_fingerprint is not None
        and manifest.cost_fingerprint != str(expected_cost_fingerprint)
    ):
        raise ShaftCostPlanCacheError("CostPlan cost fingerprint is stale.")
    expected_cache_key = _cache_key(
        plan,
        cost_fingerprint=manifest.cost_fingerprint,
    )
    if manifest.cache_key != expected_cache_key:
        raise ShaftCostPlanCacheError("CostPlan cache key is corrupt or stale.")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except ImportError:  # pragma: no cover - Shaft training targets Linux
            pass
        yield
