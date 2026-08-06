from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import time
from typing import Any

import torch.distributed as dist

from shaft.utils.distributed import get_rank, get_world_size


logger = logging.getLogger(__name__)

_MODEL_ARTIFACT_IDENTITY_VERSION = "shaft-model-artifact-identity-v1"
_IMMUTABLE_HUB_REVISION = re.compile(r"^[0-9a-fA-F]{40}$")
_DYNAMIC_MODULE_PREFIX = "transformers_modules."


class _DuplicateJSONKeyError(ValueError):
    pass


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise _DuplicateJSONKeyError(f"Duplicate JSON object key: {key!r}.")
        payload[key] = value
    return payload


@dataclass(frozen=True, slots=True)
class ResolvedModelArtifactIdentity:
    """Immutable identity of the bytes used to construct the base model."""

    kind: str
    fingerprint: str
    complete: bool
    incomplete_reasons: tuple[str, ...] = ()
    file_manifest: tuple[tuple[str, int, str], ...] = ()
    file_stat_manifest: tuple[tuple[str, int, int, int, int, int], ...] = ()
    resolved_revision: str | None = None

    def __post_init__(self) -> None:
        if not str(self.kind).strip() or not str(self.fingerprint).strip():
            raise ValueError("Model artifact identity kind/fingerprint must not be empty.")
        if bool(self.complete) == bool(self.incomplete_reasons):
            raise ValueError(
                "A complete model artifact identity cannot have incomplete reasons, "
                "and an incomplete identity must explain why."
            )
        if self.kind == "local_hf" and self.complete:
            content_shape = tuple((name, size) for name, size, _digest in self.file_manifest)
            stat_shape = tuple((entry[0], entry[1]) for entry in self.file_stat_manifest)
            if not content_shape or content_shape != stat_shape:
                raise ValueError(
                    "A complete local HF identity requires matching non-empty content "
                    "and stat manifests."
                )


@dataclass(frozen=True, slots=True)
class LocalModelArtifactLoadGuard:
    """Cheap local-file snapshot spanning one HF loader invocation.

    The guard is deliberately process-local and short lived.  It is not an
    identity cache: the content identity is still verified with a complete
    SHA-256 pass immediately after the loader returns.
    """

    artifact_fingerprint: str
    file_stat_manifest: tuple[tuple[str, int, int, int, int, int], ...]


def _fingerprint(payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _incomplete_identity(*, kind: str, source: str, reason: str) -> ResolvedModelArtifactIdentity:
    reasons = (str(reason),)
    return ResolvedModelArtifactIdentity(
        kind=kind,
        fingerprint=_fingerprint(
            {
                "version": _MODEL_ARTIFACT_IDENTITY_VERSION,
                "kind": kind,
                "source": str(source),
                "incomplete_reasons": reasons,
            }
        ),
        complete=False,
        incomplete_reasons=reasons,
    )


def _require_local_artifact_files(
    root: Path,
    files: tuple[Path, ...] | list[Path],
    *,
    label: str,
) -> tuple[Path, ...]:
    """Require lexical local-artifact entries to resolve inside their root."""

    resolved_root = root.resolve(strict=True)
    escaping: list[str] = []
    for path in files:
        try:
            path.resolve(strict=True).relative_to(resolved_root)
        except ValueError:
            try:
                escaping.append(path.relative_to(root).as_posix())
            except ValueError:
                escaping.append(str(path))
    if escaping:
        raise ValueError(
            f"Local HF {label} files must stay inside the model directory, including "
            f"through symlinks: {escaping[:8]!r}."
        )
    return tuple(files)


def _resolve_weight_files(root: Path) -> tuple[Path, ...]:
    index_path = next(
        (
            candidate
            for candidate in (
                root / "model.safetensors.index.json",
                root / "pytorch_model.bin.index.json",
            )
            if candidate.is_file()
        ),
        None,
    )
    if index_path is not None:
        try:
            payload = json.loads(
                index_path.read_text(encoding="utf-8"),
                object_pairs_hook=_strict_json_object,
            )
        except (json.JSONDecodeError, _DuplicateJSONKeyError) as exc:
            raise ValueError(f"Invalid HF weight index JSON: {index_path}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"HF weight index must be a JSON object: {index_path}")
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError(f"HF weight index has no object weight_map: {index_path}")
        if not weight_map:
            raise ValueError(f"HF weight index has an empty weight_map: {index_path}")
        names: set[str] = set()
        for tensor_name, shard_name in weight_map.items():
            if type(tensor_name) is not str or not tensor_name.strip():
                raise ValueError(
                    "HF weight index contains an empty/non-string tensor key: "
                    f"{index_path}."
                )
            if tensor_name != tensor_name.strip():
                raise ValueError(
                    "HF weight index tensor keys must not contain surrounding "
                    f"whitespace: {index_path}."
                )
            if type(shard_name) is not str or not shard_name.strip():
                raise ValueError(
                    "HF weight index contains an empty/non-string shard path for "
                    f"tensor {tensor_name!r}: {index_path}."
                )
            if shard_name != shard_name.strip():
                raise ValueError(
                    "HF weight index shard paths must not contain surrounding "
                    f"whitespace: {index_path}."
                )
            shard_path = Path(shard_name)
            if (
                shard_path.is_absolute()
                or ".." in shard_path.parts
                or "\\" in shard_name
                or shard_path.as_posix() != shard_name
                or (shard_path.parts and shard_path.parts[0].endswith(":"))
            ):
                raise ValueError(
                    "HF weight index shard paths must be canonical relative paths "
                    f"inside the model directory: tensor={tensor_name!r}, "
                    f"path={shard_name!r}."
                )
            names.add(shard_name)
        files = [index_path, *(root / name for name in sorted(names))]
        missing = [path for path in files if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "HF weight identity cannot be resolved because indexed shards are "
                f"missing: {[path.name for path in missing[:8]]}."
            )
        _require_local_artifact_files(root, files[1:], label="indexed weight shard")
        empty = [path for path in files[1:] if path.stat().st_size <= 0]
        if empty:
            raise ValueError(
                "HF weight identity cannot include zero-byte indexed shards: "
                f"{[path.name for path in empty[:8]]}."
            )
        return tuple(files)

    safetensors_files = sorted(root.glob("model*.safetensors"))
    safetensors_files = [
        path
        for path in safetensors_files
        if path.is_file() and not path.name.startswith("adapter_model")
    ]
    if safetensors_files:
        _require_local_artifact_files(root, safetensors_files, label="weight")
        empty = [path for path in safetensors_files if path.stat().st_size <= 0]
        if empty:
            raise ValueError(
                "HF weight identity cannot include zero-byte safetensors files: "
                f"{[path.name for path in empty[:8]]}."
            )
        return tuple(safetensors_files)
    binary_files = sorted(root.glob("pytorch_model*.bin"))
    binary_files = [path for path in binary_files if path.is_file()]
    _require_local_artifact_files(root, binary_files, label="weight")
    empty = [path for path in binary_files if path.stat().st_size <= 0]
    if empty:
        raise ValueError(
            "HF weight identity cannot include zero-byte PyTorch weight files: "
            f"{[path.name for path in empty[:8]]}."
        )
    return tuple(binary_files)


def _stat_signature(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "ctime_ns": int(stat.st_ctime_ns),
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
    }


def _manifest_stat_entry(relative: str, stat: dict[str, int]) -> tuple[str, int, int, int, int, int]:
    return (
        relative,
        int(stat["size"]),
        int(stat["mtime_ns"]),
        int(stat["ctime_ns"]),
        int(stat["device"]),
        int(stat["inode"]),
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _full_file_snapshot(path: Path) -> tuple[dict[str, int], str]:
    """Hash every byte and reject an observable concurrent file replacement.

    A persistent stat/probe cache is deliberately not used here. Several training
    filesystems can preserve inode timestamps for immediate same-size rewrites, and
    a sampled probe can therefore reuse a digest for bytes that no longer exist.
    Exact checkpoint identity uses complete reads for the baseline and post-load
    closure phases; the intervening load guard is intentionally metadata-only.
    """

    before = _stat_signature(path)
    digest = _file_sha256(path)
    after = _stat_signature(path)
    if after != before:
        raise RuntimeError(
            "Model artifact changed while its complete digest was being resolved: "
            f"{path}."
        )
    return before, digest


def _local_file_manifest(
    root: Path,
    files: tuple[Path, ...],
) -> tuple[
    tuple[tuple[str, int, str], ...],
    tuple[tuple[str, int, int, int, int, int], ...],
    int,
    float,
]:
    started_at = time.perf_counter()
    total_bytes = 0
    manifest: list[tuple[str, int, str]] = []
    relative_files = tuple(path.relative_to(root).as_posix() for path in files)
    baseline_stats = tuple(
        _manifest_stat_entry(relative, _stat_signature(path))
        for relative, path in zip(relative_files, files, strict=True)
    )
    for relative, path, baseline_stat in zip(
        relative_files,
        files,
        baseline_stats,
        strict=True,
    ):
        stat, digest = _full_file_snapshot(path)
        if _manifest_stat_entry(relative, stat) != baseline_stat:
            raise RuntimeError(
                "Model artifact changed after the whole-file baseline was captured: "
                f"{path}."
            )
        size = int(stat["size"])
        total_bytes += size
        manifest.append((relative, size, digest))
    final_stats = tuple(
        _manifest_stat_entry(relative, _stat_signature(path))
        for relative, path in zip(relative_files, files, strict=True)
    )
    if final_stats != baseline_stats:
        raise RuntimeError(
            "Model artifact changed while the complete multi-file manifest was "
            "being resolved."
        )
    return (
        tuple(manifest),
        baseline_stats,
        total_bytes,
        time.perf_counter() - started_at,
    )


def _complete_local_identity(
    root: Path,
    files: tuple[Path, ...],
    *,
    trust_remote_code: bool,
) -> ResolvedModelArtifactIdentity:
    manifest, stat_manifest, total_bytes, elapsed = _local_file_manifest(root, files)
    payload = {
        "version": _MODEL_ARTIFACT_IDENTITY_VERSION,
        "kind": "local_hf",
        "file_manifest": manifest,
        "trust_remote_code": bool(trust_remote_code),
    }
    logger.info(
        "[model-artifact] kind=local_hf files=%s artifact_bytes=%s "
        "full_hash_read_bytes=%s cache=disabled elapsed_seconds=%.3f",
        len(manifest),
        total_bytes,
        total_bytes,
        elapsed,
    )
    return ResolvedModelArtifactIdentity(
        kind="local_hf",
        fingerprint=_fingerprint(payload),
        complete=True,
        file_manifest=manifest,
        file_stat_manifest=stat_manifest,
    )


def _prepare_complete_local_identity_files(
    root: Path,
    *,
    trust_remote_code: bool,
) -> tuple[Path, ...] | ResolvedModelArtifactIdentity:
    source = str(root)
    if not root.is_dir():
        return _incomplete_identity(
            kind="local_hf",
            source=source,
            reason="missing_local_hf_directory",
        )
    config_path = root / "config.json"
    if not config_path.is_file():
        return _incomplete_identity(
            kind="local_hf",
            source=source,
            reason="missing_local_hf_config",
        )
    weight_files = _resolve_weight_files(root)
    if not weight_files:
        return _incomplete_identity(
            kind="local_hf",
            source=source,
            reason="missing_local_hf_weight_files",
        )
    return _local_identity_files(
        root,
        trust_remote_code=trust_remote_code,
        weight_files=weight_files,
    )


def _artifact_identity_process_group():
    return dist.new_group(
        backend="gloo",
        timeout=timedelta(hours=2),
    )


def _artifact_identity_all_gather(value: Any, *, group: Any) -> list[Any]:
    gathered: list[Any] = [None] * int(dist.get_world_size(group=group))
    dist.all_gather_object(gathered, value, group=group)
    return gathered


def _distributed_complete_local_identity(
    root: Path,
    *,
    trust_remote_code: bool,
) -> ResolvedModelArtifactIdentity:
    """Hash once per shared node artifact and retain a rank-local stat guard.

    Torchrun ranks on one host share the same model files, so reading every byte on
    every rank multiplies startup I/O without adding evidence. Each node leader
    hashes its local artifact, all node fingerprints must agree globally, and ranks
    with a distinct mount/file identity fall back to their own complete hash.
    """

    if get_world_size() <= 1:
        prepared = _prepare_complete_local_identity_files(
            root,
            trust_remote_code=trust_remote_code,
        )
        if isinstance(prepared, ResolvedModelArtifactIdentity):
            return prepared
        return _complete_local_identity(root, prepared, trust_remote_code=trust_remote_code)
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "Distributed model artifact identity requires an initialized process "
            "group before hashing local checkpoint bytes."
        )

    group = _artifact_identity_process_group()
    try:
        prepared: tuple[Path, ...] | ResolvedModelArtifactIdentity | None = None
        local_status: dict[str, Any]
        try:
            local_rank = int(os.environ["LOCAL_RANK"])
            local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
            group_rank = int(os.environ["GROUP_RANK"])
            if local_world_size <= 0 or not 0 <= local_rank < local_world_size:
                raise ValueError(
                    "Invalid torchrun local topology: "
                    f"LOCAL_RANK={local_rank}, LOCAL_WORLD_SIZE={local_world_size}."
                )
            prepared = _prepare_complete_local_identity_files(
                root,
                trust_remote_code=trust_remote_code,
            )
            if isinstance(prepared, ResolvedModelArtifactIdentity):
                identity = prepared
                stat_manifest: tuple[tuple[str, int, int, int, int, int], ...] = ()
            else:
                relative_files = tuple(path.relative_to(root).as_posix() for path in prepared)
                stat_manifest = tuple(
                    _manifest_stat_entry(relative, _stat_signature(path))
                    for relative, path in zip(relative_files, prepared, strict=True)
                )
                identity = (
                    _complete_local_identity(
                        root,
                        prepared,
                        trust_remote_code=trust_remote_code,
                    )
                    if local_rank == 0
                    else None
                )
                if identity is not None:
                    stat_manifest = identity.file_stat_manifest
            local_status = {
                "ok": True,
                "rank": get_rank(),
                "group_rank": group_rank,
                "local_rank": local_rank,
                "local_world_size": local_world_size,
                "identity": identity,
                "stat_manifest": stat_manifest,
            }
        except Exception as exc:  # noqa: BLE001 - peers must converge on failure
            local_status = {
                "ok": False,
                "rank": get_rank(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

        gathered = _artifact_identity_all_gather(local_status, group=group)
        failures = [status for status in gathered if not bool(status.get("ok"))]
        if failures:
            raise RuntimeError(
                "Distributed immutable model artifact preflight failed: "
                f"{failures}."
            )
        by_node: dict[int, list[dict[str, Any]]] = {}
        for status in gathered:
            by_node.setdefault(int(status["group_rank"]), []).append(status)
        for node_rank, statuses in by_node.items():
            local_world_sizes = {int(status["local_world_size"]) for status in statuses}
            local_ranks = {int(status["local_rank"]) for status in statuses}
            if (
                len(local_world_sizes) != 1
                or len(statuses) != next(iter(local_world_sizes))
                or local_ranks != set(range(len(statuses)))
            ):
                raise RuntimeError(
                    "Distributed model artifact identity requires complete torchrun "
                    f"node topology; node={node_rank}, statuses={statuses}."
                )

        incomplete = [
            status["identity"]
            for status in gathered
            if isinstance(status.get("identity"), ResolvedModelArtifactIdentity)
            and not status["identity"].complete
        ]
        if incomplete:
            if len(incomplete) != len(gathered) or len(
                {identity.fingerprint for identity in incomplete}
            ) != 1:
                raise ValueError(
                    "Local model artifact availability differs across distributed ranks."
                )
            return incomplete[get_rank()]

        leaders = {
            int(status["group_rank"]): status
            for status in gathered
            if int(status["local_rank"]) == 0
        }
        identities = [status["identity"] for status in leaders.values()]
        if len(identities) != len(by_node) or any(
            not isinstance(identity, ResolvedModelArtifactIdentity)
            for identity in identities
        ):
            raise RuntimeError(
                "Distributed model artifact identity did not receive exactly one "
                "content manifest per node."
            )
        canonical = identities[0]
        if any(
            identity.fingerprint != canonical.fingerprint
            or identity.file_manifest != canonical.file_manifest
            for identity in identities[1:]
        ):
            raise ValueError(
                "Local model artifact bytes differ across distributed nodes."
            )

        fallback_ranks = {
            int(status["rank"])
            for status in gathered
            if status["stat_manifest"]
            != leaders[int(status["group_rank"])]["stat_manifest"]
        }
        if fallback_ranks:
            fallback_status: dict[str, Any]
            try:
                fallback_identity = None
                if get_rank() in fallback_ranks:
                    if not isinstance(prepared, tuple):
                        raise RuntimeError(
                            "Fallback artifact hashing requires a complete local file inventory."
                        )
                    fallback_identity = _complete_local_identity(
                        root,
                        prepared,
                        trust_remote_code=trust_remote_code,
                    )
                fallback_status = {
                    "ok": True,
                    "rank": get_rank(),
                    "identity": fallback_identity,
                }
            except Exception as exc:  # noqa: BLE001 - peers must converge on failure
                fallback_status = {
                    "ok": False,
                    "rank": get_rank(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            fallback_results = _artifact_identity_all_gather(
                fallback_status,
                group=group,
            )
            fallback_failures = [
                status for status in fallback_results if not bool(status.get("ok"))
            ]
            if fallback_failures:
                raise RuntimeError(
                    "Distributed fallback model artifact hashing failed: "
                    f"{fallback_failures}."
                )
            for status in fallback_results:
                identity = status.get("identity")
                if identity is None:
                    continue
                if (
                    identity.fingerprint != canonical.fingerprint
                    or identity.file_manifest != canonical.file_manifest
                ):
                    raise ValueError(
                        "A rank-local model artifact differs from its node-leader manifest."
                    )
                gathered[int(status["rank"])]["stat_manifest"] = (
                    identity.file_stat_manifest
                )

        local_stat_manifest = gathered[get_rank()]["stat_manifest"]
        final_status: dict[str, Any]
        try:
            if not isinstance(prepared, tuple):
                raise RuntimeError(
                    "Complete distributed artifact identity lost its local file inventory."
                )
            relative_files = tuple(path.relative_to(root).as_posix() for path in prepared)
            final_stats = tuple(
                _manifest_stat_entry(relative, _stat_signature(path))
                for relative, path in zip(relative_files, prepared, strict=True)
            )
            if final_stats != local_stat_manifest:
                raise RuntimeError(
                    "Local model artifact changed while distributed content identity "
                    "was converging."
                )
            final_status = {"ok": True, "rank": get_rank()}
        except Exception as exc:  # noqa: BLE001 - peers must converge on failure
            final_status = {
                "ok": False,
                "rank": get_rank(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        final_results = _artifact_identity_all_gather(final_status, group=group)
        final_failures = [
            status for status in final_results if not bool(status.get("ok"))
        ]
        if final_failures:
            raise RuntimeError(
                "Distributed model artifact identity closure failed: "
                f"{final_failures}."
            )
        return ResolvedModelArtifactIdentity(
            kind=canonical.kind,
            fingerprint=canonical.fingerprint,
            complete=canonical.complete,
            incomplete_reasons=canonical.incomplete_reasons,
            file_manifest=canonical.file_manifest,
            file_stat_manifest=local_stat_manifest,
            resolved_revision=canonical.resolved_revision,
        )
    finally:
        dist.destroy_process_group(group)


def _local_identity_files(
    root: Path,
    *,
    trust_remote_code: bool,
    weight_files: tuple[Path, ...] | None = None,
) -> tuple[Path, ...]:
    resolved_weight_files = (
        _resolve_weight_files(root) if weight_files is None else weight_files
    )
    config_path = root / "config.json"
    config_files = (config_path,) if config_path.is_file() else ()
    if bool(trust_remote_code):
        symlink_directories = tuple(
            sorted(
                path
                for path in root.rglob("*")
                if path.is_symlink() and path.is_dir()
            )
        )
        if symlink_directories:
            raise ValueError(
                "Local HF remote-code directory symlinks are not supported by the "
                "immutable artifact identity: "
                f"{[path.relative_to(root).as_posix() for path in symlink_directories[:8]]!r}."
            )
    code_files = (
        tuple(sorted(path for path in root.rglob("*.py") if path.is_file()))
        if bool(trust_remote_code)
        else ()
    )
    files = tuple(
        dict.fromkeys((*resolved_weight_files, *config_files, *code_files))
    )
    return _require_local_artifact_files(root, files, label="identity")


def capture_local_model_artifact_load_guard(
    identity: ResolvedModelArtifactIdentity,
    *,
    root: Path,
    trust_remote_code: bool,
) -> LocalModelArtifactLoadGuard:
    """Capture a metadata-only guard without re-reading model bytes.

    The exact file inventory and the full stat tuple must still match the
    baseline resolution.  A same-stat content rewrite can pass this cheap
    phase, but it cannot pass the mandatory post-load complete digest.
    """

    if identity.kind != "local_hf" or not identity.complete:
        raise ValueError("A local artifact load guard requires a complete local HF identity.")
    files = _local_identity_files(root, trust_remote_code=trust_remote_code)
    relative_files = tuple(path.relative_to(root).as_posix() for path in files)
    expected_files = tuple(entry[0] for entry in identity.file_manifest)
    if relative_files != expected_files:
        raise RuntimeError(
            "Local HF artifact file inventory changed after identity resolution: "
            f"expected={expected_files!r}, current={relative_files!r}."
        )
    current_stats = tuple(
        _manifest_stat_entry(relative, _stat_signature(path))
        for relative, path in zip(relative_files, files, strict=True)
    )
    if current_stats != identity.file_stat_manifest:
        raise RuntimeError(
            "Local HF artifact metadata changed after identity resolution and before "
            "model loading. Re-resolve the model plan from the stable artifact."
        )
    return LocalModelArtifactLoadGuard(
        artifact_fingerprint=identity.fingerprint,
        file_stat_manifest=current_stats,
    )


def validate_local_model_artifact_load_guard(
    identity: ResolvedModelArtifactIdentity,
    guard: LocalModelArtifactLoadGuard,
    *,
    root: Path,
    trust_remote_code: bool,
) -> None:
    """Reject observable file replacement across an HF loader invocation."""

    if guard.artifact_fingerprint != identity.fingerprint:
        raise ValueError("Local artifact load guard does not belong to the resolved identity.")
    files = _local_identity_files(root, trust_remote_code=trust_remote_code)
    relative_files = tuple(path.relative_to(root).as_posix() for path in files)
    current_stats = tuple(
        _manifest_stat_entry(relative, _stat_signature(path))
        for relative, path in zip(relative_files, files, strict=True)
    )
    if current_stats != guard.file_stat_manifest:
        raise RuntimeError(
            "Local HF artifact changed while the model/tokenizer/processor loader was "
            "running. The loaded objects are discarded."
        )


def external_auto_map_repositories(
    config_payload: dict[str, Any],
    *,
    model_repo_id: str,
) -> tuple[str, ...]:
    """Return code repositories referenced outside the resolved model repo."""

    auto_map = config_payload.get("auto_map")
    if not isinstance(auto_map, dict):
        return ()
    repositories: set[str] = set()
    values: list[Any] = list(auto_map.values())
    while values:
        value = values.pop()
        if isinstance(value, (list, tuple)):
            values.extend(value)
            continue
        if not isinstance(value, str) or "--" not in value:
            continue
        repository, _separator, _reference = value.partition("--")
        repository = repository.strip().strip("/")
        if repository and repository != str(model_repo_id).strip().strip("/"):
            repositories.add(repository)
    return tuple(sorted(repositories))


def resolve_model_artifact_identity(
    model_name_or_path: str,
    *,
    model_type: str,
    uses_hf_artifacts: bool,
    trust_remote_code: bool,
    requested_revision: str | None,
    resolved_commit_hash: str | None,
    is_hub_repo: bool,
    require_immutable_local: bool = False,
    external_remote_code_repositories: tuple[str, ...] = (),
    distributed_consensus: bool = False,
) -> ResolvedModelArtifactIdentity:
    source = str(model_name_or_path).strip()
    if not uses_hf_artifacts:
        payload = {
            "version": _MODEL_ARTIFACT_IDENTITY_VERSION,
            "kind": "built_in",
            "model_type": str(model_type),
        }
        return ResolvedModelArtifactIdentity(
            kind="built_in",
            fingerprint=_fingerprint(payload),
            complete=True,
        )

    root = Path(source)
    external_repositories = tuple(
        sorted(
            {
                str(repository).strip()
                for repository in external_remote_code_repositories
                if str(repository).strip()
            }
        )
    )
    if (
        bool(distributed_consensus)
        and bool(require_immutable_local)
        and not bool(is_hub_repo)
    ):
        if bool(trust_remote_code) and external_repositories:
            return _incomplete_identity(
                kind="local_hf",
                source=source,
                reason=(
                    "unresolved_external_remote_code_revision:"
                    + ",".join(external_repositories)
                ),
            )
        return _distributed_complete_local_identity(
            root,
            trust_remote_code=trust_remote_code,
        )

    if root.is_dir():
        if not require_immutable_local:
            return _incomplete_identity(
                kind="local_hf",
                source=source,
                reason="local_hf_identity_not_materialized",
            )
        config_path = root / "config.json"
        if not config_path.is_file():
            return _incomplete_identity(
                kind="local_hf",
                source=source,
                reason="missing_local_hf_config",
            )
        if bool(trust_remote_code) and external_repositories:
            return _incomplete_identity(
                kind="local_hf",
                source=source,
                reason=(
                    "unresolved_external_remote_code_revision:"
                    + ",".join(external_repositories)
                ),
            )
        prepared = _prepare_complete_local_identity_files(
            root,
            trust_remote_code=trust_remote_code,
        )
        if isinstance(prepared, ResolvedModelArtifactIdentity):
            return prepared
        return _complete_local_identity(root, prepared, trust_remote_code=trust_remote_code)

    if is_hub_repo:
        resolved_candidate = str(resolved_commit_hash or "").strip()
        requested = str(requested_revision or "").strip()
        resolved_revision = ""
        if resolved_candidate:
            if _IMMUTABLE_HUB_REVISION.fullmatch(resolved_candidate):
                resolved_revision = resolved_candidate.lower()
            elif _IMMUTABLE_HUB_REVISION.fullmatch(requested):
                resolved_revision = requested.lower()
            else:
                return _incomplete_identity(
                    kind="hf_hub",
                    source=source,
                    reason="invalid_resolved_hub_revision",
                )
        elif _IMMUTABLE_HUB_REVISION.fullmatch(requested):
            resolved_revision = requested.lower()
        if not resolved_revision:
            return _incomplete_identity(
                kind="hf_hub",
                source=source,
                reason="unresolved_immutable_hub_revision",
            )
        if (
            _IMMUTABLE_HUB_REVISION.fullmatch(requested)
            and requested.lower() != resolved_revision
        ):
            return _incomplete_identity(
                kind="hf_hub",
                source=source,
                reason="resolved_hub_revision_mismatch",
            )
        if bool(trust_remote_code) and external_repositories:
            return _incomplete_identity(
                kind="hf_hub",
                source=source,
                reason=(
                    "unresolved_external_remote_code_revision:"
                    + ",".join(external_repositories)
                ),
            )
        payload = {
            "version": _MODEL_ARTIFACT_IDENTITY_VERSION,
            "kind": "hf_hub",
            "repo_id": source,
            "resolved_revision": resolved_revision,
        }
        return ResolvedModelArtifactIdentity(
            kind="hf_hub",
            fingerprint=_fingerprint(payload),
            complete=True,
            resolved_revision=resolved_revision,
        )

    return _incomplete_identity(
        kind="unresolved_hf",
        source=source,
        reason="unresolved_hf_model_artifact",
    )


def _runtime_dynamic_module_names(*roots: Any) -> tuple[str, ...]:
    pending = [(root, 0) for root in roots if root is not None]
    visited: set[int] = set()
    module_names: set[str] = set()
    child_attributes = (
        "model",
        "base_model",
        "module",
        "tokenizer",
        "image_processor",
        "video_processor",
    )
    while pending:
        value, depth = pending.pop()
        identity = id(value)
        if identity in visited:
            continue
        visited.add(identity)
        module_name = str(type(value).__module__)
        if module_name.startswith(_DYNAMIC_MODULE_PREFIX):
            module_names.add(module_name)
        if depth >= 5:
            continue
        for attribute in child_attributes:
            try:
                child = getattr(value, attribute, None)
            except Exception:  # noqa: BLE001 - runtime wrappers may expose guarded properties
                continue
            if child is not None and child is not value:
                pending.append((child, depth + 1))
    return tuple(sorted(module_names))


def validate_loaded_remote_code_identity(
    *,
    model: Any,
    tokenizer: Any,
    processor: Any,
    expected_model_revision: str | None,
    strict: bool,
) -> None:
    """Reject late-discovered dynamic code outside the pinned model commit.

    Model ``config.json`` does not describe every tokenizer/processor auto-map.
    In strict checkpoint mode, runtime dynamic modules must therefore expose the
    exact model commit in their Transformers module namespace. A different commit
    denotes an independently versioned code repository that the model artifact
    contract cannot currently persist and restore.
    """

    if not bool(strict):
        return
    expected = str(expected_model_revision or "").strip().lower()
    local_artifact = not expected
    if not local_artifact and not _IMMUTABLE_HUB_REVISION.fullmatch(expected):
        raise ValueError(
            "Strict remote-code validation requires an immutable 40-hex model revision."
        )
    for module_name in _runtime_dynamic_module_names(model, tokenizer, processor):
        commits = {
            part.lower()
            for part in module_name.split(".")
            if _IMMUTABLE_HUB_REVISION.fullmatch(part)
        }
        if local_artifact:
            if commits:
                raise ValueError(
                    "Loaded remote-code module for a local model comes from an external "
                    "Hub revision that is not part of the local artifact manifest: "
                    f"module={module_name!r}, code_commits={sorted(commits)!r}. Exact "
                    "checkpoint save/resume is unsafe."
                )
            continue
        if not commits:
            raise ValueError(
                "Loaded remote-code module has no immutable commit identity: "
                f"{module_name!r}. Exact checkpoint save/resume is unsafe."
            )
        if commits != {expected}:
            raise ValueError(
                "Loaded remote-code module comes from a code revision outside the "
                "pinned model commit: "
                f"module={module_name!r}, code_commits={sorted(commits)!r}, "
                f"model_commit={expected!r}. Exact checkpoint save/resume is unsafe."
            )
