from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any
import uuid

import transformers.trainer as hf_trainer_module

from shaft.config import RuntimeConfig
from shaft.model import ModelMeta, ShaftModelAdapter
from shaft.model.finetune_plan import FINETUNE_SUMMARY_FILENAME
from shaft.observability import (
    PROGRESS_SNAPSHOT_FILENAME,
    TRAINING_EFFICIENCY_FILENAME,
)
from shaft.utils.contract_schema import (
    json_bool,
    json_int,
    json_list,
    json_string,
    load_strict_json,
    require_exact_keys,
    require_json_mapping,
    validate_json_value,
)
from .batch_planning import (
    BATCHING_RUN_METADATA_FILENAME,
    BATCH_PLANNING_CHECKPOINT_COMMIT_EXTENSION,
    build_batch_planning_checkpoint_commit_payload,
    validate_batch_planning_checkpoint_commit_payload,
)
from .distributed import all_gather_objects, barrier_if_distributed
from .optimizer_plan import OPTIMIZER_SUMMARY_FILENAME


TRAINING_CHECKPOINT_COMMIT_FILENAME = "shaft_checkpoint_commit.json"
_TRAINING_CHECKPOINT_COMMIT_VERSION = "shaft-training-checkpoint-commit-v2"
_TRAINING_CHECKPOINT_COMMIT_KEYS = frozenset(
    {
        "version",
        "global_step",
        "world_size",
        "requires_grad_scaler",
        "trainer_state_sha256",
        "artifacts",
        "required_artifacts",
        "extensions",
    }
)

_RUN_METADATA_FILENAMES = frozenset(
    {
        "trainer_state.json",
        FINETUNE_SUMMARY_FILENAME,
        OPTIMIZER_SUMMARY_FILENAME,
        BATCHING_RUN_METADATA_FILENAME,
        PROGRESS_SNAPSHOT_FILENAME,
        TRAINING_EFFICIENCY_FILENAME,
    }
)

_TRAINER_CONTROL_FIELDS = (
    "should_training_stop",
    "should_epoch_stop",
    "should_save",
    "should_evaluate",
    "should_log",
)
_CHECKPOINT_STATUS_KEYS = frozenset({"ok", "error_type", "error"})
_CHECKPOINT_CALLBACK_STATUS_KEYS = frozenset({"ok", "error_type", "error", "control"})


@dataclass(frozen=True)
class CheckpointLayout:
    path: Path
    kind: str
    has_trainer_state: bool


class ShaftCheckpointProtocol(StrEnum):
    """Select the owner of checkpoint publication, discovery, and validation."""

    COMMITTED_MANIFEST = "committed_manifest"
    BACKEND_NATIVE = "backend_native"


@dataclass(frozen=True, slots=True)
class ResolvedResumeCheckpoint:
    """One validated checkpoint generation for a single startup attempt.

    The commit manifest is content-verified once during resolution.  Later
    startup phases reuse its canonical identity and a cheap stat guard instead
    of re-reading every model/optimizer shard on every rank.
    """

    path: Path
    protocol: ShaftCheckpointProtocol
    global_step: int
    generation_fingerprint: str
    commit_fingerprint: str | None
    stat_guard: tuple[tuple[str, int, int, int, int, int], ...]

    def __post_init__(self) -> None:
        if not self.path.is_absolute():
            object.__setattr__(self, "path", self.path.resolve())
        if type(self.global_step) is not int or self.global_step < 0:
            raise ValueError("Resolved resume global_step must be a non-negative integer.")
        for name, value in (
            ("generation_fingerprint", self.generation_fingerprint),
            ("commit_fingerprint", self.commit_fingerprint),
        ):
            if value is None and name == "commit_fingerprint":
                continue
            if type(value) is not str or len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"Resolved resume {name} must be a lowercase SHA-256 digest.")

    def consensus_fingerprints(self) -> dict[str, str]:
        return {
            "resume_enabled": "true",
            "resume_protocol": self.protocol.value,
            "resume_global_step": str(self.global_step),
            "resume_generation": self.generation_fingerprint,
        }

    def __fspath__(self) -> str:
        return str(self.path)


def resolve_checkpoint_protocol(distributed_strategy: str) -> ShaftCheckpointProtocol:
    """Map one normalized distributed strategy to its explicit storage protocol."""

    strategy = str(distributed_strategy).strip().lower()
    if strategy == "ddp":
        return ShaftCheckpointProtocol.COMMITTED_MANIFEST
    if strategy in {"fsdp", "deepspeed"}:
        return ShaftCheckpointProtocol.BACKEND_NATIVE
    raise ValueError(
        f"Unsupported distributed strategy for checkpoint routing: {distributed_strategy!r}."
    )


def _normalize_checkpoint_protocol(
    protocol: ShaftCheckpointProtocol | str,
) -> ShaftCheckpointProtocol:
    try:
        return ShaftCheckpointProtocol(str(protocol))
    except ValueError as exc:
        raise ValueError(f"Unsupported checkpoint protocol: {protocol!r}.") from exc


def _load_trainer_state(path: Path) -> dict[str, Any]:
    state_path = path / "trainer_state.json"
    payload = load_strict_json(state_path, role="trainer checkpoint state")
    payload = require_json_mapping(payload, role="trainer checkpoint state")
    if "global_step" not in payload:
        raise ValueError("Trainer checkpoint state has no global_step.")
    global_step = json_int(
        payload,
        "global_step",
        role="trainer checkpoint state",
    )
    if global_step < 0:
        raise ValueError("Trainer checkpoint global_step must be >= 0.")
    return dict(payload)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        dict(payload),
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _file_stat_token(path: Path, *, relative_name: str) -> tuple[str, int, int, int, int, int]:
    stat = path.stat()
    return (
        relative_name,
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
        int(stat.st_dev),
        int(stat.st_ino),
    )


def _committed_checkpoint_stat_guard(
    checkpoint: Path,
    manifest: Mapping[str, Any],
) -> tuple[tuple[str, int, int, int, int, int], ...]:
    artifacts = require_json_mapping(
        manifest["artifacts"],
        role="training checkpoint commit manifest.artifacts",
    )
    names = [*artifacts, TRAINING_CHECKPOINT_COMMIT_FILENAME]
    return tuple(
        _file_stat_token(checkpoint / name, relative_name=name)
        for name in sorted(names)
    )


def _backend_checkpoint_stat_guard(
    checkpoint: Path,
) -> tuple[tuple[str, int, int, int, int, int], ...]:
    return (
        _file_stat_token(
            checkpoint / "trainer_state.json",
            relative_name="trainer_state.json",
        ),
    )


def validate_resolved_resume_checkpoint_guard(
    resolved: ResolvedResumeCheckpoint,
) -> None:
    """Reject mutation after resolution without repeating multi-GB hashing."""

    if resolved.protocol is ShaftCheckpointProtocol.COMMITTED_MANIFEST:
        manifest = load_strict_json(
            resolved.path / TRAINING_CHECKPOINT_COMMIT_FILENAME,
            role="training checkpoint commit manifest",
        )
        manifest = require_json_mapping(
            manifest,
            role="training checkpoint commit manifest",
        )
        if _canonical_sha256(manifest) != resolved.commit_fingerprint:
            raise ValueError("Resolved training checkpoint commit marker changed during startup.")
        actual = _committed_checkpoint_stat_guard(resolved.path, manifest)
    else:
        actual = _backend_checkpoint_stat_guard(resolved.path)
    if actual != resolved.stat_guard:
        raise ValueError("Resolved training checkpoint artifacts changed during startup.")


def resume_checkpoint_consensus_fingerprints(
    resolved: ResolvedResumeCheckpoint | None,
    *,
    protocol: ShaftCheckpointProtocol | str,
) -> dict[str, str]:
    if resolved is not None:
        return resolved.consensus_fingerprints()
    return {
        "resume_enabled": "false",
        "resume_protocol": _normalize_checkpoint_protocol(protocol).value,
        "resume_global_step": "none",
        "resume_generation": "none",
    }


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_checkpoint_artifact_directories(
    checkpoint: Path,
    artifact_names: Mapping[str, Any],
) -> None:
    """Make every recorded artifact directory entry durable before publication.

    A file ``fsync`` does not persist the directory entry that names the file, nor
    the entries that name newly created parent directories.  Sync descendants
    before their parents so that each child is durable before the entry linking it
    into the checkpoint tree.  The checkpoint root is included and is synced again
    by ``_atomic_write_json`` after the commit marker is atomically replaced.
    """

    directories = {checkpoint}
    for artifact_name in artifact_names:
        relative_path = Path(artifact_name)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("Training checkpoint artifact escapes its checkpoint root.")
        relative_parent = relative_path.parent
        while relative_parent != Path("."):
            directories.add(checkpoint / relative_parent)
            relative_parent = relative_parent.parent

    def _deepest_first(directory: Path) -> tuple[int, str]:
        relative_directory = directory.relative_to(checkpoint)
        return (-len(relative_directory.parts), relative_directory.as_posix())

    for directory in sorted(directories, key=_deepest_first):
        _fsync_directory(directory)


def _checkpoint_artifact_sizes(path: Path) -> dict[str, int]:
    artifacts: dict[str, int] = {}
    for artifact in sorted(path.rglob("*")):
        if artifact.name == TRAINING_CHECKPOINT_COMMIT_FILENAME:
            continue
        if artifact.is_symlink():
            raise ValueError(
                "Training checkpoint artifacts must not be symlinks: "
                f"{artifact.relative_to(path).as_posix()}."
            )
        if not artifact.is_file():
            continue
        relative_path = artifact.relative_to(path).as_posix()
        size = int(artifact.stat().st_size)
        artifacts[relative_path] = size
    if "trainer_state.json" not in artifacts:
        raise ValueError("Training checkpoint has no trainer_state.json artifact.")
    return artifacts


def _sharded_model_artifacts(checkpoint: Path, index_name: str) -> tuple[str, ...]:
    index_path = checkpoint / index_name
    payload = load_strict_json(
        index_path,
        role=f"model shard index {index_name}",
    )
    payload = require_json_mapping(
        payload,
        role=f"model shard index {index_name}",
    )
    if "weight_map" not in payload:
        raise ValueError(f"Model shard index has no weight_map: {index_name}.")
    weight_map = require_json_mapping(
        payload["weight_map"],
        role=f"model shard index {index_name}.weight_map",
    )
    raw_shard_names = list(weight_map.values())
    if any(type(name) is not str for name in raw_shard_names):
        raise TypeError(f"Model shard index paths must be JSON strings: {index_name}.")
    if not raw_shard_names or any(not name for name in raw_shard_names):
        raise ValueError(f"Model shard index has no valid shard paths: {index_name}.")
    shard_names = sorted(set(raw_shard_names))
    if any(Path(name).is_absolute() or ".." in Path(name).parts for name in shard_names):
        raise ValueError(f"Model shard index escapes checkpoint root: {index_name}.")
    return (index_name, *shard_names)


def _required_training_artifacts(
    checkpoint: Path,
    *,
    world_size: int,
    requires_grad_scaler: bool,
) -> tuple[str, ...]:
    layout = inspect_checkpoint_layout(checkpoint)
    if layout.kind == "adapter":
        adapter_weights = next(
            (
                name
                for name in ("adapter_model.safetensors", "adapter_model.bin")
                if (checkpoint / name).is_file()
            ),
            None,
        )
        if adapter_weights is None:
            raise ValueError("Adapter checkpoint has no adapter weights.")
        model_artifacts = ("adapter_config.json", adapter_weights)
    elif layout.kind == "full":
        if (checkpoint / "model.safetensors").is_file():
            weights = ("model.safetensors",)
        elif (checkpoint / "model.safetensors.index.json").is_file():
            weights = _sharded_model_artifacts(
                checkpoint,
                "model.safetensors.index.json",
            )
        elif (checkpoint / "pytorch_model.bin").is_file():
            weights = ("pytorch_model.bin",)
        elif (checkpoint / "pytorch_model.bin.index.json").is_file():
            weights = _sharded_model_artifacts(
                checkpoint,
                "pytorch_model.bin.index.json",
            )
        else:
            raise ValueError("Full-model checkpoint has no model weights.")
        model_artifacts = ("config.json", *weights)
    else:
        raise ValueError(f"Checkpoint has no resumable model layout: {layout.kind!r}.")

    optimizer_names = tuple(
        dict.fromkeys(
            (
                str(hf_trainer_module.OPTIMIZER_NAME),
                str(hf_trainer_module.OPTIMIZER_NAME_BIN),
            )
        )
    )
    optimizer_name = next(
        (name for name in optimizer_names if (checkpoint / name).is_file()),
        None,
    )
    if optimizer_name is None:
        raise ValueError("Training checkpoint has no optimizer state.")
    scheduler_name = str(hf_trainer_module.SCHEDULER_NAME)
    if not (checkpoint / scheduler_name).is_file():
        raise ValueError("Training checkpoint has no scheduler state.")
    scaler_name = str(hf_trainer_module.SCALER_NAME)
    scaler_artifacts = (scaler_name,) if requires_grad_scaler else ()

    if int(world_size) <= 1:
        rng_names = ("rng_state.pth",)
    else:
        rng_names = tuple(f"rng_state_{rank}.pth" for rank in range(int(world_size)))
    return tuple(
        sorted(
            (
                "trainer_state.json",
                *model_artifacts,
                optimizer_name,
                scheduler_name,
                *scaler_artifacts,
                *rng_names,
            )
        )
    )


def _validate_required_artifacts(
    checkpoint: Path,
    *,
    artifacts: Mapping[str, Any],
    required_artifacts: tuple[str, ...],
) -> None:
    missing = [name for name in required_artifacts if name not in artifacts]
    if missing:
        raise ValueError(f"Training checkpoint commit is missing required artifacts: {missing}.")
    for name in required_artifacts:
        artifact = checkpoint / name
        if not artifact.is_file() or artifact.stat().st_size <= 0:
            raise ValueError(f"Training checkpoint required artifact is missing or empty: {name}.")


def _validate_checkpoint_step(path: Path, *, global_step: int) -> None:
    if not path.name.startswith("checkpoint-"):
        return
    try:
        directory_step = int(path.name.rsplit("-", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Invalid trainer checkpoint directory name: {path.name!r}.") from exc
    if directory_step != int(global_step):
        raise ValueError(
            "Trainer checkpoint directory step differs from trainer_state.json: "
            f"directory={directory_step}, state={global_step}."
        )


def _validate_backend_native_checkpoint_location(path: Path) -> dict[str, Any]:
    if not path.name.startswith("checkpoint-"):
        raise ValueError(
            "Backend-native resume requires a direct checkpoint-<step> directory, "
            f"not a run root or final-state directory: {path}."
        )
    trainer_state = _load_trainer_state(path)
    _validate_checkpoint_step(
        path,
        global_step=int(trainer_state["global_step"]),
    )
    return trainer_state


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    dict(payload),
                    allow_nan=False,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def revoke_training_checkpoint_commit(path: str | Path) -> None:
    """Remove a stale commit point before a checkpoint directory is rewritten."""

    checkpoint = Path(path)
    commit_path = checkpoint / TRAINING_CHECKPOINT_COMMIT_FILENAME
    if commit_path.exists():
        commit_path.unlink()
        _fsync_directory(checkpoint)


def commit_training_checkpoint(
    path: str | Path,
    *,
    world_size: int,
    requires_grad_scaler: bool,
) -> Path:
    """Atomically publish a complete HF/TRL training checkpoint generation."""

    checkpoint = Path(path)
    layout = inspect_checkpoint_layout(checkpoint)
    if not layout.has_trainer_state or layout.kind not in {"full", "adapter"}:
        raise ValueError(
            f"Cannot commit incomplete trainer checkpoint layout at {checkpoint}: {layout.kind!r}."
        )
    trainer_state = _load_trainer_state(checkpoint)
    global_step = trainer_state["global_step"]
    _validate_checkpoint_step(checkpoint, global_step=global_step)
    if type(world_size) is not int:
        raise TypeError("Training checkpoint world_size must be a JSON integer.")
    resolved_world_size = world_size
    if resolved_world_size <= 0:
        raise ValueError("Training checkpoint world_size must be > 0.")
    if type(requires_grad_scaler) is not bool:
        raise TypeError("Training checkpoint requires_grad_scaler must be a JSON boolean.")
    extensions: dict[str, Any] = {}
    planning_payload = build_batch_planning_checkpoint_commit_payload(checkpoint)
    if planning_payload is not None:
        extensions[BATCH_PLANNING_CHECKPOINT_COMMIT_EXTENSION] = planning_payload
    artifact_sizes = _checkpoint_artifact_sizes(checkpoint)
    required_artifacts = _required_training_artifacts(
        checkpoint,
        world_size=resolved_world_size,
        requires_grad_scaler=requires_grad_scaler,
    )
    _validate_required_artifacts(
        checkpoint,
        artifacts=artifact_sizes,
        required_artifacts=required_artifacts,
    )
    for artifact_name in artifact_sizes:
        _fsync_file(checkpoint / artifact_name)
    artifacts = {
        name: {
            "size": size,
            "sha256": _sha256(checkpoint / name),
        }
        for name, size in artifact_sizes.items()
    }
    _fsync_checkpoint_artifact_directories(checkpoint, artifact_sizes)
    manifest = {
        "version": _TRAINING_CHECKPOINT_COMMIT_VERSION,
        "global_step": global_step,
        "world_size": resolved_world_size,
        "requires_grad_scaler": requires_grad_scaler,
        # ``trainer_state.json`` is already part of the artifact table. Reuse
        # that digest instead of reading the file a second time at publication.
        "trainer_state_sha256": artifacts["trainer_state.json"]["sha256"],
        "artifacts": artifacts,
        "required_artifacts": list(required_artifacts),
        "extensions": extensions,
    }
    return _atomic_write_json(
        checkpoint / TRAINING_CHECKPOINT_COMMIT_FILENAME,
        manifest,
    )


def validate_training_checkpoint_commit(path: str | Path) -> dict[str, Any]:
    """Validate the shared commit point and all registered checkpoint extensions."""

    checkpoint = Path(path)
    commit_path = checkpoint / TRAINING_CHECKPOINT_COMMIT_FILENAME
    try:
        manifest = load_strict_json(
            commit_path,
            role="training checkpoint commit manifest",
        )
    except FileNotFoundError as exc:
        raise ValueError(f"Training checkpoint is not committed or is torn: {checkpoint}.") from exc
    manifest = require_json_mapping(
        manifest,
        role="training checkpoint commit manifest",
    )
    require_exact_keys(
        manifest,
        expected=_TRAINING_CHECKPOINT_COMMIT_KEYS,
        role="training checkpoint commit manifest",
    )
    if (
        json_string(
            manifest,
            "version",
            role="training checkpoint commit manifest",
        )
        != _TRAINING_CHECKPOINT_COMMIT_VERSION
    ):
        raise ValueError("Unsupported training checkpoint commit manifest version.")

    layout = inspect_checkpoint_layout(checkpoint)
    if not layout.has_trainer_state or layout.kind not in {"full", "adapter"}:
        raise ValueError(f"Committed training checkpoint has incomplete layout: {layout.kind!r}.")
    trainer_state = _load_trainer_state(checkpoint)
    global_step = trainer_state["global_step"]
    if (
        json_int(
            manifest,
            "global_step",
            role="training checkpoint commit manifest",
        )
        != global_step
    ):
        raise ValueError("Training checkpoint commit global_step differs from trainer state.")
    _validate_checkpoint_step(checkpoint, global_step=global_step)
    world_size = json_int(
        manifest,
        "world_size",
        role="training checkpoint commit manifest",
    )
    if world_size <= 0:
        raise ValueError("Training checkpoint commit world_size must be > 0.")
    requires_grad_scaler = json_bool(
        manifest,
        "requires_grad_scaler",
        role="training checkpoint commit manifest",
    )
    trainer_state_sha256 = json_string(
        manifest,
        "trainer_state_sha256",
        role="training checkpoint commit manifest",
    )
    if len(trainer_state_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in trainer_state_sha256
    ):
        raise ValueError(
            "Training checkpoint trainer_state_sha256 must be a lowercase SHA-256 digest."
        )
    required_artifacts = _required_training_artifacts(
        checkpoint,
        world_size=world_size,
        requires_grad_scaler=requires_grad_scaler,
    )
    stored_required_artifacts = json_list(
        manifest,
        "required_artifacts",
        role="training checkpoint commit manifest",
    )
    if any(type(name) is not str or not name for name in stored_required_artifacts):
        raise TypeError(
            "Training checkpoint required_artifacts entries must be non-empty JSON strings."
        )
    if stored_required_artifacts != list(required_artifacts):
        raise ValueError(
            "Training checkpoint commit required-artifact set differs from its layout."
        )
    required_artifact_set = frozenset(required_artifacts)

    artifacts = require_json_mapping(
        manifest["artifacts"],
        role="training checkpoint commit manifest.artifacts",
    )
    if not artifacts:
        raise ValueError("Training checkpoint commit artifacts must not be empty.")
    if "trainer_state.json" not in artifacts:
        raise ValueError("Training checkpoint commit does not bind trainer_state.json.")
    for relative_path, raw_artifact_contract in artifacts.items():
        if not relative_path:
            raise TypeError("Training checkpoint artifact paths must be non-empty strings.")
        artifact_contract = require_json_mapping(
            raw_artifact_contract,
            role=(f"training checkpoint commit manifest.artifacts.{relative_path}"),
        )
        require_exact_keys(
            artifact_contract,
            expected=frozenset({"size", "sha256"}),
            role=(f"training checkpoint commit manifest.artifacts.{relative_path}"),
        )
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Training checkpoint artifact escapes its checkpoint root.")
        artifact_path = checkpoint / relative_path
        if artifact_path.is_symlink() or not artifact_path.is_file():
            raise ValueError(f"Training checkpoint artifact is missing: {relative_path}.")
        actual_size = int(artifact_path.stat().st_size)
        expected_size = json_int(
            artifact_contract,
            "size",
            role=(f"training checkpoint commit manifest.artifacts.{relative_path}"),
        )
        if expected_size < 0:
            raise ValueError(f"Training checkpoint artifact size is invalid: {relative_path}.")
        if actual_size != expected_size:
            raise ValueError(
                f"Training checkpoint artifact size changed after commit: {relative_path}."
            )
        expected_sha256 = artifact_contract["sha256"]
        if type(expected_sha256) is not str:
            artifact_kind = "Required" if relative_path in required_artifact_set else "Optional"
            raise TypeError(
                f"{artifact_kind} checkpoint artifact sha256 must be a JSON string: "
                f"{relative_path}."
            )
        if len(expected_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in expected_sha256
        ):
            raise ValueError(f"Checkpoint artifact sha256 is invalid: {relative_path}.")
        if (
            relative_path == "trainer_state.json"
            and expected_sha256 != trainer_state_sha256
        ):
            raise ValueError(
                "Training checkpoint trainer_state_sha256 differs from its artifact digest."
            )
        if _sha256(artifact_path) != expected_sha256:
            raise ValueError(
                f"Training checkpoint artifact digest changed after commit: {relative_path}."
            )
    _validate_required_artifacts(
        checkpoint,
        artifacts=artifacts,
        required_artifacts=required_artifacts,
    )

    planning_payload = build_batch_planning_checkpoint_commit_payload(checkpoint)
    extensions = require_json_mapping(
        manifest["extensions"],
        role="training checkpoint commit manifest.extensions",
    )
    require_exact_keys(
        extensions,
        expected=(
            frozenset()
            if planning_payload is None
            else frozenset({BATCH_PLANNING_CHECKPOINT_COMMIT_EXTENSION})
        ),
        role="training checkpoint commit manifest.extensions",
    )
    validate_json_value(
        extensions,
        role="training checkpoint commit manifest.extensions",
    )
    stored_planning_payload = extensions.get(BATCH_PLANNING_CHECKPOINT_COMMIT_EXTENSION)
    if planning_payload is None:
        if stored_planning_payload is not None:
            raise ValueError("Training checkpoint commit has stale batch-planning state.")
    else:
        if not isinstance(stored_planning_payload, Mapping):
            raise TypeError("Training checkpoint commit is missing its batch-planning extension.")
        validate_batch_planning_checkpoint_commit_payload(
            checkpoint,
            stored_planning_payload,
        )
    return dict(manifest)


def training_checkpoint_is_committed(path: str | Path) -> bool:
    try:
        validate_training_checkpoint_commit(path)
        return True
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _indexed_callback_schedule(
    callbacks: list[Any],
) -> tuple[tuple[tuple[str, int], ...], dict[tuple[str, int], Any]]:
    occurrences: dict[str, int] = {}
    schedule: list[tuple[str, int]] = []
    callbacks_by_token: dict[tuple[str, int], Any] = {}
    for callback in callbacks:
        name = f"{callback.__class__.__module__}.{callback.__class__.__qualname__}"
        occurrence = occurrences.get(name, 0)
        occurrences[name] = occurrence + 1
        token = (name, occurrence)
        schedule.append(token)
        callbacks_by_token[token] = callback
    return tuple(schedule), callbacks_by_token


def _validate_shared_callback_schedule(
    peer_schedules: list[Any],
) -> tuple[tuple[str, int], ...]:
    """Require every rank to execute the same ordered callback schedule."""

    normalized_schedules: list[tuple[tuple[str, int], ...]] = []
    for rank, raw_schedule in enumerate(peer_schedules):
        if not isinstance(raw_schedule, (list, tuple)):
            raise TypeError(f"Distributed callback schedule rank {rank} must be a sequence.")
        normalized: list[tuple[str, int]] = []
        occurrences: dict[str, int] = {}
        for index, token in enumerate(raw_schedule):
            if not isinstance(token, (list, tuple)) or len(token) != 2:
                raise TypeError(
                    "Distributed callback schedule tokens must be two-item "
                    f"sequences: rank={rank}, index={index}."
                )
            name, occurrence = token
            if type(name) is not str or not name:
                raise TypeError("Distributed callback schedule names must be non-empty strings.")
            if type(occurrence) is not int or occurrence < 0:
                raise TypeError(
                    "Distributed callback schedule occurrences must be non-negative integers."
                )
            expected_occurrence = occurrences.get(name, 0)
            if occurrence != expected_occurrence:
                raise ValueError(
                    "Distributed callback schedule occurrences must be contiguous "
                    f"per callback: rank={rank}, name={name!r}, "
                    f"expected={expected_occurrence}, actual={occurrence}."
                )
            occurrences[name] = expected_occurrence + 1
            normalized.append((name, occurrence))
        normalized_schedules.append(tuple(normalized))
    if not normalized_schedules:
        raise RuntimeError("Distributed checkpoint callback schedule gather was empty.")
    shared_schedule = normalized_schedules[0]
    if any(schedule != shared_schedule for schedule in normalized_schedules[1:]):
        raise RuntimeError(
            "Distributed checkpoint requires identical ordered on_save callback "
            f"schedules across ranks: {normalized_schedules!r}."
        )
    return shared_schedule


def _strict_trainer_control_state(value: Any, *, role: str) -> dict[str, bool]:
    control = require_json_mapping(value, role=role)
    require_exact_keys(
        control,
        expected=frozenset(_TRAINER_CONTROL_FIELDS),
        role=role,
    )
    return {name: json_bool(control, name, role=role) for name in _TRAINER_CONTROL_FIELDS}


def _strict_checkpoint_status(
    value: Any,
    *,
    role: str,
    includes_control: bool,
) -> dict[str, Any]:
    status = require_json_mapping(value, role=role)
    require_exact_keys(
        status,
        expected=(
            _CHECKPOINT_CALLBACK_STATUS_KEYS if includes_control else _CHECKPOINT_STATUS_KEYS
        ),
        role=role,
    )
    ok = json_bool(status, "ok", role=role)
    error_type = status["error_type"]
    error = status["error"]
    if ok:
        if error_type is not None or error is not None:
            raise ValueError(f"Successful {role} must carry null error fields.")
    else:
        if type(error_type) is not str or not error_type.strip():
            raise TypeError(f"Failed {role}.error_type must be a non-empty JSON string.")
        if type(error) is not str or not error.strip():
            raise TypeError(f"Failed {role}.error must be a non-empty JSON string.")

    normalized: dict[str, Any] = {
        "ok": ok,
        "error_type": error_type,
        "error": error,
    }
    if includes_control:
        raw_control = status["control"]
        if raw_control is None:
            if ok:
                raise ValueError(f"Successful {role} must carry TrainerControl state.")
            normalized["control"] = None
        else:
            normalized["control"] = _strict_trainer_control_state(
                raw_control,
                role=f"{role}.control",
            )
    return normalized


def _checkpoint_error_fields(error: Exception | None) -> tuple[str | None, str | None]:
    if error is None:
        return None, None
    message = str(error).strip() or repr(error)
    return type(error).__name__, message


class ShaftCheckpointCommitMixin:
    """Give HF/TRL trainers one distributed checkpoint commit protocol."""

    def __init__(
        self,
        *args: Any,
        shaft_checkpoint_protocol: ShaftCheckpointProtocol | str = (
            ShaftCheckpointProtocol.COMMITTED_MANIFEST
        ),
        **kwargs: Any,
    ) -> None:
        checkpoint_protocol = _normalize_checkpoint_protocol(shaft_checkpoint_protocol)
        super().__init__(*args, **kwargs)
        runtime_protocol = (
            ShaftCheckpointProtocol.BACKEND_NATIVE
            if bool(
                getattr(self, "is_deepspeed_enabled", False)
                or getattr(self, "is_fsdp_enabled", False)
            )
            else ShaftCheckpointProtocol.COMMITTED_MANIFEST
        )
        if checkpoint_protocol != runtime_protocol:
            raise ValueError(
                "Trainer checkpoint protocol does not match its initialized backend: "
                f"requested={checkpoint_protocol.value!r}, runtime={runtime_protocol.value!r}."
            )
        self._shaft_checkpoint_protocol = checkpoint_protocol
        self._shaft_pending_checkpoint_path: Path | None = None
        if checkpoint_protocol is ShaftCheckpointProtocol.BACKEND_NATIVE:
            return
        # Replace only on_save so callbacks added later remain part of the same
        # converged pre-commit phase without another sentinel callback/state source.
        self.callback_handler.on_save = self._run_converged_shaft_on_save

    def _uses_shaft_checkpoint_commit(self) -> bool:
        return self._shaft_checkpoint_protocol is ShaftCheckpointProtocol.COMMITTED_MANIFEST

    def _run_converged_shaft_on_save(
        self,
        args,
        state,
        control,
        **kwargs: Any,
    ):
        control.should_save = False
        handler = self.callback_handler
        callbacks = list(handler.callbacks)
        callback_schedule, callbacks_by_token = _indexed_callback_schedule(callbacks)
        peer_schedules = all_gather_objects(callback_schedule)
        shared_schedule = _validate_shared_callback_schedule(peer_schedules)
        for callback_token in shared_schedule:
            callback = callbacks_by_token[callback_token]
            callback_error: Exception | None = None
            callback_result = None
            try:
                callback_result = callback.on_save(
                    args,
                    state,
                    control,
                    model=handler.model,
                    processing_class=handler.processing_class,
                    optimizer=handler.optimizer,
                    lr_scheduler=handler.lr_scheduler,
                    train_dataloader=handler.train_dataloader,
                    eval_dataloader=handler.eval_dataloader,
                    **kwargs,
                )
            except Exception as exc:  # noqa: BLE001 - converge each callback across ranks
                callback_error = exc
            resolved_control = callback_result if callback_result is not None else control
            self._raise_synchronized_checkpoint_callback_error(
                f"checkpoint on_save callback {callback_token[0]}",
                callback_error,
                resolved_control,
            )
            control = resolved_control
        self._commit_and_rotate_shaft_checkpoint()
        return control

    def _prepare_shaft_checkpoint_save(self, checkpoint_path: Path) -> None:
        _ = checkpoint_path

    def _save_checkpoint(self, model, trial) -> None:  # noqa: ANN001
        barrier_if_distributed()
        checkpoint_path = (
            Path(self._get_output_dir(trial=trial)) / f"checkpoint-{int(self.state.global_step)}"
        )
        if not self._uses_shaft_checkpoint_commit():
            # Telemetry transaction preparation is backend-agnostic and existed
            # before the storage protocols split. Backend-native save/rotation
            # still begins from the same prepared generation.
            prepare_error: Exception | None = None
            try:
                self._prepare_shaft_checkpoint_save(checkpoint_path)
            except Exception as exc:  # noqa: BLE001 - converge before backend collectives
                prepare_error = exc
            self._raise_synchronized_checkpoint_error(
                "checkpoint prepare",
                prepare_error,
            )
            super()._save_checkpoint(model, trial)
            return
        self._shaft_pending_checkpoint_path = checkpoint_path
        revoke_error: Exception | None = None
        if self.is_world_process_zero():
            try:
                revoke_training_checkpoint_commit(checkpoint_path)
            except Exception as exc:  # noqa: BLE001 - synchronize failure across ranks
                revoke_error = exc
        self._raise_synchronized_checkpoint_error("checkpoint begin", revoke_error)

        prepare_error: Exception | None = None
        try:
            self._prepare_shaft_checkpoint_save(checkpoint_path)
        except Exception as exc:  # noqa: BLE001 - converge prepare failures across ranks
            prepare_error = exc
        self._raise_synchronized_checkpoint_error("checkpoint prepare", prepare_error)
        save_total_limit = self.args.save_total_limit
        local_error: Exception | None = None
        # HF rotates before peer-rank save failures can converge. Under the
        # committed-manifest protocol, rotation belongs after the shared commit.
        self.args.save_total_limit = None
        try:
            super()._save_checkpoint(model, trial)
        except Exception as exc:  # noqa: BLE001 - synchronize failure across ranks
            local_error = exc
        finally:
            self.args.save_total_limit = save_total_limit
        self._raise_synchronized_checkpoint_error("checkpoint save", local_error)

    def _commit_and_rotate_shaft_checkpoint(self) -> None:
        checkpoint_path = self._shaft_pending_checkpoint_path
        validation_error: Exception | None = None
        try:
            if checkpoint_path is None:
                raise RuntimeError("Checkpoint commit phase has no pending checkpoint.")
            expected_name = f"checkpoint-{int(self.state.global_step)}"
            if checkpoint_path.name != expected_name:
                raise RuntimeError(
                    "Pending checkpoint step differs from Trainer state at commit time."
                )
        except Exception as exc:  # noqa: BLE001 - converge local state drift
            validation_error = exc
        self._raise_synchronized_checkpoint_error(
            "checkpoint commit preflight",
            validation_error,
        )
        assert checkpoint_path is not None
        commit_error: Exception | None = None
        if self.is_world_process_zero():
            try:
                commit_training_checkpoint(
                    checkpoint_path,
                    world_size=int(self.args.world_size),
                    requires_grad_scaler=(getattr(self.accelerator, "scaler", None) is not None),
                )
            except Exception as exc:  # noqa: BLE001 - synchronize failure across ranks
                commit_error = exc
        self._raise_synchronized_checkpoint_error("checkpoint commit", commit_error)

        rotation_error: Exception | None = None
        if self.args.should_save:
            try:
                hf_trainer_module.rotate_checkpoints(
                    output_dir=str(checkpoint_path.parent),
                    save_total_limit=self.args.save_total_limit,
                    best_model_checkpoint=self.state.best_model_checkpoint,
                    use_mtime=True,
                )
            except Exception as exc:  # noqa: BLE001 - synchronize failure across ranks
                rotation_error = exc
        self._raise_synchronized_checkpoint_error("checkpoint rotation", rotation_error)
        self._shaft_pending_checkpoint_path = None

    @staticmethod
    def _raise_synchronized_checkpoint_callback_error(
        operation: str,
        local_error: Exception | None,
        control: Any,
    ) -> None:
        control_state: dict[str, bool] | None = None
        synchronized_error = local_error
        try:
            control_state = {name: getattr(control, name) for name in _TRAINER_CONTROL_FIELDS}
            control_state = _strict_trainer_control_state(
                control_state,
                role=f"local {operation} TrainerControl state",
            )
        except Exception as exc:  # noqa: BLE001 - converge malformed callback results
            control_state = None
            if synchronized_error is None:
                synchronized_error = exc
        error_type, error = _checkpoint_error_fields(synchronized_error)
        statuses = all_gather_objects(
            {
                "ok": synchronized_error is None,
                "error_type": error_type,
                "error": error,
                "control": control_state,
            }
        )
        try:
            normalized_statuses = [
                _strict_checkpoint_status(
                    status,
                    role=f"distributed {operation} status rank {rank}",
                    includes_control=True,
                )
                for rank, status in enumerate(statuses)
            ]
            if not normalized_statuses:
                raise ValueError("Distributed checkpoint status gather was empty.")
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Distributed {operation} returned a malformed status envelope: {statuses!r}."
            ) from exc
        failures = [status for status in normalized_statuses if status["ok"] is False]
        if failures:
            if synchronized_error is not None:
                raise synchronized_error
            raise RuntimeError(f"Distributed {operation} failed on a peer rank: {failures!r}.")
        assert control_state is not None
        peer_controls = [status["control"] for status in normalized_statuses]
        if any(peer_control != control_state for peer_control in peer_controls):
            raise RuntimeError(
                f"Distributed {operation} produced divergent TrainerControl state: "
                f"{peer_controls!r}."
            )

    @staticmethod
    def _raise_synchronized_checkpoint_error(
        operation: str,
        local_error: Exception | None,
    ) -> None:
        error_type, error = _checkpoint_error_fields(local_error)
        statuses = all_gather_objects(
            {
                "ok": local_error is None,
                "error_type": error_type,
                "error": error,
            }
        )
        try:
            normalized_statuses = [
                _strict_checkpoint_status(
                    status,
                    role=f"distributed {operation} status rank {rank}",
                    includes_control=False,
                )
                for rank, status in enumerate(statuses)
            ]
            if not normalized_statuses:
                raise ValueError("Distributed checkpoint status gather was empty.")
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Distributed {operation} returned a malformed status envelope: {statuses!r}."
            ) from exc
        failures = [status for status in normalized_statuses if status["ok"] is False]
        if not failures:
            return
        if local_error is not None:
            raise local_error
        raise RuntimeError(f"Distributed {operation} failed on a peer rank: {failures!r}.")


def inspect_checkpoint_layout(path: str | Path) -> CheckpointLayout:
    target = Path(path)
    has_trainer_state = (target / "trainer_state.json").exists()
    has_adapter = (target / "adapter_config.json").exists() and (
        (target / "adapter_model.safetensors").exists() or (target / "adapter_model.bin").exists()
    )
    has_full_model = (target / "config.json").exists() and (
        (target / "model.safetensors").exists()
        or (target / "model.safetensors.index.json").exists()
        or (target / "pytorch_model.bin").exists()
        or (target / "pytorch_model.bin.index.json").exists()
    )
    if has_adapter:
        kind = "adapter"
    elif has_full_model:
        kind = "full"
    elif has_trainer_state:
        kind = "trainer_state_only"
    else:
        kind = "unknown"
    return CheckpointLayout(path=target, kind=kind, has_trainer_state=has_trainer_state)


def ensure_hf_export_layout(
    path: str | Path,
    *,
    finetune_mode: str,
    model_meta: ModelMeta | ShaftModelAdapter | None = None,
) -> None:
    layout = inspect_checkpoint_layout(path)
    mode = str(finetune_mode).strip().lower()
    if mode == "full":
        if layout.kind != "full":
            raise ValueError(f"Expected a full HF export at {path}, found {layout.kind!r}.")
        if model_meta is not None:
            missing = [
                name
                for name in model_meta.required_saved_files()
                if not (Path(path) / name).exists()
            ]
            if missing:
                raise ValueError(f"Missing additional saved files in export {path}: {missing}")
        return
    if mode in {"lora", "dora", "qlora"}:
        if layout.kind != "adapter":
            raise ValueError(f"Expected a PEFT adapter export at {path}, found {layout.kind!r}.")
        return
    raise ValueError(f"Unsupported finetune mode: {finetune_mode!r}.")


def resolve_best_export_dir(output_dir: str | Path) -> Path:
    return Path(output_dir) / "best"


def _resolved_committed_checkpoint(
    checkpoint: Path,
    manifest: Mapping[str, Any],
) -> ResolvedResumeCheckpoint:
    commit_fingerprint = _canonical_sha256(manifest)
    global_step = json_int(
        manifest,
        "global_step",
        role="training checkpoint commit manifest",
    )
    generation_fingerprint = _canonical_sha256(
        {
            "protocol": ShaftCheckpointProtocol.COMMITTED_MANIFEST.value,
            "global_step": global_step,
            "commit_fingerprint": commit_fingerprint,
        }
    )
    return ResolvedResumeCheckpoint(
        path=checkpoint.resolve(),
        protocol=ShaftCheckpointProtocol.COMMITTED_MANIFEST,
        global_step=global_step,
        generation_fingerprint=generation_fingerprint,
        commit_fingerprint=commit_fingerprint,
        stat_guard=_committed_checkpoint_stat_guard(checkpoint, manifest),
    )


def _resolved_backend_checkpoint(
    checkpoint: Path,
    trainer_state: Mapping[str, Any],
) -> ResolvedResumeCheckpoint:
    global_step = json_int(
        trainer_state,
        "global_step",
        role="trainer checkpoint state",
    )
    trainer_state_fingerprint = _canonical_sha256(trainer_state)
    generation_fingerprint = _canonical_sha256(
        {
            "protocol": ShaftCheckpointProtocol.BACKEND_NATIVE.value,
            "global_step": global_step,
            "trainer_state_fingerprint": trainer_state_fingerprint,
        }
    )
    return ResolvedResumeCheckpoint(
        path=checkpoint.resolve(),
        protocol=ShaftCheckpointProtocol.BACKEND_NATIVE,
        global_step=global_step,
        generation_fingerprint=generation_fingerprint,
        commit_fingerprint=None,
        stat_guard=_backend_checkpoint_stat_guard(checkpoint),
    )


def resolve_resume_checkpoint_generation(
    path: str | Path | None,
    *,
    protocol: ShaftCheckpointProtocol | str,
    require_planning_state: bool = False,
) -> ResolvedResumeCheckpoint | None:
    if path is None:
        return None
    resolved_protocol = _normalize_checkpoint_protocol(protocol)
    if resolved_protocol is ShaftCheckpointProtocol.BACKEND_NATIVE and require_planning_state:
        raise ValueError(
            "Batch-planning exact resume requires checkpoint protocol "
            f"{ShaftCheckpointProtocol.COMMITTED_MANIFEST.value!r}."
        )
    target = Path(path).resolve()
    if not target.exists():
        raise FileNotFoundError(f"resume_from checkpoint path not found: {target}")

    child_candidates: list[tuple[int, Path]] = []
    if target.is_dir():
        for candidate in target.glob("checkpoint-*"):
            if not candidate.is_dir():
                continue
            try:
                step = int(candidate.name.rsplit("-", 1)[1])
            except (IndexError, ValueError):
                continue
            child_candidates.append((step, candidate))

    # A run root is always resolved from its checkpoint children. Root-level
    # final exports or stale trainer state must never mask a newer generation.
    if child_candidates:
        if resolved_protocol is ShaftCheckpointProtocol.BACKEND_NATIVE:
            for _step, candidate in sorted(child_candidates, reverse=True):
                try:
                    trainer_state = _validate_backend_native_checkpoint_location(candidate)
                except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
                    continue
                return _resolved_backend_checkpoint(candidate, trainer_state)
        else:
            # Validate newest-first and stop at the first complete generation.
            # A successful manifest already verifies its planning extension, so
            # neither older checkpoints nor the selected checkpoint are hashed
            # again during this startup.
            for _step, candidate in sorted(child_candidates, reverse=True):
                try:
                    manifest = validate_training_checkpoint_commit(candidate)
                except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
                    continue
                extensions = require_json_mapping(
                    manifest["extensions"],
                    role="training checkpoint commit manifest.extensions",
                )
                if (
                    require_planning_state
                    and BATCH_PLANNING_CHECKPOINT_COMMIT_EXTENSION not in extensions
                ):
                    continue
                return _resolved_committed_checkpoint(candidate, manifest)
        checkpoint_kind = "planned " if require_planning_state else ""
        raise ValueError(
            f"No valid {checkpoint_kind}{resolved_protocol.value} trainer checkpoint "
            f"found under: {target}"
        )

    layout = inspect_checkpoint_layout(target)
    if resolved_protocol is ShaftCheckpointProtocol.BACKEND_NATIVE:
        if layout.has_trainer_state:
            trainer_state = _validate_backend_native_checkpoint_location(target)
            return _resolved_backend_checkpoint(target, trainer_state)
        raise ValueError(f"No backend-native trainer checkpoint found under: {target}")

    if layout.has_trainer_state and layout.kind in {"full", "adapter"}:
        manifest = validate_training_checkpoint_commit(target)
        extensions = require_json_mapping(
            manifest["extensions"],
            role="training checkpoint commit manifest.extensions",
        )
        if (
            require_planning_state
            and BATCH_PLANNING_CHECKPOINT_COMMIT_EXTENSION not in extensions
        ):
            raise ValueError(f"Checkpoint is missing valid batch-planning state: {target}")
        return _resolved_committed_checkpoint(target, manifest)

    checkpoint_kind = "planned " if require_planning_state else ""
    raise ValueError(f"No committed {checkpoint_kind}trainer checkpoint found under: {target}")


def resolve_resume_checkpoint(
    path: str | Path | None,
    *,
    protocol: ShaftCheckpointProtocol | str,
    require_planning_state: bool = False,
) -> str | None:
    """Compatibility path API; pipelines should retain the typed generation."""

    resolved = resolve_resume_checkpoint_generation(
        path,
        protocol=protocol,
        require_planning_state=require_planning_state,
    )
    return None if resolved is None else str(resolved.path)


def prune_root_output_layout(output_dir: str | Path) -> None:
    root = Path(output_dir)
    if not root.is_dir():
        return

    has_checkpoint_dir = any(
        item.is_dir() and item.name.startswith("checkpoint-") for item in root.iterdir()
    )
    if not has_checkpoint_dir and not (root / "best").exists():
        return

    layout = inspect_checkpoint_layout(root)
    if layout.kind == "unknown":
        return

    if not any(
        item.is_dir() and (item.name == "best" or item.name.startswith("checkpoint-"))
        for item in root.iterdir()
    ):
        return

    for item in root.iterdir():
        if item.name.startswith("."):
            continue
        if item.is_file() and item.name in _RUN_METADATA_FILENAMES:
            continue
        if item.is_dir() and (item.name == "best" or item.name.startswith("checkpoint-")):
            continue
        if item.is_dir():
            shutil.rmtree(item)
        elif item.is_file():
            item.unlink(missing_ok=True)


def validate_resume_checkpoint(
    path: str | Path | ResolvedResumeCheckpoint,
    *,
    finetune_mode: str,
    protocol: ShaftCheckpointProtocol | str,
) -> None:
    resolved_generation = (
        path if isinstance(path, ResolvedResumeCheckpoint) else None
    )
    checkpoint = (
        resolved_generation.path
        if resolved_generation is not None
        else Path(path)
    )
    layout = inspect_checkpoint_layout(checkpoint)
    mode = str(finetune_mode).strip().lower()
    if not layout.has_trainer_state:
        raise ValueError(
            f"resume_from requires trainer_state.json in checkpoint: {checkpoint}"
        )
    resolved_protocol = _normalize_checkpoint_protocol(protocol)
    if resolved_generation is not None:
        if resolved_generation.protocol is not resolved_protocol:
            raise ValueError(
                "Resolved resume checkpoint protocol differs from the requested protocol."
            )
        validate_resolved_resume_checkpoint_guard(resolved_generation)
    if resolved_protocol is ShaftCheckpointProtocol.BACKEND_NATIVE:
        _validate_backend_native_checkpoint_location(checkpoint)
        if mode not in {"full", "lora", "dora", "qlora"}:
            raise ValueError(f"Unsupported finetune mode: {finetune_mode!r}")
        # FSDP/DeepSpeed own the storage representation and compatibility
        # validation even when they also expose conventional-looking files.
        return
    trainer_state = _load_trainer_state(checkpoint)
    _validate_checkpoint_step(
        checkpoint,
        global_step=int(trainer_state["global_step"]),
    )
    if resolved_generation is None:
        validate_training_checkpoint_commit(checkpoint)
    if mode == "full":
        if layout.kind != "full":
            raise ValueError(
                f"Expected full-model checkpoint for resume under mode='full', found {layout.kind!r}."
            )
        return
    if mode in {"lora", "dora", "qlora"}:
        if layout.kind != "adapter":
            raise ValueError(
                f"Expected adapter checkpoint for resume under mode={mode!r}, found {layout.kind!r}."
            )
        return
    raise ValueError(f"Unsupported finetune mode: {finetune_mode!r}")


def validate_training_state_policy(config: RuntimeConfig) -> None:
    train_cfg = config.train
    eval_cfg = config.eval
    if (
        train_cfg.init_from_checkpoint is not None
        and train_cfg.resume_from_checkpoint is not None
    ):
        raise ValueError(
            "train.init_from_checkpoint and train.resume_from_checkpoint are "
            "mutually exclusive: init starts a new schedule, while resume restores "
            "the previous training state."
        )
    if not train_cfg.load_best_model_at_end:
        return
    if not eval_cfg.enabled:
        raise ValueError("train.load_best_model_at_end=true requires eval.enabled=true.")
    if train_cfg.save_strategy == "no":
        raise ValueError("load_best_model_at_end requires train.save_strategy != 'no'.")
    if eval_cfg.eval_strategy == "no":
        raise ValueError("load_best_model_at_end requires eval.eval_strategy != 'no'.")
    if train_cfg.save_strategy != eval_cfg.eval_strategy:
        raise ValueError(
            "save_strategy and eval_strategy must match when load_best_model_at_end=true."
        )
    if (
        train_cfg.save_strategy == "steps"
        and int(train_cfg.save_steps) % int(eval_cfg.eval_steps) != 0
    ):
        raise ValueError(
            "When using step-based best model loading, save_steps must be a multiple of eval_steps."
        )
