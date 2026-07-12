from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any
import uuid

from transformers import TrainerCallback
from transformers.trainer import TRAINER_STATE_NAME
from transformers.trainer_callback import ExportableState

from shaft.data import (
    ShaftBoundedBatchSampler,
    ShaftBoundedBatchingSpec,
    ShaftBoundedBatchingState,
)

from .distributed import broadcast_object_from_rank_zero, is_rank_zero


BATCHING_RUN_METADATA_FILENAME = "shaft_batching_run_metadata.json"
BOUNDED_BATCHING_CALLBACK_NAME = "ShaftBoundedBatchingCallback"

logger = logging.getLogger(__name__)


def _optional_int(payload: dict[str, Any], field_name: str) -> int | None:
    value = payload.get(field_name)
    return None if value is None else int(value)


@dataclass(frozen=True, slots=True)
class ShaftBatchingRunMetadata:
    strategy: str
    per_device_train_batch_size: int
    data_world_size: int
    gradient_accumulation_steps: int
    min_pixels: int | None
    max_pixels: int | None
    source_weights: tuple[tuple[str, float], ...]
    media_snapshot_id: str | None = None
    buffer_size: int | None = None
    cost_cache_size: int | None = None
    max_samples_per_microbatch: int | None = None
    max_padded_tokens: int | None = None
    max_vision_patches: int | None = None
    contract_fingerprint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": str(self.strategy),
            "per_device_train_batch_size": int(self.per_device_train_batch_size),
            "data_world_size": int(self.data_world_size),
            "gradient_accumulation_steps": int(self.gradient_accumulation_steps),
            "min_pixels": None if self.min_pixels is None else int(self.min_pixels),
            "max_pixels": None if self.max_pixels is None else int(self.max_pixels),
            "source_weights": {
                name: float(weight) for name, weight in self.source_weights
            },
            "media_snapshot_id": self.media_snapshot_id,
            "buffer_size": self.buffer_size,
            "cost_cache_size": self.cost_cache_size,
            "max_samples_per_microbatch": self.max_samples_per_microbatch,
            "max_padded_tokens": self.max_padded_tokens,
            "max_vision_patches": self.max_vision_patches,
            "contract_fingerprint": self.contract_fingerprint,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ShaftBatchingRunMetadata":
        source_weights = payload.get("source_weights", {})
        if not isinstance(source_weights, dict):
            raise TypeError("Batching metadata source_weights must be a mapping.")
        return cls(
            strategy=str(payload["strategy"]),
            per_device_train_batch_size=int(payload["per_device_train_batch_size"]),
            data_world_size=int(payload["data_world_size"]),
            gradient_accumulation_steps=int(payload["gradient_accumulation_steps"]),
            min_pixels=_optional_int(payload, "min_pixels"),
            max_pixels=_optional_int(payload, "max_pixels"),
            source_weights=tuple(
                sorted(
                    (str(name), float(weight))
                    for name, weight in source_weights.items()
                )
            ),
            media_snapshot_id=(
                None
                if payload.get("media_snapshot_id") is None
                else str(payload["media_snapshot_id"])
            ),
            buffer_size=_optional_int(payload, "buffer_size"),
            cost_cache_size=_optional_int(payload, "cost_cache_size"),
            max_samples_per_microbatch=_optional_int(
                payload, "max_samples_per_microbatch"
            ),
            max_padded_tokens=_optional_int(payload, "max_padded_tokens"),
            max_vision_patches=_optional_int(payload, "max_vision_patches"),
            contract_fingerprint=(
                None
                if payload.get("contract_fingerprint") is None
                else str(payload["contract_fingerprint"])
            ),
        )


def build_batching_run_metadata(
    *,
    config: Any,
    training_args: Any,
    bounded_spec: ShaftBoundedBatchingSpec | None = None,
) -> ShaftBatchingRunMetadata:
    strategy = str(config.data.batching.strategy).strip().lower()
    if (strategy == "bounded_cost_aware") != (bounded_spec is not None):
        raise ValueError(
            "Resolved bounded batching spec does not match data.batching.strategy."
        )
    return ShaftBatchingRunMetadata(
        strategy=strategy,
        per_device_train_batch_size=int(training_args.per_device_train_batch_size),
        data_world_size=int(training_args.world_size),
        gradient_accumulation_steps=int(training_args.gradient_accumulation_steps),
        min_pixels=config.data.min_pixels,
        max_pixels=config.data.max_pixels,
        source_weights=tuple(
            sorted(
                (dataset.dataset_name, float(dataset.weight))
                for dataset in config.data.datasets
                if dataset.enabled and float(dataset.weight) > 0
            )
        ),
        media_snapshot_id=config.data.media_snapshot_id,
        buffer_size=(None if bounded_spec is None else bounded_spec.buffer_size),
        cost_cache_size=(
            None if bounded_spec is None else int(config.data.batching.cost_cache_size)
        ),
        max_samples_per_microbatch=(
            None
            if bounded_spec is None
            else bounded_spec.max_samples_per_microbatch
        ),
        max_padded_tokens=(
            None if bounded_spec is None else bounded_spec.max_padded_tokens
        ),
        max_vision_patches=(
            None if bounded_spec is None else bounded_spec.max_vision_patches
        ),
        contract_fingerprint=(
            None if bounded_spec is None else bounded_spec.fingerprint
        ),
    )


def batching_run_metadata_path(path: str | Path) -> Path:
    return Path(path) / BATCHING_RUN_METADATA_FILENAME


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def write_batching_run_metadata(
    path: str | Path,
    metadata: ShaftBatchingRunMetadata,
) -> Path:
    return _atomic_write_json(batching_run_metadata_path(path), metadata.to_dict())


def publish_batching_run_metadata(
    path: str | Path,
    metadata: ShaftBatchingRunMetadata,
) -> Path:
    publish_error: Exception | None = None
    status: dict[str, Any] | None = None
    if is_rank_zero():
        try:
            target = write_batching_run_metadata(path, metadata)
            status = {"ok": True, "path": str(target)}
        except Exception as exc:  # noqa: BLE001 - propagate to every rank
            publish_error = exc
            status = {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
    status = broadcast_object_from_rank_zero(status)
    if not isinstance(status, dict) or not bool(status.get("ok")):
        if publish_error is not None:
            raise publish_error
        raise RuntimeError(f"Rank-zero batching metadata publish failed: {status!r}.")
    return Path(str(status["path"]))


def load_batching_run_metadata(path: str | Path) -> ShaftBatchingRunMetadata:
    payload = json.loads(batching_run_metadata_path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Batching run metadata must be a JSON object.")
    return ShaftBatchingRunMetadata.from_dict(payload)


def build_bounded_resume_contract_fingerprint(
    *,
    config: Any,
    training_args: Any,
) -> str:
    """Bind exact Trainer resume to duration, optimizer and scheduler semantics."""

    train = config.train
    payload = (
        "shaft-bounded-resume-contract-v1",
        int(training_args.max_steps),
        int(training_args.gradient_accumulation_steps),
        str(train.optimizer_name),
        str(train.scheduler_name),
        float(train.scheduler_num_cycles),
        float(train.scheduler_power),
        float(train.warmup_ratio),
        str(train.lr_scheduler_type),
        float(train.learning_rate),
        float(train.weight_decay),
        float(train.adam_beta1),
        float(train.adam_beta2),
        float(train.adam_epsilon),
        tuple(sorted((str(key), float(value)) for key, value in train.param_group_lrs.items())),
    )
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


def _load_bounded_callback_payload(path: str | Path) -> dict[str, Any]:
    target = Path(path) / TRAINER_STATE_NAME
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Trainer state must be a JSON object.")
    callbacks = payload.get("stateful_callbacks")
    if not isinstance(callbacks, dict):
        raise ValueError("Trainer state has no stateful callback payloads.")
    callback_payload = callbacks.get(BOUNDED_BATCHING_CALLBACK_NAME)
    if isinstance(callback_payload, list):
        callback_payload = callback_payload[-1] if callback_payload else None
    if not isinstance(callback_payload, dict):
        raise ValueError(
            "Trainer checkpoint has no committed bounded batching callback state."
        )
    return callback_payload


def checkpoint_has_bounded_batching_state(path: str | Path) -> bool:
    try:
        callback_payload = _load_bounded_callback_payload(path)
        args = callback_payload["args"]
        attributes = callback_payload["attributes"]
        if not isinstance(args, dict) or not isinstance(attributes, dict):
            return False
        spec = ShaftBoundedBatchingSpec.from_dict(args["spec"])
        state = ShaftBoundedBatchingState.from_dict(attributes["bounded_state"])
        return state.contract_fingerprint == spec.fingerprint
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def load_bounded_batching_state(
    path: str | Path,
    *,
    expected_spec: ShaftBoundedBatchingSpec,
    expected_global_step: int,
    gradient_accumulation_steps: int,
    expected_resume_contract_fingerprint: str,
) -> ShaftBoundedBatchingState:
    callback_payload = _load_bounded_callback_payload(path)
    args_payload = callback_payload.get("args")
    attributes_payload = callback_payload.get("attributes")
    if not isinstance(args_payload, dict) or not isinstance(attributes_payload, dict):
        raise TypeError("Bounded callback state must contain args and attributes mappings.")
    spec_payload = args_payload.get("spec")
    state_payload = attributes_payload.get("bounded_state")
    if not isinstance(spec_payload, dict) or not isinstance(state_payload, dict):
        raise TypeError("Bounded batching checkpoint must contain spec and state mappings.")
    actual_resume_fingerprint = str(
        args_payload.get("resume_contract_fingerprint", "")
    )
    if actual_resume_fingerprint != str(expected_resume_contract_fingerprint):
        raise ValueError(
            "Bounded batching exact-resume training contract changed; duration, "
            "gradient accumulation, optimizer, scheduler, or learning-rate semantics "
            "differ. Use init_from_checkpoint for a new training schedule."
        )
    actual_spec = ShaftBoundedBatchingSpec.from_dict(spec_payload)
    if actual_spec.fingerprint != expected_spec.fingerprint:
        expected = expected_spec.to_dict()
        actual = actual_spec.to_dict()
        differences = [
            key
            for key, value in expected.items()
            if key != "fingerprint" and actual.get(key) != value
        ]
        raise ValueError(
            "Bounded batching resume contract changed; "
            f"changed fields: {differences}. Start a new run from weights or restore "
            "the original data/topology/buffer/budget settings."
        )
    state = ShaftBoundedBatchingState.from_dict(state_payload)
    if state.contract_fingerprint != expected_spec.fingerprint:
        raise ValueError("Bounded batching state points to a different contract.")
    expected_microstep = int(expected_global_step) * int(gradient_accumulation_steps)
    if int(state.global_microstep) != expected_microstep:
        raise ValueError(
            "Bounded batching checkpoint is not aligned with trainer global_step: "
            f"state_microstep={state.global_microstep}, expected={expected_microstep}."
        )
    return state


class ShaftBoundedBatchingCallback(TrainerCallback, ExportableState):
    """Commit executed state and embed it in HF's atomic Trainer state."""

    def __init__(
        self,
        sampler: ShaftBoundedBatchSampler,
        spec: ShaftBoundedBatchingSpec,
        *,
        gradient_accumulation_steps: int,
        resume_contract_fingerprint: str,
    ) -> None:
        self.sampler = sampler
        self.spec = spec
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)
        self.resume_contract_fingerprint = str(resume_contract_fingerprint)
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be > 0.")
        if not self.resume_contract_fingerprint:
            raise ValueError("resume_contract_fingerprint must not be empty.")

    def on_step_end(self, args, state, control, **kwargs):
        _ = args, kwargs
        self.sampler.commit_global_microstep(
            int(state.global_step) * self.gradient_accumulation_steps
        )
        return control

    def state(self) -> dict[str, Any]:
        return {
            "args": {
                "spec": self.spec.to_dict(),
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "resume_contract_fingerprint": self.resume_contract_fingerprint,
            },
            "attributes": {
                "bounded_state": self.sampler.committed_state.to_dict(),
            },
        }


class ShaftBatchingMetadataCallback(TrainerCallback):
    """Publish resolved Shaft batching metadata into the active W&B run config."""

    def __init__(self, metadata: ShaftBatchingRunMetadata) -> None:
        self.metadata = metadata
        self._wandb_published = False

    @staticmethod
    def _reports_to_wandb(args: Any) -> bool:
        report_to = getattr(args, "report_to", ())
        values = {report_to} if isinstance(report_to, str) else set(report_to or ())
        return "wandb" in {str(value).strip().lower() for value in values}

    def _publish_wandb(self, args: Any, state: Any) -> None:
        if self._wandb_published or not bool(state.is_world_process_zero):
            return
        if not self._reports_to_wandb(args):
            return
        try:
            import wandb
        except ImportError:
            logger.warning(
                "W&B reporting was requested but wandb cannot be imported; "
                "batching metadata remains in %s.",
                BATCHING_RUN_METADATA_FILENAME,
            )
            return
        run = getattr(wandb, "run", None)
        if run is None:
            return
        run.config.update(
            {"shaft_batching": self.metadata.to_dict()},
            allow_val_change=True,
        )
        self._wandb_published = True

    def on_train_begin(self, args, state, control, **kwargs):
        _ = kwargs
        self._publish_wandb(args, state)
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs, kwargs
        self._publish_wandb(args, state)
        return control
