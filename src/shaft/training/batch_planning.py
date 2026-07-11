from __future__ import annotations

import json
import os
from pathlib import Path
import uuid

from transformers import TrainerCallback
from transformers.trainer_utils import get_last_checkpoint

from shaft.data.batching import (
    ShaftBatchPlanningSignature,
    ShaftFixedBatchPlanningSpec,
)
from shaft.data.dynamic_batching import ShaftDynamicBatchPlanningSpec


BATCH_PLANNING_SIGNATURE_FILENAME = "shaft_batch_planning_signature.json"


def batch_planning_signature_path(path: str | Path) -> Path:
    return Path(path) / BATCH_PLANNING_SIGNATURE_FILENAME


def write_batch_planning_signature(
    path: str | Path,
    signature: ShaftBatchPlanningSignature,
) -> Path:
    target = batch_planning_signature_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    temp_path.write_text(
        json.dumps(signature.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temp_path, target)
    return target


def load_batch_planning_signature(path: str | Path) -> ShaftBatchPlanningSignature:
    target = batch_planning_signature_path(path)
    if not target.is_file():
        raise FileNotFoundError(f"Missing batch planning signature: {target}")
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Batch planning signature must be a JSON object: {target}")
    return ShaftBatchPlanningSignature.from_dict(payload)


def validate_batch_planning_resume(
    checkpoint_path: str | Path,
    *,
    expected: ShaftBatchPlanningSignature,
) -> None:
    actual = _load_resume_batch_planning_signature(checkpoint_path)
    if actual.fingerprint == expected.fingerprint:
        return

    expected_payload = expected.to_dict()
    actual_payload = actual.to_dict()
    differences = [
        field_name
        for field_name in expected_payload
        if field_name != "fingerprint"
        and expected_payload[field_name] != actual_payload.get(field_name)
    ]
    raise ValueError(
        "Cost-aware resume requires an identical batch planning signature; "
        f"changed fields: {differences}. Resume with the original data, duration, "
        "batch/topology and processor/template settings, or start a new run from weights."
    )


def validate_batch_planning_resume_geometry(
    checkpoint_path: str | Path,
    *,
    expected: ShaftFixedBatchPlanningSpec | ShaftDynamicBatchPlanningSpec,
) -> None:
    actual = _load_resume_batch_planning_signature(checkpoint_path)
    expected_signature = ShaftBatchPlanningSignature.from_spec(
        expected,
        cost_fingerprint=actual.cost_fingerprint,
    )
    expected_payload = expected_signature.to_dict()
    actual_payload = actual.to_dict()
    differences = [
        field_name
        for field_name, expected_value in expected_payload.items()
        if field_name not in {"cost_fingerprint", "fingerprint"}
        and actual_payload.get(field_name) != expected_value
    ]
    if differences:
        raise ValueError(
            "Cost-aware resume planning geometry changed before model loading; "
            f"changed fields: {differences}. Use the original duration/batch/topology/window "
            "or start a new run from weights."
        )


def _load_resume_batch_planning_signature(
    checkpoint_path: str | Path,
) -> ShaftBatchPlanningSignature:
    checkpoint = Path(checkpoint_path)
    if checkpoint.is_dir() and not checkpoint.name.startswith("checkpoint-"):
        last_checkpoint = get_last_checkpoint(str(checkpoint))
        if last_checkpoint is not None:
            checkpoint = Path(last_checkpoint)
    # An exact resume signature belongs to the concrete checkpoint. Never borrow
    # the run-root signature: the root may already describe a newer/reused plan.
    return load_batch_planning_signature(checkpoint)


class ShaftBatchPlanningCallback(TrainerCallback):
    def __init__(self, signature: ShaftBatchPlanningSignature) -> None:
        self.signature = signature

    def on_save(self, args, state, control, **kwargs):
        _ = kwargs
        if state.is_world_process_zero:
            checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            write_batch_planning_signature(checkpoint_dir, self.signature)
        return control
