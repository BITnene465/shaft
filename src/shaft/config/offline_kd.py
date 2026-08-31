from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any


_MODES = {"dense_logits", "topk_tail"}
_DIVERGENCES = {"forward_kl", "reverse_kl", "jsd"}


@dataclass
class OfflineKDObjectiveConfig:
    mode: str = "dense_logits"
    divergence: str = "forward_kl"
    temperature: float = 1.0
    top_k: int | None = None
    token_chunk_size: int | None = None


@dataclass
class OfflineKDLossConfig:
    ce_weight: float = 0.5
    kd_weight: float = 0.5


@dataclass
class OfflineKDConfig:
    artifact_manifest: str = ""
    objective: OfflineKDObjectiveConfig = field(default_factory=OfflineKDObjectiveConfig)
    loss: OfflineKDLossConfig = field(default_factory=OfflineKDLossConfig)


def normalize_offline_kd_runtime_config(config: Any) -> None:
    offline_kd = config.offline_kd
    offline_kd.artifact_manifest = str(offline_kd.artifact_manifest).strip()
    if not offline_kd.artifact_manifest:
        raise ValueError("offline_kd.artifact_manifest must not be empty.")
    if not Path(offline_kd.artifact_manifest).is_absolute():
        raise ValueError("offline_kd.artifact_manifest must resolve to an absolute path.")

    objective = offline_kd.objective
    objective.mode = str(objective.mode).strip().lower()
    if objective.mode not in _MODES:
        raise ValueError(f"offline_kd.objective.mode must be one of {sorted(_MODES)}.")
    objective.divergence = str(objective.divergence).strip().lower()
    if objective.divergence not in _DIVERGENCES:
        raise ValueError(
            "offline_kd.objective.divergence must be one of "
            f"{sorted(_DIVERGENCES)}."
        )
    objective.temperature = float(objective.temperature)
    if not math.isfinite(objective.temperature) or objective.temperature <= 0:
        raise ValueError("offline_kd.objective.temperature must be finite and > 0.")
    if objective.top_k is not None:
        objective.top_k = int(objective.top_k)
        if objective.top_k <= 0:
            raise ValueError("offline_kd.objective.top_k must be > 0 when configured.")
    if objective.mode == "topk_tail" and objective.top_k is None:
        raise ValueError("offline_kd topk_tail mode requires objective.top_k.")
    if objective.mode == "dense_logits" and objective.top_k is not None:
        raise ValueError("offline_kd dense_logits mode does not accept objective.top_k.")
    if objective.token_chunk_size is not None:
        objective.token_chunk_size = int(objective.token_chunk_size)
        if objective.token_chunk_size <= 0:
            raise ValueError("offline_kd.objective.token_chunk_size must be > 0.")

    loss = offline_kd.loss
    for field_name in ("ce_weight", "kd_weight"):
        value = float(getattr(loss, field_name))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"offline_kd.loss.{field_name} must be finite and >= 0.")
        setattr(loss, field_name, value)
    if loss.ce_weight == 0 and loss.kd_weight == 0:
        raise ValueError("offline_kd requires a non-zero CE or KD loss weight.")
    if config.data.batching.packing.mode != "none" or config.data.batching.layout != "padded":
        raise ValueError("offline_kd currently requires packing.mode='none' and layout='padded'.")
    if config.eval.enabled:
        raise ValueError("offline_kd evaluation is not implemented; set eval.enabled=false.")
    transformed = [
        dataset.dataset_name
        for dataset in config.data.datasets
        if dataset.offline_transforms or dataset.online_transforms
    ]
    if transformed:
        raise ValueError(
            "offline_kd requires fully materialized immutable inputs and does not accept "
            f"offline/online transforms; datasets={transformed}."
        )
    if config.train.loss_name != "auto" or config.train.loss_scale != "default":
        raise ValueError(
            "offline_kd owns CE+distribution loss; train.loss_name must be 'auto' and "
            "train.loss_scale must be 'default'."
        )
