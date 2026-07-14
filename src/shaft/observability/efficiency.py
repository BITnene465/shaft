from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
from importlib import metadata
import json
import math
import platform
from pathlib import Path
from statistics import median
from typing import Any, ClassVar, Sequence

import torch
import transformers


TRAINING_EFFICIENCY_FILENAME = "shaft_training_efficiency.json"
TRAINING_EFFICIENCY_SCHEMA_VERSION = 3


@dataclass(frozen=True, slots=True)
class ShaftTrainingEfficiencyContract:
    """Immutable run identity plus the batching axes used by efficiency A/Bs."""

    algorithm: str
    model_type: str
    model_name_or_path: str
    model_plan_fingerprint: str
    finetune_mode: str
    torch_dtype: str
    attention_implementation: str
    seed: int
    max_steps: int
    num_train_epochs: float
    data_world_size: int
    gradient_accumulation_steps: int
    max_length: int | None
    min_pixels: int | None
    max_pixels: int | None
    optimizer_name: str
    scheduler_name: str
    learning_rate: float
    source_fingerprint: str
    source_contract_complete: bool
    sample_execution_fingerprint: str
    sample_stream_fingerprint: str
    software_fingerprint: str
    hardware_fingerprint: str
    measurement_protocol: str
    timing_mode: str
    batch_contract_fingerprint: str
    sequence_contract_fingerprint: str

    COMPARISON_IDENTITY_FIELDS: ClassVar[tuple[str, ...]] = (
        "algorithm",
        "model_type",
        "model_name_or_path",
        "model_plan_fingerprint",
        "finetune_mode",
        "torch_dtype",
        "attention_implementation",
        "seed",
        "max_steps",
        "num_train_epochs",
        "data_world_size",
        "gradient_accumulation_steps",
        "max_length",
        "min_pixels",
        "max_pixels",
        "optimizer_name",
        "scheduler_name",
        "learning_rate",
        "source_fingerprint",
        "source_contract_complete",
        "sample_stream_fingerprint",
        "software_fingerprint",
        "hardware_fingerprint",
        "measurement_protocol",
        "timing_mode",
    )
    EXPERIMENT_AXIS_FIELDS: ClassVar[tuple[str, ...]] = (
        "batch_contract_fingerprint",
        "sequence_contract_fingerprint",
    )

    def __post_init__(self) -> None:
        required_strings = (
            "algorithm",
            "model_type",
            "model_name_or_path",
            "model_plan_fingerprint",
            "finetune_mode",
            "torch_dtype",
            "attention_implementation",
            "optimizer_name",
            "scheduler_name",
            "source_fingerprint",
            "sample_execution_fingerprint",
            "sample_stream_fingerprint",
            "software_fingerprint",
            "hardware_fingerprint",
            "measurement_protocol",
            "timing_mode",
            "batch_contract_fingerprint",
            "sequence_contract_fingerprint",
        )
        for field_name in required_strings:
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"Training-efficiency contract {field_name} must not be empty.")
        for field_name in (
            "data_world_size",
            "gradient_accumulation_steps",
        ):
            if int(getattr(self, field_name)) <= 0:
                raise ValueError(f"Training-efficiency contract {field_name} must be > 0.")
        for field_name in ("max_length", "min_pixels", "max_pixels"):
            value = getattr(self, field_name)
            if value is not None and int(value) <= 0:
                raise ValueError(f"Training-efficiency contract {field_name} must be > 0 when set.")
        if float(self.learning_rate) <= 0:
            raise ValueError("Training-efficiency contract learning_rate must be > 0.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ShaftTrainingEfficiencyContract":
        return cls(**payload)

    def comparison_identity(self) -> dict[str, Any]:
        payload = self.to_dict()
        return {name: payload[name] for name in self.COMPARISON_IDENTITY_FIELDS}


@dataclass(frozen=True, slots=True)
class ShaftEfficiencyFrame:
    """One successfully committed, rank-local optimizer frame."""

    global_step: int
    logical_segments: int
    physical_packs: int
    useful_tokens: int
    materialized_tokens: int
    supervised_tokens: int
    weighted_supervision_mass: float | None
    weighted_supervision_coverage_microbatches: int
    sequence_length_sum: int
    sequence_length_square_sum: int
    vision_patches: int
    vision_coverage_batches: int
    microbatches: int
    host_batch_acquire_seconds: float
    batch_prepare_seconds: float
    training_step_seconds: float
    optimizer_step_seconds: float
    device_training_seconds: float | None
    update_applied: bool

    def __post_init__(self) -> None:
        count_fields = (
            "logical_segments",
            "physical_packs",
            "useful_tokens",
            "materialized_tokens",
            "supervised_tokens",
            "weighted_supervision_coverage_microbatches",
            "sequence_length_sum",
            "sequence_length_square_sum",
            "vision_patches",
            "vision_coverage_batches",
            "microbatches",
        )
        if int(self.global_step) <= 0:
            raise ValueError("Efficiency frame global_step must be > 0.")
        for name in count_fields:
            if int(getattr(self, name)) < 0:
                raise ValueError(f"Efficiency frame {name} must be >= 0.")
        if self.logical_segments <= 0 or self.physical_packs <= 0 or self.microbatches <= 0:
            raise ValueError("A committed efficiency frame cannot be empty.")
        if self.useful_tokens > self.materialized_tokens:
            raise ValueError("Efficiency useful_tokens cannot exceed materialized_tokens.")
        if self.supervised_tokens > self.useful_tokens:
            raise ValueError("Efficiency supervised_tokens cannot exceed useful_tokens.")
        if self.weighted_supervision_coverage_microbatches > self.microbatches:
            raise ValueError("Efficiency weighted-supervision coverage cannot exceed microbatches.")
        if self.vision_coverage_batches > self.microbatches:
            raise ValueError("Efficiency vision coverage cannot exceed microbatches.")
        if self.weighted_supervision_mass is not None and (
            not math.isfinite(float(self.weighted_supervision_mass))
            or float(self.weighted_supervision_mass) < 0
        ):
            raise ValueError("Efficiency weighted_supervision_mass must be finite and >= 0.")
        for name in (
            "host_batch_acquire_seconds",
            "batch_prepare_seconds",
            "training_step_seconds",
            "optimizer_step_seconds",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Efficiency frame {name} must be finite and >= 0.")
        if self.device_training_seconds is not None and (
            not math.isfinite(float(self.device_training_seconds))
            or float(self.device_training_seconds) < 0
        ):
            raise ValueError("Efficiency frame device_training_seconds must be finite and >= 0.")

    @property
    def critical_path_seconds(self) -> float:
        compute_seconds = (
            float(self.device_training_seconds)
            if self.device_training_seconds is not None
            else float(self.training_step_seconds) + float(self.optimizer_step_seconds)
        )
        return (
            float(self.host_batch_acquire_seconds)
            + float(self.batch_prepare_seconds)
            + compute_seconds
        )


@dataclass(frozen=True, slots=True)
class ShaftEfficiencyAggregate:
    first_step: int
    last_step: int
    optimizer_steps: int
    logical_segments: int
    physical_packs: int
    useful_tokens: int
    materialized_tokens: int
    supervised_tokens: int
    weighted_supervision_mass: float
    weighted_supervision_coverage_microbatches: int
    sequence_length_sum: int
    sequence_length_square_sum: int
    vision_patches: int
    vision_coverage_batches: int
    microbatches: int
    update_applied_steps: int
    host_batch_acquire_seconds: float
    batch_prepare_seconds: float
    training_step_seconds: float
    optimizer_step_seconds: float
    device_training_seconds: float
    device_timing_steps: int
    critical_path_seconds: float
    critical_path_p50_seconds: float
    critical_path_p95_seconds: float

    @classmethod
    def from_frames(
        cls,
        frames: Sequence[ShaftEfficiencyFrame],
    ) -> "ShaftEfficiencyAggregate":
        if not frames:
            raise ValueError("Efficiency aggregation requires at least one committed frame.")
        ordered = tuple(sorted(frames, key=lambda frame: frame.global_step))
        steps = [int(frame.global_step) for frame in ordered]
        expected_steps = list(range(steps[0], steps[-1] + 1))
        if steps != expected_steps:
            raise ValueError("Committed efficiency frames must be a contiguous step span.")
        critical = sorted(float(frame.critical_path_seconds) for frame in ordered)
        return cls(
            first_step=steps[0],
            last_step=steps[-1],
            optimizer_steps=len(ordered),
            logical_segments=sum(frame.logical_segments for frame in ordered),
            physical_packs=sum(frame.physical_packs for frame in ordered),
            useful_tokens=sum(frame.useful_tokens for frame in ordered),
            materialized_tokens=sum(frame.materialized_tokens for frame in ordered),
            supervised_tokens=sum(frame.supervised_tokens for frame in ordered),
            weighted_supervision_mass=sum(
                float(frame.weighted_supervision_mass or 0.0) for frame in ordered
            ),
            weighted_supervision_coverage_microbatches=sum(
                frame.weighted_supervision_coverage_microbatches for frame in ordered
            ),
            sequence_length_sum=sum(frame.sequence_length_sum for frame in ordered),
            sequence_length_square_sum=sum(frame.sequence_length_square_sum for frame in ordered),
            vision_patches=sum(frame.vision_patches for frame in ordered),
            vision_coverage_batches=sum(frame.vision_coverage_batches for frame in ordered),
            microbatches=sum(frame.microbatches for frame in ordered),
            update_applied_steps=sum(bool(frame.update_applied) for frame in ordered),
            host_batch_acquire_seconds=sum(frame.host_batch_acquire_seconds for frame in ordered),
            batch_prepare_seconds=sum(frame.batch_prepare_seconds for frame in ordered),
            training_step_seconds=sum(frame.training_step_seconds for frame in ordered),
            optimizer_step_seconds=sum(frame.optimizer_step_seconds for frame in ordered),
            device_training_seconds=sum(
                float(frame.device_training_seconds or 0.0) for frame in ordered
            ),
            device_timing_steps=sum(frame.device_training_seconds is not None for frame in ordered),
            critical_path_seconds=sum(critical),
            critical_path_p50_seconds=float(median(critical)),
            critical_path_p95_seconds=_quantile(critical, 0.95),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(self.ratios())
        return payload

    def ratios(self) -> dict[str, float]:
        mean_sequence_length = _safe_ratio(
            self.sequence_length_sum,
            self.logical_segments,
        )
        sequence_variance = max(
            _safe_ratio(self.sequence_length_square_sum, self.logical_segments)
            - mean_sequence_length * mean_sequence_length,
            0.0,
        )
        return {
            "padding_fraction": _safe_ratio(
                self.materialized_tokens - self.useful_tokens,
                self.materialized_tokens,
            ),
            "supervision_fraction": _safe_ratio(
                self.supervised_tokens,
                self.useful_tokens,
            ),
            "segments_per_pack": _safe_ratio(
                self.logical_segments,
                self.physical_packs,
            ),
            "useful_tokens_per_second": _safe_ratio(
                self.useful_tokens,
                self.critical_path_seconds,
            ),
            "logical_segments_per_second": _safe_ratio(
                self.logical_segments,
                self.critical_path_seconds,
            ),
            "vision_patches_per_second": _safe_ratio(
                self.vision_patches,
                self.critical_path_seconds,
            ),
            "supervised_tokens_per_second": _safe_ratio(
                self.supervised_tokens,
                self.critical_path_seconds,
            ),
            "data_acquire_fraction": _safe_ratio(
                self.host_batch_acquire_seconds,
                self.critical_path_seconds,
            ),
            "batch_prepare_fraction": _safe_ratio(
                self.batch_prepare_seconds,
                self.critical_path_seconds,
            ),
            "mean_sequence_length": mean_sequence_length,
            "sequence_length_std": math.sqrt(sequence_variance),
            "weighted_supervision_coverage_fraction": _safe_ratio(
                self.weighted_supervision_coverage_microbatches,
                self.microbatches,
            ),
            "vision_coverage_fraction": _safe_ratio(
                self.vision_coverage_batches,
                self.microbatches,
            ),
        }


@dataclass(frozen=True, slots=True)
class ShaftTrainingEfficiencySummary:
    initial_global_step: int
    final_global_step: int
    complete_history: bool
    world_size: int
    aggregate: ShaftEfficiencyAggregate | None
    rank_time_min_seconds: float
    rank_time_mean_seconds: float
    rank_time_max_seconds: float
    contract: ShaftTrainingEfficiencyContract | None = None
    peak_device_memory_allocated_bytes: int | None = None
    peak_device_memory_reserved_bytes: int | None = None

    def __post_init__(self) -> None:
        allocated = self.peak_device_memory_allocated_bytes
        reserved = self.peak_device_memory_reserved_bytes
        if (allocated is None) != (reserved is None):
            raise ValueError("Peak allocated/reserved memory must be both present or null.")
        if allocated is not None:
            assert reserved is not None
            if int(allocated) < 0 or int(reserved) < 0:
                raise ValueError("Peak allocated/reserved memory must be non-negative.")

    def to_dict(self) -> dict[str, Any]:
        mean = float(self.rank_time_mean_seconds)
        rank_skew = _safe_ratio(
            self.rank_time_max_seconds - self.rank_time_min_seconds,
            mean,
        )
        return {
            "schema_version": TRAINING_EFFICIENCY_SCHEMA_VERSION,
            "initial_global_step": int(self.initial_global_step),
            "final_global_step": int(self.final_global_step),
            "complete_history": bool(self.complete_history),
            "world_size": int(self.world_size),
            "rank_time_min_seconds": float(self.rank_time_min_seconds),
            "rank_time_mean_seconds": mean,
            "rank_time_max_seconds": float(self.rank_time_max_seconds),
            "rank_time_skew": rank_skew,
            "peak_device_memory_allocated_bytes": self.peak_device_memory_allocated_bytes,
            "peak_device_memory_reserved_bytes": self.peak_device_memory_reserved_bytes,
            "contract": None if self.contract is None else self.contract.to_dict(),
            "aggregate": None if self.aggregate is None else self.aggregate.to_dict(),
        }

    def log_metrics(self, *, prefix: str = "efficiency") -> dict[str, float]:
        if self.aggregate is None:
            return {}
        aggregate = self.aggregate
        ratios = aggregate.ratios()
        metrics = {
            f"{prefix}/useful_tokens_per_second": ratios["useful_tokens_per_second"],
            f"{prefix}/logical_segments_per_second": ratios["logical_segments_per_second"],
            f"{prefix}/vision_patches_per_second": ratios["vision_patches_per_second"],
            f"{prefix}/supervised_tokens_per_second": ratios["supervised_tokens_per_second"],
            f"{prefix}/padding_fraction": ratios["padding_fraction"],
            f"{prefix}/supervision_fraction": ratios["supervision_fraction"],
            f"{prefix}/segments_per_pack": ratios["segments_per_pack"],
            f"{prefix}/data_acquire_fraction": ratios["data_acquire_fraction"],
            f"{prefix}/batch_prepare_fraction": ratios["batch_prepare_fraction"],
            f"{prefix}/mean_sequence_length": ratios["mean_sequence_length"],
            f"{prefix}/sequence_length_std": ratios["sequence_length_std"],
            f"{prefix}/critical_path_p50_seconds": float(aggregate.critical_path_p50_seconds),
            f"{prefix}/critical_path_p95_seconds": float(aggregate.critical_path_p95_seconds),
            f"{prefix}/rank_time_skew": _safe_ratio(
                self.rank_time_max_seconds - self.rank_time_min_seconds,
                self.rank_time_mean_seconds,
            ),
            f"{prefix}/logical_segments": float(aggregate.logical_segments),
            f"{prefix}/physical_packs": float(aggregate.physical_packs),
            f"{prefix}/useful_tokens": float(aggregate.useful_tokens),
            f"{prefix}/vision_patches": float(aggregate.vision_patches),
            f"{prefix}/device_training_seconds": float(aggregate.device_training_seconds),
        }
        if self.peak_device_memory_allocated_bytes is not None:
            metrics[f"{prefix}/peak_device_memory_allocated_gib"] = _bytes_to_gib(
                self.peak_device_memory_allocated_bytes
            )
        if self.peak_device_memory_reserved_bytes is not None:
            metrics[f"{prefix}/peak_device_memory_reserved_gib"] = _bytes_to_gib(
                self.peak_device_memory_reserved_bytes
            )
        return metrics


def write_training_efficiency_summary(
    output_dir: str | Path,
    summary: ShaftTrainingEfficiencySummary,
) -> Path:
    path = Path(output_dir) / TRAINING_EFFICIENCY_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(summary.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def training_software_fingerprint() -> str:
    payload = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "accelerate": _package_version("accelerate"),
        "peft": _package_version("peft"),
        "trl": _package_version("trl"),
        "flash_attn": _package_version("flash-attn"),
        "flash_linear_attention": _package_version("flash-linear-attention"),
        "causal_conv1d": _package_version("causal-conv1d"),
        "cuda_runtime": torch.version.cuda,
        "cuda_driver": _cuda_driver_version(),
        "nccl": _nccl_version(),
        "shaft_source": _shaft_source_fingerprint(),
    }
    return _fingerprint_payload(payload)


def training_hardware_fingerprint(device: torch.device) -> str:
    resolved = torch.device(device)
    if resolved.type == "cuda" and torch.cuda.is_available():
        index = torch.cuda.current_device() if resolved.index is None else int(resolved.index)
        properties = torch.cuda.get_device_properties(index)
        payload = {
            "type": "cuda",
            "name": properties.name,
            "capability": [properties.major, properties.minor],
            "total_memory": properties.total_memory,
        }
    else:
        payload = {
            "type": resolved.type,
            "machine": platform.machine(),
            "processor": platform.processor(),
        }
    return _fingerprint_payload(payload)


def _fingerprint_payload(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _cuda_driver_version() -> int | None:
    getter = getattr(torch._C, "_cuda_getDriverVersion", None)
    if not callable(getter):
        return None
    try:
        return int(getter())
    except RuntimeError:
        return None


def _nccl_version() -> tuple[int, ...] | None:
    if not torch.cuda.is_available() or not hasattr(torch.cuda, "nccl"):
        return None
    try:
        value = torch.cuda.nccl.version()
    except (AttributeError, RuntimeError):
        return None
    if isinstance(value, tuple):
        return tuple(int(item) for item in value)
    return (int(value),)


@lru_cache(maxsize=1)
def _shaft_source_fingerprint() -> str:
    package_root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(package_root.rglob("*.py")):
        digest.update(path.relative_to(package_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _safe_ratio(numerator: float | int, denominator: float | int) -> float:
    denominator_value = float(denominator)
    if not math.isfinite(denominator_value) or denominator_value <= 0:
        return 0.0
    return float(numerator) / denominator_value


def _bytes_to_gib(value: float | int) -> float:
    return float(value) / float(1024**3)


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    index = (len(values) - 1) * float(probability)
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return float(values[lower])
    weight = index - lower
    return float(values[lower]) * (1.0 - weight) + float(values[upper]) * weight
