from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal


TaskKind = Literal["detection", "keypoint"]
RunStatus = Literal["created", "queued", "running", "succeeded", "failed", "cancelled"]
PredictionStatus = Literal["predicted", "failed", "skipped"]
SUPPORTED_TASKS: set[str] = {"detection", "keypoint"}


def utc_now_iso() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return value


def _require_non_empty_string(value: str, *, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return normalized


def _validate_tasks(tasks: list[TaskKind], *, field_name: str) -> list[TaskKind]:
    if not isinstance(tasks, list) or not tasks:
        raise ValueError(f"{field_name} must be a non-empty list.")
    normalized: list[TaskKind] = []
    for item in tasks:
        if item not in SUPPORTED_TASKS:
            raise ValueError(f"Unsupported task={item!r}.")
        if item not in normalized:
            normalized.append(item)
    return normalized


def _validate_bbox(bbox: list[float | int]) -> list[float]:
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError("bbox must be a list of four numbers: [x1, y1, x2, y2].")
    try:
        x1, y1, x2, y2 = [float(item) for item in bbox]
    except (TypeError, ValueError) as exc:
        raise ValueError("bbox must contain only numbers.") from exc
    if not (x2 > x1 and y2 > y1):
        raise ValueError("bbox must satisfy x2 > x1 and y2 > y1.")
    return [x1, y1, x2, y2]


def _validate_points(points: list[list[float | int]] | None, *, field_name: str) -> list[list[float]]:
    if points is None:
        return []
    if not isinstance(points, list):
        raise ValueError(f"{field_name} must be a list of [x, y] points.")
    normalized: list[list[float]] = []
    for point in points:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError(f"{field_name} must contain only [x, y] points.")
        try:
            normalized.append([float(point[0]), float(point[1])])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} points must contain only numbers.") from exc
    return normalized


@dataclass(frozen=True)
class ModelRef:
    model_id: str
    path: str
    alias: str | None = None
    checkpoint_kind: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _require_non_empty_string(self.model_id, field_name="model.model_id")
        _require_non_empty_string(self.path, field_name="model.path")


@dataclass(frozen=True)
class BenchmarkRef:
    benchmark_id: str
    root: str
    split: str
    tasks: list[TaskKind]
    manifest_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _require_non_empty_string(self.benchmark_id, field_name="benchmark.benchmark_id")
        _require_non_empty_string(self.root, field_name="benchmark.root")
        _require_non_empty_string(self.split, field_name="benchmark.split")
        _validate_tasks(self.tasks, field_name="benchmark.tasks")


@dataclass(frozen=True)
class BenchmarkManifest:
    benchmark_id: str
    tasks: list[TaskKind]
    root: str
    split: str
    manifest_path: str
    sample_count: int
    created_at: str = field(default_factory=utc_now_iso)
    source_raw_root: str | None = None
    source_manifest_path: str | None = None
    layers: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _require_non_empty_string(self.benchmark_id, field_name="benchmark_id")
        _require_non_empty_string(self.root, field_name="root")
        _require_non_empty_string(self.split, field_name="split")
        _require_non_empty_string(self.manifest_path, field_name="manifest_path")
        _validate_tasks(self.tasks, field_name="tasks")
        if self.sample_count < 0:
            raise ValueError("sample_count must be >= 0.")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class PromptRef:
    prompt_id: str
    path: str | None = None
    text_hash: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _require_non_empty_string(self.prompt_id, field_name="prompt.prompt_id")


@dataclass(frozen=True)
class InferenceParams:
    backend: str = "vllm_openai"
    endpoint: str | None = None
    served_model_name: str | None = None
    service_id: str | None = None
    cuda_visible_devices: str | None = None
    tensor_parallel_size: int | None = None
    port: int | None = None
    max_model_len: int | None = None
    gpu_memory_utilization: float | None = None
    max_num_seqs: int | None = None
    max_tokens: int = 4096
    temperature: float = 0.0
    top_p: float = 1.0
    min_pixels: int | None = None
    max_pixels: int | None = None
    batch_size: int = 1
    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _require_non_empty_string(self.backend, field_name="inference.backend")
        if self.max_tokens <= 0:
            raise ValueError("inference.max_tokens must be > 0.")
        if self.batch_size <= 0:
            raise ValueError("inference.batch_size must be > 0.")
        if self.tensor_parallel_size is not None and self.tensor_parallel_size <= 0:
            raise ValueError("inference.tensor_parallel_size must be > 0 when set.")
        if self.port is not None and self.port <= 0:
            raise ValueError("inference.port must be > 0 when set.")
        if self.max_model_len is not None and self.max_model_len <= 0:
            raise ValueError("inference.max_model_len must be > 0 when set.")
        if self.max_num_seqs is not None and self.max_num_seqs <= 0:
            raise ValueError("inference.max_num_seqs must be > 0 when set.")
        if (
            self.gpu_memory_utilization is not None
            and not (0.0 < self.gpu_memory_utilization <= 1.0)
        ):
            raise ValueError("inference.gpu_memory_utilization must be in (0, 1] when set.")
        if self.min_pixels is not None and self.min_pixels <= 0:
            raise ValueError("inference.min_pixels must be > 0 when set.")
        if self.max_pixels is not None and self.max_pixels <= 0:
            raise ValueError("inference.max_pixels must be > 0 when set.")
        if (
            self.min_pixels is not None
            and self.max_pixels is not None
            and self.max_pixels < self.min_pixels
        ):
            raise ValueError("inference.max_pixels must be >= min_pixels.")


@dataclass(frozen=True)
class EvalSpec:
    spec_id: str
    task: TaskKind
    prompt: PromptRef
    parser: str = "shaft.codec.json_any"
    metric_profile: str = "default"
    visualization_profile: str = "default"
    target_labels: list[str] = field(default_factory=list)
    inference: InferenceParams = field(default_factory=InferenceParams)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _require_non_empty_string(self.spec_id, field_name="spec.spec_id")
        if self.task not in {"detection", "keypoint"}:
            raise ValueError(f"Unsupported spec.task={self.task!r}.")
        self.prompt.validate()
        for label in self.target_labels:
            _require_non_empty_string(label, field_name="spec.target_labels[]")
        self.inference.validate()


@dataclass(frozen=True)
class EvalRunManifest:
    run_id: str
    model: ModelRef
    benchmark: BenchmarkRef
    spec: EvalSpec
    status: RunStatus = "created"
    created_at: str = field(default_factory=utc_now_iso)
    submitter: str = "local"
    shaft_commit: str | None = None
    eval_bench_commit: str | None = None
    artifact_root: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _require_non_empty_string(self.run_id, field_name="run_id")
        if self.status not in {"created", "queued", "running", "succeeded", "failed", "cancelled"}:
            raise ValueError(f"Unsupported run status={self.status!r}.")
        self.model.validate()
        self.benchmark.validate()
        self.spec.validate()
        if self.spec.task not in self.benchmark.tasks:
            raise ValueError("spec.task must be listed in benchmark.tasks.")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class PredictionInstance:
    label: str
    bbox: list[float | int]
    keypoints: list[list[float | int]] | None = None
    linestrip: list[list[float | int]] | None = None
    score: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self, *, task: TaskKind | None = None) -> None:
        if task is not None and task not in {"detection", "keypoint"}:
            raise ValueError(f"Unsupported prediction task={task!r}.")
        _require_non_empty_string(self.label, field_name="instance.label")
        _validate_bbox(self.bbox)
        _validate_points(self.keypoints, field_name="instance.keypoints")
        _validate_points(self.linestrip, field_name="instance.linestrip")
        if self.score is not None and not (0.0 <= float(self.score) <= 1.0):
            raise ValueError("instance.score must be in [0, 1] when set.")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = _jsonable(asdict(self))
        if not payload.get("keypoints"):
            payload.pop("keypoints", None)
        if not payload.get("linestrip"):
            payload.pop("linestrip", None)
        if payload.get("score") is None:
            payload.pop("score", None)
        if not payload.get("extra"):
            payload["extra"] = {}
        return payload


@dataclass(frozen=True)
class PredictionDocument:
    image: str
    instances: list[PredictionInstance]
    metadata: dict[str, Any]
    status: PredictionStatus = "predicted"
    image_id: str | None = None

    def validate(self, *, task: TaskKind | None = None) -> None:
        _require_non_empty_string(self.image, field_name="prediction.image")
        if self.status not in {"predicted", "failed", "skipped"}:
            raise ValueError(f"Unsupported prediction status={self.status!r}.")
        if not isinstance(self.metadata, dict):
            raise ValueError("prediction.metadata must be a dict.")
        for instance in self.instances:
            instance.validate(task=task if self.status == "predicted" else None)

    def to_dict(self, *, task: TaskKind | None = None) -> dict[str, Any]:
        self.validate(task=task)
        return {
            "image": self.image,
            "status": self.status,
            "instances": [instance.to_dict() for instance in self.instances],
            "metadata": _jsonable(self.metadata),
            **({"image_id": self.image_id} if self.image_id else {}),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PredictionDocument":
        if not isinstance(payload, dict):
            raise ValueError("Prediction document must be a JSON object.")
        instance_payloads = payload.get("instances", [])
        if not isinstance(instance_payloads, list):
            raise ValueError("prediction.instances must be a list.")
        instances: list[PredictionInstance] = []
        for index, item in enumerate(instance_payloads):
            if not isinstance(item, dict):
                raise ValueError(f"prediction.instances[{index}] must be an object.")
            instances.append(
                PredictionInstance(
                    label=str(item.get("label", "")),
                    bbox=list(item.get("bbox", [])),
                    keypoints=item.get("keypoints"),
                    linestrip=item.get("linestrip"),
                    score=item.get("score"),
                    extra=dict(item.get("extra") or {}),
                )
            )
        return cls(
            image=str(payload.get("image", "")),
            status=str(payload.get("status", "predicted")),  # type: ignore[arg-type]
            instances=instances,
            image_id=payload.get("image_id"),
            metadata=dict(payload.get("metadata") or {}),
        )
