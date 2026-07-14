from __future__ import annotations

from dataclasses import dataclass, field


SHAFT_BATCH_RESOURCE_NAMES = frozenset({"vision_patches"})


@dataclass
class PromptSamplingConfig:
    enabled: bool = False
    train_only: bool = True
    seed: int | None = None
    pools: dict[str, str] = field(default_factory=dict)


@dataclass
class DataScheduleConfig:
    """Choose the deterministic logical-sample stream before batching."""

    mixing: str = "weighted"
    shuffle: bool = True


@dataclass
class DataTransformsConfig:
    """Resolve per-draw sample views without owning sample order or batching."""

    prompt_sampling: PromptSamplingConfig = field(default_factory=PromptSamplingConfig)


@dataclass
class DataPackingConfig:
    """Configure logical-sequence packing independently from tensor layout."""

    mode: str = "none"


@dataclass
class DataBatchingConfig:
    """Group, size, pack, and tensorize one local training microbatch."""

    grouping: str = "none"
    cardinality: str = "fixed"
    packing: DataPackingConfig = field(default_factory=DataPackingConfig)
    layout: str = "padded"
    buffer_size: int = 64
    cost_cache_size: int = 65536
    max_tokens_per_microbatch: int | None = None
    resource_budgets: dict[str, int] = field(default_factory=dict)


@dataclass
class DatasetSourceConfig:
    dataset_name: str
    source_type: str = "jsonl_sft"
    train_path: str | None = None
    val_path: str | None = None
    train_paths: list[str] = field(default_factory=list)
    val_paths: list[str] = field(default_factory=list)
    weight: float = 1.0
    enabled: bool = True
    use_for_eval: bool = True
    offline_transforms: list[str] = field(default_factory=list)
    online_transforms: list[str] = field(default_factory=list)
    help: str | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class DataConfig:
    catalog_path: str | None = None
    catalog_names: list[str] = field(default_factory=list)
    datasets: list[DatasetSourceConfig] = field(default_factory=list)
    schedule: DataScheduleConfig = field(default_factory=DataScheduleConfig)
    transforms: DataTransformsConfig = field(default_factory=DataTransformsConfig)
    batching: DataBatchingConfig = field(default_factory=DataBatchingConfig)
    num_workers: int = 4
    prefetch_factor: int | None = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    record_cache_dir: str | None = None
    media_snapshot_id: str | None = None
    image_cache_size: int = 0
    min_pixels: int | None = 200704
    max_pixels: int | None = 1048576
    max_length: int | None = None
    add_eos_token: bool = True
