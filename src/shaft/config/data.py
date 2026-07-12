from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PromptSamplingConfig:
    enabled: bool = False
    train_only: bool = True
    seed: int | None = None
    pools: dict[str, str] = field(default_factory=dict)


@dataclass
class DataBatchingConfig:
    strategy: str = "fixed"
    buffer_size: int = 64
    cost_cache_size: int = 65536
    max_samples_per_microbatch: int | None = None
    max_padded_tokens: int | None = None
    max_vision_patches: int | None = None


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
    prompt_sampling: PromptSamplingConfig = field(default_factory=PromptSamplingConfig)
    batching: DataBatchingConfig = field(default_factory=DataBatchingConfig)
    mix_strategy: str = "weighted"
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
    shuffle: bool = True
