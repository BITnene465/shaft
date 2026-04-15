from __future__ import annotations

from dataclasses import dataclass, field


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
    offline_transforms: list[str] = field(default_factory=list)
    online_transforms: list[str] = field(default_factory=list)
    help: str | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class DataConfig:
    catalog_path: str | None = None
    catalog_names: list[str] = field(default_factory=list)
    datasets: list[DatasetSourceConfig] = field(default_factory=list)
    mix_strategy: str = "interleave_under"
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    min_pixels: int | None = 200704
    max_pixels: int | None = 1048576
    add_eos_token: bool = True
    shuffle: bool = True
