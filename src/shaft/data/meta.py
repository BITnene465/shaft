from __future__ import annotations

from dataclasses import dataclass

from shaft.config import DataConfig, DatasetSourceConfig


@dataclass(frozen=True)
class ShaftDatasetMeta:
    dataset_name: str
    source_type: str
    train_paths: tuple[str, ...]
    val_paths: tuple[str, ...]
    weight: float
    enabled: bool
    use_for_eval: bool
    offline_transforms: tuple[str, ...]
    online_transforms: tuple[str, ...]
    help: str | None = None
    tags: tuple[str, ...] = ()

    @classmethod
    def from_config(cls, config: DatasetSourceConfig) -> "ShaftDatasetMeta":
        train_paths = tuple(config.train_paths) or (() if not config.train_path else (str(config.train_path),))
        val_paths = tuple(config.val_paths) or (() if not config.val_path else (str(config.val_path),))
        return cls(
            dataset_name=config.dataset_name,
            source_type=config.source_type,
            train_paths=train_paths,
            val_paths=val_paths,
            weight=float(config.weight),
            enabled=bool(config.enabled),
            use_for_eval=bool(config.use_for_eval),
            offline_transforms=tuple(config.offline_transforms),
            online_transforms=tuple(config.online_transforms),
            help=config.help,
            tags=tuple(config.tags),
        )


def build_dataset_metas(data_config: DataConfig) -> list[ShaftDatasetMeta]:
    seen: set[str] = set()
    dataset_metas: list[ShaftDatasetMeta] = []
    for source_config in data_config.datasets:
        dataset_meta = ShaftDatasetMeta.from_config(source_config)
        if dataset_meta.dataset_name in seen:
            raise ValueError(f"Duplicate dataset_name {dataset_meta.dataset_name!r} in data.datasets.")
        seen.add(dataset_meta.dataset_name)
        dataset_metas.append(dataset_meta)
    return dataset_metas
