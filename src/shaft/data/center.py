from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from shaft.config import DataConfig

from .meta import build_dataset_metas
from .mixing import MixedDatasetBuilder
from .sources import build_data_source
from .transforms import build_offline_pipeline, build_online_pipeline

RecordT = TypeVar("RecordT")
DatasetT = TypeVar("DatasetT")
OnlineSampleTransform = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass
class ShaftPreparedRecords(Generic[RecordT]):
    train_records: list[RecordT]
    val_records: list[RecordT]
    online_transforms: list[OnlineSampleTransform]

    def build_dataset_pair(self, dataset_cls: type[DatasetT]) -> tuple[DatasetT, DatasetT]:
        return (
            dataset_cls(self.train_records, online_transforms=self.online_transforms),
            dataset_cls(self.val_records, online_transforms=self.online_transforms),
        )


class ShaftDataCenter:
    def __init__(self, data_config: DataConfig, *, seed: int = 42) -> None:
        self.data_config = data_config
        self.seed = int(seed)

    def prepare_records(self) -> ShaftPreparedRecords[Any]:
        records_by_dataset_train: dict[str, list[Any]] = {}
        records_by_dataset_val: dict[str, list[Any]] = {}
        weights: dict[str, float] = {}
        dataset_online_pipelines: dict[str, OnlineSampleTransform] = {}

        for dataset_meta in build_dataset_metas(self.data_config):
            if not dataset_meta.enabled:
                continue
            weights[dataset_meta.dataset_name] = float(dataset_meta.weight)
            source_impl = build_data_source(dataset_meta)
            offline_pipeline = build_offline_pipeline(dataset_meta.offline_transforms)
            records_by_dataset_train[dataset_meta.dataset_name] = offline_pipeline(source_impl.load_split("train"))
            records_by_dataset_val[dataset_meta.dataset_name] = offline_pipeline(source_impl.load_split("val"))
            dataset_online_pipelines[dataset_meta.dataset_name] = build_online_pipeline(
                dataset_meta.online_transforms
            )

        mixer = MixedDatasetBuilder(seed=self.seed)
        mixed_indices = mixer.build_indices(
            records_by_dataset_train,
            weights,
            strategy=self.data_config.mix_strategy,
            shuffle=self.data_config.shuffle,
        )
        train_records = [
            records_by_dataset_train[dataset_name][row_index]
            for dataset_name, row_index in mixed_indices
        ]
        val_records: list[Any] = []
        for dataset_name in sorted(records_by_dataset_val):
            val_records.extend(records_by_dataset_val[dataset_name])
        return ShaftPreparedRecords(
            train_records=train_records,
            val_records=val_records,
            online_transforms=[self._build_dataset_aware_online_transform(dataset_online_pipelines)],
        )

    def build_dataset_pair(self, dataset_cls: type[DatasetT]) -> tuple[DatasetT, DatasetT]:
        return self.prepare_records().build_dataset_pair(dataset_cls)

    @staticmethod
    def _build_dataset_aware_online_transform(
        dataset_online_pipelines: dict[str, OnlineSampleTransform],
    ) -> OnlineSampleTransform:
        def _dataset_aware_online_transform(sample: dict[str, Any]) -> dict[str, Any]:
            dataset_name = str(sample.get("dataset_name", "default"))
            pipeline = dataset_online_pipelines.get(dataset_name)
            if pipeline is None:
                return sample
            return pipeline(sample)

        return _dataset_aware_online_transform
