from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import inspect
from typing import Any, Generic, TypeVar

from shaft.config import DataConfig

from .sampler import ShaftMixedIndexSampler
from .meta import build_dataset_metas
from .sources import build_data_source
from .transforms import build_offline_pipeline, build_online_pipeline, build_prompt_sampling_transform

RecordT = TypeVar("RecordT")
DatasetT = TypeVar("DatasetT")
OnlineSampleTransform = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass
class ShaftPreparedRecords(Generic[RecordT]):
    train_records: Any
    val_records: list[RecordT]
    val_records_by_dataset: dict[str, list[RecordT]]
    online_transforms: list[OnlineSampleTransform]
    train_online_transforms: list[OnlineSampleTransform] | None = None
    eval_online_transforms: list[OnlineSampleTransform] | None = None
    train_sampler: ShaftMixedIndexSampler | None = None

    def build_dataset_pair(self, dataset_cls: type[DatasetT]) -> tuple[DatasetT, DatasetT]:
        bundle = self.build_dataset_bundle(dataset_cls)
        return bundle.train_dataset, bundle.eval_dataset

    def build_dataset_bundle(self, dataset_cls: type[DatasetT]) -> ShaftDatasetBundle[DatasetT]:
        train_length = len(self.train_sampler) if self.train_sampler is not None else None
        train_indices = list(self.train_sampler.current_indices) if self.train_sampler is not None else None
        train_online_transforms = self.train_online_transforms or self.online_transforms
        eval_online_transforms = self.eval_online_transforms or self.online_transforms
        eval_datasets_by_name = {
            dataset_name: _build_dataset(
                dataset_cls,
                records,
                online_transforms=eval_online_transforms,
                split="val",
            )
            for dataset_name, records in sorted(self.val_records_by_dataset.items())
        }
        return ShaftDatasetBundle(
            train_dataset=_build_dataset(
                dataset_cls,
                self.train_records,
                online_transforms=train_online_transforms,
                split="train",
                mixed_length=train_length,
                mixed_indices=train_indices,
                train_sampler=self.train_sampler,
            ),
            eval_dataset=_build_dataset(
                dataset_cls,
                self.val_records,
                online_transforms=eval_online_transforms,
                split="val",
            ),
            eval_datasets_by_name=eval_datasets_by_name,
            train_sampler=self.train_sampler,
        )


@dataclass
class ShaftDatasetBundle(Generic[DatasetT]):
    train_dataset: DatasetT
    eval_dataset: DatasetT
    eval_datasets_by_name: dict[str, DatasetT] | None = None
    train_sampler: ShaftMixedIndexSampler | None = None


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
            if dataset_meta.use_for_eval:
                records_by_dataset_val[dataset_meta.dataset_name] = offline_pipeline(source_impl.load_split("val"))
            dataset_online_pipelines[dataset_meta.dataset_name] = build_online_pipeline(
                dataset_meta.online_transforms
            )

        train_sampler = ShaftMixedIndexSampler(
            records_by_dataset_train,
            weights,
            strategy=self.data_config.mix_strategy,
            refresh_mode=self.data_config.mix_refresh,
            shuffle=self.data_config.shuffle,
            seed=self.seed,
            rank=0,
            world_size=1,
        )
        val_records: list[Any] = []
        for dataset_name in sorted(records_by_dataset_val):
            val_records.extend(records_by_dataset_val[dataset_name])
        dataset_aware_transform = self._build_dataset_aware_online_transform(dataset_online_pipelines)
        train_online_transforms = [dataset_aware_transform]
        eval_online_transforms = [dataset_aware_transform]
        prompt_sampling_transform = build_prompt_sampling_transform(
            self.data_config.prompt_sampling,
            default_seed=self.seed,
        )
        if prompt_sampling_transform is not None:
            train_online_transforms.append(prompt_sampling_transform)
            if not self.data_config.prompt_sampling.train_only:
                eval_online_transforms.append(prompt_sampling_transform)
        return ShaftPreparedRecords(
            train_records=records_by_dataset_train,
            val_records=val_records,
            val_records_by_dataset=records_by_dataset_val,
            online_transforms=train_online_transforms,
            train_online_transforms=train_online_transforms,
            eval_online_transforms=eval_online_transforms,
            train_sampler=train_sampler,
        )

    def build_dataset_pair(self, dataset_cls: type[DatasetT]) -> tuple[DatasetT, DatasetT]:
        return self.prepare_records().build_dataset_pair(dataset_cls)

    def build_dataset_bundle(self, dataset_cls: type[DatasetT]) -> ShaftDatasetBundle[DatasetT]:
        return self.prepare_records().build_dataset_bundle(dataset_cls)

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


def _supports_kwarg(callable_obj: Any, keyword: str) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return True
    if keyword in signature.parameters:
        return True
    return any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())


def _build_dataset(
    dataset_cls: type[DatasetT],
    records: Any,
    **kwargs: Any,
) -> DatasetT:
    filtered_kwargs = dict(kwargs)
    if not _supports_kwarg(dataset_cls, "split"):
        filtered_kwargs.pop("split", None)
    return dataset_cls(records, **filtered_kwargs)
