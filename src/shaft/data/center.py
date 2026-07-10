from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import inspect
from typing import Any, Generic, TypeVar

from shaft.config import DataConfig

from .mixing import ShaftSamplePlan
from .record_store import ShaftConcatRecordStore
from .sampler import ShaftSampleSampler
from .meta import build_dataset_metas
from .sources import build_data_source
from .transforms import build_offline_pipeline, build_online_pipeline, build_prompt_sampling_transform

RecordT = TypeVar("RecordT")
DatasetT = TypeVar("DatasetT")
OnlineSampleTransform = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass
class ShaftPreparedRecords(Generic[RecordT]):
    train_records: dict[str, Sequence[RecordT]]
    val_records: Sequence[RecordT]
    val_records_by_dataset: dict[str, Sequence[RecordT]]
    train_online_transforms: list[OnlineSampleTransform]
    eval_online_transforms: list[OnlineSampleTransform]
    train_sampler: ShaftSampleSampler
    image_cache_size: int = 0

    def build_dataset_bundle(self, dataset_cls: type[DatasetT]) -> ShaftDatasetBundle[DatasetT]:
        eval_datasets_by_name = {
            dataset_name: _build_dataset(
                dataset_cls,
                records,
                online_transforms=self.eval_online_transforms,
                split="val",
                image_cache_size=self.image_cache_size,
            )
            for dataset_name, records in sorted(self.val_records_by_dataset.items())
        }
        return ShaftDatasetBundle(
            train_dataset=_build_dataset(
                dataset_cls,
                self.train_records,
                online_transforms=self.train_online_transforms,
                split="train",
                sample_plan=self.train_sampler.plan,
                image_cache_size=self.image_cache_size,
            ),
            eval_dataset=_build_dataset(
                dataset_cls,
                self.val_records,
                online_transforms=self.eval_online_transforms,
                split="val",
                image_cache_size=self.image_cache_size,
            ),
            eval_datasets_by_name=eval_datasets_by_name,
            train_sampler=self.train_sampler,
        )


@dataclass
class ShaftDatasetBundle(Generic[DatasetT]):
    train_dataset: DatasetT
    eval_dataset: DatasetT
    eval_datasets_by_name: dict[str, DatasetT] | None = None
    train_sampler: ShaftSampleSampler | None = None


class ShaftDataCenter:
    def __init__(
        self,
        data_config: DataConfig,
        *,
        seed: int = 42,
        train_sample_budget: int | None = None,
    ) -> None:
        self.data_config = data_config
        self.seed = int(seed)
        self.train_sample_budget = (
            int(train_sample_budget) if train_sample_budget is not None else None
        )

    def prepare_records(self) -> ShaftPreparedRecords[Any]:
        records_by_dataset_train: dict[str, Sequence[Any]] = {}
        records_by_dataset_val: dict[str, Sequence[Any]] = {}
        weights: dict[str, float] = {}
        dataset_online_pipelines: dict[str, OnlineSampleTransform] = {}

        for dataset_meta in build_dataset_metas(self.data_config):
            if not dataset_meta.enabled:
                continue
            source_impl = build_data_source(
                dataset_meta,
                cache_dir=self.data_config.record_cache_dir,
            )
            offline_pipeline = build_offline_pipeline(dataset_meta.offline_transforms)
            if float(dataset_meta.weight) > 0:
                weights[dataset_meta.dataset_name] = float(dataset_meta.weight)
                records_by_dataset_train[dataset_meta.dataset_name] = offline_pipeline(
                    source_impl.load_split("train")
                )
            if dataset_meta.use_for_eval:
                records_by_dataset_val[dataset_meta.dataset_name] = offline_pipeline(source_impl.load_split("val"))
            dataset_online_pipelines[dataset_meta.dataset_name] = build_online_pipeline(
                dataset_meta.online_transforms
            )

        sample_plan = ShaftSamplePlan(
            {
                dataset_name: len(records)
                for dataset_name, records in records_by_dataset_train.items()
            },
            weights,
            strategy=self.data_config.mix_strategy,
            num_samples=self.train_sample_budget,
            shuffle=self.data_config.shuffle,
            seed=self.seed,
        )
        train_sampler = ShaftSampleSampler(
            sample_plan,
            rank=0,
            world_size=1,
        )
        val_records = ShaftConcatRecordStore(
            [records_by_dataset_val[name] for name in sorted(records_by_dataset_val)]
        )
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
            train_online_transforms=train_online_transforms,
            eval_online_transforms=eval_online_transforms,
            train_sampler=train_sampler,
            image_cache_size=self.data_config.image_cache_size,
        )

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
    filtered_kwargs = {
        key: value
        for key, value in kwargs.items()
        if _supports_kwarg(dataset_cls, key)
    }
    return dataset_cls(records, **filtered_kwargs)
