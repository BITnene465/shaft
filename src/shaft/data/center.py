from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import hashlib
import inspect
from typing import Any, Generic, TypeVar

from torch.utils.data import Sampler

from shaft.config import DataConfig

from .mixing import ShaftSamplePlan, ShaftSampleRef, ShaftSampleSchedule
from .record_store import ShaftConcatRecordStore
from .sampler import ShaftSampleSampler
from .meta import build_dataset_metas
from .sources import build_data_source
from .transforms import (
    build_offline_pipeline,
    build_online_pipeline,
    build_prompt_sampling_transform,
    is_planning_safe_online_transform,
    planning_online_transform_fingerprint,
    planning_safe_online_transform,
)

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
    train_sampler: ShaftSampleSampler | None
    train_schedule: ShaftSampleSchedule | None
    media_snapshot_id: str | None = None
    image_cache_size: int = 0
    suppress_train_decompression_bomb_warning: bool = False

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
                sample_plan=(
                    None if self.train_sampler is None else self.train_sampler.plan
                ),
                sample_schedule=self.train_schedule,
                media_snapshot_id=self.media_snapshot_id,
                image_cache_size=self.image_cache_size,
                suppress_decompression_bomb_warning=(
                    self.suppress_train_decompression_bomb_warning
                ),
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
            train_schedule=self.train_schedule,
        )


@dataclass
class ShaftDatasetBundle(Generic[DatasetT]):
    train_dataset: DatasetT
    eval_dataset: DatasetT
    eval_datasets_by_name: dict[str, DatasetT] | None = None
    train_sampler: Sampler[ShaftSampleRef] | None = None
    train_schedule: ShaftSampleSchedule | None = None


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

        source_sizes = {
            dataset_name: len(records)
            for dataset_name, records in records_by_dataset_train.items()
        }
        schedule_config = self.data_config.schedule
        train_schedule = None
        if schedule_config.mixing != "weighted" or schedule_config.shuffle:
            train_schedule = ShaftSampleSchedule(
                source_sizes,
                weights,
                strategy=schedule_config.mixing,
                shuffle=schedule_config.shuffle,
                seed=self.seed,
            )
        train_sampler: ShaftSampleSampler | None = None
        planned_grouping = self.data_config.batching.grouping in {
            "length",
            "bounded_cost",
        }
        if not planned_grouping:
            sample_plan = ShaftSamplePlan(
                source_sizes,
                weights,
                strategy=schedule_config.mixing,
                num_samples=self.train_sample_budget,
                shuffle=schedule_config.shuffle,
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
        train_dataset_aware_transform = self._build_dataset_aware_online_transform(
            {
                dataset_name: dataset_online_pipelines[dataset_name]
                for dataset_name in records_by_dataset_train
            }
        )
        eval_dataset_aware_transform = self._build_dataset_aware_online_transform(
            {
                dataset_name: dataset_online_pipelines[dataset_name]
                for dataset_name in records_by_dataset_val
            }
        )
        train_online_transforms = [train_dataset_aware_transform]
        eval_online_transforms = [eval_dataset_aware_transform]
        prompt_sampling = self.data_config.transforms.prompt_sampling
        prompt_sampling_transform = build_prompt_sampling_transform(
            prompt_sampling,
            default_seed=self.seed,
        )
        if prompt_sampling_transform is not None:
            train_online_transforms.append(prompt_sampling_transform)
            if not prompt_sampling.train_only:
                eval_online_transforms.append(prompt_sampling_transform)
        return ShaftPreparedRecords(
            train_records=records_by_dataset_train,
            val_records=val_records,
            val_records_by_dataset=records_by_dataset_val,
            train_online_transforms=train_online_transforms,
            eval_online_transforms=eval_online_transforms,
            train_sampler=train_sampler,
            train_schedule=train_schedule,
            media_snapshot_id=self.data_config.media_snapshot_id,
            image_cache_size=self.data_config.image_cache_size,
            suppress_train_decompression_bomb_warning=(
                planned_grouping
            ),
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

        if all(
            is_planning_safe_online_transform(pipeline)
            for pipeline in dataset_online_pipelines.values()
        ):
            fingerprint_payload = (
                "shaft-dataset-online-pipeline-v1",
                tuple(
                    (
                        dataset_name,
                        planning_online_transform_fingerprint(pipeline),
                    )
                    for dataset_name, pipeline in sorted(dataset_online_pipelines.items())
                ),
            )
            planning_safe_online_transform(
                _dataset_aware_online_transform,
                fingerprint=hashlib.sha256(
                    repr(fingerprint_payload).encode("utf-8")
                ).hexdigest(),
            )
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
