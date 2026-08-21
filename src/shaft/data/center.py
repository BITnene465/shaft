from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import inspect
import logging
import time
from typing import Any, Generic, TypeVar

from torch.utils.data import Sampler

from shaft.config import DataConfig
from shaft.prompting import canonical_json
from shaft.utils.distributed import (
    get_local_rank,
    get_local_world_size,
    get_world_size,
    is_rank_zero,
)

from .mixing import ShaftSamplePlan, ShaftSampleRef, ShaftSampleSchedule
from .prompt_source import ShaftPromptSource, build_prompt_source_resolver
from .record_store import ShaftConcatRecordStore
from .sampler import ShaftSampleSampler
from .meta import ShaftDatasetMeta, build_dataset_metas
from .sources import (
    BaseDataSource,
    ShaftRecordCacheTask,
    ShaftRecordCacheTaskResult,
    build_data_source,
)
from .transforms import (
    build_offline_pipeline,
    build_online_pipeline,
    is_planning_safe_online_transform,
    planning_online_transform_fingerprint,
    planning_safe_online_transform,
)

RecordT = TypeVar("RecordT")
DatasetT = TypeVar("DatasetT")
OnlineSampleTransform = Callable[[dict[str, Any]], dict[str, Any]]
logger = logging.getLogger(__name__)
_RECORD_CACHE_PLAN_VERSION = "shaft-record-cache-plan-v1"


@dataclass(frozen=True, slots=True)
class ShaftRecordCachePlan:
    tasks: tuple[ShaftRecordCacheTask, ...]
    shards: tuple[tuple[ShaftRecordCacheTask, ...], ...]
    fingerprint: str

    @classmethod
    def build(
        cls,
        tasks: Sequence[ShaftRecordCacheTask],
        *,
        shard_count: int,
    ) -> ShaftRecordCachePlan:
        resolved_shard_count = int(shard_count)
        if resolved_shard_count <= 0:
            raise ValueError("record cache shard_count must be > 0.")
        unique_tasks: dict[str, ShaftRecordCacheTask] = {}
        for task in tasks:
            existing = unique_tasks.get(task.fingerprint)
            if existing is not None:
                if (
                    existing.source_path != task.source_path
                    or existing.cache_path != task.cache_path
                    or existing.dataset_name != task.dataset_name
                    or existing.split != task.split
                ):
                    raise ValueError(
                        "Record cache fingerprint collision across distinct tasks: "
                        f"{task.fingerprint}."
                    )
                continue
            unique_tasks[task.fingerprint] = task

        ordered_tasks = sorted(
            unique_tasks.values(),
            key=lambda task: (-task.source_size_bytes, task.fingerprint),
        )
        shard_tasks: list[list[ShaftRecordCacheTask]] = [
            [] for _ in range(resolved_shard_count)
        ]
        shard_bytes = [0] * resolved_shard_count
        for task in ordered_tasks:
            shard_index = min(
                range(resolved_shard_count),
                key=lambda index: (
                    shard_bytes[index],
                    len(shard_tasks[index]),
                    index,
                ),
            )
            shard_tasks[shard_index].append(task)
            shard_bytes[shard_index] += task.source_size_bytes
        shards = tuple(tuple(shard) for shard in shard_tasks)
        payload = {
            "version": _RECORD_CACHE_PLAN_VERSION,
            "shards": [
                [task.fingerprint for task in shard]
                for shard in shards
            ],
        }
        fingerprint = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
        return cls(
            tasks=tuple(sorted(unique_tasks.values(), key=lambda task: task.fingerprint)),
            shards=shards,
            fingerprint=fingerprint,
        )

    @property
    def shard_count(self) -> int:
        return len(self.shards)

    def tasks_for_shard(self, shard_index: int) -> tuple[ShaftRecordCacheTask, ...]:
        resolved_index = int(shard_index)
        if resolved_index < 0 or resolved_index >= self.shard_count:
            raise ValueError(
                "record cache shard_index must satisfy "
                f"0 <= shard_index < {self.shard_count}, got {resolved_index}."
            )
        return self.shards[resolved_index]


@dataclass(frozen=True, slots=True)
class ShaftRecordCacheWarmupSummary:
    plan_fingerprint: str
    shard_index: int
    shard_count: int
    task_count: int
    source_size_bytes: int
    row_count: int
    duration_seconds: float


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
        record_fingerprints = tuple(
            (
                dataset_name,
                _record_sequence_fingerprint(records),
            )
            for dataset_name, records in sorted(self.train_records.items())
        )
        train_stream_fingerprint = _train_input_fingerprint(
            sample_fingerprint=(
                str(self.train_schedule.fingerprint)
                if self.train_schedule is not None
                else str(self.train_sampler.plan.stream_fingerprint)
            ),
            transforms=self.train_online_transforms,
            record_fingerprints=record_fingerprints,
            media_snapshot_id=self.media_snapshot_id,
        )
        train_execution_incomplete_reasons = tuple(
            [
                "unversioned_online_transform:"
                f"{getattr(transform, '__module__', type(transform).__module__)}."
                f"{getattr(transform, '__qualname__', type(transform).__qualname__)}"
                for transform in self.train_online_transforms
                if not is_planning_safe_online_transform(transform)
            ]
            + (
                []
                if str(self.media_snapshot_id or "").strip()
                else ["missing_media_snapshot_id"]
            )
        )
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
                sample_plan=(None if self.train_sampler is None else self.train_sampler.plan),
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
            train_execution_fingerprint=_train_input_fingerprint(
                sample_fingerprint=(
                    str(self.train_sampler.plan.fingerprint)
                    if self.train_sampler is not None
                    else str(self.train_schedule.fingerprint)
                ),
                transforms=self.train_online_transforms,
                record_fingerprints=record_fingerprints,
                media_snapshot_id=self.media_snapshot_id,
            ),
            train_execution_contract_complete=(
                not train_execution_incomplete_reasons
            ),
            train_execution_incomplete_reasons=(
                train_execution_incomplete_reasons
            ),
            train_stream_fingerprint=train_stream_fingerprint,
        )


@dataclass
class ShaftDatasetBundle(Generic[DatasetT]):
    train_dataset: DatasetT
    eval_dataset: DatasetT
    eval_datasets_by_name: dict[str, DatasetT] | None = None
    train_sampler: Sampler[ShaftSampleRef] | None = None
    train_schedule: ShaftSampleSchedule | None = None
    train_execution_fingerprint: str | None = None
    train_execution_contract_complete: bool = False
    train_execution_incomplete_reasons: tuple[str, ...] = ()
    train_stream_fingerprint: str | None = None


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
        self._prompt_source: ShaftPromptSource | None = None

    def _get_prompt_source(self) -> ShaftPromptSource:
        if self._prompt_source is None:
            self._prompt_source = build_prompt_source_resolver(
                self.data_config.prompt_sources,
                default_seed=self.seed,
            )
        return self._prompt_source

    def build_record_cache_plan(self, *, shard_count: int) -> ShaftRecordCachePlan:
        prompt_source = self._get_prompt_source()
        tasks: list[ShaftRecordCacheTask] = []
        for dataset_meta in build_dataset_metas(self.data_config):
            if not dataset_meta.enabled:
                continue
            if float(dataset_meta.weight) > 0:
                tasks.extend(
                    self._record_cache_tasks_for_split(
                        dataset_meta,
                        split="train",
                        prompt_source=prompt_source,
                    )
                )
            if dataset_meta.use_for_eval:
                tasks.extend(
                    self._record_cache_tasks_for_split(
                        dataset_meta,
                        split="val",
                        prompt_source=prompt_source,
                    )
                )
        return ShaftRecordCachePlan.build(tasks, shard_count=shard_count)

    def warm_record_caches(
        self,
        plan: ShaftRecordCachePlan,
        *,
        shard_index: int,
    ) -> ShaftRecordCacheWarmupSummary:
        tasks = plan.tasks_for_shard(shard_index)
        started_at = time.monotonic()
        results: list[ShaftRecordCacheTaskResult] = []
        for task in tasks:
            results.append(task.warm())
        return ShaftRecordCacheWarmupSummary(
            plan_fingerprint=plan.fingerprint,
            shard_index=int(shard_index),
            shard_count=plan.shard_count,
            task_count=len(tasks),
            source_size_bytes=sum(task.source_size_bytes for task in tasks),
            row_count=sum(result.row_count for result in results),
            duration_seconds=time.monotonic() - started_at,
        )

    def _warm_local_record_cache_shard(self) -> None:
        if get_world_size() <= 1:
            return
        local_world_size = get_local_world_size()
        local_rank = get_local_rank()
        if local_world_size <= 1:
            return
        if local_rank < 0 or local_rank >= local_world_size:
            raise ValueError(
                "LOCAL_RANK must satisfy 0 <= LOCAL_RANK < LOCAL_WORLD_SIZE for record "
                f"cache warmup, got rank={local_rank}, world_size={local_world_size}."
            )
        plan = self.build_record_cache_plan(shard_count=local_world_size)
        if is_rank_zero():
            shard_sizes = [
                sum(task.source_size_bytes for task in shard)
                for shard in plan.shards
            ]
            logger.info(
                "[record-cache] distributed warmup tasks=%d local_shards=%d "
                "source_bytes=%d shard_bytes=%s fingerprint=%s",
                len(plan.tasks),
                plan.shard_count,
                sum(task.source_size_bytes for task in plan.tasks),
                shard_sizes,
                plan.fingerprint,
            )
        summary = self.warm_record_caches(plan, shard_index=local_rank)
        logger.info(
            "[record-cache] local shard ready index=%d/%d tasks=%d rows=%d "
            "source_bytes=%d duration=%.2fs",
            summary.shard_index,
            summary.shard_count,
            summary.task_count,
            summary.row_count,
            summary.source_size_bytes,
            summary.duration_seconds,
        )

    def prepare_records(self) -> ShaftPreparedRecords[Any]:
        records_by_dataset_train: dict[str, Sequence[Any]] = {}
        records_by_dataset_val: dict[str, Sequence[Any]] = {}
        weights: dict[str, float] = {}
        dataset_online_pipelines: dict[str, OnlineSampleTransform] = {}
        prompt_source = self._get_prompt_source()
        self._warm_local_record_cache_shard()

        for dataset_meta in build_dataset_metas(self.data_config):
            if not dataset_meta.enabled:
                continue
            offline_pipeline = build_offline_pipeline(dataset_meta.offline_transforms)
            if float(dataset_meta.weight) > 0:
                weights[dataset_meta.dataset_name] = float(dataset_meta.weight)
                records_by_dataset_train[dataset_meta.dataset_name] = (
                    self._load_dataset_split(
                        dataset_meta,
                        split="train",
                        prompt_source=prompt_source,
                        offline_pipeline=offline_pipeline,
                    )
                )
            if dataset_meta.use_for_eval:
                records_by_dataset_val[dataset_meta.dataset_name] = (
                    self._load_dataset_split(
                        dataset_meta,
                        split="val",
                        prompt_source=prompt_source,
                        offline_pipeline=offline_pipeline,
                    )
                )
            dataset_online_pipelines[dataset_meta.dataset_name] = build_online_pipeline(
                dataset_meta.online_transforms
            )

        source_sizes = {
            dataset_name: len(records) for dataset_name, records in records_by_dataset_train.items()
        }
        schedule_config = self.data_config.schedule
        planned_grouping = self.data_config.batching.grouping in {
            "length",
            "bounded_cost",
        }
        train_schedule: ShaftSampleSchedule | None = None
        train_sampler: ShaftSampleSampler | None = None
        if planned_grouping:
            train_schedule = ShaftSampleSchedule(
                source_sizes,
                weights,
                strategy=schedule_config.mixing,
                shuffle=schedule_config.shuffle,
                seed=self.seed,
            )
        else:
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
            if schedule_config.mixing != "weighted" or schedule_config.shuffle:
                train_schedule = sample_plan.schedule
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
        train_online_transforms.append(prompt_source)
        eval_online_transforms.append(prompt_source)
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
            suppress_train_decompression_bomb_warning=(planned_grouping),
        )

    def _load_dataset_split(
        self,
        dataset_meta: ShaftDatasetMeta,
        *,
        split: str,
        prompt_source: ShaftPromptSource,
        offline_pipeline: Callable[[Sequence[Any]], Sequence[Any]],
    ) -> Sequence[Any]:
        prompt_source_records = prompt_source.prepare_records(
            dataset_meta,
            split=split,
            cache_dir=self.data_config.record_cache_dir,
            offline_pipeline=offline_pipeline,
        )
        if prompt_source_records is not None:
            return prompt_source_records
        source_impl = self._build_standard_data_source(
            dataset_meta,
            split=split,
            prompt_source=prompt_source,
        )
        return offline_pipeline(source_impl.load_split(split))

    def _record_cache_tasks_for_split(
        self,
        dataset_meta: ShaftDatasetMeta,
        *,
        split: str,
        prompt_source: ShaftPromptSource,
    ) -> tuple[ShaftRecordCacheTask, ...]:
        prompt_source_tasks = prompt_source.record_cache_tasks(
            dataset_meta,
            split=split,
            cache_dir=self.data_config.record_cache_dir,
        )
        if prompt_source_tasks is not None:
            return prompt_source_tasks
        return self._build_standard_data_source(
            dataset_meta,
            split=split,
            prompt_source=prompt_source,
        ).record_cache_tasks(split)

    def _build_standard_data_source(
        self,
        dataset_meta: ShaftDatasetMeta,
        *,
        split: str,
        prompt_source: ShaftPromptSource,
    ) -> BaseDataSource:
        validation_fingerprint = prompt_source.record_validation_fingerprint(
            dataset_meta.dataset_name,
            split=split,
        )
        return build_data_source(
            dataset_meta,
            cache_dir=self.data_config.record_cache_dir,
            record_validator=lambda record, current_split: prompt_source.validate_record(
                record,
                dataset_name=dataset_meta.dataset_name,
                split=current_split,
            ),
            validation_fingerprint=validation_fingerprint,
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
                fingerprint=hashlib.sha256(repr(fingerprint_payload).encode("utf-8")).hexdigest(),
            )
        return _dataset_aware_online_transform


def _supports_kwarg(callable_obj: Any, keyword: str) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return True
    if keyword in signature.parameters:
        return True
    return any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()
    )


def _train_input_fingerprint(
    *,
    sample_fingerprint: str,
    transforms: Sequence[OnlineSampleTransform],
    record_fingerprints: tuple[tuple[str, str], ...],
    media_snapshot_id: str | None,
) -> str:
    transform_fingerprints = tuple(
        (
            planning_online_transform_fingerprint(transform)
            if is_planning_safe_online_transform(transform)
            else f"unversioned:{transform.__module__}.{getattr(transform, '__qualname__', type(transform).__qualname__)}"
        )
        for transform in transforms
    )
    return hashlib.sha256(
        repr(
            (
                "shaft-train-execution-v3",
                str(sample_fingerprint),
                record_fingerprints,
                str(media_snapshot_id or ""),
                transform_fingerprints,
            )
        ).encode("utf-8")
    ).hexdigest()


def _record_sequence_fingerprint(records: Sequence[Any]) -> str:
    explicit = str(getattr(records, "fingerprint", "")).strip()
    if explicit:
        return explicit
    digest = hashlib.sha256(b"shaft-inline-record-sequence-v1\0")
    for record in records:
        if is_dataclass(record):
            payload = asdict(record)
        elif isinstance(record, dict):
            payload = record
        else:
            raise ValueError(
                "Training record sequences must expose a stable fingerprint or contain "
                "JSON-compatible dataclass/dict records."
            )
        encoded = canonical_json(payload).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _build_dataset(
    dataset_cls: type[DatasetT],
    records: Any,
    **kwargs: Any,
) -> DatasetT:
    filtered_kwargs = {
        key: value for key, value in kwargs.items() if _supports_kwarg(dataset_cls, key)
    }
    return dataset_cls(records, **filtered_kwargs)
