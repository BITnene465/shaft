from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from shaft.data.collator import SFTCollator
from shaft.data.dataset import SFTDataset, SFTRecord
from shaft.data.record_store import ShaftArrowRecordStore, ShaftConcatRecordStore, register_record_type
from shaft.data.registry import register_data_source
from shaft.data.sources import BaseDataSource, Split, _build_sft_record_from_raw
from shaft.training.distribution_loss import TeacherDistribution

from .artifact import (
    OfflineKDArtifactReference,
    OfflineKDArtifactStore,
    media_content_fingerprint,
)
from .media_plan import media_plan_from_item


@dataclass(init=False)
class OfflineKDRecord(SFTRecord):
    distillation_artifact_id: str
    distillation_shard: str
    distillation_row: int

    def __init__(
        self,
        *,
        distillation_ref: Mapping[str, Any] | None = None,
        distillation_artifact_id: str | None = None,
        distillation_shard: str | None = None,
        distillation_row: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if distillation_ref is None:
            if isinstance(distillation_row, str) and distillation_row.isdigit():
                distillation_row = int(distillation_row)
            distillation_ref = {
                "artifact_id": distillation_artifact_id,
                "shard": distillation_shard,
                "row": distillation_row,
            }
        elif any(
            value is not None
            for value in (distillation_artifact_id, distillation_shard, distillation_row)
        ):
            raise ValueError("Provide either distillation_ref or its stored scalar fields.")
        reference = OfflineKDArtifactReference.from_mapping(distillation_ref)
        self.distillation_artifact_id = reference.artifact_id
        self.distillation_shard = reference.shard
        self.distillation_row = reference.row

    @property
    def distillation_ref(self) -> dict[str, Any]:
        return {
            "artifact_id": self.distillation_artifact_id,
            "shard": self.distillation_shard,
            "row": self.distillation_row,
        }

    def runtime_sample_fields(self) -> Mapping[str, Any]:
        return {"distillation_ref": self.distillation_ref}


register_record_type(OfflineKDRecord)


def _build_offline_kd_record_from_raw(
    raw: dict[str, Any], *, jsonl_path: Path, line_no: int, dataset_name: str
) -> OfflineKDRecord:
    if "distillation_ref" not in raw:
        raise ValueError("Offline KD records require distillation_ref.")
    sft_raw = {key: value for key, value in raw.items() if key != "distillation_ref"}
    base = _build_sft_record_from_raw(
        sft_raw,
        jsonl_path=jsonl_path,
        line_no=line_no,
        dataset_name=dataset_name,
    )
    return OfflineKDRecord(
        image_paths=base.image_paths,
        target_text=base.target_text,
        target_reasoning_content=base.target_reasoning_content,
        dataset_name=base.dataset_name,
        sample_id=base.sample_id,
        messages=base.messages,
        system_prompt=base.system_prompt,
        user_prompt=base.user_prompt,
        prompt_args=base.prompt_args,
        extra=base.extra,
        distillation_ref=raw["distillation_ref"],
    )


def load_jsonl_offline_kd_records(
    path: str | Path,
    *,
    dataset_name: str,
    max_errors_to_report: int = 20,
    cache_dir: str | Path | None = None,
) -> ShaftArrowRecordStore[OfflineKDRecord]:
    return ShaftArrowRecordStore.from_jsonl(
        path,
        dataset_name=dataset_name,
        record_type=OfflineKDRecord,
        row_builder=_build_offline_kd_record_from_raw,
        max_errors_to_report=max_errors_to_report,
        cache_dir=cache_dir,
    )


@register_data_source("jsonl_offline_kd")
class JsonlOfflineKDDataSource(BaseDataSource):
    def load_split(self, split: Split) -> Sequence[OfflineKDRecord]:
        return ShaftConcatRecordStore(
            [
                load_jsonl_offline_kd_records(
                    path,
                    dataset_name=self.dataset_meta.dataset_name,
                    cache_dir=self.cache_dir,
                )
                for path in self._resolve_paths(split)
            ]
        )


class OfflineKDDataset(SFTDataset):
    def planning_preprocessed_image_sizes(
        self,
        item: dict[str, Any],
    ) -> tuple[tuple[int, int], ...] | None:
        media_plan = media_plan_from_item(item)
        if media_plan is None:
            return None
        return ((media_plan.target_width, media_plan.target_height),)

    def planning_image_cost_policy_signature(self) -> str:
        return "shaft-offline-kd-frozen-media-plan-cost-v1"


def _concatenate_distributions(
    distributions: list[TeacherDistribution],
) -> TeacherDistribution:
    if not distributions:
        raise ValueError("Offline KD batch requires at least one teacher distribution.")
    first = distributions[0]
    if any(
        (item.kind, item.vocab_size, item.top_k, item.temperature)
        != (first.kind, first.vocab_size, first.top_k, first.temperature)
        for item in distributions[1:]
    ):
        raise ValueError("Offline KD batch contains incompatible teacher distributions.")
    if first.kind == "dense_logits":
        return TeacherDistribution.from_dense_logits(
            torch.cat([item.dense_logits for item in distributions if item.dense_logits is not None])
        )
    tail_rows = [item.tail_log_probs for item in distributions]
    if any(item is None for item in tail_rows) and not all(
        item is None for item in tail_rows
    ):
        raise ValueError("Offline KD batch mixes top-k distributions with different tail schema.")
    return TeacherDistribution(
        kind="topk_tail",
        vocab_size=first.vocab_size,
        topk_token_ids=torch.cat(
            [item.topk_token_ids for item in distributions if item.topk_token_ids is not None]
        ),
        topk_log_probs=torch.cat(
            [item.topk_log_probs for item in distributions if item.topk_log_probs is not None]
        ),
        tail_log_probs=(
            None
            if all(item is None for item in tail_rows)
            else torch.cat([item for item in tail_rows if item is not None])
        ),
        temperature=first.temperature,
    )


class OfflineKDCollator(SFTCollator):
    SHAFT_INPUT_POLICY_VERSION = "shaft-offline-kd-collator-v1"

    def __init__(self, *args: Any, artifact_store: OfflineKDArtifactStore, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.layout != "padded" or self.packing_mode != "none":
            raise ValueError("Offline KD collator supports only padded, unpacked batches.")
        self.artifact_store = artifact_store
        self._media_plan_batch_active = False

    def _run_processor(
        self,
        prompt_texts: list[str],
        images: list[Any],
        *,
        dataset_names: list[str | None] | None = None,
    ):
        if not self._media_plan_batch_active:
            return super()._run_processor(
                prompt_texts,
                images,
                dataset_names=dataset_names,
            )
        return self.model_adapter.build_processor_batch(
            processor=self.processor,
            tokenizer=self.tokenizer,
            prompt_texts=prompt_texts,
            images=images,
            min_pixels=None,
            max_pixels=None,
            input_mode=self.input_mode,
        )

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        media_plans = [media_plan_from_item(item) for item in batch]
        if any(plan is None for plan in media_plans) and not all(
            plan is None for plan in media_plans
        ):
            raise ValueError("Offline KD batches cannot mix planned and unplanned media rows.")
        prepared_batch: list[dict[str, Any]] = []
        for item, media_plan in zip(batch, media_plans, strict=True):
            if media_plan is None:
                prepared_batch.append(item)
                continue
            raw_images = item.get("images")
            images = (
                tuple(raw_images)
                if isinstance(raw_images, (list, tuple))
                else (() if raw_images is None else (raw_images,))
            )
            if len(images) != 1:
                raise ValueError(
                    "Offline KD per-row media plans currently require exactly one image."
                )
            media_plan.validate_image_size(images[0])
            resized = self.model_adapter.prepare_rollout_image(
                images[0],
                min_pixels=media_plan.min_pixels,
                max_pixels=media_plan.max_pixels,
            )
            resolved = dict(item)
            resolved["images"] = (resized,)
            resolved["image"] = resized
            prepared_batch.append(resolved)
        self._media_plan_batch_active = bool(media_plans and media_plans[0] is not None)
        try:
            output = super().__call__(prepared_batch)
        finally:
            self._media_plan_batch_active = False
        labels = output.get("labels")
        if not isinstance(labels, torch.Tensor) or labels.ndim != 2:
            raise ValueError("Offline KD requires padded 2-D causal labels.")
        completion_mask = labels.ne(self.ignore_index)
        input_ids = output.get("input_ids")
        attention_mask = output.get("attention_mask")
        if not isinstance(input_ids, torch.Tensor) or not isinstance(attention_mask, torch.Tensor):
            raise ValueError("Offline KD requires input_ids and attention_mask tensors.")
        distributions: list[TeacherDistribution] = []
        for row_index, item in enumerate(batch):
            reference = OfflineKDArtifactReference.from_mapping(item.get("distillation_ref"))
            image_paths = tuple(item.get("image_paths") or ())
            if not image_paths:
                raise ValueError("Offline KD samples require immutable image_paths.")
            distributions.append(
                self.artifact_store.get(
                    reference,
                    completion_token_ids=labels[row_index][completion_mask[row_index]],
                    input_token_ids=input_ids[row_index][attention_mask[row_index].bool()],
                    media_sha256=media_content_fingerprint(image_paths),
                )
            )
        output["_shaft_offline_kd_completion_mask"] = completion_mask
        output["_shaft_offline_kd_teacher_distribution"] = _concatenate_distributions(
            distributions
        )
        return output
