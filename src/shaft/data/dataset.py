from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import warnings

from PIL import Image
from torch.utils.data import Dataset

from shaft.utils.qwen_pixel_budget import apply_qwen_pixel_budget

from .mixing import ShaftSamplePlan, ShaftSampleRef, ShaftSampleSchedule


@dataclass
class SFTRecord:
    image_path: str
    target_text: str
    dataset_name: str = "default"
    sample_id: str | None = None
    messages: list[dict[str, Any]] | None = None
    system_prompt: str = ""
    user_prompt: str = "Output only valid JSON. No markdown and no extra text."
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class DPORecord:
    image_path: str
    chosen_text: str
    rejected_text: str
    dataset_name: str = "default"
    sample_id: str | None = None
    messages: list[dict[str, Any]] | None = None
    system_prompt: str = ""
    user_prompt: str = "Output only valid JSON. No markdown and no extra text."
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class PPORecord:
    image_path: str | None = None
    dataset_name: str = "default"
    sample_id: str | None = None
    messages: list[dict[str, Any]] | None = None
    system_prompt: str = ""
    user_prompt: str = "Output only valid JSON. No markdown and no extra text."
    extra: dict[str, Any] = field(default_factory=dict)


class _BaseVisionDataset(Dataset):
    def __init__(
        self,
        records: Sequence[Any] | dict[str, Sequence[Any]],
        *,
        online_transforms: list[Any] | None = None,
        split: str = "train",
        sample_plan: ShaftSamplePlan | None = None,
        sample_schedule: ShaftSampleSchedule | None = None,
        media_snapshot_id: str | None = None,
        image_cache_size: int = 0,
        suppress_decompression_bomb_warning: bool = False,
    ) -> None:
        self.records = records
        self.online_transforms = list(online_transforms or [])
        self.split = str(split).strip() or "train"
        self.sample_plan = sample_plan
        self.sample_schedule = sample_schedule
        self.media_snapshot_id = str(media_snapshot_id or "").strip()
        self.image_cache_size = max(int(image_cache_size), 0)
        self.suppress_decompression_bomb_warning = bool(
            suppress_decompression_bomb_warning
        )
        self._image_cache: OrderedDict[str, Image.Image] = OrderedDict()

    def __len__(self) -> int:
        if self.sample_plan is not None:
            return len(self.sample_plan)
        if isinstance(self.records, dict):
            return sum(len(records) for records in self.records.values())
        return len(self.records)

    def _runtime_context(self, sample_ref: ShaftSampleRef | None) -> dict[str, Any]:
        context: dict[str, Any] = {"_split": self.split}
        if sample_ref is not None:
            context["_sample_context"] = sample_ref.context.to_dict()
        return context

    def _load_image(self, image_path: str):
        cached = self._image_cache.get(image_path)
        if cached is not None:
            self._image_cache.move_to_end(image_path)
            return cached.copy()
        with warnings.catch_warnings():
            if self.suppress_decompression_bomb_warning:
                warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(image_path) as image:
                decoded = image.convert("RGB")
        if self.image_cache_size > 0:
            self._image_cache[image_path] = decoded.copy()
            self._image_cache.move_to_end(image_path)
            while len(self._image_cache) > self.image_cache_size:
                self._image_cache.popitem(last=False)
        return decoded

    def _apply_online_transforms(self, sample: dict[str, Any]) -> dict[str, Any]:
        for transform in self.online_transforms:
            sample = transform(sample)
        return sample

    def _resolve_record(
        self,
        records: Sequence[Any] | dict[str, Sequence[Any]],
        index: int | ShaftSampleRef,
    ) -> tuple[Any, ShaftSampleRef | None]:
        if isinstance(records, dict):
            if isinstance(index, ShaftSampleRef):
                sample_ref = index
            elif self.sample_plan is not None:
                sample_ref = self.sample_plan.ref_at(int(index))
            else:
                position = int(index)
                names = sorted(records)
                ends: list[int] = []
                total = 0
                for name in names:
                    total += len(records[name])
                    ends.append(total)
                source_index = bisect_right(ends, position)
                if position < 0 or source_index >= len(names):
                    raise IndexError(position)
                start = 0 if source_index == 0 else ends[source_index - 1]
                return records[names[source_index]][position - start], None
            return records[sample_ref.dataset_name][sample_ref.row_index], sample_ref
        return records[int(index)], None


def _fit_image_to_pixel_budget(
    image: Any,
    *,
    min_pixels: int | None,
    max_pixels: int | None,
) -> Any:
    if not isinstance(image, Image.Image):
        return image
    if min_pixels is None and max_pixels is None:
        return image
    resized, _ = apply_qwen_pixel_budget(
        image,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    return resized


class SFTDataset(_BaseVisionDataset):
    def __init__(
        self,
        records: Sequence[SFTRecord] | dict[str, Sequence[SFTRecord]],
        *,
        online_transforms: list[Any] | None = None,
        split: str = "train",
        sample_plan: ShaftSamplePlan | None = None,
        sample_schedule: ShaftSampleSchedule | None = None,
        media_snapshot_id: str | None = None,
        image_cache_size: int = 0,
        suppress_decompression_bomb_warning: bool = False,
    ) -> None:
        super().__init__(
            records,
            online_transforms=online_transforms,
            split=split,
            sample_plan=sample_plan,
            sample_schedule=sample_schedule,
            media_snapshot_id=media_snapshot_id,
            image_cache_size=image_cache_size,
            suppress_decompression_bomb_warning=(
                suppress_decompression_bomb_warning
            ),
        )

    def _build_sample(
        self,
        record: SFTRecord,
        *,
        sample_ref: ShaftSampleRef | None,
        image: Any,
    ) -> dict[str, Any]:
        return {
            "dataset_name": record.dataset_name,
            "sample_id": record.sample_id or Path(record.image_path).stem,
            "image_path": record.image_path,
            "image": image,
            "target_text": record.target_text,
            "messages": record.messages,
            "system_prompt": record.system_prompt,
            "user_prompt": record.user_prompt,
            "extra": dict(record.extra),
            **self._runtime_context(sample_ref),
        }

    def get_planning_item(self, index: int | ShaftSampleRef) -> dict[str, Any]:
        """Resolve deterministic text transforms without decoding the image payload."""

        record, sample_ref = self._resolve_record(self.records, index)
        sample = self._build_sample(
            record,
            sample_ref=sample_ref,
            image=None,
        )
        return self._apply_online_transforms(sample)

    def __getitem__(self, index: int | ShaftSampleRef) -> dict[str, Any]:
        record, sample_ref = self._resolve_record(self.records, index)
        sample = self._build_sample(
            record,
            sample_ref=sample_ref,
            image=self._load_image(record.image_path),
        )
        return self._apply_online_transforms(sample)


class GRPODataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        *,
        template: Any,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
    ) -> None:
        self.dataset = dataset
        self.template = template
        self.min_pixels = int(min_pixels) if min_pixels is not None else None
        self.max_pixels = int(max_pixels) if max_pixels is not None else None

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int | ShaftSampleRef) -> dict[str, Any]:
        item = self.dataset[index]
        image = _fit_image_to_pixel_budget(
            item.get("image"),
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        if image is not item.get("image"):
            item = dict(item)
            item["image"] = image
        prompt = self.template.prepare_messages(self.template.resolve_messages(item))
        return {
            "prompt": prompt,
            "image": image,
            "target_text": str(item.get("target_text", "")),
            "dataset_name": item.get("dataset_name"),
            "sample_id": item.get("sample_id"),
            "image_path": item.get("image_path"),
            "extra": dict(item.get("extra", {})),
        }


class DPODataset(_BaseVisionDataset):
    def __init__(
        self,
        records: Sequence[DPORecord] | dict[str, Sequence[DPORecord]],
        *,
        online_transforms: list[Any] | None = None,
        split: str = "train",
        sample_plan: ShaftSamplePlan | None = None,
        sample_schedule: ShaftSampleSchedule | None = None,
        media_snapshot_id: str | None = None,
        image_cache_size: int = 0,
        suppress_decompression_bomb_warning: bool = False,
    ) -> None:
        super().__init__(
            records,
            online_transforms=online_transforms,
            split=split,
            sample_plan=sample_plan,
            sample_schedule=sample_schedule,
            media_snapshot_id=media_snapshot_id,
            image_cache_size=image_cache_size,
            suppress_decompression_bomb_warning=(
                suppress_decompression_bomb_warning
            ),
        )

    def __getitem__(self, index: int | ShaftSampleRef) -> dict[str, Any]:
        record, sample_ref = self._resolve_record(self.records, index)
        image = self._load_image(record.image_path)
        sample = {
            "dataset_name": record.dataset_name,
            "sample_id": record.sample_id or Path(record.image_path).stem,
            "image_path": record.image_path,
            "image": image,
            "messages": record.messages,
            "system_prompt": record.system_prompt,
            "user_prompt": record.user_prompt,
            "chosen_text": record.chosen_text,
            "rejected_text": record.rejected_text,
            "extra": dict(record.extra),
            **self._runtime_context(sample_ref),
        }
        return self._apply_online_transforms(sample)


class PPODataset(_BaseVisionDataset):
    def __init__(
        self,
        records: Sequence[PPORecord] | dict[str, Sequence[PPORecord]],
        *,
        online_transforms: list[Any] | None = None,
        split: str = "train",
        sample_plan: ShaftSamplePlan | None = None,
        sample_schedule: ShaftSampleSchedule | None = None,
        media_snapshot_id: str | None = None,
        image_cache_size: int = 0,
        suppress_decompression_bomb_warning: bool = False,
    ) -> None:
        super().__init__(
            records,
            online_transforms=online_transforms,
            split=split,
            sample_plan=sample_plan,
            sample_schedule=sample_schedule,
            media_snapshot_id=media_snapshot_id,
            image_cache_size=image_cache_size,
            suppress_decompression_bomb_warning=(
                suppress_decompression_bomb_warning
            ),
        )

    def __getitem__(self, index: int | ShaftSampleRef) -> dict[str, Any]:
        record, sample_ref = self._resolve_record(self.records, index)
        sample = {
            "dataset_name": record.dataset_name,
            "sample_id": record.sample_id or (
                Path(record.image_path).stem
                if record.image_path
                else f"row-{sample_ref.row_index if sample_ref is not None else int(index)}"
            ),
            "image_path": record.image_path,
            "messages": record.messages,
            "system_prompt": record.system_prompt,
            "user_prompt": record.user_prompt,
            "extra": dict(record.extra),
            **self._runtime_context(sample_ref),
        }
        return self._apply_online_transforms(sample)
