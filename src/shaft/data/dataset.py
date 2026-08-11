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

from .planned import ShaftPlannedSampleRef
from .mixing import ShaftSamplePlan, ShaftSampleRef, ShaftSampleSchedule


def _normalize_record_image_paths(
    *,
    image_path: str | Path | None,
    image_paths: Sequence[str | Path] | None,
    required: bool,
) -> tuple[str, ...]:
    if image_path is not None and image_paths is not None:
        raise ValueError("Provide either image_path or image_paths, not both.")
    raw_paths: Sequence[str | Path]
    if image_paths is not None:
        raw_paths = image_paths
    elif image_path is not None:
        raw_paths = (image_path,)
    else:
        raw_paths = ()
    if isinstance(raw_paths, (str, Path)):
        raw_paths = (raw_paths,)
    normalized = tuple(str(path).strip() for path in raw_paths)
    if any(not path for path in normalized):
        raise ValueError("image_paths must contain only non-empty paths.")
    if required and not normalized:
        raise ValueError("At least one image path is required.")
    return normalized


class _OrderedImageRecord:
    image_paths: tuple[str, ...]

    @property
    def image_path(self) -> str | None:
        if not self.image_paths:
            return None
        if len(self.image_paths) != 1:
            raise ValueError(
                "image_path is available only for records with exactly one image; "
                "use image_paths for ordered multi-image records."
            )
        return self.image_paths[0]


@dataclass(init=False)
class SFTRecord(_OrderedImageRecord):
    image_paths: tuple[str, ...]
    target_text: str
    dataset_name: str = "default"
    sample_id: str | None = None
    messages: list[dict[str, Any]] | None = None
    system_prompt: str = ""
    user_prompt: str = ""
    prompt_args: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        image_path: str | Path | None = None,
        target_text: str = "",
        dataset_name: str = "default",
        sample_id: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        system_prompt: str = "",
        user_prompt: str = "",
        prompt_args: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
        *,
        image_paths: Sequence[str | Path] | None = None,
    ) -> None:
        self.image_paths = _normalize_record_image_paths(
            image_path=image_path,
            image_paths=image_paths,
            required=True,
        )
        self.target_text = str(target_text)
        self.dataset_name = str(dataset_name)
        self.sample_id = sample_id
        self.messages = messages
        self.system_prompt = str(system_prompt)
        self.user_prompt = str(user_prompt)
        self.prompt_args = dict(prompt_args or {})
        self.extra = dict(extra or {})


@dataclass(init=False)
class DPORecord(_OrderedImageRecord):
    image_paths: tuple[str, ...]
    chosen_text: str
    rejected_text: str
    dataset_name: str = "default"
    sample_id: str | None = None
    messages: list[dict[str, Any]] | None = None
    system_prompt: str = ""
    user_prompt: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        image_path: str | Path | None = None,
        chosen_text: str = "",
        rejected_text: str = "",
        dataset_name: str = "default",
        sample_id: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        system_prompt: str = "",
        user_prompt: str = "",
        extra: dict[str, Any] | None = None,
        *,
        image_paths: Sequence[str | Path] | None = None,
    ) -> None:
        self.image_paths = _normalize_record_image_paths(
            image_path=image_path,
            image_paths=image_paths,
            required=True,
        )
        self.chosen_text = str(chosen_text)
        self.rejected_text = str(rejected_text)
        self.dataset_name = str(dataset_name)
        self.sample_id = sample_id
        self.messages = messages
        self.system_prompt = str(system_prompt)
        self.user_prompt = str(user_prompt)
        self.extra = dict(extra or {})


@dataclass(init=False)
class PPORecord(_OrderedImageRecord):
    image_paths: tuple[str, ...]
    dataset_name: str = "default"
    sample_id: str | None = None
    messages: list[dict[str, Any]] | None = None
    system_prompt: str = ""
    user_prompt: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        image_path: str | Path | None = None,
        dataset_name: str = "default",
        sample_id: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        system_prompt: str = "",
        user_prompt: str = "",
        extra: dict[str, Any] | None = None,
        *,
        image_paths: Sequence[str | Path] | None = None,
    ) -> None:
        self.image_paths = _normalize_record_image_paths(
            image_path=image_path,
            image_paths=image_paths,
            required=False,
        )
        self.dataset_name = str(dataset_name)
        self.sample_id = sample_id
        self.messages = messages
        self.system_prompt = str(system_prompt)
        self.user_prompt = str(user_prompt)
        self.extra = dict(extra or {})


class ShaftVisionDatasetBase(Dataset):
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

    def _load_images(self, image_paths: Sequence[str]) -> tuple[Image.Image, ...]:
        return tuple(self._load_image(image_path) for image_path in image_paths)

    def _apply_online_transforms(self, sample: dict[str, Any]) -> dict[str, Any]:
        for transform in self.online_transforms:
            sample = transform(sample)
        return sample

    def _resolve_record(
        self,
        records: Sequence[Any] | dict[str, Sequence[Any]],
        index: int | ShaftSampleRef | ShaftPlannedSampleRef,
    ) -> tuple[Any, ShaftSampleRef | None]:
        if isinstance(index, ShaftPlannedSampleRef):
            index = index.sample_ref
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

    @staticmethod
    def _attach_batch_context(
        sample: dict[str, Any],
        index: int | ShaftSampleRef | ShaftPlannedSampleRef,
    ) -> dict[str, Any]:
        if not isinstance(index, ShaftPlannedSampleRef):
            return sample
        resolved = dict(sample)
        resolved["_batch_context"] = index.batch_context.to_dict()
        return resolved


class SFTDataset(ShaftVisionDatasetBase):
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
        images: tuple[Any, ...] | None,
    ) -> dict[str, Any]:
        image_row: Any = None
        if images is not None:
            image_row = images[0] if len(images) == 1 else images
        return {
            "dataset_name": record.dataset_name,
            "sample_id": record.sample_id or Path(record.image_paths[0]).stem,
            "image_paths": record.image_paths,
            "image_path": record.image_paths[0] if len(record.image_paths) == 1 else None,
            "images": images,
            "image": image_row,
            "target_text": record.target_text,
            "messages": record.messages,
            "system_prompt": record.system_prompt,
            "user_prompt": record.user_prompt,
            "prompt_args": dict(record.prompt_args),
            "extra": dict(record.extra),
            **self._runtime_context(sample_ref),
        }

    def get_planning_item(self, index: int | ShaftSampleRef) -> dict[str, Any]:
        """Resolve deterministic text transforms without decoding the image payload."""

        record, sample_ref = self._resolve_record(self.records, index)
        sample = self._build_sample(
            record,
            sample_ref=sample_ref,
            images=None,
        )
        return self._apply_online_transforms(sample)

    def __getitem__(
        self,
        index: int | ShaftSampleRef | ShaftPlannedSampleRef,
    ) -> dict[str, Any]:
        record, sample_ref = self._resolve_record(self.records, index)
        sample = self._build_sample(
            record,
            sample_ref=sample_ref,
            images=self._load_images(record.image_paths),
        )
        sample = self._apply_online_transforms(sample)
        return self._attach_batch_context(sample, index)


class GRPODataset(Dataset):
    SHAFT_INPUT_POLICY_VERSION = "shaft-grpo-dataset-input-v1"

    def __init__(
        self,
        dataset: Dataset,
        *,
        template: Any,
        image_preprocessor: Any | None = None,
    ) -> None:
        self.dataset = dataset
        self.template = template
        self.image_preprocessor = image_preprocessor

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int | ShaftSampleRef) -> dict[str, Any]:
        item = self.dataset[index]
        raw_images = item.get("images")
        if raw_images is None:
            raw_images = item.get("image")
        images = (
            tuple(raw_images)
            if isinstance(raw_images, (list, tuple))
            else (() if raw_images is None else (raw_images,))
        )
        if len(images) != 1:
            raise ValueError("GRPO currently requires exactly one image per sample.")
        image = images[0]
        if self.image_preprocessor is not None:
            image = self.image_preprocessor(image)
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
            "image_paths": item.get("image_paths"),
            "extra": dict(item.get("extra", {})),
        }


class DPODataset(ShaftVisionDatasetBase):
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
        images = self._load_images(record.image_paths)
        sample = {
            "dataset_name": record.dataset_name,
            "sample_id": record.sample_id or Path(record.image_paths[0]).stem,
            "image_paths": record.image_paths,
            "image_path": record.image_paths[0] if len(record.image_paths) == 1 else None,
            "images": images,
            "image": images[0] if len(images) == 1 else images,
            "messages": record.messages,
            "system_prompt": record.system_prompt,
            "user_prompt": record.user_prompt,
            "chosen_text": record.chosen_text,
            "rejected_text": record.rejected_text,
            "extra": dict(record.extra),
            **self._runtime_context(sample_ref),
        }
        return self._apply_online_transforms(sample)


class PPODataset(ShaftVisionDatasetBase):
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
                Path(record.image_paths[0]).stem
                if record.image_paths
                else f"row-{sample_ref.row_index if sample_ref is not None else int(index)}"
            ),
            "image_paths": record.image_paths,
            "image_path": record.image_paths[0] if len(record.image_paths) == 1 else None,
            "messages": record.messages,
            "system_prompt": record.system_prompt,
            "user_prompt": record.user_prompt,
            "extra": dict(record.extra),
            **self._runtime_context(sample_ref),
        }
        return self._apply_online_transforms(sample)
