from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset


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
    image_path: str
    dataset_name: str = "default"
    sample_id: str | None = None
    messages: list[dict[str, Any]] | None = None
    system_prompt: str = ""
    user_prompt: str = "Output only valid JSON. No markdown and no extra text."
    extra: dict[str, Any] = field(default_factory=dict)


class _BaseVisionDataset(Dataset):
    def __init__(self, *, online_transforms: list[Any] | None = None, split: str = "train") -> None:
        self.online_transforms = list(online_transforms or [])
        self.split = str(split).strip() or "train"

    def _runtime_context(self, train_sampler: Any | None) -> dict[str, Any]:
        return {
            "_split": self.split,
            "_epoch": int(getattr(train_sampler, "epoch", 0) or 0),
        }

    def _load_image(self, image_path: str):
        return Image.open(image_path).convert("RGB")

    def _apply_online_transforms(self, sample: dict[str, Any]) -> dict[str, Any]:
        for transform in self.online_transforms:
            sample = transform(sample)
        return sample

    @staticmethod
    def _build_flat_indices(records: dict[str, list[Any]]) -> list[tuple[str, int]]:
        indices: list[tuple[str, int]] = []
        for dataset_name in sorted(records):
            indices.extend((dataset_name, row_index) for row_index in range(len(records[dataset_name])))
        return indices

    def _resolve_record(
        self,
        records: Sequence[Any] | dict[str, list[Any]],
        index: int | tuple[str, int],
        *,
        mixed_indices: Sequence[tuple[str, int]] | None = None,
    ) -> Any:
        if isinstance(records, dict):
            if isinstance(index, tuple):
                dataset_name, row_index = index
                return records[dataset_name][row_index]
            if mixed_indices is None:
                mixed_indices = self._build_flat_indices(records)
            dataset_name, row_index = mixed_indices[int(index)]
            return records[dataset_name][row_index]
        return records[int(index)]


def _fit_image_to_pixel_budget(
    image: Any,
    *,
    min_pixels: int | None,
    max_pixels: int | None,
) -> Any:
    if not isinstance(image, Image.Image):
        return image
    width, height = image.size
    if width <= 0 or height <= 0:
        return image
    area = width * height
    target_area = area
    if max_pixels is not None and area > int(max_pixels):
        target_area = int(max_pixels)
    elif min_pixels is not None and area < int(min_pixels):
        target_area = int(min_pixels)
    if target_area <= 0 or target_area == area:
        return image
    scale = math.sqrt(float(target_area) / float(area))
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))
    if new_width == width and new_height == height:
        return image
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


class SFTDataset(_BaseVisionDataset):
    def __init__(
        self,
        records: Sequence[SFTRecord] | dict[str, list[SFTRecord]],
        *,
        online_transforms: list[Any] | None = None,
        split: str = "train",
        mixed_length: int | None = None,
        mixed_indices: Sequence[tuple[str, int]] | None = None,
        train_sampler: Any | None = None,
    ) -> None:
        super().__init__(online_transforms=online_transforms, split=split)
        self.records = records
        self.mixed_length = int(mixed_length) if mixed_length is not None else None
        self.mixed_indices = list(mixed_indices) if mixed_indices is not None else None
        self.train_sampler = train_sampler

    def __len__(self) -> int:
        if self.mixed_length is not None:
            return self.mixed_length
        if isinstance(self.records, dict):
            return sum(len(records) for records in self.records.values())
        return len(self.records)

    def __getitem__(self, index: int | tuple[str, int]) -> dict[str, Any]:
        mixed_indices = getattr(self.train_sampler, "current_indices", None)
        if mixed_indices is None:
            mixed_indices = self.mixed_indices
        record = self._resolve_record(self.records, index, mixed_indices=mixed_indices)
        image = self._load_image(record.image_path)
        sample = {
            "dataset_name": record.dataset_name,
            "sample_id": record.sample_id or Path(record.image_path).stem,
            "image_path": record.image_path,
            "image": image,
            "target_text": record.target_text,
            "messages": record.messages,
            "system_prompt": record.system_prompt,
            "user_prompt": record.user_prompt,
            "extra": dict(record.extra),
            **self._runtime_context(self.train_sampler),
        }
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

    def __getitem__(self, index: int | tuple[str, int]) -> dict[str, Any]:
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
        records: Sequence[DPORecord] | dict[str, list[DPORecord]],
        *,
        online_transforms: list[Any] | None = None,
        split: str = "train",
        mixed_length: int | None = None,
        mixed_indices: Sequence[tuple[str, int]] | None = None,
        train_sampler: Any | None = None,
    ) -> None:
        super().__init__(online_transforms=online_transforms, split=split)
        self.records = records
        self.mixed_length = int(mixed_length) if mixed_length is not None else None
        self.mixed_indices = list(mixed_indices) if mixed_indices is not None else None
        self.train_sampler = train_sampler

    def __len__(self) -> int:
        if self.mixed_length is not None:
            return self.mixed_length
        if isinstance(self.records, dict):
            return sum(len(records) for records in self.records.values())
        return len(self.records)

    def __getitem__(self, index: int | tuple[str, int]) -> dict[str, Any]:
        mixed_indices = getattr(self.train_sampler, "current_indices", None)
        if mixed_indices is None:
            mixed_indices = self.mixed_indices
        record = self._resolve_record(self.records, index, mixed_indices=mixed_indices)
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
            **self._runtime_context(self.train_sampler),
        }
        return self._apply_online_transforms(sample)


class PPODataset(_BaseVisionDataset):
    def __init__(
        self,
        records: Sequence[PPORecord] | dict[str, list[PPORecord]],
        *,
        online_transforms: list[Any] | None = None,
        split: str = "train",
        mixed_length: int | None = None,
        mixed_indices: Sequence[tuple[str, int]] | None = None,
        train_sampler: Any | None = None,
    ) -> None:
        super().__init__(online_transforms=online_transforms, split=split)
        self.records = records
        self.mixed_length = int(mixed_length) if mixed_length is not None else None
        self.mixed_indices = list(mixed_indices) if mixed_indices is not None else None
        self.train_sampler = train_sampler

    def __len__(self) -> int:
        if self.mixed_length is not None:
            return self.mixed_length
        if isinstance(self.records, dict):
            return sum(len(records) for records in self.records.values())
        return len(self.records)

    def __getitem__(self, index: int | tuple[str, int]) -> dict[str, Any]:
        mixed_indices = getattr(self.train_sampler, "current_indices", None)
        if mixed_indices is None:
            mixed_indices = self.mixed_indices
        record = self._resolve_record(self.records, index, mixed_indices=mixed_indices)
        image = self._load_image(record.image_path)
        sample = {
            "dataset_name": record.dataset_name,
            "sample_id": record.sample_id or Path(record.image_path).stem,
            "image_path": record.image_path,
            "image": image,
            "messages": record.messages,
            "system_prompt": record.system_prompt,
            "user_prompt": record.user_prompt,
            "extra": dict(record.extra),
            **self._runtime_context(self.train_sampler),
        }
        return self._apply_online_transforms(sample)
