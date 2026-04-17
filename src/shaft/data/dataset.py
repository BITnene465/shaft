from __future__ import annotations

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
    def __init__(self, *, online_transforms: list[Any] | None = None) -> None:
        self.online_transforms = list(online_transforms or [])

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


class SFTDataset(_BaseVisionDataset):
    def __init__(
        self,
        records: Sequence[SFTRecord] | dict[str, list[SFTRecord]],
        *,
        online_transforms: list[Any] | None = None,
        mixed_length: int | None = None,
        mixed_indices: Sequence[tuple[str, int]] | None = None,
        train_sampler: Any | None = None,
    ) -> None:
        super().__init__(online_transforms=online_transforms)
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
        }
        return self._apply_online_transforms(sample)


class DPODataset(_BaseVisionDataset):
    def __init__(
        self,
        records: Sequence[DPORecord] | dict[str, list[DPORecord]],
        *,
        online_transforms: list[Any] | None = None,
        mixed_length: int | None = None,
        mixed_indices: Sequence[tuple[str, int]] | None = None,
        train_sampler: Any | None = None,
    ) -> None:
        super().__init__(online_transforms=online_transforms)
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
        }
        return self._apply_online_transforms(sample)


class PPODataset(_BaseVisionDataset):
    def __init__(
        self,
        records: Sequence[PPORecord] | dict[str, list[PPORecord]],
        *,
        online_transforms: list[Any] | None = None,
        mixed_length: int | None = None,
        mixed_indices: Sequence[tuple[str, int]] | None = None,
        train_sampler: Any | None = None,
    ) -> None:
        super().__init__(online_transforms=online_transforms)
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
        }
        return self._apply_online_transforms(sample)
