from __future__ import annotations

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


class SFTDataset(_BaseVisionDataset):
    def __init__(
        self,
        records: list[SFTRecord],
        *,
        online_transforms: list[Any] | None = None,
    ) -> None:
        super().__init__(online_transforms=online_transforms)
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
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
        records: list[DPORecord],
        *,
        online_transforms: list[Any] | None = None,
    ) -> None:
        super().__init__(online_transforms=online_transforms)
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
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
        records: list[PPORecord],
        *,
        online_transforms: list[Any] | None = None,
    ) -> None:
        super().__init__(online_transforms=online_transforms)
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
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
