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
    dataset_id: str = "default"
    sample_id: str | None = None
    messages: list[dict[str, Any]] | None = None
    system_prompt: str = ""
    user_prompt: str = "Output only valid JSON. No markdown and no extra text."
    extra: dict[str, Any] = field(default_factory=dict)


class SFTDataset(Dataset):
    def __init__(
        self,
        records: list[SFTRecord],
        *,
        online_transforms: list[Any] | None = None,
    ) -> None:
        self.records = records
        self.online_transforms = list(online_transforms or [])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image = Image.open(record.image_path).convert("RGB")
        sample = {
            "dataset_id": record.dataset_id,
            "sample_id": record.sample_id or Path(record.image_path).stem,
            "image_path": record.image_path,
            "image": image,
            "target_text": record.target_text,
            "messages": record.messages,
            "system_prompt": record.system_prompt,
            "user_prompt": record.user_prompt,
            "extra": dict(record.extra),
        }
        for transform in self.online_transforms:
            sample = transform(sample)
        return sample

