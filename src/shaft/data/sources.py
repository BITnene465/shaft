from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

from shaft.config import DataSourceConfig

from .dataset import SFTRecord
from .registry import DATA_SOURCE_REGISTRY, register_data_source

Split = Literal["train", "val"]


def _normalize_message_content(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, dict):
        if content.get("type") == "text":
            return [{"type": "text", "text": str(content.get("text", ""))}]
        if content.get("type") == "image":
            return [{"type": "image"}]
        if "text" in content:
            return [{"type": "text", "text": str(content["text"])}]
        return [{"type": "text", "text": str(content)}]
    if isinstance(content, list):
        items: list[dict[str, Any]] = []
        for item in content:
            items.extend(_normalize_message_content(item))
        return items
    return [{"type": "text", "text": str(content)}]


def _normalize_messages(record: dict[str, Any]) -> list[dict[str, Any]] | None:
    raw = record.get("messages")
    if raw is None:
        raw = record.get("conversation")
    if raw is None:
        raw = record.get("conversations")
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise TypeError("`messages` must be a list when provided.")
    normalized: list[dict[str, Any]] = []
    for message in raw:
        if not isinstance(message, dict):
            raise TypeError("Each message must be a dict.")
        role = str(message.get("role", "user")).strip().lower()
        content = _normalize_message_content(message.get("content", ""))
        normalized.append({"role": role, "content": content})
    return normalized


def _content_to_text(content: list[dict[str, Any]]) -> str:
    texts = [str(item.get("text", "")) for item in content if item.get("type") == "text"]
    return "".join(texts).strip()


def _extract_target_from_messages(messages: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]]]:
    if not messages:
        return None, messages
    last = messages[-1]
    if str(last.get("role", "")).strip().lower() != "assistant":
        return None, messages
    target_text = _content_to_text(last.get("content", []))
    return target_text, messages[:-1]


def _resolve_image_path(raw: dict[str, Any], jsonl_path: Path, line_no: int) -> str:
    image_obj = raw.get("image_path")
    if image_obj is None:
        image_obj = raw.get("image")
    if image_obj is None:
        images = raw.get("images")
        if isinstance(images, list) and images:
            image_obj = images[0]
    if image_obj is None:
        raise ValueError(f"Missing image path in {jsonl_path}:{line_no}. Expected image_path/image/images.")
    image_path = str(image_obj)
    if not Path(image_path).is_absolute():
        image_path = str((jsonl_path.parent / image_path).resolve())
    return image_path


def load_jsonl_records(path: str | Path, *, dataset_id: str) -> list[SFTRecord]:
    records: list[SFTRecord] = []
    jsonl_path = Path(path)
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            raw = json.loads(text)
            image_path = _resolve_image_path(raw, jsonl_path, line_no)
            messages = _normalize_messages(raw)
            target_text = raw.get("target_text")
            if target_text is None and messages is not None:
                extracted_target, prompt_messages = _extract_target_from_messages(messages)
                if extracted_target is not None:
                    target_text = extracted_target
                    messages = prompt_messages
            if target_text is None:
                raise ValueError(
                    f"Missing target text in {jsonl_path}:{line_no}. "
                    "Expected target_text or a trailing assistant message."
                )

            extra = {
                k: v
                for k, v in raw.items()
                if k not in {"image_path", "image", "images", "target_text", "messages", "conversation", "conversations"}
            }
            records.append(
                SFTRecord(
                    image_path=image_path,
                    target_text=str(target_text),
                    dataset_id=str(raw.get("dataset_id", dataset_id)),
                    sample_id=str(raw.get("sample_id", "")) or None,
                    messages=messages,
                    system_prompt=str(raw.get("system_prompt", "")),
                    user_prompt=str(
                        raw.get("user_prompt", "Output only valid JSON. No markdown and no extra text.")
                    ),
                    extra=extra,
                )
            )
    return records


class BaseDataSource(ABC):
    def __init__(self, config: DataSourceConfig) -> None:
        self.config = config

    @abstractmethod
    def load_split(self, split: Split) -> list[SFTRecord]:
        raise NotImplementedError

    def _resolve_paths(self, split: Split) -> list[str]:
        return list(self.config.train_paths if split == "train" else self.config.val_paths)


@register_data_source("jsonl_sft")
class JsonlSFTDataSource(BaseDataSource):
    def load_split(self, split: Split) -> list[SFTRecord]:
        records: list[SFTRecord] = []
        for path in self._resolve_paths(split):
            records.extend(load_jsonl_records(path, dataset_id=self.config.name))
        return records


def build_data_source(config: DataSourceConfig) -> BaseDataSource:
    source_cls = DATA_SOURCE_REGISTRY.get(config.source_type)
    return source_cls(config)

