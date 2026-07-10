from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from collections.abc import Sequence
from typing import Any, Literal

from .dataset import DPORecord, PPORecord, SFTRecord
from .meta import ShaftDatasetMeta
from .record_store import ShaftArrowRecordStore, ShaftConcatRecordStore
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
    image_path = _resolve_optional_image_path(raw, jsonl_path)
    if image_path is None:
        raise ValueError(f"Missing image path in {jsonl_path}:{line_no}. Expected image_path/image/images.")
    return image_path


def _resolve_optional_image_path(raw: dict[str, Any], jsonl_path: Path) -> str | None:
    image_obj = raw.get("image_path")
    if image_obj is None:
        image_obj = raw.get("image")
    if image_obj is None:
        images = raw.get("images")
        if isinstance(images, list) and images:
            image_obj = images[0]
    if image_obj is None:
        return None
    image_path = str(image_obj)
    if not Path(image_path).is_absolute():
        image_path = str((jsonl_path.parent / image_path).resolve())
    return image_path


def _build_sft_record_from_raw(
    raw: dict[str, Any],
    *,
    jsonl_path: Path,
    line_no: int,
    dataset_name: str,
) -> SFTRecord:
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
            "Missing target text. Expected target_text or a trailing assistant message."
        )

    raw_dataset_name = str(raw.get("dataset_name", "")).strip() or None
    extra = {
        k: v
        for k, v in raw.items()
        if k
        not in {
            "image_path",
            "image",
            "images",
            "target_text",
            "messages",
            "conversation",
            "conversations",
            "dataset_name",
            "sample_id",
            "system_prompt",
            "user_prompt",
        }
    }
    if raw_dataset_name is not None and raw_dataset_name != dataset_name:
        extra.setdefault("source_dataset_name", raw_dataset_name)
    return SFTRecord(
        image_path=image_path,
        target_text=str(target_text),
        dataset_name=dataset_name,
        sample_id=str(raw.get("sample_id", "")) or None,
        messages=messages,
        system_prompt=str(raw.get("system_prompt", "")),
        user_prompt=str(
            raw.get("user_prompt", "Output only valid JSON. No markdown and no extra text.")
        ),
        extra=extra,
    )


def _build_dpo_record_from_raw(
    raw: dict[str, Any],
    *,
    jsonl_path: Path,
    line_no: int,
    dataset_name: str,
) -> DPORecord:
    image_path = _resolve_image_path(raw, jsonl_path, line_no)
    messages = _normalize_messages(raw)
    chosen_text = raw.get("chosen_text", raw.get("chosen"))
    rejected_text = raw.get("rejected_text", raw.get("rejected"))
    if chosen_text is None or rejected_text is None:
        raise ValueError("Missing chosen/rejected fields. Expected chosen_text/chosen and rejected_text/rejected.")
    raw_dataset_name = str(raw.get("dataset_name", "")).strip() or None
    extra = {
        k: v
        for k, v in raw.items()
        if k
        not in {
            "image_path",
            "image",
            "images",
            "messages",
            "conversation",
            "conversations",
            "chosen_text",
            "chosen",
            "rejected_text",
            "rejected",
            "dataset_name",
            "sample_id",
            "system_prompt",
            "user_prompt",
        }
    }
    if raw_dataset_name is not None and raw_dataset_name != dataset_name:
        extra.setdefault("source_dataset_name", raw_dataset_name)
    return DPORecord(
        image_path=image_path,
        chosen_text=str(chosen_text),
        rejected_text=str(rejected_text),
        dataset_name=dataset_name,
        sample_id=str(raw.get("sample_id", "")) or None,
        messages=messages,
        system_prompt=str(raw.get("system_prompt", "")),
        user_prompt=str(
            raw.get("user_prompt", "Output only valid JSON. No markdown and no extra text.")
        ),
        extra=extra,
    )


def _build_ppo_record_from_raw(
    raw: dict[str, Any],
    *,
    jsonl_path: Path,
    line_no: int,
    dataset_name: str,
) -> PPORecord:
    image_path = _resolve_optional_image_path(raw, jsonl_path)
    messages = _normalize_messages(raw)
    prompt_text = str(raw.get("prompt", ""))
    user_prompt = str(raw.get("user_prompt", prompt_text))
    if messages is None and not user_prompt.strip():
        raise ValueError("Missing prompt for PPO sample. Expected messages or user_prompt/prompt.")
    if messages is not None and not any(
        _content_to_text(message.get("content", []))
        for message in messages
    ):
        raise ValueError("Current PPO path is text-only; messages must contain text content.")
    raw_dataset_name = str(raw.get("dataset_name", "")).strip() or None
    extra = {
        k: v
        for k, v in raw.items()
        if k
        not in {
            "image_path",
            "image",
            "images",
            "messages",
            "conversation",
            "conversations",
            "prompt",
            "user_prompt",
            "dataset_name",
            "sample_id",
            "system_prompt",
        }
    }
    if raw_dataset_name is not None and raw_dataset_name != dataset_name:
        extra.setdefault("source_dataset_name", raw_dataset_name)
    return PPORecord(
        image_path=image_path,
        dataset_name=dataset_name,
        sample_id=str(raw.get("sample_id", "")) or None,
        messages=messages,
        system_prompt=str(raw.get("system_prompt", "")),
        user_prompt=user_prompt
        or "Output only valid JSON. No markdown and no extra text.",
        extra=extra,
    )


def load_jsonl_sft_records(
    path: str | Path,
    *,
    dataset_name: str,
    max_errors_to_report: int = 20,
    cache_dir: str | Path | None = None,
) -> ShaftArrowRecordStore[SFTRecord]:
    return ShaftArrowRecordStore.from_jsonl(
        path,
        dataset_name=dataset_name,
        record_type=SFTRecord,
        row_builder=_build_sft_record_from_raw,
        max_errors_to_report=max_errors_to_report,
        cache_dir=cache_dir,
    )


def load_jsonl_dpo_records(
    path: str | Path,
    *,
    dataset_name: str,
    max_errors_to_report: int = 20,
    cache_dir: str | Path | None = None,
) -> ShaftArrowRecordStore[DPORecord]:
    return ShaftArrowRecordStore.from_jsonl(
        path,
        dataset_name=dataset_name,
        record_type=DPORecord,
        row_builder=_build_dpo_record_from_raw,
        max_errors_to_report=max_errors_to_report,
        cache_dir=cache_dir,
    )


def load_jsonl_ppo_records(
    path: str | Path,
    *,
    dataset_name: str,
    max_errors_to_report: int = 20,
    cache_dir: str | Path | None = None,
) -> ShaftArrowRecordStore[PPORecord]:
    return ShaftArrowRecordStore.from_jsonl(
        path,
        dataset_name=dataset_name,
        record_type=PPORecord,
        row_builder=_build_ppo_record_from_raw,
        max_errors_to_report=max_errors_to_report,
        cache_dir=cache_dir,
    )


class BaseDataSource(ABC):
    def __init__(self, dataset_meta: ShaftDatasetMeta, *, cache_dir: str | Path | None = None) -> None:
        self.dataset_meta = dataset_meta
        self.cache_dir = cache_dir

    @abstractmethod
    def load_split(self, split: Split) -> Sequence[Any]:
        raise NotImplementedError

    def _resolve_paths(self, split: Split) -> list[str]:
        return list(self.dataset_meta.train_paths if split == "train" else self.dataset_meta.val_paths)


@register_data_source("jsonl_sft")
class JsonlSFTDataSource(BaseDataSource):
    def load_split(self, split: Split) -> Sequence[SFTRecord]:
        return ShaftConcatRecordStore(
            [
                load_jsonl_sft_records(
                    path,
                    dataset_name=self.dataset_meta.dataset_name,
                    cache_dir=self.cache_dir,
                )
                for path in self._resolve_paths(split)
            ]
        )


@register_data_source("jsonl_dpo")
class JsonlDPODataSource(BaseDataSource):
    def load_split(self, split: Split) -> Sequence[DPORecord]:
        return ShaftConcatRecordStore(
            [
                load_jsonl_dpo_records(
                    path,
                    dataset_name=self.dataset_meta.dataset_name,
                    cache_dir=self.cache_dir,
                )
                for path in self._resolve_paths(split)
            ]
        )


@register_data_source("jsonl_ppo")
class JsonlPPODataSource(BaseDataSource):
    def load_split(self, split: Split) -> Sequence[PPORecord]:
        return ShaftConcatRecordStore(
            [
                load_jsonl_ppo_records(
                    path,
                    dataset_name=self.dataset_meta.dataset_name,
                    cache_dir=self.cache_dir,
                )
                for path in self._resolve_paths(split)
            ]
        )


def build_data_source(
    dataset_meta: ShaftDatasetMeta,
    *,
    cache_dir: str | Path | None = None,
) -> BaseDataSource:
    source_cls = DATA_SOURCE_REGISTRY.get(dataset_meta.source_type)
    return source_cls(dataset_meta, cache_dir=cache_dir)
