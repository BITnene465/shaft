from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from collections.abc import Sequence
from typing import Any, Callable, Literal

from shaft.utils.messages import count_message_content_type

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
        normalized_message: dict[str, Any] = {"role": role, "content": content}
        if "reasoning_content" in message and message["reasoning_content"] is not None:
            reasoning_content = message["reasoning_content"]
            if not isinstance(reasoning_content, str):
                raise TypeError("Message reasoning_content must be a string when provided.")
            normalized_message["reasoning_content"] = reasoning_content
        normalized.append(normalized_message)
    return normalized


def _content_to_text(content: list[dict[str, Any]]) -> str:
    texts = [str(item.get("text", "")) for item in content if item.get("type") == "text"]
    return "".join(texts).strip()


def _extract_target_from_messages(
    messages: list[dict[str, Any]],
) -> tuple[str | None, str | None, list[dict[str, Any]]]:
    if not messages:
        return None, None, messages
    last = messages[-1]
    if str(last.get("role", "")).strip().lower() != "assistant":
        return None, None, messages
    target_text = _content_to_text(last.get("content", []))
    target_reasoning_content = last.get("reasoning_content")
    return target_text, target_reasoning_content, messages[:-1]


def _resolve_image_paths(
    raw: dict[str, Any],
    jsonl_path: Path,
    line_no: int,
) -> tuple[str, ...]:
    image_paths = _resolve_optional_image_paths(raw, jsonl_path)
    if not image_paths:
        raise ValueError(f"Missing image path in {jsonl_path}:{line_no}. Expected image_path/image/images.")
    return image_paths


def _resolve_optional_image_paths(
    raw: dict[str, Any],
    jsonl_path: Path,
) -> tuple[str, ...]:
    present = [
        field_name
        for field_name in ("image_path", "image", "images")
        if raw.get(field_name) is not None
    ]
    if len(present) > 1:
        raise ValueError(
            "Provide exactly one of image_path, image, or images; "
            f"received {present}."
        )
    if not present:
        return ()
    field_name = present[0]
    value = raw[field_name]
    if field_name == "images":
        if not isinstance(value, list) or not value:
            raise ValueError("`images` must be a non-empty ordered list of paths.")
        raw_paths = value
    else:
        if isinstance(value, (list, tuple, dict)):
            raise ValueError(f"`{field_name}` must be one path; use `images` for multiple paths.")
        raw_paths = [value]

    resolved: list[str] = []
    for index, raw_path in enumerate(raw_paths):
        if not isinstance(raw_path, (str, Path)) or not str(raw_path).strip():
            raise ValueError(f"Invalid image path at {field_name}[{index}].")
        image_path = Path(str(raw_path).strip()).expanduser()
        if not image_path.is_absolute():
            image_path = (jsonl_path.parent / image_path).resolve()
        resolved.append(str(image_path))
    return tuple(resolved)


def _validate_message_image_count(
    messages: list[dict[str, Any]] | None,
    *,
    image_paths: tuple[str, ...],
    jsonl_path: Path,
    line_no: int,
) -> None:
    if not messages:
        return
    placeholder_count = count_message_content_type(messages, "image")
    if placeholder_count != len(image_paths):
        raise ValueError(
            "The message image placeholder count must match ordered image paths in "
            f"{jsonl_path}:{line_no}: placeholders={placeholder_count}, "
            f"images={len(image_paths)}."
        )


def _build_sft_record_from_raw(
    raw: dict[str, Any],
    *,
    jsonl_path: Path,
    line_no: int,
    dataset_name: str,
) -> SFTRecord:
    if "formulation_targets" in raw:
        raise ValueError(
            "SFT JSONL must keep one materialized target_text per row; configure aligned "
            "data.prompt_sources[*].formulation_sources instead of embedding "
            "formulation_targets."
        )
    image_paths = _resolve_image_paths(raw, jsonl_path, line_no)
    messages = _normalize_messages(raw)
    raw_prompt_args = raw.get("prompt_args")
    prompt_args = {} if raw_prompt_args is None else raw_prompt_args
    if not isinstance(prompt_args, dict):
        raise ValueError("`prompt_args` must be a JSON object when provided.")
    target_text = raw.get("target_text")
    target_reasoning_content = raw.get("target_reasoning_content")
    if target_reasoning_content is not None and not isinstance(
        target_reasoning_content,
        str,
    ):
        raise ValueError("`target_reasoning_content` must be a string when provided.")
    if target_text is None and messages is not None:
        extracted_target, extracted_reasoning, prompt_messages = (
            _extract_target_from_messages(messages)
        )
        if extracted_target is not None:
            target_text = extracted_target
            if target_reasoning_content is None:
                target_reasoning_content = extracted_reasoning
            messages = prompt_messages
    if target_text is None:
        raise ValueError(
            "Missing target text. Expected target_text or a trailing assistant message. "
            "prompt_args can render prompts but cannot generate targets."
        )
    _validate_message_image_count(
        messages,
        image_paths=image_paths,
        jsonl_path=jsonl_path,
        line_no=line_no,
    )

    if messages and prompt_args:
        raise ValueError("SFT samples cannot provide both `messages` and non-empty `prompt_args`.")

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
            "target_reasoning_content",
            "messages",
            "conversation",
            "conversations",
            "dataset_name",
            "sample_id",
            "system_prompt",
            "user_prompt",
            "prompt_args",
        }
    }
    if raw_dataset_name is not None and raw_dataset_name != dataset_name:
        extra.setdefault("source_dataset_name", raw_dataset_name)
    return SFTRecord(
        image_paths=image_paths,
        target_text="" if target_text is None else str(target_text),
        target_reasoning_content=target_reasoning_content,
        dataset_name=dataset_name,
        sample_id=str(raw.get("sample_id", "")) or None,
        messages=messages,
        system_prompt=str(raw.get("system_prompt", "")),
        user_prompt=str(raw.get("user_prompt", "")),
        prompt_args=dict(prompt_args),
        extra=extra,
    )


def _build_dpo_record_from_raw(
    raw: dict[str, Any],
    *,
    jsonl_path: Path,
    line_no: int,
    dataset_name: str,
) -> DPORecord:
    image_paths = _resolve_image_paths(raw, jsonl_path, line_no)
    messages = _normalize_messages(raw)
    chosen_text = raw.get("chosen_text", raw.get("chosen"))
    rejected_text = raw.get("rejected_text", raw.get("rejected"))
    if chosen_text is None or rejected_text is None:
        raise ValueError("Missing chosen/rejected fields. Expected chosen_text/chosen and rejected_text/rejected.")
    _validate_message_image_count(
        messages,
        image_paths=image_paths,
        jsonl_path=jsonl_path,
        line_no=line_no,
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
        image_paths=image_paths,
        chosen_text=str(chosen_text),
        rejected_text=str(rejected_text),
        dataset_name=dataset_name,
        sample_id=str(raw.get("sample_id", "")) or None,
        messages=messages,
        system_prompt=str(raw.get("system_prompt", "")),
        user_prompt=str(raw.get("user_prompt", "")),
        extra=extra,
    )


def _build_ppo_record_from_raw(
    raw: dict[str, Any],
    *,
    jsonl_path: Path,
    line_no: int,
    dataset_name: str,
) -> PPORecord:
    image_paths = _resolve_optional_image_paths(raw, jsonl_path)
    messages = _normalize_messages(raw)
    prompt_text = str(raw.get("prompt", ""))
    user_prompt = str(raw.get("user_prompt", prompt_text))
    if messages is None and not user_prompt.strip():
        raise ValueError("Missing prompt for PPO sample. Expected messages or user_prompt/prompt.")
    _validate_message_image_count(
        messages,
        image_paths=image_paths,
        jsonl_path=jsonl_path,
        line_no=line_no,
    )
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
        image_paths=image_paths,
        dataset_name=dataset_name,
        sample_id=str(raw.get("sample_id", "")) or None,
        messages=messages,
        system_prompt=str(raw.get("system_prompt", "")),
        user_prompt=user_prompt,
        extra=extra,
    )


def load_jsonl_sft_records(
    path: str | Path,
    *,
    dataset_name: str,
    max_errors_to_report: int = 20,
    cache_dir: str | Path | None = None,
    record_validator: Callable[[SFTRecord], None] | None = None,
    validation_fingerprint: str = "",
) -> ShaftArrowRecordStore[SFTRecord]:
    return ShaftArrowRecordStore.from_jsonl(
        path,
        dataset_name=dataset_name,
        record_type=SFTRecord,
        row_builder=_build_sft_record_from_raw,
        max_errors_to_report=max_errors_to_report,
        cache_dir=cache_dir,
        record_validator=record_validator,
        validation_fingerprint=validation_fingerprint,
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
    def __init__(
        self,
        dataset_meta: ShaftDatasetMeta,
        *,
        cache_dir: str | Path | None = None,
        record_validator: Callable[[Any, Split], None] | None = None,
        validation_fingerprint: str = "",
    ) -> None:
        self.dataset_meta = dataset_meta
        self.cache_dir = cache_dir
        self.record_validator = record_validator
        self.validation_fingerprint = str(validation_fingerprint)

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
                    record_validator=(
                        None
                        if self.record_validator is None
                        else lambda record: self.record_validator(record, split)
                    ),
                    validation_fingerprint=(
                        f"{self.validation_fingerprint}:{split}"
                        if self.validation_fingerprint
                        else ""
                    ),
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
    record_validator: Callable[[Any, Split], None] | None = None,
    validation_fingerprint: str = "",
) -> BaseDataSource:
    source_cls = DATA_SOURCE_REGISTRY.get(dataset_meta.source_type)
    return source_cls(
        dataset_meta,
        cache_dir=cache_dir,
        record_validator=record_validator,
        validation_fingerprint=validation_fingerprint,
    )
