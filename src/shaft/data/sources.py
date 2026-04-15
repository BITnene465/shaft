from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

from shaft.config import DataSourceConfig

from .dataset import DPORecord, PPORecord, SFTRecord
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


def _build_sft_record_from_raw(
    raw: dict[str, Any],
    *,
    jsonl_path: Path,
    line_no: int,
    dataset_id: str,
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

    extra = {
        k: v
        for k, v in raw.items()
        if k not in {"image_path", "image", "images", "target_text", "messages", "conversation", "conversations"}
    }
    return SFTRecord(
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


def _build_dpo_record_from_raw(
    raw: dict[str, Any],
    *,
    jsonl_path: Path,
    line_no: int,
    dataset_id: str,
) -> DPORecord:
    image_path = _resolve_image_path(raw, jsonl_path, line_no)
    messages = _normalize_messages(raw)
    chosen_text = raw.get("chosen_text", raw.get("chosen"))
    rejected_text = raw.get("rejected_text", raw.get("rejected"))
    if chosen_text is None or rejected_text is None:
        raise ValueError("Missing chosen/rejected fields. Expected chosen_text/chosen and rejected_text/rejected.")
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
        }
    }
    return DPORecord(
        image_path=image_path,
        chosen_text=str(chosen_text),
        rejected_text=str(rejected_text),
        dataset_id=str(raw.get("dataset_id", dataset_id)),
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
    dataset_id: str,
) -> PPORecord:
    image_path = _resolve_image_path(raw, jsonl_path, line_no)
    messages = _normalize_messages(raw)
    prompt_text = str(raw.get("prompt", ""))
    user_prompt = str(raw.get("user_prompt", prompt_text))
    if messages is None and not user_prompt.strip():
        raise ValueError("Missing prompt for PPO sample. Expected messages or user_prompt/prompt.")
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
        }
    }
    return PPORecord(
        image_path=image_path,
        dataset_id=str(raw.get("dataset_id", dataset_id)),
        sample_id=str(raw.get("sample_id", "")) or None,
        messages=messages,
        system_prompt=str(raw.get("system_prompt", "")),
        user_prompt=user_prompt
        or "Output only valid JSON. No markdown and no extra text.",
        extra=extra,
    )


def _format_error_examples(
    path: Path,
    *,
    total_errors: int,
    errors: list[tuple[int, str]],
) -> str:
    snippets = [f"L{line_no}: {message}" for line_no, message in errors]
    detail = "; ".join(snippets)
    return (
        f"Failed to parse {total_errors} row(s) in {path}. "
        f"Examples: {detail}"
    )


def _load_jsonl_records_with_builder(
    path: str | Path,
    *,
    dataset_id: str,
    row_builder,
    max_errors_to_report: int = 20,
) -> list[Any]:
    records: list[Any] = []
    jsonl_path = Path(path)
    parse_errors: list[tuple[int, str]] = []
    total_parse_errors = 0
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                raw = json.loads(text)
                if not isinstance(raw, dict):
                    raise TypeError("Each JSONL row must be a JSON object.")
                records.append(
                    row_builder(
                        raw,
                        jsonl_path=jsonl_path,
                        line_no=line_no,
                        dataset_id=dataset_id,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                total_parse_errors += 1
                if len(parse_errors) < max(1, int(max_errors_to_report)):
                    parse_errors.append((line_no, str(exc)))
    if total_parse_errors > 0:
        raise ValueError(
            _format_error_examples(
                jsonl_path,
                total_errors=total_parse_errors,
                errors=parse_errors,
            )
        )
    return records


def load_jsonl_records(
    path: str | Path,
    *,
    dataset_id: str,
    max_errors_to_report: int = 20,
) -> list[SFTRecord]:
    return _load_jsonl_records_with_builder(
        path,
        dataset_id=dataset_id,
        row_builder=_build_sft_record_from_raw,
        max_errors_to_report=max_errors_to_report,
    )


def load_jsonl_dpo_records(
    path: str | Path,
    *,
    dataset_id: str,
    max_errors_to_report: int = 20,
) -> list[DPORecord]:
    return _load_jsonl_records_with_builder(
        path,
        dataset_id=dataset_id,
        row_builder=_build_dpo_record_from_raw,
        max_errors_to_report=max_errors_to_report,
    )


def load_jsonl_ppo_records(
    path: str | Path,
    *,
    dataset_id: str,
    max_errors_to_report: int = 20,
) -> list[PPORecord]:
    return _load_jsonl_records_with_builder(
        path,
        dataset_id=dataset_id,
        row_builder=_build_ppo_record_from_raw,
        max_errors_to_report=max_errors_to_report,
    )


class BaseDataSource(ABC):
    def __init__(self, config: DataSourceConfig) -> None:
        self.config = config

    @abstractmethod
    def load_split(self, split: Split) -> list[Any]:
        raise NotImplementedError

    def _resolve_paths(self, split: Split) -> list[str]:
        paths = list(self.config.train_paths if split == "train" else self.config.val_paths)
        single_path = self.config.train_path if split == "train" else self.config.val_path
        if single_path:
            normalized_single = str(single_path).strip()
            if normalized_single and normalized_single not in paths:
                paths = [normalized_single, *paths]
        return paths


@register_data_source("jsonl_sft")
class JsonlSFTDataSource(BaseDataSource):
    def load_split(self, split: Split) -> list[SFTRecord]:
        records: list[SFTRecord] = []
        for path in self._resolve_paths(split):
            records.extend(load_jsonl_records(path, dataset_id=self.config.name))
        return records


@register_data_source("jsonl_dpo")
class JsonlDPODataSource(BaseDataSource):
    def load_split(self, split: Split) -> list[DPORecord]:
        records: list[DPORecord] = []
        for path in self._resolve_paths(split):
            records.extend(load_jsonl_dpo_records(path, dataset_id=self.config.name))
        return records


@register_data_source("jsonl_ppo")
class JsonlPPODataSource(BaseDataSource):
    def load_split(self, split: Split) -> list[PPORecord]:
        records: list[PPORecord] = []
        for path in self._resolve_paths(split):
            records.extend(load_jsonl_ppo_records(path, dataset_id=self.config.name))
        return records


def build_data_source(config: DataSourceConfig) -> BaseDataSource:
    source_cls = DATA_SOURCE_REGISTRY.get(config.source_type)
    return source_cls(config)
