from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from shaft.data import ShaftSequenceCollatorBase, ShaftVisionDatasetBase
from shaft.data.record_store import (
    ShaftArrowRecordStore,
    ShaftConcatRecordStore,
    register_record_type,
)
from shaft.data.registry import register_data_source
from shaft.data.sources import BaseDataSource, Split
from shaft.utils.messages import count_message_content_type


def _normalize_content(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, dict):
        kind = str(content.get("type", "")).strip().lower()
        if kind == "image":
            return [{"type": "image"}]
        if kind == "text" or "text" in content:
            return [{"type": "text", "text": str(content.get("text", ""))}]
        return [{"type": "text", "text": str(content)}]
    if isinstance(content, list):
        return [chunk for item in content for chunk in _normalize_content(item)]
    return [{"type": "text", "text": str(content)}]


def _normalize_messages(raw: Any) -> list[dict[str, Any]] | None:
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise TypeError("OPD messages must be a list when provided.")
    messages: list[dict[str, Any]] = []
    for message in raw:
        if not isinstance(message, dict):
            raise TypeError("Each OPD message must be a mapping.")
        role = str(message.get("role", "user")).strip().lower()
        messages.append(
            {"role": role, "content": _normalize_content(message.get("content", ""))}
        )
    return messages


def _resolve_image_paths(raw: dict[str, Any], jsonl_path: Path) -> tuple[str, ...]:
    present = [
        field_name
        for field_name in ("image_path", "image", "images")
        if raw.get(field_name) is not None
    ]
    if len(present) != 1:
        raise ValueError(
            "OPD samples require exactly one of image_path, image, or images."
        )
    field_name = present[0]
    value = raw[field_name]
    if field_name == "images":
        if not isinstance(value, list) or not value:
            raise ValueError("OPD images must be a non-empty ordered path list.")
        raw_paths = value
    else:
        if isinstance(value, (list, tuple, dict)):
            raise ValueError(f"OPD {field_name} must contain one path.")
        raw_paths = [value]
    paths: list[str] = []
    for raw_path in raw_paths:
        path = Path(str(raw_path).strip()).expanduser()
        if not str(raw_path).strip():
            raise ValueError("OPD image paths must not be empty.")
        if not path.is_absolute():
            path = (jsonl_path.parent / path).resolve()
        paths.append(str(path))
    return tuple(paths)


@dataclass
class OPDRecord:
    image_paths: tuple[str, ...]
    dataset_name: str = "default"
    sample_id: str | None = None
    messages: list[dict[str, Any]] | None = None
    system_prompt: str = ""
    user_prompt: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.image_paths = tuple(str(path).strip() for path in self.image_paths)
        if not self.image_paths or any(not path for path in self.image_paths):
            raise ValueError("OPD records require non-empty ordered image_paths.")
        self.dataset_name = str(self.dataset_name)
        self.system_prompt = str(self.system_prompt)
        self.user_prompt = str(self.user_prompt)
        self.extra = dict(self.extra or {})


register_record_type(OPDRecord)


def _build_opd_record_from_raw(
    raw: dict[str, Any],
    *,
    jsonl_path: Path,
    line_no: int,
    dataset_name: str,
) -> OPDRecord:
    image_paths = _resolve_image_paths(raw, jsonl_path)
    messages = _normalize_messages(
        raw.get("messages", raw.get("conversation", raw.get("conversations")))
    )
    user_prompt = str(raw.get("user_prompt", raw.get("prompt", "")))
    if messages:
        if str(messages[-1].get("role", "")).strip().lower() == "assistant":
            raise ValueError(
                "OPD records are prompt-only and cannot end with an assistant target."
            )
        placeholder_count = count_message_content_type(messages, "image")
        if placeholder_count != len(image_paths):
            raise ValueError(
                "OPD message image placeholders must match ordered image paths in "
                f"{jsonl_path}:{line_no}: placeholders={placeholder_count}, "
                f"images={len(image_paths)}."
            )
    elif not user_prompt.strip():
        raise ValueError("OPD records require messages or a non-empty user_prompt/prompt.")

    excluded = {
        "image_path",
        "image",
        "images",
        "messages",
        "conversation",
        "conversations",
        "prompt",
        "user_prompt",
        "system_prompt",
        "dataset_name",
        "sample_id",
    }
    return OPDRecord(
        image_paths=image_paths,
        dataset_name=dataset_name,
        sample_id=str(raw.get("sample_id", "")).strip() or None,
        messages=messages,
        system_prompt=str(raw.get("system_prompt", "")),
        user_prompt=user_prompt,
        extra={key: value for key, value in raw.items() if key not in excluded},
    )


def load_jsonl_opd_records(
    path: str | Path,
    *,
    dataset_name: str,
    max_errors_to_report: int = 20,
    cache_dir: str | Path | None = None,
) -> ShaftArrowRecordStore[OPDRecord]:
    return ShaftArrowRecordStore.from_jsonl(
        path,
        dataset_name=dataset_name,
        record_type=OPDRecord,
        row_builder=_build_opd_record_from_raw,
        max_errors_to_report=max_errors_to_report,
        cache_dir=cache_dir,
    )


@register_data_source("jsonl_opd")
class JsonlOPDDataSource(BaseDataSource):
    def load_split(self, split: Split) -> Sequence[OPDRecord]:
        return ShaftConcatRecordStore(
            [
                load_jsonl_opd_records(
                    path,
                    dataset_name=self.dataset_meta.dataset_name,
                    cache_dir=self.cache_dir,
                )
                for path in self._resolve_paths(split)
            ]
        )


class OPDDataset(ShaftVisionDatasetBase):
    def __getitem__(self, index):
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
            "extra": dict(record.extra),
            **self._runtime_context(sample_ref),
        }
        sample = self._apply_online_transforms(sample)
        return self._attach_batch_context(sample, index)


class OPDCollator(ShaftSequenceCollatorBase):
    SHAFT_INPUT_POLICY_VERSION = "shaft-opd-collator-input-v3-dual-rollout-prompts"
    DEFAULT_INPUT_MODE = "generation"

    def __init__(
        self,
        *args: Any,
        max_prompt_length: int,
        retain_rollout_media: bool = False,
        collect_telemetry: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, input_mode="generation", **kwargs)
        self.max_prompt_length = int(max_prompt_length)
        self.retain_rollout_media = bool(retain_rollout_media)
        self.collect_telemetry = bool(collect_telemetry)
        if self.max_prompt_length <= 0:
            raise ValueError("OPD max_prompt_length must be > 0.")

    @staticmethod
    def _request_id(item: dict[str, Any], *, row_index: int) -> str:
        context = item.get("_sample_context")
        payload = {
            "version": "shaft-opd-request-id-v1",
            "dataset_name": str(item.get("dataset_name", "")),
            "sample_id": str(item.get("sample_id", "")),
            "sample_context": dict(context) if isinstance(context, dict) else None,
            "fallback_row_index": row_index if not isinstance(context, dict) else None,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha256(encoded).hexdigest()

    def _generation_prompt_token_ids(
        self,
        prompt_texts: list[str],
    ) -> list[tuple[int, ...]]:
        """Tokenize rendered prompts without expanding multimodal placeholders.

        External multimodal engines own media preprocessing and expand one placeholder
        per media item. The local processor output remains the scoring truth and is
        checked against the engine's expanded prompt after generation.
        """
        encoded = self.tokenizer(
            prompt_texts,
            add_special_tokens=False,
        )
        rows = encoded["input_ids"]
        if len(rows) != len(prompt_texts):
            raise ValueError("Tokenizer changed OPD rollout prompt batch cardinality.")
        normalized = [tuple(int(value) for value in row) for row in rows]
        if any(not row for row in normalized):
            raise ValueError("Tokenizer produced an empty OPD rollout prompt.")
        return normalized

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        plans = [
            self.template.build_prompt_plan(item=item, renderer=self.chat_renderer)
            for item in batch
        ]
        prompt_texts = [plan.prompt_text for plan in plans]
        processed = self._run_processor(
            prompt_texts,
            self._processor_image_rows(batch),
            dataset_names=[item.get("dataset_name") for item in batch],
        )
        prefix_token_layouts = self._build_prefix_token_layouts(
            plans=plans,
            processed_batch=processed,
            max_length=self.max_prompt_length,
        )
        rows = [
            self.template.build_prompt_row(
                plan=plan,
                tokenizer=self.tokenizer,
                processed_batch=processed,
                row_index=row_index,
                prefix_token_layout=prefix_token_layout,
                max_length=self.max_prompt_length,
            )
            for row_index, (plan, prefix_token_layout) in enumerate(
                zip(plans, prefix_token_layouts)
            )
        ]
        eos_id = self.tokenizer.eos_token_id
        pad_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else eos_id
        )
        sequence_inputs: dict[str, Any] = {
            "input_ids": self._pad_sequences(
                [row.input_ids for row in rows],
                padding_value=int(pad_id),
            ),
            "attention_mask": self._pad_sequences(
                [row.attention_mask for row in rows],
                padding_value=0,
            ),
        }
        processor_sequence_rows = self._build_processor_sequence_rows(
            processed_batch=processed,
            rows=rows,
            processor_row_indices=tuple(range(len(rows))),
        )
        sequence_inputs.update(
            processed.collate_processor_sequence_rows(
                processor_sequence_rows,
                layout="padded",
                padding_side=self.padding_side,
            )
        )
        output = self.model_adapter.assemble_processor_training_inputs(
            processed_batch=processed,
            sequence_inputs=sequence_inputs,
            row_indices=tuple(range(len(batch))),
        )
        output["_shaft_sample_ids"] = [str(item.get("sample_id", "")) for item in batch]
        output["_shaft_rollout_request_ids"] = [
            self._request_id(item, row_index=row_index)
            for row_index, item in enumerate(batch)
        ]
        output["_shaft_rollout_prompt_ids"] = [
            tuple(
                int(value)
                for value in output["input_ids"][row_index][
                    output["attention_mask"][row_index].to(dtype=torch.bool)
                ].tolist()
            )
            for row_index in range(len(batch))
        ]
        if self.retain_rollout_media:
            image_rows = self._processor_image_rows(batch)
            output["_shaft_rollout_images"] = [
                tuple(row) if isinstance(row, (list, tuple)) else (row,)
                for row in image_rows
            ]
            output["_shaft_rollout_generation_prompt_ids"] = (
                self._generation_prompt_token_ids(prompt_texts)
            )
        if self.collect_telemetry:
            media_manifest = processed.media_manifest
            output["_shaft_opd_batch_stats"] = {
                "prompt_tokens": int(output["attention_mask"].sum().item()),
                "materialized_prompt_tokens": int(output["attention_mask"].numel()),
                "vision_patches": (
                    0 if media_manifest is None else int(media_manifest.image_patch_count)
                ),
            }
        return output
