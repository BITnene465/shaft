from __future__ import annotations

from typing import Any

import torch

from shaft.model import ShaftModelAdapter
from shaft.template import Template


class _ShaftSequenceCollatorBase:
    def __init__(
        self,
        *,
        model_adapter: ShaftModelAdapter,
        template: Template,
        processor: Any,
        tokenizer: Any,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        add_eos_token: bool = True,
        ignore_index: int = -100,
        padding_side: str = "right",
    ) -> None:
        self.model_adapter = model_adapter
        self.template = template
        self.processor = processor
        self.tokenizer = tokenizer
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.add_eos_token = bool(add_eos_token)
        self.ignore_index = int(ignore_index)
        self.padding_side = padding_side
        if self.padding_side not in {"left", "right"}:
            raise ValueError("padding_side must be 'left' or 'right'.")

    def _run_processor(self, prompt_texts: list[str], images: list[Any]) -> dict[str, torch.Tensor]:
        return self.model_adapter.build_processor_inputs(
            processor=self.processor,
            prompt_texts=prompt_texts,
            images=images,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )

    def _resolve_messages(self, item: dict[str, Any]) -> list[dict[str, Any]]:
        if item.get("messages"):
            return item["messages"]
        messages: list[dict[str, Any]] = []
        system_prompt = str(item.get("system_prompt", "")).strip()
        if system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        messages.append(
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": str(item.get("user_prompt", ""))}],
            }
        )
        return messages

    def _apply_chat_template(self, messages: list[dict[str, Any]]) -> str:
        return self.template.apply_chat_template(
            processor=self.processor,
            tokenizer=self.tokenizer,
            messages=messages,
        )

    def _tokenize_targets(self, texts: list[str]) -> list[list[int]]:
        tokenized = self.tokenizer(
            texts,
            add_special_tokens=False,
            return_attention_mask=False,
        )
        return [list(row) for row in tokenized["input_ids"]]

    def _build_supervised_rows(
        self,
        *,
        prefix_batch: dict[str, torch.Tensor],
        target_token_ids: list[list[int]],
        include_targets_in_inputs: bool,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor] | None, list[torch.Tensor]]:
        eos_id = self.tokenizer.eos_token_id
        prompt_lengths = prefix_batch["attention_mask"].sum(dim=1).tolist()
        has_mm_token_type_ids = "mm_token_type_ids" in prefix_batch
        mm_token_ids = prefix_batch.get("mm_token_type_ids")

        input_ids_rows: list[torch.Tensor] = []
        labels_rows: list[torch.Tensor] = []
        attn_rows: list[torch.Tensor] = []
        response_mask_rows: list[torch.Tensor] = []
        mm_rows: list[torch.Tensor] | None = [] if has_mm_token_type_ids else None

        for row_index, prompt_length in enumerate(prompt_lengths):
            prefix_mask = prefix_batch["attention_mask"][row_index].bool()
            prefix_ids = prefix_batch["input_ids"][row_index][prefix_mask]
            prefix_mm = mm_token_ids[row_index][prefix_mask] if has_mm_token_type_ids else None
            target_ids = list(target_token_ids[row_index])
            if self.add_eos_token and eos_id is not None and (not target_ids or target_ids[-1] != eos_id):
                target_ids.append(int(eos_id))
            target_tensor = torch.tensor(target_ids, dtype=torch.long)

            if include_targets_in_inputs:
                input_ids = torch.cat([prefix_ids, target_tensor], dim=0)
                labels = torch.cat(
                    [
                        torch.full((prefix_ids.shape[0],), self.ignore_index, dtype=torch.long),
                        target_tensor.clone(),
                    ],
                    dim=0,
                )
                response_mask = torch.cat(
                    [
                        torch.zeros((prefix_ids.shape[0],), dtype=torch.bool),
                        torch.ones((target_tensor.shape[0],), dtype=torch.bool),
                    ],
                    dim=0,
                )
                if mm_rows is not None:
                    assert prefix_mm is not None
                    mm_rows.append(torch.cat([prefix_mm, torch.zeros_like(target_tensor)], dim=0))
            else:
                input_ids = prefix_ids
                labels = torch.full((prompt_length,), self.ignore_index, dtype=torch.long)
                response_mask = torch.zeros((prompt_length,), dtype=torch.bool)
                if mm_rows is not None:
                    assert prefix_mm is not None
                    mm_rows.append(prefix_mm)

            input_ids_rows.append(input_ids)
            labels_rows.append(labels)
            attn_rows.append(torch.ones_like(input_ids))
            response_mask_rows.append(response_mask)

        return input_ids_rows, labels_rows, attn_rows, mm_rows, response_mask_rows

    def _pad_sequences(self, rows: list[torch.Tensor], *, padding_value: int) -> torch.Tensor:
        max_len = max(int(row.shape[0]) for row in rows)
        padded = []
        for row in rows:
            if int(row.shape[0]) == max_len:
                padded.append(row)
                continue
            pad = torch.full((max_len - int(row.shape[0]),), padding_value, dtype=row.dtype)
            if self.padding_side == "left":
                padded.append(torch.cat([pad, row], dim=0))
            else:
                padded.append(torch.cat([row, pad], dim=0))
        return torch.stack(padded, dim=0)

    def _repeat_on_batch_axis(self, value: Any, *, repeats: int) -> Any:
        if value is None:
            return None
        if torch.is_tensor(value):
            return torch.cat([value] * repeats, dim=0)
        if isinstance(value, list):
            return value * repeats
        if isinstance(value, tuple):
            return value * repeats
        return value


class SFTCollator(_ShaftSequenceCollatorBase):
    def __init__(
        self,
        *,
        model_adapter: ShaftModelAdapter,
        template: Template,
        processor: Any,
        tokenizer: Any,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        add_eos_token: bool = True,
        ignore_index: int = -100,
        include_targets_in_inputs: bool = True,
        padding_side: str = "right",
    ) -> None:
        super().__init__(
            model_adapter=model_adapter,
            template=template,
            processor=processor,
            tokenizer=tokenizer,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            add_eos_token=add_eos_token,
            ignore_index=ignore_index,
            padding_side=padding_side,
        )
        self.include_targets_in_inputs = bool(include_targets_in_inputs)

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        messages = [self._resolve_messages(item) for item in batch]
        prompt_texts = [self._apply_chat_template(msg) for msg in messages]
        images = [item["image"] for item in batch]
        prefix_batch = self._run_processor(prompt_texts, images)
        target_token_ids = self._tokenize_targets([str(item["target_text"]) for item in batch])
        input_rows, label_rows, attn_rows, mm_rows, _ = self._build_supervised_rows(
            prefix_batch=prefix_batch,
            target_token_ids=target_token_ids,
            include_targets_in_inputs=self.include_targets_in_inputs,
        )
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else eos_id
        out: dict[str, Any] = {
            "input_ids": self._pad_sequences(input_rows, padding_value=int(pad_id)),
            "attention_mask": self._pad_sequences(attn_rows, padding_value=0),
            "labels": self._pad_sequences(label_rows, padding_value=self.ignore_index),
            "pixel_values": prefix_batch["pixel_values"],
            "image_grid_thw": prefix_batch.get("image_grid_thw"),
            "meta": {
                "dataset_name": [item["dataset_name"] for item in batch],
                "sample_id": [item["sample_id"] for item in batch],
                "image_path": [item["image_path"] for item in batch],
                "target_text": [item["target_text"] for item in batch],
            },
        }
        if mm_rows is not None:
            out["mm_token_type_ids"] = self._pad_sequences(mm_rows, padding_value=0)
        return out


class DPOCollator(_ShaftSequenceCollatorBase):
    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        messages = [self._resolve_messages(item) for item in batch]
        prompt_texts = [self._apply_chat_template(msg) for msg in messages]
        images = [item["image"] for item in batch]
        prefix_batch = self._run_processor(prompt_texts, images)
        chosen_token_ids = self._tokenize_targets([str(item["chosen_text"]) for item in batch])
        rejected_token_ids = self._tokenize_targets([str(item["rejected_text"]) for item in batch])

        chosen_rows = self._build_supervised_rows(
            prefix_batch=prefix_batch,
            target_token_ids=chosen_token_ids,
            include_targets_in_inputs=True,
        )
        rejected_rows = self._build_supervised_rows(
            prefix_batch=prefix_batch,
            target_token_ids=rejected_token_ids,
            include_targets_in_inputs=True,
        )

        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else eos_id
        chosen_inputs, _, chosen_attn, chosen_mm_rows, chosen_completion_rows = chosen_rows
        rejected_inputs, _, rejected_attn, rejected_mm_rows, rejected_completion_rows = rejected_rows
        input_rows = [*chosen_inputs, *rejected_inputs]
        attention_rows = [*chosen_attn, *rejected_attn]
        completion_rows = [*chosen_completion_rows, *rejected_completion_rows]
        out: dict[str, Any] = {
            "input_ids": self._pad_sequences(input_rows, padding_value=int(pad_id)),
            "attention_mask": self._pad_sequences(attention_rows, padding_value=0),
            "completion_mask": self._pad_sequences(
                [row.to(dtype=torch.long) for row in completion_rows],
                padding_value=0,
            ),
            "meta": {
                "dataset_name": [item["dataset_name"] for item in batch],
                "sample_id": [item["sample_id"] for item in batch],
                "image_path": [item["image_path"] for item in batch],
            },
        }
        for key in ("pixel_values", "pixel_attention_mask", "image_grid_thw", "image_sizes"):
            if key in prefix_batch:
                out[key] = self._repeat_on_batch_axis(prefix_batch[key], repeats=2)
        if chosen_mm_rows is not None and rejected_mm_rows is not None:
            out["mm_token_type_ids"] = self._pad_sequences(
                [*chosen_mm_rows, *rejected_mm_rows],
                padding_value=0,
            )
        return out


class PPOCollator(_ShaftSequenceCollatorBase):
    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        messages = [self._resolve_messages(item) for item in batch]
        prompt_texts = [self._apply_chat_template(msg) for msg in messages]
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else eos_id
        prompt_token_ids = self._tokenize_targets(prompt_texts)
        input_rows = [torch.tensor(row, dtype=torch.long) for row in prompt_token_ids]
        attention_rows = [torch.ones_like(row) for row in input_rows]
        out: dict[str, Any] = {
            "input_ids": self._pad_sequences(input_rows, padding_value=int(pad_id)),
            "attention_mask": self._pad_sequences(attention_rows, padding_value=0),
            "meta": {
                "dataset_name": [item["dataset_name"] for item in batch],
                "sample_id": [item["sample_id"] for item in batch],
                "image_path": [item["image_path"] for item in batch],
            },
        }
        return out
