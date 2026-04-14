from __future__ import annotations

from typing import Any

import torch

from shaft.model import ModelMeta
from shaft.template import Template


class SFTCollator:
    def __init__(
        self,
        *,
        model_meta: ModelMeta,
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
        self.model_meta = model_meta
        self.template = template
        self.processor = processor
        self.tokenizer = tokenizer
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.add_eos_token = bool(add_eos_token)
        self.ignore_index = int(ignore_index)
        self.include_targets_in_inputs = bool(include_targets_in_inputs)
        self.padding_side = padding_side
        if self.padding_side not in {"left", "right"}:
            raise ValueError("padding_side must be 'left' or 'right'.")

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        messages = [self._resolve_messages(item) for item in batch]
        prompt_texts = [self._apply_chat_template(msg) for msg in messages]
        images = [item["image"] for item in batch]
        prefix_batch = self._run_processor(prompt_texts, images)

        target_tokenized = self.tokenizer(
            [item["target_text"] for item in batch],
            add_special_tokens=False,
            return_attention_mask=False,
        )
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else eos_id

        prompt_lengths = prefix_batch["attention_mask"].sum(dim=1).tolist()
        input_ids_rows: list[torch.Tensor] = []
        labels_rows: list[torch.Tensor] = []
        attn_rows: list[torch.Tensor] = []
        mm_rows: list[torch.Tensor] = []

        has_mm_token_type_ids = "mm_token_type_ids" in prefix_batch
        mm_token_ids = prefix_batch.get("mm_token_type_ids")
        for row, prompt_length in enumerate(prompt_lengths):
            prefix_mask = prefix_batch["attention_mask"][row].bool()
            prefix_ids = prefix_batch["input_ids"][row][prefix_mask]
            prefix_mm = mm_token_ids[row][prefix_mask] if has_mm_token_type_ids else None

            if self.include_targets_in_inputs:
                target_ids = list(target_tokenized["input_ids"][row])
                if self.add_eos_token and eos_id is not None and (not target_ids or target_ids[-1] != eos_id):
                    target_ids.append(eos_id)
                target_tensor = torch.tensor(target_ids, dtype=torch.long)
                input_ids = torch.cat([prefix_ids, target_tensor], dim=0)
                labels = torch.cat(
                    [
                        torch.full((prefix_ids.shape[0],), self.ignore_index, dtype=torch.long),
                        target_tensor.clone(),
                    ],
                    dim=0,
                )
                if has_mm_token_type_ids:
                    mm_rows.append(torch.cat([prefix_mm, torch.zeros_like(target_tensor)], dim=0))
            else:
                input_ids = prefix_ids
                labels = torch.full((prompt_length,), self.ignore_index, dtype=torch.long)
                if has_mm_token_type_ids:
                    mm_rows.append(prefix_mm)

            input_ids_rows.append(input_ids)
            labels_rows.append(labels)
            attn_rows.append(torch.ones_like(input_ids))

        out: dict[str, Any] = {
            "input_ids": self._pad_sequences(input_ids_rows, padding_value=pad_id),
            "attention_mask": self._pad_sequences(attn_rows, padding_value=0),
            "labels": self._pad_sequences(labels_rows, padding_value=self.ignore_index),
            "pixel_values": prefix_batch["pixel_values"],
            "image_grid_thw": prefix_batch.get("image_grid_thw"),
            "meta": {
                "dataset_id": [item["dataset_id"] for item in batch],
                "sample_id": [item["sample_id"] for item in batch],
                "image_path": [item["image_path"] for item in batch],
                "target_text": [item["target_text"] for item in batch],
            },
        }
        if has_mm_token_type_ids:
            out["mm_token_type_ids"] = self._pad_sequences(mm_rows, padding_value=0)
        return out

    def _run_processor(self, prompt_texts: list[str], images: list[Any]) -> dict[str, torch.Tensor]:
        return self.model_meta.processor_policy.build_inputs(
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
            messages.append(
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
            )
        messages.append(
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": item["user_prompt"]}],
            }
        )
        return messages

    def _apply_chat_template(self, messages: list[dict[str, Any]]) -> str:
        return self.template.apply_chat_template(
            processor=self.processor,
            tokenizer=self.tokenizer,
            messages=messages,
        )

    def _pad_sequences(self, rows: list[torch.Tensor], *, padding_value: int) -> torch.Tensor:
        max_len = max(int(row.shape[0]) for row in rows)
        padded = []
        for row in rows:
            if int(row.shape[0]) == max_len:
                padded.append(row)
                continue
            pad = torch.full((max_len - int(row.shape[0]),), padding_value, dtype=row.dtype)
            padded.append(torch.cat([pad, row], dim=0) if self.padding_side == "left" else torch.cat([row, pad], dim=0))
        return torch.stack(padded, dim=0)
