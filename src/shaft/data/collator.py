from __future__ import annotations

from typing import Any

import torch

from shaft.model import ShaftModelAdapter
from shaft.template import ShaftTemplateSupervisedRow, Template


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
        max_length: int | None = None,
        add_eos_token: bool = True,
        ignore_index: int = -100,
        padding_side: str = "right",
        loss_scale_name: str = "default",
    ) -> None:
        self.model_adapter = model_adapter
        self.template = template
        self.processor = processor
        self.tokenizer = tokenizer
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.max_length = int(max_length) if max_length is not None else None
        self.add_eos_token = bool(add_eos_token)
        self.ignore_index = int(ignore_index)
        self.padding_side = padding_side
        self.loss_scale_name = str(loss_scale_name).strip().lower() or "default"
        if self.padding_side not in {"left", "right"}:
            raise ValueError("padding_side must be 'left' or 'right'.")

    def _run_processor(self, prompt_texts: list[str], images: list[Any]) -> dict[str, torch.Tensor]:
        return self.model_adapter.build_processor_inputs(
            processor=self.processor,
            tokenizer=self.tokenizer,
            prompt_texts=prompt_texts,
            images=images,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
            padding_side=self.padding_side,
        )

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
        max_length: int | None = None,
        add_eos_token: bool = True,
        ignore_index: int = -100,
        include_targets_in_inputs: bool = True,
        include_metadata: bool = False,
        padding_side: str = "right",
        loss_scale_name: str = "default",
    ) -> None:
        super().__init__(
            model_adapter=model_adapter,
            template=template,
            processor=processor,
            tokenizer=tokenizer,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            max_length=max_length,
            add_eos_token=add_eos_token,
            ignore_index=ignore_index,
            padding_side=padding_side,
            loss_scale_name=loss_scale_name,
        )
        self.include_targets_in_inputs = bool(include_targets_in_inputs)
        self.include_metadata = bool(include_metadata)

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        plans = [
            self.template.build_supervision_plan(
                item=item,
                target_text=str(item["target_text"]),
                processor=self.processor,
                tokenizer=self.tokenizer,
                loss_scale_name=self.loss_scale_name,
            )
            for item in batch
        ]
        prompt_texts = [plan.prompt_text for plan in plans]
        images = [item["image"] for item in batch]
        prefix_batch = self._run_processor(prompt_texts, images)
        rows: list[ShaftTemplateSupervisedRow] = [
            self.template.build_supervised_row(
                plan=plan,
                model_adapter=self.model_adapter,
                processor=self.processor,
                tokenizer=self.tokenizer,
                image=item["image"],
                prefix_batch=prefix_batch,
                row_index=row_index,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
                add_eos_token=self.add_eos_token,
                ignore_index=self.ignore_index,
                include_targets_in_inputs=self.include_targets_in_inputs,
                max_length=self.max_length,
            )
            for row_index, (item, plan) in enumerate(zip(batch, plans))
        ]
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else eos_id
        out: dict[str, Any] = {
            "input_ids": self._pad_sequences([row.input_ids for row in rows], padding_value=int(pad_id)),
            "attention_mask": self._pad_sequences([row.attention_mask for row in rows], padding_value=0),
            "labels": self._pad_sequences([row.labels for row in rows], padding_value=self.ignore_index),
            "pixel_values": prefix_batch["pixel_values"],
            "image_grid_thw": prefix_batch.get("image_grid_thw"),
        }
        if self.include_metadata:
            out["meta"] = {
                "dataset_name": [item["dataset_name"] for item in batch],
                "sample_id": [item["sample_id"] for item in batch],
                "image_path": [item["image_path"] for item in batch],
                "target_text": [item["target_text"] for item in batch],
                "extra": [dict(item.get("extra", {})) for item in batch],
            }
        mm_rows = [row.mm_token_type_ids for row in rows if row.mm_token_type_ids is not None]
        if mm_rows:
            out["mm_token_type_ids"] = self._pad_sequences(mm_rows, padding_value=0)
        loss_scale_rows = [row.loss_scale for row in rows if row.loss_scale is not None]
        if loss_scale_rows:
            out["loss_scale"] = self._pad_sequences(loss_scale_rows, padding_value=0).to(dtype=torch.float32)
        return out


class DPOCollator(_ShaftSequenceCollatorBase):
    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        chosen_plans = [
            self.template.build_supervision_plan(
                item=item,
                target_text=str(item["chosen_text"]),
                processor=self.processor,
                tokenizer=self.tokenizer,
                loss_scale_name=self.loss_scale_name,
            )
            for item in batch
        ]
        rejected_plans = [
            self.template.build_supervision_plan(
                item=item,
                target_text=str(item["rejected_text"]),
                processor=self.processor,
                tokenizer=self.tokenizer,
                loss_scale_name=self.loss_scale_name,
            )
            for item in batch
        ]
        prompt_texts = [plan.prompt_text for plan in chosen_plans]
        images = [item["image"] for item in batch]
        prefix_batch = self._run_processor(prompt_texts, images)

        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else eos_id
        chosen_rows = [
            self.template.build_supervised_row(
                plan=plan,
                model_adapter=self.model_adapter,
                processor=self.processor,
                tokenizer=self.tokenizer,
                image=item["image"],
                prefix_batch=prefix_batch,
                row_index=row_index,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
                add_eos_token=self.add_eos_token,
                ignore_index=self.ignore_index,
                include_targets_in_inputs=True,
                max_length=self.max_length,
            )
            for row_index, (item, plan) in enumerate(zip(batch, chosen_plans))
        ]
        rejected_rows = [
            self.template.build_supervised_row(
                plan=plan,
                model_adapter=self.model_adapter,
                processor=self.processor,
                tokenizer=self.tokenizer,
                image=item["image"],
                prefix_batch=prefix_batch,
                row_index=row_index,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
                add_eos_token=self.add_eos_token,
                ignore_index=self.ignore_index,
                include_targets_in_inputs=True,
                max_length=self.max_length,
            )
            for row_index, (item, plan) in enumerate(zip(batch, rejected_plans))
        ]
        input_rows = [*(row.input_ids for row in chosen_rows), *(row.input_ids for row in rejected_rows)]
        attention_rows = [*(row.attention_mask for row in chosen_rows), *(row.attention_mask for row in rejected_rows)]
        completion_rows = [
            *(row.labels.ne(self.ignore_index) for row in chosen_rows),
            *(row.labels.ne(self.ignore_index) for row in rejected_rows),
        ]
        out: dict[str, Any] = {
            "input_ids": self._pad_sequences(input_rows, padding_value=int(pad_id)),
            "attention_mask": self._pad_sequences(attention_rows, padding_value=0),
            "completion_mask": self._pad_sequences(
                [row.to(dtype=torch.long) for row in completion_rows],
                padding_value=0,
            ),
        }
        for key in ("pixel_values", "pixel_attention_mask", "image_grid_thw", "image_sizes"):
            if key in prefix_batch:
                out[key] = self._repeat_on_batch_axis(prefix_batch[key], repeats=2)
        chosen_mm_rows = [row.mm_token_type_ids for row in chosen_rows if row.mm_token_type_ids is not None]
        rejected_mm_rows = [row.mm_token_type_ids for row in rejected_rows if row.mm_token_type_ids is not None]
        if chosen_mm_rows and rejected_mm_rows:
            out["mm_token_type_ids"] = self._pad_sequences(
                [*chosen_mm_rows, *rejected_mm_rows],
                padding_value=0,
            )
        return out


class PPOCollator(_ShaftSequenceCollatorBase):
    def _apply_text_only_chat_template(self, item: dict[str, Any]) -> str:
        messages = item.get("messages")
        if messages:
            text_messages = []
            for message in messages:
                content = [
                    chunk
                    for chunk in message.get("content", [])
                    if str(chunk.get("type", "")).strip().lower() != "image"
                ]
                text_messages.append({**message, "content": content})
        else:
            text_messages = []
            system_prompt = str(item.get("system_prompt", "")).strip()
            if system_prompt:
                text_messages.append(
                    {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
                )
            text_messages.append(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": str(item.get("user_prompt", ""))}],
                }
            )
        return self.template.apply_chat_template(
            processor=self.processor,
            tokenizer=self.tokenizer,
            messages=self.template.prepare_messages(text_messages),
            add_generation_prompt=None,
        )

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        prompt_texts = [self._apply_text_only_chat_template(item) for item in batch]
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else eos_id
        tokenized = self.tokenizer(
            prompt_texts,
            add_special_tokens=False,
            return_attention_mask=False,
        )
        prompt_token_ids = [list(row) for row in tokenized["input_ids"]]
        input_rows = [torch.tensor(row, dtype=torch.long) for row in prompt_token_ids]
        attention_rows = [torch.ones_like(row) for row in input_rows]
        out: dict[str, Any] = {
            "input_ids": self._pad_sequences(input_rows, padding_value=int(pad_id)),
            "attention_mask": self._pad_sequences(attention_rows, padding_value=0),
        }
        return out


class GRPOCollator:
    def __init__(self, *, template: Template) -> None:
        self.template = template

    def __call__(self, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for item in batch:
            prompt = self.template.prepare_messages(self.template.resolve_messages(item))
            rows.append(
                {
                    "prompt": prompt,
                    "image": item.get("image"),
                    "target_text": str(item.get("target_text", "")),
                    "dataset_name": item.get("dataset_name"),
                    "sample_id": item.get("sample_id"),
                    "image_path": item.get("image_path"),
                    "extra": dict(item.get("extra", {})),
                }
            )
        return rows
