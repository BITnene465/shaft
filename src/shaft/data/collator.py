from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch

from shaft.model import ShaftModelAdapter, ShaftProcessedBatch, ShaftProcessorTokenLayout
from shaft.template import (
    ShaftChatRenderer,
    ShaftTemplateSupervisedRow,
    ShaftTemplateSupervisionPlan,
    Template,
)


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
        self.chat_renderer = ShaftChatRenderer.from_components(
            processor=processor,
            tokenizer=tokenizer,
        )
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.max_length = int(max_length) if max_length is not None else None
        self.add_eos_token = bool(add_eos_token)
        self.ignore_index = int(ignore_index)
        self.padding_side = padding_side
        self.loss_scale_name = str(loss_scale_name).strip().lower() or "default"
        if self.padding_side not in {"left", "right"}:
            raise ValueError("padding_side must be 'left' or 'right'.")

    def _run_processor(self, prompt_texts: list[str], images: list[Any]) -> ShaftProcessedBatch:
        return self.model_adapter.build_processor_batch(
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

    def _build_prefix_token_layouts(
        self,
        *,
        plans: list[ShaftTemplateSupervisionPlan],
        processed_batch: ShaftProcessedBatch,
    ) -> list[ShaftProcessorTokenLayout | None]:
        layouts: list[ShaftProcessorTokenLayout | None] = []
        for row_index, plan in enumerate(plans):
            if not plan.trainable_prefix_spans:
                layouts.append(None)
                continue
            layouts.append(
                self.model_adapter.build_processor_token_layout(
                    rendered_token_ids=plan.rendered_prefix_token_ids,
                    processed_batch=processed_batch,
                    row_index=row_index,
                )
            )
        return layouts


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
                renderer=self.chat_renderer,
                loss_scale_name=self.loss_scale_name,
            )
            for item in batch
        ]
        prompt_texts = [plan.prompt_text for plan in plans]
        images = [item["image"] for item in batch]
        processed_batch = self._run_processor(prompt_texts, images)
        prefix_token_layouts = self._build_prefix_token_layouts(
            plans=plans,
            processed_batch=processed_batch,
        )
        rows: list[ShaftTemplateSupervisedRow] = [
            self.template.build_supervised_row(
                plan=plan,
                tokenizer=self.tokenizer,
                processed_batch=processed_batch,
                row_index=row_index,
                prefix_token_layout=prefix_token_layout,
                add_eos_token=self.add_eos_token,
                ignore_index=self.ignore_index,
                include_targets_in_inputs=self.include_targets_in_inputs,
                max_length=self.max_length,
            )
            for row_index, (plan, prefix_token_layout) in enumerate(
                zip(plans, prefix_token_layouts)
            )
        ]
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else eos_id
        sequence_inputs: dict[str, Any] = {
            "input_ids": self._pad_sequences([row.input_ids for row in rows], padding_value=int(pad_id)),
            "attention_mask": self._pad_sequences([row.attention_mask for row in rows], padding_value=0),
            "labels": self._pad_sequences([row.labels for row in rows], padding_value=self.ignore_index),
        }
        mm_rows = [row.mm_token_type_ids for row in rows if row.mm_token_type_ids is not None]
        if mm_rows:
            sequence_inputs["mm_token_type_ids"] = self._pad_sequences(
                mm_rows,
                padding_value=0,
            )
        loss_scale_rows = [row.loss_scale for row in rows if row.loss_scale is not None]
        if loss_scale_rows:
            sequence_inputs["loss_scale"] = self._pad_sequences(
                loss_scale_rows,
                padding_value=0,
            ).to(dtype=torch.float32)
        out = self.model_adapter.assemble_processor_training_inputs(
            processed_batch=processed_batch,
            sequence_inputs=sequence_inputs,
            row_indices=tuple(range(len(batch))),
        )
        if self.include_metadata:
            out["meta"] = {
                "dataset_name": [item["dataset_name"] for item in batch],
                "sample_id": [item["sample_id"] for item in batch],
                "image_path": [item["image_path"] for item in batch],
                "target_text": [item["target_text"] for item in batch],
                "extra": [dict(item.get("extra", {})) for item in batch],
            }
        return out


class DPOCollator(_ShaftSequenceCollatorBase):
    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        chosen_plans = [
            self.template.build_supervision_plan(
                item=item,
                target_text=str(item["chosen_text"]),
                renderer=self.chat_renderer,
                loss_scale_name=self.loss_scale_name,
            )
            for item in batch
        ]
        rejected_plans = [
            replace(plan, target_text=str(item["rejected_text"]))
            for item, plan in zip(batch, chosen_plans)
        ]
        prompt_texts = [plan.prompt_text for plan in chosen_plans]
        images = [item["image"] for item in batch]
        processed_batch = self._run_processor(prompt_texts, images)
        prefix_token_layouts = self._build_prefix_token_layouts(
            plans=chosen_plans,
            processed_batch=processed_batch,
        )

        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else eos_id
        chosen_rows = [
            self.template.build_supervised_row(
                plan=plan,
                tokenizer=self.tokenizer,
                processed_batch=processed_batch,
                row_index=row_index,
                prefix_token_layout=prefix_token_layout,
                add_eos_token=self.add_eos_token,
                ignore_index=self.ignore_index,
                include_targets_in_inputs=True,
                max_length=self.max_length,
            )
            for row_index, (plan, prefix_token_layout) in enumerate(
                zip(chosen_plans, prefix_token_layouts)
            )
        ]
        rejected_rows = [
            self.template.build_supervised_row(
                plan=plan,
                tokenizer=self.tokenizer,
                processed_batch=processed_batch,
                row_index=row_index,
                prefix_token_layout=prefix_token_layout,
                add_eos_token=self.add_eos_token,
                ignore_index=self.ignore_index,
                include_targets_in_inputs=True,
                max_length=self.max_length,
            )
            for row_index, (plan, prefix_token_layout) in enumerate(
                zip(rejected_plans, prefix_token_layouts)
            )
        ]
        input_rows = [*(row.input_ids for row in chosen_rows), *(row.input_ids for row in rejected_rows)]
        attention_rows = [*(row.attention_mask for row in chosen_rows), *(row.attention_mask for row in rejected_rows)]
        completion_rows = [
            *(row.labels.ne(self.ignore_index) for row in chosen_rows),
            *(row.labels.ne(self.ignore_index) for row in rejected_rows),
        ]
        sequence_inputs: dict[str, Any] = {
            "input_ids": self._pad_sequences(input_rows, padding_value=int(pad_id)),
            "attention_mask": self._pad_sequences(attention_rows, padding_value=0),
            "completion_mask": self._pad_sequences(
                [row.to(dtype=torch.long) for row in completion_rows],
                padding_value=0,
            ),
        }
        chosen_mm_rows = [row.mm_token_type_ids for row in chosen_rows if row.mm_token_type_ids is not None]
        rejected_mm_rows = [row.mm_token_type_ids for row in rejected_rows if row.mm_token_type_ids is not None]
        if chosen_mm_rows and rejected_mm_rows:
            sequence_inputs["mm_token_type_ids"] = self._pad_sequences(
                [*chosen_mm_rows, *rejected_mm_rows],
                padding_value=0,
            )
        return self.model_adapter.assemble_processor_training_inputs(
            processed_batch=processed_batch,
            sequence_inputs=sequence_inputs,
            row_indices=tuple(range(len(batch))) * 2,
        )


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
            renderer=self.chat_renderer,
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
