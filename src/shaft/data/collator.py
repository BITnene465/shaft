from __future__ import annotations

from typing import Any

import torch

from shaft.loss_scale import ShaftLossScaleSpec, build_loss_scale
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
        loss_scale_name: str = "default",
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
        self.loss_scale_name = str(loss_scale_name).strip().lower() or "default"
        self.loss_scale = build_loss_scale(self.loss_scale_name)
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

    def _apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool | None = None,
    ) -> str:
        return self.template.apply_chat_template(
            processor=self.processor,
            tokenizer=self.tokenizer,
            messages=messages,
            add_generation_prompt=add_generation_prompt,
        )

    def _compute_prefix_loss_scale_row(
        self,
        *,
        messages: list[dict[str, Any]],
        image: Any,
        prefix_length: int,
        loss_spec: ShaftLossScaleSpec,
    ) -> torch.Tensor:
        if prefix_length <= 0:
            return torch.zeros((0,), dtype=torch.float32)
        if float(loss_spec.prefix_scale) <= 0:
            return torch.zeros((prefix_length,), dtype=torch.float32)
        if loss_spec.base_strategy == "all":
            return torch.full((prefix_length,), float(loss_spec.prefix_scale), dtype=torch.float32)
        if loss_spec.base_strategy == "last_round":
            return torch.zeros((prefix_length,), dtype=torch.float32)

        normalized_messages = self.template.prepare_messages(messages)
        weights = torch.zeros((prefix_length,), dtype=torch.float32)
        previous_length = 0
        for idx, message in enumerate(normalized_messages):
            is_final_prompt = idx == len(normalized_messages) - 1 and str(message.get("role", "")).strip().lower() != "assistant"
            rendered = self.template.apply_chat_template(
                processor=self.processor,
                tokenizer=self.tokenizer,
                messages=normalized_messages[: idx + 1],
                add_generation_prompt=is_final_prompt,
            )
            partial_batch = self.model_adapter.build_processor_inputs(
                processor=self.processor,
                prompt_texts=[rendered],
                images=[image],
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )
            current_length = int(partial_batch["attention_mask"][0].sum().item())
            current_length = min(current_length, prefix_length)
            role = str(message.get("role", "")).strip().lower()
            if (
                current_length > previous_length
                and loss_spec.base_strategy == "default"
                and role == "assistant"
            ):
                weights[previous_length:current_length] = float(loss_spec.prefix_scale)
            previous_length = max(previous_length, current_length)
        return weights

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
        batch: list[dict[str, Any]],
        prefix_batch: dict[str, torch.Tensor],
        target_token_ids: list[list[int]],
        include_targets_in_inputs: bool,
        loss_scale_specs: list[ShaftLossScaleSpec] | None = None,
    ) -> tuple[
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor] | None,
        list[torch.Tensor] | None,
    ]:
        eos_id = self.tokenizer.eos_token_id
        prompt_lengths = prefix_batch["attention_mask"].sum(dim=1).tolist()
        has_mm_token_type_ids = "mm_token_type_ids" in prefix_batch
        mm_token_ids = prefix_batch.get("mm_token_type_ids")

        input_ids_rows: list[torch.Tensor] = []
        labels_rows: list[torch.Tensor] = []
        attn_rows: list[torch.Tensor] = []
        mm_rows: list[torch.Tensor] | None = [] if has_mm_token_type_ids else None
        loss_scale_rows: list[torch.Tensor] | None = []
        need_loss_scale_tensor = False

        for row_index, prompt_length in enumerate(prompt_lengths):
            prefix_mask = prefix_batch["attention_mask"][row_index].bool()
            prefix_ids = prefix_batch["input_ids"][row_index][prefix_mask]
            prefix_mm = mm_token_ids[row_index][prefix_mask] if has_mm_token_type_ids else None
            target_ids = list(target_token_ids[row_index])
            if self.add_eos_token and eos_id is not None and (not target_ids or target_ids[-1] != eos_id):
                target_ids.append(int(eos_id))
            target_tensor = torch.tensor(target_ids, dtype=torch.long)
            loss_spec = loss_scale_specs[row_index] if loss_scale_specs is not None else ShaftLossScaleSpec()

            if include_targets_in_inputs:
                if not loss_spec.is_binary:
                    need_loss_scale_tensor = True
                prefix_loss_scale = self._compute_prefix_loss_scale_row(
                    messages=self._resolve_messages(batch[row_index]),
                    image=batch[row_index]["image"],
                    prefix_length=int(prefix_ids.shape[0]),
                    loss_spec=loss_spec,
                )
                input_ids = torch.cat([prefix_ids, target_tensor], dim=0)
                prefix_labels = (
                    prefix_ids.clone()
                    if torch.any(prefix_loss_scale > 0)
                    else torch.full((prefix_ids.shape[0],), self.ignore_index, dtype=torch.long)
                )
                if prefix_labels.shape[0] > 0:
                    prefix_labels = prefix_labels.masked_fill(prefix_loss_scale <= 0, self.ignore_index)
                target_labels = (
                    target_tensor.clone()
                    if float(loss_spec.target_scale) > 0
                    else torch.full((target_tensor.shape[0],), self.ignore_index, dtype=torch.long)
                )
                labels = torch.cat([prefix_labels, target_labels], dim=0)
                if mm_rows is not None:
                    assert prefix_mm is not None
                    mm_rows.append(torch.cat([prefix_mm, torch.zeros_like(target_tensor)], dim=0))
                if loss_scale_rows is not None:
                    loss_scale_rows.append(
                        torch.cat(
                            [
                                prefix_loss_scale,
                                torch.full((target_tensor.shape[0],), float(loss_spec.target_scale), dtype=torch.float32),
                            ],
                            dim=0,
                        )
                    )
            else:
                input_ids = prefix_ids
                labels = torch.full((prompt_length,), self.ignore_index, dtype=torch.long)
                if mm_rows is not None:
                    assert prefix_mm is not None
                    mm_rows.append(prefix_mm)
                loss_scale_rows = None

            input_ids_rows.append(input_ids)
            labels_rows.append(labels)
            attn_rows.append(torch.ones_like(input_ids))

        if not need_loss_scale_tensor:
            loss_scale_rows = None
        return input_ids_rows, labels_rows, attn_rows, mm_rows, loss_scale_rows

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
        loss_scale_name: str = "default",
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
            loss_scale_name=loss_scale_name,
        )
        self.include_targets_in_inputs = bool(include_targets_in_inputs)

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        messages = [self._resolve_messages(item) for item in batch]
        prompt_texts = [self._apply_chat_template(msg) for msg in messages]
        images = [item["image"] for item in batch]
        prefix_batch = self._run_processor(prompt_texts, images)
        target_token_ids = self._tokenize_targets([str(item["target_text"]) for item in batch])
        loss_scale_specs = [self.loss_scale(item) for item in batch]
        input_rows, label_rows, attn_rows, mm_rows, loss_scale_rows = self._build_supervised_rows(
            batch=batch,
            prefix_batch=prefix_batch,
            target_token_ids=target_token_ids,
            include_targets_in_inputs=self.include_targets_in_inputs,
            loss_scale_specs=loss_scale_specs,
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
                "extra": [dict(item.get("extra", {})) for item in batch],
            },
        }
        if mm_rows is not None:
            out["mm_token_type_ids"] = self._pad_sequences(mm_rows, padding_value=0)
        if loss_scale_rows is not None:
            out["loss_scale"] = self._pad_sequences(loss_scale_rows, padding_value=0).to(dtype=torch.float32)
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
            batch=batch,
            prefix_batch=prefix_batch,
            target_token_ids=chosen_token_ids,
            include_targets_in_inputs=True,
        )
        rejected_rows = self._build_supervised_rows(
            batch=batch,
            prefix_batch=prefix_batch,
            target_token_ids=rejected_token_ids,
            include_targets_in_inputs=True,
        )

        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else eos_id
        chosen_inputs, chosen_labels, chosen_attn, chosen_mm_rows, _ = chosen_rows
        rejected_inputs, rejected_labels, rejected_attn, rejected_mm_rows, _ = rejected_rows
        input_rows = [*chosen_inputs, *rejected_inputs]
        attention_rows = [*chosen_attn, *rejected_attn]
        completion_rows = [
            *(row.ne(self.ignore_index) for row in chosen_labels),
            *(row.ne(self.ignore_index) for row in rejected_labels),
        ]
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
