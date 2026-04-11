from __future__ import annotations

from typing import Any

import torch

from vlm_structgen.core.registry import get_adapter_for_route


class SFTCollator:
    def __init__(
        self,
        processor,
        tokenizer,
        num_bins: int,
        task_route_options: dict[str, dict[str, Any]] | None = None,
        route_pixel_budgets: dict[str, dict[str, Any]] | None = None,
        add_eos_token: bool = True,
        ignore_index: int = -100,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        include_targets_in_inputs: bool = True,
        padding_side: str = "right",
    ) -> None:
        self.processor = processor
        self.tokenizer = tokenizer
        self.num_bins = int(num_bins)
        self.task_route_options = dict(task_route_options or {})
        self.route_pixel_budgets = dict(route_pixel_budgets or {})
        self.add_eos_token = add_eos_token
        self.ignore_index = ignore_index
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.include_targets_in_inputs = include_targets_in_inputs
        if padding_side not in {"left", "right"}:
            raise ValueError(f"Unsupported padding_side={padding_side!r}. Expected 'left' or 'right'.")
        self.padding_side = padding_side

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        messages = [self._build_messages(item["system_prompt"], item["user_prompt"]) for item in batch]
        prefix_texts = [self._apply_chat_template(message) for message in messages]
        images = [item["image"] for item in batch]
        pixel_budget_pairs = [self._resolve_pixel_budget(item) for item in batch]
        if len(set(pixel_budget_pairs)) == 1:
            # Fast path: all samples share the same pixel budget.
            min_pixels, max_pixels = pixel_budget_pairs[0]
            prefix_batch = self._run_processor(
                text=prefix_texts,
                images=images,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
        else:
            prefix_batch = self._build_prefix_batch_with_per_sample_pixel_budget(
                prefix_texts=prefix_texts,
                images=images,
                pixel_budget_pairs=pixel_budget_pairs,
            )

        target_batch = self.tokenizer(
            [item["target_text"] for item in batch],
            add_special_tokens=False,
            return_attention_mask=False,
        )
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else eos_id

        prompt_lengths = prefix_batch["attention_mask"].sum(dim=1).tolist()
        final_input_ids: list[torch.Tensor] = []
        final_labels: list[torch.Tensor] = []
        final_loss_weights: list[torch.Tensor] = []
        final_attention_masks: list[torch.Tensor] = []
        final_mm_token_type_ids: list[torch.Tensor] = []
        prompt_length_tensor: list[int] = []
        prefix_mm_token_type_ids = prefix_batch.get("mm_token_type_ids")

        for row_index, prompt_length in enumerate(prompt_lengths):
            prefix_mask = prefix_batch["attention_mask"][row_index].bool()
            prefix_ids = prefix_batch["input_ids"][row_index][prefix_mask]
            prefix_mm_ids = None
            if prefix_mm_token_type_ids is not None:
                prefix_mm_ids = prefix_mm_token_type_ids[row_index][prefix_mask]
            if self.include_targets_in_inputs:
                target_ids = list(target_batch["input_ids"][row_index])
                target_loss_weights = self._build_target_loss_weights(
                    item=batch[row_index],
                    target_ids=target_ids,
                    eos_id=eos_id,
                )
                if self.add_eos_token and eos_id is not None and (not target_ids or target_ids[-1] != eos_id):
                    target_ids.append(eos_id)
                target_tensor = torch.tensor(target_ids, dtype=torch.long)
                if len(target_loss_weights) != len(target_ids):
                    raise ValueError(
                        "SFTCollator constructed misaligned target loss weights. "
                        f"sample_id={batch[row_index].get('sample_id')!r}, "
                        f"target_ids={len(target_ids)}, target_loss_weights={len(target_loss_weights)}."
                    )
                target_loss_weight_tensor = torch.tensor(target_loss_weights, dtype=torch.float32)
                input_ids = torch.cat([prefix_ids, target_tensor], dim=0)
                labels = torch.cat(
                    [
                        torch.full((prefix_ids.shape[0],), self.ignore_index, dtype=torch.long),
                        target_tensor.clone(),
                    ],
                    dim=0,
                )
                loss_weights = torch.cat(
                    [
                        torch.ones((prefix_ids.shape[0],), dtype=torch.float32),
                        target_loss_weight_tensor,
                    ],
                    dim=0,
                )
                if prefix_mm_ids is not None:
                    target_mm_ids = torch.zeros_like(target_tensor)
                    mm_token_type_ids = torch.cat([prefix_mm_ids, target_mm_ids], dim=0)
            else:
                input_ids = prefix_ids
                labels = torch.full((prefix_ids.shape[0],), self.ignore_index, dtype=torch.long)
                loss_weights = torch.ones((prefix_ids.shape[0],), dtype=torch.float32)
                if prefix_mm_ids is not None:
                    mm_token_type_ids = prefix_mm_ids
            attention_mask = torch.ones_like(input_ids)
            final_input_ids.append(input_ids)
            final_labels.append(labels)
            final_loss_weights.append(loss_weights)
            final_attention_masks.append(attention_mask)
            if prefix_mm_ids is not None:
                final_mm_token_type_ids.append(mm_token_type_ids)
            prompt_length_tensor.append(prefix_ids.shape[0])

        padded_input_ids = self._pad_sequences(
            final_input_ids,
            padding_value=pad_id,
        )
        padded_labels = self._pad_sequences(
            final_labels,
            padding_value=self.ignore_index,
        )
        padded_loss_weights = self._pad_sequences(
            final_loss_weights,
            padding_value=1.0,
        )
        padded_attention_masks = self._pad_sequences(
            final_attention_masks,
            padding_value=0,
        )

        output = {
            "input_ids": padded_input_ids,
            "attention_mask": padded_attention_masks,
            "labels": padded_labels,
            "loss_weights": padded_loss_weights,
            "pixel_values": prefix_batch["pixel_values"],
            "image_grid_thw": prefix_batch.get("image_grid_thw"),
            "prompt_lengths": torch.tensor(prompt_length_tensor, dtype=torch.long),
            "meta": {
                "route": [item["route"] for item in batch],
                "sample_id": [item["sample_id"] for item in batch],
                "image_path": [item["image_path"] for item in batch],
                "image_width": [item["image_width"] for item in batch],
                "image_height": [item["image_height"] for item in batch],
                "system_prompt": [item["system_prompt"] for item in batch],
                "user_prompt": [item["user_prompt"] for item in batch],
                "gt_struct": [item["gt_struct"] for item in batch],
                "target_text": [item["target_text"] for item in batch],
                "loss_meta": [item.get("loss_meta") for item in batch],
            },
        }
        if prefix_mm_token_type_ids is not None:
            output["mm_token_type_ids"] = self._pad_sequences(
                final_mm_token_type_ids,
                padding_value=0,
            )
        return output

    def _build_prefix_batch_with_per_sample_pixel_budget(
        self,
        *,
        prefix_texts: list[str],
        images: list[Any],
        pixel_budget_pairs: list[tuple[int | None, int | None]],
    ) -> dict[str, Any]:
        budget_groups = self._group_row_indices_by_pixel_budget(pixel_budget_pairs)
        if len(budget_groups) <= 1:
            raise ValueError(
                "_build_prefix_batch_with_per_sample_pixel_budget expects at least two budget groups."
            )

        input_ids_rows: list[torch.Tensor | None] = [None] * len(prefix_texts)
        attention_mask_rows: list[torch.Tensor | None] = [None] * len(prefix_texts)
        mm_token_type_id_rows: list[torch.Tensor | None] = [None] * len(prefix_texts)
        pixel_values_rows: list[torch.Tensor | None] = [None] * len(prefix_texts)
        image_grid_rows: list[torch.Tensor | None] = [None] * len(prefix_texts)
        saw_mm_token_type_ids = False

        for (min_pixels, max_pixels), row_indices in budget_groups.items():
            grouped = self._run_processor(
                text=[prefix_texts[row_index] for row_index in row_indices],
                images=[images[row_index] for row_index in row_indices],
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
            grouped_size = len(row_indices)
            input_ids = grouped["input_ids"]
            attention_mask = grouped["attention_mask"]
            if int(input_ids.shape[0]) != grouped_size or int(attention_mask.shape[0]) != grouped_size:
                raise ValueError("Processor output row count does not match grouped batch size.")

            mm_token_type_ids = grouped.get("mm_token_type_ids")
            if mm_token_type_ids is not None:
                if int(mm_token_type_ids.shape[0]) != grouped_size:
                    raise ValueError("Processor mm_token_type_ids row count does not match grouped batch size.")
                saw_mm_token_type_ids = True

            grouped_image_grid = grouped.get("image_grid_thw")
            grouped_pixel_values = self._split_pixel_values_by_sample(
                pixel_values=grouped["pixel_values"],
                image_grid_thw=grouped_image_grid,
                sample_count=grouped_size,
            )
            if len(grouped_pixel_values) != grouped_size:
                raise ValueError("Failed to split processor pixel_values by sample.")

            for local_index, row_index in enumerate(row_indices):
                input_ids_rows[row_index] = input_ids[local_index]
                attention_mask_rows[row_index] = attention_mask[local_index]
                pixel_values_rows[row_index] = grouped_pixel_values[local_index]

                if grouped_image_grid is not None:
                    image_grid_rows[row_index] = grouped_image_grid[local_index : local_index + 1]

                if mm_token_type_ids is not None:
                    mm_token_type_id_rows[row_index] = mm_token_type_ids[local_index]

        if any(row is None for row in input_ids_rows) or any(row is None for row in attention_mask_rows):
            raise ValueError("Failed to build per-sample prefix tensors for mixed pixel budgets.")
        if any(row is None for row in pixel_values_rows):
            raise ValueError("Failed to build per-sample pixel tensors for mixed pixel budgets.")

        padded_input_ids = self._pad_sequences(
            [row for row in input_ids_rows if row is not None],
            padding_value=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
        )
        padded_attention_mask = self._pad_sequences(
            [row for row in attention_mask_rows if row is not None],
            padding_value=0,
        )

        merged: dict[str, Any] = {
            "input_ids": padded_input_ids,
            "attention_mask": padded_attention_mask,
            "pixel_values": torch.cat([row for row in pixel_values_rows if row is not None], dim=0),
            "image_grid_thw": None,
        }
        if any(row is not None for row in image_grid_rows):
            if any(row is None for row in image_grid_rows):
                raise ValueError("image_grid_thw is missing for part of batch in mixed pixel budget mode.")
            merged["image_grid_thw"] = torch.cat([row for row in image_grid_rows if row is not None], dim=0)
        if saw_mm_token_type_ids:
            if any(row is None for row in mm_token_type_id_rows):
                raise ValueError("mm_token_type_ids is missing for part of batch in mixed pixel budget mode.")
            merged["mm_token_type_ids"] = self._pad_sequences(
                [row for row in mm_token_type_id_rows if row is not None],
                padding_value=0,
            )
        return merged

    def _resolve_pixel_budget(self, item: dict[str, Any]) -> tuple[int | None, int | None]:
        route_key = str(item["route"])
        route_budget = dict(self.route_pixel_budgets.get(route_key, {}))
        min_pixels = route_budget.get("min_pixels", self.min_pixels)
        max_pixels = route_budget.get("max_pixels", self.max_pixels)
        min_pixels = int(min_pixels) if min_pixels is not None else None
        max_pixels = int(max_pixels) if max_pixels is not None else None
        return min_pixels, max_pixels

    def _group_row_indices_by_pixel_budget(
        self,
        pixel_budget_pairs: list[tuple[int | None, int | None]],
    ) -> dict[tuple[int | None, int | None], list[int]]:
        groups: dict[tuple[int | None, int | None], list[int]] = {}
        for row_index, budget_pair in enumerate(pixel_budget_pairs):
            groups.setdefault(budget_pair, []).append(row_index)
        return groups

    def _split_pixel_values_by_sample(
        self,
        *,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor | None,
        sample_count: int,
    ) -> list[torch.Tensor]:
        if image_grid_thw is None:
            if int(pixel_values.shape[0]) != int(sample_count):
                raise ValueError(
                    "Cannot split pixel_values without image_grid_thw: "
                    f"pixel_values_rows={int(pixel_values.shape[0])}, sample_count={sample_count}."
                )
            return [pixel_values[row_index : row_index + 1] for row_index in range(sample_count)]

        if int(image_grid_thw.shape[0]) != int(sample_count):
            raise ValueError(
                "image_grid_thw row count does not match sample count: "
                f"image_grid_rows={int(image_grid_thw.shape[0])}, sample_count={sample_count}."
            )

        token_counts = (image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]).tolist()
        rows: list[torch.Tensor] = []
        cursor = 0
        for token_count in token_counts:
            count = int(token_count)
            if count <= 0:
                raise ValueError(f"Invalid image token count parsed from image_grid_thw: {count}.")
            next_cursor = cursor + count
            rows.append(pixel_values[cursor:next_cursor])
            cursor = next_cursor
        if cursor != int(pixel_values.shape[0]):
            raise ValueError(
                "Failed to consume all pixel_values rows when splitting grouped batch: "
                f"consumed={cursor}, total={int(pixel_values.shape[0])}."
            )
        return rows

    def _run_processor(
        self,
        *,
        text: list[str],
        images: list[Any],
        min_pixels: int | None,
        max_pixels: int | None,
    ) -> dict[str, Any]:
        processor_kwargs = {
            "text": text,
            "images": images,
            "padding": True,
            "return_tensors": "pt",
        }
        if min_pixels is not None:
            processor_kwargs["min_pixels"] = int(min_pixels)
        if max_pixels is not None:
            processor_kwargs["max_pixels"] = int(max_pixels)
        return self.processor(**processor_kwargs)

    def _build_target_loss_weights(
        self,
        *,
        item: dict[str, Any],
        target_ids: list[int],
        eos_id: int | None,
    ) -> list[float]:
        adapter = self._get_adapter_for_item(item)
        target_weights = adapter.build_target_token_weights(
            str(item["target_text"]),
            loss_meta=item.get("loss_meta"),
            tokenizer=self.tokenizer,
        )
        if target_weights is None:
            raise ValueError(
                "Adapter did not provide target token weights. "
                f"sample_id={item.get('sample_id')!r}, route={item.get('route')!r}."
            )
        if len(target_weights) != len(target_ids):
            raise ValueError(
                "Adapter returned target token weights that do not match tokenizer output. "
                f"sample_id={item.get('sample_id')!r}, route={item.get('route')!r}, "
                f"target_ids={len(target_ids)}, target_weights={len(target_weights)}."
            )

        if self.add_eos_token and eos_id is not None and (not target_ids or target_ids[-1] != eos_id):
            target_weights = target_weights + [1.0]
        return target_weights

    def _get_adapter_for_item(self, item: dict[str, Any]):
        route_key = str(item["route"])
        task_options = self.task_route_options.get(route_key, {})
        return get_adapter_for_route(
            route_key=route_key,
            num_bins=self.num_bins,
            task_options_key=tuple(sorted(dict(task_options).items())),
        )

    def _build_messages(self, system_prompt: str, user_prompt: str) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if system_prompt.strip():
            messages.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                }
            )
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_prompt},
                ],
            }
        )
        return messages

    def _apply_chat_template(self, messages: list[dict[str, Any]]) -> str:
        template_owner = self.processor if hasattr(self.processor, "apply_chat_template") else self.tokenizer
        return template_owner.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _pad_sequences(self, sequences: list[torch.Tensor], padding_value: int) -> torch.Tensor:
        max_length = max(sequence.shape[0] for sequence in sequences)
        padded: list[torch.Tensor] = []
        for sequence in sequences:
            if sequence.shape[0] == max_length:
                padded.append(sequence)
                continue
            pad = torch.full(
                (max_length - sequence.shape[0],),
                padding_value,
                dtype=sequence.dtype,
            )
            if self.padding_side == "left":
                padded.append(torch.cat([pad, sequence], dim=0))
            else:
                padded.append(torch.cat([sequence, pad], dim=0))
        return torch.stack(padded, dim=0)
