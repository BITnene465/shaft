from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from .batching import ShaftCollatedBatchStats, ShaftVarlenBatchLayout
from shaft.model import ShaftModelAdapter, ShaftProcessedBatch, ShaftProcessorTokenLayout
from shaft.template import (
    ShaftChatRenderer,
    ShaftTemplatePromptPlan,
    ShaftTemplatePromptRow,
    ShaftTemplateSupervisedRow,
    ShaftTemplateSupervisionPlan,
    Template,
)


class ShaftSequenceCollatorBase:
    DEFAULT_INPUT_MODE = "training"

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
        input_mode: str | None = None,
        loss_scale_name: str = "default",
        pixel_budgets_by_dataset: Mapping[
            str,
            tuple[int | None, int | None],
        ]
        | None = None,
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
        resolved_input_mode = self.DEFAULT_INPUT_MODE if input_mode is None else input_mode
        self.input_mode = str(resolved_input_mode).strip().lower()
        self.padding_side = model_adapter.resolve_processor_padding_side(self.input_mode)
        self.loss_scale_name = str(loss_scale_name).strip().lower() or "default"
        self.pixel_budgets_by_dataset = {
            str(dataset_name): (
                int(budget[0]) if budget[0] is not None else None,
                int(budget[1]) if budget[1] is not None else None,
            )
            for dataset_name, budget in (pixel_budgets_by_dataset or {}).items()
        }

    def _resolve_pixel_budget(
        self,
        dataset_names: list[str | None] | None,
    ) -> tuple[int | None, int | None]:
        default_budget = (self.min_pixels, self.max_pixels)
        if not self.pixel_budgets_by_dataset:
            return default_budget
        normalized_names = [
            str(dataset_name).strip() if dataset_name is not None else ""
            for dataset_name in (dataset_names or [])
        ]
        if not normalized_names:
            return default_budget
        budgets = {
            self.pixel_budgets_by_dataset.get(dataset_name, default_budget)
            for dataset_name in normalized_names
        }
        if len(budgets) != 1:
            raise ValueError(
                "A processor batch cannot mix datasets with different eval pixel budgets."
            )
        return next(iter(budgets))

    def _run_processor(
        self,
        prompt_texts: list[str],
        images: list[Any],
        *,
        dataset_names: list[str | None] | None = None,
    ) -> ShaftProcessedBatch:
        min_pixels, max_pixels = self._resolve_pixel_budget(dataset_names)
        return self.model_adapter.build_processor_batch(
            processor=self.processor,
            tokenizer=self.tokenizer,
            prompt_texts=prompt_texts,
            images=images,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            input_mode=self.input_mode,
        )

    @staticmethod
    def _processor_image_rows(batch: list[dict[str, Any]]) -> list[Any]:
        rows: list[Any] = []
        for item in batch:
            raw_image_paths = item.get("image_paths")
            if isinstance(raw_image_paths, (str, bytes, Path)):
                image_paths = (str(raw_image_paths),)
            else:
                image_paths = tuple(raw_image_paths or ())
            canonical = (
                item.get("image")
                if len(image_paths) <= 1 and item.get("image") is not None
                else item.get("images")
            )
            if canonical is None:
                canonical = item.get("image")
            if isinstance(canonical, (list, tuple)):
                images = tuple(canonical)
            elif canonical is None:
                images = ()
            else:
                images = (canonical,)
            if not images:
                raise ValueError("A multimodal processor row requires at least one image.")
            rows.append(images[0] if len(images) == 1 else images)
        return rows

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
        plans: list[ShaftTemplatePromptPlan | ShaftTemplateSupervisionPlan],
        processed_batch: ShaftProcessedBatch,
        max_length: int | None = None,
    ) -> list[ShaftProcessorTokenLayout | None]:
        resolved_max_length = self.max_length if max_length is None else int(max_length)
        layouts: list[ShaftProcessorTokenLayout | None] = []
        for row_index, plan in enumerate(plans):
            prefix_length = int(
                processed_batch.model_inputs["attention_mask"][row_index].sum().item()
            )
            needs_truncation_layout = bool(
                resolved_max_length is not None
                and prefix_length >= int(resolved_max_length)
                and plan.truncatable_prefix_spans
            )
            if not getattr(plan, "trainable_prefix_spans", ()) and not needs_truncation_layout:
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

    @staticmethod
    def _build_processor_sequence_rows(
        *,
        processed_batch: ShaftProcessedBatch,
        rows: list[ShaftTemplatePromptRow | ShaftTemplateSupervisedRow],
        processor_row_indices: tuple[int, ...],
    ) -> list[dict[str, torch.Tensor]]:
        if len(rows) != len(processor_row_indices):
            raise ValueError("Processor row indices must align with template rows.")
        output: list[dict[str, torch.Tensor]] = []
        for row, processor_row_index in zip(rows, processor_row_indices, strict=True):
            prefix_length = len(row.processed_prefix_indices)
            continuation_length = int(row.input_ids.shape[0]) - prefix_length
            if continuation_length < 0:
                raise ValueError("Template row is shorter than its retained processor prefix.")
            output.append(
                processed_batch.build_processor_sequence_row(
                    row_index=processor_row_index,
                    prefix_indices=row.processed_prefix_indices,
                    continuation_length=continuation_length,
                )
            )
        return output


class SFTCollator(ShaftSequenceCollatorBase):
    SHAFT_INPUT_POLICY_VERSION = "shaft-sft-collator-input-v3-structured-truncation"

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
        input_mode: str = "training",
        loss_scale_name: str = "default",
        layout: str = "padded",
        packing_mode: str = "none",
        collect_stats: bool = True,
        pixel_budgets_by_dataset: Mapping[
            str,
            tuple[int | None, int | None],
        ]
        | None = None,
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
            input_mode=input_mode,
            loss_scale_name=loss_scale_name,
            pixel_budgets_by_dataset=pixel_budgets_by_dataset,
        )
        self.include_targets_in_inputs = bool(include_targets_in_inputs)
        self.include_metadata = bool(include_metadata)
        self.layout = str(layout).strip().lower()
        self.packing_mode = str(packing_mode).strip().lower()
        self.collect_stats = bool(collect_stats)
        if self.layout not in {"padded", "varlen"}:
            raise ValueError(f"Unsupported SFT collator layout: {self.layout!r}.")
        if self.packing_mode not in {"none", "greedy"}:
            raise ValueError(
                f"Unsupported SFT collator packing mode: {self.packing_mode!r}."
            )
        if self.packing_mode == "greedy" and self.layout != "varlen":
            raise ValueError("greedy packing requires the varlen collator layout.")
        if self.layout == "varlen" and self.padding_side != "right":
            raise ValueError("varlen SFT collation requires right-side sequence semantics.")

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        image_rows = self._processor_image_rows(batch)
        if self.layout == "varlen" and any(
            isinstance(row, (list, tuple)) and len(row) != 1
            for row in image_rows
        ):
            raise ValueError(
                "SFT multi-image rows currently require layout='padded'; "
                "varlen/packing remains fail-closed until its model runtime is validated."
            )
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
        processed_batch = self._run_processor(
            prompt_texts,
            image_rows,
            dataset_names=[item.get("dataset_name") for item in batch],
        )
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
        processor_sequence_rows = self._build_processor_sequence_rows(
            processed_batch=processed_batch,
            rows=rows,
            processor_row_indices=tuple(range(len(rows))),
        )
        varlen_plan = None
        if self.layout == "varlen":
            sequence_inputs, varlen_plan = ShaftVarlenBatchLayout.build(
                contexts=[item.get("_batch_context") for item in batch],
                input_ids=[row.input_ids for row in rows],
                labels=[row.labels for row in rows],
                loss_scales=[row.loss_scale for row in rows],
                ignore_index=self.ignore_index,
                max_sequence_length=self.max_length,
            )
            sequence_inputs.update(
                processed_batch.collate_processor_sequence_rows(
                    processor_sequence_rows,
                    layout="varlen",
                    padding_side=self.padding_side,
                )
            )
        else:
            sequence_inputs = {
                "input_ids": self._pad_sequences(
                    [row.input_ids for row in rows],
                    padding_value=int(pad_id),
                ),
                "attention_mask": self._pad_sequences(
                    [row.attention_mask for row in rows],
                    padding_value=0,
                ),
                "labels": self._pad_sequences(
                    [row.labels for row in rows],
                    padding_value=self.ignore_index,
                ),
            }
            loss_scale_rows = [
                row.loss_scale for row in rows if row.loss_scale is not None
            ]
            if loss_scale_rows:
                sequence_inputs["loss_scale"] = self._pad_sequences(
                    loss_scale_rows,
                    padding_value=0,
                ).to(dtype=torch.float32)
            sequence_inputs.update(
                processed_batch.collate_processor_sequence_rows(
                    processor_sequence_rows,
                    layout="padded",
                    padding_side=self.padding_side,
                )
            )
        out = self.model_adapter.assemble_processor_training_inputs(
            processed_batch=processed_batch,
            sequence_inputs=sequence_inputs,
            row_indices=tuple(range(len(batch))),
        )
        if varlen_plan is not None:
            out["_shaft_varlen_layout"] = varlen_plan
            if processed_batch.media_manifest is not None:
                out["_shaft_media_manifest"] = processed_batch.media_manifest
        if self.collect_stats:
            media_manifest = processed_batch.media_manifest
            out["_shaft_batch_stats"] = ShaftCollatedBatchStats.from_training_inputs(
                sequence_inputs=sequence_inputs,
                varlen_plan=varlen_plan,
                vision_patches=(
                    None
                    if media_manifest is None
                    else int(media_manifest.image_patch_count)
                ),
                ignore_index=self.ignore_index,
            )
        if self.include_metadata:
            out["meta"] = {
                "dataset_name": [item.get("dataset_name") for item in batch],
                "sample_id": [item["sample_id"] for item in batch],
                "image_path": [item.get("image_path") for item in batch],
                "image_paths": [tuple(item.get("image_paths") or ()) for item in batch],
                "target_text": [item["target_text"] for item in batch],
                "extra": [dict(item.get("extra", {})) for item in batch],
            }
        return out


class DPOCollator(ShaftSequenceCollatorBase):
    SHAFT_INPUT_POLICY_VERSION = "shaft-dpo-collator-input-v3-structured-truncation"

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
        images = self._processor_image_rows(batch)
        processed_batch = self._run_processor(
            prompt_texts,
            images,
            dataset_names=[item.get("dataset_name") for item in batch],
        )
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
        ordered_rows = [*chosen_rows, *rejected_rows]
        processor_row_indices = tuple(range(len(batch))) * 2
        processor_sequence_rows = self._build_processor_sequence_rows(
            processed_batch=processed_batch,
            rows=ordered_rows,
            processor_row_indices=processor_row_indices,
        )
        sequence_inputs.update(
            processed_batch.collate_processor_sequence_rows(
                processor_sequence_rows,
                layout="padded",
                padding_side=self.padding_side,
            )
        )
        return self.model_adapter.assemble_processor_training_inputs(
            processed_batch=processed_batch,
            sequence_inputs=sequence_inputs,
            row_indices=processor_row_indices,
        )


class PPOCollator(ShaftSequenceCollatorBase):
    SHAFT_INPUT_POLICY_VERSION = "shaft-ppo-collator-input-v1"

    # PPO batches are rollout prompts consumed by decoder-only generation, even
    # though they are produced by the training dataloader.
    DEFAULT_INPUT_MODE = "generation"

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
    SHAFT_INPUT_POLICY_VERSION = "shaft-grpo-collator-input-v1"

    def __init__(self, *, template: Template) -> None:
        self.template = template

    def __call__(self, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for item in batch:
            raw_images = item.get("images")
            if raw_images is None:
                raw_images = item.get("image")
            images = (
                tuple(raw_images)
                if isinstance(raw_images, (list, tuple))
                else (() if raw_images is None else (raw_images,))
            )
            if len(images) != 1:
                raise ValueError("GRPO currently requires exactly one image per sample.")
            prompt = self.template.prepare_messages(self.template.resolve_messages(item))
            rows.append(
                {
                    "prompt": prompt,
                    "image": images[0],
                    "target_text": str(item.get("target_text", "")),
                    "dataset_name": item.get("dataset_name"),
                    "sample_id": item.get("sample_id"),
                    "image_path": item.get("image_path"),
                    "image_paths": item.get("image_paths"),
                    "extra": dict(item.get("extra", {})),
                }
            )
        return rows
