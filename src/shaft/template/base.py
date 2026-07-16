from __future__ import annotations

from typing import Any

import torch

from shaft.loss_scale import build_loss_scale
from shaft.model.types import ShaftProcessedBatch, ShaftProcessorTokenLayout

from .types import (
    ShaftSupervisionCostEstimate,
    ShaftTemplateSupervisionPlan,
    ShaftTemplateSupervisedRow,
    Template,
)
from .rendering import ShaftChatRenderer


class ShaftChatTemplate(Template):
    name = "shaft_chat"

    def __init__(self, template_meta):
        super().__init__(template_meta)

    def apply_chat_template(
        self,
        *,
        renderer: ShaftChatRenderer,
        messages: list[dict[str, Any]],
        add_generation_prompt: bool | None = None,
    ) -> str:
        normalized_messages = self.prepare_messages(messages)
        resolved_add_generation_prompt = (
            self.template_meta.auto_add_generation_prompt
            if add_generation_prompt is None
            else bool(add_generation_prompt)
        )
        return renderer.render(
            messages=normalized_messages,
            add_generation_prompt=resolved_add_generation_prompt,
            options=self._chat_template_options(),
        )

    def _chat_template_options(self) -> dict[str, Any]:
        return {}

    def decode(self, *, tokenizer: Any, token_ids: list[int]) -> str:
        if hasattr(tokenizer, "decode"):
            return str(tokenizer.decode(token_ids, skip_special_tokens=True)).strip()
        if hasattr(tokenizer, "batch_decode"):
            decoded = tokenizer.batch_decode([token_ids], skip_special_tokens=True)
            if decoded:
                return str(decoded[0]).strip()
        return " ".join(str(x) for x in token_ids)

    def build_supervision_plan(
        self,
        *,
        item: dict[str, Any],
        target_text: str,
        renderer: ShaftChatRenderer,
        loss_scale_name: str,
    ) -> ShaftTemplateSupervisionPlan:
        messages = self.prepare_messages(self.resolve_messages(item))
        prompt_text = self.apply_chat_template(
            renderer=renderer,
            messages=messages,
        )
        loss_scale = build_loss_scale(loss_scale_name)
        loss_spec = loss_scale(item)
        rendered_prefix_token_ids: tuple[int, ...] = ()
        trainable_prefix_spans: tuple[tuple[int, int], ...] = ()
        if loss_spec.base_strategy == "default" and float(loss_spec.prefix_scale) > 0:
            assistant_indices: list[int] = []
            seen_user = False
            for index, message in enumerate(messages):
                role = str(message.get("role", "")).strip().lower()
                if role == "user":
                    seen_user = True
                elif role == "assistant" and seen_user:
                    assistant_indices.append(index)
            if assistant_indices:
                rendered_prefix_token_ids = tuple(
                    renderer.tokenize(prompt_text)
                )
                trainable_prefix_spans = self._build_trainable_prefix_spans(
                    messages=messages,
                    assistant_indices=assistant_indices,
                    rendered_prefix_token_ids=rendered_prefix_token_ids,
                    renderer=renderer,
                )
        return ShaftTemplateSupervisionPlan(
            prompt_text=prompt_text,
            target_text=str(target_text),
            loss_spec=loss_spec,
            rendered_prefix_token_ids=rendered_prefix_token_ids,
            trainable_prefix_spans=trainable_prefix_spans,
        )

    def _tokenize_target(self, *, tokenizer: Any, target_text: str) -> list[int]:
        tokenized = tokenizer(
            [target_text],
            add_special_tokens=False,
            return_attention_mask=False,
        )
        return list(tokenized["input_ids"][0])

    def _build_trainable_prefix_spans(
        self,
        *,
        messages: list[dict[str, Any]],
        assistant_indices: list[int],
        rendered_prefix_token_ids: tuple[int, ...],
        renderer: ShaftChatRenderer,
    ) -> tuple[tuple[int, int], ...]:
        _ = messages, assistant_indices, rendered_prefix_token_ids, renderer
        raise NotImplementedError(
            f"{type(self).__name__} must implement exact full-render assistant span compilation."
        )

    @staticmethod
    def _truncate_target_ids(
        target_ids: list[int],
        *,
        prefix_length: int,
        max_length: int | None,
        eos_id: int | None,
        add_eos_token: bool,
    ) -> list[int]:
        if max_length is None:
            output = list(target_ids)
            if add_eos_token and eos_id is not None and (not output or output[-1] != int(eos_id)):
                output.append(int(eos_id))
            return output

        budget = int(max_length) - int(prefix_length)
        if budget <= 0:
            return []

        eos_required = bool(
            add_eos_token
            and eos_id is not None
            and (not target_ids or target_ids[-1] != int(eos_id))
        )
        if len(target_ids) + int(eos_required) <= budget:
            output = list(target_ids)
            if eos_required:
                output.append(int(eos_id))
            return output

        # Truncated completions must not receive EOS: EOS would teach the model that a
        # partial target is a valid stopping point.
        return list(target_ids[:budget])

    def _compute_prefix_loss_scale(
        self,
        *,
        plan: ShaftTemplateSupervisionPlan,
        prefix_ids: torch.Tensor,
        prefix_token_layout: ShaftProcessorTokenLayout | None,
    ) -> torch.Tensor:
        loss_spec = plan.loss_spec
        prefix_length = int(prefix_ids.shape[0])
        weights = torch.zeros((prefix_length,), dtype=torch.float32)
        for start, end in self._resolve_prefix_supervision_spans(
            plan=plan,
            prefix_length=prefix_length,
            prefix_token_layout=prefix_token_layout,
        ):
            weights[start:end] = float(loss_spec.prefix_scale)
        return weights

    @staticmethod
    def _resolve_prefix_supervision_spans(
        *,
        plan: ShaftTemplateSupervisionPlan,
        prefix_length: int,
        prefix_token_layout: ShaftProcessorTokenLayout | None,
    ) -> tuple[tuple[int, int], ...]:
        loss_spec = plan.loss_spec
        prefix_length = int(prefix_length)
        if prefix_length <= 0 or float(loss_spec.prefix_scale) <= 0:
            return ()
        if loss_spec.base_strategy == "all":
            return ((0, prefix_length),)
        if loss_spec.base_strategy == "last_round" or not plan.trainable_prefix_spans:
            return ()
        if prefix_token_layout is None:
            raise ValueError(
                "A processor token layout is required for segmented prefix supervision."
            )
        if prefix_token_layout.rendered_token_count != len(plan.rendered_prefix_token_ids):
            raise ValueError("Processor token layout does not match the rendered prompt length.")
        if prefix_token_layout.processed_token_count != prefix_length:
            raise ValueError("Processor token layout does not match the processed prefix length.")

        projected = sorted(
            prefix_token_layout.project_span(raw_start, raw_end)
            for raw_start, raw_end in plan.trainable_prefix_spans
        )
        merged: list[tuple[int, int]] = []
        for start, end in projected:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        return tuple(merged)

    def estimate_supervision_cost(
        self,
        *,
        plan: ShaftTemplateSupervisionPlan,
        tokenizer: Any,
        prefix_token_layout: ShaftProcessorTokenLayout,
        add_eos_token: bool,
        max_length: int | None = None,
    ) -> ShaftSupervisionCostEstimate:
        prefix_length = prefix_token_layout.processed_token_count
        target_ids = self._tokenize_target(
            tokenizer=tokenizer,
            target_text=plan.target_text,
        )
        target_ids = self._truncate_target_ids(
            target_ids,
            prefix_length=prefix_length,
            max_length=max_length,
            eos_id=getattr(tokenizer, "eos_token_id", None),
            add_eos_token=add_eos_token,
        )
        prefix_spans = self._resolve_prefix_supervision_spans(
            plan=plan,
            prefix_length=prefix_length,
            prefix_token_layout=prefix_token_layout,
        )
        supervised_prefix_tokens = sum(
            max(end - max(start, 1), 0) for start, end in prefix_spans
        )
        supervised_target_tokens = 0
        if float(plan.loss_spec.target_scale) > 0:
            supervised_target_tokens = max(
                len(target_ids) - int(prefix_length == 0),
                0,
            )
        return ShaftSupervisionCostEstimate(
            llm_tokens=prefix_length + len(target_ids),
            supervised_tokens=supervised_prefix_tokens + supervised_target_tokens,
            loss_weight_sum=(
                supervised_prefix_tokens * float(plan.loss_spec.prefix_scale)
                + supervised_target_tokens * float(plan.loss_spec.target_scale)
            ),
        )

    def build_supervised_row(
        self,
        *,
        plan: ShaftTemplateSupervisionPlan,
        tokenizer: Any,
        processed_batch: ShaftProcessedBatch,
        row_index: int,
        prefix_token_layout: ShaftProcessorTokenLayout | None,
        add_eos_token: bool,
        ignore_index: int,
        include_targets_in_inputs: bool,
        max_length: int | None = None,
    ) -> ShaftTemplateSupervisedRow:
        eos_id = getattr(tokenizer, "eos_token_id", None)
        model_inputs = processed_batch.model_inputs
        prefix_mask = model_inputs["attention_mask"][row_index].bool()
        prefix_ids = model_inputs["input_ids"][row_index][prefix_mask]
        mm_token_ids = model_inputs.get("mm_token_type_ids")
        prefix_mm = mm_token_ids[row_index][prefix_mask] if mm_token_ids is not None else None

        target_ids = self._tokenize_target(tokenizer=tokenizer, target_text=plan.target_text)
        target_ids = self._truncate_target_ids(
            target_ids,
            prefix_length=int(prefix_ids.shape[0]),
            max_length=max_length,
            eos_id=eos_id,
            add_eos_token=add_eos_token,
        )
        target_tensor = torch.tensor(target_ids, dtype=torch.long)

        if include_targets_in_inputs:
            prefix_loss_scale = self._compute_prefix_loss_scale(
                plan=plan,
                prefix_ids=prefix_ids,
                prefix_token_layout=prefix_token_layout,
            )
            input_ids = torch.cat([prefix_ids, target_tensor], dim=0)
            prefix_labels = (
                prefix_ids.clone()
                if torch.any(prefix_loss_scale > 0)
                else torch.full((prefix_ids.shape[0],), ignore_index, dtype=torch.long)
            )
            if prefix_labels.shape[0] > 0:
                prefix_labels = prefix_labels.masked_fill(prefix_loss_scale <= 0, ignore_index)
            target_labels = (
                target_tensor.clone()
                if float(plan.loss_spec.target_scale) > 0
                else torch.full((target_tensor.shape[0],), ignore_index, dtype=torch.long)
            )
            labels = torch.cat([prefix_labels, target_labels], dim=0)
            attention_mask = torch.ones_like(input_ids)
            mm_row = None
            if prefix_mm is not None:
                mm_row = torch.cat([prefix_mm, torch.zeros_like(target_tensor)], dim=0)
            loss_scale = torch.cat(
                [
                    prefix_loss_scale,
                    torch.full((target_tensor.shape[0],), float(plan.loss_spec.target_scale), dtype=torch.float32),
                ],
                dim=0,
            )
            if plan.loss_spec.is_binary:
                loss_scale = None
            return ShaftTemplateSupervisedRow(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
                mm_token_type_ids=mm_row,
                loss_scale=loss_scale,
            )

        input_ids = prefix_ids
        labels = torch.full((int(prefix_ids.shape[0]),), ignore_index, dtype=torch.long)
        attention_mask = torch.ones_like(prefix_ids)
        return ShaftTemplateSupervisedRow(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            mm_token_type_ids=prefix_mm,
            loss_scale=None,
        )
