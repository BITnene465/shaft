from __future__ import annotations

from typing import Any

import torch

from shaft.loss_scale import build_loss_scale
from shaft.model.types import ShaftProcessedBatch, ShaftProcessorTokenLayout

from .types import (
    ShaftSupervisionCostEstimate,
    ShaftTemplatePromptPlan,
    ShaftTemplatePromptRow,
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
        return dict(self.template_meta.chat_template_options)

    def _prepare_target_text(self, *, item: dict[str, Any], target_text: str) -> str:
        reasoning_content = item.get("target_reasoning_content")
        if reasoning_content is not None and str(reasoning_content).strip():
            raise ValueError(
                f"Template {self.name!r} does not support target_reasoning_content."
            )
        return str(target_text)

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
        prepared_target_text = self._prepare_target_text(
            item=item,
            target_text=str(target_text),
        )
        messages = self.prepare_messages(self.resolve_messages(item))
        prompt_plan = self._build_prompt_plan(messages=messages, renderer=renderer)
        loss_scale = build_loss_scale(loss_scale_name)
        loss_spec = loss_scale(item)
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
                trainable_prefix_spans = self._build_trainable_prefix_spans(
                    messages=messages,
                    assistant_indices=assistant_indices,
                    rendered_prefix_token_ids=prompt_plan.rendered_prefix_token_ids,
                    renderer=renderer,
                )
        return ShaftTemplateSupervisionPlan(
            prompt_text=prompt_plan.prompt_text,
            target_text=prepared_target_text,
            loss_spec=loss_spec,
            rendered_prefix_token_ids=prompt_plan.rendered_prefix_token_ids,
            trainable_prefix_spans=trainable_prefix_spans,
            truncatable_prefix_spans=prompt_plan.truncatable_prefix_spans,
        )

    def build_prompt_plan(
        self,
        *,
        item: dict[str, Any],
        renderer: ShaftChatRenderer,
    ) -> ShaftTemplatePromptPlan:
        messages = self.prepare_messages(self.resolve_messages(item))
        return self._build_prompt_plan(messages=messages, renderer=renderer)

    def _build_prompt_plan(
        self,
        *,
        messages: list[dict[str, Any]],
        renderer: ShaftChatRenderer,
    ) -> ShaftTemplatePromptPlan:
        prompt_text = self.apply_chat_template(
            renderer=renderer,
            messages=messages,
        )
        rendered_prefix_token_ids = tuple(renderer.tokenize(prompt_text))
        return ShaftTemplatePromptPlan(
            prompt_text=prompt_text,
            rendered_prefix_token_ids=rendered_prefix_token_ids,
            truncatable_prefix_spans=self._build_truncatable_prefix_spans(
                messages=messages,
                rendered_prefix_token_ids=rendered_prefix_token_ids,
                renderer=renderer,
            ),
        )

    def _build_truncatable_prefix_spans(
        self,
        *,
        messages: list[dict[str, Any]],
        rendered_prefix_token_ids: tuple[int, ...],
        renderer: ShaftChatRenderer,
    ) -> tuple[tuple[int, int], ...]:
        _ = messages, rendered_prefix_token_ids, renderer
        return ()

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

    @staticmethod
    def _resolve_prefix_limit(
        *,
        prefix_length: int,
        max_length: int | None,
        reserve_supervised_target: bool,
    ) -> int:
        prefix_length = int(prefix_length)
        if max_length is None:
            return prefix_length
        limit = int(max_length)
        if prefix_length < limit:
            return prefix_length
        if reserve_supervised_target:
            if limit < 2:
                raise ValueError(
                    "data.max_length must be at least 2 when target supervision is enabled."
                )
            return limit - 1
        return limit

    @staticmethod
    def _prefix_keep_indices(
        *,
        plan: ShaftTemplatePromptPlan | ShaftTemplateSupervisionPlan,
        tokenizer: Any,
        prefix_token_layout: ShaftProcessorTokenLayout | None,
        prefix_limit: int,
        prefix_length: int,
    ) -> tuple[int, ...]:
        prefix_limit = int(prefix_limit)
        prefix_length = int(prefix_length)
        if prefix_limit < 0 or prefix_limit > prefix_length:
            raise ValueError("Invalid processed prefix truncation boundary.")
        if prefix_limit == prefix_length:
            return tuple(range(prefix_length))
        if prefix_token_layout is None:
            raise ValueError(
                "data.max_length requires an exact processor token layout before prefix "
                "truncation can preserve chat structure and media alignment."
            )
        if prefix_token_layout.processed_token_count != prefix_length:
            raise ValueError("Processor token layout does not match the processed prefix length.")
        rendered_ids = plan.rendered_prefix_token_ids
        if prefix_token_layout.rendered_token_count != len(rendered_ids):
            raise ValueError("Processor token layout does not match the rendered prompt length.")
        if not plan.truncatable_prefix_spans:
            raise ValueError(
                "data.max_length cannot truncate this prefix while preserving chat structure; "
                "increase max_length or register exact template truncation spans."
            )

        special_ids = set(int(value) for value in (getattr(tokenizer, "all_special_ids", ()) or ()))
        for attr in ("bos_token_id", "eos_token_id", "pad_token_id"):
            value = getattr(tokenizer, attr, None)
            if value is not None:
                special_ids.add(int(value))

        removable: list[int] = []
        seen: set[int] = set()
        boundaries = prefix_token_layout.processed_boundaries
        for raw_start, raw_end in plan.truncatable_prefix_spans:
            if raw_start < 0 or raw_end <= raw_start or raw_end > len(rendered_ids):
                raise ValueError("Template produced an invalid prefix truncation span.")
            for raw_index in range(raw_start, raw_end):
                if int(rendered_ids[raw_index]) in special_ids:
                    continue
                processed_start = int(boundaries[raw_index])
                processed_end = int(boundaries[raw_index + 1])
                # Expanded canonical tokens are media placeholders under the supported
                # processor policies. They and their surrounding special tokens stay intact.
                if processed_end - processed_start != 1:
                    continue
                if prefix_token_layout.intersects_protected_span(
                    processed_start,
                    processed_end,
                ):
                    continue
                if processed_start not in seen:
                    seen.add(processed_start)
                    removable.append(processed_start)

        remove_count = prefix_length - prefix_limit
        if len(removable) < remove_count:
            raise ValueError(
                "data.max_length is too small for this sample while preserving chat structure "
                "and ordered media tokens. Increase max_length or reduce the media pixel budget."
            )
        removed = set(removable[:remove_count])
        return tuple(index for index in range(prefix_length) if index not in removed)

    @classmethod
    def _truncate_processed_prefix(
        cls,
        *,
        plan: ShaftTemplatePromptPlan | ShaftTemplateSupervisionPlan,
        tokenizer: Any,
        prefix_token_layout: ShaftProcessorTokenLayout | None,
        prefix_ids: torch.Tensor,
        prefix_loss_scale: torch.Tensor,
        prefix_limit: int,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
        if int(prefix_limit) == int(prefix_ids.shape[0]):
            keep = tuple(range(int(prefix_ids.shape[0])))
            return prefix_ids, prefix_loss_scale, keep
        keep = cls._prefix_keep_indices(
            plan=plan,
            tokenizer=tokenizer,
            prefix_token_layout=prefix_token_layout,
            prefix_limit=prefix_limit,
            prefix_length=int(prefix_ids.shape[0]),
        )
        if len(keep) == int(prefix_ids.shape[0]):
            return prefix_ids, prefix_loss_scale, keep
        index = torch.tensor(keep, dtype=torch.long, device=prefix_ids.device)
        return (
            prefix_ids.index_select(0, index),
            prefix_loss_scale.index_select(0, index),
            keep,
        )

    def build_prompt_row(
        self,
        *,
        plan: ShaftTemplatePromptPlan,
        tokenizer: Any,
        processed_batch: ShaftProcessedBatch,
        row_index: int,
        prefix_token_layout: ShaftProcessorTokenLayout | None,
        max_length: int | None = None,
    ) -> ShaftTemplatePromptRow:
        model_inputs = processed_batch.model_inputs
        prefix_mask = model_inputs["attention_mask"][row_index].bool()
        prefix_ids = model_inputs["input_ids"][row_index][prefix_mask]
        prefix_limit = self._resolve_prefix_limit(
            prefix_length=int(prefix_ids.shape[0]),
            max_length=max_length,
            reserve_supervised_target=False,
        )
        prefix_ids, _, prefix_indices = self._truncate_processed_prefix(
            plan=plan,
            tokenizer=tokenizer,
            prefix_token_layout=prefix_token_layout,
            prefix_ids=prefix_ids,
            prefix_loss_scale=torch.zeros(
                (int(prefix_ids.shape[0]),),
                dtype=torch.float32,
                device=prefix_ids.device,
            ),
            prefix_limit=prefix_limit,
        )
        if max_length is not None and int(prefix_ids.shape[0]) > int(max_length):
            raise RuntimeError("Template prompt exceeds strict max_length.")
        return ShaftTemplatePromptRow(
            input_ids=prefix_ids,
            attention_mask=torch.ones_like(prefix_ids),
            processed_prefix_indices=prefix_indices,
        )

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
        full_prefix_length = prefix_token_layout.processed_token_count
        target_ids = self._tokenize_target(
            tokenizer=tokenizer,
            target_text=plan.target_text,
        )
        unbounded_target_ids = self._truncate_target_ids(
            target_ids,
            prefix_length=0,
            max_length=None,
            eos_id=getattr(tokenizer, "eos_token_id", None),
            add_eos_token=add_eos_token,
        )
        prefix_length = self._resolve_prefix_limit(
            prefix_length=full_prefix_length,
            max_length=max_length,
            reserve_supervised_target=bool(
                unbounded_target_ids and float(plan.loss_spec.target_scale) > 0
            ),
        )
        keep: tuple[int, ...] | None = None
        if prefix_length != full_prefix_length:
            keep = self._prefix_keep_indices(
                plan=plan,
                tokenizer=tokenizer,
                prefix_token_layout=prefix_token_layout,
                prefix_limit=prefix_length,
                prefix_length=full_prefix_length,
            )
            prefix_length = len(keep)
        target_ids = self._truncate_target_ids(
            target_ids,
            prefix_length=prefix_length,
            max_length=max_length,
            eos_id=getattr(tokenizer, "eos_token_id", None),
            add_eos_token=add_eos_token,
        )
        prefix_spans = self._resolve_prefix_supervision_spans(
            plan=plan,
            prefix_length=full_prefix_length,
            prefix_token_layout=prefix_token_layout,
        )
        prefix_loss_weight = 0.0
        if keep is None:
            supervised_prefix_tokens = sum(
                max(end - max(start, 1), 0) for start, end in prefix_spans
            )
            prefix_loss_weight = (
                supervised_prefix_tokens * float(plan.loss_spec.prefix_scale)
            )
        elif prefix_spans:
            prefix_weights = torch.zeros((full_prefix_length,), dtype=torch.float32)
            for start, end in prefix_spans:
                prefix_weights[start:end] = float(plan.loss_spec.prefix_scale)
            selected_prefix_weights = prefix_weights[list(keep)]
            shifted_prefix_weights = selected_prefix_weights[1:]
            supervised_prefix_tokens = int(shifted_prefix_weights.gt(0).sum().item())
            prefix_loss_weight = float(shifted_prefix_weights.sum().item())
        else:
            supervised_prefix_tokens = 0
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
                prefix_loss_weight
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

        target_ids = self._tokenize_target(tokenizer=tokenizer, target_text=plan.target_text)
        unbounded_target_ids = self._truncate_target_ids(
            target_ids,
            prefix_length=0,
            max_length=None,
            eos_id=eos_id,
            add_eos_token=add_eos_token,
        )
        prefix_loss_scale = self._compute_prefix_loss_scale(
            plan=plan,
            prefix_ids=prefix_ids,
            prefix_token_layout=prefix_token_layout,
        )
        prefix_limit = self._resolve_prefix_limit(
            prefix_length=int(prefix_ids.shape[0]),
            max_length=max_length,
            reserve_supervised_target=bool(
                include_targets_in_inputs
                and unbounded_target_ids
                and float(plan.loss_spec.target_scale) > 0
            ),
        )
        prefix_ids, prefix_loss_scale, prefix_indices = self._truncate_processed_prefix(
            plan=plan,
            tokenizer=tokenizer,
            prefix_token_layout=prefix_token_layout,
            prefix_ids=prefix_ids,
            prefix_loss_scale=prefix_loss_scale,
            prefix_limit=prefix_limit,
        )
        target_ids = self._truncate_target_ids(
            target_ids,
            prefix_length=int(prefix_ids.shape[0]),
            max_length=max_length,
            eos_id=eos_id,
            add_eos_token=add_eos_token,
        )
        target_tensor = torch.tensor(target_ids, dtype=torch.long)

        if include_targets_in_inputs:
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
            loss_scale = torch.cat(
                [
                    prefix_loss_scale,
                    torch.full((target_tensor.shape[0],), float(plan.loss_spec.target_scale), dtype=torch.float32),
                ],
                dim=0,
            )
            if plan.loss_spec.is_binary:
                loss_scale = None
            if max_length is not None and int(input_ids.shape[0]) > int(max_length):
                raise RuntimeError("Template output exceeds strict data.max_length.")
            return ShaftTemplateSupervisedRow(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
                processed_prefix_indices=prefix_indices,
                loss_scale=loss_scale,
            )

        input_ids = prefix_ids
        labels = torch.full((int(prefix_ids.shape[0]),), ignore_index, dtype=torch.long)
        attention_mask = torch.ones_like(prefix_ids)
        if max_length is not None and int(input_ids.shape[0]) > int(max_length):
            raise RuntimeError("Template output exceeds strict data.max_length.")
        return ShaftTemplateSupervisedRow(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            processed_prefix_indices=prefix_indices,
            loss_scale=None,
        )
