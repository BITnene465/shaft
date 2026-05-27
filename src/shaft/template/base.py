from __future__ import annotations

from typing import Any

import torch

from shaft.loss_scale import build_loss_scale

from .types import (
    ShaftTemplateMessagePlan,
    ShaftTemplateSupervisionPlan,
    ShaftTemplateSupervisedRow,
    Template,
)


class ShaftChatTemplate(Template):
    name = "shaft_chat"

    def __init__(self, template_meta):
        super().__init__(template_meta)

    def apply_chat_template(
        self,
        *,
        processor: Any,
        tokenizer: Any,
        messages: list[dict[str, Any]],
        add_generation_prompt: bool | None = None,
    ) -> str:
        owner = processor if hasattr(processor, "apply_chat_template") else tokenizer
        normalized_messages = self.prepare_messages(messages)
        resolved_add_generation_prompt = (
            self.template_meta.auto_add_generation_prompt
            if add_generation_prompt is None
            else bool(add_generation_prompt)
        )
        return owner.apply_chat_template(
            normalized_messages,
            tokenize=False,
            add_generation_prompt=resolved_add_generation_prompt,
        )

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
        processor: Any,
        tokenizer: Any,
        loss_scale_name: str,
    ) -> ShaftTemplateSupervisionPlan:
        messages = self.prepare_messages(self.resolve_messages(item))
        prompt_text = self.apply_chat_template(
            processor=processor,
            tokenizer=tokenizer,
            messages=messages,
        )
        loss_scale = build_loss_scale(loss_scale_name)
        loss_spec = loss_scale(item)
        message_plans: list[ShaftTemplateMessagePlan] = []
        for message in messages:
            role = str(message.get("role", "")).strip().lower()
            if loss_spec.base_strategy == "all":
                trainable = True
            elif loss_spec.base_strategy == "last_round":
                trainable = False
            else:
                trainable = role == "assistant"
            message_plans.append(ShaftTemplateMessagePlan(message=dict(message), trainable=trainable))
        return ShaftTemplateSupervisionPlan(
            messages=messages,
            prompt_text=prompt_text,
            target_text=str(target_text),
            loss_spec=loss_spec,
            message_plans=message_plans,
        )

    def _tokenize_target(self, *, tokenizer: Any, target_text: str) -> list[int]:
        tokenized = tokenizer(
            [target_text],
            add_special_tokens=False,
            return_attention_mask=False,
        )
        return list(tokenized["input_ids"][0])

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
        model_adapter: Any,
        processor: Any,
        tokenizer: Any,
        image: Any,
        prefix_length: int,
        min_pixels: int | None,
        max_pixels: int | None,
    ) -> torch.Tensor:
        loss_spec = plan.loss_spec
        if prefix_length <= 0:
            return torch.zeros((0,), dtype=torch.float32)
        if float(loss_spec.prefix_scale) <= 0:
            return torch.zeros((prefix_length,), dtype=torch.float32)
        if loss_spec.base_strategy == "all":
            return torch.full((prefix_length,), float(loss_spec.prefix_scale), dtype=torch.float32)
        if loss_spec.base_strategy == "last_round":
            return torch.zeros((prefix_length,), dtype=torch.float32)

        weights = torch.zeros((prefix_length,), dtype=torch.float32)
        previous_length = 0
        for idx, message_plan in enumerate(plan.message_plans):
            is_final_prompt = (
                idx == len(plan.message_plans) - 1
                and str(message_plan.message.get("role", "")).strip().lower() != "assistant"
            )
            rendered = self.apply_chat_template(
                processor=processor,
                tokenizer=tokenizer,
                messages=plan.messages[: idx + 1],
                add_generation_prompt=is_final_prompt,
            )
            partial_batch = model_adapter.build_processor_inputs(
                processor=processor,
                prompt_texts=[rendered],
                images=[image],
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
            current_length = int(partial_batch["attention_mask"][0].sum().item())
            current_length = min(current_length, prefix_length)
            if current_length > previous_length and message_plan.trainable:
                weights[previous_length:current_length] = float(loss_spec.prefix_scale)
            previous_length = max(previous_length, current_length)
        return weights

    def build_supervised_row(
        self,
        *,
        plan: ShaftTemplateSupervisionPlan,
        model_adapter: Any,
        processor: Any,
        tokenizer: Any,
        image: Any,
        prefix_batch: dict[str, Any],
        row_index: int,
        min_pixels: int | None,
        max_pixels: int | None,
        add_eos_token: bool,
        ignore_index: int,
        include_targets_in_inputs: bool,
        max_length: int | None = None,
    ) -> ShaftTemplateSupervisedRow:
        eos_id = getattr(tokenizer, "eos_token_id", None)
        prefix_mask = prefix_batch["attention_mask"][row_index].bool()
        prefix_ids = prefix_batch["input_ids"][row_index][prefix_mask]
        mm_token_ids = prefix_batch.get("mm_token_type_ids")
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
                model_adapter=model_adapter,
                processor=processor,
                tokenizer=tokenizer,
                image=image,
                prefix_length=int(prefix_ids.shape[0]),
                min_pixels=min_pixels,
                max_pixels=max_pixels,
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
