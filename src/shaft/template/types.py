from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from shaft.loss_scale import ShaftLossScaleSpec


@dataclass(frozen=True)
class ShaftTemplateMessagePlan:
    message: dict[str, Any]
    trainable: bool = False


@dataclass(frozen=True)
class ShaftTemplateSupervisionPlan:
    messages: list[dict[str, Any]]
    prompt_text: str
    target_text: str
    loss_spec: ShaftLossScaleSpec
    message_plans: list[ShaftTemplateMessagePlan]


@dataclass(frozen=True)
class ShaftTemplateSupervisedRow:
    input_ids: Any
    labels: Any
    attention_mask: Any
    mm_token_type_ids: Any | None = None
    loss_scale: Any | None = None


class Template(ABC):
    def __init__(self, template_meta: "TemplateMeta") -> None:
        self.template_meta = template_meta
        self.name = template_meta.template_type

    def prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized = list(messages)
        if self.template_meta.default_system and not any(str(msg.get("role", "")).strip().lower() == "system" for msg in normalized):
            normalized = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.template_meta.default_system}],
                },
                *normalized,
            ]
        return normalized

    def resolve_messages(self, item: dict[str, Any]) -> list[dict[str, Any]]:
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

    @abstractmethod
    def apply_chat_template(
        self,
        *,
        processor: Any,
        tokenizer: Any,
        messages: list[dict[str, Any]],
        add_generation_prompt: bool | None = None,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def decode(self, *, tokenizer: Any, token_ids: list[int]) -> str:
        raise NotImplementedError

    @abstractmethod
    def build_supervision_plan(
        self,
        *,
        item: dict[str, Any],
        target_text: str,
        processor: Any,
        tokenizer: Any,
        loss_scale_name: str,
    ) -> ShaftTemplateSupervisionPlan:
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError


@dataclass(frozen=True)
class TemplateMeta:
    template_type: str
    template_cls: type[Template] | None
    default_system: str | None = None
    stop_words: tuple[str, ...] = field(default_factory=tuple)
    support_multi_round: bool = True
    auto_add_generation_prompt: bool = True
    response_prefix: str = ""
    thinking_prefix: str = ""
