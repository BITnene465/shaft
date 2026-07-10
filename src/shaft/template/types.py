from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from shaft.loss_scale import ShaftLossScaleSpec

if TYPE_CHECKING:
    from shaft.model.types import ShaftProcessedBatch, ShaftProcessorTokenLayout
    from shaft.template.rendering import ShaftChatRenderer


@dataclass(frozen=True)
class ShaftTemplateSupervisionPlan:
    prompt_text: str
    target_text: str
    loss_spec: ShaftLossScaleSpec
    rendered_prefix_token_ids: tuple[int, ...] = ()
    trainable_prefix_spans: tuple[tuple[int, int], ...] = ()


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
        renderer: ShaftChatRenderer,
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
        renderer: ShaftChatRenderer,
        loss_scale_name: str,
    ) -> ShaftTemplateSupervisionPlan:
        raise NotImplementedError

    @abstractmethod
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
