from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from shaft.loss_scale import ShaftLossScaleSpec
from shaft.utils.messages import count_message_content_type

if TYPE_CHECKING:
    from shaft.model.types import ShaftProcessedBatch, ShaftProcessorTokenLayout
    from shaft.template.rendering import ShaftChatRenderer


def _item_image_count(item: dict[str, Any]) -> int:
    image_paths = item.get("image_paths")
    if image_paths is not None:
        if isinstance(image_paths, (str, bytes, Path)):
            return 1
        return len(tuple(image_paths))
    images = item.get("images")
    if images is not None:
        if isinstance(images, (str, bytes, Path)):
            return 1
        return len(tuple(images))
    image = item.get("image")
    if isinstance(image, (list, tuple)):
        return len(image)
    if image is not None or item.get("image_path") is not None:
        return 1
    return 0


def _message_image_count(messages: list[dict[str, Any]]) -> int:
    return count_message_content_type(messages, "image")


@dataclass(frozen=True)
class ShaftTemplatePromptPlan:
    prompt_text: str
    rendered_prefix_token_ids: tuple[int, ...] = ()
    truncatable_prefix_spans: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class ShaftTemplatePromptRow:
    input_ids: Any
    attention_mask: Any
    processed_prefix_indices: tuple[int, ...]


@dataclass(frozen=True)
class ShaftTemplateSupervisionPlan:
    prompt_text: str
    target_text: str
    loss_spec: ShaftLossScaleSpec
    rendered_prefix_token_ids: tuple[int, ...] = ()
    trainable_prefix_spans: tuple[tuple[int, int], ...] = ()
    truncatable_prefix_spans: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True, slots=True)
class ShaftSupervisionCostEstimate:
    llm_tokens: int
    supervised_tokens: int
    loss_weight_sum: float


@dataclass(frozen=True)
class ShaftTemplateSupervisedRow:
    input_ids: Any
    labels: Any
    attention_mask: Any
    processed_prefix_indices: tuple[int, ...]
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
        image_count = _item_image_count(item)
        if item.get("messages"):
            messages = item["messages"]
            placeholder_count = _message_image_count(messages)
            if placeholder_count != image_count:
                raise ValueError(
                    "Training message image placeholder count must match ordered "
                    f"image_paths: placeholders={placeholder_count}, images={image_count}."
                )
            return messages
        messages: list[dict[str, Any]] = []
        system_prompt = str(item.get("system_prompt", "")).strip()
        if system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        messages.append(
            {
                "role": "user",
                "content": [
                    *({"type": "image"} for _ in range(image_count)),
                    {"type": "text", "text": str(item.get("user_prompt", ""))},
                ],
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

    def build_prompt_plan(
        self,
        *,
        item: dict[str, Any],
        renderer: "ShaftChatRenderer",
    ) -> ShaftTemplatePromptPlan:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement structured prompt planning."
        )

    def build_prompt_row(
        self,
        *,
        plan: ShaftTemplatePromptPlan,
        tokenizer: Any,
        processed_batch: "ShaftProcessedBatch",
        row_index: int,
        prefix_token_layout: "ShaftProcessorTokenLayout | None",
        max_length: int | None = None,
    ) -> ShaftTemplatePromptRow:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement structured prompt rows."
        )

    @abstractmethod
    def estimate_supervision_cost(
        self,
        *,
        plan: ShaftTemplateSupervisionPlan,
        tokenizer: Any,
        prefix_token_layout: ShaftProcessorTokenLayout,
        add_eos_token: bool,
        max_length: int | None = None,
    ) -> ShaftSupervisionCostEstimate:
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
