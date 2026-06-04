from __future__ import annotations

from typing import Any

from .base import ShaftChatTemplate
from .registry import register_template
from .types import TemplateMeta


class _Qwen35VLTemplateBase(ShaftChatTemplate):
    enable_thinking: bool = False
    preserve_thinking: bool = False

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
            enable_thinking=bool(self.enable_thinking),
            preserve_thinking=bool(self.preserve_thinking),
        )


@register_template(TemplateMeta(template_type="qwen35vl", template_cls=None))
class Qwen35VLTemplate(_Qwen35VLTemplateBase):
    name = "qwen35vl"
    enable_thinking = False
    preserve_thinking = False


@register_template(TemplateMeta(template_type="qwen35vl_thinking", template_cls=None))
class Qwen35VLThinkingTemplate(_Qwen35VLTemplateBase):
    name = "qwen35vl_thinking"
    enable_thinking = True
    preserve_thinking = True
