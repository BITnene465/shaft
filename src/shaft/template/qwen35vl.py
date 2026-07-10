from __future__ import annotations

from typing import Any

from .qwen import QwenChatTemplate
from .registry import register_template
from .types import TemplateMeta


class _Qwen35VLTemplateBase(QwenChatTemplate):
    enable_thinking: bool = False
    preserve_thinking: bool = False

    def _chat_template_options(self) -> dict[str, Any]:
        return {
            "enable_thinking": bool(self.enable_thinking),
            "preserve_thinking": bool(self.preserve_thinking),
        }


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
