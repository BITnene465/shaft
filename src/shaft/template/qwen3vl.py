from __future__ import annotations

from .qwen import QwenChatTemplate
from .registry import register_template
from .types import TemplateMeta


@register_template(TemplateMeta(template_type="qwen3vl", template_cls=None))
class Qwen3VLTemplate(QwenChatTemplate):
    name = "qwen3vl"
