from __future__ import annotations

from .base import ShaftChatTemplate
from .registry import register_template
from .types import TemplateMeta


@register_template(TemplateMeta(template_type="qwen3vl", template_cls=None))
class Qwen3VLTemplate(ShaftChatTemplate):
    name = "qwen3vl"
