from __future__ import annotations

from .base import ShaftChatTemplate
from .registry import register_template
from .types import TemplateMeta


@register_template(TemplateMeta(template_type="smoke_vlm", template_cls=None))
class SmokeVLMTemplate(ShaftChatTemplate):
    name = "smoke_vlm"
