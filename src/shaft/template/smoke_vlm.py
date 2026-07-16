from __future__ import annotations

from .delimited import ShaftDelimitedChatTemplate
from .registry import register_template
from .types import TemplateMeta


@register_template(TemplateMeta(template_type="smoke_vlm", template_cls=None))
class SmokeVLMTemplate(ShaftDelimitedChatTemplate):
    name = "smoke_vlm"
    assistant_start = "<|smoke_start|>assistant\n"
    message_end = "<|smoke_end|>\n"
