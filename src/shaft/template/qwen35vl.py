from __future__ import annotations

from typing import Any

from .qwen import QwenChatTemplate
from .registry import register_template
from .types import TemplateMeta


class _Qwen35ArchitectureTemplateBase(QwenChatTemplate):
    def _prepare_target_text(self, *, item: dict[str, Any], target_text: str) -> str:
        options = dict(self.template_meta.chat_template_options)
        thinking_enabled = bool(options.get("enable_thinking", False))
        raw_reasoning = item.get("target_reasoning_content")
        reasoning_content = (
            "" if raw_reasoning is None else str(raw_reasoning).strip()
        )
        target_text = str(target_text)

        if not thinking_enabled:
            if reasoning_content:
                raise ValueError(
                    f"Qwen non-thinking template {self.name!r} cannot train "
                    "target_reasoning_content; select the matching *_thinking template."
                )
            return target_text

        if reasoning_content:
            if "<think>" in target_text or "</think>" in target_text:
                raise ValueError(
                    "Structured target_reasoning_content cannot be combined with think tags "
                    "inside target_text."
                )
            return f"{reasoning_content}\n</think>\n\n{target_text}"

        if "<think>" in target_text:
            raise ValueError(
                "Qwen thinking prompts already open <think>; preformatted target_text must "
                "start with the reasoning continuation and contain only the closing </think>."
            )
        if "</think>" not in target_text:
            raise ValueError(
                "Qwen thinking training requires target_reasoning_content or a preformatted "
                "target_text continuation containing </think>."
            )
        return target_text


@register_template(
    TemplateMeta(
        template_type="qwen35vl",
        template_cls=None,
        chat_template_options=(("enable_thinking", False),),
    )
)
class Qwen35VLTemplate(_Qwen35ArchitectureTemplateBase):
    name = "qwen35vl"


@register_template(
    TemplateMeta(
        template_type="qwen35vl_thinking",
        template_cls=None,
        chat_template_options=(("enable_thinking", True),),
    )
)
class Qwen35VLThinkingTemplate(_Qwen35ArchitectureTemplateBase):
    name = "qwen35vl_thinking"


@register_template(
    TemplateMeta(
        template_type="qwen36vl",
        template_cls=None,
        chat_template_options=(
            ("enable_thinking", False),
            ("preserve_thinking", False),
        ),
    )
)
class Qwen36VLTemplate(_Qwen35ArchitectureTemplateBase):
    name = "qwen36vl"


@register_template(
    TemplateMeta(
        template_type="qwen36vl_thinking",
        template_cls=None,
        chat_template_options=(
            ("enable_thinking", True),
            ("preserve_thinking", True),
        ),
    )
)
class Qwen36VLThinkingTemplate(_Qwen35ArchitectureTemplateBase):
    name = "qwen36vl_thinking"


@register_template(
    TemplateMeta(
        template_type="qwen38vl",
        template_cls=None,
        chat_template_options=(
            ("enable_thinking", False),
            ("preserve_thinking", False),
        ),
    )
)
class Qwen38VLTemplate(_Qwen35ArchitectureTemplateBase):
    name = "qwen38vl"


def _qwen38_thinking_options(
    reasoning_effort: str,
) -> tuple[tuple[str, bool | str], ...]:
    return (
        ("enable_thinking", True),
        ("preserve_thinking", True),
        ("reasoning_effort", reasoning_effort),
    )


@register_template(
    TemplateMeta(
        template_type="qwen38vl_thinking",
        template_cls=None,
        chat_template_options=_qwen38_thinking_options("xhigh"),
    )
)
class Qwen38VLThinkingTemplate(_Qwen35ArchitectureTemplateBase):
    name = "qwen38vl_thinking"


@register_template(
    TemplateMeta(
        template_type="qwen38vl_thinking_medium",
        template_cls=None,
        chat_template_options=_qwen38_thinking_options("medium"),
    )
)
class Qwen38VLMediumThinkingTemplate(_Qwen35ArchitectureTemplateBase):
    name = "qwen38vl_thinking_medium"


@register_template(
    TemplateMeta(
        template_type="qwen38vl_thinking_low",
        template_cls=None,
        chat_template_options=_qwen38_thinking_options("low"),
    )
)
class Qwen38VLLowThinkingTemplate(_Qwen35ArchitectureTemplateBase):
    name = "qwen38vl_thinking_low"
