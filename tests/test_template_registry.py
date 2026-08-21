from __future__ import annotations

import pytest

from shaft.model import build_model_meta
from shaft.template import (
    ShaftChatRenderer,
    build_template,
    build_template_meta,
    resolve_template_meta,
)


def _renderer(tokenizer) -> ShaftChatRenderer:
    return ShaftChatRenderer.from_components(processor=object(), tokenizer=tokenizer)


def test_build_template_returns_expected_name() -> None:
    template = build_template("smoke_vlm")
    assert template.name == "smoke_vlm"


def test_build_template_meta_returns_expected_fields() -> None:
    meta = build_template_meta("qwen3vl")
    assert meta.template_type == "qwen3vl"
    assert meta.template_cls.__name__ == "Qwen3VLTemplate"
    assert meta.default_system is None
    assert meta.support_multi_round is True
    assert meta.auto_add_generation_prompt is True
    assert meta.stop_words == ()


def test_resolve_template_meta_uses_model_default() -> None:
    model_meta = build_model_meta("qwen3vl")
    meta = resolve_template_meta(model_meta=model_meta)
    assert meta.template_type == "qwen3vl"


def test_resolve_template_meta_uses_qwen35vl_model_default() -> None:
    model_meta = build_model_meta("qwen35vl")
    meta = resolve_template_meta(model_meta=model_meta)
    assert meta.template_type == "qwen35vl"


def test_resolve_template_meta_uses_qwen38vl_non_thinking_default() -> None:
    model_meta = build_model_meta("qwen38vl")
    meta = resolve_template_meta(model_meta=model_meta)
    assert meta.template_type == "qwen38vl"


def test_resolve_template_meta_accepts_model_adapter() -> None:
    model_adapter = build_model_meta("smoke_vlm").resolve_adapter(model_name_or_path="models/Smoke-VLM")
    meta = resolve_template_meta(model_adapter=model_adapter)
    assert meta.template_type == "smoke_vlm"


def test_template_instance_carries_meta() -> None:
    template = build_template("smoke_vlm")
    assert template.template_meta.template_type == "smoke_vlm"
    assert template.template_meta.support_multi_round is True


def test_template_default_system_is_injected() -> None:
    meta = build_template_meta("smoke_vlm")
    template = meta.template_cls(meta.__class__(**{**meta.__dict__, "default_system": "You are system."}))

    class _Tokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            assert tokenize is False
            assert add_generation_prompt is True
            assert messages[0]["role"] == "system"
            return "ok"

    tokenizer = _Tokenizer()
    assert template.apply_chat_template(
        renderer=_renderer(tokenizer),
        messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
    ) == "ok"


@pytest.mark.parametrize(
    ("template_name", "expected_options"),
    [
        ("qwen35vl", {"enable_thinking": False}),
        ("qwen35vl_thinking", {"enable_thinking": True}),
        ("qwen36vl", {"enable_thinking": False, "preserve_thinking": False}),
        ("qwen36vl_thinking", {"enable_thinking": True, "preserve_thinking": True}),
        ("qwen38vl", {"enable_thinking": False, "preserve_thinking": False}),
        (
            "qwen38vl_thinking",
            {
                "enable_thinking": True,
                "preserve_thinking": True,
                "reasoning_effort": "xhigh",
            },
        ),
        (
            "qwen38vl_thinking_medium",
            {
                "enable_thinking": True,
                "preserve_thinking": True,
                "reasoning_effort": "medium",
            },
        ),
        (
            "qwen38vl_thinking_low",
            {
                "enable_thinking": True,
                "preserve_thinking": True,
                "reasoning_effort": "low",
            },
        ),
    ],
)
def test_qwen35_architecture_templates_forward_exact_product_options(
    template_name: str,
    expected_options: dict[str, object],
) -> None:
    template = build_template(template_name)
    captured = {}

    class _Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            captured["messages"] = messages
            captured["kwargs"] = kwargs
            return "ok"

    assert (
        template.apply_chat_template(
            renderer=_renderer(_Tokenizer()),
            messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        )
        == "ok"
    )
    assert captured["kwargs"]["tokenize"] is False
    assert captured["kwargs"]["add_generation_prompt"] is True
    assert {
        key: value
        for key, value in captured["kwargs"].items()
        if key not in {"tokenize", "add_generation_prompt"}
    } == expected_options


def test_template_respects_generation_prompt_flag() -> None:
    meta = build_template_meta("smoke_vlm")
    template = meta.template_cls(meta.__class__(**{**meta.__dict__, "auto_add_generation_prompt": False}))

    class _Tokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            assert add_generation_prompt is False
            return "ok"

    tokenizer = _Tokenizer()
    assert template.apply_chat_template(
        renderer=_renderer(tokenizer),
        messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
    ) == "ok"
