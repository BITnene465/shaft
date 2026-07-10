from __future__ import annotations

from shaft.model import build_model_meta
from shaft.template import (
    TEMPLATE_REGISTRY,
    ShaftChatRenderer,
    build_template,
    build_template_meta,
    resolve_template_meta,
)


def _renderer(tokenizer) -> ShaftChatRenderer:
    return ShaftChatRenderer.from_components(processor=object(), tokenizer=tokenizer)


def test_qwen3vl_template_registered() -> None:
    assert TEMPLATE_REGISTRY.has("qwen3vl")


def test_qwen35vl_templates_registered() -> None:
    assert TEMPLATE_REGISTRY.has("qwen35vl")
    assert TEMPLATE_REGISTRY.has("qwen35vl_thinking")


def test_smoke_template_registered() -> None:
    assert TEMPLATE_REGISTRY.has("smoke_vlm")


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


def test_qwen35vl_template_disables_thinking_by_default() -> None:
    template = build_template("qwen35vl")
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
    assert captured["kwargs"]["enable_thinking"] is False
    assert captured["kwargs"]["preserve_thinking"] is False


def test_qwen35vl_thinking_template_enables_thinking_explicitly() -> None:
    template = build_template("qwen35vl_thinking")
    captured = {}

    class _Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            captured["kwargs"] = kwargs
            return "ok"

    assert (
        template.apply_chat_template(
            renderer=_renderer(_Tokenizer()),
            messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        )
        == "ok"
    )
    assert captured["kwargs"]["enable_thinking"] is True
    assert captured["kwargs"]["preserve_thinking"] is True


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
