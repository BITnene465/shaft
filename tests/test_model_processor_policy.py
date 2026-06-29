from __future__ import annotations

from shaft.model import build_model_meta


def test_processor_policy_controls_pixel_budget_forwarding() -> None:
    model_adapter = build_model_meta("qwen3vl").resolve_adapter(
        model_name_or_path="models/Qwen3-VL-4B-Instruct"
    )
    captured = {}

    class _Processor:
        def __call__(self, **kwargs):
            captured.update(kwargs)
            return {"ok": True}

    _ = model_adapter.build_processor_inputs(
        processor=_Processor(),
        prompt_texts=["hello"],
        images=["img"],
        min_pixels=16,
        max_pixels=32,
    )
    assert "min_pixels" not in captured
    assert "max_pixels" not in captured
    assert captured["images_kwargs"] == {"min_pixels": 16, "max_pixels": 32}


def test_processor_policy_can_disable_pixel_budget() -> None:
    model_adapter = build_model_meta("smoke_vlm").resolve_adapter(
        model_name_or_path="models/Smoke-VLM"
    )
    captured = {}

    class _Processor:
        def __call__(self, **kwargs):
            captured.update(kwargs)
            return {"ok": True}

    _ = model_adapter.build_processor_inputs(
        processor=_Processor(),
        prompt_texts=["hello"],
        images=["img"],
        min_pixels=16,
        max_pixels=32,
    )
    assert "min_pixels" not in captured
    assert "max_pixels" not in captured
    assert "images_kwargs" not in captured


def test_processor_policy_temporarily_controls_padding_side() -> None:
    model_adapter = build_model_meta("qwen3vl").resolve_adapter(
        model_name_or_path="models/Qwen3-VL-4B-Instruct"
    )
    captured = {}

    class _Tokenizer:
        def __init__(self) -> None:
            self.padding_side = "right"

    tokenizer = _Tokenizer()

    class _Processor:
        def __init__(self, tokenizer_obj) -> None:
            self.tokenizer = tokenizer_obj

        def __call__(self, **kwargs):
            captured["padding_side_during_call"] = self.tokenizer.padding_side
            return {"ok": True, "kwargs": kwargs}

    processor = _Processor(tokenizer)
    _ = model_adapter.build_processor_inputs(
        processor=processor,
        tokenizer=tokenizer,
        prompt_texts=["hello"],
        images=["img"],
        min_pixels=16,
        max_pixels=32,
        padding_side="left",
    )
    assert captured["padding_side_during_call"] == "left"
    assert tokenizer.padding_side == "right"
