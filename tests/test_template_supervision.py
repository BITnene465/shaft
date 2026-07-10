from __future__ import annotations

import inspect

import pytest
import torch
from PIL import Image

from shaft.loss_scale import ShaftLossScaleSpec
from shaft.model import ShaftProcessedBatch, ShaftProcessorTokenLayout, build_model_meta
from shaft.template import (
    ShaftChatRenderer,
    ShaftTemplateSupervisionPlan,
    build_template,
)
from shaft.template.base import ShaftChatTemplate
from shaft.template.types import Template
from shaft.template.types import TemplateMeta


class _FakeTokenizer:
    eos_token_id = 2
    pad_token_id = 0
    eos_token = "</s>"

    def __call__(self, texts, add_special_tokens=False, return_attention_mask=False):
        _ = add_special_tokens, return_attention_mask
        ids = []
        for text in texts:
            token_count = max(len(str(text).split()), 1)
            ids.append([10 + i for i in range(token_count)])
        return {"input_ids": ids}


class _FakeProcessor:
    tokenizer = _FakeTokenizer()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        _ = tokenize, add_generation_prompt
        return " ".join(chunk.get("text", "") for m in messages for chunk in m.get("content", []))

    def __call__(self, text, images, padding=True, return_tensors="pt", **kwargs):
        _ = images, padding, return_tensors, kwargs
        tokenized = self.tokenizer(text, add_special_tokens=False, return_attention_mask=False)["input_ids"]
        batch_size = len(tokenized)
        max_len = max(len(ids) for ids in tokenized)
        input_ids = []
        attention_mask = []
        for ids in tokenized:
            row = list(ids) + [self.tokenizer.pad_token_id] * (max_len - len(ids))
            input_ids.append(row)
            attention_mask.append([1] * len(ids) + [0] * (max_len - len(ids)))
        pixel_values = torch.randn(batch_size, 3, 2, 2)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "pixel_values": pixel_values,
        }


class _StrictUserQueryProcessor(_FakeProcessor):
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        if not any(str(message.get("role", "")).strip().lower() == "user" for message in messages):
            raise RuntimeError("No user query found in messages.")
        return super().apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
        )


class _ChatMLTokenizer(_FakeTokenizer):
    def __init__(self) -> None:
        self._token_ids: dict[str, int] = {}

    def __call__(self, texts, add_special_tokens=False, return_attention_mask=False):
        _ = add_special_tokens, return_attention_mask
        if isinstance(texts, str):
            texts = [texts]
        rows = []
        for text in texts:
            row = []
            for token in str(text).split():
                if token not in self._token_ids:
                    self._token_ids[token] = 10 + len(self._token_ids)
                row.append(self._token_ids[token])
            rows.append(row)
        return {"input_ids": rows}


class _ChatMLProcessor:
    def __init__(self) -> None:
        self.tokenizer = _ChatMLTokenizer()
        self.render_count = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        _ = tokenize
        self.render_count += 1
        rendered = []
        for message in messages:
            role = str(message.get("role", "user"))
            text = " ".join(
                str(chunk.get("text", ""))
                for chunk in message.get("content", [])
                if chunk.get("type") == "text"
            )
            rendered.append(f"<|im_start|>{role}\n{text}\n<|im_end|>\n")
        if add_generation_prompt:
            rendered.append("<|im_start|>assistant\n")
        return "".join(rendered)

    def __call__(self, text, images, padding=True, return_tensors="pt", **kwargs):
        _ = images, padding, return_tensors, kwargs
        tokenized = self.tokenizer(text)["input_ids"]
        max_len = max(len(row) for row in tokenized)
        input_ids = [row + [self.tokenizer.pad_token_id] * (max_len - len(row)) for row in tokenized]
        attention_mask = [[1] * len(row) + [0] * (max_len - len(row)) for row in tokenized]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "pixel_values": torch.zeros((len(text), 3, 2, 2), dtype=torch.float32),
        }


def _renderer(processor, tokenizer) -> ShaftChatRenderer:
    return ShaftChatRenderer.from_components(processor=processor, tokenizer=tokenizer)


def test_supervised_row_api_cannot_reprocess_multimodal_inputs() -> None:
    parameters = set(inspect.signature(Template.build_supervised_row).parameters)
    assert parameters.isdisjoint(
        {"model_adapter", "processor", "image", "min_pixels", "max_pixels"}
    )
    assert "prefix_token_layout" in parameters

    plan_parameters = set(inspect.signature(Template.build_supervision_plan).parameters)
    assert plan_parameters.isdisjoint({"processor", "tokenizer", "image"})
    assert "renderer" in plan_parameters

    cost_parameters = set(inspect.signature(Template.estimate_supervision_cost).parameters)
    assert cost_parameters.isdisjoint({"model_adapter", "processor", "image"})
    assert "prefix_token_layout" in cost_parameters


def test_template_cost_estimate_matches_supervised_row_and_causal_shift() -> None:
    template = build_template("qwen3vl")
    tokenizer = _FakeTokenizer()
    plan = ShaftTemplateSupervisionPlan(
        prompt_text="prompt",
        target_text="alpha beta",
        loss_spec=ShaftLossScaleSpec(
            base_strategy="default",
            prefix_scale=0.5,
            target_scale=2.0,
        ),
        rendered_prefix_token_ids=(10, 99, 11, 12),
        trainable_prefix_spans=((1, 3),),
    )
    prefix_layout = ShaftProcessorTokenLayout(
        processed_boundaries=(0, 1, 5, 6, 7)
    )
    processed_batch = ShaftProcessedBatch(
        model_inputs={
            "input_ids": torch.arange(7, dtype=torch.long).unsqueeze(0),
            "attention_mask": torch.ones((1, 7), dtype=torch.long),
        },
        batch_size=1,
    )

    estimate = template.estimate_supervision_cost(
        plan=plan,
        tokenizer=tokenizer,
        prefix_token_layout=prefix_layout,
        add_eos_token=True,
    )
    row = template.build_supervised_row(
        plan=plan,
        tokenizer=tokenizer,
        processed_batch=processed_batch,
        row_index=0,
        prefix_token_layout=prefix_layout,
        add_eos_token=True,
        ignore_index=-100,
        include_targets_in_inputs=True,
    )
    shifted_valid = row.labels[1:].ne(-100)

    assert estimate.llm_tokens == int(row.attention_mask.sum()) == 10
    assert estimate.supervised_tokens == int(shifted_valid.sum()) == 8
    assert row.loss_scale is not None
    assert estimate.loss_weight_sum == pytest.approx(
        float(row.loss_scale[1:][shifted_valid].sum())
    )


def test_qwen_template_compiles_closed_chatml_assistant_spans() -> None:
    processor = _ChatMLProcessor()
    item = _build_item()

    plan = build_template("qwen3vl").build_supervision_plan(
        item=item,
        target_text=item["target_text"],
        renderer=_renderer(processor, processor.tokenizer),
        loss_scale_name="default",
    )

    assert plan.trainable_prefix_spans == ((3, 6),)
    assert len(plan.rendered_prefix_token_ids) == 10
    assert processor.render_count == 1


def test_qwen_template_compiles_multiple_assistant_spans_across_tool_messages() -> None:
    processor = _ChatMLProcessor()
    item = _build_item()
    item["messages"] = [
        {"role": "system", "content": [{"type": "text", "text": "be exact"}]},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "first"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "tool request"}]},
        {"role": "tool", "content": [{"type": "text", "text": "tool result"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "intermediate"}]},
        {"role": "user", "content": [{"type": "text", "text": "finish"}]},
    ]

    plan = build_template("qwen3vl").build_supervision_plan(
        item=item,
        target_text=item["target_text"],
        renderer=_renderer(processor, processor.tokenizer),
        loss_scale_name="default",
    )

    assert len(plan.trainable_prefix_spans) == 2
    assert plan.trainable_prefix_spans[0][1] <= plan.trainable_prefix_spans[1][0]
    assert processor.render_count == 1


def test_generic_chat_template_has_no_partial_render_supervision_fallback() -> None:
    processor = _ChatMLProcessor()
    item = _build_item()
    template = ShaftChatTemplate(TemplateMeta(template_type="generic", template_cls=None))

    with pytest.raises(NotImplementedError, match="exact full-render assistant span"):
        template.build_supervision_plan(
            item=item,
            target_text=item["target_text"],
            renderer=_renderer(processor, processor.tokenizer),
            loss_scale_name="default",
        )

    assert processor.render_count == 1


def _build_item():
    image = Image.new("RGB", (16, 16), color=(255, 255, 255))
    return {
        "dataset_name": "demo",
        "sample_id": "sample-1",
        "image_path": "/tmp/demo.png",
        "image": image,
        "target_text": "{\"round\":2}",
        "messages": [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "first"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "{\"round\":1}"}]},
            {"role": "user", "content": [{"type": "text", "text": "second"}]},
        ],
        "system_prompt": "",
        "user_prompt": "",
        "extra": {},
    }


def test_template_build_supervision_plan_marks_message_trainability() -> None:
    template = build_template("qwen3vl")
    processor = _ChatMLProcessor()
    tokenizer = processor.tokenizer
    item = _build_item()

    default_plan = template.build_supervision_plan(
        item=item,
        target_text=item["target_text"],
        renderer=_renderer(processor, tokenizer),
        loss_scale_name="default",
    )
    last_round_plan = template.build_supervision_plan(
        item=item,
        target_text=item["target_text"],
        renderer=_renderer(processor, tokenizer),
        loss_scale_name="last_round",
    )
    all_plan = template.build_supervision_plan(
        item=item,
        target_text=item["target_text"],
        renderer=_renderer(processor, tokenizer),
        loss_scale_name="all",
    )

    assert default_plan.trainable_prefix_spans == ((3, 6),)
    assert len(default_plan.rendered_prefix_token_ids) == 10
    assert last_round_plan.trainable_prefix_spans == ()
    assert all_plan.trainable_prefix_spans == ()


def test_template_build_supervised_row_controls_prefix_supervision() -> None:
    template = build_template("qwen3vl")
    processor = _ChatMLProcessor()
    tokenizer = processor.tokenizer
    model_adapter = build_model_meta("qwen3vl").resolve_adapter(
        model_name_or_path="models/Qwen3-VL-4B-Instruct"
    )
    item = _build_item()

    plans = {
        name: template.build_supervision_plan(
            item=item,
            target_text=item["target_text"],
            renderer=_renderer(processor, tokenizer),
            loss_scale_name=name,
        )
        for name in ("default", "last_round", "all")
    }
    processed_batch = model_adapter.build_processor_batch(
        processor=processor,
        prompt_texts=[plans["default"].prompt_text],
        images=[item["image"]],
        min_pixels=None,
        max_pixels=None,
    )
    default_layout = model_adapter.build_processor_token_layout(
        rendered_token_ids=plans["default"].rendered_prefix_token_ids,
        processed_batch=processed_batch,
        row_index=0,
    )

    default_row = template.build_supervised_row(
        plan=plans["default"],
        tokenizer=tokenizer,
        processed_batch=processed_batch,
        row_index=0,
        prefix_token_layout=default_layout,
        add_eos_token=True,
        ignore_index=-100,
        include_targets_in_inputs=True,
    )
    last_round_row = template.build_supervised_row(
        plan=plans["last_round"],
        tokenizer=tokenizer,
        processed_batch=processed_batch,
        row_index=0,
        prefix_token_layout=None,
        add_eos_token=True,
        ignore_index=-100,
        include_targets_in_inputs=True,
    )
    all_row = template.build_supervised_row(
        plan=plans["all"],
        tokenizer=tokenizer,
        processed_batch=processed_batch,
        row_index=0,
        prefix_token_layout=None,
        add_eos_token=True,
        ignore_index=-100,
        include_targets_in_inputs=True,
    )

    prefix_length = int(processed_batch.model_inputs["attention_mask"][0].sum())
    default_prefix_supervised = int(
        torch.sum(default_row.labels[:prefix_length].ne(-100)).item()
    )
    last_round_prefix_supervised = int(
        torch.sum(last_round_row.labels[:prefix_length].ne(-100)).item()
    )
    all_prefix_supervised = int(
        torch.sum(all_row.labels[:prefix_length].ne(-100)).item()
    )

    assert default_prefix_supervised == 3
    assert last_round_prefix_supervised == 0
    assert all_prefix_supervised == prefix_length
    assert int(default_row.labels[-1].item()) == tokenizer.eos_token_id


def test_template_maps_assistant_span_across_expanded_multimodal_tokens() -> None:
    template = build_template("qwen3vl")
    processor = _ChatMLProcessor()
    tokenizer = processor.tokenizer
    item = _build_item()
    plan = template.build_supervision_plan(
        item=item,
        target_text=item["target_text"],
        renderer=_renderer(processor, tokenizer),
        loss_scale_name="default",
    )
    rendered_ids = list(plan.rendered_prefix_token_ids)
    processed_ids = [rendered_ids[0], rendered_ids[0], rendered_ids[0], *rendered_ids[1:]]
    processed_batch = ShaftProcessedBatch(
        model_inputs={
            "input_ids": torch.tensor([processed_ids], dtype=torch.long),
            "attention_mask": torch.ones((1, len(processed_ids)), dtype=torch.long),
            "mm_token_type_ids": torch.tensor(
                [[1, 1, 1, *([0] * (len(rendered_ids) - 1))]],
                dtype=torch.long,
            ),
        },
        batch_size=1,
    )
    model_adapter = build_model_meta("qwen3vl").resolve_adapter(
        model_name_or_path="models/Qwen3-VL-4B-Instruct"
    )
    prefix_token_layout = model_adapter.build_processor_token_layout(
        rendered_token_ids=plan.rendered_prefix_token_ids,
        processed_batch=processed_batch,
        row_index=0,
    )

    row = template.build_supervised_row(
        plan=plan,
        tokenizer=tokenizer,
        processed_batch=processed_batch,
        row_index=0,
        prefix_token_layout=prefix_token_layout,
        add_eos_token=True,
        ignore_index=-100,
        include_targets_in_inputs=True,
    )

    projected_start, projected_end = prefix_token_layout.project_span(
        *plan.trainable_prefix_spans[0]
    )
    assert torch.all(row.labels[projected_start:projected_end].ne(-100))
    assert int(row.labels[:projected_start].ne(-100).sum()) == 0


def test_processor_policy_rejects_inexact_multimodal_prefix_alignment() -> None:
    template = build_template("qwen3vl")
    processor = _ChatMLProcessor()
    tokenizer = processor.tokenizer
    item = _build_item()
    plan = template.build_supervision_plan(
        item=item,
        target_text=item["target_text"],
        renderer=_renderer(processor, tokenizer),
        loss_scale_name="default",
    )
    model_adapter = build_model_meta("smoke_vlm").resolve_adapter(
        model_name_or_path="models/Smoke-VLM"
    )

    with pytest.raises(ValueError, match="register a model-specific processor policy"):
        model_adapter.build_processor_token_layout(
            rendered_token_ids=plan.rendered_prefix_token_ids,
            processed_batch=ShaftProcessedBatch(
                model_inputs={
                    "input_ids": torch.tensor([[99, 11, 12]], dtype=torch.long),
                    "attention_mask": torch.ones((1, 3), dtype=torch.long),
                },
                batch_size=1,
            ),
            row_index=0,
        )


def test_template_prefix_loss_scale_skips_system_only_prefix() -> None:
    template = build_template("smoke_vlm")
    processor = _StrictUserQueryProcessor()
    tokenizer = _FakeTokenizer()
    model_adapter = build_model_meta("smoke_vlm").resolve_adapter(model_name_or_path="models/Smoke-VLM")
    image = Image.new("RGB", (16, 16), color=(255, 255, 255))
    item = {
        "dataset_name": "demo",
        "sample_id": "sample-2",
        "image_path": "/tmp/demo.png",
        "image": image,
        "target_text": "{\"ok\":true}",
        "system_prompt": "Return compact JSON.",
        "user_prompt": "Locate the object.",
        "extra": {},
    }

    plan = template.build_supervision_plan(
        item=item,
        target_text=item["target_text"],
        renderer=_renderer(processor, tokenizer),
        loss_scale_name="default",
    )
    processed_batch = model_adapter.build_processor_batch(
        processor=processor,
        prompt_texts=[plan.prompt_text],
        images=[image],
        min_pixels=None,
        max_pixels=None,
    )
    row = template.build_supervised_row(
        plan=plan,
        tokenizer=tokenizer,
        processed_batch=processed_batch,
        row_index=0,
        prefix_token_layout=None,
        add_eos_token=True,
        ignore_index=-100,
        include_targets_in_inputs=True,
    )

    assert row.input_ids.shape[0] == row.labels.shape[0]
