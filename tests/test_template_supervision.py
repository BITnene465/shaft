from __future__ import annotations

import torch
from PIL import Image

from shaft.model import build_model_meta
from shaft.template import build_template


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
    template = build_template("smoke_vlm")
    processor = _FakeProcessor()
    tokenizer = _FakeTokenizer()
    item = _build_item()

    default_plan = template.build_supervision_plan(
        item=item,
        target_text=item["target_text"],
        processor=processor,
        tokenizer=tokenizer,
        loss_scale_name="default",
    )
    last_round_plan = template.build_supervision_plan(
        item=item,
        target_text=item["target_text"],
        processor=processor,
        tokenizer=tokenizer,
        loss_scale_name="last_round",
    )
    all_plan = template.build_supervision_plan(
        item=item,
        target_text=item["target_text"],
        processor=processor,
        tokenizer=tokenizer,
        loss_scale_name="all",
    )

    assert [segment.trainable for segment in default_plan.message_plans] == [False, True, False]
    assert [segment.trainable for segment in last_round_plan.message_plans] == [False, False, False]
    assert [segment.trainable for segment in all_plan.message_plans] == [True, True, True]


def test_template_build_supervised_row_controls_prefix_supervision() -> None:
    template = build_template("smoke_vlm")
    processor = _FakeProcessor()
    tokenizer = _FakeTokenizer()
    model_adapter = build_model_meta("smoke_vlm").resolve_adapter(model_name_or_path="models/Smoke-VLM")
    item = _build_item()

    plans = {
        name: template.build_supervision_plan(
            item=item,
            target_text=item["target_text"],
            processor=processor,
            tokenizer=tokenizer,
            loss_scale_name=name,
        )
        for name in ("default", "last_round", "all")
    }
    prefix_batch = model_adapter.build_processor_inputs(
        processor=processor,
        prompt_texts=[plans["default"].prompt_text],
        images=[item["image"]],
        min_pixels=None,
        max_pixels=None,
    )

    default_row = template.build_supervised_row(
        plan=plans["default"],
        model_adapter=model_adapter,
        processor=processor,
        tokenizer=tokenizer,
        image=item["image"],
        prefix_batch=prefix_batch,
        row_index=0,
        min_pixels=None,
        max_pixels=None,
        add_eos_token=True,
        ignore_index=-100,
        include_targets_in_inputs=True,
    )
    last_round_row = template.build_supervised_row(
        plan=plans["last_round"],
        model_adapter=model_adapter,
        processor=processor,
        tokenizer=tokenizer,
        image=item["image"],
        prefix_batch=prefix_batch,
        row_index=0,
        min_pixels=None,
        max_pixels=None,
        add_eos_token=True,
        ignore_index=-100,
        include_targets_in_inputs=True,
    )
    all_row = template.build_supervised_row(
        plan=plans["all"],
        model_adapter=model_adapter,
        processor=processor,
        tokenizer=tokenizer,
        image=item["image"],
        prefix_batch=prefix_batch,
        row_index=0,
        min_pixels=None,
        max_pixels=None,
        add_eos_token=True,
        ignore_index=-100,
        include_targets_in_inputs=True,
    )

    default_prefix_supervised = int(torch.sum(default_row.labels[:3].ne(-100)).item())
    last_round_prefix_supervised = int(torch.sum(last_round_row.labels[:3].ne(-100)).item())
    all_prefix_supervised = int(torch.sum(all_row.labels[:3].ne(-100)).item())

    assert default_prefix_supervised == 1
    assert last_round_prefix_supervised == 0
    assert all_prefix_supervised == 3
    assert int(default_row.labels[-1].item()) == tokenizer.eos_token_id
