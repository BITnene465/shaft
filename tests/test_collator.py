from __future__ import annotations

import torch
from PIL import Image

from shaft.data import DPOCollator, PPOCollator, SFTCollator
from shaft.model import build_model_meta
from shaft.template import build_template


class _FakeTokenizer:
    eos_token_id = 2
    pad_token_id = 0
    eos_token = "</s>"

    def __call__(self, texts, add_special_tokens=False, return_attention_mask=False):
        ids = []
        for text in texts:
            token_count = max(len(str(text).split()), 1)
            ids.append([10 + i for i in range(token_count)])
        return {"input_ids": ids}


class _FakeProcessor:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        _ = tokenize, add_generation_prompt
        return " ".join(chunk.get("text", "") for m in messages for chunk in m.get("content", []))

    def __call__(self, text, images, padding=True, return_tensors="pt", **kwargs):
        _ = text, images, padding, return_tensors, kwargs
        input_ids = torch.tensor([[1, 2, 3], [1, 2, 0]], dtype=torch.long)
        attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.long)
        pixel_values = torch.randn(2, 3, 2, 2)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        }


def test_sft_collator_builds_labels() -> None:
    model_adapter = build_model_meta("smoke_vlm").resolve_adapter(model_name_or_path="models/Smoke-VLM")
    collator = SFTCollator(
        model_adapter=model_adapter,
        template=build_template("smoke_vlm"),
        processor=_FakeProcessor(),
        tokenizer=_FakeTokenizer(),
    )
    image = Image.new("RGB", (16, 16), color=(255, 255, 255))
    batch = [
        {
            "dataset_name": "a",
            "sample_id": "a1",
            "image_path": "/tmp/a.png",
            "image": image,
            "target_text": "{\"k\":1}",
            "messages": None,
            "system_prompt": "",
            "user_prompt": "Locate.",
            "extra": {},
        },
        {
            "dataset_name": "b",
            "sample_id": "b1",
            "image_path": "/tmp/b.png",
            "image": image,
            "target_text": "{\"k\":2}",
            "messages": None,
            "system_prompt": "",
            "user_prompt": "Locate.",
            "extra": {},
        },
    ]
    out = collator(batch)
    assert "labels" in out
    assert out["input_ids"].shape[0] == 2
    assert out["labels"].shape[0] == 2
    assert "dataset_name" in out["meta"]


def test_dpo_collator_builds_pairwise_batches() -> None:
    model_adapter = build_model_meta("smoke_vlm").resolve_adapter(model_name_or_path="models/Smoke-VLM")
    collator = DPOCollator(
        model_adapter=model_adapter,
        template=build_template("smoke_vlm"),
        processor=_FakeProcessor(),
        tokenizer=_FakeTokenizer(),
    )
    image = Image.new("RGB", (16, 16), color=(255, 255, 255))
    batch = [
        {
            "dataset_name": "a",
            "sample_id": "a1",
            "image_path": "/tmp/a.png",
            "image": image,
            "chosen_text": "{\"ok\":1}",
            "rejected_text": "{\"ok\":0}",
            "messages": None,
            "system_prompt": "",
            "user_prompt": "Locate.",
            "extra": {},
        },
        {
            "dataset_name": "a",
            "sample_id": "a2",
            "image_path": "/tmp/a2.png",
            "image": image,
            "chosen_text": "{\"ok\":1}",
            "rejected_text": "{\"ok\":0}",
            "messages": None,
            "system_prompt": "",
            "user_prompt": "Locate.",
            "extra": {},
        },
    ]
    out = collator(batch)
    assert out["input_ids"].shape[0] == 4
    assert out["attention_mask"].shape == out["input_ids"].shape
    assert out["completion_mask"].shape == out["input_ids"].shape
    assert out["pixel_values"].shape[0] == 4


def test_ppo_collator_builds_query_only_batch() -> None:
    model_adapter = build_model_meta("smoke_vlm").resolve_adapter(model_name_or_path="models/Smoke-VLM")
    collator = PPOCollator(
        model_adapter=model_adapter,
        template=build_template("smoke_vlm"),
        processor=_FakeProcessor(),
        tokenizer=_FakeTokenizer(),
    )
    image = Image.new("RGB", (16, 16), color=(255, 255, 255))
    batch = [
        {
            "dataset_name": "a",
            "sample_id": "a1",
            "image_path": "/tmp/a.png",
            "image": image,
            "messages": None,
            "system_prompt": "",
            "user_prompt": "Locate.",
            "extra": {},
        },
        {
            "dataset_name": "a",
            "sample_id": "a2",
            "image_path": "/tmp/a2.png",
            "image": image,
            "messages": None,
            "system_prompt": "",
            "user_prompt": "Locate.",
            "extra": {},
        },
    ]
    out = collator(batch)
    assert out["input_ids"].shape[0] == 2
    assert out["attention_mask"].shape == out["input_ids"].shape
