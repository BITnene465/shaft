from __future__ import annotations

import torch
from PIL import Image

from shaft.data import DPOCollator, PPOCollator, SFTCollator
from shaft.loss_scale import LOSS_SCALE_REGISTRY, ShaftLossScale, ShaftLossScaleSpec, register_loss_scale
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


if not LOSS_SCALE_REGISTRY.has("test_weighted"):
    @register_loss_scale("test_weighted")
    class _TestWeightedLossScale(ShaftLossScale):
        is_binary = False

        def get_loss_scale(self, item):
            _ = item
            return ShaftLossScaleSpec(base_strategy="all", prefix_scale=0.25, target_scale=1.0)


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


def test_sft_collator_loss_scale_all_supervises_prefix_tokens() -> None:
    model_adapter = build_model_meta("smoke_vlm").resolve_adapter(model_name_or_path="models/Smoke-VLM")
    collator = SFTCollator(
        model_adapter=model_adapter,
        template=build_template("smoke_vlm"),
        processor=_FakeProcessor(),
        tokenizer=_FakeTokenizer(),
        loss_scale_name="all",
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
    ]
    out = collator(batch)
    assert "loss_scale" not in out
    prefix_len = 1
    assert torch.all(out["labels"][0, :prefix_len] != -100)


def test_sft_collator_emits_loss_scale_tensor_for_weighted_strategy() -> None:
    model_adapter = build_model_meta("smoke_vlm").resolve_adapter(model_name_or_path="models/Smoke-VLM")
    collator = SFTCollator(
        model_adapter=model_adapter,
        template=build_template("smoke_vlm"),
        processor=_FakeProcessor(),
        tokenizer=_FakeTokenizer(),
        loss_scale_name="test_weighted",
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
    ]
    out = collator(batch)
    assert "loss_scale" in out
    assert out["loss_scale"].dtype == torch.float32
    assert torch.any(torch.isclose(out["loss_scale"][0], torch.tensor(0.25, dtype=torch.float32)))


def test_sft_collator_appends_eos_to_inputs_and_labels() -> None:
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
            "target_text": "tail head",
            "messages": None,
            "system_prompt": "",
            "user_prompt": "Locate.",
            "extra": {},
        },
    ]
    out = collator(batch)
    assert int(out["input_ids"][0, -1].item()) == _FakeTokenizer.eos_token_id
    assert int(out["labels"][0, -1].item()) == _FakeTokenizer.eos_token_id


def test_sft_collator_truncates_target_tokens_without_eos_when_over_max_length() -> None:
    model_adapter = build_model_meta("smoke_vlm").resolve_adapter(model_name_or_path="models/Smoke-VLM")
    collator = SFTCollator(
        model_adapter=model_adapter,
        template=build_template("smoke_vlm"),
        processor=_FakeProcessor(),
        tokenizer=_FakeTokenizer(),
        max_length=4,
    )
    image = Image.new("RGB", (16, 16), color=(255, 255, 255))
    batch = [
        {
            "dataset_name": "a",
            "sample_id": "a1",
            "image_path": "/tmp/a.png",
            "image": image,
            "target_text": "one two three four five",
            "messages": None,
            "system_prompt": "",
            "user_prompt": "Locate.",
            "extra": {},
        },
    ]

    out = collator(batch)

    assert out["input_ids"].shape[1] == 4
    assert int(out["input_ids"][0, -1].item()) != _FakeTokenizer.eos_token_id
    assert int(out["labels"][0, -1].item()) != _FakeTokenizer.eos_token_id
    assert int(torch.sum(out["labels"][0].ne(-100)).item()) == 3


def test_sft_collator_default_supervises_previous_assistant_rounds() -> None:
    model_adapter = build_model_meta("smoke_vlm").resolve_adapter(model_name_or_path="models/Smoke-VLM")
    image = Image.new("RGB", (16, 16), color=(255, 255, 255))
    batch = [
        {
            "dataset_name": "a",
            "sample_id": "a1",
            "image_path": "/tmp/a.png",
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
        },
    ]
    default_collator = SFTCollator(
        model_adapter=model_adapter,
        template=build_template("smoke_vlm"),
        processor=_FakeProcessor(),
        tokenizer=_FakeTokenizer(),
        loss_scale_name="default",
    )
    last_round_collator = SFTCollator(
        model_adapter=model_adapter,
        template=build_template("smoke_vlm"),
        processor=_FakeProcessor(),
        tokenizer=_FakeTokenizer(),
        loss_scale_name="last_round",
    )
    default_out = default_collator(batch)
    last_round_out = last_round_collator(batch)
    default_prefix_supervised = int(torch.sum(default_out["labels"][0, :3].ne(-100)).item())
    last_round_prefix_supervised = int(torch.sum(last_round_out["labels"][0, :3].ne(-100)).item())
    assert default_prefix_supervised == 1
    assert last_round_prefix_supervised == 0


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
