from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from PIL import Image

from shaft.data import DPOCollator, PPOCollator, SFTCollator
from shaft.loss_scale import LOSS_SCALE_REGISTRY, ShaftLossScale, ShaftLossScaleSpec, register_loss_scale
from shaft.model import ProcessorPolicy, build_model_meta
from shaft.model.smoke_vlm import SmokeProcessor, SmokeTokenizer
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

    def __init__(self) -> None:
        self.call_count = 0
        self.last_kwargs = {}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        _ = tokenize, add_generation_prompt
        self.last_messages = messages
        return " ".join(chunk.get("text", "") for m in messages for chunk in m.get("content", []))

    def __call__(self, text, images, padding=True, return_tensors="pt", **kwargs):
        _ = images, padding, return_tensors
        self.call_count += 1
        self.last_kwargs = dict(kwargs)
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
            "model_specific_mask": torch.arange(batch_size, dtype=torch.long).unsqueeze(1),
        }


class _CountingSmokeProcessor(SmokeProcessor):
    def __init__(self) -> None:
        super().__init__(tokenizer=SmokeTokenizer())
        self.call_count = 0

    def __call__(self, *args, **kwargs):
        self.call_count += 1
        return super().__call__(*args, **kwargs)


def _build_smoke_adapter():
    adapter = build_model_meta("smoke_vlm").resolve_adapter(
        model_name_or_path="models/Smoke-VLM"
    )
    return replace(
        adapter,
        processor_policy=ProcessorPolicy(
            supports_pixel_budget=False,
            sample_aligned_model_input_names=("pixel_values", "model_specific_mask"),
        ),
    )


def _sft_item(
    *,
    sample_id: str,
    target_text: str = "answer",
    batch_context: dict[str, int | str] | None = None,
) -> dict[str, object]:
    item: dict[str, object] = {
        "dataset_name": "fixture",
        "sample_id": sample_id,
        "image_path": f"/tmp/{sample_id}.png",
        "image": Image.new("RGB", (16, 16), color=(255, 255, 255)),
        "target_text": target_text,
        "messages": None,
        "system_prompt": "",
        "user_prompt": "Locate.",
        "extra": {},
    }
    if batch_context is not None:
        item["_batch_context"] = batch_context
    return item


def _batch_context(
    *,
    pack_index: int,
    segment_index: int,
    pack_segment_count: int,
    global_microstep: int = 5,
    plan_fingerprint: str = "plan-v1",
    local_batch_id: int = 0,
) -> dict[str, int | str]:
    return {
        "global_microstep": global_microstep,
        "plan_fingerprint": plan_fingerprint,
        "local_batch_id": local_batch_id,
        "pack_index": pack_index,
        "segment_index": segment_index,
        "pack_segment_count": pack_segment_count,
    }


if not LOSS_SCALE_REGISTRY.has("test_weighted"):
    @register_loss_scale("test_weighted")
    class _TestWeightedLossScale(ShaftLossScale):
        is_binary = False

        def get_loss_scale(self, item):
            _ = item
            return ShaftLossScaleSpec(base_strategy="all", prefix_scale=0.25, target_scale=1.0)


if not LOSS_SCALE_REGISTRY.has("test_weighted_default"):
    @register_loss_scale("test_weighted_default")
    class _TestWeightedDefaultLossScale(ShaftLossScale):
        is_binary = False

        def get_loss_scale(self, item):
            _ = item
            return ShaftLossScaleSpec(
                base_strategy="default",
                prefix_scale=0.25,
                target_scale=1.0,
            )


def test_sft_collator_builds_labels() -> None:
    model_adapter = _build_smoke_adapter()
    processor = _FakeProcessor()
    collator = SFTCollator(
        model_adapter=model_adapter,
        template=build_template("smoke_vlm"),
        processor=processor,
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
    assert "meta" not in out
    assert processor.call_count == 1
    assert out["model_specific_mask"].flatten().tolist() == [0, 1]

    eval_collator = SFTCollator(
        model_adapter=model_adapter,
        template=build_template("smoke_vlm"),
        processor=_FakeProcessor(),
        tokenizer=_FakeTokenizer(),
        include_metadata=True,
    )
    eval_out = eval_collator(batch)
    assert "dataset_name" in eval_out["meta"]


def test_sft_collator_routes_resolved_eval_pixel_budget_by_dataset() -> None:
    adapter = _build_smoke_adapter()
    adapter = replace(
        adapter,
        processor_policy=replace(adapter.processor_policy, supports_pixel_budget=True),
    )
    processor = _FakeProcessor()
    collator = SFTCollator(
        model_adapter=adapter,
        template=build_template("smoke_vlm"),
        processor=processor,
        tokenizer=processor.tokenizer,
        min_pixels=100,
        max_pixels=1000,
        pixel_budgets_by_dataset={"fixture": (300, 3000)},
    )

    collator([_sft_item(sample_id="one")])

    assert processor.last_kwargs["images_kwargs"] == {
        "min_pixels": 300,
        "max_pixels": 3000,
    }


def test_sft_collator_without_dataset_overrides_accepts_missing_dataset_name() -> None:
    item = _sft_item(sample_id="legacy")
    item.pop("dataset_name")
    collator = SFTCollator(
        model_adapter=_build_smoke_adapter(),
        template=build_template("smoke_vlm"),
        processor=_FakeProcessor(),
        tokenizer=_FakeTokenizer(),
        include_metadata=True,
    )

    output = collator([item])

    assert output["input_ids"].shape[0] == 1
    assert output["meta"]["dataset_name"] == [None]


@pytest.mark.parametrize("dataset_name", [None, "unknown"])
def test_sft_collator_uses_default_budget_for_missing_or_unknown_dataset_name(
    dataset_name: str | None,
) -> None:
    item = _sft_item(sample_id="default")
    if dataset_name is None:
        item.pop("dataset_name")
    else:
        item["dataset_name"] = dataset_name
    adapter = _build_smoke_adapter()
    adapter = replace(
        adapter,
        processor_policy=replace(adapter.processor_policy, supports_pixel_budget=True),
    )
    processor = _FakeProcessor()
    collator = SFTCollator(
        model_adapter=adapter,
        template=build_template("smoke_vlm"),
        processor=processor,
        tokenizer=processor.tokenizer,
        min_pixels=100,
        max_pixels=1000,
        pixel_budgets_by_dataset={"fixture": (300, 3000)},
    )

    collator([item])

    assert processor.last_kwargs["images_kwargs"] == {
        "min_pixels": 100,
        "max_pixels": 1000,
    }


def test_sft_collator_allows_default_and_override_with_the_same_eval_budget() -> None:
    first = _sft_item(sample_id="first")
    second = _sft_item(sample_id="second")
    first.pop("dataset_name")
    second["dataset_name"] = "override"
    collator = SFTCollator(
        model_adapter=_build_smoke_adapter(),
        template=build_template("smoke_vlm"),
        processor=_FakeProcessor(),
        tokenizer=_FakeTokenizer(),
        min_pixels=300,
        max_pixels=3000,
        pixel_budgets_by_dataset={"override": (300, 3000)},
    )

    output = collator([first, second])

    assert output["input_ids"].shape[0] == 2


def test_sft_collator_rejects_default_and_override_with_different_eval_budgets() -> None:
    first = _sft_item(sample_id="first")
    second = _sft_item(sample_id="second")
    first.pop("dataset_name")
    second["dataset_name"] = "override"
    collator = SFTCollator(
        model_adapter=_build_smoke_adapter(),
        template=build_template("smoke_vlm"),
        processor=_FakeProcessor(),
        tokenizer=_FakeTokenizer(),
        min_pixels=300,
        max_pixels=3000,
        pixel_budgets_by_dataset={"override": (400, 4000)},
    )

    with pytest.raises(ValueError, match="cannot mix datasets"):
        collator([first, second])


def test_sft_varlen_collator_flattens_planned_segments_without_padding() -> None:
    adapter = _build_smoke_adapter()
    processor = _FakeProcessor()
    collator = SFTCollator(
        model_adapter=adapter,
        template=build_template("smoke_vlm"),
        processor=processor,
        tokenizer=_FakeTokenizer(),
        max_length=32,
        layout="varlen",
        packing_mode="greedy",
        loss_scale_name="test_weighted",
    )
    batch = [
        _sft_item(
            sample_id="a",
            target_text="one two",
            batch_context=_batch_context(
                pack_index=0,
                segment_index=0,
                pack_segment_count=2,
            ),
        ),
        _sft_item(
            sample_id="b",
            target_text="three",
            batch_context=_batch_context(
                pack_index=0,
                segment_index=1,
                pack_segment_count=2,
            ),
        ),
        _sft_item(
            sample_id="c",
            target_text="four five six",
            batch_context=_batch_context(
                pack_index=1,
                segment_index=0,
                pack_segment_count=1,
            ),
        ),
    ]

    out = collator(batch)

    assert out["input_ids"].shape[0] == 1
    assert out["labels"].shape == out["input_ids"].shape
    assert out["loss_scale"].shape == out["input_ids"].shape
    assert "attention_mask" not in out
    layout = out["_shaft_varlen_layout"]
    assert layout.physical_pack_count == 2
    assert layout.logical_segment_count == 3
    assert layout.pack_lengths == tuple(
        sum(segment.length for segment in layout.segments if segment.pack_index == pack)
        for pack in range(2)
    )
    assert int(out["input_ids"].shape[1]) == sum(
        segment.length for segment in layout.segments
    )
    for segment in layout.segments:
        assert int(out["labels"][0, segment.start].item()) == -100
        assert float(out["loss_scale"][0, segment.start].item()) == 0.0


@pytest.mark.parametrize(
    "broken_context",
    [
        None,
        _batch_context(
            pack_index=0,
            segment_index=0,
            pack_segment_count=2,
        ),
        _batch_context(
            pack_index=1,
            segment_index=0,
            pack_segment_count=1,
            plan_fingerprint="other-plan",
        ),
    ],
)
def test_sft_varlen_collator_rejects_missing_or_inconsistent_plan_context(
    broken_context: dict[str, int | str] | None,
) -> None:
    collator = SFTCollator(
        model_adapter=_build_smoke_adapter(),
        template=build_template("smoke_vlm"),
        processor=_FakeProcessor(),
        tokenizer=_FakeTokenizer(),
        max_length=32,
        layout="varlen",
        packing_mode="greedy",
    )
    batch = [
        _sft_item(
            sample_id="a",
            batch_context=_batch_context(
                pack_index=0,
                segment_index=0,
                pack_segment_count=2,
            ),
        ),
        _sft_item(sample_id="b", batch_context=broken_context),
    ]

    with pytest.raises(ValueError, match="varlen.*plan|plan.*varlen"):
        collator(batch)


def test_sft_collator_loss_scale_all_supervises_prefix_tokens() -> None:
    model_adapter = _build_smoke_adapter()
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
    model_adapter = _build_smoke_adapter()
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
    model_adapter = _build_smoke_adapter()
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
    model_adapter = _build_smoke_adapter()
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


def test_sft_collator_uses_per_row_dynamic_prefix_length_for_target_budget() -> None:
    model_adapter = _build_smoke_adapter()
    collator = SFTCollator(
        model_adapter=model_adapter,
        template=build_template("smoke_vlm"),
        processor=_FakeProcessor(),
        tokenizer=_FakeTokenizer(),
        max_length=5,
    )
    image = Image.new("RGB", (16, 16), color=(255, 255, 255))
    batch = [
        {
            "dataset_name": "a",
            "sample_id": "short-prefix",
            "image_path": "/tmp/a.png",
            "image": image,
            "target_text": "one two three four five",
            "messages": None,
            "system_prompt": "",
            "user_prompt": "short",
            "extra": {},
        },
        {
            "dataset_name": "b",
            "sample_id": "long-prefix",
            "image_path": "/tmp/b.png",
            "image": image,
            "target_text": "one two three four five",
            "messages": None,
            "system_prompt": "",
            "user_prompt": "alpha beta gamma delta",
            "extra": {},
        },
    ]

    out = collator(batch)

    supervised_counts = [int(row.ne(-100).sum().item()) for row in out["labels"]]
    assert supervised_counts == [4, 1]
    assert out["input_ids"].shape == torch.Size([2, 5])
    assert int(out["labels"][0, -1].item()) != _FakeTokenizer.eos_token_id
    assert int(out["labels"][1, -1].item()) != _FakeTokenizer.eos_token_id


def test_sft_collator_default_supervises_previous_assistant_rounds() -> None:
    model_adapter = _build_smoke_adapter()
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
    default_processor = _CountingSmokeProcessor()
    default_collator = SFTCollator(
        model_adapter=model_adapter,
        template=build_template("smoke_vlm"),
        processor=default_processor,
        tokenizer=default_processor.tokenizer,
        loss_scale_name="default",
    )
    last_round_processor = _CountingSmokeProcessor()
    last_round_collator = SFTCollator(
        model_adapter=model_adapter,
        template=build_template("smoke_vlm"),
        processor=last_round_processor,
        tokenizer=last_round_processor.tokenizer,
        loss_scale_name="last_round",
    )
    default_out = default_collator(batch)
    last_round_out = last_round_collator(batch)
    default_supervised = int(torch.sum(default_out["labels"][0].ne(-100)).item())
    last_round_supervised = int(torch.sum(last_round_out["labels"][0].ne(-100)).item())
    assert default_supervised > last_round_supervised
    assert default_processor.call_count == 1
    assert last_round_processor.call_count == 1


def test_sft_collator_weighted_default_scales_historical_assistant_once() -> None:
    model_adapter = _build_smoke_adapter()
    processor = _CountingSmokeProcessor()
    image = Image.new("RGB", (16, 16), color=(255, 255, 255))
    collator = SFTCollator(
        model_adapter=model_adapter,
        template=build_template("smoke_vlm"),
        processor=processor,
        tokenizer=processor.tokenizer,
        loss_scale_name="test_weighted_default",
    )

    out = collator(
        [
            {
                "dataset_name": "a",
                "sample_id": "a1",
                "image_path": "/tmp/a.png",
                "image": image,
                "target_text": "answer two",
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "image"}, {"type": "text", "text": "first"}],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "answer one"}],
                    },
                    {"role": "user", "content": [{"type": "text", "text": "second"}]},
                ],
                "system_prompt": "",
                "user_prompt": "",
                "extra": {},
            }
        ]
    )

    assert processor.call_count == 1
    assert "loss_scale" in out
    assert torch.any(torch.isclose(out["loss_scale"], torch.tensor(0.25)))
    assert torch.any(torch.isclose(out["loss_scale"], torch.tensor(1.0)))


def test_dpo_collator_builds_pairwise_batches() -> None:
    model_adapter = _build_smoke_adapter()
    processor = _FakeProcessor()
    collator = DPOCollator(
        model_adapter=model_adapter,
        template=build_template("smoke_vlm"),
        processor=processor,
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
    assert processor.call_count == 1
    assert out["model_specific_mask"].flatten().tolist() == [0, 1, 0, 1]


def test_ppo_collator_builds_left_padded_generation_queries() -> None:
    model_adapter = _build_smoke_adapter()
    processor = _FakeProcessor()
    collator = PPOCollator(
        model_adapter=model_adapter,
        template=build_template("smoke_vlm"),
        processor=processor,
        tokenizer=_FakeTokenizer(),
    )
    batch = [
        {
            "dataset_name": "a",
            "sample_id": "a1",
            "image_path": "/tmp/a.png",
            "messages": None,
            "system_prompt": "",
            "user_prompt": "Locate.",
            "extra": {},
        },
        {
            "dataset_name": "a",
            "sample_id": "a2",
            "image_path": "/tmp/a2.png",
            "messages": None,
            "system_prompt": "",
            "user_prompt": "Locate the requested object.",
            "extra": {},
        },
    ]
    out = collator(batch)

    assert collator.input_mode == "generation"
    assert collator.padding_side == "left"
    assert out["input_ids"].shape[0] == 2
    assert out["attention_mask"].shape == out["input_ids"].shape
    assert out["input_ids"].tolist() == [
        [0, 0, 0, 10],
        [10, 11, 12, 13],
    ]
    assert out["attention_mask"].tolist() == [
        [0, 0, 0, 1],
        [1, 1, 1, 1],
    ]
    assert all(
        chunk.get("type") != "image"
        for message in processor.last_messages
        for chunk in message.get("content", [])
    )
