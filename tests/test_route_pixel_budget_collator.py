from __future__ import annotations

import unittest

import torch
from PIL import Image

from vlm_structgen.core.data.collator import SFTCollator


class _DummyTokenizer:
    eos_token_id = 9000
    pad_token_id = 0

    def __call__(
        self,
        text,
        *,
        add_special_tokens: bool = False,
        return_attention_mask: bool = False,
        **_kwargs,
    ):
        del add_special_tokens
        del return_attention_mask
        if isinstance(text, str):
            return {"input_ids": [index + 1 for index, _ in enumerate(text)]}
        return {"input_ids": [[index + 1 for index, _ in enumerate(item)] for item in text]}


class _RecordingProcessor:
    def __init__(self) -> None:
        self.calls: list[dict[str, int | None]] = []

    def apply_chat_template(self, messages, *, tokenize: bool = False, add_generation_prompt: bool = True):
        del tokenize
        rendered = []
        for message in messages:
            for content in message.get("content", []):
                if content.get("type") == "text":
                    rendered.append(str(content.get("text", "")))
                elif content.get("type") == "image":
                    rendered.append("<image>")
        if add_generation_prompt:
            rendered.append("<gen>")
        return "".join(rendered)

    def __call__(
        self,
        *,
        text,
        images,
        padding: bool,
        return_tensors: str,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        **_kwargs,
    ):
        del images
        if not padding or return_tensors != "pt":
            raise ValueError("test processor expects padding=True and return_tensors='pt'")
        self.calls.append(
            {
                "batch_size": len(text),
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
            }
        )

        # Make token lengths depend on max_pixels to validate mixed-budget merge.
        target_len = 8 if max_pixels == 200 else 5
        input_ids = torch.arange(1, target_len + 1, dtype=torch.long).repeat(len(text), 1)
        attention_mask = torch.ones_like(input_ids)
        pixel_values = torch.ones((len(text) * 2, 4), dtype=torch.float32)
        image_grid_thw = torch.tensor([[1, 1, 2] for _ in text], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }


def _make_item(route: str, image: Image.Image) -> dict:
    return {
        "route": route,
        "sample_id": f"{route}-sample",
        "image_path": "unused.png",
        "image": image,
        "image_width": 100,
        "image_height": 100,
        "system_prompt": "",
        "user_prompt": "predict",
        "target_text": "{\"ok\":1}",
        "gt_struct": {},
    }


class RoutePixelBudgetCollatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = _DummyTokenizer()
        self.processor = _RecordingProcessor()
        self.image = Image.new("RGB", (8, 8), color="black")

    def test_uses_per_route_pixel_budget_when_batch_is_mixed(self) -> None:
        collator = SFTCollator(
            processor=self.processor,
            tokenizer=self.tokenizer,
            route_pixel_budgets={
                "grounding/arrow": {"min_pixels": 100, "max_pixels": 200},
                "keypoint_sequence/arrow": {"min_pixels": 300, "max_pixels": 400},
            },
            add_eos_token=True,
            include_targets_in_inputs=True,
        )
        batch = [
            _make_item("grounding/arrow", self.image),
            _make_item("keypoint_sequence/arrow", self.image),
            _make_item("grounding/arrow", self.image),
        ]
        output = collator(batch)

        self.assertEqual(len(self.processor.calls), 2)
        budgets = {(call["min_pixels"], call["max_pixels"]) for call in self.processor.calls}
        self.assertEqual(budgets, {(100, 200), (300, 400)})
        self.assertEqual(sorted(call["batch_size"] for call in self.processor.calls), [1, 2])
        self.assertEqual(output["input_ids"].shape[0], 3)
        self.assertEqual(output["pixel_values"].shape[0], 6)
        self.assertEqual(output["meta"]["route"], ["grounding/arrow", "keypoint_sequence/arrow", "grounding/arrow"])

    def test_keeps_fast_path_when_pixel_budget_is_homogeneous(self) -> None:
        collator = SFTCollator(
            processor=self.processor,
            tokenizer=self.tokenizer,
            route_pixel_budgets={
                "grounding/arrow": {"min_pixels": 100, "max_pixels": 200},
                "keypoint_sequence/arrow": {"min_pixels": 100, "max_pixels": 200},
            },
            add_eos_token=True,
            include_targets_in_inputs=True,
        )
        batch = [
            _make_item("grounding/arrow", self.image),
            _make_item("keypoint_sequence/arrow", self.image),
        ]
        _ = collator(batch)

        self.assertEqual(len(self.processor.calls), 1)
        self.assertEqual(self.processor.calls[0]["batch_size"], 2)
        self.assertEqual(self.processor.calls[0]["min_pixels"], 100)
        self.assertEqual(self.processor.calls[0]["max_pixels"], 200)


if __name__ == "__main__":
    unittest.main()
