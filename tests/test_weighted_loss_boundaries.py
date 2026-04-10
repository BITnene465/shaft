from __future__ import annotations

import unittest
from unittest.mock import patch

import torch
from PIL import Image

from vlm_structgen.core.data.collator import SFTCollator
from vlm_structgen.core.train.weighted_loss import compute_weighted_token_ce_loss
from vlm_structgen.domains.arrow.codecs.grounding import GroundingCodec
from vlm_structgen.domains.arrow.codecs.keypoint_sequence import KeypointSequenceCodec
from vlm_structgen.tasks.bootstrap import ensure_builtin_task_adapters_registered


class DummyTokenizer:
    eos_token_id = 9000
    pad_token_id = 0

    def __call__(
        self,
        text,
        *,
        add_special_tokens: bool = False,
        return_attention_mask: bool = False,
        return_offsets_mapping: bool = False,
        **_kwargs,
    ):
        del add_special_tokens
        del return_attention_mask
        if isinstance(text, str):
            input_ids = self._encode_text(text)
            payload = {"input_ids": input_ids}
            if return_offsets_mapping:
                payload["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
            return payload
        input_ids = [self._encode_text(item) for item in text]
        payload = {"input_ids": input_ids}
        if return_offsets_mapping:
            payload["offset_mapping"] = [
                [(index, index + 1) for index in range(len(item))]
                for item in text
            ]
        return payload

    @staticmethod
    def _encode_text(text: str) -> list[int]:
        return [index + 1 for index, _char in enumerate(text)]


class NoOffsetTokenizer(DummyTokenizer):
    def __call__(
        self,
        text,
        *,
        add_special_tokens: bool = False,
        return_attention_mask: bool = False,
        return_offsets_mapping: bool = False,
        **kwargs,
    ):
        del return_offsets_mapping
        return super().__call__(
            text,
            add_special_tokens=add_special_tokens,
            return_attention_mask=return_attention_mask,
            return_offsets_mapping=False,
            **kwargs,
        )


class DummyProcessor:
    def __init__(self, tokenizer: DummyTokenizer) -> None:
        self.tokenizer = tokenizer

    def apply_chat_template(self, messages, *, tokenize: bool = False, add_generation_prompt: bool = True):
        del tokenize
        rendered_parts: list[str] = []
        for message in messages:
            for content in message.get("content", []):
                content_type = content.get("type")
                if content_type == "text":
                    rendered_parts.append(str(content.get("text", "")))
                elif content_type == "image":
                    rendered_parts.append("<image>")
        if add_generation_prompt:
            rendered_parts.append("<gen>")
        return "".join(rendered_parts)

    def __call__(
        self,
        *,
        text,
        images,
        padding: bool = True,
        return_tensors: str = "pt",
        **_kwargs,
    ):
        del images
        del padding
        if return_tensors != "pt":
            raise ValueError("DummyProcessor only supports return_tensors='pt'.")
        encoded = self.tokenizer(text, add_special_tokens=False)
        input_id_rows = [torch.tensor(row, dtype=torch.long) for row in encoded["input_ids"]]
        max_length = max(row.shape[0] for row in input_id_rows)
        padded_ids = []
        padded_masks = []
        for row in input_id_rows:
            pad_length = max_length - row.shape[0]
            if pad_length > 0:
                padded_ids.append(
                    torch.cat([row, torch.full((pad_length,), self.tokenizer.pad_token_id, dtype=torch.long)])
                )
                padded_masks.append(
                    torch.cat([torch.ones_like(row), torch.zeros((pad_length,), dtype=torch.long)])
                )
            else:
                padded_ids.append(row)
                padded_masks.append(torch.ones_like(row))
        return {
            "input_ids": torch.stack(padded_ids, dim=0),
            "attention_mask": torch.stack(padded_masks, dim=0),
            "pixel_values": torch.zeros((len(text), 3, 2, 2), dtype=torch.float32),
        }


class DummyModelOutputs:
    def __init__(self, logits: torch.Tensor, loss: torch.Tensor | None = None) -> None:
        self.logits = logits
        self.loss = torch.tensor(123.0) if loss is None else loss


class WeightedLossBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        ensure_builtin_task_adapters_registered()
        self.tokenizer = DummyTokenizer()
        self.processor = DummyProcessor(self.tokenizer)
        self.image = Image.new("RGB", (4, 4), color="black")

    def _build_training_item(
        self,
        *,
        route: str,
        target_text: str,
        loss_meta: dict,
        loss_weight_enabled: bool,
    ) -> dict:
        route_options = {
            "grounding/arrow": {
                "bbox_token_loss_weight": 2.0 if loss_weight_enabled else 1.0,
                "label_token_loss_weight": 1.5 if loss_weight_enabled else 1.0,
            },
            "keypoint_sequence/arrow": {
                "coordinate_token_loss_weight": 1.5 if loss_weight_enabled else 1.0,
            },
        }
        return {
            "route": route,
            "sample_id": f"{route}-sample",
            "image_path": "unused.png",
            "image": self.image,
            "image_width": 100,
            "image_height": 100,
            "system_prompt": "",
            "user_prompt": "predict",
            "target_text": target_text,
            "loss_meta": loss_meta,
            "gt_struct": {},
            "_route_options": route_options,
        }

    def _build_collator(self, item: dict, *, include_targets_in_inputs: bool = True) -> SFTCollator:
        return SFTCollator(
            processor=self.processor,
            tokenizer=self.tokenizer,
            num_bins=1000,
            task_route_options=item["_route_options"],
            add_eos_token=True,
            include_targets_in_inputs=include_targets_in_inputs,
        )

    def test_collator_appends_eos_and_aligned_loss_weights_for_grounding(self) -> None:
        codec = GroundingCodec(num_bins=1000)
        target_text, loss_meta = codec.encode_with_loss_meta(
            {"instances": [{"label": "single_arrow", "bbox": [10, 20, 30, 40], "keypoints": []}]},
            image_width=100,
            image_height=100,
        )
        item = self._build_training_item(
            route="grounding/arrow",
            target_text=target_text,
            loss_meta=loss_meta,
            loss_weight_enabled=True,
        )
        collator = self._build_collator(item)

        batch = collator([item])
        target_ids = self.tokenizer(target_text, add_special_tokens=False)["input_ids"]

        self.assertEqual(batch["labels"].shape, batch["loss_weights"].shape)
        target_labels = batch["labels"][0][batch["labels"][0] != -100].tolist()
        target_loss_weights = batch["loss_weights"][0][batch["labels"][0] != -100].tolist()
        self.assertEqual(target_labels[-1], self.tokenizer.eos_token_id)
        self.assertEqual(len(target_labels), len(target_ids) + 1)
        self.assertEqual(len(target_loss_weights), len(target_labels))
        self.assertEqual(target_loss_weights[-1], 1.0)
        self.assertTrue(any(weight > 1.0 for weight in target_loss_weights[:-1]))

    def test_collator_appends_eos_and_aligned_loss_weights_for_keypoint_sequence(self) -> None:
        codec = KeypointSequenceCodec(num_bins=1000)
        target_text, loss_meta = codec.encode_with_loss_meta(
            [[10, 20], [30, 40], [50, 60]],
            image_width=100,
            image_height=100,
        )
        item = self._build_training_item(
            route="keypoint_sequence/arrow",
            target_text=target_text,
            loss_meta=loss_meta,
            loss_weight_enabled=True,
        )
        collator = self._build_collator(item)

        batch = collator([item])
        target_labels = batch["labels"][0][batch["labels"][0] != -100].tolist()
        target_loss_weights = batch["loss_weights"][0][batch["labels"][0] != -100].tolist()
        self.assertEqual(target_labels[-1], self.tokenizer.eos_token_id)
        self.assertEqual(target_loss_weights[-1], 1.0)
        self.assertTrue(any(weight > 1.0 for weight in target_loss_weights[:-1]))

    def test_collator_raises_when_adapter_weights_do_not_match_target_tokenization(self) -> None:
        item = self._build_training_item(
            route="grounding/arrow",
            target_text='[{"label":"single_arrow","bbox_2d":[1,2,3,4]}]',
            loss_meta={"field_char_spans": {"label": [[11, 23]], "bbox_2d": [[34, 35]]}},
            loss_weight_enabled=True,
        )
        collator = self._build_collator(item)

        class BadAdapter:
            def build_target_token_weights(self, target_text, *, loss_meta, tokenizer):
                del target_text
                del loss_meta
                del tokenizer
                return [1.0]

        with patch("vlm_structgen.core.data.collator.get_adapter_for_route", return_value=BadAdapter()):
            with self.assertRaisesRegex(ValueError, "target token weights"):
                collator([item])

    def test_weighted_loss_consumes_only_precomputed_loss_weights(self) -> None:
        logits = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 3.0, 0.0],
                    [0.0, 0.0, 3.0],
                ]
            ],
            dtype=torch.float32,
        )
        labels = torch.tensor([[-100, 1, 2]], dtype=torch.long)
        loss_weights = torch.tensor([[1.0, 2.0, 5.0]], dtype=torch.float32)
        batch = {
            "labels": labels,
            "loss_weights": loss_weights,
        }
        outputs = DummyModelOutputs(logits=logits, loss=torch.tensor(999.0))

        actual = compute_weighted_token_ce_loss(outputs, batch)

        shift_logits = logits[:, :-1].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        shift_weights = loss_weights[:, 1:].contiguous()
        token_loss = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
        ).view_as(shift_labels)
        valid_mask = (shift_labels != -100).to(token_loss.dtype)
        expected = (token_loss * shift_weights * valid_mask).sum() / (shift_weights * valid_mask).sum()
        self.assertTrue(torch.isclose(actual, expected))

    def test_collator_raises_when_weighted_route_is_missing_loss_meta(self) -> None:
        item = self._build_training_item(
            route="grounding/arrow",
            target_text='[{"label":"single_arrow","bbox_2d":[1,2,3,4]}]',
            loss_meta=None,
            loss_weight_enabled=True,
        )
        collator = self._build_collator(item)

        with self.assertRaisesRegex(ValueError, "requires loss_meta"):
            collator([item])

    def test_collator_raises_when_weighted_route_lacks_offset_mapping(self) -> None:
        codec = KeypointSequenceCodec(num_bins=1000)
        target_text, loss_meta = codec.encode_with_loss_meta(
            [[10, 20], [30, 40]],
            image_width=100,
            image_height=100,
        )
        item = self._build_training_item(
            route="keypoint_sequence/arrow",
            target_text=target_text,
            loss_meta=loss_meta,
            loss_weight_enabled=True,
        )
        collator = SFTCollator(
            processor=DummyProcessor(NoOffsetTokenizer()),
            tokenizer=NoOffsetTokenizer(),
            num_bins=1000,
            task_route_options=item["_route_options"],
            add_eos_token=True,
            include_targets_in_inputs=True,
        )

        with self.assertRaisesRegex(ValueError, "requires tokenizer offset_mapping"):
            collator([item])

    def test_weighted_loss_raises_when_loss_weights_are_missing(self) -> None:
        logits = torch.zeros((1, 3, 4), dtype=torch.float32)
        labels = torch.tensor([[-100, 1, 2]], dtype=torch.long)
        outputs = DummyModelOutputs(logits=logits, loss=torch.tensor(999.0))

        with self.assertRaisesRegex(ValueError, "loss_weights"):
            compute_weighted_token_ce_loss(outputs, {"labels": labels})

    def test_weight_configs_must_be_greater_or_equal_to_one(self) -> None:
        with self.assertRaisesRegex(ValueError, "bbox_token_loss_weight must be >= 1.0"):
            SFTCollator(
                processor=self.processor,
                tokenizer=self.tokenizer,
                num_bins=1000,
                task_route_options={"grounding/arrow": {"bbox_token_loss_weight": 0.5}},
                add_eos_token=True,
                include_targets_in_inputs=True,
            )._get_adapter_for_item({"route": "grounding/arrow"})

        with self.assertRaisesRegex(ValueError, "coordinate_token_loss_weight must be >= 1.0"):
            SFTCollator(
                processor=self.processor,
                tokenizer=self.tokenizer,
                num_bins=1000,
                task_route_options={"keypoint_sequence/arrow": {"coordinate_token_loss_weight": 0.5}},
                add_eos_token=True,
                include_targets_in_inputs=True,
            )._get_adapter_for_item({"route": "keypoint_sequence/arrow"})


if __name__ == "__main__":
    unittest.main()
