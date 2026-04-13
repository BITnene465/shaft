from __future__ import annotations

import unittest

import torch
from PIL import Image

from vlm_structgen.core.config import ExperimentRuntimeConfig
from vlm_structgen.core.data.collator import SFTCollator
from vlm_structgen.core.train.trainer import Trainer
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
        **_kwargs,
    ):
        del add_special_tokens
        del return_attention_mask
        if isinstance(text, str):
            return {"input_ids": [index + 1 for index, _ in enumerate(text)]}
        return {"input_ids": [[index + 1 for index, _ in enumerate(item)] for item in text]}


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


def _make_item(route: str, image: Image.Image, target_text: str = '{"ok":1}') -> dict:
    return {
        "route": route,
        "sample_id": f"{route}-sample",
        "image_path": "unused.png",
        "image": image,
        "image_width": 100,
        "image_height": 100,
        "system_prompt": "",
        "user_prompt": "predict",
        "target_text": target_text,
        "gt_struct": {},
    }


class _TinyTrainDataset:
    def __len__(self) -> int:
        return 1


class _ToyCausalLM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scalar = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, **kwargs):  # noqa: D401
        del kwargs
        loss = self.scalar * self.scalar
        return type("Output", (), {"loss": loss})()


class SFTLossBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        ensure_builtin_task_adapters_registered()
        self.tokenizer = DummyTokenizer()
        self.processor = DummyProcessor(self.tokenizer)
        self.image = Image.new("RGB", (4, 4), color="black")

    def test_collator_appends_eos_and_does_not_emit_loss_weights(self) -> None:
        collator = SFTCollator(
            processor=self.processor,
            tokenizer=self.tokenizer,
            add_eos_token=True,
            include_targets_in_inputs=True,
        )
        batch = collator([_make_item("grounding/arrow", self.image)])

        self.assertNotIn("loss_weights", batch)
        target_labels = batch["labels"][0][batch["labels"][0] != -100].tolist()
        self.assertTrue(target_labels)
        self.assertEqual(target_labels[-1], self.tokenizer.eos_token_id)

    def test_collator_eval_mode_keeps_all_labels_ignored(self) -> None:
        collator = SFTCollator(
            processor=self.processor,
            tokenizer=self.tokenizer,
            add_eos_token=True,
            include_targets_in_inputs=False,
            padding_side="left",
        )
        batch = collator([_make_item("keypoint_sequence/arrow", self.image)])
        self.assertNotIn("loss_weights", batch)
        self.assertTrue(torch.all(batch["labels"] == -100).item())

    def test_trainer_step_uses_model_loss_without_loss_weights(self) -> None:
        model = _ToyCausalLM()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)
        config = ExperimentRuntimeConfig()
        config.train.grad_accum_steps = 1
        config.train.max_grad_norm = 1.0
        config.train.bf16 = False

        trainer = Trainer(
            model=model,
            tokenizer=self.tokenizer,
            processor=self.processor,
            train_dataloader=_TinyTrainDataset(),
            val_dataloader=None,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            device=torch.device("cpu"),
            rank=0,
            world_size=1,
            evaluator=None,
            logger=None,
        )

        batch = {
            "input_ids": torch.ones((1, 4), dtype=torch.long),
            "attention_mask": torch.ones((1, 4), dtype=torch.long),
            "labels": torch.tensor([[-100, 1, 2, 3]], dtype=torch.long),
            "pixel_values": torch.zeros((1, 3, 2, 2), dtype=torch.float32),
            "meta": {"route": ["grounding/arrow"]},
        }
        metrics = trainer.train_one_step(batch)
        self.assertIn("train/loss", metrics)
        self.assertGreaterEqual(metrics["train/loss"], 0.0)


if __name__ == "__main__":
    unittest.main()
