from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

from shaft.config.training import (
    EvalConfig,
    EvalDatasetPolicyConfig,
    EvalMetricConfig,
    EvalNormalizerConfig,
)
from shaft.template.base import ShaftChatTemplate


class FakeOnlineEvalTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        mapping = {
            (101,): '{"ok": 1}',
            (102,): "oops",
        }
        token_text = {
            91: '["solid", "dashed"]',
            92: " prompt tail ",
            201: '{"ok": 1}',
            202: '{"ok": 2}',
        }
        key = tuple(int(x) for x in token_ids if not skip_special_tokens or int(x) not in {0, 2})
        if key in mapping:
            return mapping[key]
        return "".join(token_text.get(token_id, "") for token_id in key)


class FakeOnlineEvalTemplateMeta:
    template_type = "fake"
    default_system = None
    auto_add_generation_prompt = True


class FakeOnlineEvalPromptCollator:
    def __init__(self) -> None:
        self.template = ShaftChatTemplate(FakeOnlineEvalTemplateMeta())
        self.tokenizer = FakeOnlineEvalTokenizer()


class FakeOnlineEvalModel:
    def __init__(self) -> None:
        self.training = False
        self.grad_enabled_during_generate = None
        self.generate_kwargs = None
        self.use_cache_during_generate = None
        self.config = SimpleNamespace(use_cache=False)
        self.generation_config = SimpleNamespace(use_cache=False)

    def eval(self):
        self.training = False
        return self

    def train(self):
        self.training = True
        return self

    def generate(self, **kwargs):
        self.generate_kwargs = dict(kwargs)
        self.grad_enabled_during_generate = torch.is_grad_enabled()
        self.use_cache_during_generate = (
            bool(self.config.use_cache),
            bool(self.generation_config.use_cache),
        )
        return torch.tensor([[11, 12, 101, 2], [21, 22, 102, 2]], dtype=torch.long)


class FakeOnlineEvalTrainer:
    def __init__(self, batches):
        self._batches = batches
        self.data_collator = None
        self.model = FakeOnlineEvalModel()

    def get_eval_dataloader(self, eval_dataset):
        _ = eval_dataset
        return list(self._batches)

    def _prepare_inputs(self, inputs):
        return inputs


class OnlineEvalPrepareHookTrainer(FakeOnlineEvalTrainer):
    def __init__(self, batches):
        super().__init__(batches)
        self.online_prepare_called = False

    def _prepare_inputs(self, inputs):
        _ = inputs
        raise AssertionError("online eval should not call the trainer rollout _prepare_inputs")

    def prepare_online_eval_inputs(self, inputs):
        self.online_prepare_called = True
        return inputs


class MappedOnlineEvalModel(FakeOnlineEvalModel):
    def generate(self, **kwargs):
        self.generate_kwargs = dict(kwargs)
        self.grad_enabled_during_generate = torch.is_grad_enabled()
        self.use_cache_during_generate = (
            bool(self.config.use_cache),
            bool(self.generation_config.use_cache),
        )
        first_token = int(kwargs["input_ids"][0, 0])
        completion_token = 101 if first_token == 11 else 102
        prompt_ids = kwargs["input_ids"][0].tolist()
        return torch.tensor([prompt_ids + [completion_token, 2]], dtype=torch.long)


class LeftPaddedOnlineEvalModel(FakeOnlineEvalModel):
    def generate(self, **kwargs):
        self.generate_kwargs = dict(kwargs)
        self.grad_enabled_during_generate = torch.is_grad_enabled()
        self.use_cache_during_generate = (
            bool(self.config.use_cache),
            bool(self.generation_config.use_cache),
        )
        rows = []
        for index, prompt_ids in enumerate(kwargs["input_ids"].tolist()):
            completion_token = 201 if index == 0 else 202
            rows.append(prompt_ids + [completion_token, 2])
        return torch.tensor(rows, dtype=torch.long)


class LeftPaddedOnlineEvalTrainer(FakeOnlineEvalTrainer):
    def __init__(self, batches):
        super().__init__(batches)
        self.model = LeftPaddedOnlineEvalModel()


class MappedOnlineEvalTrainer(FakeOnlineEvalTrainer):
    def __init__(self, batches_by_dataset):
        self._batches_by_dataset = batches_by_dataset
        self.data_collator = None
        self.model = MappedOnlineEvalModel()
        self.eval_dataset = batches_by_dataset

    def get_eval_dataloader(self, eval_dataset):
        return list(self._batches_by_dataset[eval_dataset])


class PersistentWorkerCacheOnlineEvalTrainer(FakeOnlineEvalTrainer):
    def __init__(self, batches_by_dataset):
        self._batches_by_dataset = batches_by_dataset
        self.data_collator = None
        self.model = MappedOnlineEvalModel()
        self.eval_dataset = batches_by_dataset
        self._cached = None

    def get_eval_dataloader(self, eval_dataset):
        if isinstance(eval_dataset, str):
            return list(self._batches_by_dataset[eval_dataset])
        if self._cached is None:
            first_key = next(iter(self._batches_by_dataset))
            self._cached = list(self._batches_by_dataset[first_key])
        return list(self._cached)


def online_eval_config(datasets: dict[str, EvalDatasetPolicyConfig]) -> EvalConfig:
    return EvalConfig(enabled=True, online_metrics_enabled=True, datasets=datasets)


def json_target_policy(
    *,
    metrics: list[str],
    primary_metric: str | None = None,
    weight: float = 1.0,
) -> EvalDatasetPolicyConfig:
    return EvalDatasetPolicyConfig(
        prediction_codec="json_object",
        target_adapter="target_text",
        target_adapter_params={"codec": "json_object"},
        metrics=[EvalMetricConfig(name=name) for name in metrics],
        primary_metric=primary_metric or metrics[0],
        normalizer=EvalNormalizerConfig(type="identity"),
        weight=weight,
    )


def text_target_policy(
    *,
    metrics: list[str],
    primary_metric: str | None = None,
    normalizer: EvalNormalizerConfig | None = None,
    weight: float = 1.0,
) -> EvalDatasetPolicyConfig:
    return EvalDatasetPolicyConfig(
        prediction_codec="text",
        target_adapter="target_text",
        metrics=[EvalMetricConfig(name=name) for name in metrics],
        primary_metric=primary_metric or metrics[0],
        normalizer=normalizer or EvalNormalizerConfig(type="identity"),
        weight=weight,
    )


def online_eval_batch(
    *,
    input_ids: list[list[int]],
    dataset_names: list[str],
    sample_ids: list[str],
    target_texts: list[str],
    attention_mask: list[list[int]] | None = None,
    image_paths: list[str] | None = None,
    include_pixels: bool = True,
) -> dict[str, Any]:
    width = len(input_ids[0])
    if attention_mask is None:
        attention_mask = [[1] * width for _ in input_ids]
    if image_paths is None:
        image_paths = [f"{sample_id}.png" for sample_id in sample_ids]
    batch_size = len(input_ids)
    batch: dict[str, Any] = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.full((batch_size, 2), -100, dtype=torch.long),
        "meta": {
            "dataset_name": dataset_names,
            "sample_id": sample_ids,
            "image_path": image_paths,
            "target_text": target_texts,
            "extra": [{} for _ in range(batch_size)],
        },
    }
    if include_pixels:
        batch["pixel_values"] = torch.zeros((batch_size, 3, 4, 4), dtype=torch.float32)
    return batch
