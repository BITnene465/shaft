from __future__ import annotations

import logging

import pytest
import torch

from shaft.codec import ShaftCodecResult
from shaft.config.training import (
    EvalConfig,
    EvalDatasetPolicyConfig,
    EvalMetricConfig,
    EvalNormalizerConfig,
)
from shaft.template.base import ShaftChatTemplate
from shaft.training.online_eval import (
    ShaftOnlineEvalRunner,
    ShaftOnlineEvalSample,
    ShaftTargetResult,
)


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        mapping = {
            (101,): '{"ok": 1}',
            (102,): "oops",
        }
        key = tuple(int(x) for x in token_ids if not skip_special_tokens or int(x) not in {0, 2})
        return mapping.get(key, "")


class _FakeTemplateMeta:
    template_type = "fake"
    default_system = None
    auto_add_generation_prompt = True


class _FakePromptCollator:
    def __init__(self) -> None:
        self.template = ShaftChatTemplate(_FakeTemplateMeta())
        self.tokenizer = _FakeTokenizer()


class _FakeModel:
    def __init__(self) -> None:
        self.training = False
        self.grad_enabled_during_generate = None

    def eval(self):
        self.training = False
        return self

    def generate(self, **kwargs):
        _ = kwargs
        self.grad_enabled_during_generate = torch.is_grad_enabled()
        return torch.tensor([[11, 12, 101, 2], [21, 22, 102, 2]], dtype=torch.long)


class _FakeTrainer:
    def __init__(self, batches):
        self._batches = batches
        self.data_collator = None
        self.model = _FakeModel()

    def get_eval_dataloader(self, eval_dataset):
        _ = eval_dataset
        return list(self._batches)

    def _prepare_inputs(self, inputs):
        return inputs


def test_online_eval_runner_aggregates_metrics_and_logs(caplog) -> None:
    caplog.set_level(logging.INFO)
    eval_config = EvalConfig(
        enabled=True,
        online_metrics_enabled=True,
        datasets={
            "ds_a": EvalDatasetPolicyConfig(
                prediction_codec="json_object",
                target_adapter="target_text",
                target_adapter_params={"codec": "json_object"},
                metrics=[
                    EvalMetricConfig(name="parse_success"),
                    EvalMetricConfig(name="exact_match"),
                ],
                primary_metric="exact_match",
                normalizer=EvalNormalizerConfig(type="identity"),
                weight=0.25,
            ),
            "ds_b": EvalDatasetPolicyConfig(
                prediction_codec="json_object",
                target_adapter="target_text",
                target_adapter_params={"codec": "json_object"},
                metrics=[
                    EvalMetricConfig(name="parse_success"),
                    EvalMetricConfig(name="exact_match"),
                ],
                primary_metric="exact_match",
                normalizer=EvalNormalizerConfig(type="identity"),
                weight=0.75,
            ),
        },
    )
    runner = ShaftOnlineEvalRunner(
        eval_config=eval_config,
        prompt_collator=_FakePromptCollator(),
    )
    batch = {
        "input_ids": torch.tensor([[11, 12], [21, 22]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1], [1, 1]], dtype=torch.long),
        "pixel_values": torch.zeros((2, 3, 4, 4), dtype=torch.float32),
        "labels": torch.full((2, 2), -100, dtype=torch.long),
        "meta": {
            "dataset_name": ["ds_a", "ds_b"],
            "sample_id": ["a", "b"],
            "image_path": ["a.png", "b.png"],
            "target_text": ['{"ok": 1}', '{"ok": 2}'],
            "extra": [{}, {}],
        },
    }
    trainer = _FakeTrainer([batch])
    metrics = runner.evaluate(trainer, eval_dataset=object(), metric_key_prefix="eval")
    assert metrics["eval_ds_a_parse_success"] == 1.0
    assert metrics["eval_ds_a_exact_match"] == 1.0
    assert metrics["eval_ds_a_score"] == 1.0
    assert metrics["eval_ds_b_parse_success"] == 0.0
    assert metrics["eval_ds_b_exact_match"] == 0.0
    assert metrics["eval_ds_b_score"] == 0.0
    assert metrics["eval_final_score"] == 0.25
    assert trainer.model.grad_enabled_during_generate is False
    assert "dataset=ds_a" in caplog.text
    assert "dataset=ds_b" in caplog.text
    assert "final_score=0.25" in caplog.text


def test_online_eval_runner_normalizes_with_range() -> None:
    eval_config = EvalConfig(
        enabled=True,
        online_metrics_enabled=True,
        datasets={
            "ds": EvalDatasetPolicyConfig(
                prediction_codec="text",
                target_adapter="target_text",
                metrics=[EvalMetricConfig(name="exact_match")],
                primary_metric="exact_match",
                normalizer=EvalNormalizerConfig(type="range", min_value=0.0, max_value=2.0),
                weight=1.0,
            ),
        },
    )
    runner = ShaftOnlineEvalRunner(
        eval_config=eval_config,
        prompt_collator=_FakePromptCollator(),
    )
    entries = [
        ShaftOnlineEvalSample(
            dataset_name="ds",
            sample_id="x",
            prediction=ShaftCodecResult(raw_text="a", parsed="a", valid=True, partial=False, error_type=None, error=None),
            target=ShaftTargetResult(value="a", valid=True, error=None),
            meta={},
        ),
        ShaftOnlineEvalSample(
            dataset_name="ds",
            sample_id="y",
            prediction=ShaftCodecResult(raw_text="b", parsed="b", valid=True, partial=False, error_type=None, error=None),
            target=ShaftTargetResult(value="c", valid=True, error=None),
            meta={},
        ),
    ]
    metrics = runner.aggregate_samples(entries, metric_key_prefix="eval")
    assert metrics["eval_ds_exact_match"] == 0.5
    assert metrics["eval_ds_score"] == 0.25


def test_online_eval_runner_skips_dataset_without_samples(caplog) -> None:
    caplog.set_level(logging.WARNING)
    eval_config = EvalConfig(
        enabled=True,
        online_metrics_enabled=True,
        datasets={
            "ds_a": EvalDatasetPolicyConfig(
                prediction_codec="text",
                target_adapter="target_text",
                metrics=[EvalMetricConfig(name="exact_match")],
                primary_metric="exact_match",
                normalizer=EvalNormalizerConfig(type="identity"),
                weight=0.25,
            ),
            "ds_b": EvalDatasetPolicyConfig(
                prediction_codec="text",
                target_adapter="target_text",
                metrics=[EvalMetricConfig(name="exact_match")],
                primary_metric="exact_match",
                normalizer=EvalNormalizerConfig(type="identity"),
                weight=0.75,
            ),
        },
    )
    runner = ShaftOnlineEvalRunner(
        eval_config=eval_config,
        prompt_collator=_FakePromptCollator(),
    )
    entries = [
        ShaftOnlineEvalSample(
            dataset_name="ds_a",
            sample_id="x",
            prediction=ShaftCodecResult(raw_text="a", parsed="a", valid=True, partial=False, error_type=None, error=None),
            target=ShaftTargetResult(value="a", valid=True, error=None),
            meta={},
        ),
    ]
    metrics = runner.aggregate_samples(entries, metric_key_prefix="eval")
    assert metrics["eval_ds_a_score"] == pytest.approx(1.0)
    assert "eval_ds_b_score" not in metrics
    assert metrics["eval_final_score"] == pytest.approx(1.0)
    assert "dataset=ds_b has no samples" in caplog.text
