from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
import torch

from shaft.codec import ShaftCodecResult
from shaft.config.training import (
    EvalConfig,
    EvalDatasetPolicyConfig,
    EvalMetricConfig,
    EvalNormalizerConfig,
)
from shaft.metrics import build_eval_metric
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
        self.generate_kwargs = None
        self.use_cache_during_generate = None
        self.config = SimpleNamespace(use_cache=False)
        self.generation_config = SimpleNamespace(use_cache=False)

    def eval(self):
        self.training = False
        return self

    def generate(self, **kwargs):
        self.generate_kwargs = dict(kwargs)
        self.grad_enabled_during_generate = torch.is_grad_enabled()
        self.use_cache_during_generate = (
            bool(self.config.use_cache),
            bool(self.generation_config.use_cache),
        )
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


class _OnlinePrepareHookTrainer(_FakeTrainer):
    def __init__(self, batches):
        super().__init__(batches)
        self.online_prepare_called = False

    def _prepare_inputs(self, inputs):
        _ = inputs
        raise AssertionError("online eval should not call the trainer rollout _prepare_inputs")

    def prepare_online_eval_inputs(self, inputs):
        self.online_prepare_called = True
        return inputs


class _MappedFakeModel(_FakeModel):
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


class _LeftPaddedGenerateModel(_FakeModel):
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


class _LeftPaddedTrainer(_FakeTrainer):
    def __init__(self, batches):
        super().__init__(batches)
        self.model = _LeftPaddedGenerateModel()


class _MappedTrainer(_FakeTrainer):
    def __init__(self, batches_by_dataset):
        self._batches_by_dataset = batches_by_dataset
        self.data_collator = None
        self.model = _MappedFakeModel()
        self.eval_dataset = batches_by_dataset

    def get_eval_dataloader(self, eval_dataset):
        return list(self._batches_by_dataset[eval_dataset])


class _PersistentWorkerCacheTrainer(_FakeTrainer):
    def __init__(self, batches_by_dataset):
        self._batches_by_dataset = batches_by_dataset
        self.data_collator = None
        self.model = _MappedFakeModel()
        self.eval_dataset = batches_by_dataset
        self._cached = None

    def get_eval_dataloader(self, eval_dataset):
        if isinstance(eval_dataset, str):
            return list(self._batches_by_dataset[eval_dataset])
        if self._cached is None:
            first_key = next(iter(self._batches_by_dataset))
            self._cached = list(self._batches_by_dataset[first_key])
        return list(self._cached)


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
    assert trainer.model.generate_kwargs["do_sample"] is False
    assert trainer.model.generate_kwargs["temperature"] == 1.0
    assert trainer.model.generate_kwargs["top_p"] == 1.0
    assert trainer.model.generate_kwargs["top_k"] == 50
    assert trainer.model.use_cache_during_generate == (True, True)
    assert trainer.model.config.use_cache is False
    assert trainer.model.generation_config.use_cache is False
    assert "dataset=ds_a" in caplog.text
    assert "dataset=ds_b" in caplog.text
    assert "final_score=0.25" in caplog.text


def test_online_eval_runner_uses_online_prepare_hook() -> None:
    eval_config = EvalConfig(
        enabled=True,
        online_metrics_enabled=True,
        datasets={
            "ds": EvalDatasetPolicyConfig(
                prediction_codec="json_object",
                target_adapter="target_text",
                target_adapter_params={"codec": "json_object"},
                metrics=[EvalMetricConfig(name="parse_success")],
                primary_metric="parse_success",
                normalizer=EvalNormalizerConfig(type="identity"),
                weight=1.0,
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
        "labels": torch.full((2, 2), -100, dtype=torch.long),
        "meta": {
            "dataset_name": ["ds", "ds"],
            "sample_id": ["a", "b"],
            "image_path": ["a.png", "b.png"],
            "target_text": ['{"ok": 1}', '{"ok": 2}'],
            "extra": [{}, {}],
        },
    }
    trainer = _OnlinePrepareHookTrainer([batch])

    metrics = runner.evaluate(trainer, eval_dataset=object(), metric_key_prefix="eval")

    assert trainer.online_prepare_called is True
    assert metrics["eval_ds_parse_success"] == pytest.approx(0.5)


def test_online_eval_runner_supports_named_eval_datasets() -> None:
    eval_config = EvalConfig(
        enabled=True,
        online_metrics_enabled=True,
        datasets={
            "ds_a": EvalDatasetPolicyConfig(
                prediction_codec="json_object",
                target_adapter="target_text",
                target_adapter_params={"codec": "json_object"},
                metrics=[EvalMetricConfig(name="exact_match")],
                primary_metric="exact_match",
                normalizer=EvalNormalizerConfig(type="identity"),
                weight=0.25,
            ),
            "ds_b": EvalDatasetPolicyConfig(
                prediction_codec="json_object",
                target_adapter="target_text",
                target_adapter_params={"codec": "json_object"},
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
    batch_a = {
        "input_ids": torch.tensor([[11, 12]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
        "pixel_values": torch.zeros((1, 3, 4, 4), dtype=torch.float32),
        "labels": torch.full((1, 2), -100, dtype=torch.long),
        "meta": {
            "dataset_name": ["ds_a"],
            "sample_id": ["a"],
            "image_path": ["a.png"],
            "target_text": ['{"ok": 1}'],
            "extra": [{}],
        },
    }
    batch_b = {
        "input_ids": torch.tensor([[21, 22]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
        "pixel_values": torch.zeros((1, 3, 4, 4), dtype=torch.float32),
        "labels": torch.full((1, 2), -100, dtype=torch.long),
        "meta": {
            "dataset_name": ["ds_b"],
            "sample_id": ["b"],
            "image_path": ["b.png"],
            "target_text": ['{"ok": 2}'],
            "extra": [{}],
        },
    }
    trainer = _MappedTrainer({"ds_a": [batch_a], "ds_b": [batch_b]})
    metrics = runner.evaluate(
        trainer,
        eval_dataset={"ds_a": "ds_a", "ds_b": "ds_b"},
        metric_key_prefix="eval",
    )
    assert metrics["eval_ds_a_exact_match"] == pytest.approx(1.0)
    assert metrics["eval_ds_b_exact_match"] == pytest.approx(0.0)
    assert metrics["eval_final_score"] == pytest.approx(0.25)


def test_online_eval_runner_slices_left_padded_decoder_prompts_at_input_width() -> None:
    eval_config = EvalConfig(
        enabled=True,
        online_metrics_enabled=True,
        datasets={
            "ds": EvalDatasetPolicyConfig(
                prediction_codec="json_object",
                target_adapter="target_text",
                target_adapter_params={"codec": "json_object"},
                metrics=[
                    EvalMetricConfig(name="parse_success"),
                    EvalMetricConfig(name="exact_match"),
                ],
                primary_metric="exact_match",
                normalizer=EvalNormalizerConfig(type="identity"),
                weight=1.0,
            ),
        },
    )
    runner = ShaftOnlineEvalRunner(
        eval_config=eval_config,
        prompt_collator=_FakePromptCollator(),
    )
    batch = {
        "input_ids": torch.tensor(
            [
                [0, 0, 91, 92],
                [81, 82, 83, 84],
            ],
            dtype=torch.long,
        ),
        "attention_mask": torch.tensor(
            [
                [0, 0, 1, 1],
                [1, 1, 1, 1],
            ],
            dtype=torch.long,
        ),
        "pixel_values": torch.zeros((2, 3, 4, 4), dtype=torch.float32),
        "labels": torch.full((2, 2), -100, dtype=torch.long),
        "meta": {
            "dataset_name": ["ds", "ds"],
            "sample_id": ["left-padded", "full-width"],
            "image_path": ["left.png", "right.png"],
            "target_text": ['{"ok": 1}', '{"ok": 2}'],
            "extra": [{}, {}],
        },
    }
    trainer = _LeftPaddedTrainer([batch])
    metrics = runner.evaluate(trainer, eval_dataset=object(), metric_key_prefix="eval")

    assert metrics["eval_ds_parse_success"] == pytest.approx(1.0)
    assert metrics["eval_ds_exact_match"] == pytest.approx(1.0)


def test_online_eval_runner_uses_named_eval_keys_to_avoid_cached_eval_dataloader_collision() -> None:
    eval_config = EvalConfig(
        enabled=True,
        online_metrics_enabled=True,
        datasets={
            "ds_a": EvalDatasetPolicyConfig(
                prediction_codec="json_object",
                target_adapter="target_text",
                target_adapter_params={"codec": "json_object"},
                metrics=[EvalMetricConfig(name="exact_match")],
                primary_metric="exact_match",
                normalizer=EvalNormalizerConfig(type="identity"),
                weight=0.25,
            ),
            "ds_b": EvalDatasetPolicyConfig(
                prediction_codec="json_object",
                target_adapter="target_text",
                target_adapter_params={"codec": "json_object"},
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
    batch_a = {
        "input_ids": torch.tensor([[11, 12]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
        "pixel_values": torch.zeros((1, 3, 4, 4), dtype=torch.float32),
        "labels": torch.full((1, 2), -100, dtype=torch.long),
        "meta": {
            "dataset_name": ["ds_a"],
            "sample_id": ["a"],
            "image_path": ["a.png"],
            "target_text": ['{"ok": 1}'],
            "extra": [{}],
        },
    }
    batch_b = {
        "input_ids": torch.tensor([[21, 22]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
        "pixel_values": torch.zeros((1, 3, 4, 4), dtype=torch.float32),
        "labels": torch.full((1, 2), -100, dtype=torch.long),
        "meta": {
            "dataset_name": ["ds_b"],
            "sample_id": ["b"],
            "image_path": ["b.png"],
            "target_text": ['{"ok": 2}'],
            "extra": [{}],
        },
    }
    trainer = _PersistentWorkerCacheTrainer({"ds_a": [batch_a], "ds_b": [batch_b]})
    metrics = runner.evaluate(
        trainer,
        eval_dataset={"ds_a": object(), "ds_b": object()},
        metric_key_prefix="eval",
    )
    assert metrics["eval_ds_a_exact_match"] == pytest.approx(1.0)
    assert metrics["eval_ds_b_exact_match"] == pytest.approx(0.0)
    assert metrics["eval_final_score"] == pytest.approx(0.25)


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


def test_online_eval_runner_final_score_is_dataset_weighted_not_sample_weighted() -> None:
    eval_config = EvalConfig(
        enabled=True,
        online_metrics_enabled=True,
        datasets={
            "layout": EvalDatasetPolicyConfig(
                prediction_codec="text",
                target_adapter="target_text",
                metrics=[EvalMetricConfig(name="exact_match")],
                primary_metric="exact_match",
                normalizer=EvalNormalizerConfig(type="identity"),
                weight=0.8,
            ),
            "keypoint": EvalDatasetPolicyConfig(
                prediction_codec="text",
                target_adapter="target_text",
                metrics=[EvalMetricConfig(name="exact_match")],
                primary_metric="exact_match",
                normalizer=EvalNormalizerConfig(type="identity"),
                weight=0.2,
            ),
        },
    )
    runner = ShaftOnlineEvalRunner(
        eval_config=eval_config,
        prompt_collator=_FakePromptCollator(),
    )
    entries = [
        ShaftOnlineEvalSample(
            dataset_name="layout",
            sample_id="layout-0",
            prediction=ShaftCodecResult(raw_text="ok", parsed="ok", valid=True, partial=False, error_type=None, error=None),
            target=ShaftTargetResult(value="ok", valid=True, error=None),
            meta={},
        ),
    ]
    for index in range(5):
        entries.append(
            ShaftOnlineEvalSample(
                dataset_name="keypoint",
                sample_id=f"keypoint-{index}",
                prediction=ShaftCodecResult(raw_text="miss", parsed="miss", valid=True, partial=False, error_type=None, error=None),
                target=ShaftTargetResult(value="ok", valid=True, error=None),
                meta={},
            )
        )
    metrics = runner.aggregate_samples(entries, metric_key_prefix="eval")
    assert metrics["eval_layout_exact_match"] == pytest.approx(1.0)
    assert metrics["eval_keypoint_exact_match"] == pytest.approx(0.0)
    assert metrics["eval_final_score"] == pytest.approx(0.8)


def test_online_eval_runner_deduplicates_gathered_samples_before_metrics() -> None:
    eval_config = EvalConfig(
        enabled=True,
        online_metrics_enabled=True,
        datasets={
            "ds": EvalDatasetPolicyConfig(
                prediction_codec="text",
                target_adapter="target_text",
                metrics=[EvalMetricConfig(name="parse_success")],
                primary_metric="parse_success",
                normalizer=EvalNormalizerConfig(type="identity"),
                weight=1.0,
            ),
        },
    )
    runner = ShaftOnlineEvalRunner(
        eval_config=eval_config,
        prompt_collator=_FakePromptCollator(),
    )
    valid = ShaftCodecResult(
        raw_text='{"ok": 1}',
        parsed={"ok": 1},
        valid=True,
        partial=False,
        error_type=None,
        error=None,
    )
    invalid_duplicate = ShaftCodecResult(
        raw_text="oops",
        parsed=None,
        valid=False,
        partial=False,
        error_type="json_decode_error",
        error="bad json",
    )
    entries = [
        ShaftOnlineEvalSample(
            dataset_name="ds",
            sample_id="sample-0",
            prediction=valid,
            target=ShaftTargetResult(value='{"ok": 1}', valid=True, error=None),
            meta={"image_path": "same.png"},
        ),
        ShaftOnlineEvalSample(
            dataset_name="ds",
            sample_id="sample-0",
            prediction=invalid_duplicate,
            target=ShaftTargetResult(value='{"ok": 1}', valid=True, error=None),
            meta={"image_path": "same.png"},
        ),
    ]

    metrics = runner.aggregate_samples(entries, metric_key_prefix="eval")

    assert metrics["eval_ds_parse_success"] == pytest.approx(1.0)
    assert metrics["eval_final_score"] == pytest.approx(1.0)


def test_parse_metrics_distinguish_complete_and_partial_json() -> None:
    parse_success = build_eval_metric("parse_success")
    parse_partial_rate = build_eval_metric("parse_partial_rate")

    complete = ShaftCodecResult(
        raw_text='{"ok": 1}',
        parsed={"ok": 1},
        valid=True,
        partial=False,
        error_type=None,
        error=None,
    )
    partial = ShaftCodecResult(
        raw_text='{"ok": 1',
        parsed={"ok": 1},
        valid=True,
        partial=True,
        error_type=None,
        error=None,
    )
    invalid = ShaftCodecResult(
        raw_text="oops",
        parsed=None,
        valid=False,
        partial=False,
        error_type="json_decode_error",
        error="bad json",
    )

    for prediction in (complete, partial, invalid):
        parse_success.update(prediction=prediction, target=None, sample_meta={})
        parse_partial_rate.update(prediction=prediction, target=None, sample_meta={})

    assert parse_success.compute() == pytest.approx(1.0 / 3.0)
    assert parse_partial_rate.compute() == pytest.approx(1.0 / 3.0)


def test_keypoint_pck_uses_normalized_coordinate_scale_by_default() -> None:
    metric = build_eval_metric("keypoint_pck")
    prediction = ShaftCodecResult(
        raw_text="",
        parsed={
            "keypoints_2d": [[530, 500], [500, 470]],
            "stroke_pattern": "solid",
            "geometry_style": "straight",
        },
        valid=True,
        partial=False,
        error_type=None,
        error=None,
    )
    target = {
        "keypoints_2d": [[500, 500], [500, 500]],
        "stroke_pattern": "solid",
        "geometry_style": "straight",
    }

    metric.update(
        prediction=prediction,
        target=target,
        sample_meta={"extra": {"image_width": 100, "image_height": 100}},
    )

    assert metric.compute() == pytest.approx(1.0)
