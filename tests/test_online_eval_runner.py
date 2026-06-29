from __future__ import annotations

import logging

import pytest

from shaft.training.online_eval import ShaftOnlineEvalRunner
from tests.support.online_eval import (
    FakeOnlineEvalPromptCollator,
    FakeOnlineEvalTrainer,
    LeftPaddedOnlineEvalTrainer,
    MappedOnlineEvalTrainer,
    OnlineEvalPrepareHookTrainer,
    PersistentWorkerCacheOnlineEvalTrainer,
    json_target_policy,
    online_eval_batch,
    online_eval_config,
)


def test_online_eval_runner_aggregates_metrics_and_logs(caplog) -> None:
    caplog.set_level(logging.INFO)
    runner = ShaftOnlineEvalRunner(
        eval_config=online_eval_config(
            {
                "ds_a": json_target_policy(
                    metrics=["parse_success", "exact_match"],
                    primary_metric="exact_match",
                    weight=0.25,
                ),
                "ds_b": json_target_policy(
                    metrics=["parse_success", "exact_match"],
                    primary_metric="exact_match",
                    weight=0.75,
                ),
            }
        ),
        prompt_collator=FakeOnlineEvalPromptCollator(),
    )
    trainer = FakeOnlineEvalTrainer(
        [
            online_eval_batch(
                input_ids=[[11, 12], [21, 22]],
                dataset_names=["ds_a", "ds_b"],
                sample_ids=["a", "b"],
                image_paths=["a.png", "b.png"],
                target_texts=['{"ok": 1}', '{"ok": 2}'],
            )
        ]
    )

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
    runner = ShaftOnlineEvalRunner(
        eval_config=online_eval_config(
            {
                "ds": json_target_policy(
                    metrics=["parse_success"],
                    primary_metric="parse_success",
                )
            }
        ),
        prompt_collator=FakeOnlineEvalPromptCollator(),
    )
    trainer = OnlineEvalPrepareHookTrainer(
        [
            online_eval_batch(
                input_ids=[[11, 12], [21, 22]],
                dataset_names=["ds", "ds"],
                sample_ids=["a", "b"],
                image_paths=["a.png", "b.png"],
                target_texts=['{"ok": 1}', '{"ok": 2}'],
                include_pixels=False,
            )
        ]
    )

    metrics = runner.evaluate(trainer, eval_dataset=object(), metric_key_prefix="eval")

    assert trainer.online_prepare_called is True
    assert metrics["eval_ds_parse_success"] == pytest.approx(0.5)


def test_online_eval_runner_supports_named_eval_datasets() -> None:
    runner = ShaftOnlineEvalRunner(
        eval_config=online_eval_config(
            {
                "ds_a": json_target_policy(metrics=["exact_match"], weight=0.25),
                "ds_b": json_target_policy(metrics=["exact_match"], weight=0.75),
            }
        ),
        prompt_collator=FakeOnlineEvalPromptCollator(),
    )
    trainer = MappedOnlineEvalTrainer(
        {
            "ds_a": [
                online_eval_batch(
                    input_ids=[[11, 12]],
                    dataset_names=["ds_a"],
                    sample_ids=["a"],
                    target_texts=['{"ok": 1}'],
                )
            ],
            "ds_b": [
                online_eval_batch(
                    input_ids=[[21, 22]],
                    dataset_names=["ds_b"],
                    sample_ids=["b"],
                    target_texts=['{"ok": 2}'],
                )
            ],
        }
    )

    metrics = runner.evaluate(
        trainer,
        eval_dataset={"ds_a": "ds_a", "ds_b": "ds_b"},
        metric_key_prefix="eval",
    )

    assert metrics["eval_ds_a_exact_match"] == pytest.approx(1.0)
    assert metrics["eval_ds_b_exact_match"] == pytest.approx(0.0)
    assert metrics["eval_final_score"] == pytest.approx(0.25)


def test_online_eval_runner_slices_left_padded_decoder_prompts_at_input_width() -> None:
    runner = ShaftOnlineEvalRunner(
        eval_config=online_eval_config(
            {
                "ds": json_target_policy(
                    metrics=["parse_success", "exact_match"],
                    primary_metric="exact_match",
                )
            }
        ),
        prompt_collator=FakeOnlineEvalPromptCollator(),
    )
    trainer = LeftPaddedOnlineEvalTrainer(
        [
            online_eval_batch(
                input_ids=[[0, 0, 91, 92], [81, 82, 83, 84]],
                attention_mask=[[0, 0, 1, 1], [1, 1, 1, 1]],
                dataset_names=["ds", "ds"],
                sample_ids=["left-padded", "full-width"],
                image_paths=["left.png", "right.png"],
                target_texts=['{"ok": 1}', '{"ok": 2}'],
            )
        ]
    )

    metrics = runner.evaluate(trainer, eval_dataset=object(), metric_key_prefix="eval")

    assert metrics["eval_ds_parse_success"] == pytest.approx(1.0)
    assert metrics["eval_ds_exact_match"] == pytest.approx(1.0)


def test_online_eval_runner_uses_named_eval_keys_to_avoid_cached_eval_dataloader_collision() -> None:
    runner = ShaftOnlineEvalRunner(
        eval_config=online_eval_config(
            {
                "ds_a": json_target_policy(metrics=["exact_match"], weight=0.25),
                "ds_b": json_target_policy(metrics=["exact_match"], weight=0.75),
            }
        ),
        prompt_collator=FakeOnlineEvalPromptCollator(),
    )
    trainer = PersistentWorkerCacheOnlineEvalTrainer(
        {
            "ds_a": [
                online_eval_batch(
                    input_ids=[[11, 12]],
                    dataset_names=["ds_a"],
                    sample_ids=["a"],
                    target_texts=['{"ok": 1}'],
                )
            ],
            "ds_b": [
                online_eval_batch(
                    input_ids=[[21, 22]],
                    dataset_names=["ds_b"],
                    sample_ids=["b"],
                    target_texts=['{"ok": 2}'],
                )
            ],
        }
    )

    metrics = runner.evaluate(
        trainer,
        eval_dataset={"ds_a": object(), "ds_b": object()},
        metric_key_prefix="eval",
    )

    assert metrics["eval_ds_a_exact_match"] == pytest.approx(1.0)
    assert metrics["eval_ds_b_exact_match"] == pytest.approx(0.0)
    assert metrics["eval_final_score"] == pytest.approx(0.25)
