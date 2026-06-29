from __future__ import annotations

import logging

import pytest

from shaft.codec import ShaftCodecResult
from shaft.config.training import EvalNormalizerConfig
from shaft.training.online_eval import (
    ShaftOnlineEvalRunner,
    ShaftOnlineEvalSample,
    ShaftTargetResult,
)
from tests.support.online_eval import (
    FakeOnlineEvalPromptCollator,
    online_eval_config,
    text_target_policy,
)


def _sample(
    *,
    dataset_name: str,
    sample_id: str,
    prediction_text: str,
    target_text: str,
    valid: bool = True,
    image_path: str | None = None,
) -> ShaftOnlineEvalSample:
    return ShaftOnlineEvalSample(
        dataset_name=dataset_name,
        sample_id=sample_id,
        prediction=ShaftCodecResult(
            raw_text=prediction_text,
            parsed=prediction_text if valid else None,
            valid=valid,
            partial=False,
            error_type=None if valid else "json_decode_error",
            error=None if valid else "bad json",
        ),
        target=ShaftTargetResult(value=target_text, valid=True, error=None),
        meta={"image_path": image_path} if image_path is not None else {},
    )


def test_online_eval_runner_normalizes_with_range() -> None:
    runner = ShaftOnlineEvalRunner(
        eval_config=online_eval_config(
            {
                "ds": text_target_policy(
                    metrics=["exact_match"],
                    normalizer=EvalNormalizerConfig(type="range", min_value=0.0, max_value=2.0),
                )
            }
        ),
        prompt_collator=FakeOnlineEvalPromptCollator(),
    )
    entries = [
        _sample(dataset_name="ds", sample_id="x", prediction_text="a", target_text="a"),
        _sample(dataset_name="ds", sample_id="y", prediction_text="b", target_text="c"),
    ]

    metrics = runner.aggregate_samples(entries, metric_key_prefix="eval")

    assert metrics["eval_ds_exact_match"] == 0.5
    assert metrics["eval_ds_score"] == 0.25


def test_online_eval_runner_skips_dataset_without_samples(caplog) -> None:
    caplog.set_level(logging.WARNING)
    runner = ShaftOnlineEvalRunner(
        eval_config=online_eval_config(
            {
                "ds_a": text_target_policy(metrics=["exact_match"], weight=0.25),
                "ds_b": text_target_policy(metrics=["exact_match"], weight=0.75),
            }
        ),
        prompt_collator=FakeOnlineEvalPromptCollator(),
    )
    entries = [_sample(dataset_name="ds_a", sample_id="x", prediction_text="a", target_text="a")]

    metrics = runner.aggregate_samples(entries, metric_key_prefix="eval")

    assert metrics["eval_ds_a_score"] == pytest.approx(1.0)
    assert "eval_ds_b_score" not in metrics
    assert metrics["eval_final_score"] == pytest.approx(1.0)
    assert "dataset=ds_b has no samples" in caplog.text


def test_online_eval_runner_final_score_is_dataset_weighted_not_sample_weighted() -> None:
    runner = ShaftOnlineEvalRunner(
        eval_config=online_eval_config(
            {
                "layout": text_target_policy(metrics=["exact_match"], weight=0.8),
                "keypoint": text_target_policy(metrics=["exact_match"], weight=0.2),
            }
        ),
        prompt_collator=FakeOnlineEvalPromptCollator(),
    )
    entries = [
        _sample(
            dataset_name="layout",
            sample_id="layout-0",
            prediction_text="ok",
            target_text="ok",
        )
    ]
    entries.extend(
        _sample(
            dataset_name="keypoint",
            sample_id=f"keypoint-{index}",
            prediction_text="miss",
            target_text="ok",
        )
        for index in range(5)
    )

    metrics = runner.aggregate_samples(entries, metric_key_prefix="eval")

    assert metrics["eval_layout_exact_match"] == pytest.approx(1.0)
    assert metrics["eval_keypoint_exact_match"] == pytest.approx(0.0)
    assert metrics["eval_final_score"] == pytest.approx(0.8)


def test_online_eval_runner_deduplicates_gathered_samples_before_metrics() -> None:
    runner = ShaftOnlineEvalRunner(
        eval_config=online_eval_config(
            {
                "ds": text_target_policy(
                    metrics=["parse_success"],
                    primary_metric="parse_success",
                )
            }
        ),
        prompt_collator=FakeOnlineEvalPromptCollator(),
    )
    entries = [
        ShaftOnlineEvalSample(
            dataset_name="ds",
            sample_id="sample-0",
            prediction=ShaftCodecResult(
                raw_text='{"ok": 1}',
                parsed={"ok": 1},
                valid=True,
                partial=False,
                error_type=None,
                error=None,
            ),
            target=ShaftTargetResult(value='{"ok": 1}', valid=True, error=None),
            meta={"image_path": "same.png"},
        ),
        _sample(
            dataset_name="ds",
            sample_id="sample-0",
            prediction_text="oops",
            target_text='{"ok": 1}',
            valid=False,
            image_path="same.png",
        ),
    ]

    metrics = runner.aggregate_samples(entries, metric_key_prefix="eval")

    assert metrics["eval_ds_parse_success"] == pytest.approx(1.0)
    assert metrics["eval_final_score"] == pytest.approx(1.0)
