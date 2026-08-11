from __future__ import annotations

from typing import Any

from shaft.config import resolve_eval_input_policy
from shaft.data import DPOCollator
from shaft.training.eval_policy import log_eval_input_policy


def build_dpo_collator(
    context: Any,
    *,
    min_pixels: int | None,
    max_pixels: int | None,
    pixel_budgets_by_dataset: dict[str, tuple[int | None, int | None]] | None = None,
) -> DPOCollator:
    artifacts = context.artifacts
    config = context.config
    return DPOCollator(
        model_adapter=artifacts.model_adapter,
        template=artifacts.template,
        processor=artifacts.processor,
        tokenizer=artifacts.tokenizer,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        max_length=config.data.max_length,
        add_eos_token=config.data.add_eos_token,
        pixel_budgets_by_dataset=pixel_budgets_by_dataset,
    )


def resolve_rl_eval_input_policy(context: Any):
    policy = resolve_eval_input_policy(
        context.config.eval,
        train_min_pixels=context.config.data.min_pixels,
        train_max_pixels=context.config.data.max_pixels,
    )
    if context.config.eval.enabled:
        log_eval_input_policy(
            policy=policy,
            model_adapter=context.artifacts.model_adapter,
        )
    return policy


def resolve_named_eval_dataset(context: Any) -> tuple[Any, bool]:
    config = context.config
    bundle = context.dataset_bundle
    use_named = bool(
        config.eval.enabled
        and bundle.eval_datasets_by_name
        and config.eval.datasets
        and (
            config.eval.loss_metrics_enabled
            or config.eval.online_metrics_enabled
            or config.eval.metric_for_best_model
            in {"eval_final_loss", "eval_final_score"}
        )
    )
    return (
        bundle.eval_datasets_by_name if use_named else bundle.eval_dataset,
        use_named,
    )
