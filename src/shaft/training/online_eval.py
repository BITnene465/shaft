from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

import torch
import torch.distributed as dist

from shaft.codec import ShaftCodecResult, decode_with_codec
from shaft.config.training import EvalConfig, EvalDatasetPolicyConfig
from shaft.metrics import build_eval_metric
from shaft.model.generation import restore_model_use_cache, set_model_use_cache
from shaft.plugins import Registry
from shaft.utils.distributed import get_rank, get_world_size, is_distributed, is_rank_zero

logger = logging.getLogger(__name__)

TARGET_ADAPTER_REGISTRY: Registry = Registry("online_eval_target_adapter")


def register_target_adapter(name: str):
    return TARGET_ADAPTER_REGISTRY.register(str(name).strip().lower())


@dataclass(frozen=True)
class ShaftTargetResult:
    value: Any
    valid: bool
    error: str | None = None


@dataclass(frozen=True)
class ShaftOnlineEvalSample:
    dataset_name: str
    sample_id: str
    prediction: ShaftCodecResult
    target: ShaftTargetResult
    meta: dict[str, Any]


@register_target_adapter("target_text")
def target_adapter_text(sample_meta: dict[str, Any], params: dict[str, Any]) -> ShaftTargetResult:
    raw_value = sample_meta.get("target_text")
    if raw_value is None:
        return ShaftTargetResult(value=None, valid=False, error="Missing target_text in sample meta.")
    codec_name = str(params.get("codec", "")).strip().lower()
    if codec_name:
        decoded = decode_with_codec(codec_name, str(raw_value))
        if not decoded.valid:
            return ShaftTargetResult(value=None, valid=False, error=decoded.error)
        return ShaftTargetResult(value=decoded.parsed, valid=True, error=None)
    return ShaftTargetResult(value=str(raw_value), valid=True, error=None)


@register_target_adapter("extra_field")
def target_adapter_extra_field(sample_meta: dict[str, Any], params: dict[str, Any]) -> ShaftTargetResult:
    field_name = str(params.get("field", "")).strip()
    if not field_name:
        return ShaftTargetResult(value=None, valid=False, error="extra_field target adapter requires param 'field'.")
    current: Any = sample_meta.get("extra", {})
    for token in field_name.split("."):
        if not isinstance(current, dict) or token not in current:
            return ShaftTargetResult(
                value=None,
                valid=False,
                error=f"Field {field_name!r} not found in sample meta extra.",
            )
        current = current[token]
    codec_name = str(params.get("codec", "")).strip().lower()
    if codec_name:
        if not isinstance(current, str):
            return ShaftTargetResult(
                value=None,
                valid=False,
                error=f"Field {field_name!r} must be a string when codec is configured.",
            )
        decoded = decode_with_codec(codec_name, current)
        if not decoded.valid:
            return ShaftTargetResult(value=None, valid=False, error=decoded.error)
        return ShaftTargetResult(value=decoded.parsed, valid=True, error=None)
    return ShaftTargetResult(value=current, valid=True, error=None)


class ShaftOnlineEvalRunner:
    def __init__(self, *, eval_config: EvalConfig, prompt_collator: Any) -> None:
        self.eval_config = eval_config
        self.prompt_collator = prompt_collator

    @property
    def enabled(self) -> bool:
        return bool(self.eval_config.enabled and self.eval_config.online_metrics_enabled and self.eval_config.datasets)

    def evaluate(
        self,
        trainer: Any,
        *,
        eval_dataset: Any,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        if not self.enabled or eval_dataset is None:
            return {}
        entries = self.collect_samples(trainer, eval_dataset=eval_dataset)
        metrics = self.aggregate_samples(entries, metric_key_prefix=metric_key_prefix)
        self.log_metrics(metrics, metric_key_prefix=metric_key_prefix)
        return metrics

    def collect_samples(self, trainer: Any, *, eval_dataset: Any) -> list[ShaftOnlineEvalSample]:
        dataloader = self._get_prompt_eval_dataloader(trainer, eval_dataset)
        local_entries: list[ShaftOnlineEvalSample] = []
        model = trainer.model
        was_training = bool(getattr(model, "training", False))
        model.eval()
        previous_use_cache = set_model_use_cache(model, enabled=True)
        try:
            with torch.inference_mode():
                for batch in dataloader:
                    meta = batch.pop("meta", None)
                    if not isinstance(meta, dict):
                        raise ValueError("Online eval requires batch meta from collator.")
                    batch.pop("labels", None)
                    prompt_lengths = batch["attention_mask"].sum(dim=1).tolist()
                    prepared = trainer._prepare_inputs(batch)
                    generated_tokens = model.generate(**prepared, **self._build_generation_kwargs())
                    if isinstance(generated_tokens, tuple):
                        generated_tokens = generated_tokens[0]
                    batch_entries = self._decode_batch_entries(
                        generated_tokens=generated_tokens,
                        prompt_lengths=prompt_lengths,
                        meta=meta,
                    )
                    local_entries.extend(batch_entries)
        finally:
            restore_model_use_cache(model, previous_use_cache)
        if was_training:
            model.train()
        return self._gather_entries(local_entries)

    def aggregate_samples(
        self,
        entries: list[ShaftOnlineEvalSample],
        *,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        metrics: dict[str, float] = {}
        weighted_sum = 0.0
        total_weight = 0.0
        for dataset_name in sorted(self.eval_config.datasets):
            policy = self.eval_config.datasets[dataset_name]
            dataset_entries = [
                entry
                for entry in entries
                if entry.dataset_name == dataset_name
            ]
            if not dataset_entries:
                logger.warning(
                    "[eval] dataset=%s has no samples in this evaluation pass; skipping score aggregation",
                    dataset_name,
                )
                continue
            metric_values = self._compute_dataset_metrics(policy, dataset_entries)
            for metric_name, metric_value in metric_values.items():
                metrics[f"{metric_key_prefix}_{dataset_name}_{metric_name}"] = metric_value
            score = self._normalize_score(metric_values[policy.primary_metric], policy)
            metrics[f"{metric_key_prefix}_{dataset_name}_score"] = score
            weighted_sum += score * float(policy.weight)
            total_weight += float(policy.weight)
        if total_weight == 0:
            logger.warning("[eval] no dataset produced online metrics; final_score defaults to 0.0")
        metrics[f"{metric_key_prefix}_final_score"] = weighted_sum / total_weight if total_weight > 0 else 0.0
        return metrics

    def log_metrics(self, metrics: dict[str, float], *, metric_key_prefix: str = "eval") -> None:
        if not is_rank_zero():
            return
        for dataset_name in sorted(self.eval_config.datasets):
            policy = self.eval_config.datasets[dataset_name]
            score_key = f"{metric_key_prefix}_{dataset_name}_score"
            if score_key not in metrics:
                continue
            parts = []
            for metric in policy.metrics:
                key = f"{metric_key_prefix}_{dataset_name}_{metric.name}"
                if key in metrics:
                    parts.append(f"{metric.name}={metrics[key]:.4g}")
            logger.info(
                "[eval] dataset=%s %s normalized_score=%.4g weight=%.4g",
                dataset_name,
                " ".join(parts),
                float(metrics.get(score_key, 0.0)),
                float(policy.weight),
            )
        logger.info(
            "[eval] final_score=%.4g metric_for_best_model=%s",
            float(metrics.get(f"{metric_key_prefix}_final_score", 0.0)),
            self.eval_config.metric_for_best_model,
        )

    def _get_prompt_eval_dataloader(self, trainer: Any, eval_dataset: Any):
        original_collator = getattr(trainer, "data_collator", None)
        trainer.data_collator = self.prompt_collator
        try:
            return trainer.get_eval_dataloader(eval_dataset)
        finally:
            trainer.data_collator = original_collator

    def _build_generation_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "max_new_tokens": int(self.eval_config.max_new_tokens),
            "do_sample": bool(self.eval_config.do_sample),
        }
        if self.eval_config.do_sample and float(self.eval_config.temperature) > 0:
            kwargs["temperature"] = float(self.eval_config.temperature)
        else:
            kwargs["temperature"] = 1.0
            kwargs["top_p"] = 1.0
            kwargs["top_k"] = 50
        tokenizer = getattr(self.prompt_collator, "tokenizer", None)
        if tokenizer is not None:
            if getattr(tokenizer, "pad_token_id", None) is not None:
                kwargs["pad_token_id"] = int(tokenizer.pad_token_id)
            if getattr(tokenizer, "eos_token_id", None) is not None:
                kwargs["eos_token_id"] = int(tokenizer.eos_token_id)
        return kwargs

    def _decode_batch_entries(
        self,
        *,
        generated_tokens: torch.Tensor,
        prompt_lengths: list[int],
        meta: dict[str, Any],
    ) -> list[ShaftOnlineEvalSample]:
        tokenizer = self.prompt_collator.tokenizer
        template = self.prompt_collator.template
        is_encoder_decoder = bool(
            getattr(getattr(getattr(self.prompt_collator, "model_adapter", None), "capabilities", None), "is_encoder_decoder", False)
        )
        entries: list[ShaftOnlineEvalSample] = []
        batch_size = int(generated_tokens.shape[0])
        for index in range(batch_size):
            row = generated_tokens[index]
            prompt_length = int(prompt_lengths[index])
            completion_ids = row if is_encoder_decoder else row[prompt_length:]
            raw_text = template.decode(tokenizer=tokenizer, token_ids=completion_ids.tolist())
            sample_meta = {
                "dataset_name": meta["dataset_name"][index],
                "sample_id": meta["sample_id"][index],
                "image_path": meta["image_path"][index],
                "target_text": meta["target_text"][index],
                "extra": meta.get("extra", [{}] * batch_size)[index],
            }
            dataset_name = str(sample_meta["dataset_name"])
            policy = self.eval_config.datasets.get(dataset_name)
            if policy is None:
                raise KeyError(f"Online eval policy for dataset={dataset_name!r} is missing.")
            prediction = decode_with_codec(policy.prediction_codec, raw_text)
            target = self._resolve_target(policy, sample_meta)
            if not target.valid:
                raise ValueError(
                    f"Target adapter failed for dataset={dataset_name!r}, sample_id={sample_meta['sample_id']!r}: {target.error}"
                )
            entries.append(
                ShaftOnlineEvalSample(
                    dataset_name=dataset_name,
                    sample_id=str(sample_meta["sample_id"]),
                    prediction=prediction,
                    target=target,
                    meta=sample_meta,
                )
            )
        return entries

    def _resolve_target(self, policy: EvalDatasetPolicyConfig, sample_meta: dict[str, Any]) -> ShaftTargetResult:
        adapter_name = str(policy.target_adapter).strip().lower()
        adapter = TARGET_ADAPTER_REGISTRY.get(adapter_name)
        return adapter(sample_meta, dict(policy.target_adapter_params))

    def _compute_dataset_metrics(
        self,
        policy: EvalDatasetPolicyConfig,
        entries: list[ShaftOnlineEvalSample],
    ) -> dict[str, float]:
        metric_instances = {
            metric.name: build_eval_metric(metric.name, params=metric.params)
            for metric in policy.metrics
        }
        for entry in sorted(entries, key=lambda item: (item.dataset_name, item.sample_id)):
            for metric in metric_instances.values():
                metric.update(
                    prediction=entry.prediction,
                    target=entry.target.value,
                    sample_meta=entry.meta,
                )
        return {
            metric_name: float(metric.compute())
            for metric_name, metric in metric_instances.items()
        }

    def _normalize_score(self, value: float, policy: EvalDatasetPolicyConfig) -> float:
        normalizer = policy.normalizer
        if normalizer.type == "identity":
            return float(min(max(value, 0.0), 1.0))
        if normalizer.type == "range":
            assert normalizer.min_value is not None and normalizer.max_value is not None
            min_value = float(normalizer.min_value)
            max_value = float(normalizer.max_value)
            score = (float(value) - min_value) / (max_value - min_value)
            return float(min(max(score, 0.0), 1.0))
        raise ValueError(f"Unsupported normalizer.type={normalizer.type!r}.")

    def _gather_entries(self, local_entries: list[ShaftOnlineEvalSample]) -> list[ShaftOnlineEvalSample]:
        if not is_distributed():
            return local_entries
        gathered: list[list[ShaftOnlineEvalSample] | None] = [None] * get_world_size()
        dist.all_gather_object(gathered, local_entries)
        if get_rank() == 0:
            merged: list[ShaftOnlineEvalSample] = []
            for part in gathered:
                if part:
                    merged.extend(part)
        else:
            merged = []
        objects: list[Any] = [merged]
        dist.broadcast_object_list(objects, src=0)
        return list(objects[0])
