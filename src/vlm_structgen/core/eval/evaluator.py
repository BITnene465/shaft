from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import MutableMapping
from typing import Any

import torch

from vlm_structgen.core.registry import get_adapter_for_route
from vlm_structgen.core.routing import decode_route_token, encode_route_token, route_metric_label
from vlm_structgen.core.utils.distributed import reduce_numeric_dict, reset_model_runtime_state, unwrap_model
from vlm_structgen.core.utils.generation import (
    build_generate_kwargs,
    trim_generated_ids_at_eos,
)
from vlm_structgen.core.utils.logging import create_progress_bar


class Evaluator:
    def __init__(
        self,
        num_bins: int,
        tokenizer,
        max_new_tokens: int,
        num_beams: int = 1,
        do_sample: bool = False,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        use_cache: bool = True,
        task_route_options: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.use_cache = use_cache
        self.num_bins = int(num_bins)
        self.task_route_options = dict(task_route_options or {})

    def evaluate_model(self, model: torch.nn.Module, dataloader) -> dict[str, float]:
        counts = self._empty_counts()
        raw_model = unwrap_model(model)
        raw_model.eval()
        progress = create_progress_bar(total=len(dataloader), desc="eval", leave=True)
        eval_started_at = time.perf_counter()
        processed_steps = 0
        with torch.no_grad():
            for batch in dataloader:
                batch_counts = self.evaluate_batch(raw_model, batch)
                for key, value in batch_counts.items():
                    counts[key] += value
                processed_steps += 1
                if progress is not None:
                    samples = max(counts["samples"], 1.0)
                    parse_rate_lenient = counts["parse_success_lenient"] / samples
                    parse_rate_strict = counts["parse_success_strict"] / samples
                    avg_step_seconds = (time.perf_counter() - eval_started_at) / max(processed_steps, 1)
                    progress.set_postfix(
                        {
                            "parseL": f"{parse_rate_lenient:.2f}",
                            "parseS": f"{parse_rate_strict:.2f}",
                            "step_s": f"{avg_step_seconds:.2f}",
                        }
                    )
                    progress.update(1)
        if progress is not None:
            progress.close()
        reduced = reduce_numeric_dict(counts, average=False)
        return self.summarize(reduced)

    def evaluate_batch(self, model: torch.nn.Module, batch: dict[str, Any]) -> dict[str, float]:
        generate_inputs = {
            "input_ids": batch["input_ids"].to(next(model.parameters()).device),
            "attention_mask": batch["attention_mask"].to(next(model.parameters()).device),
            "pixel_values": batch["pixel_values"].to(next(model.parameters()).device),
        }
        if batch.get("mm_token_type_ids") is not None:
            generate_inputs["mm_token_type_ids"] = batch["mm_token_type_ids"].to(next(model.parameters()).device)
        generate_inputs.update(
            build_generate_kwargs(
                self.tokenizer,
                generation_config=getattr(model, "generation_config", None),
                num_bins=self.num_bins,
                prompt_lengths=batch["prompt_lengths"].tolist(),
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
                do_sample=self.do_sample,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                use_cache=self.use_cache,
            )
        )
        input_context_length = int(generate_inputs["input_ids"].shape[1])
        if batch.get("image_grid_thw") is not None:
            generate_inputs["image_grid_thw"] = batch["image_grid_thw"].to(next(model.parameters()).device)
        reset_model_runtime_state(model)
        generated = model.generate(**generate_inputs)
        counts = self._empty_counts()
        eos_token_id = generate_inputs.get("eos_token_id")

        for row_index, _prompt_length in enumerate(batch["prompt_lengths"].tolist()):
            counts["samples"] += 1.0
            route_key = str(batch["meta"]["route"][row_index])
            route_token = encode_route_token(route_key)
            route_options = dict(self.task_route_options.get(route_key, {}))
            adapter = get_adapter_for_route(
                route_key=route_key,
                num_bins=self.num_bins,
                task_options_key=tuple(sorted(route_options.items())),
            )
            generated_ids = generated[row_index, input_context_length:]
            trimmed_ids = trim_generated_ids_at_eos(generated_ids, eos_token_id)
            decoded_text = self.tokenizer.decode(trimmed_ids, skip_special_tokens=False)
            strict_text = self.tokenizer.decode(trimmed_ids, skip_special_tokens=True)
            image_width = int(batch["meta"]["image_width"][row_index])
            image_height = int(batch["meta"]["image_height"][row_index])
            gt_struct = batch["meta"]["gt_struct"][row_index]
            parse_error_lenient = None
            pred_struct = None
            try:
                pred_struct = adapter.decode(
                    decoded_text,
                    image_width=image_width,
                    image_height=image_height,
                )
                counts["parse_success_lenient"] += 1.0
                counts[f"__route__::{route_token}::parse_success_lenient"] += 1.0
            except Exception as exc:  # noqa: BLE001
                pred_struct = adapter.empty_prediction()
                parse_error_lenient = str(exc)
            if parse_error_lenient is None:
                try:
                    adapter.decode(
                        strict_text,
                        image_width=image_width,
                        image_height=image_height,
                        strict=True,
                    )
                    counts["parse_success_strict"] += 1.0
                    counts[f"__route__::{route_token}::parse_success_strict"] += 1.0
                except Exception:  # noqa: BLE001
                    pass
            local_counts = adapter.score_prediction(
                gt_struct,
                pred_struct,
                eval_options=route_options,
            )
            for key, value in local_counts.items():
                counts[key] += value
                counts[f"__route__::{route_token}::{key}"] += value
            counts[f"__route__::{route_token}::samples"] += 1.0
        return counts

    def summarize(self, counts: dict[str, float]) -> dict[str, float]:
        samples = max(float(counts.get("samples", 0.0)), 1.0)
        summary = {
            "val/parse_rate_lenient": float(counts.get("parse_success_lenient", 0.0)) / samples,
            "val/parse_rate_strict": float(counts.get("parse_success_strict", 0.0)) / samples,
        }
        route_counts = self._extract_route_counts(counts)
        route_summaries: dict[str, dict[str, float]] = {}
        route_scores: list[tuple[float, float]] = []
        for route_key, route_count in sorted(route_counts.items(), key=lambda item: item[0]):
            adapter = get_adapter_for_route(
                route_key=route_key,
                num_bins=self.num_bins,
                task_options_key=tuple(sorted(dict(self.task_route_options.get(
                    route_key,
                    {},
                )).items())),
            )
            route_summary = adapter.summarize_eval_counts(route_count)
            route_summaries[route_key] = route_summary
            route_prefix = f"val/routes/{route_metric_label(route_key)}"
            summary[f"{route_prefix}/samples"] = float(route_count.get("samples", 0.0))
            for metric_name, metric_value in route_summary.items():
                summary[f"{route_prefix}/{metric_name}"] = metric_value

            spec = self._resolve_route_metric_spec(
                route_key=route_key,
                default_metric_name=adapter.default_eval_primary_metric(),
            )
            primary_metric_name = str(spec["metric_name"])
            if primary_metric_name not in route_summary:
                available_metrics = sorted(route_summary.keys())
                raise ValueError(
                    "Missing primary metric in route summary. "
                    f"route={route_key!r}, expected_metric={primary_metric_name!r}, "
                    f"available_metrics={available_metrics}."
                )
            primary_metric_value = route_summary[primary_metric_name]
            normalized_primary_metric = self._normalize_metric_value(
                primary_metric_value,
                normalizer=str(spec["normalizer"]),
                metric_min=spec["metric_min"],
                metric_max=spec["metric_max"],
            )
            summary[f"{route_prefix}/normalized_primary_metric"] = normalized_primary_metric
            summary[f"{route_prefix}/primary_metric_weight"] = float(spec["weight"])
            route_scores.append((normalized_primary_metric, float(spec["weight"])))

        if route_scores:
            total_weight = sum(weight for _score, weight in route_scores)
            if total_weight > 0:
                summary["val/multi_task_score"] = sum(score * weight for score, weight in route_scores) / total_weight

        if len(route_summaries) == 1:
            only_route_summary = next(iter(route_summaries.values()))
            for metric_name, metric_value in only_route_summary.items():
                summary[f"val/{metric_name}"] = float(metric_value)
        return summary

    def _extract_route_counts(self, counts: dict[str, float]) -> dict[str, dict[str, float]]:
        route_counts: dict[str, dict[str, float]] = defaultdict(dict)
        for key, value in counts.items():
            if not key.startswith("__route__::"):
                continue
            _prefix, route_token, stat_name = key.split("::", 2)
            route_counts[decode_route_token(route_token)][stat_name] = float(value)
        return route_counts

    def _resolve_route_metric_spec(
        self,
        *,
        route_key: str,
        default_metric_name: str,
    ) -> dict[str, Any]:
        route_options = dict(self.task_route_options.get(route_key, {}))
        metric_name = route_options.get("eval_primary_metric")
        if metric_name is None or metric_name == "":
            metric_name = str(default_metric_name)
        metric_name = self._normalize_metric_name(metric_name)
        return {
            "metric_name": metric_name,
            "weight": float(route_options.get("eval_metric_weight", 1.0)),
            "normalizer": str(route_options.get("eval_metric_normalizer", "identity")),
            "metric_min": route_options.get("eval_metric_min"),
            "metric_max": route_options.get("eval_metric_max"),
        }

    def _normalize_metric_value(
        self,
        value: float,
        *,
        normalizer: str,
        metric_min: float | None,
        metric_max: float | None,
    ) -> float:
        normalized = float(value)
        if normalizer in {"identity", "none"}:
            return self._clamp01(normalized)
        if normalizer in {"inverse", "one_minus"}:
            return self._clamp01(1.0 - normalized)
        if normalizer in {"minmax", "scale"}:
            low = 0.0 if metric_min is None else float(metric_min)
            high = 1.0 if metric_max is None else float(metric_max)
            if high <= low:
                return self._clamp01(normalized)
            return self._clamp01((normalized - low) / (high - low))
        raise ValueError(f"Unsupported metric normalizer: {normalizer!r}.")

    def _normalize_metric_name(self, metric_name: str) -> str:
        normalized_name = str(metric_name).strip()
        if normalized_name.startswith("val/"):
            return normalized_name.removeprefix("val/")
        return normalized_name

    def _clamp01(self, value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def _empty_counts(self) -> MutableMapping[str, float]:
        counts: MutableMapping[str, float] = defaultdict(float)
        counts.update({
            "samples": 0.0,
            "parse_success_lenient": 0.0,
            "parse_success_strict": 0.0,
        })
        return counts
