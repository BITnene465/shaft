from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import torch

from vlm_structgen.core.registry import get_adapter
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
        bbox_iou_threshold: float = 0.5,
        strict_point_distance_px: float = 8.0,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.use_cache = use_cache
        self.bbox_iou_threshold = bbox_iou_threshold
        self.strict_point_distance_px = strict_point_distance_px
        self.num_bins = int(num_bins)
        self.task_route_options = dict(task_route_options or {})

    def evaluate_model(self, model: torch.nn.Module, dataloader) -> dict[str, float]:
        counts = self._empty_counts()
        raw_model = unwrap_model(model)
        raw_model.eval()
        progress = create_progress_bar(total=len(dataloader), desc="eval", leave=True)
        with torch.no_grad():
            for batch in dataloader:
                batch_counts = self.evaluate_batch(raw_model, batch)
                for key, value in batch_counts.items():
                    counts[key] += value
                if progress is not None:
                    samples = max(counts["samples"], 1.0)
                    parse_rate_lenient = counts["parse_success_lenient"] / samples
                    parse_rate_strict = counts["parse_success_strict"] / samples
                    if counts["stage2_samples"] > 0 and counts["structured_samples"] == 0 and counts["grounding_samples"] == 0:
                        e2e = counts["end_to_end_correct"] / max(counts["gt_instances"], 1.0)
                        l2_mean = counts["point_distance_sum"] / max(counts["point_count"], 1.0)
                        progress.set_postfix(
                            {
                                "parseL": f"{parse_rate_lenient:.2f}",
                                "parseS": f"{parse_rate_strict:.2f}",
                                "e2e": f"{e2e:.2f}",
                                "l2": f"{l2_mean:.1f}",
                            }
                        )
                    else:
                        precision = counts["bbox_tp"] / max(counts["bbox_tp"] + counts["bbox_fp"], 1.0)
                        recall = counts["bbox_tp"] / max(counts["bbox_tp"] + counts["bbox_fn"], 1.0)
                        progress.set_postfix(
                            {
                                "parseL": f"{parse_rate_lenient:.2f}",
                                "parseS": f"{parse_rate_strict:.2f}",
                                "p": f"{precision:.2f}",
                                "r": f"{recall:.2f}",
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
            task_type = batch["meta"]["task_type"][row_index]
            domain_type = batch["meta"]["domain_type"][row_index]
            route_key = self._route_key(task_type=str(task_type), domain_type=str(domain_type))
            adapter = get_adapter(
                task_type=task_type,
                domain_type=domain_type,
                num_bins=self.num_bins,
                task_options_key=tuple(sorted(dict(self.task_route_options.get(
                    route_key,
                    {},
                )).items())),
            )
            generated_ids = generated[row_index, input_context_length:]
            trimmed_ids = trim_generated_ids_at_eos(generated_ids, eos_token_id)
            decoded_text = self.tokenizer.decode(trimmed_ids, skip_special_tokens=False)
            strict_text = self.tokenizer.decode(trimmed_ids, skip_special_tokens=True)
            image_width = int(batch["meta"]["image_width"][row_index])
            image_height = int(batch["meta"]["image_height"][row_index])
            gt_struct = batch["meta"]["gt_struct"][row_index]
            parse_error_lenient = None
            parse_error_strict = None
            pred_struct = None
            counts[adapter.task_bucket_key] += 1.0
            try:
                pred_struct = adapter.decode(
                    decoded_text,
                    image_width=image_width,
                    image_height=image_height,
                )
                counts["parse_success_lenient"] += 1.0
                counts[f"__route__::{task_type}::{domain_type}::parse_success_lenient"] += 1.0
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
                    counts[f"__route__::{task_type}::{domain_type}::parse_success_strict"] += 1.0
                except Exception as exc:  # noqa: BLE001
                    parse_error_strict = str(exc)
            else:
                parse_error_strict = parse_error_lenient
            local_counts = adapter.score_prediction(
                gt_struct,
                pred_struct,
                bbox_iou_threshold=self.bbox_iou_threshold,
                strict_point_distance_px=self.strict_point_distance_px,
            )
            for key, value in local_counts.items():
                counts[key] += value
                counts[f"__route__::{task_type}::{domain_type}::{key}"] += value
            counts[f"__route__::{task_type}::{domain_type}::samples"] += 1.0
        return counts

    def summarize(self, counts: dict[str, float]) -> dict[str, float]:
        summary = self._summarize_global_counts(counts)
        route_counts = self._extract_route_counts(counts)
        route_scores: list[tuple[float, float]] = []
        for (task_type, domain_type), route_count in sorted(
            route_counts.items(),
            key=lambda item: (item[0][0], item[0][1]),
        ):
            route_key = self._route_key(task_type=task_type, domain_type=domain_type)
            route_summary = self._summarize_route_counts(route_count, task_type=task_type, domain_type=domain_type)
            route_prefix = f"val/routes/{task_type}__{domain_type}"
            summary[f"{route_prefix}/samples"] = float(route_count.get("samples", 0.0))
            for metric_name, metric_value in route_summary.items():
                summary[f"{route_prefix}/{metric_name}"] = metric_value

            spec = self._resolve_route_metric_spec(task_type=task_type, domain_type=domain_type)
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
        return summary

    def _summarize_global_counts(self, counts: dict[str, float]) -> dict[str, float]:
        if counts["stage2_samples"] > 0 and counts["structured_samples"] == 0 and counts["grounding_samples"] == 0:
            samples = max(counts["samples"], 1.0)
            point_count = max(counts["point_count"], 1.0)
            gt_instances = max(counts["gt_instances"], 1.0)
            matched = max(counts["gt_instances"], 1.0)
            return {
                "val/parse_rate_lenient": counts["parse_success_lenient"] / samples,
                "val/parse_rate_strict": counts["parse_success_strict"] / samples,
                "val/keypoint_l2_mean": counts["point_distance_sum"] / point_count,
                "val/keypoint_count_acc": counts["keypoint_count_exact"] / matched,
                "val/end_to_end_score": counts["end_to_end_correct"] / gt_instances,
            }
        if counts["grounding_samples"] > 0 and counts["structured_samples"] == 0 and counts["stage2_samples"] == 0:
            samples = max(counts["samples"], 1.0)
            tp = counts["bbox_tp"]
            fp = counts["bbox_fp"]
            fn = counts["bbox_fn"]
            matched = max(tp, 1.0)
            precision = tp / max(tp + fp, 1.0)
            recall = tp / max(tp + fn, 1.0)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)
            return {
                "val/parse_rate_lenient": counts["parse_success_lenient"] / samples,
                "val/parse_rate_strict": counts["parse_success_strict"] / samples,
                "val/bbox_precision_at_iou50": precision,
                "val/bbox_f1_at_iou50": f1,
                "val/bbox_recall_at_iou50": recall,
                "val/bbox_iou_mean": counts["bbox_iou_sum"] / matched,
            }
        samples = max(counts["samples"], 1.0)
        tp = counts["bbox_tp"]
        fp = counts["bbox_fp"]
        fn = counts["bbox_fn"]
        matched = max(tp, 1.0)
        point_count = max(counts["point_count"], 1.0)
        gt_instances = max(counts["gt_instances"], 1.0)
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        return {
            "val/parse_rate_lenient": counts["parse_success_lenient"] / samples,
            "val/parse_rate_strict": counts["parse_success_strict"] / samples,
            "val/bbox_precision_at_iou50": precision,
            "val/bbox_f1_at_iou50": f1,
            "val/bbox_recall_at_iou50": recall,
            "val/bbox_iou_mean": counts["bbox_iou_sum"] / matched,
            "val/keypoint_l2_mean": counts["point_distance_sum"] / point_count,
            "val/keypoint_count_acc": counts["keypoint_count_exact"] / matched,
            "val/end_to_end_score": counts["end_to_end_correct"] / gt_instances,
        }

    def _summarize_route_counts(
        self,
        counts: dict[str, float],
        *,
        task_type: str,
        domain_type: str,
    ) -> dict[str, float]:
        samples = max(counts.get("samples", 0.0), 1.0)
        if task_type == "grounding":
            tp = counts.get("bbox_tp", 0.0)
            fp = counts.get("bbox_fp", 0.0)
            fn = counts.get("bbox_fn", 0.0)
            matched = max(tp, 1.0)
            precision = tp / max(tp + fp, 1.0)
            recall = tp / max(tp + fn, 1.0)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)
            return {
                "parse_rate_lenient": counts.get("parse_success_lenient", 0.0) / samples,
                "parse_rate_strict": counts.get("parse_success_strict", 0.0) / samples,
                "bbox_precision_at_iou50": precision,
                "bbox_f1_at_iou50": f1,
                "bbox_recall_at_iou50": recall,
                "bbox_iou_mean": counts.get("bbox_iou_sum", 0.0) / matched,
            }
        if task_type == "keypoint_sequence":
            point_count = max(counts.get("point_count", 0.0), 1.0)
            gt_instances = max(counts.get("gt_instances", 0.0), 1.0)
            matched = max(counts.get("gt_instances", 0.0), 1.0)
            return {
                "parse_rate_lenient": counts.get("parse_success_lenient", 0.0) / samples,
                "parse_rate_strict": counts.get("parse_success_strict", 0.0) / samples,
                "keypoint_l2_mean": counts.get("point_distance_sum", 0.0) / point_count,
                "keypoint_count_acc": counts.get("keypoint_count_exact", 0.0) / matched,
                "end_to_end_score": counts.get("end_to_end_correct", 0.0) / gt_instances,
            }
        if task_type == "joint_structure":
            tp = counts.get("bbox_tp", 0.0)
            fp = counts.get("bbox_fp", 0.0)
            fn = counts.get("bbox_fn", 0.0)
            matched = max(tp, 1.0)
            point_count = max(counts.get("point_count", 0.0), 1.0)
            gt_instances = max(counts.get("gt_instances", 0.0), 1.0)
            precision = tp / max(tp + fp, 1.0)
            recall = tp / max(tp + fn, 1.0)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)
            return {
                "parse_rate_lenient": counts.get("parse_success_lenient", 0.0) / samples,
                "parse_rate_strict": counts.get("parse_success_strict", 0.0) / samples,
                "bbox_precision_at_iou50": precision,
                "bbox_f1_at_iou50": f1,
                "bbox_recall_at_iou50": recall,
                "bbox_iou_mean": counts.get("bbox_iou_sum", 0.0) / matched,
                "keypoint_l2_mean": counts.get("point_distance_sum", 0.0) / point_count,
                "keypoint_count_acc": counts.get("keypoint_count_exact", 0.0) / matched,
                "end_to_end_score": counts.get("end_to_end_correct", 0.0) / gt_instances,
            }
        raise ValueError(f"Unsupported route for evaluation summary: {task_type!r}/{domain_type!r}.")

    def _extract_route_counts(self, counts: dict[str, float]) -> dict[tuple[str, str], dict[str, float]]:
        route_counts: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
        for key, value in counts.items():
            if not key.startswith("__route__::"):
                continue
            _prefix, task_type, domain_type, stat_name = key.split("::", 3)
            route_counts[(task_type, domain_type)][stat_name] = float(value)
        return route_counts

    def _resolve_route_metric_spec(self, *, task_type: str, domain_type: str) -> dict[str, Any]:
        route_key = self._route_key(task_type=task_type, domain_type=domain_type)
        route_options = dict(self.task_route_options.get(route_key, {}))
        metric_name = route_options.get("eval_primary_metric")
        if metric_name is None or metric_name == "":
            metric_name = self._default_primary_metric(task_type=task_type)
        metric_name = self._normalize_metric_name(
            metric_name,
        )
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

    def _default_primary_metric(self, *, task_type: str) -> str:
        if task_type == "grounding":
            return "bbox_f1_at_iou50"
        if task_type in {"keypoint_sequence", "joint_structure"}:
            return "end_to_end_score"
        raise ValueError(f"Unsupported task for primary metric resolution: {task_type!r}.")

    def _route_key(self, *, task_type: str, domain_type: str) -> str:
        return f"{task_type}/{domain_type}"

    def _clamp01(self, value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def _empty_counts(self) -> dict[str, float]:
        return {
            "samples": 0.0,
            "parse_success_lenient": 0.0,
            "parse_success_strict": 0.0,
            "structured_samples": 0.0,
            "grounding_samples": 0.0,
            "stage2_samples": 0.0,
            "gt_instances": 0.0,
            "pred_instances": 0.0,
            "bbox_tp": 0.0,
            "bbox_fp": 0.0,
            "bbox_fn": 0.0,
            "bbox_iou_sum": 0.0,
            "point_distance_sum": 0.0,
            "point_count": 0.0,
            "keypoint_count_exact": 0.0,
            "end_to_end_correct": 0.0,
        }
