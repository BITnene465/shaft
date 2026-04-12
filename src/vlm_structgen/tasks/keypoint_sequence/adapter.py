from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from vlm_structgen.core.registry import register_task_adapter
from vlm_structgen.domains.arrow.codecs.keypoint_sequence import KeypointSequenceCodec
from vlm_structgen.domains.arrow.task_support import BaseArrowAdapter, empty_counts


@dataclass
class ArrowKeypointSequenceAdapter(BaseArrowAdapter):
    task_type: str = field(init=False, default="keypoint_sequence")
    coordinate_token_loss_weight: float = 1.0

    def __post_init__(self) -> None:
        if float(self.coordinate_token_loss_weight) < 1.0:
            raise ValueError(
                f"coordinate_token_loss_weight must be >= 1.0, got {self.coordinate_token_loss_weight!r}."
            )

    @property
    def weighted_loss_enabled(self) -> bool:
        return float(self.coordinate_token_loss_weight) > 1.0

    def build_gt_struct_from_record(self, record: dict[str, Any]) -> dict[str, Any]:
        instances = record.get("instances", [])
        if len(instances) != 1:
            raise ValueError("keypoint_sequence samples must contain exactly one instance.")
        instance = instances[0]
        return {
            "label": instance["label"],
            "keypoints": instance["keypoints"],
        }

    def encode_target_text(self, gt_struct: dict[str, Any], *, image_width: int, image_height: int) -> str:
        return self.codec.encode(gt_struct.get("keypoints", []), image_width=image_width, image_height=image_height)

    def build_training_target(
        self,
        gt_struct: dict[str, Any],
        *,
        image_width: int,
        image_height: int,
    ) -> dict[str, Any]:
        target_text, loss_meta = self.codec.encode_with_loss_meta(
            gt_struct.get("keypoints", []),
            image_width=image_width,
            image_height=image_height,
        )
        return {
            "target_text": target_text,
            "loss_meta": loss_meta,
        }

    def decode_with_meta(
        self,
        text: str,
        *,
        image_width: int,
        image_height: int,
        strict: bool = False,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return self.codec.decode_with_meta(text, image_width=image_width, image_height=image_height, strict=strict)

    def decode(self, text: str, *, image_width: int, image_height: int, strict: bool = False) -> dict[str, Any]:
        return self.codec.decode(text, image_width=image_width, image_height=image_height, strict=strict)

    def empty_prediction(self) -> dict[str, Any]:
        return {"keypoints": [], "keypoints_2d": []}

    def score_prediction(
        self,
        gt_struct: dict[str, Any],
        pred_struct: dict[str, Any],
        *,
        eval_options: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        strict_point_distance_px = float(dict(eval_options or {}).get("strict_point_distance_px", 8.0))
        counts = empty_counts()
        gt_points = gt_struct.get("keypoints", [])
        pred_points = pred_struct.get("keypoints", [])
        counts["gt_instances"] = 1.0
        counts["pred_instances"] = 1.0 if pred_points else 0.0
        if len(gt_points) == len(pred_points):
            counts["keypoint_count_exact"] = 1.0
        point_limit = min(len(gt_points), len(pred_points))
        all_points_strict = len(gt_points) == len(pred_points)
        for point_index in range(point_limit):
            gx, gy = gt_points[point_index][:2]
            px, py = pred_points[point_index][:2]
            distance = math.dist((gx, gy), (px, py))
            counts["point_distance_sum"] += distance
            counts["point_count"] += 1.0
            if distance > strict_point_distance_px:
                all_points_strict = False
        if point_limit != len(gt_points) or point_limit != len(pred_points):
            all_points_strict = False
        if all_points_strict:
            counts["end_to_end_correct"] = 1.0
        return counts

    def summarize_eval_counts(self, counts: dict[str, float]) -> dict[str, float]:
        samples = max(counts.get("samples", 0.0), 1.0)
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

    def default_eval_primary_metric(self) -> str:
        return "end_to_end_score"

    def build_target_token_weights(
        self,
        target_text: str,
        *,
        loss_meta: dict[str, Any] | None,
        tokenizer,
    ) -> list[float] | None:
        if not self.weighted_loss_enabled:
            return [1.0] * len(tokenizer(target_text, add_special_tokens=False)["input_ids"])
        if loss_meta is None:
            raise ValueError(
                "keypoint_sequence weighted token loss requires loss_meta.field_char_spans. "
                f"route={self.task_type}/{self.domain_type}"
            )
        encoded = tokenizer(
            target_text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_offsets_mapping=True,
        )
        offsets = encoded.get("offset_mapping")
        input_ids = encoded.get("input_ids")
        if offsets is None or input_ids is None:
            raise ValueError(
                "keypoint_sequence weighted token loss requires tokenizer offset_mapping/input_ids. "
                f"route={self.task_type}/{self.domain_type}"
            )

        field_spans = dict(loss_meta.get("field_char_spans", {}))
        weighted_spans = [
            (int(start), int(end), float(self.coordinate_token_loss_weight))
            for start, end in field_spans.get("coordinates", [])
        ]

        weights: list[float] = []
        for start, end in offsets:
            token_weight = 1.0
            if end > start:
                for span_start, span_end, span_weight in weighted_spans:
                    if max(int(start), span_start) < min(int(end), span_end):
                        token_weight = max(token_weight, span_weight)
            weights.append(float(token_weight))
        return weights


def build_keypoint_sequence_adapter(*, domain_type: str, num_bins: int, task_options: dict[str, Any] | None = None):
    task_options = dict(task_options or {})
    if domain_type == "arrow":
        return ArrowKeypointSequenceAdapter(
            codec=KeypointSequenceCodec(num_bins=num_bins),
            coordinate_token_loss_weight=float(task_options.get("coordinate_token_loss_weight", 1.0)),
        )
    raise ValueError(f"Unsupported keypoint_sequence domain_type: {domain_type!r}")


register_task_adapter("keypoint_sequence", build_keypoint_sequence_adapter)
