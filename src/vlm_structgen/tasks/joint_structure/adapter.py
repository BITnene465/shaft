from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from vlm_structgen.core.registry import register_task_adapter
from vlm_structgen.domains.arrow.codecs.structure import ArrowCodec
from vlm_structgen.domains.arrow.task_support import BaseArrowAdapter, empty_counts, match_instances


@dataclass
class ArrowJointStructureAdapter(BaseArrowAdapter):
    task_type: str = field(init=False, default="joint_structure")

    def build_gt_struct_from_record(self, record: dict[str, Any]) -> dict[str, Any]:
        return {
            "instances": [
                {
                    "label": instance["label"],
                    "bbox": instance["bbox"],
                    "keypoints": instance["keypoints"],
                }
                for instance in record.get("instances", [])
            ]
        }

    def encode_target_text(self, gt_struct: dict[str, Any], *, image_width: int, image_height: int) -> str:
        return self.codec.encode(gt_struct, image_width=image_width, image_height=image_height)

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

    def score_prediction(
        self,
        gt_struct: dict[str, Any],
        pred_struct: dict[str, Any],
        *,
        eval_options: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        eval_options = dict(eval_options or {})
        bbox_iou_threshold = float(eval_options.get("bbox_iou_threshold", 0.5))
        strict_point_distance_px = float(eval_options.get("strict_point_distance_px", 8.0))
        counts = empty_counts()
        gt_instances = gt_struct.get("instances", [])
        pred_instances = pred_struct.get("instances", [])
        counts["gt_instances"] = float(len(gt_instances))
        counts["pred_instances"] = float(len(pred_instances))

        matches = match_instances(gt_instances, pred_instances, bbox_iou_threshold=bbox_iou_threshold)
        matched_gt = set()
        matched_pred = set()
        for gt_index, pred_index, iou_value in matches:
            matched_gt.add(gt_index)
            matched_pred.add(pred_index)
            counts["bbox_tp"] += 1.0
            counts["bbox_iou_sum"] += iou_value

            gt_instance = gt_instances[gt_index]
            pred_instance = pred_instances[pred_index]
            gt_points = gt_instance["keypoints"]
            pred_points = pred_instance["keypoints"]
            if len(gt_points) == len(pred_points):
                counts["keypoint_count_exact"] += 1.0
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
            if all_points_strict and iou_value >= bbox_iou_threshold:
                counts["end_to_end_correct"] += 1.0

        counts["bbox_fp"] = float(len(pred_instances) - len(matched_pred))
        counts["bbox_fn"] = float(len(gt_instances) - len(matched_gt))
        return counts

    def summarize_eval_counts(self, counts: dict[str, float]) -> dict[str, float]:
        samples = max(counts.get("samples", 0.0), 1.0)
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

    def default_eval_primary_metric(self) -> str:
        return "end_to_end_score"


def build_joint_structure_adapter(*, domain_type: str, num_bins: int, task_options: dict[str, Any] | None = None):
    del task_options
    if domain_type == "arrow":
        return ArrowJointStructureAdapter(codec=ArrowCodec(num_bins=num_bins))
    raise ValueError(f"Unsupported joint_structure domain_type: {domain_type!r}")


register_task_adapter("joint_structure", build_joint_structure_adapter)
