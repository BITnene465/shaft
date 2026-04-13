from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vlm_structgen.core.registry import register_task_adapter
from vlm_structgen.domains.arrow.codecs.grounding import GroundingCodec
from vlm_structgen.domains.arrow.task_support import BaseArrowAdapter, empty_counts, match_instances


@dataclass
class ArrowGroundingAdapter(BaseArrowAdapter):
    task_type: str = field(init=False, default="grounding")

    def build_gt_struct_from_record(self, record: dict[str, Any]) -> dict[str, Any]:
        return {
            "instances": [
                {
                    "label": instance["label"],
                    "bbox": instance["bbox"],
                    "keypoints": [],
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
        bbox_iou_threshold = float(dict(eval_options or {}).get("bbox_iou_threshold", 0.5))
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

        counts["bbox_fp"] = float(len(pred_instances) - len(matched_pred))
        counts["bbox_fn"] = float(len(gt_instances) - len(matched_gt))
        return counts

    def summarize_eval_counts(self, counts: dict[str, float]) -> dict[str, float]:
        samples = max(counts.get("samples", 0.0), 1.0)
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

    def default_eval_primary_metric(self) -> str:
        return "bbox_f1_at_iou50"


def build_grounding_adapter(*, domain_type: str, num_bins: int, task_options: dict[str, Any] | None = None):
    del task_options
    if domain_type == "arrow":
        return ArrowGroundingAdapter(codec=GroundingCodec(num_bins=num_bins))
    raise ValueError(f"Unsupported grounding domain_type: {domain_type!r}")


register_task_adapter("grounding", build_grounding_adapter)
