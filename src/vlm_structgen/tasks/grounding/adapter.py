from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vlm_structgen.core.registry import register_task_adapter
from vlm_structgen.core.train.weighted_loss import compute_weighted_token_ce_loss
from vlm_structgen.domains.arrow.codecs.grounding import GroundingCodec
from vlm_structgen.domains.arrow.task_support import BaseArrowAdapter, empty_counts, match_instances


@dataclass
class ArrowGroundingAdapter(BaseArrowAdapter):
    task_type: str = field(init=False, default="grounding")
    bbox_token_loss_weight: float = 1.0
    label_token_loss_weight: float = 1.0

    def __post_init__(self) -> None:
        for field_name, value in (
            ("bbox_token_loss_weight", self.bbox_token_loss_weight),
            ("label_token_loss_weight", self.label_token_loss_weight),
        ):
            if float(value) < 1.0:
                raise ValueError(f"{field_name} must be >= 1.0, got {value!r}.")

    @property
    def weighted_loss_enabled(self) -> bool:
        return float(self.bbox_token_loss_weight) > 1.0 or float(self.label_token_loss_weight) > 1.0

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

    def build_training_target(
        self,
        gt_struct: dict[str, Any],
        *,
        image_width: int,
        image_height: int,
    ) -> dict[str, Any]:
        target_text, loss_meta = self.codec.encode_with_loss_meta(
            gt_struct,
            image_width=image_width,
            image_height=image_height,
        )
        return {
            "target_text": target_text,
            "loss_meta": loss_meta,
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

    def compute_loss(self, model_outputs, batch: dict[str, Any], *, tokenizer=None) -> object:
        del tokenizer
        if not self.weighted_loss_enabled:
            return model_outputs.loss
        return compute_weighted_token_ce_loss(
            model_outputs,
            batch,
        )

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
                "grounding weighted token loss requires loss_meta.field_char_spans. "
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
                "grounding weighted token loss requires tokenizer offset_mapping/input_ids. "
                f"route={self.task_type}/{self.domain_type}"
            )

        field_spans = dict(loss_meta.get("field_char_spans", {}))
        weighted_spans: list[tuple[int, int, float]] = []
        for start, end in field_spans.get("label", []):
            weighted_spans.append((int(start), int(end), float(self.label_token_loss_weight)))
        for start, end in field_spans.get("bbox_2d", []):
            weighted_spans.append((int(start), int(end), float(self.bbox_token_loss_weight)))

        weights: list[float] = []
        for start, end in offsets:
            token_weight = 1.0
            if end > start:
                for span_start, span_end, span_weight in weighted_spans:
                    if max(int(start), span_start) < min(int(end), span_end):
                        token_weight = max(token_weight, span_weight)
            weights.append(float(token_weight))
        return weights


def build_grounding_adapter(*, domain_type: str, num_bins: int, task_options: dict[str, Any] | None = None):
    task_options = dict(task_options or {})
    if domain_type == "arrow":
        return ArrowGroundingAdapter(
            codec=GroundingCodec(num_bins=num_bins),
            bbox_token_loss_weight=float(task_options.get("bbox_token_loss_weight", 1.0)),
            label_token_loss_weight=float(task_options.get("label_token_loss_weight", 1.0)),
        )
    raise ValueError(f"Unsupported grounding domain_type: {domain_type!r}")


register_task_adapter("grounding", build_grounding_adapter)
