#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from eval_common import (  # noqa: E402
    _build_generation_config,
    _build_infer_adapter,
    _collect_checkpoints,
    _evaluate_records_with_infer,
    _load_records,
    _optional_bool,
    _write_json,
)
from shaft.config import load_config  # noqa: E402


@dataclass(frozen=True)
class EvalTaskSpec:
    name: str
    input_jsonl: str
    dataset_name: str
    codec: str
    metrics: tuple[str, ...]
    primary_metric: str


TASK_SPECS: dict[str, EvalTaskSpec] = {
    "grounding_arrow": EvalTaskSpec(
        name="grounding_arrow",
        input_jsonl="data/grounding_arrow/sft/val.jsonl",
        dataset_name="grounding_arrow",
        codec="json_list",
        metrics=("parse_success", "det_f1", "det_iou"),
        primary_metric="det_f1",
    ),
    "grounding_layout": EvalTaskSpec(
        name="grounding_layout",
        input_jsonl="data/grounding_layout/sft/val.jsonl",
        dataset_name="grounding_layout",
        codec="json_list",
        metrics=("parse_success", "det_f1", "det_iou"),
        primary_metric="det_f1",
    ),
    "keypoint_arrow": EvalTaskSpec(
        name="keypoint_arrow",
        input_jsonl="data/keypoint_arrow/sft/val.jsonl",
        dataset_name="keypoint_arrow",
        codec="json_object",
        metrics=("parse_success", "subattr_fields", "keypoint_pck"),
        primary_metric="keypoint_pck",
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Temporary offline eval for grounding mixed training checkpoints.")
    parser.add_argument(
        "--config",
        default="configs/train/train_sft_4b_grounding.yaml",
        help="Path to training YAML config.",
    )
    parser.add_argument(
        "--tasks",
        default="grounding_arrow,grounding_layout,keypoint_arrow",
        help=f"Comma-separated tasks. Available: {','.join(TASK_SPECS)}",
    )
    parser.add_argument("--checkpoint", action="append", default=[], help="Checkpoint directory.")
    parser.add_argument("--checkpoint-root", default=None, help="Directory to scan checkpoint-*.")
    parser.add_argument("--include-best", action="store_true", default=False, help="Include best/.")
    parser.add_argument("--include-final", action="store_true", default=False, help="Include root ckpt.")
    parser.add_argument("--output-root", default=None, help="Directory for eval outputs.")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit per dataset.")
    parser.add_argument("--batch-size", type=int, default=1, help="Offline eval batch size.")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Override max_new_tokens.")
    parser.add_argument("--do-sample", type=_optional_bool, default=None, help="Override do_sample.")
    parser.add_argument("--temperature", type=float, default=None, help="Override temperature.")
    parser.add_argument("--top-p", type=float, default=None, help="Override top_p.")
    parser.add_argument("--top-k", type=int, default=None, help="Override top_k.")
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="Override repetition_penalty.",
    )
    parser.add_argument("--min-pixels", type=int, default=None, help="Override min_pixels.")
    parser.add_argument("--max-pixels", type=int, default=None, help="Override max_pixels.")
    parser.add_argument("--device", default=None, help="Override device, for example cuda:0.")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=False,
        help="Continue when one checkpoint fails.",
    )
    parser.add_argument(
        "--save-visualizations",
        type=_optional_bool,
        default=True,
        help="Save prediction visualizations.",
    )
    return parser


def _parse_tasks(raw: str) -> list[EvalTaskSpec]:
    names = [item.strip().lower() for item in str(raw).split(",") if item.strip()]
    if not names:
        raise ValueError("tasks cannot be empty.")
    unknown = [name for name in names if name not in TASK_SPECS]
    if unknown:
        raise KeyError(f"Unknown tasks: {unknown}. Available: {sorted(TASK_SPECS)}")
    return [TASK_SPECS[name] for name in names]


def _build_checkpoint_summary(
    *,
    checkpoint: Path,
    task_summaries: dict[str, dict[str, object]],
) -> dict[str, object]:
    primary_scores: list[float] = []
    weighted_scores: list[tuple[float, int]] = []
    for task_name, summary in task_summaries.items():
        spec = TASK_SPECS[task_name]
        metrics = summary.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        raw_score = metrics.get(spec.primary_metric)
        if raw_score is None:
            continue
        score = float(raw_score)
        primary_scores.append(score)
        weighted_scores.append((score, int(summary.get("num_samples", 0))))
    total_weight = sum(weight for _, weight in weighted_scores)
    return {
        "checkpoint": str(checkpoint),
        "datasets": task_summaries,
        "macro_primary_score": (
            float(sum(primary_scores) / len(primary_scores)) if primary_scores else 0.0
        ),
        "weighted_primary_score": (
            float(sum(score * weight for score, weight in weighted_scores) / total_weight)
            if total_weight > 0
            else 0.0
        ),
    }


def main() -> None:
    args = build_parser().parse_args()
    base_config = load_config(args.config)
    task_specs = _parse_tasks(args.tasks)

    explicit = [Path(path).expanduser() for path in args.checkpoint]
    checkpoint_root = Path(args.checkpoint_root).expanduser() if args.checkpoint_root else None
    if checkpoint_root is None and not explicit:
        checkpoint_root = Path(base_config.experiment.output_dir).resolve()
    checkpoints = _collect_checkpoints(
        explicit=explicit,
        root=checkpoint_root,
        include_best=bool(args.include_best),
        include_final=bool(args.include_final),
    )
    if not checkpoints:
        raise ValueError("No valid checkpoints found.")

    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else Path(base_config.experiment.output_dir).resolve() / "eval" / "grounding_suite"
    )
    generation = _build_generation_config(base_config, args)

    records_by_task = {
        spec.name: _load_records(Path(spec.input_jsonl).expanduser().resolve(), spec.dataset_name)
        for spec in task_specs
    }

    checkpoint_summaries: list[dict[str, object]] = []
    failed: list[dict[str, str]] = []
    for checkpoint in checkpoints:
        start = time.perf_counter()
        try:
            infer = _build_infer_adapter(
                checkpoint=checkpoint,
                base_config=base_config,
                generation=generation,
                device=args.device,
                min_pixels=args.min_pixels,
                max_pixels=args.max_pixels,
            )
            try:
                task_summaries: dict[str, dict[str, object]] = {}
                for spec in task_specs:
                    summary, _ = _evaluate_records_with_infer(
                        infer=infer,
                        checkpoint=checkpoint,
                        records=records_by_task[spec.name],
                        codec=spec.codec,
                        metrics=spec.metrics,
                        generation=generation,
                        output_dir=output_root / checkpoint.name / spec.name,
                        batch_size=max(1, int(args.batch_size)),
                        limit=args.limit,
                        save_visualizations=bool(args.save_visualizations),
                    )
                    task_summaries[spec.name] = summary
                    score = float(summary["metrics"].get(spec.primary_metric, 0.0))
                    print(f"[grounding_suite] {checkpoint.name} {spec.name} {spec.primary_metric}={score:.4f}")
            finally:
                del infer
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            checkpoint_summary = _build_checkpoint_summary(checkpoint=checkpoint, task_summaries=task_summaries)
            checkpoint_summary["runtime_sec"] = float(time.perf_counter() - start)
            checkpoint_summary["status"] = "ok"
            _write_json(output_root / checkpoint.name / "summary.json", checkpoint_summary)
            checkpoint_summaries.append(checkpoint_summary)
            print(
                f"[grounding_suite] {checkpoint.name} done in {checkpoint_summary['runtime_sec']:.2f}s "
                f"macro={checkpoint_summary['macro_primary_score']:.4f}"
            )
        except Exception as exc:  # noqa: BLE001
            failed.append(
                {
                    "checkpoint": str(checkpoint),
                    "status": "failed",
                    "error": str(exc),
                }
            )
            print(f"[grounding_suite] {checkpoint.name} failed: {exc}")
            if not args.continue_on_error:
                raise

    aggregate = {
        "task_name": "grounding_suite",
        "config": str(Path(args.config).resolve()),
        "tasks": [spec.name for spec in task_specs],
        "checkpoints": checkpoint_summaries,
        "failed_checkpoints": failed,
        "checkpoint_count": len(checkpoint_summaries),
        "failed_count": len(failed),
    }
    _write_json(output_root / "summary.json", aggregate)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
