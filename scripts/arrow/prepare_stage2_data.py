#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from vlm_structgen.domains.arrow.data.two_stage import prepare_stage2_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare stage2 crop dataset from processed arrow annotations.")
    parser.add_argument("--input-dir", required=True, help="Directory containing processed train.jsonl and val.jsonl.")
    parser.add_argument("--output-dir", required=True, help="Directory to write stage2 JSONL and crop images.")
    parser.add_argument("--padding-ratio", type=float, default=None, help="Legacy single padding ratio fallback for stage2 train crops.")
    parser.add_argument("--train-padding-ratios", type=str, default="0.2,0.3,0.45", help="Comma-separated padding ratios for stage2 train crops.")
    parser.add_argument("--val-padding-ratio", type=float, default=0.3, help="Padding ratio used for stage2 validation crops.")
    parser.add_argument("--num-bins", type=int, default=1000, help="Coordinate quantization bins for prompt/target serialization.")
    parser.add_argument("--num-workers", type=int, default=None, help="Number of worker processes for per-image crop export.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = prepare_stage2_data(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        padding_ratio=args.padding_ratio,
        train_padding_ratios=[float(value) for value in args.train_padding_ratios.split(",") if value.strip()],
        val_padding_ratio=args.val_padding_ratio,
        num_bins=args.num_bins,
        num_workers=args.num_workers,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
