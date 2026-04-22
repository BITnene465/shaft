#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from eval_common import build_parser, run_eval  # noqa: E402


def main() -> None:
    parser = build_parser(
        task_name="arrow_subattr_eval",
        default_input="data/keypoint_arrow/sft/val.jsonl",
        default_dataset_name="keypoint_arrow",
        default_codec="json_object",
        default_metrics=("parse_success", "subattr_fields", "keypoint_pck"),
        default_output_subdir="eval/arrow_subattr",
    )
    run_eval(parser.parse_args())


if __name__ == "__main__":
    main()
