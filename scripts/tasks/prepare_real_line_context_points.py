#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from shaft.data.real_line_points import prepare_real_line_point_selection


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select every non-empty human line path from compact raw annotations."
    )
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--train-split",
        type=Path,
        default=Path("data/raw/splits/grounding_layout.train.txt"),
    )
    parser.add_argument(
        "--exclude-manifest",
        type=Path,
        default=Path("data/raw/splits/vlm.test.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/reconstruction_v5_7_selection/line_points_real/train.jsonl"),
    )
    parser.add_argument("--workers", type=int, default=40)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    try:
        summary = prepare_real_line_point_selection(
            raw_root=args.raw_root,
            train_split=args.train_split,
            exclude_manifest=args.exclude_manifest,
            output=args.output,
            workers=args.workers,
            clean=args.clean,
        )
    except (FileExistsError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
