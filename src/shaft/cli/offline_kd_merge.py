from __future__ import annotations

import argparse

from shaft.offline_kd import merge_offline_kd_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge compatible Offline-KD map artifacts.")
    parser.add_argument("--input-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    merge_offline_kd_artifacts(args.input_dir, output_dir=args.output_dir)
