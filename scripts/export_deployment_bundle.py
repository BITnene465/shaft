#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from vlm_structgen.core.modeling import AdapterBundleSpec, export_deployment_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a deployment bundle with one base model and multiple LoRA adapters.")
    parser.add_argument(
        "--base-source-dir",
        required=True,
        help="Base model source directory. For current training checkpoints, pass a LoRA checkpoint with base_model/ inside.",
    )
    parser.add_argument(
        "--adapter",
        action="append",
        nargs=2,
        metavar=("ROUTE", "CHECKPOINT_DIR"),
        default=[],
        help="Adapter mapping in the form ROUTE CHECKPOINT_DIR. Repeat for multiple LoRA checkpoints.",
    )
    parser.add_argument("--output-dir", required=True, help="Output deployment bundle directory.")
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Overwrite output-dir if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter_specs = [
        AdapterBundleSpec(route=route, checkpoint_dir=Path(checkpoint_dir))
        for route, checkpoint_dir in args.adapter
    ]
    result = export_deployment_bundle(
        base_source_dir=args.base_source_dir,
        adapter_specs=adapter_specs,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print("[export] success")
    print(f"[export] output_dir: {result.output_dir}")
    print(f"[export] base_model_dir: {result.base_model_dir}")
    print(f"[export] adapters_manifest_path: {result.adapters_manifest_path}")
    for route, adapter_dir in sorted(result.adapter_dirs.items()):
        print(f"[export] adapter[{route}]: {adapter_dir}")


if __name__ == "__main__":
    main()
