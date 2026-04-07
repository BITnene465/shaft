#!/usr/bin/env python
from __future__ import annotations

import argparse

from vlm_structgen.core.modeling import merge_lora_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge a LoRA training checkpoint into full model weights.")
    parser.add_argument("--checkpoint-dir", required=True, help="Training checkpoint directory (e.g. .../checkpoints/best).")
    parser.add_argument("--output-dir", required=True, help="Output directory for merged full weights.")
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Optional training config path. Only needed when checkpoint meta has no config payload "
            "or when --no-prefer-checkpoint-meta is used."
        ),
    )
    parser.add_argument(
        "--prefer-checkpoint-meta",
        dest="prefer_checkpoint_meta",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer runtime config from checkpoint meta when available.",
    )
    parser.add_argument("--device", default=None, help="Optional device override, e.g. cuda:1 or cpu.")
    parser.add_argument(
        "--safe-serialization",
        dest="safe_serialization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save merged HF weights as safetensors when enabled.",
    )
    parser.add_argument(
        "--export-ft-checkpoint",
        dest="export_ft_checkpoint",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Export a full-ft checkpoint bundle under output_dir/ft_checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = merge_lora_checkpoint(
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        config_path=args.config,
        prefer_checkpoint_meta=args.prefer_checkpoint_meta,
        device_name=args.device,
        safe_serialization=args.safe_serialization,
        export_ft_checkpoint=args.export_ft_checkpoint,
    )
    print("[merge] success")
    print(f"[merge] checkpoint_dir: {result.checkpoint_dir}")
    print(f"[merge] output_dir: {result.output_dir}")
    print(f"[merge] model_source: {result.model_source}")
    print(f"[merge] used_checkpoint_meta_config: {result.used_checkpoint_meta_config}")
    print(f"[merge] ft_checkpoint_dir: {result.ft_checkpoint_dir}")


if __name__ == "__main__":
    main()
