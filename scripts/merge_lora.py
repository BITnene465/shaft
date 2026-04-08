#!/usr/bin/env python
from __future__ import annotations

import argparse

from vlm_structgen.core.modeling import merge_lora_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a dense model or merge a LoRA adapter into full weights.")
    parser.add_argument("--dense-model", default=None, help="Optional dense model path/name override.")
    parser.add_argument(
        "--lora-adapter",
        default=None,
        help="Optional LoRA adapter directory. Omit to export the dense model only.",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory for exported weights.")
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
        dense_model_name_or_path=args.dense_model,
        lora_adapter_path=args.lora_adapter,
        output_dir=args.output_dir,
        config_path=args.config,
        prefer_checkpoint_meta=args.prefer_checkpoint_meta,
        device_name=args.device,
        safe_serialization=args.safe_serialization,
        export_ft_checkpoint=args.export_ft_checkpoint,
    )
    print("[merge] success")
    print(f"[merge] output_dir: {result.output_dir}")
    print(f"[merge] dense_model_dir: {result.dense_model_dir}")
    print(f"[merge] lora_adapter_dir: {result.lora_adapter_dir}")
    print(f"[merge] model_source: {result.model_source}")
    print(f"[merge] used_checkpoint_meta_config: {result.used_checkpoint_meta_config}")
    print(f"[merge] ft_checkpoint_dir: {result.ft_checkpoint_dir}")


if __name__ == "__main__":
    main()
