#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor, AutoTokenizer, BitsAndBytesConfig

from vlm_structgen.core.infer import load_inference_runner
from vlm_structgen.core.utils.distributed import unwrap_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export quantization bundle for a stage1 checkpoint.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Training checkpoint dir (adapter-only LoRA checkpoint or legacy state_dict layout).",
    )
    parser.add_argument("--config", default="configs/infer/infer_stage1_grounding.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--quant-mode", choices=["int8", "int4"], default="int8")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save-quantized-model", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def _quant_config(mode: str) -> tuple[BitsAndBytesConfig, dict[str, Any]]:
    if mode == "int8":
        cfg = BitsAndBytesConfig(load_in_8bit=True)
        return cfg, {"load_in_8bit": True}
    cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    return cfg, {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": "bfloat16",
    }


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_dir = output_dir / "merged_fp16"
    merged_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load checkpoint with repo-native pipeline.
    runner = load_inference_runner(
        checkpoint_path=str((project_root / args.checkpoint) if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)),
        config_path=str((project_root / args.config) if not Path(args.config).is_absolute() else Path(args.config)),
        device_name=args.device,
    )

    model = unwrap_model(runner.artifacts.model)

    # 2) Merge LoRA if available, then save merged model/tokenizer/processor.
    if hasattr(model, "merge_and_unload") and callable(getattr(model, "merge_and_unload")):
        model = model.merge_and_unload()

    if hasattr(model, "config"):
        model.config.save_pretrained(merged_dir)
    model.save_pretrained(merged_dir, safe_serialization=True)
    runner.artifacts.tokenizer.save_pretrained(merged_dir)
    runner.artifacts.processor.save_pretrained(merged_dir)

    # 3) Create one-click quantization bundle.
    _cfg, quant_dict = _quant_config(args.quant_mode)
    bundle = {
        "format_version": 1,
        "quant_mode": args.quant_mode,
        "merged_model_dir": str(merged_dir),
        "checkpoint": str(args.checkpoint),
        "infer_config": str(args.config),
        "device": args.device,
        "quant_load_kwargs": quant_dict,
        "notes": "Use scripts/acc_exp/load_quant_bundle.py for one-click loading.",
    }

    quantized_save_status = {"attempted": False, "saved": False, "path": None, "error": None}

    # 4) Optional: try to save quantized weights directly.
    if args.save_quantized_model:
        quantized_save_status["attempted"] = True
        try:
            quant_cfg, _ = _quant_config(args.quant_mode)
            from transformers import Qwen3VLForConditionalGeneration

            q_model = Qwen3VLForConditionalGeneration.from_pretrained(
                merged_dir,
                quantization_config=quant_cfg,
                device_map="auto",
                trust_remote_code=True,
            )
            quantized_dir = output_dir / f"quantized_{args.quant_mode}"
            quantized_dir.mkdir(parents=True, exist_ok=True)
            q_model.save_pretrained(quantized_dir, safe_serialization=True)
            AutoTokenizer.from_pretrained(merged_dir, trust_remote_code=True).save_pretrained(quantized_dir)
            AutoProcessor.from_pretrained(merged_dir, trust_remote_code=True).save_pretrained(quantized_dir)
            quantized_save_status.update({"saved": True, "path": str(quantized_dir)})
        except Exception as exc:  # noqa: BLE001
            quantized_save_status["error"] = str(exc)

    bundle["quantized_save"] = quantized_save_status

    bundle_path = output_dir / "quant_bundle.json"
    bundle_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"bundle": str(bundle_path), "merged_model_dir": str(merged_dir), "quantized_save": quantized_save_status}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
