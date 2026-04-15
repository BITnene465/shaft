from __future__ import annotations

import argparse
import json

from .hf import inspect_hf_artifact, merge_peft_adapter, validate_hf_artifact


def _as_bool(text: str) -> bool:
    normalized = str(text).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {text!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HF-compatible export and merge tools.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Inspect checkpoint/export layout.")
    inspect_parser.add_argument("--path", required=True)

    validate_parser = subparsers.add_parser("validate", help="Validate HF/PEFT export layout.")
    validate_parser.add_argument("--path", required=True)
    validate_parser.add_argument("--finetune-mode", required=True, choices=["full", "lora", "dora", "qlora"])
    validate_parser.add_argument("--model-type", default=None)
    validate_parser.add_argument("--model-name-or-path", default=None)
    validate_parser.add_argument("--template", default=None)

    merge_parser = subparsers.add_parser("merge-peft", help="Merge a PEFT adapter into a HF full export.")
    merge_parser.add_argument("--model-type", required=True)
    merge_parser.add_argument("--adapter-path", required=True)
    merge_parser.add_argument("--output-dir", required=True)
    merge_parser.add_argument("--base-model", dest="base_model_path", default=None)
    merge_parser.add_argument("--template", default=None)
    merge_parser.add_argument("--trust-remote-code", type=_as_bool, default=True)
    merge_parser.add_argument("--torch-dtype", default="bfloat16")
    merge_parser.add_argument("--safe-serialization", type=_as_bool, default=True)
    merge_parser.add_argument("--max-shard-size", default="5GB")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "inspect":
        layout = inspect_hf_artifact(args.path)
        print(
            json.dumps(
                {
                    "path": str(layout.path),
                    "kind": layout.kind,
                    "has_trainer_state": layout.has_trainer_state,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.command == "validate":
        layout = validate_hf_artifact(
            args.path,
            finetune_mode=args.finetune_mode,
            model_type=args.model_type,
            model_name_or_path=args.model_name_or_path,
            template=args.template,
        )
        print(
            json.dumps(
                {
                    "path": str(layout.path),
                    "kind": layout.kind,
                    "validated_as": args.finetune_mode,
                    "ok": True,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.command == "merge-peft":
        result = merge_peft_adapter(
            model_type=args.model_type,
            adapter_path=args.adapter_path,
            output_dir=args.output_dir,
            base_model_path=args.base_model_path,
            template=args.template,
            trust_remote_code=args.trust_remote_code,
            torch_dtype=args.torch_dtype,
            safe_serialization=args.safe_serialization,
            max_shard_size=args.max_shard_size,
        )
        print(
            json.dumps(
                {
                    "output_dir": str(result.output_dir),
                    "base_model_path": result.base_model_path,
                    "adapter_path": str(result.adapter_path),
                    "kind": result.layout.kind,
                    "ok": True,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    raise ValueError(f"Unsupported command: {args.command!r}")
