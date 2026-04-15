from __future__ import annotations

import argparse
import json
from typing import Any

from shaft.infer import InferPipeline, load_infer_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run configurable multi-stage inference pipeline.")
    parser.add_argument("--config", required=True, help="Infer pipeline YAML config path.")
    parser.add_argument("--image", required=True, help="Image path.")
    parser.add_argument(
        "--inputs",
        default="{}",
        help="JSON string for initial pipeline context (default: {}).",
    )
    return parser


def _json_default(value: Any) -> Any:
    if hasattr(value, "__dict__"):
        return value.__dict__
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = load_infer_config(args.config)
    pipeline = InferPipeline.from_config(config)
    inputs = json.loads(args.inputs)
    outputs = pipeline.run(image_path=args.image, inputs=inputs)
    print(json.dumps(outputs, ensure_ascii=False, indent=2, default=_json_default))
