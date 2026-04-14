#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from shaft.infer import InferPipeline, load_infer_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run configurable multi-stage inference pipeline.")
    parser.add_argument("--config", required=True, help="Infer pipeline YAML config path.")
    parser.add_argument("--image", required=True, help="Image path.")
    parser.add_argument(
        "--inputs",
        default="{}",
        help="JSON string for initial pipeline context (default: {}).",
    )
    args = parser.parse_args()

    config = load_infer_config(args.config)
    pipeline = InferPipeline.from_config(config)
    inputs = json.loads(args.inputs)
    outputs = pipeline.run(image_path=args.image, inputs=inputs)
    print(json.dumps(outputs, ensure_ascii=False, indent=2, default=lambda x: x.__dict__))


if __name__ == "__main__":
    main()

