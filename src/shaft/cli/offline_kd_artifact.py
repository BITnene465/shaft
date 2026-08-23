from __future__ import annotations

import argparse

import torch

from shaft.config import load_config
from shaft.offline_kd.producer import (
    OfflineKDDistributionSpec,
    produce_offline_kd_artifact,
)


_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an atomic offline teacher-distribution artifact."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Fixed-target SFT input config; the producer materializes its canonical sequence.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--expected-teacher-checkpoint-fingerprint",
        help="Optional assertion; Shaft always computes the manifest identity from immutable bytes.",
    )
    parser.add_argument(
        "--denylist",
        required=True,
        help="Explicit shaft-offline-kd-denylist-v1 JSON; use an empty contract intentionally.",
    )
    parser.add_argument("--mode", choices=("dense_logits", "topk_tail"), required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--shard-rows", type=int, default=128)
    parser.add_argument("--shard-max-bytes", type=int, default=512 * 1024 * 1024)
    parser.add_argument("--storage-dtype", choices=tuple(_DTYPES), default="float16")
    parser.add_argument("--backend", choices=("hf", "vllm"), default="hf")
    parser.add_argument("--scoring-batch-size", type=int, default=1)
    parser.add_argument(
        "--max-rows",
        type=int,
        help="Optional deterministic producer canary size; becomes part of the sample plan identity.",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-length", type=int)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    distribution_spec = OfflineKDDistributionSpec(
        mode=args.mode,
        temperature=args.temperature if args.mode == "topk_tail" else None,
        top_k=args.top_k,
    )
    produce_offline_kd_artifact(
        config,
        output_dir=args.output_dir,
        distribution_spec=distribution_spec,
        denylist_path=args.denylist,
        shard_rows=args.shard_rows,
        shard_max_bytes=args.shard_max_bytes,
        storage_dtype=_DTYPES[args.storage_dtype],
        scorer_backend=args.backend,
        scoring_batch_size=args.scoring_batch_size,
        max_rows=args.max_rows,
        vllm_options=(
            {
                "tensor_parallel_size": args.tensor_parallel_size,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "max_model_length": args.max_model_length,
                "enforce_eager": args.enforce_eager,
            }
            if args.backend == "vllm"
            else None
        ),
        resume=args.resume,
        expected_teacher_checkpoint_fingerprint=(
            args.expected_teacher_checkpoint_fingerprint
        ),
    )
