#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run detection pseudo-KD as independent TP1 GPU maps, then merge."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--denylist", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument(
        "--scoring-batch-size",
        type=int,
        default=16,
        help=(
            "Maximum active async vLLM requests per GPU; each completed request is immediately "
            "replaced after single-sample CPU preparation."
        ),
    )
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--max-model-length", type=int, default=16384)
    parser.add_argument("--generation-max-tokens", type=int, default=8000)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume incomplete rank artifacts and skip ranks with a completed manifest.",
    )
    parser.add_argument("--rank-offset", type=int, default=0)
    parser.add_argument(
        "--source-world-size",
        type=int,
        help="Global GPU/rank count; defaults to --gpus for a single host.",
    )
    parser.add_argument(
        "--allow-existing-root",
        action="store_true",
        help="Allow another host to share the output root while writing disjoint ranks.",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Wait for every global rank manifest and merge them into output-root/merged.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.gpus <= 0 or args.scoring_batch_size <= 0:
        raise ValueError("--gpus and --scoring-batch-size must be > 0")
    source_world_size = args.source_world_size or args.gpus
    if args.rank_offset < 0 or args.rank_offset + args.gpus > source_world_size:
        raise ValueError("Local global-rank range must fit inside --source-world-size.")
    output_root = Path(args.output_root).resolve()
    if output_root.exists() and not (args.resume or args.allow_existing_root):
        raise FileExistsError(f"Map output root already exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=args.resume or args.allow_existing_root)
    processes: list[tuple[int, subprocess.Popen[bytes]]] = []
    logs = []
    try:
        for local_rank in range(args.gpus):
            rank = args.rank_offset + local_rank
            rank_dir = output_root / f"rank-{rank:02d}"
            if args.resume and (rank_dir / "manifest.json").is_file():
                continue
            log_path = output_root / f"rank-{rank:02d}.log"
            log = log_path.open("ab" if args.resume else "wb")
            logs.append(log)
            command = [
                sys.executable,
                "scripts/build_offline_kd_artifact.py",
                "--config",
                args.config,
                "--output-dir",
                str(rank_dir),
                "--denylist",
                args.denylist,
                "--mode",
                "topk_tail",
                "--temperature",
                "1",
                "--top-k",
                "64",
                "--backend",
                "vllm",
                "--tensor-parallel-size",
                "1",
                "--scoring-batch-size",
                str(args.scoring_batch_size),
                "--max-model-length",
                str(args.max_model_length),
                "--pseudo-label-task",
                "detection",
                "--generation-max-tokens",
                str(args.generation_max_tokens),
                "--source-rank",
                str(rank),
                "--source-world-size",
                str(source_world_size),
                # Always use the resumable writer protocol. On a fresh rank this creates the
                # deterministic .building directory; after interruption the same command
                # validates and resumes its committed shard state.
                "--resume",
            ]
            if args.max_rows is not None:
                command.extend(["--max-rows", str(args.max_rows)])
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = str(local_rank)
            environment["TRITON_CACHE_DIR"] = (
                f"/tmp/shaft-detection-kd-triton-{os.getpid()}-{rank}"
            )
            environment["VLLM_CACHE_ROOT"] = (
                f"/tmp/shaft-detection-kd-vllm-{os.getpid()}-{rank}"
            )
            environment["TORCHINDUCTOR_CACHE_DIR"] = (
                f"/tmp/shaft-detection-kd-inductor-{os.getpid()}-{rank}"
            )
            processes.append(
                (
                    rank,
                    subprocess.Popen(
                        command, stdout=log, stderr=subprocess.STDOUT, env=environment
                    ),
                )
            )
        failures = []
        for rank, process in processes:
            return_code = process.wait()
            if return_code != 0:
                failures.append((rank, return_code))
        if failures:
            raise RuntimeError(f"Detection pseudo-KD map workers failed: {failures}")
        if args.merge:
            while not all(
                (output_root / f"rank-{rank:02d}" / "manifest.json").is_file()
                for rank in range(source_world_size)
            ):
                time.sleep(30)
            merge_command = [
                sys.executable,
                "scripts/merge_offline_kd_artifacts.py",
                "--output-dir",
                str(output_root / "merged"),
            ]
            for rank in range(source_world_size):
                merge_command.extend(
                    ["--input-dir", str(output_root / f"rank-{rank:02d}")]
                )
            if not (output_root / "merged" / "manifest.json").is_file():
                subprocess.run(merge_command, check=True)
    except BaseException:
        for _, process in processes:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
        for _, process in processes:
            if process.poll() is None:
                process.wait()
        raise
    finally:
        for log in logs:
            log.close()


if __name__ == "__main__":
    main()
