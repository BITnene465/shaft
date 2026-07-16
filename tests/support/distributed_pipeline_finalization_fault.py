from __future__ import annotations

import os
from pathlib import Path
import sys

import torch.distributed as dist

from shaft.config import load_config
from shaft.pipeline import run_sft
import shaft.pipeline.sft as sft_pipeline


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: distributed_pipeline_finalization_fault.py CONFIG MODE"
        )
    config = load_config(Path(sys.argv[1]))
    mode = str(sys.argv[2])
    if mode == "ensure":
        config.train.save_final_model = True

        def _failing_ensure(*args, **kwargs) -> None:
            _ = args, kwargs
            raise OSError("synthetic rank-zero pipeline export validation failure")

        sft_pipeline.ensure_hf_export_layout = _failing_ensure
        expected = "synthetic rank-zero pipeline export validation failure"
    elif mode == "prune":

        def _failing_prune(*args, **kwargs) -> None:
            _ = args, kwargs
            raise OSError("synthetic rank-zero pipeline output prune failure")

        sft_pipeline.prune_root_output_layout = _failing_prune
        expected = "synthetic rank-zero pipeline output prune failure"
    else:
        raise ValueError(f"unsupported mode: {mode}")

    try:
        run_sft(config)
    except Exception as exc:  # noqa: BLE001 - compare the converged peer failure
        local_failure = (f"{type(exc).__module__}.{type(exc).__qualname__}", str(exc))
    else:  # pragma: no cover - script must observe the injected failure
        raise AssertionError(f"pipeline finalization fault did not fail: {mode}")

    peer_failures: list[tuple[str, str] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(peer_failures, local_failure)
    if any(failure != peer_failures[0] for failure in peer_failures[1:]):
        raise AssertionError(f"ranks observed different pipeline failures: {peer_failures!r}")
    if expected not in local_failure[1]:
        raise AssertionError(f"missing expected failure {expected!r}: {local_failure!r}")
    if dist.get_rank() == 0:
        print(f"pipeline finalization {mode} convergence ok", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
