from __future__ import annotations

import os
from pathlib import Path
import sys

from shaft.config import load_config
import shaft.pipeline.sft as sft_pipeline
from shaft.training.distributed import destroy_process_group_if_initialized


def _fail_nonzero_rank_load(*args, **kwargs):
    _ = args, kwargs
    raise PermissionError("injected nonzero-rank CostPlan mmap failure")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: distributed_cost_plan_load_fault.py CONFIG")
    if int(os.environ.get("RANK", "0")) == 1:
        sft_pipeline.load_cost_plan_manifest = _fail_nonzero_rank_load
    try:
        sft_pipeline.run_sft(load_config(Path(sys.argv[1])))
    finally:
        destroy_process_group_if_initialized()


if __name__ == "__main__":
    main()
