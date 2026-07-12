from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import sys

import torch.distributed as dist
from transformers.trainer_callback import TrainerState

from shaft.config import load_config
from shaft.data import ShaftSFTSampleCostProvider as _RealCostProvider
from shaft.pipeline import run_sft
import shaft.pipeline.sft as sft_pipeline


class _RankDependentCostProvider:
    def __init__(self, *args, mode: str, **kwargs) -> None:
        if mode == "constructor_failure" and dist.get_rank() == 1:
            raise RuntimeError("synthetic rank-local provider construction failure")
        self._provider = _RealCostProvider(*args, **kwargs)
        self.fingerprint = self._provider.fingerprint
        self.mode = mode

    def __call__(self, sample_ref):
        if dist.get_rank() == 1 and int(sample_ref.context.draw_id) == 0:
            if self.mode == "provider_failure":
                raise FileNotFoundError("synthetic rank-local media failure")
            cost = self._provider(sample_ref)
            return replace(cost, llm_tokens=cost.llm_tokens + 1)
        return self._provider(sample_ref)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: distributed_bounded_fault.py CONFIG MODE")
    config_path = Path(sys.argv[1])
    mode = str(sys.argv[2])
    if mode in {"constructor_failure", "provider_failure", "cost_drift"}:
        sft_pipeline.ShaftSFTSampleCostProvider = lambda *args, **kwargs: (
            _RankDependentCostProvider(*args, mode=mode, **kwargs)
        )
    elif mode == "checkpoint_write_failure":
        original_save = TrainerState.save_to_json

        def _failing_save(self, json_path: str) -> None:
            if dist.get_rank() == 0 and "checkpoint-2" in str(json_path):
                raise OSError("synthetic bounded trainer-state write failure")
            original_save(self, json_path)

        TrainerState.save_to_json = _failing_save
    else:
        raise ValueError(f"unsupported mode: {mode}")

    config = load_config(config_path)
    run_sft(config)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
