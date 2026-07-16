from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist
from transformers.trainer_callback import TrainerCallback, TrainerState

from shaft.config import load_config
from shaft.data import ShaftSFTSampleCostProvider as _RealCostProvider
from shaft.pipeline import run_sft
import shaft.pipeline.sft as sft_pipeline
from shaft.training.sft_trainer import ShaftSFTTrainer


class _RankLocalOnSaveFailure(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):  # noqa: ANN001
        _ = args, kwargs
        if dist.get_rank() == 1 and int(state.global_step) == 2:
            raise RuntimeError("synthetic peer-rank on_save callback failure")
        return control


class _CollectiveAfterOnSaveFailure(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):  # noqa: ANN001
        _ = kwargs
        if int(state.global_step) == 2:
            marker = Path(args.output_dir) / (
                f"unexpected_post_failure_collective_rank{dist.get_rank()}.txt"
            )
            marker.write_text("entered\n", encoding="utf-8")
        dist.barrier()
        return control


class _RankLocalCollectiveOnSave(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):  # noqa: ANN001
        _ = state, kwargs
        marker = Path(args.output_dir) / (
            f"unexpected_rank_local_collective_rank{dist.get_rank()}.txt"
        )
        marker.write_text("entered\n", encoding="utf-8")
        dist.barrier()
        return control


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
    elif mode == "checkpoint_peer_rng_failure":
        original_torch_save = torch.save

        def _failing_torch_save(obj, path, *args, **kwargs):
            path_text = str(path)
            if (
                dist.get_rank() == 1
                and "checkpoint-2" in path_text
                and Path(path_text).name.startswith("rng_state")
            ):
                raise OSError("synthetic peer-rank RNG-state write failure")
            return original_torch_save(obj, path, *args, **kwargs)

        torch.save = _failing_torch_save
    elif mode == "checkpoint_peer_on_save_failure":
        original_init = ShaftSFTTrainer.__init__

        def _init_with_rank_local_on_save_failure(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self.callback_handler.callbacks[0:0] = [
                _RankLocalOnSaveFailure(),
                _CollectiveAfterOnSaveFailure(),
            ]

        ShaftSFTTrainer.__init__ = _init_with_rank_local_on_save_failure
    elif mode == "checkpoint_rank_local_callback_schedule":
        original_init = ShaftSFTTrainer.__init__

        def _init_with_rank_local_collective(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            if dist.get_rank() == 0:
                self.callback_handler.callbacks.insert(0, _RankLocalCollectiveOnSave())

        ShaftSFTTrainer.__init__ = _init_with_rank_local_collective
    else:
        raise ValueError(f"unsupported mode: {mode}")

    config = load_config(config_path)
    run_sft(config)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
