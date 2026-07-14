from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys

import torch
import torch.distributed as dist

from shaft.observability import ShaftTrainingEfficiencyContract
from shaft.training.efficiency import ShaftTrainingEfficiencyMonitor


_TIMING_FIELDS = {
    "host_batch_acquire_seconds",
    "batch_prepare_seconds",
    "training_step_seconds",
    "optimizer_step_seconds",
    "device_training_seconds",
}


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: distributed_efficiency_checkpoint_validate.py "
            "CHECKPOINT GLOBAL_STEP OUTPUT"
        )
    checkpoint = Path(sys.argv[1])
    global_step = int(sys.argv[2])
    output_path = Path(sys.argv[3])

    dist.init_process_group("gloo")
    try:
        rank = dist.get_rank()
        snapshot_path = checkpoint / f"shaft_training_efficiency_rank{rank}.json"
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        contract = ShaftTrainingEfficiencyContract.from_dict(dict(snapshot["contract"]))
        monitor = ShaftTrainingEfficiencyMonitor.from_checkpoint(
            output_dir=checkpoint.parent,
            checkpoint_dir=checkpoint,
            checkpoint_global_step=global_step,
            device_timing=contract.timing_mode == "cuda_optimizer_frame",
            persist=False,
            contract=contract,
        )
        if not monitor.complete_history:
            raise AssertionError("Checkpoint telemetry did not restore complete history.")
        if monitor.initial_global_step != 0:
            raise AssertionError("Complete telemetry must begin at optimizer step zero.")
        if len(monitor.committed_frames) != global_step:
            raise AssertionError("Checkpoint telemetry frame count differs from global step.")
        if monitor.committed_frames[-1].global_step != global_step:
            raise AssertionError("Checkpoint telemetry does not cover its final global step.")

        normalized_frames = [
            {
                key: value
                for key, value in asdict(frame).items()
                if key not in _TIMING_FIELDS
            }
            for frame in monitor.committed_frames
        ]
        rank_frames: list[list[dict] | None] = [None] * dist.get_world_size()
        dist.all_gather_object(rank_frames, normalized_frames)
        summary, _ = monitor.finalize(
            final_global_step=global_step,
            # Gloo still uses CPU collectives. The logical device selects the
            # timing-coverage validation encoded by the persisted contract.
            device=torch.device(
                "cuda" if contract.timing_mode == "cuda_optimizer_frame" else "cpu"
            ),
        )
        if summary.aggregate is None or summary.aggregate.optimizer_steps != global_step:
            raise AssertionError("Restored telemetry could not produce a complete aggregate.")
        if rank == 0:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(
                    {
                        "complete_history": summary.complete_history,
                        "initial_global_step": summary.initial_global_step,
                        "final_global_step": summary.final_global_step,
                        "rank_frames": rank_frames,
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
