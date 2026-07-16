from __future__ import annotations

import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist

from shaft.pipeline.execution import finalize_training_outputs


class _CollectiveSaveTrainer:
    def __init__(self, mode: str) -> None:
        self.mode = mode

    def save_model(self, *, output_dir: str) -> None:
        _ = output_dir
        marker = torch.tensor([dist.get_rank() + 1], dtype=torch.int64)
        dist.all_reduce(marker)
        if int(marker.item()) != 3:
            raise AssertionError(f"unexpected model-save collective result: {marker.item()}")
        if self.mode == "save_model" and dist.get_rank() == 1:
            raise OSError("synthetic rank-local final model save failure")

    def save_state(self) -> None:
        if self.mode == "save_state" and dist.get_rank() == 0:
            raise OSError("synthetic rank-zero final state failure")


def _run_fault(output_dir: Path, mode: str) -> None:
    best_export_dir = output_dir / mode / "best"

    def _validate_export(_path: Path) -> None:
        if mode == "ensure" and dist.get_rank() == 0:
            raise OSError("synthetic rank-zero export validation failure")

    def _prune_output() -> None:
        if mode == "prune" and dist.get_rank() == 0:
            raise OSError("synthetic rank-zero output prune failure")

    try:
        finalize_training_outputs(
            trainer=_CollectiveSaveTrainer(mode),
            best_export_dir=(
                best_export_dir if mode in {"save_model", "ensure", "prune"} else None
            ),
            save_final_state=mode == "save_state",
            validate_export=(
                _validate_export if mode in {"save_model", "ensure", "prune"} else None
            ),
            prune_output=_prune_output,
        )
    except Exception as exc:  # noqa: BLE001 - compare the converged peer failure
        local_failure = (f"{type(exc).__module__}.{type(exc).__qualname__}", str(exc))
    else:  # pragma: no cover - script must observe every injected failure
        raise AssertionError(f"distributed finalization fault did not fail: {mode}")

    peer_failures: list[tuple[str, str] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(peer_failures, local_failure)
    if any(failure != peer_failures[0] for failure in peer_failures[1:]):
        raise AssertionError(f"ranks observed different finalization failures: {peer_failures!r}")
    expected = {
        "save_model": "synthetic rank-local final model save failure",
        "ensure": "synthetic rank-zero export validation failure",
        "save_state": "synthetic rank-zero final state failure",
        "prune": "synthetic rank-zero output prune failure",
    }[mode]
    if expected not in local_failure[1]:
        raise AssertionError(f"missing expected failure {expected!r}: {local_failure!r}")
    dist.barrier()


def _reject_export_path_drift_before_side_effects(output_dir: Path) -> None:
    calls = {
        "save_model": 0,
        "validate_export": 0,
        "save_state": 0,
        "prune_output": 0,
    }

    class _TrackingTrainer:
        def save_model(self, *, output_dir: str) -> None:
            _ = output_dir
            calls["save_model"] += 1

        def save_state(self) -> None:
            calls["save_state"] += 1

    def _validate_export(_path: Path) -> None:
        calls["validate_export"] += 1

    def _prune_output() -> None:
        calls["prune_output"] += 1

    rank_export_dir = output_dir / f"rank-{dist.get_rank()}" / "best"
    try:
        finalize_training_outputs(
            trainer=_TrackingTrainer(),
            best_export_dir=rank_export_dir,
            save_final_state=True,
            validate_export=_validate_export,
            prune_output=_prune_output,
        )
    except ValueError as exc:
        failure = str(exc)
    else:  # pragma: no cover - path drift must fail before model save
        raise AssertionError("rank-divergent final export path was accepted")

    if "post-training-save-preflight" not in failure:
        raise AssertionError(f"unexpected path-drift failure: {failure!r}")
    peer_calls: list[dict[str, int] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(peer_calls, calls)
    if any(peer != calls for peer in peer_calls):
        raise AssertionError(f"finalization side-effect counts differ: {peer_calls!r}")
    if any(value != 0 for value in calls.values()):
        raise AssertionError(f"path drift reached finalization side effects: {calls!r}")
    dist.barrier()
    if dist.get_rank() == 0:
        print("final export path drift rejected before side effects", flush=True)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: distributed_finalization_fault.py OUTPUT_DIR")
    output_dir = Path(sys.argv[1])
    dist.init_process_group(backend="gloo")
    try:
        for mode in ("save_model", "ensure", "save_state", "prune"):
            _run_fault(output_dir, mode)
        _reject_export_path_drift_before_side_effects(output_dir)
        if dist.get_rank() == 0:
            print("distributed training finalization convergence ok", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
