from __future__ import annotations

import ast
from pathlib import Path

import torch
import torch.distributed as dist

from shaft.training.checkpointing import (
    ResolvedResumeCheckpoint,
    ShaftCheckpointProtocol,
    resume_checkpoint_consensus_fingerprints,
)
from shaft.training.resume_contract import converge_training_contract_fingerprints


def main() -> None:
    dist.init_process_group("gloo")
    try:
        rank = dist.get_rank()
        resolved = ResolvedResumeCheckpoint(
            path=Path.cwd() / "checkpoint-17",
            protocol=ShaftCheckpointProtocol.COMMITTED_MANIFEST,
            global_step=17,
            generation_fingerprint=("a" if rank == 0 else "b") * 64,
            commit_fingerprint=("c" if rank == 0 else "d") * 64,
            stat_guard=(),
        )
        failure: str | None = None
        try:
            converge_training_contract_fingerprints(
                stage="resume-generation",
                fingerprints=resume_checkpoint_consensus_fingerprints(
                    resolved,
                    protocol=ShaftCheckpointProtocol.COMMITTED_MANIFEST,
                ),
            )
        except (RuntimeError, ValueError) as exc:
            failure = str(exc)
        else:
            raise AssertionError("Different resume checkpoint generations were accepted.")

        failures: list[str | None] = [None] * dist.get_world_size()
        dist.all_gather_object(failures, failure)
        if any(message != failures[0] for message in failures[1:]):
            raise AssertionError(f"Resume generation error differed by rank: {failures!r}.")
        marker = "Distributed resume-generation contract differs across ranks: "
        if failure is None or marker not in failure:
            raise AssertionError(f"Resume generation drift was not diagnosed: {failure!r}.")
        peer_fingerprints = ast.literal_eval(failure.partition(marker)[2].removesuffix("."))
        if not isinstance(peer_fingerprints, list) or len(peer_fingerprints) != 2:
            raise AssertionError(
                f"Resume generation drift did not report two peer identities: {failure!r}."
            )
        differing_fields = {
            name
            for name in peer_fingerprints[0]
            if peer_fingerprints[0][name] != peer_fingerprints[1][name]
        }
        if differing_fields != {"resume_generation"}:
            raise AssertionError(
                "Same-step resume drift was not isolated to resume_generation: "
                f"{differing_fields!r}; failure={failure!r}."
            )
        if {item["resume_global_step"] for item in peer_fingerprints} != {"17"}:
            raise AssertionError(f"Resume global step unexpectedly drifted: {failure!r}.")

        ready = torch.tensor([1], dtype=torch.int64)
        dist.all_reduce(ready)
        if ready.item() != dist.get_world_size():
            raise AssertionError("Ranks did not converge after resume generation rejection.")
        if rank == 0:
            print("same-step resume generation drift rejected", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
