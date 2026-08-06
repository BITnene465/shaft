from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import torch.distributed as dist

from shaft.config import RuntimeConfig
import shaft.model.artifact_identity as artifact_identity
from shaft.model.resolution import (
    materialize_resolved_model_artifact_identity,
    resolve_model_plan,
    validate_resolved_model_artifact,
)


def main(model_dir_raw: str, output_path_raw: str, mode: str) -> None:
    dist.init_process_group(backend="gloo")
    rank = int(os.environ["RANK"])
    model_dir = Path(model_dir_raw)
    output_path = Path(output_path_raw)

    original_hash = artifact_identity._file_sha256
    original_stat = artifact_identity._stat_signature
    hashed_files: list[str] = []

    def counted_hash(path: Path) -> str:
        hashed_files.append(path.name)
        return original_hash(path)

    artifact_identity._file_sha256 = counted_hash
    if mode not in {"shared", "distinct-stat"}:
        raise ValueError(f"Unsupported artifact identity probe mode: {mode!r}.")
    if mode == "distinct-stat" and rank == 1:
        def distinct_stat(path: Path) -> dict[str, int]:
            value = dict(original_stat(path))
            value["inode"] += 10_000
            return value

        artifact_identity._stat_signature = distinct_stat
    try:
        config = RuntimeConfig()
        config.model.model_type = "qwen3vl"
        config.model.model_name_or_path = str(model_dir)
        plan = resolve_model_plan(
            config,
            require_immutable_artifact=False,
        )
        plan = materialize_resolved_model_artifact_identity(plan)
        validate_resolved_model_artifact(plan)
    finally:
        artifact_identity._file_sha256 = original_hash
        artifact_identity._stat_signature = original_stat

    gathered: list[dict[str, object] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(
        gathered,
        {
            "rank": rank,
            "hashed_files": hashed_files,
            "fingerprint": plan.artifact_identity.fingerprint,
            "manifest": plan.artifact_identity.file_manifest,
        },
    )
    if rank == 0:
        output_path.write_text(json.dumps(gathered, indent=2), encoding="utf-8")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
