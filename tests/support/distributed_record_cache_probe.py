from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import torch.distributed as dist

from shaft.config import DatasetSourceConfig, RuntimeConfig
from shaft.data import ShaftDataCenter
from shaft.data.sources import ShaftRecordCacheTask


def _write_sources(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for index in range(4):
        path = root / f"source-{index}.jsonl"
        rows = [
            {
                "image_path": str(root / "unused.png"),
                "sample_id": f"{index}-{row_index}",
                "target_text": "{}",
            }
            for row_index in range(index + 1)
        ]
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: distributed_record_cache_probe.py OUTPUT_DIR")
    output_dir = Path(sys.argv[1]).resolve()
    rank = int(os.environ["RANK"])
    dist.init_process_group("gloo")
    try:
        if rank == 0:
            _write_sources(output_dir)
        dist.barrier()

        config = RuntimeConfig()
        config.data.record_cache_dir = str(output_dir / "cache")
        config.data.datasets = [
            DatasetSourceConfig(
                dataset_name=f"dataset-{index}",
                train_path=str(output_dir / f"source-{index}.jsonl"),
                use_for_eval=False,
            )
            for index in range(4)
        ]
        warmed_sources: list[str] = []
        original_warm = ShaftRecordCacheTask.warm

        def traced_warm(self: ShaftRecordCacheTask):
            warmed_sources.append(Path(self.source_path).name)
            return original_warm(self)

        ShaftRecordCacheTask.warm = traced_warm
        try:
            prepared = ShaftDataCenter(config.data, seed=17).prepare_records()
        finally:
            ShaftRecordCacheTask.warm = original_warm
        if set(prepared.train_records) != {
            "dataset-0",
            "dataset-1",
            "dataset-2",
            "dataset-3",
        }:
            raise RuntimeError("Distributed cache probe did not load every dataset.")
        (output_dir / f"rank-{rank}.json").write_text(
            json.dumps(sorted(warmed_sources)),
            encoding="utf-8",
        )
        dist.barrier()

        if rank == 0:
            assignments = [
                set(json.loads((output_dir / f"rank-{index}.json").read_text(encoding="utf-8")))
                for index in range(int(os.environ["LOCAL_WORLD_SIZE"]))
            ]
            if any(len(assignment) != 2 for assignment in assignments):
                raise RuntimeError(f"Unexpected local cache task counts: {assignments}")
            if assignments[0] & assignments[1]:
                raise RuntimeError(f"Local ranks warmed overlapping cache tasks: {assignments}")
            expected = {f"source-{index}.jsonl" for index in range(4)}
            if set().union(*assignments) != expected:
                raise RuntimeError(f"Local ranks did not cover every cache task: {assignments}")
            cache_files = list((output_dir / "cache").glob("*.arrow"))
            if len(cache_files) != 4:
                raise RuntimeError(f"Expected four Arrow caches, found {len(cache_files)}")
            print("distributed record cache warmup probe passed", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
