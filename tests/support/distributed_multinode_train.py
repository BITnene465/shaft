from __future__ import annotations

import json
import os
from pathlib import Path
import sys

from shaft.cli.train import main as train_main
from shaft.utils.distributed import (
    all_gather_objects,
    get_rank,
    initialize_process_group_if_needed,
)


def _required_int_env(name: str) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        raise RuntimeError(f"Missing torchrun topology environment variable: {name}.")
    try:
        return int(raw_value)
    except ValueError as exc:
        raise RuntimeError(f"Invalid torchrun topology environment variable {name}={raw_value!r}.") from exc


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: distributed_multinode_train.py CONFIG_PATH TOPOLOGY_RESULT_PATH"
        )
    config_path = Path(sys.argv[1]).resolve()
    result_path = Path(sys.argv[2]).resolve()

    initialize_process_group_if_needed(use_cpu=True)
    local_topology = {
        name.lower(): _required_int_env(name)
        for name in (
            "RANK",
            "WORLD_SIZE",
            "LOCAL_RANK",
            "LOCAL_WORLD_SIZE",
            "GROUP_RANK",
        )
    }
    gathered = all_gather_objects(local_topology)

    train_main(["sft", "--config", str(config_path)])

    if get_rank() == 0:
        result_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = result_path.with_suffix(result_path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(gathered, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary_path, result_path)


if __name__ == "__main__":
    main()
