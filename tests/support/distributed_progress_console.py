from __future__ import annotations

import logging
import os
from pathlib import Path
import sys
import time

from shaft.config import LoggingConfig
from shaft.observability.logging import configure_logging
from shaft.observability.progress import (
    ShaftProgressManager,
    ShaftTerminalProgressSink,
)


def _wait_for(path: Path, *, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for {path}")
        time.sleep(0.01)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: distributed_progress_console.py SYNC_DIR")
    sync_dir = Path(sys.argv[1])
    sync_dir.mkdir(parents=True, exist_ok=True)
    ready = sync_dir / "rank-zero-ready"
    peer_done = sync_dir / "rank-one-done"
    rank = int(os.environ.get("RANK", "0"))
    configure_logging(LoggingConfig(rank_zero_only=True), run_id="progress-smoke")
    logger = logging.getLogger("shaft.progress_smoke")

    if rank != 0:
        _wait_for(ready)
        logger.warning("rank-one-warning-must-be-hidden")
        peer_done.touch()
        return

    sink = ShaftTerminalProgressSink(
        stream=sys.stderr,
        width=72,
        refresh_interval=0.0,
    )
    manager = ShaftProgressManager(run_id="progress-smoke", sinks=[sink])
    train = manager.start_task(
        "train",
        label="train",
        total=10_000,
        unit="step",
        display_rate=True,
    )
    ready.touch()
    _wait_for(peer_done)
    logger.warning("rank-zero-warning")
    train.update(current=1, metrics={"lr": "2.5–5e-7"})
    train.complete()
    manager.close()


if __name__ == "__main__":
    main()
