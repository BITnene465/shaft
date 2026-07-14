from __future__ import annotations

import logging
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from shaft.config import LoggingConfig
from shaft.observability.logging import configure_logging
from shaft.observability.progress import build_progress_manager


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
    config = SimpleNamespace(
        experiment=SimpleNamespace(
            run_id="progress-smoke",
            name="progress-smoke",
            output_dir=str(sync_dir),
        ),
        progress=SimpleNamespace(
            enabled=True,
            display="interactive",
            width=72,
            refresh_interval=0.0,
            leave_completed=False,
            log_interval=30.0,
            persist=False,
        ),
    )
    manager = build_progress_manager(config, stream=sys.stderr)

    if rank != 0:
        if manager.enabled:
            raise RuntimeError("nonzero rank unexpectedly created a progress sink")
        _wait_for(ready)
        logger.warning("rank-one-warning-must-be-hidden")
        peer_done.touch()
        manager.close()
        return

    if not manager.enabled:
        raise RuntimeError("rank zero did not create the configured progress sink")
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
