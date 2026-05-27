#!/usr/bin/env python3
from __future__ import annotations

import signal
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_BENCH_ROOT = REPO_ROOT / "projects" / "eval_bench"
sys.path = [
    item
    for item in sys.path
    if Path(item or ".").resolve() != SCRIPT_DIR
]
if str(EVAL_BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_BENCH_ROOT))

from eval_bench.cli import main  # noqa: E402


if __name__ == "__main__":
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    main()
