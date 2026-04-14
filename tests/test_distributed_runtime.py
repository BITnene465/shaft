from __future__ import annotations

import logging

from shaft.config import LoggingConfig
from shaft.observability.logging import _RankFilter, configure_logging
from shaft.training.distributed import barrier_if_distributed


def test_rank_filter_suppresses_non_warning_logs_on_nonzero_rank() -> None:
    rank_filter = _RankFilter(rank=1, rank_zero_only=True)
    info_record = logging.LogRecord("x", logging.INFO, __file__, 1, "hello", (), None)
    warning_record = logging.LogRecord("x", logging.WARNING, __file__, 1, "warn", (), None)
    assert rank_filter.filter(info_record) is False
    assert rank_filter.filter(warning_record) is True


def test_configure_logging_accepts_rank_zero_only_flag() -> None:
    cfg = LoggingConfig(rank_zero_only=True)
    configure_logging(cfg, run_id="demo")


def test_barrier_if_distributed_noop_without_dist() -> None:
    barrier_if_distributed()
