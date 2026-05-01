from __future__ import annotations

import logging

from shaft.config import LoggingConfig
from shaft.observability.logging import _RankFilter, configure_logging
from shaft.training.distributed import barrier_if_distributed
from shaft.training.distributed import destroy_process_group_if_initialized
from shaft.utils import distributed as distributed_utils


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


def test_barrier_if_distributed_passes_nccl_device_ids(monkeypatch) -> None:
    class _FakeDist:
        barrier_kwargs = None

        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def is_initialized() -> bool:
            return True

        @staticmethod
        def get_world_size() -> int:
            return 2

        @staticmethod
        def get_backend() -> str:
            return "nccl"

        @classmethod
        def barrier(cls, **kwargs) -> None:
            cls.barrier_kwargs = kwargs

    monkeypatch.setattr(distributed_utils, "dist", _FakeDist)
    monkeypatch.setattr(distributed_utils.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(distributed_utils.torch.cuda, "current_device", lambda: 1)

    barrier_if_distributed()

    assert _FakeDist.barrier_kwargs == {"device_ids": [1]}


def test_destroy_process_group_if_initialized_calls_dist_destroy(monkeypatch) -> None:
    class _FakeDist:
        destroyed = False

        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def is_initialized() -> bool:
            return True

        @classmethod
        def destroy_process_group(cls) -> None:
            cls.destroyed = True

    monkeypatch.setattr(distributed_utils, "dist", _FakeDist)

    destroy_process_group_if_initialized()

    assert _FakeDist.destroyed is True
