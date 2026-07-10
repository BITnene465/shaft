from __future__ import annotations

import logging
from pathlib import Path

from shaft.config import LoggingConfig
from shaft.observability.logging import configure_logging
from shaft.training.distributed import barrier_if_distributed
from shaft.training.distributed import destroy_process_group_if_initialized
from shaft.utils import distributed as distributed_utils


def test_configure_logging_suppresses_info_on_nonzero_rank(
    tmp_path: Path,
    monkeypatch,
) -> None:
    log_path = tmp_path / "rank.log"
    monkeypatch.setattr("shaft.observability.logging.get_rank", lambda: 1)
    configure_logging(
        LoggingConfig(rank_zero_only=True, file_path=str(log_path)),
        run_id="demo",
    )

    logger = logging.getLogger("shaft.test")
    logger.info("hidden-info")
    logger.warning("visible-warning")
    for handler in logging.getLogger().handlers:
        handler.flush()

    content = log_path.read_text(encoding="utf-8")
    assert "hidden-info" not in content
    assert "visible-warning" in content


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
