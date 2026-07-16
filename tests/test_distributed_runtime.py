from __future__ import annotations

import json
import logging
from pathlib import Path

from huggingface_hub import logging as hub_logging
import pytest

from shaft.config import LoggingConfig
from shaft.observability.logging import configure_logging
from shaft.training.distributed import barrier_if_distributed
from shaft.training.distributed import all_gather_objects
from shaft.training.distributed import broadcast_object_from_rank_zero
from shaft.training.distributed import destroy_process_group_if_initialized
from shaft.utils import distributed as distributed_utils


def test_configure_logging_suppresses_all_structured_logs_on_nonzero_rank(
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
    logger.warning("hidden-warning")
    logger.error("hidden-error")
    hub_logging.get_logger("huggingface_hub.rank_test").error("hidden-hub-error")
    for handler in logging.getLogger().handlers:
        handler.flush()

    content = log_path.read_text(encoding="utf-8")
    assert "hidden-info" not in content
    assert "hidden-warning" not in content
    assert "hidden-error" not in content
    assert "hidden-hub-error" not in content


def test_configure_logging_keeps_rank_zero_warnings(
    tmp_path: Path,
    monkeypatch,
) -> None:
    log_path = tmp_path / "rank-zero.log"
    monkeypatch.setattr("shaft.observability.logging.get_rank", lambda: 0)
    configure_logging(
        LoggingConfig(rank_zero_only=True, file_path=str(log_path)),
        run_id="demo",
    )

    logging.getLogger("shaft.test").warning("visible-warning")
    for handler in logging.getLogger().handlers:
        handler.flush()

    assert "visible-warning" in log_path.read_text(encoding="utf-8")


@pytest.mark.parametrize("fmt", ["text", "json"])
def test_all_rank_logging_identifies_rank_and_uses_per_rank_files(
    tmp_path: Path,
    monkeypatch,
    fmt: str,
) -> None:
    log_path = tmp_path / "all-ranks.log"
    monkeypatch.setattr("shaft.observability.logging.get_rank", lambda: 1)
    monkeypatch.setattr("shaft.observability.logging.get_world_size", lambda: 2)
    configure_logging(
        LoggingConfig(
            fmt=fmt,
            rank_zero_only=False,
            file_path=str(log_path),
        ),
        run_id="demo",
    )

    logging.getLogger("shaft.test").warning("rank-local-warning")
    hub_logging.get_logger("huggingface_hub.rank_test").warning("rank-local-hub-warning")
    for handler in logging.getLogger().handlers:
        handler.flush()

    ranked_path = tmp_path / "all-ranks.rank1.log"
    assert ranked_path.exists()
    assert not log_path.exists()
    content = ranked_path.read_text(encoding="utf-8")
    if fmt == "json":
        payloads = [json.loads(line) for line in content.splitlines()]
        assert all(payload["rank"] == 1 for payload in payloads)
        assert any(payload["msg"] == "rank-local-warning" for payload in payloads)
        assert sum(payload["msg"] == "rank-local-hub-warning" for payload in payloads) == 1
    else:
        assert "rank=1" in content
        assert "rank-local-warning" in content
        assert content.count("rank-local-hub-warning") == 1


def test_barrier_if_distributed_noop_without_dist() -> None:
    barrier_if_distributed()


def test_broadcast_object_noop_without_dist() -> None:
    payload = {"ok": True}
    assert broadcast_object_from_rank_zero(payload) is payload


def test_object_gather_noop_without_dist() -> None:
    payload = {"fingerprint": "same"}
    assert all_gather_objects(payload) == [payload]


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


def test_process_group_honors_explicit_cpu_intent_when_cuda_is_visible(monkeypatch) -> None:
    class _FakeDist:
        backend = None

        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def is_initialized() -> bool:
            return False

        @classmethod
        def init_process_group(cls, *, backend: str) -> None:
            cls.backend = backend

    def _unexpected_set_device(_local_rank: int) -> None:
        raise AssertionError("An explicit CPU run must not select a CUDA device.")

    monkeypatch.setattr(distributed_utils, "dist", _FakeDist)
    monkeypatch.setattr(distributed_utils, "get_world_size", lambda: 2)
    monkeypatch.setattr(distributed_utils.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(distributed_utils.torch.cuda, "set_device", _unexpected_set_device)

    distributed_utils.initialize_process_group_if_needed(use_cpu=True)

    assert _FakeDist.backend == "gloo"


def test_process_group_rejects_truthy_non_boolean_cpu_intent() -> None:
    with pytest.raises(TypeError, match="use_cpu intent must be a boolean"):
        distributed_utils.initialize_process_group_if_needed(use_cpu="false")  # type: ignore[arg-type]


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
