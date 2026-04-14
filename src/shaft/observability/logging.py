from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm
from transformers.utils import logging as hf_logging

from shaft.config import LoggingConfig
from shaft.utils.distributed import get_rank
from .context import get_log_context, set_log_context


class _TqdmStreamHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:  # noqa: BLE001
            self.handleError(record)


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        context = get_log_context()
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            **context,
        }
        event = getattr(record, "event", None)
        if event is not None:
            payload["event"] = event
        event_fields = getattr(record, "event_fields", None)
        if isinstance(event_fields, dict):
            payload.update(event_fields)
        return json.dumps(payload, ensure_ascii=False)


class _ContextFilter(logging.Filter):
    _DEFAULTS = {
        "run_id": "-",
        "algorithm": "-",
        "rank": 0,
    }

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
        context = get_log_context()
        merged: dict[str, Any] = dict(self._DEFAULTS)
        merged.update(context)
        merged["rank"] = get_rank()
        for key, value in merged.items():
            if not hasattr(record, key):
                setattr(record, key, value)
        return True


class _RankFilter(logging.Filter):
    def __init__(self, *, rank: int, rank_zero_only: bool) -> None:
        super().__init__()
        self.rank = int(rank)
        self.rank_zero_only = bool(rank_zero_only)

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
        if not self.rank_zero_only:
            return True
        if self.rank == 0:
            return True
        return int(record.levelno) >= int(logging.WARNING)


def configure_logging(config: LoggingConfig, *, run_id: str | None = None) -> None:
    if run_id is not None:
        set_log_context(run_id=run_id)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, config.level, logging.INFO))
    context_filter = _ContextFilter()
    rank_filter = _RankFilter(rank=get_rank(), rank_zero_only=config.rank_zero_only)

    if config.fmt == "json":
        formatter: logging.Formatter = _JsonFormatter()
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | run_id=%(run_id)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    stream_handler = _TqdmStreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.addFilter(context_filter)
    stream_handler.addFilter(rank_filter)
    root.addHandler(stream_handler)

    if config.file_path:
        file_path = Path(config.file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.addFilter(context_filter)
        file_handler.addFilter(rank_filter)
        root.addHandler(file_handler)

    hf_logging.set_verbosity(getattr(logging, config.level, logging.INFO))

    logging.getLogger(__name__).info("logging configured")
