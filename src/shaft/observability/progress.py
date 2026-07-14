from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
import re
import sys
import threading
import time
from typing import Any, Protocol, TextIO
from types import MappingProxyType
import unicodedata
import uuid

from shaft.utils.distributed import is_rank_zero


PROGRESS_SNAPSHOT_FILENAME = "shaft_progress.json"
_PROGRESS_SCHEMA_VERSION = 1
_LIFECYCLE_EVENTS = frozenset({"started", "succeeded", "failed", "cancelled"})

logger = logging.getLogger(__name__)

_ANSI_CONTROL_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_metric(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _normalize_metric(item())
        except (TypeError, ValueError, RuntimeError):
            pass
    return str(value)


def _single_line_text(value: Any) -> str:
    text = _ANSI_CONTROL_RE.sub("", str(value))
    without_controls = "".join(
        " " if unicodedata.category(character) == "Cc" else character for character in text
    )
    return " ".join(without_controls.split())


def _terminal_character_width(character: str) -> int:
    if unicodedata.combining(character) or unicodedata.category(character) in {
        "Cf",
        "Me",
        "Mn",
    }:
        return 0
    return 2 if unicodedata.east_asian_width(character) in {"F", "W"} else 1


def _display_width(value: str) -> int:
    return sum(_terminal_character_width(character) for character in value)


def _truncate_display(value: str, width: int) -> str:
    rendered: list[str] = []
    current_width = 0
    for character in value:
        character_width = _terminal_character_width(character)
        if current_width + character_width > width:
            break
        rendered.append(character)
        current_width += character_width
    return "".join(rendered)


def _pad_display(value: str, width: int) -> str:
    return value + (" " * max(width - _display_width(value), 0))


def _format_label(value: str, *, width: int = 7) -> str:
    return _pad_display(_truncate_display(_single_line_text(value), width), width)


@dataclass(frozen=True, slots=True)
class ShaftProgressTaskSnapshot:
    task_id: str
    label: str
    state: str
    current: int
    total: int | None
    unit: str
    message: str | None
    metrics: Mapping[str, Any]
    parent_task_id: str | None
    started_at: str
    updated_at: str
    finished_at: str | None
    generation: int
    order: int
    summary_on_complete: bool
    display_rate: bool

    @property
    def progress_fraction(self) -> float | None:
        if self.total is None or self.total <= 0:
            return None
        return min(max(self.current / self.total, 0.0), 1.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "label": self.label,
            "state": self.state,
            "current": self.current,
            "total": self.total,
            "unit": self.unit,
            "message": self.message,
            "metrics": dict(self.metrics),
            "parent_task_id": self.parent_task_id,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "finished_at": self.finished_at,
            "generation": self.generation,
            "order": self.order,
            "summary_on_complete": self.summary_on_complete,
            "display_rate": self.display_rate,
        }


@dataclass(frozen=True, slots=True)
class ShaftProgressSnapshot:
    run_id: str
    attempt_id: str
    status: str
    active_task_id: str | None
    updated_at: str
    tasks: Mapping[str, ShaftProgressTaskSnapshot]
    schema_version: int = _PROGRESS_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "status": self.status,
            "active_task_id": self.active_task_id,
            "updated_at": self.updated_at,
            "tasks": {
                task_id: task.to_dict()
                for task_id, task in sorted(
                    self.tasks.items(),
                    key=lambda item: item[1].order,
                )
            },
        }


@dataclass(frozen=True, slots=True)
class ShaftProgressEvent:
    kind: str
    task_id: str
    revision: int = 0


class ShaftProgressSink(Protocol):
    def publish(
        self,
        snapshot: ShaftProgressSnapshot,
        event: ShaftProgressEvent,
    ) -> None: ...

    def close(self) -> None: ...


@dataclass(slots=True)
class _TaskRecord:
    task_id: str
    label: str
    state: str
    current: int
    total: int | None
    unit: str
    message: str | None
    metrics: dict[str, Any]
    parent_task_id: str | None
    started_at: str
    updated_at: str
    finished_at: str | None
    generation: int
    order: int
    summary_on_complete: bool
    display_rate: bool

    def snapshot(self) -> ShaftProgressTaskSnapshot:
        return ShaftProgressTaskSnapshot(
            task_id=self.task_id,
            label=self.label,
            state=self.state,
            current=self.current,
            total=self.total,
            unit=self.unit,
            message=self.message,
            metrics=MappingProxyType(dict(self.metrics)),
            parent_task_id=self.parent_task_id,
            started_at=self.started_at,
            updated_at=self.updated_at,
            finished_at=self.finished_at,
            generation=self.generation,
            order=self.order,
            summary_on_complete=self.summary_on_complete,
            display_rate=self.display_rate,
        )


class ShaftProgressTask:
    def __init__(
        self,
        manager: ShaftProgressManager,
        *,
        task_id: str,
        generation: int,
    ) -> None:
        self._manager = manager
        self.task_id = task_id
        self.generation = int(generation)

    def update(
        self,
        *,
        current: int | None = None,
        total: int | None = None,
        message: str | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        self._manager._update_task(
            self.task_id,
            self.generation,
            current=current,
            total=total,
            message=message,
            metrics=metrics,
        )

    def advance(
        self,
        amount: int = 1,
        *,
        message: str | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        self._manager._advance_task(
            self.task_id,
            self.generation,
            amount=amount,
            message=message,
            metrics=metrics,
        )

    def set_total(self, total: int | None) -> None:
        self._manager._set_task_total(
            self.task_id,
            self.generation,
            total=total,
        )

    def complete(
        self,
        *,
        message: str | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        self._manager._finish_task(
            self.task_id,
            self.generation,
            state="succeeded",
            message=message,
            metrics=metrics,
        )

    def fail(self, message: str | None = None) -> None:
        self._manager._finish_task(
            self.task_id,
            self.generation,
            state="failed",
            message=message,
            metrics=None,
        )

    def cancel(self, message: str | None = None) -> None:
        self._manager._finish_task(
            self.task_id,
            self.generation,
            state="cancelled",
            message=message,
            metrics=None,
        )

    def __enter__(self) -> ShaftProgressTask:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        _ = exc_type, traceback
        if exc_value is None:
            self.complete()
        else:
            self.fail(str(exc_value) or type(exc_value).__name__)
        return False


class ShaftProgressManager:
    """Single in-process truth for progress tasks; sinks only render snapshots."""

    def __init__(
        self,
        *,
        run_id: str,
        attempt_id: str | None = None,
        sinks: Iterable[ShaftProgressSink] = (),
    ) -> None:
        self.run_id = str(run_id).strip() or "shaft"
        self.attempt_id = str(attempt_id or uuid.uuid4().hex)
        self._sinks = tuple(sinks)
        self._tasks: dict[str, _TaskRecord] = {}
        self._generations: dict[str, int] = {}
        self._order = 0
        self._revision = 0
        self._last_published_revision = 0
        self._pending_publications: dict[
            int, tuple[ShaftProgressSnapshot, ShaftProgressEvent]
        ] = {}
        self._closing = False
        self._closed = False
        self._lock = threading.RLock()
        self._publish_lock = threading.RLock()

    @property
    def enabled(self) -> bool:
        return bool(self._sinks)

    @property
    def snapshot(self) -> ShaftProgressSnapshot:
        with self._lock:
            return self._build_snapshot()

    def is_task_active(self, task_id: str) -> bool:
        with self._lock:
            task = self._tasks.get(str(task_id))
            return task is not None and task.state == "running"

    def record_failure(self, message: str) -> None:
        """Mark active work failed, preserving a failure even before tasks start."""

        with self._lock:
            self._ensure_open()
            active = sorted(
                (task for task in self._tasks.values() if task.state == "running"),
                key=lambda task: task.order,
                reverse=True,
            )
            already_failed = any(task.state == "failed" for task in self._tasks.values())
        for task in active:
            self._finish_task(
                task.task_id,
                task.generation,
                state="failed",
                message=str(message),
                metrics=None,
            )
        if active or already_failed:
            return
        failure = self.start_task(
            "run.failure",
            label="run",
            unit="phase",
            message=str(message),
        )
        failure.fail(str(message))

    def start_task(
        self,
        task_id: str,
        *,
        label: str,
        total: int | None = None,
        initial: int = 0,
        unit: str = "it",
        message: str | None = None,
        parent_task_id: str | None = None,
        metrics: dict[str, Any] | None = None,
        summary_on_complete: bool = False,
        display_rate: bool = False,
    ) -> ShaftProgressTask:
        task_id = str(task_id).strip()
        label = str(label).strip()
        if not task_id or not label:
            raise ValueError("Progress task_id and label must not be empty.")
        total = None if total is None else int(total)
        initial = int(initial)
        if total is not None and total < 0:
            raise ValueError("Progress total must be >= 0 when set.")
        if initial < 0 or (total is not None and initial > total):
            raise ValueError("Progress initial must be within [0, total].")
        parent_task_id = None if parent_task_id is None else str(parent_task_id).strip()
        with self._lock:
            self._ensure_open()
            existing = self._tasks.get(task_id)
            if existing is not None and existing.state == "running":
                raise RuntimeError(f"Progress task is already running: {task_id}")
            if parent_task_id is not None:
                parent = self._tasks.get(parent_task_id)
                if parent is None or parent.state != "running":
                    raise ValueError(f"Progress parent task is not running: {parent_task_id}")
            generation = self._generations.get(task_id, 0) + 1
            self._generations[task_id] = generation
            self._order += 1
            now = _utc_now()
            record = _TaskRecord(
                task_id=task_id,
                label=label,
                state="running",
                current=initial,
                total=total,
                unit=str(unit).strip() or "it",
                message=None if message is None else str(message),
                metrics={
                    str(key): _normalize_metric(value) for key, value in (metrics or {}).items()
                },
                parent_task_id=parent_task_id,
                started_at=now,
                updated_at=now,
                finished_at=None,
                generation=generation,
                order=self._order,
                summary_on_complete=bool(summary_on_complete),
                display_rate=bool(display_rate),
            )
            self._tasks[task_id] = record
            snapshot, event = self._commit_locked("started", task_id)
        self._publish(snapshot, event)
        return ShaftProgressTask(
            self,
            task_id=task_id,
            generation=generation,
        )

    def _advance_task(
        self,
        task_id: str,
        generation: int,
        *,
        amount: int,
        message: str | None,
        metrics: dict[str, Any] | None,
    ) -> None:
        with self._lock:
            task = self._require_running(task_id, generation)
            self._apply_task_update_locked(
                task,
                current=task.current + int(amount),
                total=None,
                message=message,
                metrics=metrics,
            )
            snapshot, event = self._commit_locked("updated", task_id)
        self._publish(snapshot, event)

    def _update_task(
        self,
        task_id: str,
        generation: int,
        *,
        current: int | None,
        total: int | None,
        message: str | None,
        metrics: dict[str, Any] | None,
    ) -> None:
        with self._lock:
            task = self._require_running(task_id, generation)
            self._apply_task_update_locked(
                task,
                current=current,
                total=total,
                message=message,
                metrics=metrics,
            )
            snapshot, event = self._commit_locked("updated", task_id)
        self._publish(snapshot, event)

    @staticmethod
    def _apply_task_update_locked(
        task: _TaskRecord,
        *,
        current: int | None,
        total: int | None,
        message: str | None,
        metrics: dict[str, Any] | None,
    ) -> None:
        if total is not None:
            total = int(total)
            if total < 0 or total < task.current:
                raise ValueError("Progress total cannot be below current.")
            task.total = total
        if current is not None:
            current = int(current)
            if current < task.current:
                raise ValueError("Progress current must be monotonic.")
            if task.total is not None and current > task.total:
                raise ValueError("Progress current cannot exceed total.")
            task.current = current
        if message is not None:
            task.message = str(message)
        if metrics:
            task.metrics.update(
                {str(key): _normalize_metric(value) for key, value in metrics.items()}
            )
        task.updated_at = _utc_now()

    def _set_task_total(
        self,
        task_id: str,
        generation: int,
        *,
        total: int | None,
    ) -> None:
        with self._lock:
            task = self._require_running(task_id, generation)
            if total is not None:
                total = int(total)
                if total < task.current:
                    raise ValueError("Progress total cannot be below current.")
            task.total = total
            task.updated_at = _utc_now()
            snapshot, event = self._commit_locked("updated", task_id)
        self._publish(snapshot, event)

    def _finish_task(
        self,
        task_id: str,
        generation: int,
        *,
        state: str,
        message: str | None,
        metrics: dict[str, Any] | None,
    ) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.generation != int(generation):
                return
            if task.state != "running":
                return
            if metrics:
                task.metrics.update(
                    {str(key): _normalize_metric(value) for key, value in metrics.items()}
                )
            if message is not None:
                task.message = str(message)
            now = _utc_now()
            task.state = state
            task.updated_at = now
            task.finished_at = now
            snapshot, event = self._commit_locked(state, task_id)
        self._publish(snapshot, event)

    def _require_running(self, task_id: str, generation: int) -> _TaskRecord:
        self._ensure_open()
        task = self._tasks.get(str(task_id))
        if task is None or task.generation != int(generation):
            raise RuntimeError(f"Progress task handle is stale: {task_id}")
        if task.state != "running":
            raise RuntimeError(f"Progress task is not running: {task_id}")
        return task

    def _ensure_open(self) -> None:
        if self._closing or self._closed:
            raise RuntimeError("Progress manager is closing or closed.")

    def _build_snapshot(self) -> ShaftProgressSnapshot:
        tasks = {task_id: task.snapshot() for task_id, task in self._tasks.items()}
        active = [task for task in tasks.values() if task.state == "running"]
        active_task = max(active, key=lambda task: task.order) if active else None
        if active_task is not None:
            status = "running"
        elif any(task.state == "failed" for task in tasks.values()):
            status = "failed"
        elif any(task.state == "cancelled" for task in tasks.values()):
            status = "cancelled"
        elif tasks:
            status = "succeeded"
        else:
            status = "idle"
        return ShaftProgressSnapshot(
            run_id=self.run_id,
            attempt_id=self.attempt_id,
            status=status,
            active_task_id=None if active_task is None else active_task.task_id,
            updated_at=_utc_now(),
            tasks=MappingProxyType(tasks),
        )

    def _commit_locked(
        self,
        kind: str,
        task_id: str,
    ) -> tuple[ShaftProgressSnapshot, ShaftProgressEvent]:
        self._revision += 1
        snapshot = self._build_snapshot()
        event = ShaftProgressEvent(kind, task_id, revision=self._revision)
        self._pending_publications[event.revision] = (snapshot, event)
        return snapshot, event

    def _publish(
        self,
        snapshot: ShaftProgressSnapshot,
        event: ShaftProgressEvent,
    ) -> None:
        with self._publish_lock:
            if event.revision <= self._last_published_revision:
                return
            next_revision = self._last_published_revision + 1
            while True:
                with self._lock:
                    publication = self._pending_publications.pop(next_revision, None)
                if publication is None:
                    return
                next_snapshot, next_event = publication
                for sink in self._sinks:
                    try:
                        sink.publish(next_snapshot, next_event)
                    except Exception:  # noqa: BLE001 - progress must not stop training
                        with self._lock:
                            self._sinks = tuple(
                                candidate
                                for candidate in self._sinks
                                if candidate is not sink
                            )
                        try:
                            sink.close()
                        except Exception:  # noqa: BLE001 - best-effort sink cleanup
                            pass
                        logger.warning(
                            "disabled failed progress sink %s",
                            type(sink).__name__,
                            exc_info=True,
                        )
                self._last_published_revision = next_revision
                next_revision += 1

    def close(self) -> None:
        with self._lock:
            if self._closing or self._closed:
                return
            self._closing = True
            active = sorted(
                (task for task in self._tasks.values() if task.state == "running"),
                key=lambda task: task.order,
                reverse=True,
            )
        for task in active:
            self._finish_task(
                task.task_id,
                task.generation,
                state="cancelled",
                message=task.message,
                metrics=None,
            )
        with self._lock:
            self._closed = True
        with self._publish_lock:
            for sink in self._sinks:
                try:
                    sink.close()
                except Exception:  # noqa: BLE001 - progress cleanup is best-effort
                    logger.warning(
                        "failed to close progress sink %s",
                        type(sink).__name__,
                        exc_info=True,
                    )


def _compact_number(value: int | float) -> str:
    number = float(value)
    absolute = abs(number)
    suffix = ""
    scaled = number
    if absolute >= 1_000_000:
        scaled = number / 1_000_000
        suffix = "m"
    elif absolute >= 1_000:
        scaled = number / 1_000
        suffix = "k"
    if suffix:
        scaled_absolute = abs(scaled)
        decimal_places = 2 if scaled_absolute < 10 else 1 if scaled_absolute < 100 else 0
        factor = 10**decimal_places
        truncated = math.trunc(scaled * factor) / factor
        rendered = f"{truncated:.{decimal_places}f}"
        if decimal_places:
            rendered = rendered.rstrip("0").rstrip(".")
        return f"{rendered}{suffix}"
    if number.is_integer():
        return str(int(number))
    return f"{number:.3g}"


def format_progress_percentage(fraction: float) -> str:
    """Format a bounded fraction without claiming completion before it is exact."""

    percentage = min(max(float(fraction) * 100.0, 0.0), 100.0)
    if percentage == 0.0:
        return "0%"
    if percentage == 100.0:
        return "100%"
    if percentage < 0.01:
        return "<0.01%"
    if percentage < 1.0:
        floored = math.floor(percentage * 100.0) / 100.0
        return f"{floored:.2f}%"
    if percentage < 10.0:
        floored = math.floor(percentage * 10.0) / 10.0
        return f"{floored:.1f}%"
    if percentage < 99.0:
        return f"{math.floor(percentage):.0f}%"
    floored = math.floor(percentage * 10.0) / 10.0
    return f"{floored:.1f}%"


def _format_rate(rate: float, *, unit: str) -> str:
    normalized_unit = _single_line_text(unit).strip().lower()
    if rate <= 0:
        return ""
    if normalized_unit == "step":
        seconds_per_step = 1.0 / rate
        if seconds_per_step < 0.01:
            return f"{_compact_number(rate)}step/s"
        if seconds_per_step < 1:
            rendered = f"{seconds_per_step:.3f}"
        elif seconds_per_step < 10:
            rendered = f"{seconds_per_step:.2f}"
        elif seconds_per_step < 100:
            rendered = f"{seconds_per_step:.1f}"
        else:
            rendered = f"{seconds_per_step:.0f}"
        if "." in rendered:
            rendered = rendered.rstrip("0").rstrip(".")
        return f"{rendered}s/step"
    return f"{_compact_number(rate)}/s"


def _stream_supports_unicode(stream: TextIO) -> bool:
    encoding = getattr(stream, "encoding", None)
    if not encoding:
        return True
    try:
        "▏·█".encode(str(encoding))
    except (LookupError, UnicodeEncodeError):
        return False
    return True


def _encoding_safe_text(value: str, encoding: str | None) -> str:
    if not encoding:
        return value
    try:
        value.encode(encoding)
        return value
    except UnicodeEncodeError:
        translated = value.translate(
            str.maketrans(
                {
                    "–": "-",
                    "—": "-",
                    "−": "-",
                    "…": "...",
                }
            )
        )
        return translated.encode(encoding, errors="replace").decode(encoding)
    except LookupError:
        return value.encode("ascii", errors="replace").decode("ascii")


def _format_progress_bar(
    fraction: float,
    *,
    current: int,
    width: int = 8,
    unicode: bool,
) -> str:
    bounded = min(max(float(fraction), 0.0), 1.0)
    if unicode:
        if bounded >= 1.0:
            return "█" * width
        partials = "▏▎▍▌▋▊▉"
        scaled = bounded * width
        full = min(int(scaled), width)
        partial_index = int((scaled - full) * 8)
        if current > 0 and full == 0 and partial_index == 0:
            partial_index = 1
        partial = (
            partials[partial_index - 1]
            if partial_index > 0 and full < width
            else ""
        )
        return "█" * full + partial + "·" * (width - full - len(partial))
    if bounded >= 1.0:
        return "#" * width
    full = min(int(bounded * width), width)
    head = ">" if current > 0 and full < width else ""
    return "#" * full + head + "." * (width - full - len(head))


def _format_duration(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    if 0 < seconds < 1:
        return "<1s"
    seconds = math.ceil(seconds)
    if seconds >= 3600:
        hours, remainder = divmod(seconds, 3600)
        minutes = remainder // 60
        return f"{hours}h{minutes:02d}m" if minutes else f"{hours}h"
    if seconds >= 60:
        minutes, remainder = divmod(seconds, 60)
        return f"{minutes}m{remainder:02d}s"
    return f"{seconds}s"


def _format_metric(key: str, value: Any) -> str:
    label = _single_line_text(
        {
            "learning_rate": "lr",
            "eval_loss": "loss",
        }.get(str(key), str(key))
    )
    if isinstance(value, float):
        if value != 0 and abs(value) < 1e-3:
            mantissa, exponent = f"{value:.1e}".split("e", maxsplit=1)
            rendered = f"{mantissa.removesuffix('.0')}e{int(exponent)}"
        else:
            rendered = f"{value:.3g}"
    else:
        rendered = _single_line_text(value)
    return f"{label} {rendered}"


def _ordered_metrics(metrics: Mapping[str, Any]) -> list[tuple[str, Any]]:
    priority = {"loss": 0, "eval_loss": 0, "lr": 1, "learning_rate": 1}
    return sorted(
        metrics.items(),
        key=lambda item: (priority.get(str(item[0]), 2), str(item[0])),
    )


def _progress_task_field(task: Any, name: str, default: Any = None) -> Any:
    if isinstance(task, Mapping):
        return task.get(name, default)
    return getattr(task, name, default)


def _progress_task_order(task: Any) -> int:
    try:
        return int(_progress_task_field(task, "order", 0) or 0)
    except (OverflowError, TypeError, ValueError):
        return 0


def select_progress_display_task_id(
    tasks: Mapping[str, Any],
    *,
    active_task_id: str | None,
    status: str,
) -> str | None:
    """Select the shared foreground task from a progress snapshot."""

    if active_task_id is not None and str(active_task_id) in tasks:
        return str(active_task_id)
    normalized_status = str(status or "")
    if normalized_status in {"failed", "cancelled"}:
        terminal_tasks = [
            (str(task_id), task)
            for task_id, task in tasks.items()
            if str(_progress_task_field(task, "state", "")) == normalized_status
        ]
        if terminal_tasks:
            return max(
                terminal_tasks,
                key=lambda item: _progress_task_order(item[1]),
            )[0]
    if "train" in tasks:
        return "train"
    if not tasks:
        return None
    return max(
        tasks.items(),
        key=lambda item: _progress_task_order(item[1]),
    )[0]


class ShaftTerminalProgressPresentation:
    """Pure width-aware presentation policy for one terminal progress line."""

    def __init__(
        self,
        *,
        width: int,
        unicode: bool,
        encoding: str | None = None,
    ) -> None:
        self.width = max(int(width), 40)
        self.unicode = bool(unicode)
        self.encoding = encoding

    def format_failure(
        self,
        task: ShaftProgressTaskSnapshot,
        *,
        state: str,
    ) -> str:
        progress = _compact_number(task.current)
        if task.total is not None:
            progress = f"{progress}/{_compact_number(task.total)}"
        parts = [_format_label(task.label), state, progress]
        if task.message:
            parts.append(_single_line_text(task.message))
        return self._finalize(" ".join(parts))

    def format_task(
        self,
        task: ShaftProgressTaskSnapshot,
        *,
        rate: float,
    ) -> str:
        fraction = task.progress_fraction
        if fraction is None:
            parts = [_format_label(task.label)]
            if task.message:
                parts.append(_single_line_text(task.message))
            else:
                parts.append(_compact_number(task.current))
            return self._finalize(" ".join(parts))

        progress_text = (
            f"{_compact_number(task.current)}/{_compact_number(task.total or 0)}"
        )
        percentage = format_progress_percentage(fraction)
        bar = _format_progress_bar(
            fraction,
            current=task.current,
            unicode=self.unicode,
        )
        full_parts = [_format_label(task.label), bar, progress_text, percentage]
        compact_parts = [_format_label(task.label), progress_text, percentage]
        rate_text = _format_rate(rate, unit=task.unit) if task.display_rate else ""
        if rate_text and self._width([*full_parts, rate_text]) > self.width:
            parts = compact_parts
        else:
            parts = full_parts
        if rate_text:
            self._append_if_fits(parts, rate_text)
        if rate > 0 and task.total is not None and task.current < task.total:
            eta = f"ETA {_format_duration((task.total - task.current) / rate)}"
            self._append_if_fits(parts, eta)
        for metric_key, metric_value in _ordered_metrics(task.metrics)[:2]:
            self._append_if_fits(parts, _format_metric(metric_key, metric_value))
        return self._finalize(" ".join(parts))

    def _finalize(self, value: str) -> str:
        safe = _encoding_safe_text(value, self.encoding)
        return _truncate_display(safe, self.width)

    @staticmethod
    def _width(parts: list[str]) -> int:
        return _display_width(" ".join(parts))

    def _append_if_fits(self, parts: list[str], value: str) -> bool:
        if self._width([*parts, value]) > self.width:
            return False
        parts.append(value)
        return True


class ShaftTerminalProgressSink:
    """Render the foreground task on exactly one bounded terminal line."""

    def __init__(
        self,
        *,
        stream: TextIO | None = None,
        width: int = 72,
        refresh_interval: float = 0.5,
        leave_completed: bool = False,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.stream = stream or sys.stderr
        self.width = max(int(width), 40)
        self.refresh_interval = max(float(refresh_interval), 0.0)
        self.leave_completed = bool(leave_completed)
        self.clock = clock
        self._stream_encoding = getattr(self.stream, "encoding", None)
        self.presentation = ShaftTerminalProgressPresentation(
            width=self.width,
            unicode=_stream_supports_unicode(self.stream),
            encoding=self._stream_encoding,
        )
        self._last_rendered_at = float("-inf")
        self._last_line = ""
        self._line_visible = False
        self._rate_samples: dict[
            tuple[str, int], deque[tuple[float, int]]
        ] = {}
        self._active_rate_key: tuple[str, int] | None = None
        self._rate_paused_at: dict[tuple[str, int], float] = {}
        self._closed = False
        self._lock = threading.RLock()
        _register_terminal_sink(self)

    def publish(
        self,
        snapshot: ShaftProgressSnapshot,
        event: ShaftProgressEvent,
    ) -> None:
        with self._lock:
            if self._closed:
                return
            now = self.clock()
            force = event.kind in _LIFECYCLE_EVENTS
            if not force and now - self._last_rendered_at < self.refresh_interval:
                return
            self._last_rendered_at = now
            active = (
                None if snapshot.active_task_id is None else snapshot.tasks[snapshot.active_task_id]
            )
            self._switch_active_rate_task(active, now=now)
            completed = snapshot.tasks.get(event.task_id)
            try:
                if active is not None:
                    if (
                        completed is not None
                        and event.kind in {"failed", "cancelled"}
                        and completed.task_id != active.task_id
                    ):
                        self._draw(self._format_failure(completed, state=event.kind))
                        self.stream.write("\n")
                        self.stream.flush()
                        self._line_visible = False
                        self._last_line = ""
                    elif (
                        completed is not None
                        and event.kind == "succeeded"
                        and completed.task_id != active.task_id
                        and self.leave_completed
                    ):
                        self._draw(self._format_task(completed, now=now))
                        self.stream.write("\n")
                        self.stream.flush()
                        self._line_visible = False
                        self._last_line = ""
                    self._draw(self._format_task(active, now=now))
                    return
                if (
                    snapshot.status in {"failed", "cancelled"}
                    and snapshot.tasks
                ):
                    terminal_task_id = select_progress_display_task_id(
                        snapshot.tasks,
                        active_task_id=snapshot.active_task_id,
                        status=snapshot.status,
                    )
                    if terminal_task_id is None:
                        self._clear()
                        return
                    terminal_task = snapshot.tasks[terminal_task_id]
                    self._draw(
                        self._format_failure(terminal_task, state=snapshot.status)
                    )
                    self.stream.write("\n")
                    self.stream.flush()
                    self._line_visible = False
                    self._last_line = ""
                    return
                if completed is not None and event.kind in {"failed", "cancelled"}:
                    self._draw(self._format_failure(completed, state=event.kind))
                    self.stream.write("\n")
                    self.stream.flush()
                    self._line_visible = False
                    self._last_line = ""
                    return
                if (
                    completed is not None
                    and event.kind == "succeeded"
                    and (completed.summary_on_complete or self.leave_completed)
                ):
                    self._draw(self._format_task(completed, now=now))
                    self.stream.write("\n")
                    self.stream.flush()
                    self._line_visible = False
                    self._last_line = ""
                    return
                self._clear()
            finally:
                if completed is not None and event.kind in {
                    "succeeded",
                    "failed",
                    "cancelled",
                }:
                    self._release_rate_task(completed)

    def _format_failure(
        self,
        task: ShaftProgressTaskSnapshot,
        *,
        state: str,
    ) -> str:
        return self.presentation.format_failure(task, state=state)

    def _format_task(
        self,
        task: ShaftProgressTaskSnapshot,
        *,
        now: float,
    ) -> str:
        return self.presentation.format_task(
            task,
            rate=self._observe_rate(task, now=now),
        )

    def _observe_rate(
        self,
        task: ShaftProgressTaskSnapshot,
        *,
        now: float,
    ) -> float:
        key = (task.task_id, task.generation)
        samples = self._rate_samples.setdefault(key, deque(maxlen=20))
        if not samples or samples[-1][1] != int(task.current):
            samples.append((float(now), int(task.current)))
        if len(samples) < 2:
            return 0.0
        started_at, started_current = samples[0]
        finished_at, finished_current = samples[-1]
        elapsed = finished_at - started_at
        completed = finished_current - started_current
        return completed / elapsed if elapsed > 0 and completed > 0 else 0.0

    def _switch_active_rate_task(
        self,
        task: ShaftProgressTaskSnapshot | None,
        *,
        now: float,
    ) -> None:
        next_key = None if task is None else (task.task_id, task.generation)
        if next_key == self._active_rate_key:
            return
        if self._active_rate_key is not None:
            self._rate_paused_at[self._active_rate_key] = float(now)
        if next_key is not None and next_key in self._rate_paused_at:
            paused_at = self._rate_paused_at.pop(next_key)
            paused_for = max(float(now) - paused_at, 0.0)
            samples = self._rate_samples.get(next_key)
            if samples and paused_for > 0:
                self._rate_samples[next_key] = deque(
                    (
                        (sample_time + paused_for, current)
                        for sample_time, current in samples
                    ),
                    maxlen=samples.maxlen,
                )
        self._active_rate_key = next_key

    def _release_rate_task(self, task: ShaftProgressTaskSnapshot) -> None:
        key = (task.task_id, task.generation)
        self._rate_samples.pop(key, None)
        self._rate_paused_at.pop(key, None)
        if self._active_rate_key == key:
            self._active_rate_key = None

    def _draw(self, line: str) -> None:
        line = _encoding_safe_text(line, self._stream_encoding)
        line = _truncate_display(line, self.width)
        line_width = _display_width(line)
        padded_width = max(_display_width(self._last_line), line_width)
        self.stream.write("\r" + line + (" " * (padded_width - line_width)))
        self.stream.flush()
        self._last_line = line
        self._line_visible = True

    def _clear(self) -> None:
        if not self._line_visible:
            return
        self.stream.write("\r" + (" " * _display_width(self._last_line)) + "\r")
        self.stream.flush()
        self._last_line = ""
        self._line_visible = False

    def write_message(self, message: str) -> None:
        with self._lock:
            redraw = self._last_line if self._line_visible else ""
            self._clear()
            rendered = _encoding_safe_text(str(message), self._stream_encoding)
            self.stream.write(rendered + "\n")
            self.stream.flush()
            if redraw:
                self._draw(redraw)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                self._clear()
            except Exception:  # noqa: BLE001 - unregister a broken terminal below
                self._last_line = ""
                self._line_visible = False
            finally:
                self._closed = True
        _unregister_terminal_sink(self)


class ShaftPlainProgressSink:
    """Emit sparse progress lines for CI, redirected logs, and non-interactive subprocesses."""

    def __init__(
        self,
        *,
        writer: Callable[[str], None] | None = None,
        stream: TextIO | None = None,
        log_interval: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.writer = writer or (lambda message: progress_safe_write(message, stream=stream))
        self.log_interval = max(float(log_interval), 0.0)
        self.clock = clock
        self._last_logged_at: dict[str, float] = {}

    def publish(
        self,
        snapshot: ShaftProgressSnapshot,
        event: ShaftProgressEvent,
    ) -> None:
        task = snapshot.tasks[event.task_id]
        now = self.clock()
        force = event.kind in _LIFECYCLE_EVENTS
        last = self._last_logged_at.get(task.task_id, float("-inf"))
        if not force and now - last < self.log_interval:
            return
        self._last_logged_at[task.task_id] = now
        progress = _compact_number(task.current)
        if task.total is not None:
            progress = f"{progress}/{_compact_number(task.total)}"
        metrics = " ".join(
            _format_metric(key, value) for key, value in _ordered_metrics(task.metrics)[:2]
        )
        line = (
            f"progress {_single_line_text(task.label)} {event.kind} {progress} "
            f"{_single_line_text(task.unit)}"
        )
        if metrics:
            line = f"{line} {metrics}"
        if task.message:
            line = f"{line} message={_single_line_text(task.message)}"
        self.writer(line)

    def close(self) -> None:
        return None


class ShaftJsonProgressSink:
    """Persist the latest progress tree as one atomically replaced JSON snapshot."""

    def __init__(
        self,
        path: str | Path,
        *,
        persist_interval: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.path = Path(path)
        self.persist_interval = max(float(persist_interval), 0.0)
        self.clock = clock
        self._last_written_at = float("-inf")
        self._latest: ShaftProgressSnapshot | None = None
        self._latest_revision = 0
        self._lock = threading.RLock()

    def publish(
        self,
        snapshot: ShaftProgressSnapshot,
        event: ShaftProgressEvent,
    ) -> None:
        with self._lock:
            if event.revision and event.revision < self._latest_revision:
                return
            self._latest = snapshot
            self._latest_revision = max(self._latest_revision, event.revision)
            now = self.clock()
            force = event.kind in _LIFECYCLE_EVENTS
            if not force and now - self._last_written_at < self.persist_interval:
                return
            self._write(snapshot)
            self._last_written_at = now

    def _write(self, snapshot: ShaftProgressSnapshot) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(
                json.dumps(
                    snapshot.to_dict(),
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                ),
                encoding="utf-8",
            )
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def close(self) -> None:
        with self._lock:
            if self._latest is not None:
                self._write(self._latest)


_TERMINAL_SINK_LOCK = threading.RLock()
_ACTIVE_TERMINAL_SINK: ShaftTerminalProgressSink | None = None


def _register_terminal_sink(sink: ShaftTerminalProgressSink) -> None:
    global _ACTIVE_TERMINAL_SINK
    with _TERMINAL_SINK_LOCK:
        _ACTIVE_TERMINAL_SINK = sink


def _unregister_terminal_sink(sink: ShaftTerminalProgressSink) -> None:
    global _ACTIVE_TERMINAL_SINK
    with _TERMINAL_SINK_LOCK:
        if _ACTIVE_TERMINAL_SINK is sink:
            _ACTIVE_TERMINAL_SINK = None


def progress_safe_write(message: str, *, stream: TextIO | None = None) -> None:
    with _TERMINAL_SINK_LOCK:
        sink = _ACTIVE_TERMINAL_SINK
    if sink is not None:
        try:
            sink.write_message(str(message))
            return
        except Exception:  # noqa: BLE001 - terminal loss must not break the caller
            sink.close()
    target = stream or sys.stderr
    rendered = _encoding_safe_text(str(message), getattr(target, "encoding", None))
    target.write(rendered + "\n")
    target.flush()


def resolve_run_id(config: Any) -> str:
    configured = str(config.experiment.run_id or "").strip()
    fallback = str(config.experiment.name or "").strip()
    return configured or fallback or "shaft"


def build_progress_manager(
    config: Any,
    *,
    stream: TextIO | None = None,
) -> ShaftProgressManager:
    progress = config.progress
    run_id = resolve_run_id(config)
    sinks: list[ShaftProgressSink] = []
    if not bool(progress.enabled) or not is_rank_zero():
        return ShaftProgressManager(run_id=run_id)

    stream = stream or sys.stderr
    display = str(progress.display).strip().lower()
    if display == "auto":
        display = "interactive" if bool(stream.isatty()) else "plain"
    if display == "interactive":
        sinks.append(
            ShaftTerminalProgressSink(
                stream=stream,
                width=progress.width,
                refresh_interval=progress.refresh_interval,
                leave_completed=progress.leave_completed,
            )
        )
    elif display == "plain":
        sinks.append(
            ShaftPlainProgressSink(
                stream=stream,
                log_interval=progress.log_interval,
            )
        )
    elif display != "off":
        raise ValueError(f"Unsupported progress display mode: {display!r}.")

    if bool(progress.persist):
        sinks.append(
            ShaftJsonProgressSink(
                Path(config.experiment.output_dir) / PROGRESS_SNAPSHOT_FILENAME,
            )
        )
    return ShaftProgressManager(run_id=run_id, sinks=sinks)
