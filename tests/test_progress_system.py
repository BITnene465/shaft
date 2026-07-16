from __future__ import annotations

import errno
import io
import json
import logging
import os
from pathlib import Path
import re
import threading
import time

from huggingface_hub import logging as hub_logging
from huggingface_hub import utils as hf_hub_utils
import pytest
from transformers.utils import logging as hf_logging

from shaft.config import LoggingConfig
from shaft.observability import configure_logging
from shaft.observability.progress import (
    PROGRESS_SNAPSHOT_FILENAME,
    ShaftJsonProgressSink,
    ShaftPlainProgressSink,
    ShaftProgressEvent,
    ShaftProgressManager,
    ShaftTerminalProgressSink,
    _display_width,
    _truncate_display,
    progress_safe_write,
)


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += float(seconds)


class _TTY(io.StringIO):
    def isatty(self) -> bool:
        return True


class _AsciiTTY(_TTY):
    @property
    def encoding(self) -> str:
        return "ascii"

    def write(self, value: str) -> int:
        value.encode("ascii")
        return super().write(value)


class _FailingTTY(_TTY):
    def __init__(self) -> None:
        super().__init__()
        self.fail_writes = False

    def write(self, value: str) -> int:
        if self.fail_writes:
            raise OSError("terminal disconnected")
        return super().write(value)


class _RecordingSink:
    def __init__(self) -> None:
        self.records = []
        self.closed = False

    def publish(self, snapshot, event) -> None:  # noqa: ANN001
        self.records.append((snapshot, event))

    def close(self) -> None:
        self.closed = True


class _BrokenSink:
    def publish(self, snapshot, event) -> None:  # noqa: ANN001
        _ = snapshot, event
        raise OSError("broken progress destination")

    def close(self) -> None:
        return None


class _BrokenCloseSink:
    def publish(self, snapshot, event) -> None:  # noqa: ANN001
        _ = snapshot, event

    def close(self) -> None:
        raise OSError("cannot close progress destination")


def test_progress_manager_keeps_one_task_state_truth_across_nested_phases() -> None:
    sink = _RecordingSink()
    manager = ShaftProgressManager(
        run_id="run-1",
        attempt_id="attempt-1",
        sinks=[sink],
    )

    train = manager.start_task(
        "train",
        label="train",
        total=100,
        initial=10,
        unit="step",
        summary_on_complete=True,
    )
    train.update(current=11, metrics={"loss": 1.25, "lr": 2e-5})
    evaluation = manager.start_task(
        "eval.loss",
        label="eval",
        parent_task_id="train",
        total=2,
        unit="batch",
    )
    evaluation.advance(1)

    assert manager.snapshot.active_task_id == "eval.loss"
    evaluation.complete()
    assert manager.snapshot.active_task_id == "train"

    train.update(current=12)
    train.complete(message="training complete")
    snapshot = manager.snapshot

    assert snapshot.status == "succeeded"
    assert snapshot.active_task_id is None
    assert snapshot.tasks["train"].current == 12
    assert snapshot.tasks["train"].metrics == {"loss": 1.25, "lr": 2e-5}
    assert snapshot.tasks["eval.loss"].state == "succeeded"
    manager.close()
    assert sink.closed is True


def test_progress_snapshots_are_read_only_for_all_sinks() -> None:
    manager = ShaftProgressManager(run_id="run", sinks=[_RecordingSink()])
    task = manager.start_task(
        "train",
        label="train",
        total=2,
        metrics={"loss": 1.0},
    )
    snapshot = manager.snapshot

    with pytest.raises(TypeError):
        snapshot.tasks["other"] = snapshot.tasks["train"]  # type: ignore[index]
    with pytest.raises(TypeError):
        snapshot.tasks["train"].metrics["loss"] = 0.0  # type: ignore[index]

    assert manager.snapshot.tasks["train"].metrics == {"loss": 1.0}
    task.complete()


def test_terminal_sink_uses_one_short_physical_line_for_nested_tasks() -> None:
    clock = _Clock()
    stream = _TTY()
    sink = ShaftTerminalProgressSink(
        stream=stream,
        width=72,
        refresh_interval=0.0,
        leave_completed=False,
        clock=clock,
    )
    manager = ShaftProgressManager(run_id="run", sinks=[sink])
    train = manager.start_task(
        "train",
        label="train",
        total=100,
        unit="step",
        summary_on_complete=True,
    )
    clock.advance(1)
    train.update(current=20, metrics={"loss": 1.2345, "lr": 2e-5})
    evaluation = manager.start_task(
        "eval.loss",
        label="eval",
        parent_task_id="train",
        total=2,
        unit="batch",
    )
    clock.advance(1)
    evaluation.advance(1)
    evaluation.complete()
    train.complete()
    manager.close()

    output = stream.getvalue()
    assert output.count("\n") == 1
    assert "train" in output
    assert "eval" in output
    assert any("eval" in segment and "eta" in segment for segment in output.split("\r"))
    assert "loss 1.23" in output
    assert "lr 2e-5" in output
    assert output.rfind("loss 1.23") < output.rfind("lr 2e-5")
    assert "\x1b[" not in output
    assert all(len(segment.rstrip()) <= 72 for segment in output.split("\r"))


def test_terminal_long_run_has_visible_early_progress_and_step_time() -> None:
    clock = _Clock()
    stream = _TTY()
    sink = ShaftTerminalProgressSink(
        stream=stream,
        width=72,
        refresh_interval=0.0,
        clock=clock,
    )
    manager = ShaftProgressManager(run_id="run", sinks=[sink])
    train = manager.start_task(
        "train",
        label="train",
        total=10_000,
        unit="step",
        display_rate=True,
    )

    clock.advance(163.5)
    train.update(
        current=25,
        metrics={"loss": 7.9, "lr": "2.5–5e-7"},
    )

    line = stream.getvalue().split("\r")[-1].rstrip()
    assert "╸" in line and "─" in line
    assert "▏" not in line and "·" not in line
    assert "25/10k" in line
    assert "0.25%" in line
    assert "6.54s/it" in line
    assert "eta " in line
    assert "loss 7.9" in line
    assert "lr 2.5–5e-7" in line
    assert len(line) <= 72
    train.complete()
    manager.close()


def test_terminal_narrow_layout_keeps_progress_and_speed_before_optional_fields() -> None:
    clock = _Clock()
    stream = _TTY()
    sink = ShaftTerminalProgressSink(
        stream=stream,
        width=40,
        refresh_interval=0.0,
        clock=clock,
    )
    manager = ShaftProgressManager(run_id="run", sinks=[sink])
    train = manager.start_task(
        "train",
        label="train",
        total=10_000,
        unit="step",
        display_rate=True,
    )

    clock.advance(163.5)
    train.update(current=25, metrics={"loss": 7.9, "lr": "2.5–5e-7"})

    line = stream.getvalue().split("\r")[-1].rstrip()
    assert "25/10k" in line
    assert "0.25%" in line
    assert "6.54s/it" in line
    assert len(line) <= 40
    train.complete()
    manager.close()


def test_terminal_ascii_stream_uses_a_readable_bar_fallback() -> None:
    stream = _AsciiTTY()
    sink = ShaftTerminalProgressSink(
        stream=stream,
        width=72,
        refresh_interval=0.0,
    )
    manager = ShaftProgressManager(run_id="run", sinks=[sink])
    train = manager.start_task("train", label="train", total=10_000, unit="step")

    train.update(current=25, metrics={"lr": "2.5–5e-7", "note": "训练"})

    line = stream.getvalue().split("\r")[-1]
    assert ">-------" in line
    assert "╸" not in line and "─" not in line
    assert "lr 2.5-5e-7" in line
    line.encode("ascii")
    train.complete()
    manager.close()


def test_terminal_never_reports_one_hundred_percent_before_completion() -> None:
    stream = _TTY()
    sink = ShaftTerminalProgressSink(
        stream=stream,
        width=72,
        refresh_interval=0.0,
    )
    manager = ShaftProgressManager(run_id="run", sinks=[sink])
    task = manager.start_task("train", label="train", total=10_000, unit="step")

    task.update(current=9_999)
    incomplete = stream.getvalue().split("\r")[-1]
    assert "99.9%" in incomplete
    assert "100%" not in incomplete
    assert "10k/10k" not in incomplete

    task.update(current=10_000)
    complete = stream.getvalue().split("\r")[-1]
    assert "100%" in complete
    task.complete()
    manager.close()


def test_terminal_fast_steps_and_subsecond_eta_never_render_fake_zero() -> None:
    clock = _Clock()
    stream = _TTY()
    sink = ShaftTerminalProgressSink(
        stream=stream,
        width=72,
        refresh_interval=0.0,
        clock=clock,
    )
    manager = ShaftProgressManager(run_id="run", sinks=[sink])
    task = manager.start_task(
        "train",
        label="train",
        total=101,
        unit="step",
        display_rate=True,
    )

    clock.advance(0.5)
    task.update(current=100)

    line = stream.getvalue().split("\r")[-1]
    assert "200it/s" in line
    assert "eta <1s" in line
    assert "0s/it" not in line
    task.complete()
    manager.close()


def test_terminal_extreme_horizon_and_slow_step_never_round_to_fake_zero() -> None:
    clock = _Clock()
    stream = _TTY()
    sink = ShaftTerminalProgressSink(
        stream=stream,
        width=72,
        refresh_interval=0.0,
        clock=clock,
    )
    manager = ShaftProgressManager(run_id="run", sinks=[sink])
    train = manager.start_task(
        "train",
        label="train",
        total=1_000_000,
        unit="step",
        display_rate=True,
    )

    clock.advance(100)
    train.update(current=1)

    line = stream.getvalue().split("\r")[-1]
    assert "<0.01%" in line
    assert "100s/it" in line
    assert "1s/it" not in line
    train.complete()
    manager.close()


def test_leave_completed_prints_nested_phase_then_restores_parent() -> None:
    stream = _TTY()
    sink = ShaftTerminalProgressSink(
        stream=stream,
        width=72,
        refresh_interval=0.0,
        leave_completed=True,
    )
    manager = ShaftProgressManager(run_id="run", sinks=[sink])
    train = manager.start_task("train", label="train", total=10, unit="step")
    evaluation = manager.start_task(
        "eval.loss",
        label="eval",
        total=1,
        unit="batch",
        parent_task_id="train",
    )
    evaluation.update(current=1)
    evaluation.complete()

    output = stream.getvalue()
    assert output.count("\n") == 1
    assert output.split("\n", maxsplit=1)[1].startswith("\rtrain")
    train.complete()
    manager.close()


def test_nested_eval_time_is_excluded_from_resumed_train_step_rate() -> None:
    clock = _Clock()
    stream = _TTY()
    sink = ShaftTerminalProgressSink(
        stream=stream,
        width=72,
        refresh_interval=0.0,
        clock=clock,
    )
    manager = ShaftProgressManager(run_id="run", sinks=[sink])
    train = manager.start_task(
        "train",
        label="train",
        total=10,
        unit="step",
        display_rate=True,
    )
    clock.advance(10)
    train.update(current=1)
    evaluation = manager.start_task(
        "eval.loss",
        label="eval",
        total=1,
        unit="batch",
        parent_task_id="train",
        display_rate=True,
    )
    clock.advance(100)
    evaluation.update(current=1)
    evaluation.complete()
    clock.advance(10)
    train.update(current=2)

    line = stream.getvalue().split("\r")[-1]
    assert "10s/it" in line
    assert "60s/it" not in line
    train.complete()
    manager.close()


def test_terminal_releases_finished_task_rate_history() -> None:
    clock = _Clock()
    stream = _TTY()
    sink = ShaftTerminalProgressSink(
        stream=stream,
        width=72,
        refresh_interval=0.0,
        clock=clock,
    )
    manager = ShaftProgressManager(run_id="run", sinks=[sink])
    task = manager.start_task(
        "eval.loss",
        label="eval",
        total=1,
        unit="batch",
        display_rate=True,
    )
    clock.advance(1)
    task.update(current=1)
    task.complete()

    assert sink._rate_samples == {}
    assert sink._rate_paused_at == {}
    assert sink._active_rate_key is None
    manager.close()


def test_progress_safe_write_preserves_the_active_terminal_line() -> None:
    stream = _TTY()
    sink = ShaftTerminalProgressSink(
        stream=stream,
        width=72,
        refresh_interval=0.0,
    )
    manager = ShaftProgressManager(run_id="run", sinks=[sink])
    task = manager.start_task("train", label="train", total=10, unit="step")
    task.update(current=2)

    progress_safe_write("worker ready", stream=stream)
    output = stream.getvalue()

    assert output.count("worker ready\n") == 1
    assert output.split("worker ready\n", maxsplit=1)[1].startswith("\rtrain")
    task.complete()
    manager.close()


def test_progress_safe_write_routes_by_stream_and_restores_previous_sink() -> None:
    first_stream = _TTY()
    second_stream = _TTY()
    first_sink = ShaftTerminalProgressSink(stream=first_stream, refresh_interval=0.0)
    second_sink = ShaftTerminalProgressSink(stream=second_stream, refresh_interval=0.0)
    first_manager = ShaftProgressManager(run_id="first", sinks=[first_sink])
    second_manager = ShaftProgressManager(run_id="second", sinks=[second_sink])
    first_task = first_manager.start_task("first", label="first", total=2)
    second_task = second_manager.start_task("second", label="second", total=2)

    progress_safe_write("first log", stream=first_stream)
    progress_safe_write("second log", stream=second_stream)
    assert first_stream.getvalue().count("first log\n") == 1
    assert "second log" not in first_stream.getvalue()
    assert second_stream.getvalue().count("second log\n") == 1
    assert "first log" not in second_stream.getvalue()

    second_task.complete()
    second_manager.close()
    progress_safe_write("restored log")
    assert first_stream.getvalue().count("restored log\n") == 1

    first_task.complete()
    first_manager.close()


def test_hf_logs_are_quiet_at_info_but_debug_remains_available() -> None:
    stream = _TTY()
    sink = ShaftTerminalProgressSink(
        stream=stream,
        width=72,
        refresh_interval=0.0,
    )
    manager = ShaftProgressManager(run_id="run", sinks=[sink])
    task = manager.start_task("train", label="train", total=10, unit="step")
    root = logging.getLogger()
    previous_handlers = list(root.handlers)
    previous_level = root.level
    previous_hf_verbosity = hf_logging.get_verbosity()
    previous_hf_progress = hf_logging.is_progress_bar_enabled()
    previous_hub_progress = hf_hub_utils.are_progress_bars_disabled()
    hub_root = logging.getLogger("huggingface_hub")
    previous_hub_handlers = list(hub_root.handlers)
    previous_hub_level = hub_root.level
    previous_hub_propagate = hub_root.propagate
    try:
        configure_logging(LoggingConfig(level="INFO"), run_id="run")
        assert hf_logging.is_progress_bar_enabled() is False
        assert hf_hub_utils.are_progress_bars_disabled() is True
        hf_logger = hf_logging.get_logger("transformers.progress_test")
        hub_logger = hub_logging.get_logger("huggingface_hub.progress_test")
        hf_logger.info("hf detail")
        hf_logger.warning("hf warning")
        hub_logger.info("hub detail")
        hub_logger.warning("hub warning")

        output = stream.getvalue()
        assert "hf detail" not in output
        assert "hub detail" not in output
        assert output.count("hf warning\n") == 1
        assert output.split("hf warning\n", maxsplit=1)[1].startswith("\rtrain")
        assert output.count("hub warning\n") == 1
        assert output.split("hub warning\n", maxsplit=1)[1].startswith("\rtrain")

        configure_logging(LoggingConfig(level="DEBUG"), run_id="run")
        hf_logger.info("hf debug detail")
        hub_logger.info("hub debug detail")
        assert stream.getvalue().count("hf debug detail\n") == 1
        assert stream.getvalue().count("hub debug detail\n") == 1
    finally:
        task.complete()
        manager.close()
        root.handlers[:] = previous_handlers
        root.setLevel(previous_level)
        hf_logging.disable_propagation()
        hf_logging.enable_default_handler()
        hf_logging.set_verbosity(previous_hf_verbosity)
        hub_root.handlers[:] = previous_hub_handlers
        hub_root.setLevel(previous_hub_level)
        hub_root.propagate = previous_hub_propagate
        if previous_hf_progress:
            hf_logging.enable_progress_bar()
        else:
            hf_logging.disable_progress_bar()
        if previous_hub_progress:
            hf_hub_utils.disable_progress_bars()
        else:
            hf_hub_utils.enable_progress_bars()


def test_plain_sink_throttles_updates_but_always_reports_lifecycle() -> None:
    clock = _Clock()
    lines: list[str] = []
    sink = ShaftPlainProgressSink(
        writer=lines.append,
        log_interval=30.0,
        clock=clock,
    )
    manager = ShaftProgressManager(run_id="run", sinks=[sink])
    task = manager.start_task("train", label="train", total=100, unit="step")
    for current in range(1, 20):
        task.update(current=current)
    assert len(lines) == 1

    clock.advance(30)
    task.update(current=20, metrics={"loss": 1.0})
    task.complete()

    assert len(lines) == 3
    assert "started" in lines[0]
    assert "20/100" in lines[1]
    assert "succeeded" in lines[2]


def test_plain_sink_is_not_suppressed_by_application_log_level() -> None:
    stream = io.StringIO()
    sink = ShaftPlainProgressSink(
        stream=stream,
        log_interval=30.0,
    )
    manager = ShaftProgressManager(run_id="run", sinks=[sink])

    task = manager.start_task("train", label="train", total=2, unit="step")
    task.fail("failed even when logging is disabled")

    output = stream.getvalue()
    assert "progress train started 0/2 step" in output
    assert "progress train failed 0/2 step" in output
    assert "\r" not in output and "\x1b[" not in output


def test_json_sink_persists_latest_snapshot_atomically(tmp_path: Path) -> None:
    clock = _Clock()
    target = tmp_path / PROGRESS_SNAPSHOT_FILENAME
    sink = ShaftJsonProgressSink(
        target,
        persist_interval=60.0,
        clock=clock,
    )
    manager = ShaftProgressManager(
        run_id="run-json",
        attempt_id="attempt-json",
        sinks=[sink],
    )
    task = manager.start_task("train", label="train", total=4, unit="step")
    task.update(current=3, metrics={"loss": 0.75})
    task.complete()
    manager.close()

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["run_id"] == "run-json"
    assert payload["attempt_id"] == "attempt-json"
    assert payload["status"] == "succeeded"
    assert payload["active_task_id"] is None
    assert payload["tasks"]["train"]["current"] == 3
    assert payload["tasks"]["train"]["metrics"] == {"loss": 0.75}
    assert not list(tmp_path.glob(".*.tmp"))


def test_progress_sink_failure_is_isolated_from_training_state() -> None:
    healthy = _RecordingSink()
    manager = ShaftProgressManager(
        run_id="run",
        sinks=[_BrokenSink(), healthy],
    )

    task = manager.start_task("train", label="train", total=2)
    task.advance()
    task.complete()

    assert manager.snapshot.tasks["train"].state == "succeeded"
    assert [event.kind for _, event in healthy.records] == [
        "started",
        "updated",
        "succeeded",
    ]

    close_manager = ShaftProgressManager(
        run_id="close-run",
        sinks=[_BrokenCloseSink(), healthy],
    )
    close_manager.start_task("phase", label="phase").complete()
    close_manager.close()
    assert healthy.closed is True


def test_broken_terminal_is_unregistered_before_logging_falls_back() -> None:
    terminal_stream = _FailingTTY()
    sink = ShaftTerminalProgressSink(
        stream=terminal_stream,
        width=72,
        refresh_interval=0.0,
    )
    manager = ShaftProgressManager(run_id="run", sinks=[sink])
    task = manager.start_task("train", label="train", total=2)
    terminal_stream.fail_writes = True

    task.update(current=1)

    assert manager.snapshot.tasks["train"].current == 1
    fallback = io.StringIO()
    progress_safe_write("terminal lost", stream=fallback)
    assert fallback.getvalue() == "terminal lost\n"
    task.complete()
    manager.close()


def test_progress_safe_write_falls_back_when_active_terminal_breaks_first() -> None:
    terminal_stream = _FailingTTY()
    sink = ShaftTerminalProgressSink(
        stream=terminal_stream,
        width=72,
        refresh_interval=0.0,
    )
    manager = ShaftProgressManager(run_id="run", sinks=[sink])
    task = manager.start_task("train", label="train", total=2)
    terminal_stream.fail_writes = True
    fallback = io.StringIO()

    progress_safe_write("terminal lost", stream=fallback)

    assert fallback.getvalue() == "terminal lost\n"
    task.complete()
    manager.close()


def test_record_failure_marks_active_work_or_run_as_failed() -> None:
    manager = ShaftProgressManager(run_id="run", sinks=[_RecordingSink()])
    train = manager.start_task("train", label="train", total=2)
    manager.record_failure("CUDA out of memory")

    assert manager.snapshot.status == "failed"
    assert manager.snapshot.tasks["train"].state == "failed"
    assert manager.snapshot.tasks["train"].message == "CUDA out of memory"
    train.fail("ignored stale completion")

    idle_manager = ShaftProgressManager(run_id="idle", sinks=[_RecordingSink()])
    idle_manager.record_failure("startup failed")
    assert idle_manager.snapshot.status == "failed"
    assert idle_manager.snapshot.tasks["run.failure"].state == "failed"


def test_terminal_nested_failure_is_visible_and_remains_the_final_status() -> None:
    stream = _TTY()
    sink = ShaftTerminalProgressSink(
        stream=stream,
        width=72,
        refresh_interval=0.0,
    )
    manager = ShaftProgressManager(run_id="run", sinks=[sink])
    train = manager.start_task(
        "train",
        label="train",
        total=10,
        unit="step",
        summary_on_complete=True,
    )
    evaluation = manager.start_task(
        "eval.loss",
        label="eval",
        total=2,
        unit="batch",
        parent_task_id="train",
    )

    evaluation.fail("validation crashed")
    train.update(current=10)
    train.complete(message="training loop returned")

    output = stream.getvalue()
    assert "eval × failed 0/2 validation crashed\n" in output
    final_line = output.rstrip("\n").split("\n")[-1]
    assert "eval × failed 0/2 validation crashed" in final_line
    assert manager.snapshot.status == "failed"
    manager.close()


def test_progress_advance_is_atomic_across_threads() -> None:
    manager = ShaftProgressManager(run_id="run", sinks=[_RecordingSink()])
    task = manager.start_task("work", label="work", total=8_000)
    barrier = threading.Barrier(8)

    def _advance_many() -> None:
        barrier.wait()
        for _ in range(1_000):
            task.advance()

    threads = [threading.Thread(target=_advance_many) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert all(not thread.is_alive() for thread in threads)
    assert manager.snapshot.tasks["work"].current == 8_000
    task.complete()
    manager.close()


def test_progress_manager_publishes_concurrent_updates_in_commit_order() -> None:
    class _BlockingSink:
        def __init__(self) -> None:
            self.entered = threading.Event()
            self.release = threading.Event()
            self.currents: list[int] = []

        def publish(self, snapshot, event: ShaftProgressEvent) -> None:  # noqa: ANN001
            current = snapshot.tasks[event.task_id].current
            if event.kind == "updated" and current == 1:
                self.entered.set()
                assert self.release.wait(timeout=5)
            if event.kind == "updated":
                self.currents.append(current)

        def close(self) -> None:
            return None

    sink = _BlockingSink()
    manager = ShaftProgressManager(run_id="run", sinks=[sink])
    task = manager.start_task("train", label="train", total=2)
    errors: list[BaseException] = []

    def _update(current: int) -> None:
        try:
            task.update(current=current)
        except BaseException as exc:  # noqa: BLE001 - asserted below
            errors.append(exc)

    first = threading.Thread(target=_update, args=(1,))
    first.start()
    assert sink.entered.wait(timeout=5)
    second = threading.Thread(target=_update, args=(2,))
    second.start()
    deadline = time.monotonic() + 5
    while manager.snapshot.tasks["train"].current != 2 and time.monotonic() < deadline:
        time.sleep(0.001)
    assert manager.snapshot.tasks["train"].current == 2
    sink.release.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert errors == []
    assert sink.currents == [1, 2]
    task.complete()
    manager.close()


def test_progress_close_prevents_a_new_task_during_terminal_publication() -> None:
    class _BlockingCancelSink:
        def __init__(self) -> None:
            self.entered = threading.Event()
            self.release = threading.Event()

        def publish(self, snapshot, event: ShaftProgressEvent) -> None:  # noqa: ANN001
            _ = snapshot
            if event.kind == "cancelled":
                self.entered.set()
                assert self.release.wait(timeout=5)

        def close(self) -> None:
            return None

    sink = _BlockingCancelSink()
    manager = ShaftProgressManager(run_id="run", sinks=[sink])
    manager.start_task("train", label="train", total=2)
    close_thread = threading.Thread(target=manager.close)
    close_thread.start()
    assert sink.entered.wait(timeout=5)

    with pytest.raises(RuntimeError, match="closed"):
        manager.start_task("late", label="late")

    sink.release.set()
    close_thread.join(timeout=5)
    assert not close_thread.is_alive()
    assert all(task.state != "running" for task in manager.snapshot.tasks.values())


def test_terminal_failure_leaves_one_bounded_summary_line() -> None:
    stream = _TTY()
    sink = ShaftTerminalProgressSink(
        stream=stream,
        width=40,
        refresh_interval=0.0,
    )
    manager = ShaftProgressManager(run_id="run", sinks=[sink])
    train = manager.start_task("train", label="train", total=10, unit="step")
    train.update(current=4)

    manager.record_failure("CUDA out of memory")

    output = stream.getvalue()
    assert output.count("\n") == 1
    assert "train × failed 4/10 CUDA out of memory" in output
    assert all(len(segment.rstrip()) <= 40 for segment in output.split("\r"))
    manager.close()


def test_terminal_width_counts_cjk_as_two_display_cells() -> None:
    stream = _TTY()
    sink = ShaftTerminalProgressSink(
        stream=stream,
        width=40,
        refresh_interval=0.0,
    )
    manager = ShaftProgressManager(run_id="run", sinks=[sink])
    task = manager.start_task("train", label="训练", total=10, unit="step")
    task.update(current=4)
    task.fail("显存不足" * 20)

    for segment in stream.getvalue().split("\r"):
        display_cells = sum(
            2 if "\u4e00" <= character <= "\u9fff" else 1 for character in segment.rstrip()
        )
        assert display_cells <= 40
    manager.close()


def test_progress_sinks_sanitize_multiline_and_terminal_control_text() -> None:
    stream = _TTY()
    terminal = ShaftTerminalProgressSink(
        stream=stream,
        width=72,
        refresh_interval=0.0,
    )
    terminal_manager = ShaftProgressManager(run_id="run", sinks=[terminal])
    task = terminal_manager.start_task(
        "unsafe",
        label="bad\n\x1b[31mlabel",
        message="line1\r\nline2\x1b[0m",
        metrics={"note\nkey": "value\r\x1b[32mgreen"},
        summary_on_complete=True,
    )
    task.complete()
    terminal_manager.close()

    output = stream.getvalue()
    assert output.count("\n") == 1
    assert "\x1b[" not in output
    assert "bad lab" in output

    lines: list[str] = []
    plain_manager = ShaftProgressManager(
        run_id="plain",
        sinks=[ShaftPlainProgressSink(writer=lines.append, log_interval=0.0)],
    )
    plain = plain_manager.start_task(
        "unsafe",
        label="bad\n\x1b[31mlabel",
        message="line1\r\nline2\x1b[0m",
        metrics={"note\nkey": "value\r\x1b[32mgreen"},
    )
    plain.complete()

    assert lines
    assert all("\n" not in line and "\r" not in line and "\x1b" not in line for line in lines)


def test_terminal_cost_task_displays_rate_and_eta() -> None:
    clock = _Clock()
    stream = _TTY()
    sink = ShaftTerminalProgressSink(
        stream=stream,
        width=72,
        refresh_interval=0.0,
        clock=clock,
    )
    manager = ShaftProgressManager(run_id="run", sinks=[sink])
    task = manager.start_task(
        "startup.cost_estimate",
        label="cost",
        total=20,
        unit="sample",
        display_rate=True,
    )
    clock.advance(2)
    task.update(current=10)

    assert "5 sample/s" in stream.getvalue()
    assert "eta 2s" in stream.getvalue()
    task.complete()
    manager.close()


def test_terminal_indeterminate_task_uses_spinner_and_useful_rate() -> None:
    clock = _Clock()
    stream = _TTY()
    sink = ShaftTerminalProgressSink(
        stream=stream,
        width=72,
        refresh_interval=0.0,
        clock=clock,
    )
    manager = ShaftProgressManager(run_id="run", sinks=[sink])
    task = manager.start_task(
        "download",
        label="download",
        unit="file",
        display_rate=True,
    )
    clock.advance(2)
    task.update(current=10, message="model shards")

    line = stream.getvalue().split("\r")[-1].rstrip()
    assert any(spinner in line for spinner in "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
    assert "10 file" in line
    assert "5 file/s" in line
    assert "model shards" in line
    task.complete()
    manager.close()


def test_terminal_indeterminate_spinner_animates_without_manager_updates() -> None:
    stream = _TTY()
    sink = ShaftTerminalProgressSink(
        stream=stream,
        width=72,
        refresh_interval=0.0,
    )
    manager = ShaftProgressManager(run_id="run", sinks=[sink])
    manager.start_task("download", label="download", message="model shards")

    deadline = time.monotonic() + 1.0
    frames: set[str] = set()
    while time.monotonic() < deadline:
        frames.update(character for character in stream.getvalue() if character in "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
        if len(frames) >= 2:
            break
        time.sleep(0.02)

    assert len(frames) >= 2
    manager.close()
    closed_output = stream.getvalue()
    time.sleep(0.12)
    assert stream.getvalue() == closed_output


def test_terminal_spinner_failure_unregisters_without_thread_exception() -> None:
    stream = _FailingTTY()
    sink = ShaftTerminalProgressSink(stream=stream, width=72, refresh_interval=0.0)
    manager = ShaftProgressManager(run_id="run", sinks=[sink])
    task = manager.start_task("download", label="download", message="model shards")
    stream.fail_writes = True

    deadline = time.monotonic() + 1.0
    while not sink._closed and time.monotonic() < deadline:
        time.sleep(0.02)

    assert sink._closed is True
    task.complete()
    manager.close()


def test_terminal_color_is_scoped_to_interactive_presentation() -> None:
    stream = _TTY()
    sink = ShaftTerminalProgressSink(
        stream=stream,
        width=72,
        refresh_interval=0.0,
        color=True,
    )
    manager = ShaftProgressManager(run_id="run", sinks=[sink])
    task = manager.start_task("train", label="train", total=10, unit="step")
    task.update(current=4, metrics={"loss": 1.25})

    line = stream.getvalue().split("\r")[-1].rstrip()
    assert "\x1b[1;36mtrain\x1b[0m" in line
    assert "\x1b[36m" in line
    plain = re.sub(r"\x1b\[[0-9;]*m", "", line)
    assert "train 40%" in plain
    assert len(plain) <= 72
    task.complete()
    manager.close()


def test_terminal_honors_no_color_even_when_color_is_forced_by_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.setenv("NO_COLOR", "")
    stream = _TTY()
    sink = ShaftTerminalProgressSink(stream=stream, refresh_interval=0.0)
    manager = ShaftProgressManager(run_id="run", sinks=[sink])
    task = manager.start_task("train", label="train", total=2)
    task.update(current=1)

    assert "\x1b[" not in stream.getvalue()
    task.complete()
    manager.close()


def _read_pty(master_fd: int) -> str:
    chunks: list[bytes] = []
    os.set_blocking(master_fd, False)
    while True:
        try:
            chunk = os.read(master_fd, 65_536)
        except BlockingIOError:
            break
        except OSError as exc:
            if exc.errno == errno.EIO:  # Linux PTY signals that the slave closed.
                break
            raise
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks).decode("utf-8", errors="replace")


def test_real_tty_uses_erase_line_and_color_but_no_color_keeps_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("CLICOLOR", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)

    master_fd, slave_fd = os.openpty()
    stream = os.fdopen(slave_fd, "w", encoding="utf-8", buffering=1)
    try:
        sink = ShaftTerminalProgressSink(
            stream=stream,
            width=72,
            width_provider=lambda: 72,
            refresh_interval=0.0,
        )
        manager = ShaftProgressManager(run_id="run", sinks=[sink])
        task = manager.start_task("train", label="train", total=2)
        task.update(current=1)
        task.complete()
        manager.close()
        stream.close()
        colored = _read_pty(master_fd)
    finally:
        if not stream.closed:
            stream.close()
        os.close(master_fd)

    assert "\r\x1b[2K" in colored
    assert "\x1b[1;36mtrain\x1b[0m" in colored

    monkeypatch.setenv("NO_COLOR", "")
    master_fd, slave_fd = os.openpty()
    stream = os.fdopen(slave_fd, "w", encoding="utf-8", buffering=1)
    try:
        sink = ShaftTerminalProgressSink(
            stream=stream,
            width=72,
            width_provider=lambda: 72,
            refresh_interval=0.0,
        )
        manager = ShaftProgressManager(run_id="run", sinks=[sink])
        task = manager.start_task("train", label="train", total=2)
        task.update(current=1)
        task.complete()
        manager.close()
        stream.close()
        colorless = _read_pty(master_fd)
    finally:
        if not stream.closed:
            stream.close()
        os.close(master_fd)

    assert "\r\x1b[2K" in colorless
    assert "\x1b[1;36m" not in colorless
    assert "\x1b[36m" not in colorless


def test_terminal_layout_tracks_terminal_resize_without_losing_core_fields() -> None:
    clock = _Clock()
    columns = [96]
    stream = _TTY()
    sink = ShaftTerminalProgressSink(
        stream=stream,
        width=96,
        width_provider=lambda: columns[0],
        refresh_interval=0.0,
        clock=clock,
    )
    manager = ShaftProgressManager(run_id="run", sinks=[sink])
    task = manager.start_task(
        "train",
        label="train",
        total=10_000,
        unit="step",
        display_rate=True,
    )
    clock.advance(10)
    task.update(current=10, metrics={"loss": 1.5})
    wide = stream.getvalue().split("\r")[-1].rstrip()

    columns[0] = 72
    clock.advance(10)
    task.update(current=20, metrics={"loss": 1.4, "tok/s": 4_103.7})
    medium = stream.getvalue().split("\r")[-1].rstrip()

    columns[0] = 40
    clock.advance(10)
    task.update(current=30, metrics={"loss": 1.3, "tok/s": 4_203.7})
    narrow = stream.getvalue().split("\r")[-1].rstrip()

    assert len(wide) <= 96
    assert len(medium) <= 72
    assert len(narrow) <= 40
    assert "0.30%" in narrow
    assert "30/10k" in narrow
    assert "1s/it" in narrow
    assert "tok 4.1k/s" in medium
    assert "loss 1.5" in wide
    task.complete()
    manager.close()


@pytest.mark.parametrize("columns", [12, 15, 20])
def test_terminal_ultra_narrow_layout_never_splits_a_core_field(columns: int) -> None:
    stream = _TTY()
    sink = ShaftTerminalProgressSink(
        stream=stream,
        width=96,
        width_provider=lambda: columns,
        refresh_interval=0.0,
    )
    manager = ShaftProgressManager(run_id="run", sinks=[sink])
    task = manager.start_task(
        "train",
        label="训练任务",
        total=1_234_567,
        unit="step",
    )
    task.update(current=12_345)

    line = stream.getvalue().split("\r")[-1].rstrip()
    assert _display_width(line) <= columns
    assert "12.3k/1.23m" in line
    assert not line.endswith(("/", ".", "k/1", "m/1"))
    task.complete()
    manager.close()


@pytest.mark.parametrize("columns", [1, 2, 5, 8, 11])
def test_terminal_tiny_width_never_exceeds_physical_columns(columns: int) -> None:
    stream = _TTY()
    sink = ShaftTerminalProgressSink(
        stream=stream,
        width=96,
        width_provider=lambda: columns,
        refresh_interval=0.0,
    )
    manager = ShaftProgressManager(run_id="run", sinks=[sink])
    task = manager.start_task("train", label="训练任务", total=1_234_567, unit="step")
    task.update(current=12_345)

    for line in stream.getvalue().split("\r"):
        assert _display_width(line.rstrip()) <= columns
    task.complete()
    manager.close()


def test_log_redraw_reflows_after_terminal_resize() -> None:
    columns = [72]
    stream = _TTY()
    sink = ShaftTerminalProgressSink(
        stream=stream,
        width=96,
        width_provider=lambda: columns[0],
        refresh_interval=0.0,
    )
    manager = ShaftProgressManager(run_id="run", sinks=[sink])
    task = manager.start_task("train", label="train", total=10_000, unit="step")
    task.update(current=25, metrics={"loss": 7.9, "tok/s": 4_103.7})

    columns[0] = 20
    progress_safe_write("worker ready", stream=stream)
    redraw = stream.getvalue().split("worker ready\n", maxsplit=1)[1].split("\r")[-1].rstrip()

    assert _display_width(redraw) <= 20
    assert "25/10k" in redraw
    assert "tok " not in redraw
    task.complete()
    manager.close()


def test_terminal_width_and_truncation_preserve_grapheme_clusters() -> None:
    examples = ("👩‍💻", "🏳️‍🌈", "🇨🇳", "e\u0301")
    expected_widths = (2, 2, 2, 1)

    for value, expected_width in zip(examples, expected_widths, strict=True):
        assert _display_width(value) == expected_width
        assert _truncate_display(value + "x", expected_width) == value

    assert "\u200d" not in _truncate_display("👩‍💻", 1)


def test_terminal_bar_uses_the_last_cell_before_exact_completion() -> None:
    stream = _TTY()
    sink = ShaftTerminalProgressSink(stream=stream, width=40, refresh_interval=0.0)
    manager = ShaftProgressManager(run_id="run", sinks=[sink])
    task = manager.start_task("train", label="train", total=1_000, unit="step")
    task.update(current=999)

    line = stream.getvalue().split("\r")[-1]
    assert "━━━━━━━╸" in line
    assert "100%" not in line
    task.complete()
    manager.close()


def test_terminal_prioritizes_loss_and_token_throughput_over_lr() -> None:
    clock = _Clock()
    stream = _TTY()
    sink = ShaftTerminalProgressSink(
        stream=stream,
        width=72,
        refresh_interval=0.0,
        clock=clock,
    )
    manager = ShaftProgressManager(run_id="run", sinks=[sink])
    task = manager.start_task(
        "train",
        label="train",
        total=10_000,
        unit="step",
        display_rate=True,
    )
    clock.advance(163.5)
    task.update(
        current=25,
        metrics={"loss": 7.9, "tok/s": 4_103.7, "lr": "2.5–5e-7"},
    )

    line = stream.getvalue().split("\r")[-1].rstrip()
    assert "loss 7.9" in line
    assert "tok 4.1k/s" in line
    assert line.index("loss 7.9") < line.index("tok 4.1k/s")
    assert "lr " not in line
    task.complete()
    manager.close()


def test_terminal_does_not_backfill_lower_priority_metric_after_omission() -> None:
    stream = _TTY()
    sink = ShaftTerminalProgressSink(stream=stream, width=50, refresh_interval=0.0)
    manager = ShaftProgressManager(run_id="run", sinks=[sink])
    task = manager.start_task("train", label="train", total=10_000, unit="step")
    task.update(
        current=25,
        metrics={
            "loss": 7.9,
            "tok/s": "rolling-throughput-value-is-unavailable",
            "lr": 2e-5,
        },
    )

    line = stream.getvalue().split("\r")[-1].rstrip()
    assert "loss 7.9" in line
    assert "tok " not in line
    assert "lr " not in line
    task.complete()
    manager.close()
