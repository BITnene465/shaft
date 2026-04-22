from __future__ import annotations

from types import SimpleNamespace

import pytest

from shaft.training.progress_callback import ShaftProgressCallback


class _DummyBar:
    def __init__(self) -> None:
        self.updated_by: list[int] = []
        self.postfix: dict[str, str] = {}
        self.closed = False

    def update(self, value: int) -> None:
        self.updated_by.append(int(value))

    def set_postfix(self, value, refresh: bool = False) -> None:  # noqa: ANN001, FBT002
        _ = refresh
        self.postfix = dict(value)

    def close(self) -> None:
        self.closed = True


class _DummyScheduler:
    def __init__(self, lr: float) -> None:
        self.lr = float(lr)

    def get_last_lr(self) -> list[float]:
        return [self.lr]


def _build_state(*, global_step: int, max_steps: int, world_zero: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        global_step=int(global_step),
        max_steps=int(max_steps),
        is_world_process_zero=bool(world_zero),
    )


def test_progress_callback_initializes_training_bar_from_resume_step(monkeypatch: pytest.MonkeyPatch) -> None:
    callback = ShaftProgressCallback()
    bar = _DummyBar()
    captured: dict[str, int | str | float | bool | None] = {}

    def _fake_create_progress_bar(**kwargs):  # noqa: ANN003
        captured.update(kwargs)
        return bar

    monkeypatch.setattr("shaft.training.progress_callback.create_progress_bar", _fake_create_progress_bar)

    state = _build_state(global_step=12021, max_steps=24042)
    callback.on_train_begin(
        args=object(),
        state=state,
        control=object(),
        optimizer=SimpleNamespace(param_groups=[{"lr": 5e-6}]),
        lr_scheduler=_DummyScheduler(5e-6),
    )

    assert callback.current_step == 12021
    assert captured["initial"] == 12021
    assert captured["total"] == 24042
    assert bar.postfix["learning_rate"] == "5e-06"


def test_progress_callback_updates_learning_rate_every_step(monkeypatch: pytest.MonkeyPatch) -> None:
    callback = ShaftProgressCallback()
    bar = _DummyBar()
    monkeypatch.setattr("shaft.training.progress_callback.create_progress_bar", lambda **_: bar)

    scheduler = _DummyScheduler(5e-6)
    optimizer = SimpleNamespace(param_groups=[{"lr": 5e-6}])
    callback.on_train_begin(
        args=object(),
        state=_build_state(global_step=12021, max_steps=24042),
        control=object(),
        optimizer=optimizer,
        lr_scheduler=scheduler,
    )

    scheduler.lr = 4e-6
    optimizer.param_groups[0]["lr"] = 4e-6
    callback.on_step_end(
        args=object(),
        state=_build_state(global_step=12025, max_steps=24042),
        control=object(),
        optimizer=optimizer,
        lr_scheduler=scheduler,
    )

    assert callback.current_step == 12025
    assert bar.updated_by == [4]
    assert bar.postfix["learning_rate"] == "4e-06"


def test_progress_callback_keeps_learning_rate_visible_after_logs(monkeypatch: pytest.MonkeyPatch) -> None:
    callback = ShaftProgressCallback()
    bar = _DummyBar()
    monkeypatch.setattr("shaft.training.progress_callback.create_progress_bar", lambda **_: bar)

    callback.on_train_begin(
        args=object(),
        state=_build_state(global_step=0, max_steps=100),
        control=object(),
        optimizer=SimpleNamespace(param_groups=[{"lr": 2e-5}]),
        lr_scheduler=_DummyScheduler(2e-5),
    )

    callback.on_log(
        args=object(),
        state=_build_state(global_step=1, max_steps=100),
        control=object(),
        logs={"loss": 1.23456, "grad_norm": 0.98765},
    )

    assert bar.postfix["learning_rate"] == "2e-05"
    assert bar.postfix["loss"] == "1.235"
    assert bar.postfix["grad_norm"] == "0.9877"
