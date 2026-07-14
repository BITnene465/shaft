from __future__ import annotations

from types import SimpleNamespace

from shaft.observability import ShaftProgressManager
from shaft.training.progress_callback import ShaftProgressCallback


class _RecordingSink:
    def __init__(self) -> None:
        self.records = []

    def publish(self, snapshot, event) -> None:  # noqa: ANN001
        self.records.append((snapshot, event))

    def close(self) -> None:
        return None


class _DummyScheduler:
    def __init__(self, lr: float | list[float]) -> None:
        self.lr = lr

    def get_last_lr(self) -> list[float]:
        if isinstance(self.lr, list):
            return list(self.lr)
        return [float(self.lr)]


class _BrokenScheduler:
    def get_last_lr(self) -> list[float]:
        raise RuntimeError("scheduler state is unavailable")


def _build_state(
    *,
    global_step: int,
    max_steps: int,
    world_zero: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        global_step=int(global_step),
        max_steps=int(max_steps),
        is_world_process_zero=bool(world_zero),
    )


def _build_callback() -> tuple[ShaftProgressCallback, ShaftProgressManager]:
    manager = ShaftProgressManager(run_id="run", sinks=[_RecordingSink()])
    return ShaftProgressCallback(manager), manager


def test_progress_callback_resumes_from_trainer_state_and_tracks_current_lr() -> None:
    callback, manager = _build_callback()
    scheduler = _DummyScheduler(5e-6)
    optimizer = SimpleNamespace(param_groups=[{"lr": 5e-6}])
    callback.on_train_begin(
        args=object(),
        state=_build_state(global_step=12021, max_steps=24042),
        control=object(),
        optimizer=optimizer,
        lr_scheduler=scheduler,
    )

    task = manager.snapshot.tasks["train"]
    assert task.current == 12021
    assert task.total == 24042
    assert task.metrics == {"lr": 5e-6}
    assert task.display_rate is True

    scheduler.lr = 4e-6
    callback.on_step_end(
        args=object(),
        state=_build_state(global_step=12025, max_steps=24042),
        control=object(),
        optimizer=optimizer,
        lr_scheduler=scheduler,
    )

    task = manager.snapshot.tasks["train"]
    assert task.current == 12025
    assert task.metrics == {"lr": 4e-6}


def test_progress_callback_exposes_loss_token_throughput_and_current_lr() -> None:
    callback, manager = _build_callback()
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
        logs={
            "loss": 1.23456,
            "learning_rate": 1.5e-5,
            "grad_norm": 0.98765,
            "eval_loss": 3.0,
            "efficiency/useful_tokens_per_second": 4_103.7,
        },
    )

    assert manager.snapshot.tasks["train"].metrics == {
        "loss": 1.23456,
        "lr": 2e-5,
        "tok/s": 4_103.7,
        "grad_norm": 0.98765,
    }


def test_progress_callback_reports_all_optimizer_group_lrs_as_one_compact_range() -> None:
    callback, manager = _build_callback()
    scheduler = _DummyScheduler([2.5e-7, 2.5e-7, 5e-7, 5e-7])
    optimizer = SimpleNamespace(
        param_groups=[
            {"lr": 2.5e-7},
            {"lr": 2.5e-7},
            {"lr": 5e-7},
            {"lr": 5e-7},
        ]
    )

    callback.on_train_begin(
        args=object(),
        state=_build_state(global_step=25, max_steps=10_000),
        control=object(),
        optimizer=optimizer,
        lr_scheduler=scheduler,
    )

    assert manager.snapshot.tasks["train"].metrics == {"lr": "2.5–5e-7"}


def test_progress_callback_falls_back_to_optimizer_when_scheduler_probe_fails() -> None:
    callback, manager = _build_callback()
    optimizer = SimpleNamespace(param_groups=[{"lr": 2.5e-7}, {"lr": 5e-7}])

    callback.on_train_begin(
        args=object(),
        state=_build_state(global_step=0, max_steps=10),
        control=object(),
        optimizer=optimizer,
        lr_scheduler=_BrokenScheduler(),
    )

    assert manager.snapshot.tasks["train"].metrics == {"lr": "2.5–5e-7"}


def test_progress_callback_rejects_partial_invalid_lr_ranges_and_falls_back() -> None:
    callback, manager = _build_callback()
    optimizer = SimpleNamespace(param_groups=[{"lr": 1e-6}, {"lr": 2e-6}])

    callback.on_train_begin(
        args=object(),
        state=_build_state(global_step=0, max_steps=10),
        control=object(),
        optimizer=optimizer,
        lr_scheduler=_DummyScheduler([5e-7, float("nan")]),
    )

    assert manager.snapshot.tasks["train"].metrics == {"lr": "1–2e-6"}


def test_progress_callback_temporarily_foregrounds_eval_then_restores_train() -> None:
    callback, manager = _build_callback()
    state = _build_state(global_step=3, max_steps=10)
    callback.on_train_begin(args=object(), state=state, control=object())

    args = SimpleNamespace(eval_strategy="steps")
    dataloader = [object(), object()]
    callback.on_prediction_step(
        args=args,
        state=state,
        control=object(),
        eval_dataloader=dataloader,
    )
    callback.on_prediction_step(
        args=args,
        state=state,
        control=object(),
        eval_dataloader=dataloader,
    )

    snapshot = manager.snapshot
    assert snapshot.active_task_id == "eval.loss"
    assert snapshot.tasks["eval.loss"].current == 2
    assert snapshot.tasks["eval.loss"].parent_task_id == "train"

    callback.on_evaluate(
        args=args,
        state=state,
        control=object(),
        metrics={"eval_loss": 0.75},
    )
    snapshot = manager.snapshot
    assert snapshot.active_task_id == "train"
    assert snapshot.tasks["eval.loss"].state == "succeeded"
    assert snapshot.tasks["eval.loss"].metrics == {"loss": 0.75}

    callback.on_train_end(args=args, state=state, control=object())
    assert manager.snapshot.tasks["train"].state == "succeeded"


def test_progress_callback_accumulates_named_eval_dataloader_totals() -> None:
    callback, manager = _build_callback()
    state = _build_state(global_step=3, max_steps=10)
    args = SimpleNamespace(eval_strategy="steps")
    callback.on_train_begin(args=args, state=state, control=object())
    first_loader = [object(), object()]
    second_loader = [object(), object(), object()]

    for _ in first_loader:
        callback.on_prediction_step(
            args=args,
            state=state,
            control=object(),
            eval_dataloader=first_loader,
        )
    assert manager.snapshot.tasks["eval.loss"].current == 2
    assert manager.snapshot.tasks["eval.loss"].total == 2

    callback.on_prediction_step(
        args=args,
        state=state,
        control=object(),
        eval_dataloader=second_loader,
    )
    task = manager.snapshot.tasks["eval.loss"]
    assert task.current == 3
    assert task.total == 5
    for _ in second_loader[1:]:
        callback.on_prediction_step(
            args=args,
            state=state,
            control=object(),
            eval_dataloader=second_loader,
        )

    callback.on_evaluate(args=args, state=state, control=object())
    task = manager.snapshot.tasks["eval.loss"]
    assert task.state == "succeeded"
    assert task.current == 5
    assert task.total == 5


def test_progress_callback_detects_a_reused_named_eval_dataloader() -> None:
    callback, manager = _build_callback()
    state = _build_state(global_step=3, max_steps=10)
    args = SimpleNamespace(eval_strategy="steps")
    callback.on_train_begin(args=args, state=state, control=object())
    reused_loader = [object(), object()]

    for _ in range(2):
        for _ in reused_loader:
            callback.on_prediction_step(
                args=args,
                state=state,
                control=object(),
                eval_dataloader=reused_loader,
            )

    task = manager.snapshot.tasks["eval.loss"]
    assert task.current == 4
    assert task.total == 4


def test_progress_callback_does_not_publish_from_nonzero_rank() -> None:
    callback, manager = _build_callback()
    callback.on_train_begin(
        args=object(),
        state=_build_state(global_step=0, max_steps=10, world_zero=False),
        control=object(),
    )

    assert manager.snapshot.tasks == {}
