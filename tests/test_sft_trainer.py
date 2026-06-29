from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from transformers.trainer_callback import PrinterCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.trainer_utils import IntervalStrategy, SaveStrategy

from shaft.config.training import EvalConfig, EvalDatasetPolicyConfig
from shaft.data import SFTDataset, SFTRecord, ShaftMixedIndexSampler
from shaft.training import ShaftEpochIntervalCallback
from shaft.training.optimizer_plan import build_resolved_optimizer_plan
from shaft.training.sft_trainer import ShaftSFTTrainer
from tests.support.training import StaticOnlineEvalRunner
from tests.support.training import TinyModel as _TinyModel
from tests.support.training import build_training_args
from tests.support.training import capture_trainer_logs, eval_loop_output


pytestmark = pytest.mark.component


def test_shaft_trainer_uses_custom_components() -> None:
    model = _TinyModel()
    args = build_training_args(
        output_dir="/tmp/shaft_trainer_smoke",
    )
    trainer = ShaftSFTTrainer(
        model=model,
        args=args,
        train_dataset=[],
        eval_dataset=[],
        data_collator=lambda x: x,
        loss_name="causal_lm",
        optimizer_name="adamw_torch",
        scheduler_name="linear",
        scheduler_num_cycles=2.0,
        scheduler_power=1.5,
    )
    assert not any(
        isinstance(callback, PrinterCallback) for callback in trainer.callback_handler.callbacks
    )
    device = next(model.parameters()).device
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3]], device=device),
        "labels": torch.tensor([[1, 2, 3]], device=device),
        "loss_scale": torch.tensor([[0.0, 1.0, 1.0]], device=device),
    }
    with patch("shaft.training.optimizer_mixin.build_optimizer_and_plan") as mocked_build_optim:
        mocked_build_optim.return_value = (
            torch.optim.AdamW(model.parameters(), lr=1e-3),
            build_resolved_optimizer_plan(
                model=model,
                args=args,
                finetune_plan=None,
                model_adapter=None,
                param_group_lrs={},
            ),
        )
        trainer.create_optimizer()
        mocked_build_optim.assert_called_once()
        _, kwargs = mocked_build_optim.call_args
        assert kwargs["param_group_lrs"] == {}
        assert kwargs["model_adapter"] is None
        assert kwargs["finetune_plan"] is None
    with patch("shaft.training.optimizer_mixin.build_scheduler") as mocked_build_sched:
        mocked_build_sched.return_value = torch.optim.lr_scheduler.LambdaLR(
            trainer.optimizer, lambda _: 1.0
        )
        trainer.create_scheduler(10)
        mocked_build_sched.assert_called_once()
        _, kwargs = mocked_build_sched.call_args
        assert kwargs["num_cycles"] == pytest.approx(2.0)
        assert kwargs["power"] == pytest.approx(1.5)
    loss = trainer.compute_loss(model, inputs)
    assert isinstance(loss, torch.Tensor)
    assert "loss_scale" not in (model.last_forward_kwargs or {})


def test_optimizer_summary_is_written_only_on_rank_zero(tmp_path, monkeypatch) -> None:
    model = _TinyModel()
    args = build_training_args(
        output_dir=str(tmp_path),
    )
    trainer = ShaftSFTTrainer(
        model=model,
        args=args,
        train_dataset=[],
        eval_dataset=[],
        data_collator=lambda x: x,
        loss_name="causal_lm",
        optimizer_name="adamw_torch",
        scheduler_name="linear",
    )

    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    with patch("shaft.training.optimizer_mixin.write_resolved_optimizer_summary") as mocked_write:
        trainer.create_optimizer()

    mocked_write.assert_not_called()


def test_shaft_trainer_uses_custom_train_sampler() -> None:
    model = _TinyModel()
    args = build_training_args(
        output_dir="/tmp/shaft_trainer_sampler",
    )
    records = {
        "a": [
            SFTRecord(image_path="/tmp/a.png", target_text="{}", dataset_name="a", sample_id="a0")
        ],
        "b": [
            SFTRecord(image_path="/tmp/b.png", target_text="{}", dataset_name="b", sample_id="b0")
        ],
    }
    sampler = ShaftMixedIndexSampler(
        records,
        {"a": 1.0, "b": 1.0},
        strategy="concat",
        refresh_mode="epoch_refresh",
        shuffle=False,
        seed=3,
        rank=0,
        world_size=1,
    )
    train_dataset = SFTDataset(
        records, mixed_length=len(sampler), mixed_indices=sampler.current_indices
    )
    trainer = ShaftSFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=[],
        train_sampler=sampler,
        data_collator=lambda batch: batch,
    )

    train_dataloader = trainer.get_train_dataloader()
    assert trainer._get_train_sampler(train_dataset) is sampler
    assert train_dataloader.batch_sampler.sampler is sampler


def test_shaft_trainer_evaluate_merges_online_metrics() -> None:
    model = _TinyModel()
    args = build_training_args(
        output_dir="/tmp/shaft_trainer_eval_smoke",
        per_device_eval_batch_size=1,
    )
    trainer = ShaftSFTTrainer(
        model=model,
        args=args,
        train_dataset=[],
        eval_dataset=[{"sample_id": "x"}],
        data_collator=lambda x: x,
        online_eval_runner=StaticOnlineEvalRunner(
            {
                "eval_final_score": 0.8,
                "eval_ds_a_exact_match": 0.7,
            }
        ),
    )
    trainer.get_eval_dataloader = lambda eval_dataset=None: []  # type: ignore[method-assign]
    trainer.evaluation_loop = lambda *a, **k: eval_loop_output({"eval_loss": 0.2})  # type: ignore[method-assign]
    logged = capture_trainer_logs(trainer)
    trainer.callback_handler.on_evaluate = lambda args, state, control, metrics: control  # type: ignore[method-assign]
    metrics = trainer.evaluate()
    assert metrics["eval_loss"] == pytest.approx(0.2)
    assert metrics["eval_final_score"] == pytest.approx(0.8)
    assert metrics["eval_ds_a_exact_match"] == pytest.approx(0.7)
    assert logged == [{"eval_loss": 0.2, "eval_final_score": 0.8, "eval_ds_a_exact_match": 0.7}]


def test_shaft_trainer_evaluate_aggregates_final_loss_for_named_eval_datasets() -> None:
    model = _TinyModel()
    args = build_training_args(
        output_dir="/tmp/shaft_trainer_eval_named",
        per_device_eval_batch_size=1,
    )
    trainer = ShaftSFTTrainer(
        model=model,
        args=args,
        train_dataset=[],
        eval_dataset={"ds_a": [{"sample_id": "a"}], "ds_b": [{"sample_id": "b"}]},
        data_collator=lambda x: x,
        online_eval_runner=StaticOnlineEvalRunner(
            {
                "eval_final_score": 0.8,
                "eval_ds_a_exact_match": 0.7,
                "eval_ds_b_exact_match": 0.9,
            }
        ),
        eval_config=EvalConfig(
            enabled=True,
            loss_metrics_enabled=True,
            online_metrics_enabled=True,
            datasets={
                "ds_a": EvalDatasetPolicyConfig(weight=0.25),
                "ds_b": EvalDatasetPolicyConfig(weight=0.75),
            },
        ),
    )
    trainer.get_eval_dataloader = lambda eval_dataset=None: []  # type: ignore[method-assign]

    def _fake_evaluation_loop(*args, **kwargs):
        prefix = kwargs["metric_key_prefix"]
        values = {
            "eval_ds_a": 0.4,
            "eval_ds_b": 0.2,
        }
        return eval_loop_output({f"{prefix}_loss": values[prefix]})

    trainer.evaluation_loop = _fake_evaluation_loop  # type: ignore[method-assign]
    logged = capture_trainer_logs(trainer)
    trainer.callback_handler.on_evaluate = lambda args, state, control, metrics: control  # type: ignore[method-assign]
    metrics = trainer.evaluate()
    assert metrics["eval_ds_a_loss"] == pytest.approx(0.4)
    assert metrics["eval_ds_b_loss"] == pytest.approx(0.2)
    assert metrics["eval_final_loss"] == pytest.approx(0.25)
    assert metrics["eval_final_score"] == pytest.approx(0.8)
    assert metrics["eval_ds_a_exact_match"] == pytest.approx(0.7)
    assert metrics["eval_ds_b_exact_match"] == pytest.approx(0.9)
    assert logged == [
        {
            "eval_ds_a_loss": 0.4,
            "eval_ds_b_loss": 0.2,
            "eval_final_loss": 0.25,
            "eval_final_score": 0.8,
            "eval_ds_a_exact_match": 0.7,
            "eval_ds_b_exact_match": 0.9,
        }
    ]


def test_shaft_trainer_evaluate_reports_only_eval_loss_without_online_eval() -> None:
    model = _TinyModel()
    args = build_training_args(
        output_dir="/tmp/shaft_trainer_eval_loss_only",
        per_device_eval_batch_size=1,
    )
    trainer = ShaftSFTTrainer(
        model=model,
        args=args,
        train_dataset=[],
        eval_dataset=[{"sample_id": "x"}],
        data_collator=lambda x: x,
    )
    trainer.get_eval_dataloader = lambda eval_dataset=None: []  # type: ignore[method-assign]
    trainer.evaluation_loop = lambda *a, **k: eval_loop_output(  # type: ignore[method-assign]
        {"eval_loss": 0.3, "eval_samples_per_second": 12.0}
    )
    logged = capture_trainer_logs(trainer)
    trainer.callback_handler.on_evaluate = lambda args, state, control, metrics: control  # type: ignore[method-assign]
    metrics = trainer.evaluate()
    assert metrics["eval_loss"] == pytest.approx(0.3)
    assert "eval_samples_per_second" in metrics
    assert logged == [{"eval_loss": 0.3}]


def test_epoch_interval_callback_gates_eval_and_save_until_interval_or_final_epoch() -> None:
    callback = ShaftEpochIntervalCallback(eval_epoch_interval=2, save_epoch_interval=2)
    args = SimpleNamespace(
        eval_strategy=IntervalStrategy.EPOCH,
        save_strategy=SaveStrategy.EPOCH,
        num_train_epochs=5,
    )

    control = TrainerControl(should_evaluate=True, should_save=True)
    state = TrainerState(epoch=1.0)
    result = callback.on_epoch_end(args, state, control)
    assert result.should_evaluate is False
    assert result.should_save is False

    control = TrainerControl(should_evaluate=True, should_save=True)
    state = TrainerState(epoch=2.0)
    result = callback.on_epoch_end(args, state, control)
    assert result.should_evaluate is True
    assert result.should_save is True

    control = TrainerControl(should_evaluate=True, should_save=True)
    state = TrainerState(epoch=5.0)
    result = callback.on_epoch_end(args, state, control)
    assert result.should_evaluate is True
    assert result.should_save is True
