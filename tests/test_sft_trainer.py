from __future__ import annotations

import random
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from transformers.trainer_callback import PrinterCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.trainer_utils import IntervalStrategy, SaveStrategy

from shaft.config.training import EvalConfig, EvalDatasetPolicyConfig
from shaft.data import SFTDataset, SFTRecord, ShaftSamplePlan, ShaftSampleSampler
from shaft.training import ShaftEpochIntervalCallback
from shaft.training.online_eval import ShaftOnlineEvalRunner
from shaft.training.optimizer_plan import build_resolved_optimizer_plan
from shaft.training.sft_trainer import ShaftSFTTrainer
from tests.support.training import StaticOnlineEvalRunner
from tests.support.training import TinyModel as _TinyModel
from tests.support.training import build_training_args
from tests.support.training import capture_trainer_logs, eval_loop_output


pytestmark = pytest.mark.component


class _TaggedEvalCollator:
    def __init__(self, source: str) -> None:
        self.source = source

    def __call__(self, rows):
        return {
            "source": self.source,
            "sample_ids": [str(row["sample_id"]) for row in rows],
            "input_ids": torch.tensor(
                [row.get("input_ids", [1, 2]) for row in rows],
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                [row.get("labels", [1, 2]) for row in rows],
                dtype=torch.long,
            ),
        }


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
    assert model.last_forward_labels is None


def test_shaft_trainer_delegates_private_varlen_inputs_before_device_transfer() -> None:
    model = _TinyModel()
    args = build_training_args(output_dir="/tmp/shaft_trainer_varlen_prepare")

    class _SequenceAdapter:
        def __init__(self) -> None:
            self.seen_model = None
            self.seen_layout = None

        def prepare_sequence_training_inputs(self, *, model, inputs):
            self.seen_model = model
            prepared = dict(inputs)
            self.seen_layout = prepared.pop("_shaft_varlen_layout")
            prepared["position_ids"] = torch.arange(
                prepared["input_ids"].shape[-1],
                dtype=torch.long,
            ).view(1, 1, -1)
            return prepared

    adapter = _SequenceAdapter()
    trainer = ShaftSFTTrainer(
        model=model,
        args=args,
        train_dataset=[],
        eval_dataset=[],
        data_collator=lambda x: x,
        model_adapter=adapter,
    )
    layout = object()

    prepared = trainer._prepare_inputs(
        {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "_shaft_varlen_layout": layout,
        }
    )

    assert adapter.seen_model is model
    assert adapter.seen_layout is layout
    assert "_shaft_varlen_layout" not in prepared
    assert prepared["position_ids"].shape == (1, 1, 3)


def test_custom_train_batch_sampler_keeps_batches_on_host_until_prepare_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _SingleBatchSampler:
        batch_size = None
        drop_last = True

        def __iter__(self):
            yield [0]

        def __len__(self):
            return 1

    model = _TinyModel()
    args = build_training_args(output_dir="/tmp/shaft_trainer_host_planned_batch")
    trainer = ShaftSFTTrainer(
        model=model,
        args=args,
        train_dataset=[torch.tensor([1, 2, 3])],
        eval_dataset=[],
        train_batch_sampler=_SingleBatchSampler(),
        data_collator=lambda rows: {"input_ids": torch.stack(rows)},
    )
    prepare_data_loader = trainer.accelerator.prepare_data_loader
    observed_device_placement: list[bool | None] = []

    def _capture_prepare_data_loader(
        dataloader,
        device_placement=None,
        slice_fn_for_dispatch=None,
    ):
        observed_device_placement.append(device_placement)
        return prepare_data_loader(
            dataloader,
            device_placement=device_placement,
            slice_fn_for_dispatch=slice_fn_for_dispatch,
        )

    monkeypatch.setattr(
        trainer.accelerator,
        "prepare_data_loader",
        _capture_prepare_data_loader,
    )

    batch = next(iter(trainer.get_train_dataloader()))

    assert observed_device_placement == [False]
    assert batch["input_ids"].device.type == "cpu"
    prepared = trainer._prepare_inputs(batch)
    assert prepared["input_ids"].device == trainer.args.device


def test_shaft_trainer_uses_a_distinct_padded_eval_collator() -> None:
    model = _TinyModel()
    args = build_training_args(output_dir="/tmp/shaft_trainer_eval_collator")

    def train_collator(rows):
        return {"source": "train", "rows": rows}

    def eval_collator(rows):
        return {"source": "eval", "rows": rows}

    trainer = ShaftSFTTrainer(
        model=model,
        args=args,
        train_dataset=[{"value": 1}],
        eval_dataset=[{"value": 2}],
        data_collator=train_collator,
        eval_data_collator=eval_collator,
    )

    batch = next(iter(trainer.get_eval_dataloader()))

    assert batch["source"] == "eval"
    assert trainer.data_collator is train_collator


def test_loss_and_online_eval_have_distinct_persistent_loader_namespaces() -> None:
    dataset = [{"sample_id": "one", "input_ids": [1, 2], "labels": [1, 2]}]
    train_collator = _TaggedEvalCollator("train")
    loss_collator = _TaggedEvalCollator("loss")
    online_collator = _TaggedEvalCollator("online")
    trainer = ShaftSFTTrainer(
        model=_TinyModel(),
        args=build_training_args(
            output_dir="/tmp/shaft_trainer_eval_loader_namespaces",
            per_device_eval_batch_size=1,
            dataloader_num_workers=1,
            dataloader_persistent_workers=True,
            remove_unused_columns=False,
        ),
        train_dataset=[],
        eval_dataset=dataset,
        data_collator=train_collator,
        eval_data_collator=loss_collator,
    )
    runner = ShaftOnlineEvalRunner(
        eval_config=EvalConfig(),
        prompt_collator=online_collator,
    )

    loss_loader = trainer.get_eval_dataloader()
    assert next(iter(loss_loader))["source"] == "loss"
    online_loader = runner._get_prompt_eval_dataloaders(trainer, dataset)[0]
    assert next(iter(online_loader))["source"] == "online"

    assert trainer.get_eval_dataloader() is loss_loader
    assert runner._get_prompt_eval_dataloaders(trainer, dataset)[0] is online_loader
    assert online_loader is not loss_loader
    assert set(trainer._eval_dataloaders) == {
        "eval",
        f"shaft-online:default:{id(dataset)}",
    }


def test_named_loss_eval_datasets_do_not_share_a_persistent_loader() -> None:
    eval_datasets = {
        "a": [{"sample_id": "a", "input_ids": [1, 2], "labels": [1, 2]}],
        "b": [{"sample_id": "b", "input_ids": [2, 3], "labels": [2, 3]}],
    }
    trainer = ShaftSFTTrainer(
        model=_TinyModel(),
        args=build_training_args(
            output_dir="/tmp/shaft_trainer_named_eval_loader_namespaces",
            per_device_eval_batch_size=1,
            dataloader_num_workers=1,
            dataloader_persistent_workers=True,
            remove_unused_columns=False,
        ),
        train_dataset=[],
        eval_dataset=eval_datasets,
        data_collator=_TaggedEvalCollator("train"),
        eval_data_collator=_TaggedEvalCollator("loss"),
    )
    observed: dict[str, list[str]] = {}

    def _evaluation_loop(dataloader, *args, metric_key_prefix, **kwargs):
        _ = args, kwargs
        batch = next(iter(dataloader))
        observed[metric_key_prefix] = batch["sample_ids"]
        return eval_loop_output(
            {f"{metric_key_prefix}_loss": 1.0},
            num_samples=1,
        )

    trainer.evaluation_loop = _evaluation_loop  # type: ignore[method-assign]
    trainer._evaluate_named_datasets(
        eval_datasets=eval_datasets,
        ignore_keys=None,
        metric_key_prefix="eval",
    )

    assert observed == {"eval_a": ["a"], "eval_b": ["b"]}
    assert {"a", "b"}.issubset(trainer._eval_dataloaders)


def test_shaft_trainer_counts_weighted_optimizer_batch_denominator() -> None:
    model = _TinyModel()
    args = build_training_args(output_dir="/tmp/shaft_trainer_denominator")
    args.average_tokens_across_devices = True
    trainer = ShaftSFTTrainer(
        model=model,
        args=args,
        train_dataset=[],
        eval_dataset=[],
        data_collator=lambda x: x,
    )
    batches = [
        {
            "labels": torch.tensor([[0, 1, 2, -100]], dtype=torch.long),
            "loss_scale": torch.tensor([[0.0, 0.5, 2.0, 0.0]]),
        },
        {
            "labels": torch.tensor([[0, 3, 4, 5]], dtype=torch.long),
            "loss_scale": torch.tensor([[0.0, 1.0, 1.0, 1.0]]),
        },
    ]

    denominator = trainer._get_num_items_in_batch(batches, torch.device("cpu"))

    assert denominator is not None
    assert float(denominator) == pytest.approx(5.5)


def test_weighted_denominator_is_not_divided_in_average_tokens_mode() -> None:
    model = _TinyModel()
    args = build_training_args(output_dir="/tmp/shaft_trainer_data_parallel")
    args.average_tokens_across_devices = True
    args._n_gpu = 2
    trainer = ShaftSFTTrainer(
        model=model,
        args=args,
        train_dataset=[],
        eval_dataset=[],
        data_collator=lambda x: x,
    )
    batches = [
        {
            "labels": torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
            "loss_scale": torch.tensor([[0.0, 0.5, 2.0, 3.0]]),
        }
    ]

    denominator = trainer._get_num_items_in_batch(batches, torch.device("cpu"))

    assert denominator is not None
    assert float(denominator) == pytest.approx(5.5)


def test_weighted_denominator_preserves_fractions_when_removing_replicas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _TinyModel()
    args = build_training_args(output_dir="/tmp/shaft_trainer_replicated_denominator")
    args.average_tokens_across_devices = True
    trainer = ShaftSFTTrainer(
        model=model,
        args=args,
        train_dataset=[],
        eval_dataset=[],
        data_collator=lambda x: x,
    )
    monkeypatch.setattr(
        trainer.accelerator.state,
        "parallelism_config",
        SimpleNamespace(non_data_parallel_size=2),
    )
    batches = [
        {
            "labels": torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
            "loss_scale": torch.tensor([[0.0, 0.5, 2.0, 3.0]]),
        }
    ]

    denominator = trainer._get_num_items_in_batch(batches, torch.device("cpu"))

    assert denominator is not None
    assert float(denominator) == pytest.approx(2.75)


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
    plan = ShaftSamplePlan(
        {name: len(rows) for name, rows in records.items()},
        {"a": 1.0, "b": 1.0},
        strategy="concat",
        shuffle=False,
        seed=3,
    )
    sampler = ShaftSampleSampler(plan, rank=0, world_size=1)
    train_dataset = SFTDataset(records, sample_plan=plan)
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


def test_shaft_trainer_rejects_pre_sharded_train_sampler() -> None:
    model = _TinyModel()
    args = build_training_args(
        output_dir="/tmp/shaft_trainer_pre_sharded_sampler",
    )
    plan = ShaftSamplePlan(
        {"a": 2},
        {"a": 1.0},
        strategy="concat",
        shuffle=False,
        seed=3,
    )

    with pytest.raises(ValueError, match="unsharded"):
        ShaftSFTTrainer(
            model=model,
            args=args,
            train_dataset=[0, 1],
            eval_dataset=[],
            train_sampler=ShaftSampleSampler(plan, rank=1, world_size=2),
            data_collator=lambda batch: batch,
        )


def test_shaft_trainer_uses_variable_train_batch_sampler() -> None:
    class _VariableBatchSampler:
        batch_size = None
        drop_last = True

        def __iter__(self):
            yield [0]
            yield [1, 2]

        def __len__(self):
            return 2

    model = _TinyModel()
    args = build_training_args(
        output_dir="/tmp/shaft_trainer_batch_sampler",
    )
    trainer = ShaftSFTTrainer(
        model=model,
        args=args,
        train_dataset=[0, 1, 2],
        eval_dataset=[],
        train_batch_sampler=_VariableBatchSampler(),
        data_collator=lambda batch: batch,
    )

    train_dataloader = trainer.get_train_dataloader()

    assert list(train_dataloader) == [[0], [1, 2]]
    assert trainer.accelerator.even_batches is True
    initial_values = trainer.set_initial_training_values(args, train_dataloader)
    assert len(initial_values) >= 7


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


def test_training_evaluation_preserves_host_rng_state() -> None:
    trainer = ShaftSFTTrainer(
        model=_TinyModel(),
        args=build_training_args(output_dir="/tmp/shaft_trainer_eval_rng"),
        train_dataset=[],
        eval_dataset=[],
        data_collator=lambda batch: batch,
    )
    trainer.is_in_train = True

    def _consume_rng(**kwargs):
        _ = kwargs
        random.random()
        np.random.random()
        torch.rand(())
        return {"eval_loss": 0.0}

    trainer._evaluate_impl = _consume_rng  # type: ignore[method-assign]
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()

    assert trainer.evaluate() == {"eval_loss": 0.0}
    assert random.getstate() == python_state
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    assert torch.equal(torch.random.get_rng_state(), torch_state)


def test_training_evaluation_restores_cuda_and_host_rng_after_exception() -> None:
    trainer = ShaftSFTTrainer(
        model=_TinyModel(),
        args=build_training_args(output_dir="/tmp/shaft_trainer_eval_rng_error"),
        train_dataset=[],
        eval_dataset=[],
        data_collator=lambda batch: batch,
    )
    trainer.is_in_train = True

    def _consume_rng_and_fail(**kwargs):
        _ = kwargs
        random.random()
        np.random.random()
        torch.rand(())
        raise RuntimeError("synthetic eval failure")

    trainer._evaluate_impl = _consume_rng_and_fail  # type: ignore[method-assign]
    random.seed(19)
    np.random.seed(19)
    torch.manual_seed(19)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_state = torch.tensor([1, 2, 3], dtype=torch.uint8)

    with patch("torch.cuda.is_available", return_value=True):
        with patch("torch.cuda.is_initialized", return_value=True):
            with patch("torch.cuda.current_device", return_value=1):
                with patch("torch.cuda.get_rng_state", return_value=cuda_state) as get_rng:
                    with patch("torch.cuda.set_rng_state") as set_rng:
                        with pytest.raises(RuntimeError, match="synthetic eval failure"):
                            trainer.evaluate()

    get_rng.assert_called_once_with(1)
    set_rng.assert_called_once_with(cuda_state, 1)
    assert random.getstate() == python_state
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    assert torch.equal(torch.random.get_rng_state(), torch_state)


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
