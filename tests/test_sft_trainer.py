from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import random
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from peft import LoraConfig, get_peft_model
import pytest
import torch
from accelerate.data_loader import BatchSamplerShard, DataLoaderShard, skip_first_batches
from torch.utils.data import BatchSampler
from transformers import PretrainedConfig, PreTrainedModel, Trainer
from transformers.trainer_callback import PrinterCallback, TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.trainer_utils import IntervalStrategy, SaveStrategy

from shaft.config.training import EvalConfig, EvalDatasetPolicyConfig
from shaft.data import (
    SFTDataset,
    SFTRecord,
    ShaftCollatedBatchStats,
    ShaftSamplePlan,
    ShaftBatchPlanningSpec,
    ShaftPlannedBatchSampler,
    ShaftSampleCost,
    ShaftSampleRef,
    ShaftSampleSchedule,
    ShaftSampleSampler,
)
from shaft.training import ShaftEpochIntervalCallback
from shaft.training.checkpointing import (
    MODEL_ONLY_CHECKPOINT_COMMIT_FILENAME,
    ShaftCheckpointProtocol,
    ensure_hf_export_layout,
    resolve_resume_checkpoint,
    validate_model_only_checkpoint,
    validate_training_checkpoint_commit,
)
from shaft.training.efficiency import ShaftTrainingEfficiencyCallback
from shaft.training.efficiency import ShaftTrainingEfficiencyMonitor
from shaft.training.online_eval import ShaftOnlineEvalRunner
from shaft.training.optimizer_plan import build_resolved_optimizer_plan
from shaft.training.sft_trainer import ShaftSFTTrainer
from shaft.training.train_sampler_mixin import ShaftTrainSamplerMixin
from shaft.model.types import (
    ModelModuleGroups,
    ShaftAuxiliaryLossTerm,
    ShaftEvalAuxiliaryMetric,
    ShaftEvalAuxiliaryStatistic,
    TrainingObjectivePolicy,
)
from shaft.model import build_model_meta
from tests.support.training import StaticOnlineEvalRunner
from tests.support.training import TinyModel as _TinyModel
from tests.support.training import build_training_args
from tests.support.training import capture_trainer_logs, eval_loop_output


pytestmark = pytest.mark.component


def test_fsdp_peft_best_model_load_fails_closed() -> None:
    trainer = object.__new__(ShaftSFTTrainer)
    trainer.is_fsdp_enabled = True
    trainer._shaft_uses_peft = True

    with pytest.raises(RuntimeError, match="load_best_model_at_end"):
        trainer._load_best_model()


def test_fsdp_peft_resume_preloads_before_wrap_and_only_skips_native_model(
    tmp_path: Path,
) -> None:
    checkpoint = (tmp_path / "checkpoint-1").resolve()
    other_checkpoint = (tmp_path / "checkpoint-2").resolve()
    artifact = SimpleNamespace(path=str(checkpoint))
    trainer = object.__new__(ShaftSFTTrainer)
    trainer.model = get_peft_model(
        _TinyCheckpointModel(_TinyCheckpointConfig()),
        LoraConfig(target_modules=["fc"], r=2, lora_alpha=4),
    )
    trainer.is_fsdp_enabled = True
    trainer.model_init = None
    trainer.args = SimpleNamespace(output_dir=str(tmp_path), world_size=1)
    trainer._shaft_preloaded_fsdp_peft_checkpoint = None
    trainer._shaft_resume_peft_artifact = artifact
    events: list[tuple[str, object]] = []
    result = object()

    def fake_preload(model, checkpoint_dir, *, resolved_artifact=None):
        assert model is trainer.model
        events.append(
            (
                "preload",
                (Path(checkpoint_dir).resolve(), resolved_artifact),
            )
        )
        return (7, 11)

    def fake_parent_model_load(self, resume_from_checkpoint, model=None):
        _ = self, model
        events.append(("parent-model-load", Path(resume_from_checkpoint).resolve()))

    def fake_parent_optimizer_load(self, resume_from_checkpoint):
        _ = self
        events.append(("parent-optimizer-load", Path(resume_from_checkpoint).resolve()))

    def fake_parent_train(
        self,
        resume_from_checkpoint=None,
        trial=None,
        ignore_keys_for_eval=None,
    ):
        _ = trial, ignore_keys_for_eval
        events.append(("parent-train", resume_from_checkpoint))
        self._load_from_checkpoint(str(checkpoint))
        self._load_from_checkpoint(str(other_checkpoint))
        self._load_optimizer_and_scheduler(str(checkpoint))
        return result

    with (
        patch(
            "shaft.training.sft_trainer.get_last_checkpoint",
            return_value=str(checkpoint),
        ),
        patch("shaft.training.sft_trainer.load_peft_checkpoint", fake_preload),
        patch.object(Trainer, "_load_from_checkpoint", fake_parent_model_load),
        patch.object(
            Trainer,
            "_load_optimizer_and_scheduler",
            fake_parent_optimizer_load,
        ),
        patch.object(Trainer, "train", fake_parent_train),
    ):
        actual = trainer.train(resume_from_checkpoint=True)

    assert actual is result
    assert events == [
        ("preload", (checkpoint, artifact)),
        ("parent-train", str(checkpoint)),
        ("parent-model-load", other_checkpoint),
        ("parent-optimizer-load", checkpoint),
    ]


def test_fsdp_peft_resume_rejects_model_init_before_preload(tmp_path: Path) -> None:
    checkpoint = (tmp_path / "checkpoint-1").resolve()
    trainer = object.__new__(ShaftSFTTrainer)
    trainer.model = get_peft_model(
        _TinyCheckpointModel(_TinyCheckpointConfig()),
        LoraConfig(target_modules=["fc"], r=2, lora_alpha=4),
    )
    trainer.is_fsdp_enabled = True
    trainer.model_init = lambda: trainer.model
    trainer.args = SimpleNamespace(output_dir=str(tmp_path))
    trainer._shaft_preloaded_fsdp_peft_checkpoint = None
    trainer._shaft_resume_peft_artifact = SimpleNamespace(path=str(checkpoint))

    with (
        patch("shaft.training.sft_trainer.load_peft_checkpoint") as preload,
        pytest.raises(ValueError, match="model_init"),
    ):
        trainer.train(resume_from_checkpoint=str(checkpoint))
    preload.assert_not_called()


def test_sampler_resume_step_reader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-2"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text(
        '{"global_step": 2, "global_step": 2}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Cannot read trainer global_step"):
        ShaftTrainSamplerMixin._checkpoint_global_step(checkpoint)


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


class _TinyCheckpointConfig(PretrainedConfig):
    model_type = "shaft_tiny_checkpoint"

    def __init__(self, vocab_size: int = 16, **kwargs) -> None:
        super().__init__(**kwargs)
        self.vocab_size = int(vocab_size)


class _TinyCheckpointModel(PreTrainedModel):
    config_class = _TinyCheckpointConfig

    def __init__(self, config: _TinyCheckpointConfig) -> None:
        super().__init__(config)
        self.emb = torch.nn.Embedding(config.vocab_size, 8)
        self.fc = torch.nn.Linear(8, config.vocab_size)
        self.post_init()

    def forward(self, input_ids=None, labels=None, **kwargs):
        _ = labels, kwargs
        hidden = self.emb(input_ids)
        return SimpleNamespace(logits=self.fc(hidden))


class _DeepSpeedPlaceholderModel(torch.nn.Module):
    main_input_name = "input_ids"

    def __init__(self) -> None:
        super().__init__()
        self.emb = torch.nn.Embedding(2, 2)
        self.emb.weight = torch.nn.Parameter(torch.empty(0))
        self.emb.weight.ds_numel = 19
        self.weight = torch.nn.Parameter(torch.empty(0))
        self.weight.ds_numel = 23
        self.bias = torch.nn.Parameter(torch.empty(0))
        self.bias.ds_shape = (5,)

    def forward(self, input_ids=None, **kwargs):
        _ = input_ids, kwargs
        raise AssertionError("The parameter-count test must not execute forward.")


class _AuxiliaryOutput:
    def __init__(self, logits: torch.Tensor, auxiliary_loss: torch.Tensor) -> None:
        self.logits = logits
        self.auxiliary_loss = auxiliary_loss


class _AuxiliaryModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logit_scale = torch.nn.Parameter(torch.tensor(1.0))
        self.router_scale = torch.nn.Parameter(torch.tensor(2.0))
        self.last_forward_kwargs: dict[str, object] = {}
        self.last_forward_labels = None

    def forward(self, input_ids=None, labels=None, **kwargs):
        self.last_forward_kwargs = dict(kwargs)
        self.last_forward_labels = labels
        batch, sequence = input_ids.shape
        logits = torch.zeros(
            (batch, sequence, 4),
            dtype=self.logit_scale.dtype,
            device=self.logit_scale.device,
        ) + self.logit_scale * 0.0
        return _AuxiliaryOutput(logits, self.router_scale)


class _AuxiliaryCheckpointModel(_TinyCheckpointModel):
    def __init__(self, config: _TinyCheckpointConfig) -> None:
        super().__init__(config)
        self.router_scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, input_ids=None, labels=None, **kwargs):
        _ = labels, kwargs
        hidden = self.emb(input_ids)
        auxiliary_loss = self.router_scale * input_ids[:, 0].to(
            dtype=self.router_scale.dtype
        ).mean()
        return _AuxiliaryOutput(self.fc(hidden), auxiliary_loss)


class _AuxiliaryPolicy(TrainingObjectivePolicy):
    def auxiliary_loss_names(self):
        return ("router_aux_loss",)

    def prepare_sft_forward_inputs(self, *, model, inputs):
        _ = model
        prepared = dict(inputs)
        prepared["output_router_logits"] = True
        return prepared

    def resolve_sft_auxiliary_loss_terms(self, *, model, outputs, inputs):
        _ = model, inputs
        return (
            ShaftAuxiliaryLossTerm(
                name="router_aux_loss",
                value=outputs.auxiliary_loss,
                coefficient=0.25,
            ),
        )

    def resolve_sft_eval_auxiliary_statistics(self, *, model, outputs, inputs):
        _ = model
        batch_size = int(inputs["input_ids"].shape[0])
        values = outputs.auxiliary_loss.detach().reshape(1, 1).expand(
            batch_size,
            1,
        )
        return (
            ShaftEvalAuxiliaryStatistic(
                name="router_aux_mean",
                coefficient=0.25,
                coefficient_key="router_aux_loss",
                components={
                    "sum": values,
                    "count": torch.ones_like(values),
                },
            ),
        )

    def finalize_sft_eval_auxiliary_statistics(self, statistics):
        assert len(statistics) == 1
        statistic = statistics[0]
        self.last_finalized_coefficient = float(statistic.coefficient)
        value = statistic.components["sum"].sum() / statistic.components[
            "count"
        ].sum()
        return (
            ShaftEvalAuxiliaryMetric(
                name=statistic.name,
                value=value,
                coefficient_key=statistic.coefficient_key,
                coefficient=statistic.coefficient,
            ),
        )


class _AuxiliaryAdapter:
    def __init__(self) -> None:
        self.policy = _AuxiliaryPolicy()
        self.module_groups = ModelModuleGroups()

    def prepare_sft_forward_inputs(self, *, model, inputs):
        return self.policy.prepare_sft_forward_inputs(model=model, inputs=inputs)

    def auxiliary_loss_names(self):
        return self.policy.auxiliary_loss_names()

    def resolve_sft_auxiliary_loss_terms(self, *, model, outputs, inputs):
        return self.policy.resolve_sft_auxiliary_loss_terms(
            model=model,
            outputs=outputs,
            inputs=inputs,
        )

    def resolve_sft_eval_auxiliary_statistics(self, *, model, outputs, inputs):
        return self.policy.resolve_sft_eval_auxiliary_statistics(
            model=model,
            outputs=outputs,
            inputs=inputs,
        )

    def finalize_sft_eval_auxiliary_statistics(self, statistics):
        return self.policy.finalize_sft_eval_auxiliary_statistics(statistics)


class _UndeclaredAuxiliaryPolicy(_AuxiliaryPolicy):
    def auxiliary_loss_names(self):
        return ()


class _DuplicateAuxiliaryPolicy(_AuxiliaryPolicy):
    def resolve_sft_auxiliary_loss_terms(self, *, model, outputs, inputs):
        terms = super().resolve_sft_auxiliary_loss_terms(
            model=model,
            outputs=outputs,
            inputs=inputs,
        )
        return (*terms, *terms)


class _UndeclaredEvalCoefficientPolicy(_AuxiliaryPolicy):
    def resolve_sft_eval_auxiliary_statistics(self, *, model, outputs, inputs):
        statistics = super().resolve_sft_eval_auxiliary_statistics(
            model=model,
            outputs=outputs,
            inputs=inputs,
        )
        statistic = statistics[0]
        return (
            ShaftEvalAuxiliaryStatistic(
                name=statistic.name,
                coefficient_key="other_auxiliary_loss",
                coefficient=statistic.coefficient,
                components=statistic.components,
            ),
        )


class _FixedPreferenceModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, input_ids=None, labels=None, **kwargs):
        _ = labels, kwargs
        batch, sequence = input_ids.shape
        logits = torch.zeros(
            (batch, sequence, 3),
            dtype=self.anchor.dtype,
            device=self.anchor.device,
        )
        logits[..., 0] = 5.0 + self.anchor * 0.0
        return {"logits": logits}


def test_sft_floating_point_ops_uses_pre_wrap_parameter_count(tmp_path: Path) -> None:
    model = _TinyCheckpointModel(_TinyCheckpointConfig())
    trainer = ShaftSFTTrainer(
        model=model,
        args=build_training_args(tmp_path),
    )
    inputs = {"input_ids": torch.ones((2, 3), dtype=torch.long)}

    expected = 6 * inputs["input_ids"].numel() * (16 * 8 + 16)
    assert trainer.floating_point_ops(inputs) == expected

    # ZeRO-3 may temporarily change the live logical parameter view after a
    # gathered checkpoint save. FLOP telemetry must retain its construction-time
    # model-size denominator across save/resume boundaries.
    model.fc.weight.ds_numel = 29
    assert trainer.floating_point_ops(inputs) == expected


def test_sft_floating_point_ops_counts_deepspeed_placeholders(tmp_path: Path) -> None:
    trainer = ShaftSFTTrainer(
        model=_DeepSpeedPlaceholderModel(),
        args=build_training_args(tmp_path),
    )
    inputs = {"input_ids": torch.ones((2, 3), dtype=torch.long)}

    # The embedding placeholder is excluded exactly like HF; ds_numel and
    # ds_shape retain the two non-embedding parameters' full shapes.
    assert trainer.floating_point_ops(inputs) == 6 * 6 * (23 + 5)


def test_fixed_sft_commits_after_efficiency_on_save_and_is_resolvable(
    tmp_path,
) -> None:
    class _LateTelemetryCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            _ = kwargs
            checkpoint = Path(args.output_dir) / f"checkpoint-{int(state.global_step)}"
            (checkpoint / "late_telemetry.json").write_text(
                json.dumps({"global_step": int(state.global_step)}),
                encoding="utf-8",
            )
            return control

    monitor = ShaftTrainingEfficiencyMonitor(
        output_dir=tmp_path,
        device_timing=False,
    )
    stats = ShaftCollatedBatchStats(
        logical_segments=1,
        physical_packs=1,
        useful_tokens=3,
        materialized_tokens=3,
        supervised_tokens=2,
        weighted_supervision_mass=2.0,
        sequence_length_sum=3,
        sequence_length_square_sum=9,
        vision_patches=None,
    )

    def collate(rows):
        return {
            "input_ids": torch.tensor([row["input_ids"] for row in rows]),
            "labels": torch.tensor([row["labels"] for row in rows]),
            "_shaft_batch_stats": stats,
        }

    trainer = ShaftSFTTrainer(
        model=_TinyCheckpointModel(_TinyCheckpointConfig()),
        args=build_training_args(
            output_dir=tmp_path,
            max_steps=1,
            save_strategy="steps",
            save_steps=1,
            save_total_limit=1,
            logging_strategy="no",
            disable_tqdm=True,
            remove_unused_columns=False,
        ),
        train_dataset=[{"input_ids": [1, 2, 3], "labels": [1, 2, 3]}],
        data_collator=collate,
        efficiency_monitor=monitor,
        callbacks=[ShaftTrainingEfficiencyCallback(monitor)],
    )
    trainer.add_callback(_LateTelemetryCallback())

    trainer.train()

    checkpoint = tmp_path / "checkpoint-1"
    manifest = validate_training_checkpoint_commit(checkpoint)
    assert "shaft_training_efficiency_checkpoint_transaction.json" in manifest["artifacts"]
    assert "shaft_training_efficiency_rank0.json" in manifest["artifacts"]
    assert "late_telemetry.json" in manifest["artifacts"]
    assert resolve_resume_checkpoint(
        checkpoint,
        protocol=ShaftCheckpointProtocol.COMMITTED_MANIFEST,
    ) == str(checkpoint)


def test_model_only_periodic_checkpoint_is_deployable_init_only_snapshot(
    tmp_path: Path,
) -> None:
    model = _TinyCheckpointModel(_TinyCheckpointConfig())
    trainer = ShaftSFTTrainer(
        model=model,
        shaft_max_shard_size="1KB",
        args=build_training_args(
            output_dir=tmp_path,
            max_steps=1,
            save_strategy="steps",
            save_steps=1,
            save_total_limit=2,
            save_only_model=True,
            logging_strategy="no",
            disable_tqdm=True,
            remove_unused_columns=False,
        ),
        train_dataset=[{"input_ids": [1, 2, 3], "labels": [1, 2, 3]}],
        data_collator=lambda rows: {
            "input_ids": torch.tensor([row["input_ids"] for row in rows]),
            "labels": torch.tensor([row["labels"] for row in rows]),
        },
    )

    trainer.train()

    checkpoint = tmp_path / "checkpoint-1"
    snapshot = validate_model_only_checkpoint(checkpoint)
    assert snapshot["global_step"] == 1
    assert snapshot["kind"] == "full"
    assert (checkpoint / MODEL_ONLY_CHECKPOINT_COMMIT_FILENAME).is_file()
    assert not (checkpoint / "optimizer.pt").exists()
    assert not (checkpoint / "scheduler.pt").exists()
    assert not (checkpoint / "scaler.pt").exists()
    assert not list(checkpoint.glob("rng_state*.pth"))
    assert not list(checkpoint.glob("global_step*"))
    assert (checkpoint / "model.safetensors.index.json").is_file()
    assert len(list(checkpoint.glob("model-*-of-*.safetensors"))) > 1
    assert "save_pretrained" not in model.__dict__
    ensure_hf_export_layout(checkpoint, finetune_mode="full")
    restored = _TinyCheckpointModel.from_pretrained(checkpoint)
    for name, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name])
    with pytest.raises(ValueError, match="model-only.*cannot.*resume"):
        resolve_resume_checkpoint(
            checkpoint,
            protocol=ShaftCheckpointProtocol.COMMITTED_MANIFEST,
        )


def _assert_nested_state_equal(expected, actual) -> None:
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert torch.equal(expected, actual)
        return
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert expected.keys() == actual.keys()
        for key in expected:
            _assert_nested_state_equal(expected[key], actual[key])
        return
    if isinstance(expected, (list, tuple)):
        assert isinstance(actual, type(expected))
        assert len(expected) == len(actual)
        for expected_item, actual_item in zip(expected, actual, strict=True):
            _assert_nested_state_equal(expected_item, actual_item)
        return
    assert expected == actual


def test_fixed_sft_checkpoint_actual_resume_matches_uninterrupted_step_four(
    tmp_path: Path,
) -> None:
    class _RecordingTrainer(ShaftSFTTrainer):
        def __init__(self, *args, **kwargs) -> None:
            self.step_losses: list[float] = []
            super().__init__(*args, **kwargs)

        def training_step(self, model, inputs, num_items_in_batch=None):
            loss = super().training_step(model, inputs, num_items_in_batch)
            self.step_losses.append(float(loss.detach()))
            return loss

    torch.manual_seed(31)
    initial_model = _TinyCheckpointModel(_TinyCheckpointConfig())
    initial_state = deepcopy(initial_model.state_dict())
    rows = [
        {"input_ids": [1, 2, 3], "labels": [1, 2, 3]},
        {"input_ids": [3, 2, 1], "labels": [3, 2, 1]},
    ]

    def collate(batch):
        return {
            "input_ids": torch.tensor([row["input_ids"] for row in batch]),
            "labels": torch.tensor([row["labels"] for row in batch]),
        }

    uninterrupted_model = _TinyCheckpointModel(_TinyCheckpointConfig())
    uninterrupted_model.load_state_dict(initial_state)
    uninterrupted = _RecordingTrainer(
        model=uninterrupted_model,
        args=build_training_args(
            output_dir=tmp_path / "uninterrupted",
            max_steps=4,
            save_strategy="steps",
            save_steps=2,
            save_total_limit=2,
            gradient_accumulation_steps=2,
            logging_strategy="no",
            disable_tqdm=True,
            remove_unused_columns=False,
        ),
        train_dataset=rows,
        data_collator=collate,
    )
    uninterrupted.train()

    checkpoint_two = tmp_path / "uninterrupted" / "checkpoint-2"
    resolved_checkpoint = resolve_resume_checkpoint(
        checkpoint_two,
        protocol=ShaftCheckpointProtocol.COMMITTED_MANIFEST,
    )
    expected_model_state = deepcopy(uninterrupted.model.state_dict())
    expected_optimizer_state = deepcopy(uninterrupted.optimizer.state_dict())
    expected_scheduler_state = deepcopy(uninterrupted.lr_scheduler.state_dict())

    resumed_model = _TinyCheckpointModel(_TinyCheckpointConfig())
    resumed = ShaftSFTTrainer(
        model=resumed_model,
        args=build_training_args(
            output_dir=tmp_path / "resumed",
            max_steps=4,
            gradient_accumulation_steps=2,
            save_strategy="no",
            logging_strategy="no",
            disable_tqdm=True,
            remove_unused_columns=False,
        ),
        train_dataset=rows,
        data_collator=collate,
    )
    resumed_result = resumed.train(resume_from_checkpoint=resolved_checkpoint)

    assert resumed.state.global_step == 4
    expected_resume_window_loss = sum(uninterrupted.step_losses[-4:]) / 2.0
    assert resumed_result.metrics["train_loss"] == pytest.approx(
        expected_resume_window_loss
    )
    assert resumed_result.training_loss == pytest.approx(
        resumed_result.metrics["train_loss"]
    )
    runtime = float(resumed_result.metrics["train_runtime"])
    assert resumed_result.metrics["train_steps_per_second"] == pytest.approx(
        round(2.0 / runtime, 3)
    )
    assert resumed_result.metrics["train_samples_per_second"] == pytest.approx(
        round(4.0 / runtime, 3)
    )
    _assert_nested_state_equal(expected_model_state, resumed.model.state_dict())
    _assert_nested_state_equal(expected_optimizer_state, resumed.optimizer.state_dict())
    _assert_nested_state_equal(expected_scheduler_state, resumed.lr_scheduler.state_dict())


def test_sft_exact_resume_preserves_unflushed_loss_and_auxiliary_log_window(
    tmp_path: Path,
) -> None:
    class _LogCapture(TrainerCallback):
        def __init__(self) -> None:
            self.events: list[tuple[int, dict[str, float]]] = []

        def on_log(self, args, state, control, logs=None, **kwargs):
            _ = args, kwargs
            self.events.append((int(state.global_step), dict(logs or {})))
            return control

    torch.manual_seed(37)
    initial_model = _AuxiliaryCheckpointModel(_TinyCheckpointConfig())
    initial_state = deepcopy(initial_model.state_dict())
    rows = [
        {"input_ids": [token, 2, 3], "labels": [token, 2, 3]}
        for token in (1, 2, 3, 4)
    ]

    def collate(batch):
        return {
            "input_ids": torch.tensor([row["input_ids"] for row in batch]),
            "labels": torch.tensor([row["labels"] for row in batch]),
        }

    def build_trainer(output_dir: Path, *, save_steps: int):
        model = _AuxiliaryCheckpointModel(_TinyCheckpointConfig())
        model.load_state_dict(initial_state)
        capture = _LogCapture()
        trainer = ShaftSFTTrainer(
            model=model,
            args=build_training_args(
                output_dir=output_dir,
                max_steps=4,
                per_device_train_batch_size=1,
                save_strategy="steps",
                save_steps=save_steps,
                save_total_limit=2,
                logging_strategy="steps",
                logging_steps=4,
                disable_tqdm=True,
                remove_unused_columns=False,
                full_determinism=True,
            ),
            train_dataset=rows,
            data_collator=collate,
            model_adapter=_AuxiliaryAdapter(),
            callbacks=[capture],
        )
        return trainer, capture

    uninterrupted, fresh_capture = build_trainer(
        tmp_path / "uninterrupted-reporting",
        save_steps=2,
    )
    uninterrupted.train()
    checkpoint_two = tmp_path / "uninterrupted-reporting" / "checkpoint-2"

    resumed, resumed_capture = build_trainer(
        tmp_path / "resumed-reporting",
        save_steps=4,
    )
    resumed.train(resume_from_checkpoint=checkpoint_two)

    _assert_nested_state_equal(
        uninterrupted.model.state_dict(),
        resumed.model.state_dict(),
    )
    fresh_step = next(
        entry
        for entry in uninterrupted.state.log_history
        if entry.get("step") == 4 and "loss" in entry
    )
    resumed_step = next(
        entry
        for entry in resumed.state.log_history
        if entry.get("step") == 4 and "loss" in entry
    )
    assert resumed_step["loss"] == fresh_step["loss"]
    assert resumed_step["aux/router_aux_loss"] == fresh_step[
        "aux/router_aux_loss"
    ]
    assert resumed_step["aux/router_aux_loss_weighted"] == fresh_step[
        "aux/router_aux_loss_weighted"
    ]
    event_keys = (
        "loss",
        "aux/router_aux_loss",
        "aux/router_aux_loss_weighted",
    )
    fresh_event = next(
        payload
        for step, payload in fresh_capture.events
        if step == 4 and "loss" in payload
    )
    resumed_event = next(
        payload
        for step, payload in resumed_capture.events
        if step == 4 and "loss" in payload
    )
    assert {key: resumed_event[key] for key in event_keys} == {
        key: fresh_event[key] for key in event_keys
    }


def test_sft_reporting_state_rejects_hf_callback_replacement(tmp_path: Path) -> None:
    with pytest.raises(
        ValueError,
        match="restore_callback_states_from_checkpoint=True",
    ):
        ShaftSFTTrainer(
            model=_TinyCheckpointModel(_TinyCheckpointConfig()),
            args=build_training_args(
                output_dir=tmp_path / "callback-replacement",
                restore_callback_states_from_checkpoint=True,
            ),
        )


def test_sft_reporting_snapshot_rejects_incomplete_rank_state() -> None:
    snapshot = {
        "version": "shaft-sft-reporting-state-v1",
        "global_step": 2,
        "world_size": 2,
        "ranks": [
            {
                "rank": 0,
                "global_step": 2,
                "tr_loss": 1.0,
                "total_loss_scalar": 0.0,
                "globalstep_last_logged": 0,
                "auxiliary": {},
            }
        ],
    }

    with pytest.raises(ValueError, match="does not contain every rank"):
        ShaftSFTTrainer._normalize_reporting_snapshot(
            snapshot,
            expected_global_step=2,
            expected_world_size=2,
        )


def test_sft_noop_resume_preserves_pending_final_train_loss(tmp_path: Path) -> None:
    rows = [
        {"input_ids": [token, 2, 3], "labels": [token, 2, 3]}
        for token in (1, 2)
    ]

    def collate(batch):
        return {
            "input_ids": torch.tensor([row["input_ids"] for row in batch]),
            "labels": torch.tensor([row["labels"] for row in batch]),
        }

    torch.manual_seed(41)
    initial = _TinyCheckpointModel(_TinyCheckpointConfig()).state_dict()

    def build(output_dir: Path, *, save_strategy: str):
        model = _TinyCheckpointModel(_TinyCheckpointConfig())
        model.load_state_dict(initial)
        return ShaftSFTTrainer(
            model=model,
            args=build_training_args(
                output_dir=output_dir,
                max_steps=2,
                save_strategy=save_strategy,
                save_steps=2,
                logging_strategy="steps",
                logging_steps=4,
                disable_tqdm=True,
                remove_unused_columns=False,
                full_determinism=True,
            ),
            train_dataset=rows,
            data_collator=collate,
        )

    fresh = build(tmp_path / "noop-fresh", save_strategy="steps")
    fresh_result = fresh.train()
    resumed = build(tmp_path / "noop-resumed", save_strategy="no")
    resumed_result = resumed.train(
        resume_from_checkpoint=tmp_path / "noop-fresh" / "checkpoint-2"
    )

    assert resumed.state.global_step == 2
    assert resumed_result.metrics["train_loss"] == fresh_result.metrics["train_loss"]


def test_sft_pending_reporting_loss_preserves_hf_nonfinite_filter_state(
    tmp_path: Path,
) -> None:
    trainer = ShaftSFTTrainer(
        model=_TinyCheckpointModel(_TinyCheckpointConfig()),
        args=build_training_args(output_dir=tmp_path / "nonfinite-reporting"),
    )
    trainer.state.global_step = 2
    trainer._globalstep_last_logged = 0
    trainer._total_loss_scalar = 4.0
    trainer._shaft_pending_reporting_loss = 4.0
    trainer._shaft_pending_total_provisional = 4.0

    with patch.object(
        Trainer,
        "training_step",
        return_value=torch.tensor(float("nan")),
    ):
        filtered = trainer.training_step(trainer.model, {})

    assert float(filtered) == pytest.approx(4.0 + 4.0 / 3.0)
    assert trainer._shaft_pending_reporting_loss is None
    assert trainer._shaft_pending_total_provisional is None
    assert trainer._total_loss_scalar == 0.0


def test_fixed_sft_epoch_checkpoint_actual_resume_preserves_cycle_and_state(
    tmp_path: Path,
) -> None:
    class _CycleAwareDataset:
        def __init__(self) -> None:
            self.seen: list[tuple[int, int, int]] = []

        def __len__(self) -> int:
            return 2

        def __getitem__(self, ref: ShaftSampleRef) -> dict[str, list[int]]:
            assert isinstance(ref, ShaftSampleRef)
            self.seen.append(
                (
                    ref.context.plan_cycle,
                    ref.context.draw_id,
                    ref.row_index,
                )
            )
            token = 1 + int(ref.context.transform_seed % 14)
            next_token = 1 + token % 14
            return {
                "input_ids": [token, next_token],
                "labels": [token, next_token],
            }

    def _collate(rows):
        return {
            "input_ids": torch.tensor(
                [row["input_ids"] for row in rows],
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                [row["labels"] for row in rows],
                dtype=torch.long,
            ),
        }

    def _rng_snapshot():
        numpy_state = np.random.get_state()
        return (
            random.getstate(),
            (
                numpy_state[0],
                numpy_state[1].copy(),
                numpy_state[2],
                numpy_state[3],
                numpy_state[4],
            ),
            torch.get_rng_state().clone(),
        )

    plan = ShaftSamplePlan(
        {"a": 2},
        {"a": 1.0},
        strategy="concat",
        shuffle=True,
        seed=73,
    )
    torch.manual_seed(31)
    initial_model = _TinyCheckpointModel(_TinyCheckpointConfig())
    initial_state = deepcopy(initial_model.state_dict())
    uninterrupted_dataset = _CycleAwareDataset()
    uninterrupted_model = _TinyCheckpointModel(_TinyCheckpointConfig())
    uninterrupted_model.load_state_dict(initial_state)
    uninterrupted = ShaftSFTTrainer(
        model=uninterrupted_model,
        args=build_training_args(
            output_dir=tmp_path / "epoch-uninterrupted",
            num_train_epochs=2,
            save_strategy="steps",
            save_steps=1,
            save_total_limit=4,
            logging_strategy="no",
            disable_tqdm=True,
            remove_unused_columns=False,
            seed=73,
            data_seed=73,
        ),
        train_dataset=uninterrupted_dataset,
        train_sampler=ShaftSampleSampler(plan, rank=0, world_size=1),
        data_collator=_collate,
    )
    uninterrupted.train()

    checkpoint_one = tmp_path / "epoch-uninterrupted" / "checkpoint-1"
    resolved_checkpoint = resolve_resume_checkpoint(
        checkpoint_one,
        protocol=ShaftCheckpointProtocol.COMMITTED_MANIFEST,
    )
    expected_model_state = deepcopy(uninterrupted.model.state_dict())
    expected_optimizer_state = deepcopy(uninterrupted.optimizer.state_dict())
    expected_scheduler_state = deepcopy(uninterrupted.lr_scheduler.state_dict())
    expected_rng_state = _rng_snapshot()
    resumed_dataset = _CycleAwareDataset()
    resumed = ShaftSFTTrainer(
        model=_TinyCheckpointModel(_TinyCheckpointConfig()),
        args=build_training_args(
            output_dir=tmp_path / "epoch-resumed",
            num_train_epochs=2,
            save_strategy="no",
            logging_strategy="no",
            disable_tqdm=True,
            remove_unused_columns=False,
            seed=73,
            data_seed=73,
        ),
        train_dataset=resumed_dataset,
        train_sampler=ShaftSampleSampler(plan, rank=0, world_size=1),
        data_collator=_collate,
    )
    resumed.train(resume_from_checkpoint=resolved_checkpoint)
    actual_rng_state = _rng_snapshot()

    assert uninterrupted.state.global_step == resumed.state.global_step == 4
    assert [cycle for cycle, _, _ in uninterrupted_dataset.seen] == [0, 0, 1, 1]
    assert [cycle for cycle, _, _ in resumed_dataset.seen] == [0, 1, 1]
    assert [draw_id for _, draw_id, _ in resumed_dataset.seen] == [1, 2, 3]
    _assert_nested_state_equal(expected_model_state, resumed.model.state_dict())
    _assert_nested_state_equal(expected_optimizer_state, resumed.optimizer.state_dict())
    _assert_nested_state_equal(expected_scheduler_state, resumed.lr_scheduler.state_dict())
    assert expected_rng_state[0] == actual_rng_state[0]
    assert expected_rng_state[1][0] == actual_rng_state[1][0]
    assert np.array_equal(expected_rng_state[1][1], actual_rng_state[1][1])
    assert expected_rng_state[1][2:] == actual_rng_state[1][2:]
    assert torch.equal(expected_rng_state[2], actual_rng_state[2])


def test_efficiency_cuda_lifecycle_is_wired_around_trainer_and_optimizer() -> None:
    operations: list[str] = []

    class _SpyMonitor:
        enabled = True
        device_timing = True

        def start_device_frame(self) -> None:
            operations.append("device-start")

        def record_training_step(self, seconds: float) -> None:
            assert seconds >= 0
            operations.append("training-finished")

        def start_optimizer_step(self) -> None:
            operations.append("optimizer-start")

        def finish_optimizer_step(self) -> None:
            operations.append("optimizer-finished")

    model = _TinyModel()
    args = build_training_args(output_dir="/tmp/shaft_efficiency_lifecycle")
    monitor = _SpyMonitor()
    trainer = ShaftSFTTrainer(
        model=model,
        args=args,
        train_dataset=[],
        data_collator=lambda rows: rows,
        efficiency_monitor=monitor,
    )
    trainer.args._setup_devices = torch.device("cuda")

    def _parent_training_step(*args, **kwargs):
        del args, kwargs
        operations.append("forward-backward")
        return torch.tensor(1.0)

    with patch("transformers.Trainer.training_step", _parent_training_step):
        trainer.training_step(model, {"input_ids": torch.tensor([[1]])})

    callback = ShaftTrainingEfficiencyCallback(monitor)  # type: ignore[arg-type]
    control = TrainerControl()
    state = TrainerState(global_step=0)
    callback.on_pre_optimizer_step(trainer.args, state, control)
    callback.on_optimizer_step(trainer.args, state, control)

    assert operations == [
        "device-start",
        "forward-backward",
        "training-finished",
        "optimizer-start",
        "optimizer-finished",
    ]


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
                model_adapter=None,
                param_group_lrs={},
            ),
        )
        trainer.create_optimizer()
        mocked_build_optim.assert_called_once()
        _, kwargs = mocked_build_optim.call_args
        assert kwargs["param_group_lrs"] == {}
        assert kwargs["model_adapter"] is None
        assert "finetune_plan" not in kwargs
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


def test_sft_trainer_combines_model_owned_auxiliary_objective_once() -> None:
    model = _AuxiliaryModel()
    adapter = _AuxiliaryAdapter()
    trainer = ShaftSFTTrainer(
        model=model,
        args=build_training_args(
            output_dir="/tmp/shaft_trainer_auxiliary_objective",
            gradient_accumulation_steps=2,
        ),
        train_dataset=[],
        data_collator=lambda rows: rows,
        model_adapter=adapter,
        loss_name="auto",
    )
    trainer.current_gradient_accumulation_steps = 2
    inputs = {
        "input_ids": torch.tensor([[0, 1, 2]], dtype=torch.long),
        "labels": torch.tensor([[0, 1, 2]], dtype=torch.long),
    }

    loss = trainer.compute_loss(model, inputs, num_items_in_batch=2)
    expected_causal_ce = torch.log(torch.tensor(4.0))
    expected_auxiliary = model.router_scale * 0.25 / 2.0

    torch.testing.assert_close(loss, expected_causal_ce + expected_auxiliary)
    loss.backward()
    assert model.router_scale.grad == pytest.approx(0.125)
    assert model.last_forward_kwargs["output_router_logits"] is True
    assert model.last_forward_labels is None
    trainer.log({"loss": float(loss.detach())})
    assert trainer.state.log_history[-1]["aux/router_aux_loss"] == pytest.approx(2.0)
    assert trainer.state.log_history[-1]["aux/router_aux_loss_weighted"] == pytest.approx(
        0.5
    )


def test_sft_trainer_overrides_declared_auxiliary_loss_weight_consistently() -> None:
    model = _AuxiliaryModel()
    adapter = _AuxiliaryAdapter()
    trainer = ShaftSFTTrainer(
        model=model,
        args=build_training_args(
            output_dir="/tmp/shaft_trainer_auxiliary_weight_override",
            gradient_accumulation_steps=2,
        ),
        train_dataset=[],
        data_collator=lambda rows: rows,
        model_adapter=adapter,
        loss_name="auto",
        auxiliary_loss_weights={"router_aux_loss": 0.5},
    )
    trainer.current_gradient_accumulation_steps = 2
    inputs = {
        "input_ids": torch.tensor([[0, 1, 2]], dtype=torch.long),
        "labels": torch.tensor([[0, 1, 2]], dtype=torch.long),
    }

    model.train()
    train_loss = trainer.compute_loss(model, inputs, num_items_in_batch=2)
    expected_causal_ce = torch.log(torch.tensor(4.0))
    torch.testing.assert_close(
        train_loss,
        expected_causal_ce + model.router_scale * 0.5 / 2.0,
    )
    trainer.log({"loss": float(train_loss.detach())})
    assert trainer.state.log_history[-1]["aux/router_aux_loss"] == pytest.approx(2.0)
    assert trainer.state.log_history[-1]["aux/router_aux_loss_weighted"] == pytest.approx(
        1.0
    )

    trainer._begin_eval_objective_collection()
    model.eval()
    eval_loss = trainer.compute_loss(model, inputs, num_items_in_batch=2)
    eval_metrics = trainer._finish_eval_objective_collection(metric_key_prefix="eval")
    torch.testing.assert_close(eval_loss, expected_causal_ce)
    assert eval_metrics["eval_aux/router_aux_mean"] == pytest.approx(2.0)
    assert eval_metrics["eval_aux/router_aux_mean_weighted"] == pytest.approx(1.0)
    assert adapter.policy.last_finalized_coefficient == pytest.approx(0.25)


def test_sft_trainer_rejects_unknown_auxiliary_loss_weight() -> None:
    with pytest.raises(ValueError, match="Unknown SFT auxiliary loss weight.*typo"):
        ShaftSFTTrainer(
            model=_AuxiliaryModel(),
            args=build_training_args(
                output_dir="/tmp/shaft_trainer_unknown_auxiliary_weight",
            ),
            train_dataset=[],
            data_collator=lambda rows: rows,
            model_adapter=_AuxiliaryAdapter(),
            auxiliary_loss_weights={"typo": 0.5},
        )


@pytest.mark.parametrize("value", [[], "", False])
def test_sft_trainer_rejects_falsey_non_mapping_auxiliary_weights(value) -> None:
    with pytest.raises(TypeError, match="must be a mapping"):
        ShaftSFTTrainer(
            model=_AuxiliaryModel(),
            args=build_training_args(
                output_dir="/tmp/shaft_trainer_invalid_auxiliary_weight_shape",
            ),
            train_dataset=[],
            data_collator=lambda rows: rows,
            model_adapter=_AuxiliaryAdapter(),
            auxiliary_loss_weights=value,
        )


@pytest.mark.parametrize(
    ("policy", "message"),
    [
        (_UndeclaredAuxiliaryPolicy(), "undeclared auxiliary loss term"),
        (_DuplicateAuxiliaryPolicy(), "duplicate auxiliary loss term"),
    ],
)
def test_sft_trainer_rejects_invalid_model_auxiliary_term_contract(
    policy,
    message: str,
) -> None:
    model = _AuxiliaryModel()
    adapter = _AuxiliaryAdapter()
    adapter.policy = policy
    trainer = ShaftSFTTrainer(
        model=model,
        args=build_training_args(
            output_dir="/tmp/shaft_trainer_invalid_auxiliary_term_contract",
        ),
        train_dataset=[],
        data_collator=lambda rows: rows,
        model_adapter=adapter,
    )
    inputs = {
        "input_ids": torch.tensor([[0, 1, 2]], dtype=torch.long),
        "labels": torch.tensor([[0, 1, 2]], dtype=torch.long),
    }

    with pytest.raises(ValueError, match=message):
        trainer.compute_loss(model, inputs, num_items_in_batch=2)


def test_eval_auxiliary_coefficient_key_must_be_explicitly_canonical() -> None:
    with pytest.raises(ValueError, match="coefficient_key must be a canonical"):
        ShaftEvalAuxiliaryStatistic(
            name="router_balance",
            coefficient_key=" Router_Aux_Loss ",
            coefficient=0.1,
            components={"count": torch.ones((1, 1))},
        )


def test_sft_trainer_rejects_undeclared_eval_auxiliary_coefficient_key() -> None:
    model = _AuxiliaryModel()
    adapter = _AuxiliaryAdapter()
    adapter.policy = _UndeclaredEvalCoefficientPolicy()
    trainer = ShaftSFTTrainer(
        model=model,
        args=build_training_args(
            output_dir="/tmp/shaft_trainer_undeclared_eval_auxiliary_key",
        ),
        train_dataset=[],
        data_collator=lambda rows: rows,
        model_adapter=adapter,
    )
    model.eval()

    with pytest.raises(ValueError, match="undeclared coefficient_key"):
        trainer.compute_loss(
            model,
            {
                "input_ids": torch.tensor([[0, 1, 2]], dtype=torch.long),
                "labels": torch.tensor([[0, 1, 2]], dtype=torch.long),
            },
            num_items_in_batch=2,
        )


def test_fp16_overflow_log_never_persists_non_finite_trainer_state() -> None:
    args = build_training_args(output_dir="/tmp/shaft_trainer_fp16_overflow_log")
    args.fp16 = True
    trainer = ShaftSFTTrainer(
        model=_TinyModel(),
        args=args,
        train_dataset=[],
        data_collator=lambda rows: rows,
        loss_name="auto",
    )

    trainer.log({"loss": 1.25, "grad_norm": float("nan")})
    overflow_entry = trainer.state.log_history[-1]
    assert overflow_entry["loss"] == pytest.approx(1.25)
    assert overflow_entry["grad_norm_overflow"] == pytest.approx(1.0)
    assert "grad_norm" not in overflow_entry

    trainer.log({"loss": 1.0, "grad_norm": 3.5})
    finite_entry = trainer.state.log_history[-1]
    assert finite_entry["grad_norm"] == pytest.approx(3.5)
    assert "grad_norm_overflow" not in finite_entry


def test_sft_eval_auxiliary_objective_is_not_scaled_by_training_ga_or_mixed_with_train() -> None:
    model = _AuxiliaryModel()
    adapter = _AuxiliaryAdapter()
    trainer = ShaftSFTTrainer(
        model=model,
        args=build_training_args(
            output_dir="/tmp/shaft_trainer_eval_auxiliary_objective",
            gradient_accumulation_steps=5,
        ),
        train_dataset=[],
        data_collator=lambda rows: rows,
        model_adapter=adapter,
        loss_name="auto",
    )
    trainer.current_gradient_accumulation_steps = 5
    inputs = {
        "input_ids": torch.tensor([[0, 1, 2]], dtype=torch.long),
        "labels": torch.tensor([[0, 1, 2]], dtype=torch.long),
    }

    model.train()
    train_loss = trainer.compute_loss(model, inputs, num_items_in_batch=2)
    expected_causal_ce = torch.log(torch.tensor(4.0))
    torch.testing.assert_close(
        train_loss,
        expected_causal_ce + model.router_scale * 0.25 / 5.0,
    )

    trainer._begin_eval_objective_collection()
    model.eval()
    eval_loss = trainer.compute_loss(model, inputs, num_items_in_batch=2)
    torch.testing.assert_close(
        eval_loss,
        expected_causal_ce,
    )
    eval_metrics = trainer._finish_eval_objective_collection(
        metric_key_prefix="eval",
    )
    assert eval_metrics["eval_loss"] == pytest.approx(float(expected_causal_ce))
    assert eval_metrics["eval_aux/router_aux_mean"] == pytest.approx(2.0)
    assert eval_metrics["eval_aux/router_aux_mean_weighted"] == pytest.approx(0.5)

    trainer.log(eval_metrics)
    assert "aux/router_aux_loss" not in trainer.state.log_history[-1]
    trainer.log({"loss": float(train_loss.detach())})
    assert trainer.state.log_history[-1]["aux/router_aux_loss"] == pytest.approx(2.0)


def test_sft_eval_loss_is_invariant_to_batch_partition_and_token_normalized(
    tmp_path: Path,
) -> None:
    rows = [
        {"input_ids": [2, 0, 0, 0, 0], "labels": [2, 0, 0, 0, 0]},
        {"input_ids": [2, 1], "labels": [2, 1]},
    ]

    def collate(batch):
        width = max(len(row["input_ids"]) for row in batch)
        input_ids = []
        labels = []
        for row in batch:
            padding = width - len(row["input_ids"])
            input_ids.append(row["input_ids"] + [0] * padding)
            labels.append(row["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    losses = []
    for batch_size in (1, 2):
        trainer = ShaftSFTTrainer(
            model=_FixedPreferenceModel(),
            args=build_training_args(
                output_dir=tmp_path / f"eval-batch-{batch_size}",
                per_device_eval_batch_size=batch_size,
                disable_tqdm=True,
            ),
            train_dataset=[],
            eval_dataset=rows,
            data_collator=collate,
            loss_name="auto",
        )
        losses.append(float(trainer.evaluate()["eval_loss"]))

    log_partition = torch.logsumexp(torch.tensor([5.0, 0.0, 0.0]), dim=0)
    low_loss = log_partition - 5.0
    high_loss = log_partition
    expected = float((4.0 * low_loss + high_loss) / 5.0)
    assert losses[0] == pytest.approx(expected)
    assert losses[1] == pytest.approx(expected)


def test_qwen35_moe_sft_train_objective_and_eval_metric_match_upstream() -> None:
    from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig

    torch.manual_seed(73)
    model = Qwen3_5MoeForCausalLM(
        Qwen3_5MoeTextConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            moe_intermediate_size=16,
            shared_expert_intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            layer_types=["full_attention"],
            num_experts=4,
            num_experts_per_tok=2,
            max_position_embeddings=64,
            router_aux_loss_coef=0.01,
            use_cache=False,
        )
    )
    adapter = build_model_meta("qwen35vl").resolve_adapter(
        model_name_or_path="Qwen3.5-35B-A3B"
    )
    trainer = ShaftSFTTrainer(
        model=model,
        args=build_training_args(output_dir="/tmp/shaft_qwen35_moe_objective"),
        train_dataset=[],
        data_collator=lambda rows: rows,
        model_adapter=adapter,
        loss_name="auto",
    )
    input_ids = torch.randint(0, 64, (2, 8))
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()

    model.train()
    train_loss, train_outputs = trainer.compute_loss(
        model,
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        },
        return_outputs=True,
        num_items_in_batch=labels[:, 1:].numel(),
    )
    with torch.no_grad():
        upstream_train_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_router_logits=True,
            use_cache=False,
        )
    torch.testing.assert_close(train_loss, upstream_train_outputs.loss)
    assert train_outputs.aux_loss is not None
    assert train_outputs.aux_loss.requires_grad is True
    train_loss.backward()
    router_gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if name.endswith(".mlp.gate.weight")
    ]
    assert router_gradients
    assert all(gradient is not None for gradient in router_gradients)
    assert any(bool(torch.count_nonzero(gradient).item()) for gradient in router_gradients)

    model.zero_grad(set_to_none=True)
    model.eval()
    with torch.no_grad():
        shaft_loss, shaft_outputs = trainer.compute_loss(
            model,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            },
            return_outputs=True,
            num_items_in_batch=labels[:, 1:].numel(),
        )
        upstream_eval_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_router_logits=True,
            use_cache=False,
        )

    assert shaft_outputs.aux_loss is not None
    assert float(shaft_outputs.aux_loss) > 0.0
    expected_ce = upstream_eval_outputs.loss - 0.01 * upstream_eval_outputs.aux_loss
    torch.testing.assert_close(shaft_loss, expected_ce)
    eval_metrics = trainer._finish_eval_objective_collection(
        metric_key_prefix="eval",
    )
    assert eval_metrics["eval_loss"] == pytest.approx(float(expected_ce))
    assert eval_metrics["eval_aux/router_global_balance"] == pytest.approx(
        float(upstream_eval_outputs.aux_loss)
    )
    assert eval_metrics[
        "eval_aux/router_global_balance_weighted"
    ] == pytest.approx(float(upstream_eval_outputs.aux_loss) * 0.01)


def test_qwen35_moe_eval_metrics_are_batch_partition_invariant(tmp_path: Path) -> None:
    from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig

    torch.manual_seed(79)
    base_model = Qwen3_5MoeForCausalLM(
        Qwen3_5MoeTextConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            moe_intermediate_size=16,
            shared_expert_intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            layer_types=["full_attention"],
            num_experts=4,
            num_experts_per_tok=2,
            max_position_embeddings=64,
            router_aux_loss_coef=0.01,
            use_cache=False,
        )
    )
    initial_state = deepcopy(base_model.state_dict())
    rows = [
        {"input_ids": [2, 7, 11, 5, 3, 13, 17]},
        {"input_ids": [2, 19, 23, 29]},
        {"input_ids": [2, 31, 37]},
    ]

    def collate(batch):
        width = max(len(row["input_ids"]) for row in batch)
        input_ids = []
        attention_mask = []
        labels = []
        for row in batch:
            padding = width - len(row["input_ids"])
            input_ids.append(row["input_ids"] + [0] * padding)
            attention_mask.append([1] * len(row["input_ids"]) + [0] * padding)
            labels.append(row["input_ids"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    metric_sets = []
    for batch_size in (1, 2, 3):
        model = Qwen3_5MoeForCausalLM(base_model.config)
        model.load_state_dict(initial_state)
        trainer = ShaftSFTTrainer(
            model=model,
            args=build_training_args(
                output_dir=tmp_path / f"moe-eval-batch-{batch_size}",
                per_device_eval_batch_size=batch_size,
                disable_tqdm=True,
            ),
            train_dataset=[],
            eval_dataset=rows,
            data_collator=collate,
            model_adapter=build_model_meta("qwen35vl").resolve_adapter(
                model_name_or_path="Qwen3.5-35B-A3B"
            ),
            loss_name="auto",
        )
        metrics = trainer.evaluate()
        metric_sets.append(
            (
                float(metrics["eval_loss"]),
                float(metrics["eval_aux/router_global_balance"]),
            )
        )

    for actual in metric_sets[1:]:
        assert actual[0] == pytest.approx(metric_sets[0][0], abs=1e-6)
        assert actual[1] == pytest.approx(metric_sets[0][1], abs=1e-6)


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

    observed_even_batches = []
    prepare = trainer.accelerator.prepare

    def capture_prepare(*args, **kwargs):
        observed_even_batches.append(trainer.accelerator.even_batches)
        return prepare(*args, **kwargs)

    with patch.object(trainer.accelerator, "prepare", side_effect=capture_prepare):
        train_dataloader = trainer.get_train_dataloader()
    assert trainer._get_train_sampler(train_dataset) is sampler
    assert train_dataloader.batch_sampler.sampler is sampler
    assert observed_even_batches == [False]
    assert trainer.accelerator.even_batches is True


def test_train_sampler_mixin_sets_plan_cycle_before_hf_wraps_resumed_loader() -> None:
    plan = ShaftSamplePlan(
        {"a": 8},
        {"a": 1.0},
        strategy="concat",
        shuffle=True,
        seed=3,
    )
    sampler = ShaftSampleSampler(plan, rank=0, world_size=1)

    class _RefDataset:
        def __len__(self):
            return len(plan)

        def __getitem__(self, ref):
            return ref

    rank_zero_batches = BatchSamplerShard(
        BatchSampler(sampler, batch_size=2, drop_last=False),
        num_processes=2,
        process_index=0,
        split_batches=False,
        even_batches=False,
    )
    train_dataloader = DataLoaderShard(
        _RefDataset(),
        batch_sampler=rank_zero_batches,
        collate_fn=lambda rows: rows,
    )

    class _HFRunEpochProbe:
        def _run_epoch(self, model, epoch, train_dataloader, *args, **kwargs):
            _ = model, args, kwargs
            # Match the HF 5.10 resume order that caused the regression: wrap the
            # distributed loader first, then call set_epoch on the extra wrapper.
            resumed = skip_first_batches(train_dataloader, 1)
            resumed.set_epoch(epoch)
            return tuple(ref for batch in resumed for ref in batch)

    class _TrainerProbe(ShaftTrainSamplerMixin, _HFRunEpochProbe):
        pass

    trainer = object.__new__(_TrainerProbe)
    trainer.train_sampler = sampler

    refs = trainer._run_epoch(None, 2, train_dataloader)

    assert {ref.context.plan_cycle for ref in refs} == {2}
    assert [ref.context.draw_id for ref in refs] == [20, 21]


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


def test_rank_local_planned_dataloader_is_not_accelerate_sharded_again() -> None:
    schedule = ShaftSampleSchedule(
        {"ds": 32},
        {"ds": 1.0},
        strategy="concat",
        shuffle=False,
        seed=7,
    )

    class _CostProvider:
        fingerprint = "fixture-cost-v1"

        def __call__(self, sample_ref):
            _ = sample_ref
            return ShaftSampleCost(
                llm_tokens=4,
                supervised_tokens=2,
                vision_patches=0,
                exact=True,
            )

    spec = ShaftBatchPlanningSpec(
        data_world_size=1,
        buffer_size=8,
        per_device_microbatch_size=2,
        max_tokens_per_microbatch=16,
        resource_budgets=(),
        seed=7,
        sample_schedule_fingerprint=schedule.fingerprint,
        cost_fingerprint=_CostProvider.fingerprint,
    )
    sampler = ShaftPlannedBatchSampler(
        schedule,
        cost_provider=_CostProvider(),
        spec=spec,
        global_microstep_count=2,
        planning_frame_size=1,
        process_index=0,
    )

    class _RefDataset:
        def __len__(self):
            return 32

        def __getitem__(self, ref):
            return ref

    trainer = ShaftSFTTrainer(
        model=_TinyModel(),
        args=build_training_args(
            output_dir="/tmp/shaft_rank_local_planned_loader",
            per_device_train_batch_size=2,
        ),
        train_dataset=_RefDataset(),
        eval_dataset=[],
        train_batch_sampler=sampler,
        data_collator=lambda rows: rows,
    )

    with patch.object(
        trainer.accelerator,
        "prepare_data_loader",
        side_effect=AssertionError("rank-local planned loader must not be prepared"),
    ):
        train_dataloader = trainer.get_train_dataloader()

    assert not isinstance(train_dataloader.batch_sampler, BatchSamplerShard)
    assert [len(batch) for batch in train_dataloader] == [2, 2]


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
