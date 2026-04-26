from __future__ import annotations

from types import SimpleNamespace
import warnings
from unittest.mock import patch

import pytest
import torch
from transformers.trainer_callback import PrinterCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.trainer_utils import IntervalStrategy, SaveStrategy
from shaft.algorithms.rlhf_utils import (
    build_trl_grpo_config,
    build_ppo_value_and_reward_models,
    build_trl_dpo_config,
    build_trl_ppo_config,
    validate_ppo_runtime_requirements,
)
from shaft.config import FinetuneConfig, FreezeConfig
from shaft.config import DPOConfig as ShaftDPOConfig
from shaft.config import GRPOConfig as ShaftGRPOConfig
from shaft.config import PPOConfig as ShaftPPOConfig
from shaft.config import GRPORewardConfig
from shaft.algorithms.grpo_rewards import build_grpo_reward_functions
from shaft.config.training import EvalConfig, EvalDatasetPolicyConfig
from shaft.model import build_model_meta
from shaft.model.finetune import apply_resolved_finetune_plan
from shaft.model.finetune_plan import build_resolved_finetune_plan
from shaft.model.smoke_vlm import SmokeVLMConfig, SmokeVLMModel
from transformers import TrainingArguments
from shaft.data import SFTRecord, SFTDataset, ShaftMixedIndexSampler

from shaft.training.loss import LOSS_REGISTRY, auto_loss, build_loss, causal_lm_cross_entropy, causal_lm_loss
from shaft.training.muon import Muon
from shaft.training.optimizer import OPTIMIZER_REGISTRY, build_optimizer
from shaft.training import ShaftDPOTrainer, ShaftGRPOTrainer, ShaftPPOTrainer
from shaft.training import ShaftEpochIntervalCallback
from shaft.training.optimizer_plan import build_resolved_optimizer_plan, summarize_resolved_optimizer_plan
from shaft.training.scheduler import SCHEDULER_REGISTRY, build_scheduler
from shaft.training.sft_trainer import ShaftSFTTrainer


class _DummyOutput:
    def __init__(self, loss: torch.Tensor | None = None, logits: torch.Tensor | None = None):
        self.loss = loss
        self.logits = logits


class _TinyModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 16):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab_size, 8)
        self.fc = torch.nn.Linear(8, vocab_size)
        self.config = type("Cfg", (), {"hidden_size": 8})()
        self.last_forward_kwargs = None

    def forward(self, input_ids=None, labels=None, **kwargs):
        self.last_forward_kwargs = dict(kwargs)
        hidden = self.emb(input_ids)
        logits = self.fc(hidden)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return _DummyOutput(loss=loss, logits=logits)


def _build_smoke_model() -> SmokeVLMModel:
    return SmokeVLMModel(SmokeVLMConfig())


def _build_smoke_adapter():
    return build_model_meta("smoke_vlm").resolve_adapter(model_name_or_path="models/smoke-vlm")


def test_loss_functions() -> None:
    assert LOSS_REGISTRY.has("auto")
    assert LOSS_REGISTRY.has("causal_lm")
    assert build_loss("auto") is auto_loss
    logits = torch.randn(2, 3, 8)
    labels = torch.tensor([[1, 2, -100], [3, 4, 5]])
    out = _DummyOutput(loss=None, logits=logits)
    loss = causal_lm_loss(outputs=out, labels=labels, ignore_index=-100)
    assert isinstance(loss, torch.Tensor)
    assert float(loss) > 0.0

    out2 = _DummyOutput(loss=torch.tensor(1.25), logits=logits)
    loss2 = auto_loss(outputs=out2, labels=labels, ignore_index=-100)
    assert float(loss2) == pytest.approx(1.25)


def test_causal_lm_cross_entropy_supports_weighted_loss_scale() -> None:
    logits = torch.tensor(
        [
            [
                [0.0, 0.0, 5.0],
                [0.0, 5.0, 0.0],
                [5.0, 0.0, 0.0],
                [5.0, 0.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([[0, 1, 2, 0]], dtype=torch.long)
    weighted = causal_lm_cross_entropy(
        logits=logits,
        labels=labels,
        loss_scale=torch.tensor([[0.0, 0.5, 1.0, 1.0]], dtype=torch.float32),
    )
    unweighted = causal_lm_cross_entropy(logits=logits, labels=labels)
    assert isinstance(weighted, torch.Tensor)
    assert isinstance(unweighted, torch.Tensor)
    assert weighted.ndim == 0
    assert unweighted.ndim == 0
    assert float(weighted) < float(unweighted)


def test_causal_lm_cross_entropy_includes_last_eos_and_shift_is_exact() -> None:
    vocab_size = 8
    labels = torch.tensor([[-100, 3, 4, 2]], dtype=torch.long)
    perfect_logits = torch.full((1, 4, vocab_size), -10.0, dtype=torch.float32)
    perfect_logits[0, 0, 3] = 10.0
    perfect_logits[0, 1, 4] = 10.0
    perfect_logits[0, 2, 2] = 10.0

    misaligned_logits = torch.full((1, 4, vocab_size), -10.0, dtype=torch.float32)
    misaligned_logits[0, 0, 4] = 10.0
    misaligned_logits[0, 1, 2] = 10.0
    misaligned_logits[0, 2, 0] = 10.0

    perfect_loss = causal_lm_cross_entropy(logits=perfect_logits, labels=labels)
    misaligned_loss = causal_lm_cross_entropy(logits=misaligned_logits, labels=labels)

    assert float(perfect_loss) < 1e-3
    assert float(misaligned_loss) > 1.0


def test_optimizer_and_scheduler() -> None:
    assert OPTIMIZER_REGISTRY.has("adamw_torch")
    assert OPTIMIZER_REGISTRY.has("muon")
    assert SCHEDULER_REGISTRY.has("cosine")
    assert SCHEDULER_REGISTRY.has("cosine_with_restarts")
    assert SCHEDULER_REGISTRY.has("polynomial")
    model = _TinyModel()
    args = TrainingArguments(
        output_dir="/tmp/shaft_training_modules",
        learning_rate=1e-3,
        per_device_train_batch_size=1,
        use_cpu=True,
        report_to=[],
    )
    optimizer = build_optimizer(
        model=model,
        args=args,
        optimizer_name="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
    )
    assert isinstance(optimizer, torch.optim.Optimizer)
    scheduler = build_scheduler(
        scheduler_name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=10,
    )
    assert scheduler is not None

    scheduler_restart = build_scheduler(
        scheduler_name="cosine_with_restarts",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=10,
        num_cycles=2.0,
    )
    assert scheduler_restart is not None

    scheduler_poly = build_scheduler(
        scheduler_name="polynomial",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=10,
        power=2.0,
    )
    assert scheduler_poly is not None

    muon = build_optimizer(
        model=model,
        args=args,
        optimizer_name="muon",
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
    )
    assert isinstance(muon, Muon)


def test_optimizer_supports_param_group_lrs_for_full_finetune() -> None:
    model = _build_smoke_model()
    adapter = _build_smoke_adapter()
    finetune = FinetuneConfig(mode="full", freeze=FreezeConfig(groups=["generator"]))
    plan = build_resolved_finetune_plan(model, finetune, model_adapter=adapter)
    apply_resolved_finetune_plan(model, plan, finetune=finetune)
    args = TrainingArguments(
        output_dir="/tmp/shaft_optimizer_groups_full",
        learning_rate=1e-3,
        per_device_train_batch_size=1,
        use_cpu=True,
        report_to=[],
        weight_decay=0.1,
    )

    resolved = build_resolved_optimizer_plan(
        model=model,
        args=args,
        finetune_plan=plan,
        model_adapter=adapter,
        param_group_lrs={"language_model": 2.5e-4},
    )

    logical_groups = {group.logical_group for group in resolved.groups}
    assert logical_groups == {"language_model"}
    assert all(group.lr == pytest.approx(2.5e-4) for group in resolved.groups)
    assert {group.weight_decay for group in resolved.groups} == {0.1, 0.0}


def test_optimizer_supports_no_decay_name_patterns() -> None:
    model = _build_smoke_model()
    adapter = _build_smoke_adapter()
    finetune = FinetuneConfig(mode="full", freeze=FreezeConfig(groups=["generator"]))
    plan = build_resolved_finetune_plan(model, finetune, model_adapter=adapter)
    apply_resolved_finetune_plan(model, plan, finetune=finetune)
    args = TrainingArguments(
        output_dir="/tmp/shaft_optimizer_groups_no_decay_name_patterns",
        learning_rate=1e-3,
        per_device_train_batch_size=1,
        use_cpu=True,
        report_to=[],
        weight_decay=0.1,
    )

    baseline = build_resolved_optimizer_plan(
        model=model,
        args=args,
        finetune_plan=plan,
        model_adapter=adapter,
    )
    baseline_group = next(
        group for group in baseline.groups if any(name.endswith("embed_tokens.weight") for name in group.parameter_names)
    )
    assert baseline_group.decay is True
    assert baseline_group.weight_decay == pytest.approx(0.1)

    resolved = build_resolved_optimizer_plan(
        model=model,
        args=args,
        finetune_plan=plan,
        model_adapter=adapter,
        no_decay_name_patterns=["embed_tokens.weight"],
    )
    embed_group = next(
        group for group in resolved.groups if any(name.endswith("embed_tokens.weight") for name in group.parameter_names)
    )
    assert embed_group.decay is False
    assert embed_group.weight_decay == pytest.approx(0.0)


def test_optimizer_supports_param_group_lrs_for_lora_and_modules_to_save() -> None:
    model = _build_smoke_model()
    adapter = _build_smoke_adapter()
    finetune = FinetuneConfig(
        mode="dora",
        target_modules=["all-linear"],
        freeze=FreezeConfig(trainable_prefixes=["lm_head"]),
    )
    plan = build_resolved_finetune_plan(model, finetune, model_adapter=adapter)
    wrapped = apply_resolved_finetune_plan(model, plan, finetune=finetune)
    args = TrainingArguments(
        output_dir="/tmp/shaft_optimizer_groups_dora",
        learning_rate=1e-3,
        per_device_train_batch_size=1,
        use_cpu=True,
        report_to=[],
    )

    resolved = build_resolved_optimizer_plan(
        model=wrapped,
        args=args,
        finetune_plan=plan,
        model_adapter=adapter,
        param_group_lrs={"lora_params": 5e-4, "modules_to_save": 2e-4},
    )

    lora_groups = [group for group in resolved.groups if group.logical_group == "lora_params"]
    modules_to_save_groups = [group for group in resolved.groups if group.logical_group == "modules_to_save"]
    assert lora_groups
    assert modules_to_save_groups
    assert all(group.lr == pytest.approx(5e-4) for group in lora_groups)
    assert all(group.lr == pytest.approx(2e-4) for group in modules_to_save_groups)
    assert any(
        "lora_magnitude_vector" in name
        for group in lora_groups
        for name in group.parameter_names
    )
    assert any(
        ".modules_to_save." in name
        for group in modules_to_save_groups
        for name in group.parameter_names
    )


def test_optimizer_summary_reports_grouped_learning_rates() -> None:
    model = _build_smoke_model()
    adapter = _build_smoke_adapter()
    finetune = FinetuneConfig(
        mode="dora",
        target_modules=["all-linear"],
        freeze=FreezeConfig(trainable_prefixes=["lm_head"]),
    )
    plan = build_resolved_finetune_plan(model, finetune, model_adapter=adapter)
    wrapped = apply_resolved_finetune_plan(model, plan, finetune=finetune)
    args = TrainingArguments(
        output_dir="/tmp/shaft_optimizer_summary",
        learning_rate=1e-3,
        per_device_train_batch_size=1,
        use_cpu=True,
        report_to=[],
    )

    resolved = build_resolved_optimizer_plan(
        model=wrapped,
        args=args,
        finetune_plan=plan,
        model_adapter=adapter,
        param_group_lrs={"lora_params": 5e-4, "modules_to_save": 2e-4},
    )
    summary = summarize_resolved_optimizer_plan(resolved)

    assert summary.total_trainable_params > 0
    assert summary.group_count == len(summary.groups)
    assert any(group.logical_group == "lora_params" and group.lr == pytest.approx(5e-4) for group in summary.groups)
    assert any(
        group.logical_group == "modules_to_save" and group.lr == pytest.approx(2e-4)
        for group in summary.groups
    )


def test_shaft_trainer_uses_custom_components() -> None:
    model = _TinyModel()
    args = TrainingArguments(
        output_dir="/tmp/shaft_trainer_smoke",
        learning_rate=1e-3,
        per_device_train_batch_size=1,
        use_cpu=True,
        report_to=[],
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
    assert not any(isinstance(callback, PrinterCallback) for callback in trainer.callback_handler.callbacks)
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
        mocked_build_sched.return_value = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda _: 1.0)
        trainer.create_scheduler(10)
        mocked_build_sched.assert_called_once()
        _, kwargs = mocked_build_sched.call_args
        assert kwargs["num_cycles"] == pytest.approx(2.0)
        assert kwargs["power"] == pytest.approx(1.5)
    loss = trainer.compute_loss(model, inputs)
    assert isinstance(loss, torch.Tensor)
    assert "loss_scale" not in (model.last_forward_kwargs or {})


def test_shaft_trainer_uses_custom_train_sampler() -> None:
    model = _TinyModel()
    args = TrainingArguments(
        output_dir="/tmp/shaft_trainer_sampler",
        per_device_train_batch_size=1,
        use_cpu=True,
        report_to=[],
    )
    records = {
        "a": [SFTRecord(image_path="/tmp/a.png", target_text="{}", dataset_name="a", sample_id="a0")],
        "b": [SFTRecord(image_path="/tmp/b.png", target_text="{}", dataset_name="b", sample_id="b0")],
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
    train_dataset = SFTDataset(records, mixed_length=len(sampler), mixed_indices=sampler.current_indices)
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
    args = TrainingArguments(
        output_dir="/tmp/shaft_trainer_eval_smoke",
        learning_rate=1e-3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        use_cpu=True,
        report_to=[],
    )

    class _FakeOnlineEvalRunner:
        def evaluate(self, trainer, *, eval_dataset, metric_key_prefix="eval"):
            _ = trainer, eval_dataset, metric_key_prefix
            return {
                "eval_final_score": 0.8,
                "eval_ds_a_exact_match": 0.7,
            }

    trainer = ShaftSFTTrainer(
        model=model,
        args=args,
        train_dataset=[],
        eval_dataset=[{"sample_id": "x"}],
        data_collator=lambda x: x,
        online_eval_runner=_FakeOnlineEvalRunner(),
    )
    trainer.get_eval_dataloader = lambda eval_dataset=None: []  # type: ignore[method-assign]
    trainer.evaluation_loop = lambda *a, **k: SimpleNamespace(metrics={"eval_loss": 0.2}, num_samples=1)  # type: ignore[method-assign]
    logged: list[dict[str, float]] = []
    trainer.log = lambda metrics, start_time=None: logged.append(dict(metrics))  # type: ignore[method-assign]
    trainer.callback_handler.on_evaluate = lambda args, state, control, metrics: control  # type: ignore[method-assign]
    metrics = trainer.evaluate()
    assert metrics["eval_loss"] == pytest.approx(0.2)
    assert metrics["eval_final_score"] == pytest.approx(0.8)
    assert metrics["eval_ds_a_exact_match"] == pytest.approx(0.7)
    assert logged == [{"eval_loss": 0.2, "eval_final_score": 0.8, "eval_ds_a_exact_match": 0.7}]


def test_shaft_trainer_evaluate_aggregates_final_loss_for_named_eval_datasets() -> None:
    model = _TinyModel()
    args = TrainingArguments(
        output_dir="/tmp/shaft_trainer_eval_named",
        learning_rate=1e-3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        use_cpu=True,
        report_to=[],
    )

    class _FakeOnlineEvalRunner:
        def evaluate(self, trainer, *, eval_dataset, metric_key_prefix="eval"):
            _ = trainer, eval_dataset, metric_key_prefix
            return {
                "eval_final_score": 0.8,
                "eval_ds_a_exact_match": 0.7,
                "eval_ds_b_exact_match": 0.9,
            }

    trainer = ShaftSFTTrainer(
        model=model,
        args=args,
        train_dataset=[],
        eval_dataset={"ds_a": [{"sample_id": "a"}], "ds_b": [{"sample_id": "b"}]},
        data_collator=lambda x: x,
        online_eval_runner=_FakeOnlineEvalRunner(),
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
        return SimpleNamespace(metrics={f"{prefix}_loss": values[prefix]}, num_samples=1)

    trainer.evaluation_loop = _fake_evaluation_loop  # type: ignore[method-assign]
    logged: list[dict[str, float]] = []
    trainer.log = lambda metrics, start_time=None: logged.append(dict(metrics))  # type: ignore[method-assign]
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
    args = TrainingArguments(
        output_dir="/tmp/shaft_trainer_eval_loss_only",
        learning_rate=1e-3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        use_cpu=True,
        report_to=[],
    )
    trainer = ShaftSFTTrainer(
        model=model,
        args=args,
        train_dataset=[],
        eval_dataset=[{"sample_id": "x"}],
        data_collator=lambda x: x,
    )
    trainer.get_eval_dataloader = lambda eval_dataset=None: []  # type: ignore[method-assign]
    trainer.evaluation_loop = lambda *a, **k: SimpleNamespace(  # type: ignore[method-assign]
        metrics={"eval_loss": 0.3, "eval_samples_per_second": 12.0},
        num_samples=1,
    )
    logged: list[dict[str, float]] = []
    trainer.log = lambda metrics, start_time=None: logged.append(dict(metrics))  # type: ignore[method-assign]
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


def test_build_trl_dpo_config_from_training_args() -> None:
    args = TrainingArguments(
        output_dir="/tmp/shaft_dpo_config_smoke",
        learning_rate=1e-3,
        per_device_train_batch_size=1,
        use_cpu=True,
        report_to=[],
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dpo_args = build_trl_dpo_config(
            train_args=args,
            rlhf_config=ShaftDPOConfig(
                beta=0.2,
                label_smoothing=0.05,
                loss_type="sigmoid",
                precompute_ref_log_probs=True,
                use_weighting=True,
            ),
        )
    assert all("push_to_hub_token" not in str(w.message) for w in caught)
    assert dpo_args.beta == pytest.approx(0.2)
    assert dpo_args.label_smoothing == pytest.approx(0.05)
    assert dpo_args.loss_type == ["sigmoid"]
    assert dpo_args.precompute_ref_log_probs is True
    assert dpo_args.use_weighting is True


def test_build_trl_ppo_config_from_training_args() -> None:
    args = TrainingArguments(
        output_dir="/tmp/shaft_ppo_config_smoke",
        learning_rate=1e-3,
        per_device_train_batch_size=1,
        use_cpu=True,
        report_to=[],
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ppo_args = build_trl_ppo_config(
            train_args=args,
            rlhf_config=ShaftPPOConfig(
                cliprange=0.2,
                cliprange_value=0.2,
                kl_coef=0.03,
                vf_coef=0.2,
                gamma=0.99,
                lam=0.95,
                whiten_rewards=True,
                response_length=64,
                temperature=0.7,
                num_ppo_epochs=2,
                num_mini_batches=1,
                local_rollout_forward_batch_size=8,
                num_sample_generations=0,
                stop_token="eos",
                train_value_backbone=False,
            ),
        )
    assert all("push_to_hub_token" not in str(w.message) for w in caught)
    assert ppo_args.cliprange == pytest.approx(0.2)
    assert ppo_args.cliprange_value == pytest.approx(0.2)
    assert ppo_args.kl_coef == pytest.approx(0.03)
    assert ppo_args.vf_coef == pytest.approx(0.2)
    assert ppo_args.response_length == 64
    assert ppo_args.temperature == pytest.approx(0.7)
    assert ppo_args.num_ppo_epochs == 2


def test_build_trl_grpo_config_from_training_args() -> None:
    args = TrainingArguments(
        output_dir="/tmp/shaft_grpo_config_smoke",
        learning_rate=1e-3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        use_cpu=True,
        report_to=[],
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        grpo_args = build_trl_grpo_config(
            train_args=args,
            rlhf_config=ShaftGRPOConfig(
                beta=0.01,
                num_generations=4,
                max_completion_length=96,
                temperature=0.8,
                top_p=0.95,
                top_k=16,
                min_p=0.05,
                repetition_penalty=1.1,
                use_vllm=False,
            ),
        )
    assert all("push_to_hub_token" not in str(w.message) for w in caught)
    assert grpo_args.beta == pytest.approx(0.01)
    assert grpo_args.num_generations == 4
    assert grpo_args.max_completion_length == 96
    assert grpo_args.temperature == pytest.approx(0.8)
    assert grpo_args.top_p == pytest.approx(0.95)
    assert grpo_args.top_k == 16
    assert grpo_args.min_p == pytest.approx(0.05)
    assert grpo_args.repetition_penalty == pytest.approx(1.1)


def test_shaft_rlhf_trainer_classes_are_importable() -> None:
    assert isinstance(ShaftDPOTrainer, type)
    assert isinstance(ShaftPPOTrainer, type)
    assert isinstance(ShaftGRPOTrainer, type)


def test_build_grpo_reward_functions_supports_exact_match_and_parse_success() -> None:
    reward_funcs = build_grpo_reward_functions(
        [
            GRPORewardConfig(name="parse_success", codec="json_any", weight=0.5),
            GRPORewardConfig(name="exact_match", codec="json_any", weight=2.0),
        ]
    )
    parse_reward, exact_reward = reward_funcs
    assert parse_reward(
        completions=['{"ok": 1}', 'not-json'],
        target_text=['{"ok": 1}', '{"ok": 0}'],
    ) == [0.5, 0.0]
    assert exact_reward(
        completions=['{"ok": 1}', '{"ok": 0}'],
        target_text=['{"ok": 1}', '{"ok": 1}'],
    ) == [2.0, 0.0]
    assert exact_reward(
        completions=[
            [{"role": "assistant", "content": [{"type": "text", "text": '{"ok": 1}'}]}],
            [{"role": "assistant", "content": '{"ok": 0}'}],
        ],
        target_text=['{"ok": 1}', '{"ok": 1}'],
    ) == [2.0, 0.0]


def test_ppo_requires_explicit_random_reward_opt_in() -> None:
    model = _TinyModel()
    with pytest.raises(ValueError, match="allow_untrained_reward_model"):
        build_ppo_value_and_reward_models(
            model=model,
            train_value_backbone=False,
            value_model_mode="shared_backbone",
            reward_model_mode="adapter_disabled_policy",
            allow_untrained_reward_model=False,
        )
    value_model, reward_model = build_ppo_value_and_reward_models(
        model=model,
        train_value_backbone=False,
        value_model_mode="copy_backbone",
        reward_model_mode="copy_backbone",
        allow_untrained_reward_model=True,
    )
    assert isinstance(value_model, torch.nn.Module)
    assert isinstance(reward_model, torch.nn.Module)


def test_ppo_multimodal_guard_requires_opt_in() -> None:
    meta = build_model_meta("smoke_vlm")
    with pytest.raises(ValueError, match="allow_text_only_multimodal_ppo"):
        validate_ppo_runtime_requirements(
            model_meta=meta,
            model=_TinyModel(),
            finetune_mode="lora",
            rlhf_config=ShaftPPOConfig(allow_text_only_multimodal_ppo=False),
        )
    validate_ppo_runtime_requirements(
        model_meta=meta,
        model=_TinyModel(),
        finetune_mode="lora",
        rlhf_config=ShaftPPOConfig(
            allow_text_only_multimodal_ppo=True,
            reward_model_mode="copy_backbone",
        ),
    )


def test_ppo_shared_value_backbone_keeps_policy_trainable() -> None:
    model = _TinyModel()
    value_model, reward_model = build_ppo_value_and_reward_models(
        model=model,
        train_value_backbone=False,
        value_model_mode="shared_backbone",
        reward_model_mode="copy_backbone",
        allow_untrained_reward_model=True,
    )
    assert value_model.backbone is not None
    assert any(param.requires_grad for param in model.parameters())
    assert all(not param.requires_grad for param in reward_model.score.parameters())


def test_ppo_rejects_full_finetune_mode() -> None:
    meta = build_model_meta("smoke_vlm")
    with pytest.raises(ValueError, match="finetune.mode='full'"):
        validate_ppo_runtime_requirements(
            model_meta=meta,
            model=_TinyModel(),
            finetune_mode="full",
            rlhf_config=ShaftPPOConfig(allow_text_only_multimodal_ppo=True),
        )
