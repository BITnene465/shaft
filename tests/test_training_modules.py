from __future__ import annotations

import warnings
from unittest.mock import patch

import pytest
import torch
from shaft.algorithms.rlhf_utils import (
    build_ppo_value_and_reward_models,
    build_trl_dpo_config,
    build_trl_ppo_config,
    validate_ppo_runtime_requirements,
)
from shaft.config import DPOConfig as ShaftDPOConfig
from shaft.config import PPOConfig as ShaftPPOConfig
from shaft.model import build_model_meta
from transformers import TrainingArguments

from shaft.training.loss import LOSS_REGISTRY, auto_loss, build_loss, causal_lm_loss
from shaft.training.muon import Muon
from shaft.training.optimizer import OPTIMIZER_REGISTRY, build_optimizer
from shaft.training.rlhf import ShaftDPOTrainer, ShaftPPOTrainer
from shaft.training.scheduler import SCHEDULER_REGISTRY, build_scheduler
from shaft.training.trainer import ShaftSFTTrainer


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

    def forward(self, input_ids=None, labels=None, **kwargs):
        _ = kwargs
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
    device = next(model.parameters()).device
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3]], device=device),
        "labels": torch.tensor([[1, 2, 3]], device=device),
    }
    with patch("shaft.training.trainer.build_optimizer") as mocked_build_optim:
        mocked_build_optim.return_value = torch.optim.AdamW(model.parameters(), lr=1e-3)
        trainer.create_optimizer()
        mocked_build_optim.assert_called_once()
    with patch("shaft.training.trainer.build_scheduler") as mocked_build_sched:
        mocked_build_sched.return_value = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda _: 1.0)
        trainer.create_scheduler(10)
        mocked_build_sched.assert_called_once()
        _, kwargs = mocked_build_sched.call_args
        assert kwargs["num_cycles"] == pytest.approx(2.0)
        assert kwargs["power"] == pytest.approx(1.5)
    loss = trainer.compute_loss(model, inputs)
    assert isinstance(loss, torch.Tensor)


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


def test_shaft_rlhf_trainer_classes_are_importable() -> None:
    assert isinstance(ShaftDPOTrainer, type)
    assert isinstance(ShaftPPOTrainer, type)


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
