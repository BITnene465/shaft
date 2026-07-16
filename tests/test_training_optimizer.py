from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch.utils.data import DataLoader

from shaft.config import FinetuneConfig, FreezeConfig
from shaft.model import build_model_meta
from shaft.model.finetune import apply_resolved_finetune_plan
from shaft.model.finetune_plan import build_resolved_finetune_plan
from shaft.model.smoke_vlm import SmokeVLMConfig, SmokeVLMModel
from shaft.training.muon import Muon
from shaft.training.optimizer import OPTIMIZER_REGISTRY, build_optimizer
from shaft.training.optimizer_mixin import ShaftOptimizerMixin
from shaft.training.optimizer_plan import (
    ShaftOptimizerParamGroup,
    ShaftResolvedOptimizerPlan,
    build_resolved_optimizer_plan,
    summarize_resolved_optimizer_plan,
)
from shaft.training.scheduler import SCHEDULER_REGISTRY, build_scheduler
from shaft.training.sft_trainer import ShaftSFTTrainer
from tests.support.training import TinyModel as _TinyModel
from tests.support.training import build_training_args


pytestmark = pytest.mark.component


def _build_smoke_model() -> SmokeVLMModel:
    return SmokeVLMModel(SmokeVLMConfig())


def _build_smoke_adapter():
    return build_model_meta("smoke_vlm").resolve_adapter(model_name_or_path="models/smoke-vlm")


def test_optimizer_and_scheduler() -> None:
    assert OPTIMIZER_REGISTRY.has("adamw_torch")
    assert OPTIMIZER_REGISTRY.has("muon")
    assert SCHEDULER_REGISTRY.has("cosine")
    assert SCHEDULER_REGISTRY.has("cosine_with_restarts")
    assert SCHEDULER_REGISTRY.has("polynomial")
    model = _TinyModel()
    args = build_training_args(
        output_dir="/tmp/shaft_training_modules",
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
    args = build_training_args(
        output_dir="/tmp/shaft_optimizer_groups_full",
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
    args = build_training_args(
        output_dir="/tmp/shaft_optimizer_groups_no_decay_name_patterns",
        weight_decay=0.1,
    )

    baseline = build_resolved_optimizer_plan(
        model=model,
        args=args,
        finetune_plan=plan,
        model_adapter=adapter,
    )
    baseline_group = next(
        group
        for group in baseline.groups
        if any(name.endswith("embed_tokens.weight") for name in group.parameter_names)
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
        group
        for group in resolved.groups
        if any(name.endswith("embed_tokens.weight") for name in group.parameter_names)
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
    args = build_training_args(
        output_dir="/tmp/shaft_optimizer_groups_dora",
    )

    resolved = build_resolved_optimizer_plan(
        model=wrapped,
        args=args,
        finetune_plan=plan,
        model_adapter=adapter,
        param_group_lrs={"lora_params": 5e-4, "modules_to_save": 2e-4},
    )

    lora_groups = [group for group in resolved.groups if group.logical_group == "lora_params"]
    modules_to_save_groups = [
        group for group in resolved.groups if group.logical_group == "modules_to_save"
    ]
    assert lora_groups
    assert modules_to_save_groups
    assert all(group.lr == pytest.approx(5e-4) for group in lora_groups)
    assert all(group.lr == pytest.approx(2e-4) for group in modules_to_save_groups)
    assert any(
        "lora_magnitude_vector" in name for group in lora_groups for name in group.parameter_names
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
    args = build_training_args(
        output_dir="/tmp/shaft_optimizer_summary",
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
    assert any(
        group.logical_group == "lora_params" and group.lr == pytest.approx(5e-4)
        for group in summary.groups
    )
    assert any(
        group.logical_group == "modules_to_save" and group.lr == pytest.approx(2e-4)
        for group in summary.groups
    )


def test_optimizer_summary_uses_deepspeed_global_parameter_counts() -> None:
    ds_numel_param = torch.nn.Parameter(torch.empty(0), requires_grad=True)
    ds_numel_param.ds_numel = 13
    ds_shape_param = torch.nn.Parameter(torch.empty(0), requires_grad=True)
    ds_shape_param.ds_shape = (2, 3, 5)
    plan = ShaftResolvedOptimizerPlan(
        groups=(
            ShaftOptimizerParamGroup(
                logical_group="language_model",
                decay=True,
                lr=1e-5,
                weight_decay=0.03,
                parameter_names=("layer.ds_numel", "layer.ds_shape"),
                parameters=(ds_numel_param, ds_shape_param),
            ),
        )
    )

    summary = summarize_resolved_optimizer_plan(plan)

    assert summary.total_trainable_params == 43
    assert summary.groups[0].num_parameters == 43
    assert summary.groups[0].num_tensors == 2


def test_optimizer_grouping_uses_deepspeed_global_parameter_ndim() -> None:
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.empty(0), requires_grad=True)
    model.weight.ds_shape = (4, 4)
    model.bias = torch.nn.Parameter(torch.empty(0), requires_grad=True)
    model.bias.ds_shape = (4,)
    args = build_training_args(
        output_dir="/tmp/shaft_optimizer_deepspeed_ndim",
        weight_decay=0.03,
    )

    resolved = build_resolved_optimizer_plan(model=model, args=args)

    groups_by_decay = {group.decay: group for group in resolved.groups}
    assert set(groups_by_decay) == {False, True}
    assert groups_by_decay[True].parameter_names == ("weight",)
    assert groups_by_decay[True].to_optimizer_group()["weight_decay"] == pytest.approx(0.03)
    assert groups_by_decay[False].parameter_names == ("bias",)
    assert groups_by_decay[False].to_optimizer_group()["weight_decay"] == pytest.approx(0.0)


def test_optimizer_mixin_accepts_delayed_wrapped_model_and_validates_plan() -> None:
    model = _TinyModel()
    args = build_training_args(output_dir="/tmp/shaft_optimizer_delayed")
    plan = build_resolved_optimizer_plan(model=model, args=args)
    consumer = object.__new__(ShaftOptimizerMixin)
    consumer.optimizer = None
    consumer.model = model
    consumer.args = args
    consumer.optimizer_name = "adamw_torch"
    consumer.adam_beta1 = 0.9
    consumer.adam_beta2 = 0.999
    consumer.adam_epsilon = 1e-8
    consumer.finetune_plan = None
    consumer.model_adapter = None
    consumer.param_group_lrs = {}
    consumer.no_decay_name_patterns = []
    consumer.resolved_optimizer_plan = plan
    consumer.resolved_optimizer_summary = None

    with patch("shaft.training.optimizer_mixin.is_rank_zero", return_value=False):
        optimizer = consumer.create_optimizer(model=model)

    assert isinstance(optimizer, torch.optim.Optimizer)
    assert consumer.resolved_optimizer_plan.fingerprint == plan.fingerprint

    drifted = _TinyModel()
    drifted.extra = torch.nn.Parameter(torch.ones(1))
    consumer.optimizer = None
    consumer.resolved_optimizer_plan = plan
    with pytest.raises(ValueError, match="Wrapped-model optimizer plan differs"):
        consumer.create_optimizer(model=drifted)


def test_sft_trainer_hf_delayed_fsdp_optimizer_uses_wrapped_parameters(
    tmp_path,
) -> None:
    original_model = _TinyModel()
    wrapped_model = deepcopy(original_model)
    args = build_training_args(output_dir=tmp_path)
    original_plan = build_resolved_optimizer_plan(model=original_model, args=args)
    trainer = ShaftSFTTrainer(
        model=original_model,
        args=args,
        train_dataset=[],
        resolved_optimizer_plan=original_plan,
    )
    original_parameter_ids = {id(parameter) for parameter in original_model.parameters()}
    wrapped_parameter_ids = {id(parameter) for parameter in wrapped_model.parameters()}
    assert tuple(original_model.state_dict()) == tuple(wrapped_model.state_dict())
    assert original_parameter_ids.isdisjoint(wrapped_parameter_ids)

    def prepare(value):
        if value is original_model:
            return wrapped_model
        return value

    trainer.is_fsdp_enabled = True
    fsdp_plugin = SimpleNamespace(fsdp_version=1)
    with (
        patch.object(
            trainer.accelerator.state,
            "fsdp_plugin",
            fsdp_plugin,
            create=True,
        ),
        patch.object(trainer.accelerator, "prepare", side_effect=prepare),
        patch("shaft.training.optimizer_mixin.is_rank_zero", return_value=False),
    ):
        prepared_model, _ = trainer._prepare_for_training(
            max_steps=1,
            train_dataloader=DataLoader([0]),
            resume_from_checkpoint=None,
        )

    optimizer_parameter_ids = {
        id(parameter)
        for group in trainer.optimizer.param_groups
        for parameter in group["params"]
    }
    assert prepared_model is wrapped_model
    assert trainer.model is wrapped_model
    assert optimizer_parameter_ids == wrapped_parameter_ids
    assert optimizer_parameter_ids.isdisjoint(original_parameter_ids)
    assert trainer.resolved_optimizer_plan.fingerprint == original_plan.fingerprint


def test_sft_trainer_hf_delayed_fsdp_optimizer_rejects_wrapped_plan_drift(
    tmp_path,
) -> None:
    original_model = _TinyModel()
    drifted_wrapper = torch.nn.Module()
    drifted_wrapper.wrapped = deepcopy(original_model)
    args = build_training_args(output_dir=tmp_path)
    trainer = ShaftSFTTrainer(
        model=original_model,
        args=args,
        train_dataset=[],
        resolved_optimizer_plan=build_resolved_optimizer_plan(
            model=original_model,
            args=args,
        ),
    )

    def prepare(value):
        if value is original_model:
            return drifted_wrapper
        return value

    trainer.is_fsdp_enabled = True
    fsdp_plugin = SimpleNamespace(fsdp_version=1)
    with (
        patch.object(
            trainer.accelerator.state,
            "fsdp_plugin",
            fsdp_plugin,
            create=True,
        ),
        patch.object(trainer.accelerator, "prepare", side_effect=prepare),
        patch("shaft.training.optimizer_mixin.is_rank_zero", return_value=False),
        pytest.raises(ValueError, match="Wrapped-model optimizer plan differs"),
    ):
        trainer._prepare_for_training(
            max_steps=1,
            train_dataloader=DataLoader([0]),
            resume_from_checkpoint=None,
        )
