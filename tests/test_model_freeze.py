from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from shaft.config import FinetuneConfig, FreezeConfig
from shaft.model import (
    DefaultPeftPolicy,
    ModelCapabilities,
    ModelModuleGroups,
    ProcessorPolicy,
    ShaftModelAdapter,
)
from shaft.model.finetune import apply_finetune_strategy
from shaft.model.freeze import (
    apply_full_freeze,
    build_freeze_plan,
    resolve_adapter_modules_to_save,
    resolve_adapter_target_modules,
)


class _TinyFreezeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = torch.nn.Sequential(
            torch.nn.Linear(4, 4),
            torch.nn.Linear(4, 4),
        )
        self.vision_tower = torch.nn.Sequential(torch.nn.Linear(4, 4))
        self.aligner = torch.nn.Linear(4, 4)
        self.lm_head = torch.nn.Linear(4, 4, bias=False)


def _build_adapter() -> ShaftModelAdapter:
    return ShaftModelAdapter(
        model_type="dummy",
        family="dummy",
        model_name_or_path="dummy",
        template_type="smoke_vlm",
        capabilities=ModelCapabilities(supports_pixel_budget=False, is_multimodal=True),
        module_groups=ModelModuleGroups(
            language_model=("language_model",),
            vision_tower=("vision_tower",),
            aligner=("aligner",),
            generator=("lm_head",),
        ),
        processor_policy=ProcessorPolicy(supports_pixel_budget=False),
        peft_policy=DefaultPeftPolicy(target_modules=["all-linear"]),
    )


def _trainable_parameter_names(model: torch.nn.Module) -> set[str]:
    return {name for name, parameter in model.named_parameters() if parameter.requires_grad}


def test_full_mode_freeze_group_disables_vision_tower_parameters() -> None:
    model = _TinyFreezeModel()
    adapter = _build_adapter()
    finetune = FinetuneConfig(mode="full", freeze=FreezeConfig(groups=["vision_tower"]))

    apply_finetune_strategy(model, finetune, model_adapter=adapter)

    trainable = _trainable_parameter_names(model)
    assert "vision_tower.0.weight" not in trainable
    assert "vision_tower.0.bias" not in trainable
    assert "language_model.0.weight" in trainable
    assert "aligner.weight" in trainable


def test_full_mode_trainable_prefix_override_wins_last() -> None:
    model = _TinyFreezeModel()
    adapter = _build_adapter()
    finetune = FinetuneConfig(
        mode="full",
        freeze=FreezeConfig(
            prefixes=["language_model"],
            trainable_prefixes=["language_model.1"],
        ),
    )

    apply_finetune_strategy(model, finetune, model_adapter=adapter)

    trainable = _trainable_parameter_names(model)
    assert "language_model.0.weight" not in trainable
    assert "language_model.0.bias" not in trainable
    assert "language_model.1.weight" in trainable
    assert "language_model.1.bias" in trainable


def test_full_mode_trainable_regex_override_wins_last() -> None:
    model = _TinyFreezeModel()
    adapter = _build_adapter()
    finetune = FinetuneConfig(
        mode="full",
        freeze=FreezeConfig(
            groups=["language_model", "vision_tower", "aligner", "generator"],
            trainable_regex=".*lm_head.*",
        ),
    )

    apply_finetune_strategy(model, finetune, model_adapter=adapter)

    trainable = _trainable_parameter_names(model)
    assert trainable == {"lm_head.weight"}


def test_full_mode_freeze_regex_matches_parameters() -> None:
    model = _TinyFreezeModel()
    adapter = _build_adapter()
    plan = build_freeze_plan(
        model_adapter=adapter,
        finetune=FinetuneConfig(mode="full", freeze=FreezeConfig(regex=".*aligner.*")),
    )

    apply_full_freeze(model, plan)

    trainable = _trainable_parameter_names(model)
    assert "aligner.weight" not in trainable
    assert "aligner.bias" not in trainable
    assert "language_model.0.weight" in trainable


def test_lora_all_linear_filters_frozen_groups_and_adds_modules_to_save() -> None:
    model = _TinyFreezeModel()
    adapter = _build_adapter()
    finetune = FinetuneConfig(
        mode="lora",
        target_modules=["all-linear"],
        freeze=FreezeConfig(groups=["vision_tower"], trainable_prefixes=["lm_head"]),
    )
    plan = build_freeze_plan(model_adapter=adapter, finetune=finetune)

    filtered_targets = resolve_adapter_target_modules(model, finetune.target_modules, plan=plan)
    modules_to_save = resolve_adapter_modules_to_save(model, plan=plan, target_modules=filtered_targets)

    assert isinstance(filtered_targets, list)
    assert all(not name.startswith("vision_tower") for name in filtered_targets)
    assert "language_model.0" in filtered_targets
    assert "aligner" in filtered_targets
    assert modules_to_save == ["lm_head"]


def test_lora_explicit_target_modules_remain_authoritative() -> None:
    model = _TinyFreezeModel()
    adapter = _build_adapter()
    finetune = FinetuneConfig(
        mode="lora",
        target_modules=["vision_tower.0", "aligner"],
        freeze=FreezeConfig(groups=["vision_tower"]),
    )
    plan = build_freeze_plan(model_adapter=adapter, finetune=finetune)

    filtered_targets = resolve_adapter_target_modules(model, finetune.target_modules, plan=plan)

    assert filtered_targets == ["vision_tower.0", "aligner"]


def test_lora_explicit_target_modules_do_not_raise_when_freeze_groups_overlap() -> None:
    model = _TinyFreezeModel()
    adapter = _build_adapter()
    finetune = FinetuneConfig(
        mode="lora",
        target_modules=["vision_tower.0", "aligner"],
        freeze=FreezeConfig(groups=["vision_tower", "aligner"]),
    )
    plan = build_freeze_plan(model_adapter=adapter, finetune=finetune)

    filtered_targets = resolve_adapter_target_modules(model, finetune.target_modules, plan=plan)
    assert filtered_targets == ["vision_tower.0", "aligner"]


def test_apply_finetune_strategy_passes_filtered_targets_and_modules_to_save_to_peft() -> None:
    model = _TinyFreezeModel()
    adapter = _build_adapter()
    finetune = FinetuneConfig(
        mode="lora",
        target_modules=["all-linear"],
        freeze=FreezeConfig(groups=["vision_tower"], trainable_prefixes=["lm_head"]),
    )
    captured = {}

    def _fake_get_peft_model(model, peft_config):
        captured["target_modules"] = peft_config.target_modules
        captured["modules_to_save"] = list(peft_config.modules_to_save or [])
        return model

    with patch("shaft.model.finetune.get_peft_model", side_effect=_fake_get_peft_model):
        wrapped = apply_finetune_strategy(model, finetune, model_adapter=adapter)

    assert wrapped is model
    assert "vision_tower.0" not in captured["target_modules"]
    assert "aligner" in captured["target_modules"]
    assert captured["modules_to_save"] == ["lm_head"]
