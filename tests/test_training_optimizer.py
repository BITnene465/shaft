from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
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
    canonicalize_optimizer_parameter_name,
    summarize_resolved_optimizer_plan,
    write_resolved_optimizer_summary,
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


class _ExactNamedParameterModel(torch.nn.Module):
    def __init__(self, names: list[str]) -> None:
        super().__init__()
        self._exact_names = tuple(names)
        self._exact_parameters = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.ones(2, 2)) for _ in names]
        )

    def named_parameters(self, *args, **kwargs):
        _ = args, kwargs
        return iter(zip(self._exact_names, self._exact_parameters, strict=True))


@pytest.mark.parametrize(
    ("raw_name", "expected"),
    [
        (
            "model.language_model.layers.0.self_attn.q_proj.weight",
            "model.language_model.layers.0.self_attn.q_proj.weight",
        ),
        (
            "base_model.model.model.visual.blocks.0.attn.qkv.lora_A.default.weight",
            "model.visual.blocks.0.attn.qkv.lora_A.weight",
        ),
        (
            "base_model.model.model.visual.blocks.0.attn.qkv.lora_B.weight",
            "model.visual.blocks.0.attn.qkv.lora_B.weight",
        ),
        (
            "base_model.model.model.language_model.embed_tokens."
            "lora_embedding_A.default",
            "model.language_model.embed_tokens.lora_embedding_A",
        ),
        (
            "base_model.model.model.language_model.embed_tokens.lora_embedding_B",
            "model.language_model.embed_tokens.lora_embedding_B",
        ),
        (
            "base_model.model.model.visual.blocks.0.attn.qkv."
            "lora_magnitude_vector.default.weight",
            "model.visual.blocks.0.attn.qkv.lora_magnitude_vector.weight",
        ),
        (
            "base_model.model.model.visual.blocks.0.attn.qkv.lora_magnitude_vector",
            "model.visual.blocks.0.attn.qkv.lora_magnitude_vector.weight",
        ),
        (
            "base_model.model.lm_head.modules_to_save.default.weight",
            "lm_head.weight",
        ),
        (
            "_fsdp_wrapped_module.base_model.model.model.visual.merger."
            "linear_fc1.lora_B.default.weight",
            "model.visual.merger.linear_fc1.lora_B.weight",
        ),
        (
            "base_model.model.model.language_model.layers.0.mlp.experts."
            "base_layer.lora_A.default.weight",
            "model.language_model.layers.0.mlp.experts.base_layer.lora_A.weight",
        ),
    ],
)
def test_optimizer_parameter_name_canonicalization(raw_name: str, expected: str) -> None:
    canonical = canonicalize_optimizer_parameter_name(raw_name)

    assert canonical == expected
    assert canonicalize_optimizer_parameter_name(canonical) == canonical


@pytest.mark.parametrize(
    "raw_name",
    [
        "",
        "base_model.model",
        "lm_head.modules_to_save.default",
        "model.visual.qkv.lora_A.default",
        "model.visual.qkv.lora_magnitude_vector.default",
    ],
)
def test_optimizer_parameter_name_canonicalization_rejects_incomplete_wrappers(
    raw_name: str,
) -> None:
    with pytest.raises(ValueError, match="parameter name|PEFT wrapper"):
        canonicalize_optimizer_parameter_name(raw_name)


@pytest.mark.parametrize(
    "raw_name",
    [
        "model.visual.blocks.0.attn.qkv.weight",
        "base_model.model.model.visual.blocks.0.attn.qkv.lora_A.default.weight",
        "base_model.model.model.visual.blocks.0.attn.qkv."
        "lora_magnitude_vector.default.weight",
        "base_model.model.model.visual.blocks.0.attn.qkv.lora_B.default.weight",
    ],
    ids=("full", "lora", "dora", "qlora"),
)
def test_optimizer_uses_identical_structural_grouping_for_all_finetune_modes(
    raw_name: str,
) -> None:
    model = _ExactNamedParameterModel([raw_name])
    adapter = build_model_meta("qwen36vl").resolve_adapter(
        model_name_or_path="models/Qwen3.6-27B"
    )
    args = build_training_args(output_dir="/tmp/shaft_optimizer_mode_agnostic")

    plan = build_resolved_optimizer_plan(
        model=model,
        args=args,
        model_adapter=adapter,
        param_group_lrs={"vision_tower": 3e-6},
    )

    assert {group.module_group for group in plan.groups} == {"vision_tower"}
    assert all(group.lr == pytest.approx(3e-6) for group in plan.groups)


def test_qwen36_observed_runtime_parameter_paths_resolve_all_structural_groups() -> None:
    adapter = build_model_meta("qwen36vl").resolve_adapter(
        model_name_or_path="models/Qwen3.6-27B"
    )
    expected = {
        (
            "base_model.model.model.language_model.layers.0.linear_attn."
            "in_proj_qkv.lora_A.default.weight"
        ): "language_model",
        (
            "base_model.model.model.visual.blocks.0.attn.qkv.lora_B.default.weight"
        ): "vision_tower",
        (
            "base_model.model.model.visual.merger.linear_fc1.lora_A.default.weight"
        ): "aligner",
        "base_model.model.lm_head.modules_to_save.default.weight": "generator",
    }

    resolved = {
        raw_name: adapter.module_groups.resolve_group_for_name(
            canonicalize_optimizer_parameter_name(raw_name)
        )
        for raw_name in expected
    }

    assert resolved == expected
    assert all(group is not None for group in resolved.values())


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
        model_adapter=adapter,
        param_group_lrs={"language_model": 2.5e-4},
    )

    module_groups = {group.module_group for group in resolved.groups}
    assert module_groups == {"language_model"}
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
        model_adapter=adapter,
    )
    baseline_group = next(
        group
        for group in baseline.groups
        if any(name.endswith("embed_tokens.weight") for name in group.raw_parameter_names)
    )
    assert baseline_group.decay is True
    assert baseline_group.weight_decay == pytest.approx(0.1)

    resolved = build_resolved_optimizer_plan(
        model=model,
        args=args,
        model_adapter=adapter,
        no_decay_name_patterns=["embed_tokens.weight"],
    )
    embed_group = next(
        group
        for group in resolved.groups
        if any(name.endswith("embed_tokens.weight") for name in group.raw_parameter_names)
    )
    assert embed_group.decay is False
    assert embed_group.weight_decay == pytest.approx(0.0)


def test_optimizer_groups_dora_and_modules_to_save_by_model_structure() -> None:
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
        model_adapter=adapter,
        param_group_lrs={"language_model": 5e-4, "generator": 2e-4},
    )

    language_groups = [group for group in resolved.groups if group.module_group == "language_model"]
    generator_groups = [group for group in resolved.groups if group.module_group == "generator"]
    assert language_groups
    assert generator_groups
    assert all(group.lr == pytest.approx(5e-4) for group in language_groups)
    assert all(group.lr == pytest.approx(2e-4) for group in generator_groups)
    assert any(
        "lora_magnitude_vector" in name
        for group in language_groups
        for name in group.raw_parameter_names
    )
    assert any(
        ".modules_to_save." in name
        for group in generator_groups
        for name in group.raw_parameter_names
    )


def test_optimizer_uses_longest_structural_prefix_for_aligner_lora() -> None:
    raw_name = (
        "base_model.model.model.visual.merger.linear_fc1.lora_A.default.weight"
    )
    model = _ExactNamedParameterModel([raw_name])
    adapter = build_model_meta("qwen36vl").resolve_adapter(
        model_name_or_path="models/Qwen3.6-27B"
    )
    args = build_training_args(output_dir="/tmp/shaft_optimizer_aligner")

    plan = build_resolved_optimizer_plan(
        model=model,
        args=args,
        model_adapter=adapter,
        param_group_lrs={"aligner": 4e-6},
    )

    assert {group.module_group for group in plan.groups} == {"aligner"}
    assert all(group.lr == pytest.approx(4e-6) for group in plan.groups)


def test_optimizer_rejects_unresolved_trainable_parameter_for_formal_model() -> None:
    model = _ExactNamedParameterModel(["model.unowned.weight"])
    adapter = build_model_meta("qwen36vl").resolve_adapter(
        model_name_or_path="models/Qwen3.6-27B"
    )
    args = build_training_args(output_dir="/tmp/shaft_optimizer_unresolved")

    with pytest.raises(
        ValueError,
        match=r"raw_name='model\.unowned\.weight'.*canonical_name='model\.unowned\.weight'",
    ):
        build_resolved_optimizer_plan(model=model, args=args, model_adapter=adapter)


def test_optimizer_rejects_differential_lr_without_structural_metadata() -> None:
    model = _TinyModel()
    args = build_training_args(output_dir="/tmp/shaft_optimizer_unstructured")

    with pytest.raises(ValueError, match="requires model module-group metadata"):
        build_resolved_optimizer_plan(
            model=model,
            args=args,
            param_group_lrs={"vision_tower": 3e-6},
        )


def test_optimizer_rejects_non_positive_structural_lr_at_runtime_boundary() -> None:
    model = _ExactNamedParameterModel(["model.visual.blocks.0.attn.qkv.weight"])
    adapter = build_model_meta("qwen36vl").resolve_adapter(
        model_name_or_path="models/Qwen3.6-27B"
    )
    args = build_training_args(output_dir="/tmp/shaft_optimizer_invalid_lr")

    with pytest.raises(ValueError, match="finite and positive"):
        build_resolved_optimizer_plan(
            model=model,
            args=args,
            model_adapter=adapter,
            param_group_lrs={"vision_tower": 0.0},
        )


def test_optimizer_rejects_configured_group_that_has_no_trainable_parameters() -> None:
    model = _ExactNamedParameterModel(
        ["model.language_model.layers.0.self_attn.q_proj.weight"]
    )
    adapter = build_model_meta("qwen36vl").resolve_adapter(
        model_name_or_path="models/Qwen3.6-27B"
    )
    args = build_training_args(output_dir="/tmp/shaft_optimizer_unconsumed")

    with pytest.raises(
        ValueError,
        match=(
            r"configured_group='vision_tower'.*configured_lr=3e-06.*"
            r"trainable_groups=.*language_model.*model_type='qwen36vl'"
        ),
    ):
        build_resolved_optimizer_plan(
            model=model,
            args=args,
            model_adapter=adapter,
            param_group_lrs={"vision_tower": 3e-6},
        )


def test_qwen35_moe_peft_auto_resolves_fused_experts_and_router(tmp_path: Path) -> None:
    from peft import PeftModel
    from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig

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
            use_cache=False,
        )
    )
    base_dir = tmp_path / "base"
    model.save_pretrained(base_dir)
    model = Qwen3_5MoeForCausalLM.from_pretrained(base_dir)
    adapter = build_model_meta("qwen35vl").resolve_adapter(
        model_name_or_path="Qwen3.5-35B-A3B"
    )
    finetune = FinetuneConfig(
        mode="lora",
        target_modules=[],
        target_parameters=["auto"],
        lora_r=2,
        lora_alpha=4,
    )

    plan = build_resolved_finetune_plan(model, finetune, model_adapter=adapter)

    assert plan.adapter_plan is not None
    assert plan.adapter_plan.resolved_target_modules == ()
    assert plan.adapter_plan.resolved_target_parameters == (
        "model.layers.0.mlp.experts.gate_up_proj",
        "model.layers.0.mlp.experts.down_proj",
        "model.layers.0.mlp.gate.weight",
    )
    wrapped = apply_resolved_finetune_plan(model, plan, finetune=finetune)
    trainable = {
        name: parameter
        for name, parameter in wrapped.named_parameters()
        if parameter.requires_grad
    }
    frozen = {
        name: parameter
        for name, parameter in wrapped.named_parameters()
        if not parameter.requires_grad
    }
    assert any("mlp.gate.lora_A" in name for name in trainable)
    assert any("mlp.experts" in name and "lora_A" in name for name in trainable)
    assert frozen
    assert all("lora_" in name for name in trainable)

    inputs = torch.randint(0, 64, (2, 8))
    output = wrapped(
        input_ids=inputs,
        use_cache=False,
        output_router_logits=True,
    )
    output.logits.sum().backward()
    assert all(parameter.grad is not None for parameter in trainable.values())
    torch.optim.SGD(trainable.values(), lr=0.01).step()

    adapter_dir = tmp_path / "adapter"
    wrapped.save_pretrained(adapter_dir)
    adapter_config = (adapter_dir / "adapter_config.json").read_text(encoding="utf-8")
    assert '"target_parameters"' in adapter_config
    reloaded = PeftModel.from_pretrained(
        Qwen3_5MoeForCausalLM.from_pretrained(base_dir),
        adapter_dir,
    ).eval()
    wrapped.eval()
    with torch.no_grad():
        expected_logits = wrapped(input_ids=inputs, use_cache=False).logits
        reloaded_logits = reloaded(input_ids=inputs, use_cache=False).logits
    torch.testing.assert_close(reloaded_logits, expected_logits)

    merged = reloaded.merge_and_unload(safe_merge=True).eval()
    with torch.no_grad():
        merged_logits = merged(input_ids=inputs, use_cache=False).logits
    torch.testing.assert_close(merged_logits, expected_logits)


def test_qwen3vl_moe_peft_auto_trains_and_reloads_fused_parameters(tmp_path: Path) -> None:
    from peft import PeftModel
    from transformers import Qwen3VLMoeConfig, Qwen3VLMoeForConditionalGeneration

    config = Qwen3VLMoeConfig(
        text_config={
            "vocab_size": 64,
            "hidden_size": 32,
            "intermediate_size": 64,
            "moe_intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "max_position_embeddings": 64,
            "use_cache": False,
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 4,
            "in_channels": 3,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "out_hidden_size": 32,
            "num_position_embeddings": 256,
            "deepstack_visual_indexes": [0],
        },
        image_token_id=60,
        video_token_id=61,
        vision_start_token_id=58,
        vision_end_token_id=59,
    )
    config._experts_implementation = "grouped_mm"
    base_dir = tmp_path / "qwen3vl-moe-base"
    Qwen3VLMoeForConditionalGeneration(config).save_pretrained(base_dir)
    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        base_dir,
        experts_implementation="grouped_mm",
    )
    adapter = build_model_meta("qwen3vl").resolve_adapter(
        model_name_or_path="Qwen3-VL-30B-A3B-Instruct"
    )
    finetune = FinetuneConfig(
        mode="lora",
        target_modules=[],
        target_parameters=["auto"],
        lora_r=2,
        lora_alpha=4,
    )

    plan = build_resolved_finetune_plan(model, finetune, model_adapter=adapter)

    assert plan.adapter_plan is not None
    assert plan.adapter_plan.resolved_target_parameters == (
        "model.language_model.layers.0.mlp.experts.gate_up_proj",
        "model.language_model.layers.0.mlp.experts.down_proj",
        "model.language_model.layers.0.mlp.gate.weight",
    )
    wrapped = apply_resolved_finetune_plan(model, plan, finetune=finetune)
    trainable = {
        name: parameter
        for name, parameter in wrapped.named_parameters()
        if parameter.requires_grad
    }
    frozen = {
        name: parameter
        for name, parameter in wrapped.named_parameters()
        if not parameter.requires_grad
    }
    assert any("mlp.gate.lora_A" in name for name in trainable)
    assert any("mlp.experts" in name and "lora_A" in name for name in trainable)
    assert frozen
    assert all("lora_" in name for name in trainable)

    inputs = torch.randint(0, 50, (2, 8))
    output = wrapped(
        input_ids=inputs,
        use_cache=False,
        output_router_logits=True,
    )
    (output.logits.float().mean() + output.aux_loss).backward()
    assert all(parameter.grad is not None for parameter in trainable.values())
    torch.optim.SGD(trainable.values(), lr=0.01).step()

    adapter_dir = tmp_path / "qwen3vl-moe-adapter"
    wrapped.save_pretrained(adapter_dir)
    reloaded = PeftModel.from_pretrained(
        Qwen3VLMoeForConditionalGeneration.from_pretrained(
            base_dir,
            experts_implementation="grouped_mm",
        ),
        adapter_dir,
    ).eval()
    wrapped.eval()
    with torch.no_grad():
        expected_logits = wrapped(input_ids=inputs, use_cache=False).logits
        reloaded_logits = reloaded(input_ids=inputs, use_cache=False).logits
    torch.testing.assert_close(reloaded_logits, expected_logits)


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
        model_adapter=adapter,
        param_group_lrs={"language_model": 5e-4, "generator": 2e-4},
    )
    summary = summarize_resolved_optimizer_plan(resolved)

    assert summary.total_trainable_params > 0
    assert summary.group_count == len(summary.groups)
    assert any(
        group.module_group == "language_model" and group.lr == pytest.approx(5e-4)
        for group in summary.groups
    )
    assert any(
        group.module_group == "generator" and group.lr == pytest.approx(2e-4)
        for group in summary.groups
    )
    assert all(group.sample_raw_parameter_names for group in summary.groups)
    assert all(group.sample_canonical_parameter_names for group in summary.groups)
    assert summary.to_log_dict() == resolved.to_log_dict()


def test_optimizer_summary_json_is_derived_from_resolved_plan(tmp_path: Path) -> None:
    model = _ExactNamedParameterModel(
        ["base_model.model.model.visual.blocks.0.attn.qkv.lora_A.default.weight"]
    )
    adapter = build_model_meta("qwen36vl").resolve_adapter(
        model_name_or_path="models/Qwen3.6-27B"
    )
    args = build_training_args(output_dir=tmp_path)
    plan = build_resolved_optimizer_plan(
        model=model,
        args=args,
        model_adapter=adapter,
        param_group_lrs={"vision_tower": 3e-6},
    )

    path = write_resolved_optimizer_summary(tmp_path, plan)
    payload = json.loads(path.read_text(encoding="utf-8"))

    expected = json.loads(json.dumps(plan.summary().to_dict()))
    assert payload == expected
    assert payload["groups"][0]["module_group"] == "vision_tower"
    assert payload["groups"][0]["sample_raw_parameter_names"]
    assert payload["groups"][0]["sample_canonical_parameter_names"]


def test_optimizer_summary_uses_deepspeed_global_parameter_counts() -> None:
    ds_numel_param = torch.nn.Parameter(torch.empty(0), requires_grad=True)
    ds_numel_param.ds_numel = 13
    ds_shape_param = torch.nn.Parameter(torch.empty(0), requires_grad=True)
    ds_shape_param.ds_shape = (2, 3, 5)
    plan = ShaftResolvedOptimizerPlan(
        groups=(
            ShaftOptimizerParamGroup(
                module_group="language_model",
                decay=True,
                lr=1e-5,
                weight_decay=0.03,
                raw_parameter_names=("layer.ds_numel", "layer.ds_shape"),
                canonical_parameter_names=("layer.ds_numel", "layer.ds_shape"),
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
    assert groups_by_decay[True].raw_parameter_names == ("weight",)
    assert groups_by_decay[True].to_optimizer_group()["weight_decay"] == pytest.approx(0.03)
    assert groups_by_decay[False].raw_parameter_names == ("bias",)
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
    consumer.model_adapter = None
    consumer.param_group_lrs = {}
    consumer.no_decay_name_patterns = []
    consumer.resolved_optimizer_plan = plan

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


def test_fsdp_wrapper_prefix_preserves_optimizer_plan_fingerprint() -> None:
    original_model = _TinyModel()
    wrapped_model = torch.nn.Module()
    wrapped_model._fsdp_wrapped_module = deepcopy(original_model)
    args = build_training_args(output_dir="/tmp/shaft_optimizer_fsdp_prefix")

    original = build_resolved_optimizer_plan(model=original_model, args=args)
    wrapped = build_resolved_optimizer_plan(model=wrapped_model, args=args)

    assert wrapped.fingerprint == original.fingerprint
    assert all(
        raw.startswith("_fsdp_wrapped_module.")
        for group in wrapped.groups
        for raw in group.raw_parameter_names
    )
    assert tuple(
        name for group in wrapped.groups for name in group.canonical_parameter_names
    ) == tuple(
        name for group in original.groups for name in group.canonical_parameter_names
    )


def test_cosine_scheduler_preserves_structural_group_lr_ratio() -> None:
    language = torch.nn.Parameter(torch.ones(1))
    vision = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW(
        [
            {"params": [language], "lr": 1e-5},
            {"params": [vision], "lr": 3e-6},
        ]
    )
    scheduler = build_scheduler(
        scheduler_name="cosine",
        optimizer=optimizer,
        num_warmup_steps=2,
        num_training_steps=10,
        num_cycles=0.5,
    )

    observed = []
    for _ in range(9):
        optimizer.step()
        scheduler.step()
        language_lr, vision_lr = scheduler.get_last_lr()
        if language_lr > 0:
            observed.append(vision_lr / language_lr)

    assert observed
    assert all(ratio == pytest.approx(0.3) for ratio in observed)


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
