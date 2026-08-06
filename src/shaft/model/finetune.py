from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)

from shaft.config import FinetuneConfig

from .finetune_plan import ShaftResolvedFinetunePlan, build_resolved_finetune_plan
from .generation import disable_model_cache_for_training
from .parameters import parameter_numel
from .resolution import (
    ResolvedAdapterInit,
    load_resolved_adapter_weights,
    resolve_adapter_artifact,
)
from .types import ShaftModelAdapter


@dataclass
class FinetuneSummary:
    mode: str
    total_params: int
    trainable_params: int


def _count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = 0
    trainable = 0
    for parameter in model.parameters():
        count = parameter_numel(parameter)
        total += count
        if parameter.requires_grad:
            trainable += count
    return total, trainable


def summarize_finetune(model: torch.nn.Module, mode: str) -> FinetuneSummary:
    total, trainable = _count_parameters(model)
    return FinetuneSummary(mode=str(mode), total_params=total, trainable_params=trainable)


def load_peft_checkpoint(
    model: torch.nn.Module,
    checkpoint_dir: str | Path,
    *,
    resolved_artifact: ResolvedAdapterInit | None = None,
) -> tuple[int, int]:
    """Load and verify one complete PEFT state before distributed wrapping."""

    if not isinstance(model, PeftModel):
        raise TypeError("PEFT checkpoint loading requires an unwrapped PeftModel.")
    checkpoint = Path(checkpoint_dir).resolve()
    artifact = resolved_artifact or resolve_adapter_artifact(checkpoint)
    if Path(artifact.path).resolve() != checkpoint:
        raise ValueError(
            "Resolved PEFT artifact path differs from the requested checkpoint: "
            f"artifact={artifact.path}, checkpoint={checkpoint}."
        )
    saved_state = load_resolved_adapter_weights(artifact)

    expected_state = get_peft_model_state_dict(model, adapter_name="default")
    missing = sorted(set(expected_state).difference(saved_state))
    unexpected = sorted(set(saved_state).difference(expected_state))
    shape_mismatches = sorted(
        name
        for name in set(expected_state).intersection(saved_state)
        if tuple(expected_state[name].shape) != tuple(saved_state[name].shape)
    )
    if missing or unexpected or shape_mismatches:
        raise ValueError(
            "PEFT checkpoint state does not exactly match the prepared adapter: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}, "
            f"shape_mismatches={shape_mismatches[:8]}."
        )

    # PEFT may normalize modules_to_save keys in-place.  Keep the canonical
    # file state separate so post-load verification always compares against
    # exactly what was serialized without cloning multi-GB tensors.
    state_for_load = dict(saved_state)
    load_result = set_peft_model_state_dict(
        model,
        state_for_load,
        adapter_name="default",
    )
    if tuple(getattr(load_result, "unexpected_keys", ()) or ()):
        raise ValueError(
            "PEFT checkpoint load left unexpected model keys: "
            f"{tuple(load_result.unexpected_keys)[:8]}."
        )
    loaded_state = get_peft_model_state_dict(model, adapter_name="default")
    unequal = [
        name
        for name in sorted(saved_state)
        if not torch.equal(
            loaded_state[name].detach().cpu(),
            saved_state[name].detach().cpu(),
        )
    ]
    if unequal:
        raise RuntimeError(
            "PEFT checkpoint state changed while loading into the training model: "
            f"{unequal[:8]}."
        )
    return len(saved_state), sum(int(tensor.numel()) for tensor in saved_state.values())


def apply_resolved_finetune_plan(
    model: torch.nn.Module,
    plan: ShaftResolvedFinetunePlan,
    *,
    finetune: FinetuneConfig,
    gradient_checkpointing: bool = False,
) -> torch.nn.Module:
    # A training model must never build generation caches. Inference and online
    # evaluation explicitly enable and restore cache state around generation.
    disable_model_cache_for_training(model)

    if plan.mode == "full":
        trainable_names = set(plan.parameter_plan.trainable_parameter_names)
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name in trainable_names)
        return model

    if plan.mode not in {"lora", "dora", "qlora"}:
        raise ValueError(f"Unsupported finetune mode: {plan.mode!r}")

    if plan.mode == "qlora":
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=bool(gradient_checkpointing),
        )

    if plan.adapter_plan is None:
        raise ValueError(f"Missing adapter finetune plan for mode={plan.mode!r}.")

    if plan.adapter_plan.resolved_target_parameters:
        if plan.mode == "dora":
            raise ValueError("PEFT target_parameters do not support DoRA.")
        if float(finetune.lora_dropout) != 0.0:
            raise ValueError("PEFT target_parameters require lora_dropout=0.")

    peft_config = LoraConfig(
        r=plan.adapter_plan.peft_signature.r,
        lora_alpha=plan.adapter_plan.peft_signature.lora_alpha,
        lora_dropout=float(finetune.lora_dropout),
        bias=plan.adapter_plan.peft_signature.lora_bias,
        target_modules=list(plan.adapter_plan.resolved_target_modules),
        target_parameters=list(plan.adapter_plan.resolved_target_parameters),
        modules_to_save=list(plan.adapter_plan.modules_to_save),
        task_type=TaskType.CAUSAL_LM,
        use_dora=plan.adapter_plan.peft_signature.use_dora,
        use_rslora=plan.adapter_plan.peft_signature.use_rslora,
    )
    wrapped = get_peft_model(model, peft_config)
    if gradient_checkpointing and hasattr(wrapped, "enable_input_require_grads"):
        wrapped.enable_input_require_grads()
    return wrapped


def apply_finetune_strategy(
    model: torch.nn.Module,
    finetune: FinetuneConfig,
    *,
    model_adapter: ShaftModelAdapter,
    gradient_checkpointing: bool = False,
) -> torch.nn.Module:
    plan = build_resolved_finetune_plan(model, finetune, model_adapter=model_adapter)
    wrapped = apply_resolved_finetune_plan(
        model,
        plan,
        finetune=finetune,
        gradient_checkpointing=gradient_checkpointing,
    )
    setattr(wrapped, "_shaft_finetune_plan", plan)
    return wrapped


def make_bnb_4bit_config(finetune: FinetuneConfig, *, dtype: torch.dtype | str) -> Any:
    try:
        from transformers import BitsAndBytesConfig
    except Exception as exc:  # noqa: BLE001
        raise ImportError("BitsAndBytesConfig is unavailable in current transformers version.") from exc
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=bool(finetune.qlora_use_double_quant),
        bnb_4bit_quant_type=str(finetune.qlora_quant_type),
        bnb_4bit_compute_dtype=dtype if isinstance(dtype, torch.dtype) else torch.bfloat16,
    )
