from __future__ import annotations

import copy
from contextlib import nullcontext
from typing import TYPE_CHECKING

import torch
from transformers import TrainingArguments

from shaft.config import DPOConfig as ShaftDPOConfig
from shaft.config import PPOConfig as ShaftPPOConfig

from shaft.training.rlhf import _DPO_IMPORT_ERROR, _PPO_IMPORT_ERROR

if TYPE_CHECKING:
    from shaft.model.types import ModelMeta

if _DPO_IMPORT_ERROR is None:
    from trl import DPOConfig as TRLDPOConfig
else:
    TRLDPOConfig = None  # type: ignore[assignment]

if _PPO_IMPORT_ERROR is None:
    from trl.experimental.ppo import PPOConfig as TRLPPOConfig
else:
    TRLPPOConfig = None  # type: ignore[assignment]


def _normalize_training_args_payload(train_args: TrainingArguments) -> dict[str, object]:
    """Build TRL config payload from TrainingArguments without deprecated token placeholders.

    `TrainingArguments.to_dict()` redacts secret fields into placeholder strings such as
    `<PUSH_TO_HUB_TOKEN>`, which are interpreted as non-None values by downstream configs
    and trigger deprecation warnings. We normalize these fields back to the runtime values.
    """
    payload = train_args.to_dict()
    runtime_values = vars(train_args)

    # Drop deprecated push_to_hub aliases entirely to avoid FutureWarning in HF >= 4.56.
    payload.pop("push_to_hub_token", None)
    payload.pop("push_to_hub_model_id", None)
    payload.pop("push_to_hub_organization", None)

    # Keep modern hub fields as real runtime values (not redacted placeholders).
    payload["hub_token"] = runtime_values.get("hub_token")
    payload["hub_model_id"] = runtime_values.get("hub_model_id")
    payload["hub_private_repo"] = runtime_values.get("hub_private_repo")

    return payload


def build_reference_model(*, model: torch.nn.Module, finetune_mode: str) -> torch.nn.Module | None:
    mode = str(finetune_mode).strip().lower()
    if mode in {"lora", "dora", "qlora"} and callable(getattr(model, "disable_adapter", None)):
        return None
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for parameter in ref_model.parameters():
        parameter.requires_grad_(False)
    return ref_model


def validate_ppo_runtime_requirements(
    *,
    model_meta: ModelMeta,
    model: torch.nn.Module,
    finetune_mode: str,
    rlhf_config: ShaftPPOConfig,
) -> None:
    if model_meta.capabilities.is_multimodal and not bool(rlhf_config.allow_text_only_multimodal_ppo):
        raise ValueError(
            "Current TRL PPO path is text-only and does not support multimodal rollout inputs. "
            "Set rlhf.ppo.allow_text_only_multimodal_ppo=true only for smoke/debug runs."
        )
    mode = str(finetune_mode).strip().lower()
    if mode == "full":
        raise ValueError(
            "Shaft PPO currently does not support finetune.mode='full'. "
            "Use lora/dora/qlora to keep PPO memory bounded and reward/reference behavior stable."
        )
    if mode not in {"lora", "dora", "qlora"}:
        raise ValueError(f"Unsupported finetune mode for PPO: {mode!r}.")
    if str(rlhf_config.value_model_mode).strip().lower() == "shared_backbone" and bool(
        rlhf_config.train_value_backbone
    ):
        raise ValueError(
            "rlhf.ppo.value_model_mode='shared_backbone' is incompatible with train_value_backbone=true."
        )
    if str(rlhf_config.reward_model_mode).strip().lower() == "adapter_disabled_policy" and not callable(
        getattr(model, "disable_adapter", None)
    ):
        raise ValueError(
            "rlhf.ppo.reward_model_mode='adapter_disabled_policy' requires a PEFT policy model "
            "that provides disable_adapter()."
        )


def build_trl_dpo_config(*, train_args: TrainingArguments, rlhf_config: ShaftDPOConfig):
    if TRLDPOConfig is None:
        raise ImportError(
            "TRL DPO config is unavailable. Install RLHF deps: `uv pip install -e \".[rlhf]\"`."
        ) from _DPO_IMPORT_ERROR
    payload = _normalize_training_args_payload(train_args)
    payload.update(
        {
            "beta": float(rlhf_config.beta),
            "label_smoothing": float(rlhf_config.label_smoothing),
            "loss_type": str(rlhf_config.loss_type),
            "precompute_ref_log_probs": bool(rlhf_config.precompute_ref_log_probs),
            "use_weighting": bool(rlhf_config.use_weighting),
        }
    )
    return TRLDPOConfig(**payload)


def _resolve_hidden_size(model: torch.nn.Module) -> int:
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError("Cannot resolve hidden_size from model without config.")
    direct = getattr(config, "hidden_size", None)
    if direct is not None:
        return int(direct)
    text_config = getattr(config, "text_config", None)
    if text_config is not None and getattr(text_config, "hidden_size", None) is not None:
        return int(text_config.hidden_size)
    if getattr(config, "d_model", None) is not None:
        return int(config.d_model)
    raise ValueError("Cannot resolve hidden_size for PPO value/reward scorer.")


class ShaftCausalLMScorer(torch.nn.Module):
    """Wraps a causal LM backbone with a TRL-compatible scalar score head."""

    base_model_prefix = "backbone"

    def __init__(
        self,
        *,
        backbone,
        hidden_size: int,
        train_backbone: bool,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.score = torch.nn.Linear(int(hidden_size), 1, bias=False)
        if not train_backbone and isinstance(self.backbone, torch.nn.Module):
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)


class _PolicyBackboneProxy:
    def __init__(self, policy_model: torch.nn.Module) -> None:
        self.policy_model = policy_model

    def __call__(self, *args, **kwargs):
        return self.policy_model(*args, **kwargs)


class _AdapterDisabledPolicyBackboneProxy:
    def __init__(self, policy_model: torch.nn.Module) -> None:
        self.policy_model = policy_model
        disable_adapter = getattr(policy_model, "disable_adapter", None)
        if not callable(disable_adapter):
            raise ValueError("Policy model does not provide disable_adapter().")
        self._disable_adapter = disable_adapter

    def __call__(self, *args, **kwargs):
        context = self._disable_adapter()
        if context is None:
            context = nullcontext()
        with context:
            return self.policy_model(*args, **kwargs)


def build_ppo_value_and_reward_models(
    *,
    model: torch.nn.Module,
    train_value_backbone: bool,
    value_model_mode: str,
    reward_model_mode: str,
    allow_untrained_reward_model: bool,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    if not bool(allow_untrained_reward_model):
        raise ValueError(
            "PPO reward model is currently created with an untrained scalar head by default. "
            "Set rlhf.ppo.allow_untrained_reward_model=true only for smoke/debug runs, "
            "or add a real reward-model loading path first."
        )
    hidden_size = _resolve_hidden_size(model)
    value_mode = str(value_model_mode).strip().lower()
    reward_mode = str(reward_model_mode).strip().lower()
    if value_mode == "shared_backbone":
        value_backbone = _PolicyBackboneProxy(model)
    elif value_mode == "copy_backbone":
        value_backbone = copy.deepcopy(model)
    else:
        raise ValueError(f"Unsupported value_model_mode: {value_model_mode!r}.")
    value_model = ShaftCausalLMScorer(
        backbone=value_backbone,
        hidden_size=hidden_size,
        train_backbone=bool(train_value_backbone),
    )
    if reward_mode == "adapter_disabled_policy":
        reward_backbone = _AdapterDisabledPolicyBackboneProxy(model)
    elif reward_mode == "copy_backbone":
        reward_backbone = copy.deepcopy(model)
    else:
        raise ValueError(f"Unsupported reward_model_mode: {reward_model_mode!r}.")
    reward_model = ShaftCausalLMScorer(
        backbone=reward_backbone,
        hidden_size=hidden_size,
        train_backbone=False,
    )
    reward_model.eval()
    for parameter in reward_model.score.parameters():
        parameter.requires_grad_(False)
    if isinstance(reward_model.backbone, torch.nn.Module):
        for parameter in reward_model.backbone.parameters():
            parameter.requires_grad_(False)
    return value_model, reward_model


def build_trl_ppo_config(*, train_args: TrainingArguments, rlhf_config: ShaftPPOConfig):
    if TRLPPOConfig is None:
        raise ImportError(
            "TRL PPO config is unavailable. Install RLHF deps: `uv pip install -e \".[rlhf]\"`."
        ) from _PPO_IMPORT_ERROR
    payload = _normalize_training_args_payload(train_args)
    payload.update(
        {
            "cliprange": float(rlhf_config.cliprange),
            "cliprange_value": float(rlhf_config.cliprange_value),
            "kl_coef": float(rlhf_config.kl_coef),
            "vf_coef": float(rlhf_config.vf_coef),
            "gamma": float(rlhf_config.gamma),
            "lam": float(rlhf_config.lam),
            "whiten_rewards": bool(rlhf_config.whiten_rewards),
            "response_length": int(rlhf_config.response_length),
            "temperature": float(rlhf_config.temperature),
            "num_ppo_epochs": int(rlhf_config.num_ppo_epochs),
            "num_mini_batches": int(rlhf_config.num_mini_batches),
            "local_rollout_forward_batch_size": int(rlhf_config.local_rollout_forward_batch_size),
            "num_sample_generations": int(rlhf_config.num_sample_generations),
            "stop_token": str(rlhf_config.stop_token) if rlhf_config.stop_token is not None else None,
        }
    )
    return TRLPPOConfig(**payload)
