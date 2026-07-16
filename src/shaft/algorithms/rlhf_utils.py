from __future__ import annotations

import copy
from contextlib import nullcontext
from importlib import metadata
import math
from pathlib import Path
from typing import TYPE_CHECKING

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version
import torch
from transformers import TrainingArguments

from shaft.config import DPOConfig as ShaftDPOConfig
from shaft.config import GRPOConfig as ShaftGRPOConfig
from shaft.config import PPOConfig as ShaftPPOConfig
from shaft.data.sampler import ShaftGroupedSampleContract

from shaft.training.trl_trainers import _DPO_IMPORT_ERROR, _GRPO_IMPORT_ERROR, _PPO_IMPORT_ERROR
from shaft.utils.contract_schema import json_int, load_strict_json, require_json_mapping

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

if _GRPO_IMPORT_ERROR is None:
    from trl import GRPOConfig as TRLGRPOConfig
else:
    TRLGRPOConfig = None  # type: ignore[assignment]


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


def _precision_model_init_kwargs(train_args: TrainingArguments) -> dict[str, object]:
    if bool(getattr(train_args, "bf16", False)):
        return {"dtype": "bfloat16"}
    if bool(getattr(train_args, "fp16", False)):
        return {"dtype": "float16"}
    return {}


def _set_default_model_init_kwargs(
    payload: dict[str, object],
    defaults: dict[str, object],
) -> None:
    if not defaults:
        return
    model_init_kwargs = payload.get("model_init_kwargs")
    if model_init_kwargs is None:
        payload["model_init_kwargs"] = dict(defaults)
    elif isinstance(model_init_kwargs, dict):
        for key, value in defaults.items():
            model_init_kwargs.setdefault(key, value)


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
    if model_meta.capabilities.is_multimodal and not bool(
        rlhf_config.allow_text_only_multimodal_ppo
    ):
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
    if str(
        rlhf_config.reward_model_mode
    ).strip().lower() == "adapter_disabled_policy" and not callable(
        getattr(model, "disable_adapter", None)
    ):
        raise ValueError(
            "rlhf.ppo.reward_model_mode='adapter_disabled_policy' requires a PEFT policy model "
            "that provides disable_adapter()."
        )


def build_trl_dpo_config(*, train_args: TrainingArguments, rlhf_config: ShaftDPOConfig):
    if TRLDPOConfig is None:
        raise ImportError(
            'TRL DPO config is unavailable. Install RLHF deps: `uv pip install -e ".[rlhf]"`.'
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
            'TRL PPO config is unavailable. Install RLHF deps: `uv pip install -e ".[rlhf]"`.'
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
            "stop_token": str(rlhf_config.stop_token)
            if rlhf_config.stop_token is not None
            else None,
        }
    )
    return TRLPPOConfig(**payload)


def build_trl_grpo_config(*, train_args: TrainingArguments, rlhf_config: ShaftGRPOConfig):
    if TRLGRPOConfig is None:
        raise ImportError(
            'TRL GRPO config is unavailable. Install RLHF deps: `uv pip install -e ".[rlhf]"`.'
        ) from _GRPO_IMPORT_ERROR
    payload = _normalize_training_args_payload(train_args)
    _set_default_model_init_kwargs(payload, _precision_model_init_kwargs(train_args))
    rollout_config = copy.deepcopy(rlhf_config.rollout)
    vllm_config = copy.deepcopy(rlhf_config.vllm)
    if rlhf_config.num_generations is not None:
        rollout_config.num_generations = int(rlhf_config.num_generations)
    if rlhf_config.num_generations_eval is not None:
        rollout_config.num_generations_eval = int(rlhf_config.num_generations_eval)
    if rlhf_config.max_completion_length is not None:
        rollout_config.max_completion_length = int(rlhf_config.max_completion_length)
    if rlhf_config.temperature is not None:
        rollout_config.temperature = float(rlhf_config.temperature)
    if rlhf_config.top_p is not None:
        rollout_config.top_p = float(rlhf_config.top_p)
    if rlhf_config.top_k is not None:
        rollout_config.top_k = int(rlhf_config.top_k)
    if rlhf_config.min_p is not None:
        rollout_config.min_p = float(rlhf_config.min_p)
    if rlhf_config.repetition_penalty is not None:
        rollout_config.repetition_penalty = float(rlhf_config.repetition_penalty)
    if rlhf_config.use_vllm is not None:
        vllm_config.enabled = bool(rlhf_config.use_vllm)
    world_size = int(getattr(train_args, "world_size", 1) or 1)
    base_global_batch = max(1, int(train_args.per_device_train_batch_size) * world_size)
    steps_per_generation = max(1, int(train_args.gradient_accumulation_steps))
    num_generations = int(rollout_config.num_generations)
    while (base_global_batch * steps_per_generation) % num_generations != 0:
        steps_per_generation += 1
    generation_kwargs = dict(rollout_config.generation_kwargs)
    payload.update(
        {
            "beta": float(rlhf_config.beta),
            "reward_weights": [float(reward.weight) for reward in rlhf_config.reward_functions],
            "num_generations": num_generations,
            "num_generations_eval": (
                int(rollout_config.num_generations_eval)
                if rollout_config.num_generations_eval is not None
                else None
            ),
            "max_completion_length": int(rollout_config.max_completion_length),
            "temperature": float(rollout_config.temperature),
            "top_p": float(rollout_config.top_p),
            "top_k": int(rollout_config.top_k),
            "min_p": float(rollout_config.min_p) if rollout_config.min_p is not None else None,
            "generation_kwargs": generation_kwargs or None,
            "cache_implementation": (
                str(rollout_config.cache_implementation)
                if rollout_config.cache_implementation is not None
                else None
            ),
            "use_transformers_paged": bool(rollout_config.use_transformers_paged),
            "repetition_penalty": float(rollout_config.repetition_penalty),
            "use_vllm": bool(vllm_config.enabled),
            "vllm_mode": str(vllm_config.mode),
            "vllm_model_impl": str(vllm_config.model_impl),
            "vllm_enable_sleep_mode": bool(vllm_config.enable_sleep_mode),
            "vllm_structured_outputs_regex": vllm_config.structured_outputs_regex,
            "vllm_server_base_url": vllm_config.server_base_url,
            "vllm_server_host": str(vllm_config.server_host),
            "vllm_server_port": int(vllm_config.server_port),
            "vllm_server_timeout": float(vllm_config.server_timeout),
            "vllm_group_port": int(vllm_config.group_port),
            "vllm_gpu_memory_utilization": float(vllm_config.gpu_memory_utilization),
            "vllm_max_model_length": (
                int(vllm_config.max_model_length)
                if vllm_config.max_model_length is not None
                else None
            ),
            "vllm_tensor_parallel_size": int(vllm_config.tensor_parallel_size),
            "steps_per_generation": steps_per_generation,
            # ShaftSamplePlan is the only shuffle owner.  The grouped sampler only
            # expands GRPO mini-repeat/repeat structure; allowing TRL to shuffle a
            # second time can cross source-local permutation cycles.
            "shuffle_dataset": False,
        }
    )
    return TRLGRPOConfig(**payload)


def resolve_grpo_grouped_sample_contract(args: object) -> ShaftGroupedSampleContract:
    """Resolve TRL GRPO arguments into the generic grouped-repeat contract once."""

    num_generations = int(getattr(args, "num_generations"))
    generation_batch_size = int(getattr(args, "generation_batch_size"))
    if generation_batch_size % num_generations != 0:
        raise ValueError(
            "GRPO generation_batch_size must be divisible by num_generations: "
            f"{generation_batch_size} vs {num_generations}."
        )
    return ShaftGroupedSampleContract(
        mini_repeat_count=num_generations,
        batch_size=generation_batch_size // num_generations,
        iteration_count=int(getattr(args, "num_iterations")),
        steps_per_iteration=int(getattr(args, "steps_per_generation")),
    )


def resolve_grpo_checkpoint_step_cadence(args: object) -> int:
    """Return the safe cadence within a stream of full accumulation steps.

    TRL does not checkpoint its private ``_step`` or ``_buffered_inputs`` state.
    A checkpoint is therefore exactly resumable only after a whole generation
    reuse cycle. Checkpoints are written after optimizer steps, while TRL tracks
    the cycle in local training microsteps. Epoch tails can contain a shortened
    accumulation step and must additionally use the real epoch geometry below.
    """

    gradient_accumulation = int(getattr(args, "gradient_accumulation_steps"))
    generate_every = int(getattr(args, "steps_per_generation")) * int(
        getattr(args, "num_iterations")
    )
    if gradient_accumulation <= 0 or generate_every <= 0:
        raise ValueError("GRPO checkpoint cadence values must be > 0.")
    return generate_every // math.gcd(gradient_accumulation, generate_every)


def _grpo_save_strategy(args: object) -> str:
    save_strategy_value = getattr(getattr(args, "save_strategy", "no"), "value", None)
    return (
        str(
            save_strategy_value
            if save_strategy_value is not None
            else getattr(args, "save_strategy", "no")
        )
        .strip()
        .lower()
    )


def _grpo_uses_vllm(args: object) -> bool:
    use_vllm = getattr(args, "use_vllm", False)
    if type(use_vllm) is not bool:
        raise ValueError(
            f"GRPO use_vllm must be a boolean before runtime validation; got {use_vllm!r}."
        )
    return use_vllm


def validate_grpo_rollout_checkpointability(
    args: object,
    *,
    resume_requested: bool = False,
) -> None:
    """Reject rollout backends whose sampling state is not checkpointed."""

    if not _grpo_uses_vllm(args):
        return
    save_strategy = _grpo_save_strategy(args)
    if save_strategy == "no" and not bool(resume_requested):
        return
    raise ValueError(
        "GRPO vLLM sampled rollout cannot publish or resume an exact checkpoint: "
        "the vLLM engine/server RNG state is not persisted by TRL or Shaft. Set "
        "train.save_strategy='no' and do not resume until rollout requests use a "
        "checkpointed or canonical per-request seed."
    )


def _grpo_vllm_compatibility_error(
    reason: str,
    *,
    trl_version: str,
    required_spec: str,
    installed_version: str,
) -> ValueError:
    return ValueError(
        "GRPO vLLM runtime is incompatible with the installed TRL package: "
        f"trl_version={trl_version}, required_vllm_spec={required_spec}, "
        f"installed_vllm_version={installed_version}. {reason}"
    )


def validate_grpo_vllm_runtime_compatibility(args: object) -> None:
    """Fail closed unless TRL's active ``vllm`` extra accepts the runtime version.

    TRL owns the supported vLLM version window. Reading that window from package
    metadata avoids duplicating an immediately stale compatibility table in Shaft.
    The validation deliberately remains separate from exact-resume rollout-state
    checks: a compatible backend can still be non-checkpointable.
    """

    if not _grpo_uses_vllm(args):
        return

    try:
        trl_version = str(metadata.version("trl"))
    except Exception as exc:
        raise _grpo_vllm_compatibility_error(
            "TRL distribution metadata is unavailable.",
            trl_version="<missing>",
            required_spec="<unresolved>",
            installed_version="<unresolved>",
        ) from exc

    try:
        raw_requirements = metadata.requires("trl")
    except Exception as exc:
        raise _grpo_vllm_compatibility_error(
            "TRL dependency metadata cannot be read.",
            trl_version=trl_version,
            required_spec="<unresolved>",
            installed_version="<unresolved>",
        ) from exc

    try:
        installed_version = str(metadata.version("vllm"))
    except Exception:
        installed_version = "<missing>"

    matching_requirements: list[Requirement] = []
    for raw_requirement in raw_requirements or ():
        try:
            requirement = Requirement(raw_requirement)
        except (InvalidRequirement, TypeError, ValueError) as exc:
            raise _grpo_vllm_compatibility_error(
                f"TRL contains malformed dependency metadata: {raw_requirement!r}.",
                trl_version=trl_version,
                required_spec=f"<malformed:{raw_requirement!r}>",
                installed_version=installed_version,
            ) from exc
        if canonicalize_name(requirement.name) != "vllm":
            continue
        marker = requirement.marker
        if marker is None:
            continue
        try:
            enabled_with_extra = marker.evaluate({"extra": "vllm"})
            enabled_without_extra = marker.evaluate({"extra": ""})
        except Exception as exc:
            raise _grpo_vllm_compatibility_error(
                f"TRL's vLLM environment marker cannot be evaluated: {marker!s}.",
                trl_version=trl_version,
                required_spec=f"<malformed-marker:{marker!s}>",
                installed_version=installed_version,
            ) from exc
        if enabled_with_extra and not enabled_without_extra:
            matching_requirements.append(requirement)

    specs = {
        str(requirement.specifier)
        for requirement in matching_requirements
        if requirement.url is None and str(requirement.specifier)
    }
    invalid_candidates = [
        requirement
        for requirement in matching_requirements
        if requirement.url is not None or not str(requirement.specifier)
    ]
    if invalid_candidates or len(specs) != 1:
        resolved = ",".join(sorted(specs)) if specs else "<missing>"
        if invalid_candidates:
            resolved = f"<ambiguous:{resolved}>"
        elif len(specs) > 1:
            resolved = f"<ambiguous:{resolved}>"
        raise _grpo_vllm_compatibility_error(
            "TRL does not publish one unambiguous conditional vLLM version specifier.",
            trl_version=trl_version,
            required_spec=resolved,
            installed_version=installed_version,
        )

    required_spec = next(iter(specs))
    if installed_version == "<missing>":
        raise _grpo_vllm_compatibility_error(
            "The vLLM distribution is not installed.",
            trl_version=trl_version,
            required_spec=required_spec,
            installed_version=installed_version,
        )
    try:
        parsed_version = Version(installed_version)
    except InvalidVersion as exc:
        raise _grpo_vllm_compatibility_error(
            "The installed vLLM distribution has a malformed version.",
            trl_version=trl_version,
            required_spec=required_spec,
            installed_version=installed_version,
        ) from exc
    requirement = matching_requirements[0]
    if not requirement.specifier.contains(parsed_version, prereleases=True):
        raise _grpo_vllm_compatibility_error(
            "Install a vLLM release inside TRL's declared compatibility window.",
            trl_version=trl_version,
            required_spec=required_spec,
            installed_version=installed_version,
        )


def _grpo_microstep_at_optimizer_step(
    global_step: int,
    *,
    epoch_microsteps: int,
    gradient_accumulation_steps: int,
) -> int:
    """Map an optimizer boundary to the actual consumed local microstep count."""

    updates_per_epoch = int(math.ceil(epoch_microsteps / gradient_accumulation_steps))
    complete_epochs, step_in_epoch = divmod(int(global_step), updates_per_epoch)
    return complete_epochs * epoch_microsteps + min(
        step_in_epoch * gradient_accumulation_steps,
        epoch_microsteps,
    )


def _grpo_total_optimizer_steps(args: object, *, updates_per_epoch: int) -> int:
    configured_max_steps = int(getattr(args, "max_steps", -1))
    if configured_max_steps > 0:
        return configured_max_steps
    num_train_epochs = float(getattr(args, "num_train_epochs", 0.0))
    if not math.isfinite(num_train_epochs) or num_train_epochs <= 0:
        raise ValueError("GRPO checkpoint cadence requires max_steps > 0 or num_train_epochs > 0.")
    return int(math.ceil(num_train_epochs * updates_per_epoch))


def validate_grpo_checkpoint_cadence(
    args: object,
    *,
    epoch_microsteps: int | None = None,
    resume_checkpoint: str | Path | None = None,
) -> None:
    """Prove every configured save/resume boundary against real epoch geometry."""

    save_strategy = _grpo_save_strategy(args)
    if save_strategy not in {"no", "steps", "epoch"}:
        raise ValueError(
            "GRPO resumable checkpointing currently supports only "
            "save_strategy='no', cadence-aligned save_strategy='steps', or "
            "save_strategy='epoch' when every optimizer boundary is safe; "
            f"got save_strategy={save_strategy!r}."
        )
    if save_strategy == "no" and resume_checkpoint is None:
        return
    if epoch_microsteps is None or int(epoch_microsteps) <= 0:
        raise ValueError(
            "GRPO checkpoint cadence requires the proven rank-local epoch_microsteps geometry."
        )

    gradient_accumulation = int(getattr(args, "gradient_accumulation_steps"))
    generate_every = int(getattr(args, "steps_per_generation")) * int(
        getattr(args, "num_iterations")
    )
    epoch_microsteps = int(epoch_microsteps)
    if gradient_accumulation <= 0 or generate_every <= 0:
        raise ValueError("GRPO checkpoint cadence values must be > 0.")
    if epoch_microsteps % generate_every != 0:
        raise ValueError(
            "GRPO grouped epoch must end on a complete generation-reuse cycle: "
            f"epoch_microsteps={epoch_microsteps}, generate_every={generate_every}."
        )
    updates_per_epoch = int(math.ceil(epoch_microsteps / gradient_accumulation))

    def _boundary(global_step: int) -> tuple[int, int]:
        microstep = _grpo_microstep_at_optimizer_step(
            global_step,
            epoch_microsteps=epoch_microsteps,
            gradient_accumulation_steps=gradient_accumulation,
        )
        return microstep, microstep % generate_every

    if save_strategy == "steps":
        total_optimizer_steps = _grpo_total_optimizer_steps(
            args,
            updates_per_epoch=updates_per_epoch,
        )
        raw_save_steps = float(getattr(args, "save_steps"))
        if not raw_save_steps.is_integer() or raw_save_steps <= 0:
            raise ValueError("GRPO exact-resume save cadence requires an integer save_steps > 0.")
        save_steps = int(raw_save_steps)
        save_target_count = total_optimizer_steps // save_steps
        # Boundary phases repeat every K/gcd(K, save_steps) save targets. Check
        # one complete residue cycle, or the finite configured run if shorter.
        target_period = updates_per_epoch // math.gcd(
            updates_per_epoch,
            save_steps,
        )
        for target_index in range(
            1,
            min(save_target_count, target_period) + 1,
        ):
            global_step = target_index * save_steps
            microstep, phase = _boundary(global_step)
            if phase:
                raise ValueError(
                    "GRPO save target is inside a generation-reuse cycle after "
                    "accounting for shortened epoch accumulation: "
                    f"save_steps={save_steps}, global_step={global_step}, "
                    f"microstep={microstep}, phase={phase}, "
                    f"generate_every={generate_every}, "
                    f"epoch_microsteps={epoch_microsteps}."
                )
    elif save_strategy == "epoch":
        total_optimizer_steps = _grpo_total_optimizer_steps(
            args,
            updates_per_epoch=updates_per_epoch,
        )
        # Complete grouped epochs are safe because epoch_microsteps is a multiple
        # of generate_every. HF also emits an epoch save when a max-step or
        # fractional-epoch run stops partway through its final epoch.
        if total_optimizer_steps % updates_per_epoch:
            microstep, phase = _boundary(total_optimizer_steps)
            if phase:
                raise ValueError(
                    "GRPO save_strategy='epoch' would save a partial epoch inside "
                    "a generation-reuse cycle: "
                    f"global_step={total_optimizer_steps}, microstep={microstep}, "
                    f"phase={phase}, generate_every={generate_every}, "
                    f"epoch_microsteps={epoch_microsteps}."
                )

    if resume_checkpoint is None:
        return
    state_path = Path(resume_checkpoint) / "trainer_state.json"
    try:
        state_payload = load_strict_json(
            state_path,
            role=f"GRPO trainer state {state_path}",
        )
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"GRPO exact resume requires a readable trainer_state.json global_step: {state_path}."
        ) from exc
    state_payload = require_json_mapping(
        state_payload,
        role=f"GRPO trainer state {state_path}",
    )
    if "global_step" not in state_payload:
        raise ValueError(
            f"GRPO exact resume requires trainer_state.json global_step: {state_path}."
        )
    global_step = json_int(
        state_payload,
        "global_step",
        role=f"GRPO trainer state {state_path}",
    )
    if global_step < 0:
        raise ValueError("GRPO resume checkpoint global_step must be >= 0.")
    microstep, phase = _boundary(global_step)
    if phase:
        raise ValueError(
            "GRPO checkpoint is inside a generation-reuse cycle and cannot be "
            "resumed exactly because TRL does not persist its generation buffer: "
            f"global_step={global_step}, microstep={microstep}, phase={phase}, "
            f"gradient_accumulation_steps={gradient_accumulation}, "
            f"generate_every={generate_every}, epoch_microsteps={epoch_microsteps}."
        )
