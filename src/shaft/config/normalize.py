from __future__ import annotations

from .schema import RuntimeConfig

_MIX_STRATEGIES = {"concat", "interleave_under", "interleave_over"}
_ALGORITHMS = {"sft", "dpo", "ppo"}
_FINETUNE_MODES = {"full", "lora", "dora", "qlora"}
_LOSS_NAMES = {"auto", "causal_lm"}
_DPO_LOSS_TYPES = {"sigmoid"}
_PPO_VALUE_MODEL_MODES = {"shared_backbone", "copy_backbone"}
_PPO_REWARD_MODEL_MODES = {"adapter_disabled_policy", "copy_backbone"}
_LOG_LEVELS = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}
_LOG_FORMATS = {"text", "json"}


def normalize_runtime_config(config: RuntimeConfig) -> RuntimeConfig:
    config.algorithm.name = str(config.algorithm.name).strip().lower()
    if config.algorithm.name not in _ALGORITHMS:
        raise ValueError(f"Unsupported algorithm.name={config.algorithm.name!r}. Expected one of {_ALGORITHMS}.")

    config.data.mix_strategy = str(config.data.mix_strategy).strip().lower()
    if config.data.mix_strategy not in _MIX_STRATEGIES:
        raise ValueError(f"Unsupported data.mix_strategy={config.data.mix_strategy!r}.")
    config.data.dataset_refs = [str(x).strip() for x in config.data.dataset_refs if str(x).strip()]
    if config.data.registry_path is not None:
        config.data.registry_path = str(config.data.registry_path).strip() or None

    finetune = config.model.finetune
    finetune.mode = str(finetune.mode).strip().lower()
    if finetune.mode not in _FINETUNE_MODES:
        raise ValueError(f"Unsupported model.finetune.mode={finetune.mode!r}.")
    finetune.lora_bias = str(finetune.lora_bias).strip().lower()
    if not finetune.target_modules:
        finetune.target_modules = ["auto"]
    finetune.target_modules = [str(x).strip() for x in finetune.target_modules if str(x).strip()]
    if not finetune.target_modules:
        raise ValueError("model.finetune.target_modules cannot be empty.")

    for dataset in config.data.datasets:
        dataset.source_type = str(dataset.source_type).strip().lower()
        dataset.train_paths = [str(x).strip() for x in dataset.train_paths if str(x).strip()]
        dataset.val_paths = [str(x).strip() for x in dataset.val_paths if str(x).strip()]
        if dataset.train_path:
            dataset.train_paths = [str(dataset.train_path).strip(), *dataset.train_paths]
        if dataset.val_path:
            dataset.val_paths = [str(dataset.val_path).strip(), *dataset.val_paths]
        if not dataset.train_paths:
            raise ValueError(f"data.datasets[{dataset.name}].train_paths cannot be empty.")
        if not dataset.val_paths and bool(config.sft.eval.enabled):
            raise ValueError(f"data.datasets[{dataset.name}].val_paths cannot be empty.")
        if config.algorithm.name == "sft" and dataset.source_type == "jsonl_ppo":
            raise ValueError(f"data.datasets[{dataset.name}] uses jsonl_ppo but algorithm is sft.")
        if config.algorithm.name == "sft" and dataset.source_type == "jsonl_dpo":
            raise ValueError(f"data.datasets[{dataset.name}] uses jsonl_dpo but algorithm is sft.")
        if config.algorithm.name == "dpo" and dataset.source_type == "jsonl_sft":
            raise ValueError(f"data.datasets[{dataset.name}] uses jsonl_sft but algorithm is dpo.")
        if config.algorithm.name == "dpo" and dataset.source_type == "jsonl_ppo":
            raise ValueError(f"data.datasets[{dataset.name}] uses jsonl_ppo but algorithm is dpo.")
        if config.algorithm.name == "ppo" and dataset.source_type == "jsonl_sft":
            raise ValueError(f"data.datasets[{dataset.name}] uses jsonl_sft but algorithm is ppo.")
        if config.algorithm.name == "ppo" and dataset.source_type == "jsonl_dpo":
            raise ValueError(f"data.datasets[{dataset.name}] uses jsonl_dpo but algorithm is ppo.")

    train = config.sft.train
    train.optimizer_name = str(train.optimizer_name).strip().lower()
    train.scheduler_name = str(train.scheduler_name).strip().lower()
    if train.scheduler_name in {"", "auto"}:
        train.scheduler_name = str(train.lr_scheduler_type).strip().lower()
    train.loss_name = str(train.loss_name).strip().lower()
    if train.loss_name not in _LOSS_NAMES:
        raise ValueError(f"Unsupported sft.train.loss_name={train.loss_name!r}.")
    train.lr_scheduler_type = str(train.lr_scheduler_type).strip().lower()
    if isinstance(train.save_strategy, bool):
        train.save_strategy = "no" if not train.save_strategy else "steps"
    else:
        train.save_strategy = str(train.save_strategy).strip().lower()
    if train.save_strategy not in {"no", "steps", "epoch"}:
        raise ValueError(f"Unsupported sft.train.save_strategy={train.save_strategy!r}.")
    if int(train.epochs) <= 0:
        raise ValueError("sft.train.epochs must be > 0.")
    if int(train.max_steps) == 0:
        raise ValueError("sft.train.max_steps cannot be 0. Use -1 or >0.")
    if float(train.scheduler_num_cycles) <= 0:
        raise ValueError("sft.train.scheduler_num_cycles must be > 0.")
    if float(train.scheduler_power) <= 0:
        raise ValueError("sft.train.scheduler_power must be > 0.")

    eval_cfg = config.sft.eval
    if isinstance(eval_cfg.eval_strategy, bool):
        eval_cfg.eval_strategy = "no" if not eval_cfg.eval_strategy else "steps"
    else:
        eval_cfg.eval_strategy = str(eval_cfg.eval_strategy).strip().lower()
    if eval_cfg.eval_strategy not in {"no", "steps", "epoch"}:
        raise ValueError(f"Unsupported sft.eval.eval_strategy={eval_cfg.eval_strategy!r}.")

    dpo_cfg = config.rlhf.dpo
    dpo_cfg.loss_type = str(dpo_cfg.loss_type).strip().lower()
    if dpo_cfg.loss_type not in _DPO_LOSS_TYPES:
        raise ValueError(f"Unsupported rlhf.dpo.loss_type={dpo_cfg.loss_type!r}.")
    if float(dpo_cfg.beta) <= 0:
        raise ValueError("rlhf.dpo.beta must be > 0.")
    if not (0.0 <= float(dpo_cfg.label_smoothing) < 1.0):
        raise ValueError("rlhf.dpo.label_smoothing must be in [0, 1).")

    ppo_cfg = config.rlhf.ppo
    if not (0.0 < float(ppo_cfg.cliprange) < 1.0):
        raise ValueError("rlhf.ppo.cliprange must be in (0, 1).")
    if not (0.0 < float(ppo_cfg.cliprange_value) < 1.0):
        raise ValueError("rlhf.ppo.cliprange_value must be in (0, 1).")
    if float(ppo_cfg.kl_coef) < 0:
        raise ValueError("rlhf.ppo.kl_coef must be >= 0.")
    if float(ppo_cfg.vf_coef) < 0:
        raise ValueError("rlhf.ppo.vf_coef must be >= 0.")
    if not (0.0 <= float(ppo_cfg.gamma) <= 1.0):
        raise ValueError("rlhf.ppo.gamma must be in [0, 1].")
    if not (0.0 <= float(ppo_cfg.lam) <= 1.0):
        raise ValueError("rlhf.ppo.lam must be in [0, 1].")
    if int(ppo_cfg.response_length) <= 0:
        raise ValueError("rlhf.ppo.response_length must be > 0.")
    if float(ppo_cfg.temperature) <= 0:
        raise ValueError("rlhf.ppo.temperature must be > 0.")
    if int(ppo_cfg.num_ppo_epochs) <= 0:
        raise ValueError("rlhf.ppo.num_ppo_epochs must be > 0.")
    if int(ppo_cfg.num_mini_batches) <= 0:
        raise ValueError("rlhf.ppo.num_mini_batches must be > 0.")
    if int(ppo_cfg.local_rollout_forward_batch_size) <= 0:
        raise ValueError("rlhf.ppo.local_rollout_forward_batch_size must be > 0.")
    if int(ppo_cfg.num_sample_generations) < 0:
        raise ValueError("rlhf.ppo.num_sample_generations must be >= 0.")
    if ppo_cfg.stop_token is not None:
        ppo_cfg.stop_token = str(ppo_cfg.stop_token).strip().lower() or None
    ppo_cfg.value_model_mode = str(ppo_cfg.value_model_mode).strip().lower()
    if ppo_cfg.value_model_mode not in _PPO_VALUE_MODEL_MODES:
        raise ValueError(
            f"Unsupported rlhf.ppo.value_model_mode={ppo_cfg.value_model_mode!r}. "
            f"Expected one of {_PPO_VALUE_MODEL_MODES}."
        )
    ppo_cfg.reward_model_mode = str(ppo_cfg.reward_model_mode).strip().lower()
    if ppo_cfg.reward_model_mode not in _PPO_REWARD_MODEL_MODES:
        raise ValueError(
            f"Unsupported rlhf.ppo.reward_model_mode={ppo_cfg.reward_model_mode!r}. "
            f"Expected one of {_PPO_REWARD_MODEL_MODES}."
        )
    ppo_cfg.allow_untrained_reward_model = bool(ppo_cfg.allow_untrained_reward_model)
    ppo_cfg.allow_text_only_multimodal_ppo = bool(ppo_cfg.allow_text_only_multimodal_ppo)

    config.plugins.hooks = [str(x).strip().lower() for x in config.plugins.hooks if str(x).strip()]
    config.plugins.interceptors = [
        str(x).strip().lower() for x in config.plugins.interceptors if str(x).strip()
    ]

    config.logging.level = str(config.logging.level).strip().upper()
    if config.logging.level not in _LOG_LEVELS:
        raise ValueError(f"Unsupported logging.level={config.logging.level!r}.")
    config.logging.fmt = str(config.logging.fmt).strip().lower()
    if config.logging.fmt not in _LOG_FORMATS:
        raise ValueError(f"Unsupported logging.fmt={config.logging.fmt!r}.")
    config.logging.rank_zero_only = bool(config.logging.rank_zero_only)

    config.progress.mininterval = float(config.progress.mininterval)
    if config.progress.mininterval <= 0:
        raise ValueError("progress.mininterval must be > 0.")
    return config
