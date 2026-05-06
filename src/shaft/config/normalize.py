from __future__ import annotations

import re

from .runtime import RuntimeConfig

_MIX_STRATEGIES = {"concat", "interleave_under", "interleave_over"}
_MIX_REFRESH_MODES = {"static", "epoch_refresh"}
_ALGORITHMS = {"sft", "dpo", "ppo", "grpo"}
_FINETUNE_MODES = {"full", "lora", "dora", "qlora"}
_LOSS_NAMES = {"auto", "causal_lm"}
_DPO_LOSS_TYPES = {"sigmoid"}
_PPO_VALUE_MODEL_MODES = {"shared_backbone", "copy_backbone"}
_PPO_REWARD_MODEL_MODES = {"adapter_disabled_policy", "copy_backbone"}
_LOG_LEVELS = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}
_LOG_FORMATS = {"text", "json"}
_ONLINE_EVAL_NORMALIZERS = {"identity", "range"}
_FREEZE_GROUPS = {"language_model", "vision_tower", "aligner", "generator"}
_PARAM_GROUP_LR_KEYS = {
    "language_model",
    "vision_tower",
    "aligner",
    "generator",
    "lora_params",
    "modules_to_save",
}


def _normalize_string_list(values: list[str]) -> list[str]:
    normalized = [str(item).strip() for item in values if str(item).strip()]
    return list(dict.fromkeys(normalized))


def _validate_optional_regex(value: str | None, field_name: str) -> str | None:
    normalized = str(value).strip() if value is not None else ""
    if not normalized:
        return None
    try:
        re.compile(normalized)
    except re.error as exc:
        raise ValueError(f"{field_name} is not a valid regex: {exc}") from exc
    return normalized


def normalize_runtime_config(config: RuntimeConfig) -> RuntimeConfig:
    config.algorithm.name = str(config.algorithm.name).strip().lower()
    if config.algorithm.name not in _ALGORITHMS:
        raise ValueError(f"Unsupported algorithm.name={config.algorithm.name!r}. Expected one of {_ALGORITHMS}.")

    config.data.mix_strategy = str(config.data.mix_strategy).strip().lower()
    if config.data.mix_strategy not in _MIX_STRATEGIES:
        raise ValueError(f"Unsupported data.mix_strategy={config.data.mix_strategy!r}.")
    config.data.mix_refresh = str(config.data.mix_refresh).strip().lower()
    if config.data.mix_refresh not in _MIX_REFRESH_MODES:
        raise ValueError(f"Unsupported data.mix_refresh={config.data.mix_refresh!r}.")
    if config.algorithm.name == "grpo" and config.data.mix_refresh != "static":
        raise ValueError(
            "GRPO currently requires data.mix_refresh='static' because TRL GRPO uses its own repeated "
            "train sampler for grouped generations."
        )
    config.data.catalog_names = [str(x).strip() for x in config.data.catalog_names if str(x).strip()]
    if config.data.catalog_path is not None:
        config.data.catalog_path = str(config.data.catalog_path).strip() or None

    finetune = config.model.finetune
    finetune.mode = str(finetune.mode).strip().lower()
    if finetune.mode not in _FINETUNE_MODES:
        raise ValueError(f"Unsupported model.finetune.mode={finetune.mode!r}.")
    finetune.lora_bias = str(finetune.lora_bias).strip().lower()
    finetune.freeze.groups = _normalize_string_list([str(value).lower() for value in finetune.freeze.groups])
    invalid_groups = sorted(set(finetune.freeze.groups) - _FREEZE_GROUPS)
    if invalid_groups:
        raise ValueError(
            f"Unsupported model.finetune.freeze.groups={invalid_groups!r}. Expected only {_FREEZE_GROUPS}."
        )
    finetune.freeze.prefixes = _normalize_string_list(finetune.freeze.prefixes)
    finetune.freeze.trainable_prefixes = _normalize_string_list(finetune.freeze.trainable_prefixes)
    finetune.freeze.regex = _validate_optional_regex(
        finetune.freeze.regex,
        "model.finetune.freeze.regex",
    )
    finetune.freeze.trainable_regex = _validate_optional_regex(
        finetune.freeze.trainable_regex,
        "model.finetune.freeze.trainable_regex",
    )
    if not finetune.target_modules:
        finetune.target_modules = ["auto"]
    finetune.target_modules = _normalize_string_list(finetune.target_modules)
    if not finetune.target_modules:
        raise ValueError("model.finetune.target_modules cannot be empty.")

    for dataset in config.data.datasets:
        dataset.dataset_name = str(dataset.dataset_name).strip()
        if not dataset.dataset_name:
            raise ValueError("data.datasets[*].dataset_name cannot be empty.")
        dataset.source_type = str(dataset.source_type).strip().lower()
        dataset.enabled = bool(dataset.enabled)
        dataset.use_for_eval = bool(dataset.use_for_eval)
        dataset.train_paths = [str(x).strip() for x in dataset.train_paths if str(x).strip()]
        dataset.val_paths = [str(x).strip() for x in dataset.val_paths if str(x).strip()]
        dataset.offline_transforms = _normalize_string_list(dataset.offline_transforms)
        dataset.online_transforms = _normalize_string_list(dataset.online_transforms)
        dataset.tags = _normalize_string_list(dataset.tags)
        if dataset.help is not None:
            dataset.help = str(dataset.help).strip() or None
        if dataset.train_path:
            dataset.train_paths = [str(dataset.train_path).strip(), *dataset.train_paths]
        if dataset.val_path:
            dataset.val_paths = [str(dataset.val_path).strip(), *dataset.val_paths]
        if not dataset.train_paths:
            raise ValueError(f"data.datasets[{dataset.dataset_name}].train_paths cannot be empty.")
        if not dataset.val_paths and bool(config.eval.enabled) and dataset.enabled and dataset.use_for_eval:
            raise ValueError(f"data.datasets[{dataset.dataset_name}].val_paths cannot be empty.")
        if config.algorithm.name == "sft" and dataset.source_type == "jsonl_ppo":
            raise ValueError(f"data.datasets[{dataset.dataset_name}] uses jsonl_ppo but algorithm is sft.")
        if config.algorithm.name == "sft" and dataset.source_type == "jsonl_dpo":
            raise ValueError(f"data.datasets[{dataset.dataset_name}] uses jsonl_dpo but algorithm is sft.")
        if config.algorithm.name == "dpo" and dataset.source_type == "jsonl_sft":
            raise ValueError(f"data.datasets[{dataset.dataset_name}] uses jsonl_sft but algorithm is dpo.")
        if config.algorithm.name == "dpo" and dataset.source_type == "jsonl_ppo":
            raise ValueError(f"data.datasets[{dataset.dataset_name}] uses jsonl_ppo but algorithm is dpo.")
        if config.algorithm.name == "ppo" and dataset.source_type == "jsonl_sft":
            raise ValueError(f"data.datasets[{dataset.dataset_name}] uses jsonl_sft but algorithm is ppo.")
        if config.algorithm.name == "ppo" and dataset.source_type == "jsonl_dpo":
            raise ValueError(f"data.datasets[{dataset.dataset_name}] uses jsonl_dpo but algorithm is ppo.")
        if config.algorithm.name == "grpo" and dataset.source_type != "jsonl_sft":
            raise ValueError(
                f"data.datasets[{dataset.dataset_name}] uses {dataset.source_type} but algorithm is grpo. "
                "GRPO currently expects jsonl_sft data."
            )

    if bool(config.eval.enabled):
        has_eval_dataset = any(
            dataset.enabled and dataset.use_for_eval
            for dataset in config.data.datasets
        )
        if not has_eval_dataset:
            raise ValueError(
                "eval.enabled=true requires at least one dataset with use_for_eval=true."
            )

    train = config.train
    train.optimizer_name = str(train.optimizer_name).strip().lower()
    train.scheduler_name = str(train.scheduler_name).strip().lower()
    if train.scheduler_name in {"", "auto"}:
        train.scheduler_name = str(train.lr_scheduler_type).strip().lower()
    train.loss_name = str(train.loss_name).strip().lower()
    if train.loss_name not in _LOSS_NAMES:
        raise ValueError(f"Unsupported train.loss_name={train.loss_name!r}.")
    train.loss_scale = str(train.loss_scale).strip().lower() or "default"
    from shaft.loss_scale import build_loss_scale
    try:
        build_loss_scale(train.loss_scale)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Unsupported train.loss_scale={train.loss_scale!r}.") from exc
    train.lr_scheduler_type = str(train.lr_scheduler_type).strip().lower()
    if isinstance(train.save_strategy, bool):
        train.save_strategy = "no" if not train.save_strategy else "steps"
    else:
        train.save_strategy = str(train.save_strategy).strip().lower()
    if train.save_strategy not in {"no", "steps", "epoch"}:
        raise ValueError(f"Unsupported train.save_strategy={train.save_strategy!r}.")
    if int(train.save_epoch_interval) <= 0:
        raise ValueError("train.save_epoch_interval must be > 0.")
    if int(train.epochs) <= 0:
        raise ValueError("train.epochs must be > 0.")
    if int(train.max_steps) == 0:
        raise ValueError("train.max_steps cannot be 0. Use -1 or >0.")
    if float(train.scheduler_num_cycles) <= 0:
        raise ValueError("train.scheduler_num_cycles must be > 0.")
    if float(train.scheduler_power) <= 0:
        raise ValueError("train.scheduler_power must be > 0.")
    normalized_param_group_lrs: dict[str, float] = {}
    for key, value in dict(train.param_group_lrs).items():
        normalized_key = str(key).strip().lower()
        if not normalized_key:
            raise ValueError("train.param_group_lrs contains an empty key.")
        if normalized_key not in _PARAM_GROUP_LR_KEYS:
            raise ValueError(
                f"Unsupported train.param_group_lrs key={normalized_key!r}. "
                f"Expected only {_PARAM_GROUP_LR_KEYS}."
            )
        try:
            normalized_value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"train.param_group_lrs[{normalized_key!r}] must be a positive float."
            ) from exc
        if normalized_value <= 0:
            raise ValueError(
                f"train.param_group_lrs[{normalized_key!r}] must be > 0."
            )
        normalized_param_group_lrs[normalized_key] = normalized_value
    train.param_group_lrs = normalized_param_group_lrs
    normalized_no_decay_name_patterns: list[str] = []
    seen_no_decay_name_patterns: set[str] = set()
    for raw_pattern in list(train.no_decay_name_patterns):
        normalized_pattern = str(raw_pattern).strip().lower()
        if not normalized_pattern or normalized_pattern in seen_no_decay_name_patterns:
            continue
        seen_no_decay_name_patterns.add(normalized_pattern)
        normalized_no_decay_name_patterns.append(normalized_pattern)
    train.no_decay_name_patterns = normalized_no_decay_name_patterns

    eval_cfg = config.eval
    if isinstance(eval_cfg.eval_strategy, bool):
        eval_cfg.eval_strategy = "no" if not eval_cfg.eval_strategy else "steps"
    else:
        eval_cfg.eval_strategy = str(eval_cfg.eval_strategy).strip().lower()
    if eval_cfg.eval_strategy not in {"no", "steps", "epoch"}:
        raise ValueError(f"Unsupported eval.eval_strategy={eval_cfg.eval_strategy!r}.")
    if int(eval_cfg.epoch_interval) <= 0:
        raise ValueError("eval.epoch_interval must be > 0.")
    if int(eval_cfg.max_new_tokens) <= 0:
        raise ValueError("eval.max_new_tokens must be > 0.")
    eval_cfg.metric_for_best_model = str(eval_cfg.metric_for_best_model).strip()
    if not eval_cfg.metric_for_best_model:
        raise ValueError("eval.metric_for_best_model cannot be empty.")
    eval_cfg.loss_metrics_enabled = bool(eval_cfg.loss_metrics_enabled)
    eval_cfg.online_metrics_enabled = bool(eval_cfg.online_metrics_enabled)
    normalized_policies: dict[str, object] = {}
    for dataset_name, policy in config.eval.datasets.items():
        normalized_name = str(dataset_name).strip()
        if not normalized_name:
            raise ValueError("eval.datasets contains an empty dataset key.")
        policy.prediction_codec = str(policy.prediction_codec).strip().lower()
        if not policy.prediction_codec:
            raise ValueError(f"eval.datasets.{normalized_name}.prediction_codec cannot be empty.")
        policy.target_adapter = str(policy.target_adapter).strip().lower()
        if not policy.target_adapter:
            raise ValueError(f"eval.datasets.{normalized_name}.target_adapter cannot be empty.")
        if not isinstance(policy.target_adapter_params, dict):
            raise ValueError(f"eval.datasets.{normalized_name}.target_adapter_params must be a mapping.")
        normalized_metrics: list[object] = []
        seen_metric_names: set[str] = set()
        for metric in policy.metrics:
            metric.name = str(metric.name).strip().lower()
            if not metric.name:
                raise ValueError(f"eval.datasets.{normalized_name}.metrics[*].name cannot be empty.")
            if metric.name in seen_metric_names:
                raise ValueError(
                    f"eval.datasets.{normalized_name}.metrics contains duplicate metric {metric.name!r}."
                )
            if not isinstance(metric.params, dict):
                raise ValueError(
                    f"eval.datasets.{normalized_name}.metrics[{metric.name}].params must be a mapping."
                )
            seen_metric_names.add(metric.name)
            normalized_metrics.append(metric)
        policy.metrics = normalized_metrics
        policy.primary_metric = str(policy.primary_metric).strip().lower()
        if policy.primary_metric and policy.primary_metric not in seen_metric_names:
            raise ValueError(
                f"eval.datasets.{normalized_name}.primary_metric={policy.primary_metric!r} "
                "must appear in metrics."
            )
        policy.normalizer.type = str(policy.normalizer.type).strip().lower()
        if policy.normalizer.type not in _ONLINE_EVAL_NORMALIZERS:
            raise ValueError(
                f"Unsupported eval.datasets.{normalized_name}.normalizer.type={policy.normalizer.type!r}."
            )
        if policy.normalizer.type == "range":
            if policy.normalizer.min_value is None or policy.normalizer.max_value is None:
                raise ValueError(
                    f"eval.datasets.{normalized_name}.normalizer range requires min_value and max_value."
                )
            if float(policy.normalizer.max_value) <= float(policy.normalizer.min_value):
                raise ValueError(
                    f"eval.datasets.{normalized_name}.normalizer.max_value must be > min_value."
                )
        if float(policy.weight) <= 0:
            raise ValueError(f"eval.datasets.{normalized_name}.weight must be > 0.")
        normalized_policies[normalized_name] = policy
    eval_cfg.datasets = normalized_policies
    configured_dataset_names = {
        dataset.dataset_name
        for dataset in config.data.datasets
        if dataset.enabled and dataset.use_for_eval
    }
    if eval_cfg.datasets or eval_cfg.online_metrics_enabled or eval_cfg.metric_for_best_model == "eval_final_loss":
        if not eval_cfg.enabled:
            raise ValueError("dataset-policy eval requires eval.enabled=true.")
        if not eval_cfg.datasets:
            raise ValueError(
                "dataset-policy eval requires eval.datasets to be configured for final_loss/final_score aggregation."
            )
        missing_policies = sorted(configured_dataset_names - set(eval_cfg.datasets.keys()))
        if missing_policies:
            raise ValueError(
                f"dataset-policy eval is missing policies for datasets: {missing_policies}."
            )
        unknown_policies = sorted(set(eval_cfg.datasets.keys()) - configured_dataset_names)
        if unknown_policies:
            raise ValueError(
                f"eval.datasets contains unknown dataset policies: {unknown_policies}."
            )
    if eval_cfg.online_metrics_enabled:
        from shaft.codec import CODEC_REGISTRY
        from shaft.metrics import EVAL_METRIC_REGISTRY
        from shaft.training.online_eval import TARGET_ADAPTER_REGISTRY

        if config.algorithm.name not in {"sft", "grpo"}:
            raise ValueError(
                "eval.online_metrics_enabled is currently only supported for "
                "algorithm.name in {'sft', 'grpo'}."
            )
        if eval_cfg.do_sample:
            raise ValueError("eval.online_metrics_enabled requires greedy decoding; set eval.do_sample=false.")
        for dataset_name, policy in eval_cfg.datasets.items():
            if not CODEC_REGISTRY.has(policy.prediction_codec):
                raise ValueError(
                    f"eval.datasets.{dataset_name}.prediction_codec={policy.prediction_codec!r} is unregistered. "
                    f"Registered codecs: {sorted(CODEC_REGISTRY.keys())}."
                )
            if not TARGET_ADAPTER_REGISTRY.has(policy.target_adapter):
                raise ValueError(
                    f"eval.datasets.{dataset_name}.target_adapter={policy.target_adapter!r} is unregistered. "
                    f"Registered target adapters: {sorted(TARGET_ADAPTER_REGISTRY.keys())}."
                )
            if not policy.metrics:
                raise ValueError(f"eval.datasets.{dataset_name}.metrics cannot be empty.")
            if not policy.primary_metric:
                raise ValueError(f"eval.datasets.{dataset_name}.primary_metric cannot be empty.")
            for metric in policy.metrics:
                if not EVAL_METRIC_REGISTRY.has(metric.name):
                    raise ValueError(
                        f"eval.datasets.{dataset_name}.metrics includes unregistered metric {metric.name!r}. "
                        f"Registered metrics: {sorted(EVAL_METRIC_REGISTRY.keys())}."
                    )
    if eval_cfg.metric_for_best_model == "eval_loss":
        if eval_cfg.online_metrics_enabled and eval_cfg.datasets:
            eval_cfg.metric_for_best_model = "eval_final_score"
            eval_cfg.greater_is_better = True
        elif eval_cfg.datasets and eval_cfg.loss_metrics_enabled:
            eval_cfg.metric_for_best_model = "eval_final_loss"
            eval_cfg.greater_is_better = False
    elif eval_cfg.metric_for_best_model == "eval_final_score":
        if not eval_cfg.online_metrics_enabled:
            raise ValueError("eval.metric_for_best_model=eval_final_score requires eval.online_metrics_enabled=true.")
        eval_cfg.greater_is_better = True
    elif eval_cfg.metric_for_best_model == "eval_final_loss":
        if not eval_cfg.loss_metrics_enabled:
            raise ValueError("eval.metric_for_best_model=eval_final_loss requires eval.loss_metrics_enabled=true.")
        if not eval_cfg.datasets:
            raise ValueError("eval.metric_for_best_model=eval_final_loss requires eval.datasets to be configured.")
        eval_cfg.greater_is_better = False

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

    grpo_cfg = config.rlhf.grpo
    if float(grpo_cfg.beta) < 0:
        raise ValueError("rlhf.grpo.beta must be >= 0.")

    rollout_cfg = grpo_cfg.rollout
    vllm_cfg = grpo_cfg.vllm
    # Backward-compatible flat aliases. New configs should use rollout/vllm as the
    # canonical GRPO runtime structure.
    if grpo_cfg.num_generations is not None:
        rollout_cfg.num_generations = int(grpo_cfg.num_generations)
    if grpo_cfg.num_generations_eval is not None:
        rollout_cfg.num_generations_eval = int(grpo_cfg.num_generations_eval)
    if grpo_cfg.max_completion_length is not None:
        rollout_cfg.max_completion_length = int(grpo_cfg.max_completion_length)
    if grpo_cfg.temperature is not None:
        rollout_cfg.temperature = float(grpo_cfg.temperature)
    if grpo_cfg.top_p is not None:
        rollout_cfg.top_p = float(grpo_cfg.top_p)
    if grpo_cfg.top_k is not None:
        rollout_cfg.top_k = int(grpo_cfg.top_k)
    if grpo_cfg.min_p is not None:
        rollout_cfg.min_p = float(grpo_cfg.min_p)
    if grpo_cfg.repetition_penalty is not None:
        rollout_cfg.repetition_penalty = float(grpo_cfg.repetition_penalty)
    if grpo_cfg.use_vllm is not None:
        vllm_cfg.enabled = bool(grpo_cfg.use_vllm)

    if int(rollout_cfg.num_generations) <= 0:
        raise ValueError("rlhf.grpo.rollout.num_generations must be > 0.")
    if rollout_cfg.num_generations_eval is not None and int(rollout_cfg.num_generations_eval) <= 0:
        raise ValueError("rlhf.grpo.rollout.num_generations_eval must be > 0 when configured.")
    if int(rollout_cfg.max_completion_length) <= 0:
        raise ValueError("rlhf.grpo.rollout.max_completion_length must be > 0.")
    if float(rollout_cfg.temperature) <= 0:
        raise ValueError("rlhf.grpo.rollout.temperature must be > 0.")
    if not (0.0 < float(rollout_cfg.top_p) <= 1.0):
        raise ValueError("rlhf.grpo.rollout.top_p must be in (0, 1].")
    if int(rollout_cfg.top_k) < 0:
        raise ValueError("rlhf.grpo.rollout.top_k must be >= 0.")
    if rollout_cfg.min_p is not None and not (0.0 <= float(rollout_cfg.min_p) <= 1.0):
        raise ValueError("rlhf.grpo.rollout.min_p must be in [0, 1].")
    if float(rollout_cfg.repetition_penalty) <= 0:
        raise ValueError("rlhf.grpo.rollout.repetition_penalty must be > 0.")
    if not isinstance(rollout_cfg.generation_kwargs, dict):
        raise ValueError("rlhf.grpo.rollout.generation_kwargs must be a mapping.")
    rollout_cfg.cache_implementation = (
        str(rollout_cfg.cache_implementation).strip()
        if rollout_cfg.cache_implementation is not None
        else None
    )
    rollout_cfg.use_transformers_paged = bool(rollout_cfg.use_transformers_paged)

    vllm_cfg.enabled = bool(vllm_cfg.enabled)
    vllm_cfg.mode = str(vllm_cfg.mode).strip().lower()
    if vllm_cfg.mode not in {"server", "colocate"}:
        raise ValueError("rlhf.grpo.vllm.mode must be 'server' or 'colocate'.")
    vllm_cfg.model_impl = str(vllm_cfg.model_impl).strip().lower()
    if vllm_cfg.model_impl not in {"vllm", "transformers"}:
        raise ValueError("rlhf.grpo.vllm.model_impl must be 'vllm' or 'transformers'.")
    vllm_cfg.enable_sleep_mode = bool(vllm_cfg.enable_sleep_mode)
    vllm_cfg.structured_outputs_regex = (
        str(vllm_cfg.structured_outputs_regex)
        if vllm_cfg.structured_outputs_regex is not None
        else None
    )
    vllm_cfg.server_base_url = (
        str(vllm_cfg.server_base_url).strip() if vllm_cfg.server_base_url is not None else None
    )
    vllm_cfg.server_host = str(vllm_cfg.server_host).strip() or "0.0.0.0"
    if int(vllm_cfg.server_port) <= 0:
        raise ValueError("rlhf.grpo.vllm.server_port must be > 0.")
    if float(vllm_cfg.server_timeout) <= 0:
        raise ValueError("rlhf.grpo.vllm.server_timeout must be > 0.")
    if int(vllm_cfg.group_port) <= 0:
        raise ValueError("rlhf.grpo.vllm.group_port must be > 0.")
    if not (0.0 < float(vllm_cfg.gpu_memory_utilization) <= 1.0):
        raise ValueError("rlhf.grpo.vllm.gpu_memory_utilization must be in (0, 1].")
    if vllm_cfg.max_model_length is not None and int(vllm_cfg.max_model_length) <= 0:
        raise ValueError("rlhf.grpo.vllm.max_model_length must be > 0 when configured.")
    if int(vllm_cfg.tensor_parallel_size) <= 0:
        raise ValueError("rlhf.grpo.vllm.tensor_parallel_size must be > 0.")

    grpo_cfg.num_generations = int(rollout_cfg.num_generations)
    grpo_cfg.num_generations_eval = (
        int(rollout_cfg.num_generations_eval)
        if rollout_cfg.num_generations_eval is not None
        else None
    )
    grpo_cfg.max_completion_length = int(rollout_cfg.max_completion_length)
    grpo_cfg.temperature = float(rollout_cfg.temperature)
    grpo_cfg.top_p = float(rollout_cfg.top_p)
    grpo_cfg.top_k = int(rollout_cfg.top_k)
    grpo_cfg.min_p = float(rollout_cfg.min_p) if rollout_cfg.min_p is not None else None
    grpo_cfg.repetition_penalty = float(rollout_cfg.repetition_penalty)
    grpo_cfg.use_vllm = bool(vllm_cfg.enabled)
    if not grpo_cfg.reward_functions:
        raise ValueError("rlhf.grpo.reward_functions cannot be empty.")
    from shaft.algorithms.grpo_rewards import GRPO_REWARD_REGISTRY
    from shaft.codec import CODEC_REGISTRY

    normalized_rewards: list[object] = []
    for reward in grpo_cfg.reward_functions:
        reward.name = str(reward.name).strip().lower()
        if not reward.name:
            raise ValueError("rlhf.grpo.reward_functions[*].name cannot be empty.")
        if not GRPO_REWARD_REGISTRY.has(reward.name):
            raise ValueError(
                f"rlhf.grpo.reward_functions[{reward.name}].name is unregistered. "
                f"Registered rewards: {sorted(GRPO_REWARD_REGISTRY.keys())}."
            )
        reward.codec = str(reward.codec).strip().lower()
        if not reward.codec:
            raise ValueError(f"rlhf.grpo.reward_functions[{reward.name}].codec cannot be empty.")
        if not CODEC_REGISTRY.has(reward.codec):
            raise ValueError(
                f"rlhf.grpo.reward_functions[{reward.name}].codec={reward.codec!r} is unregistered. "
                f"Registered codecs: {sorted(CODEC_REGISTRY.keys())}."
            )
        if float(reward.weight) <= 0:
            raise ValueError(f"rlhf.grpo.reward_functions[{reward.name}].weight must be > 0.")
        if not isinstance(reward.params, dict):
            raise ValueError(f"rlhf.grpo.reward_functions[{reward.name}].params must be a mapping.")
        normalized_rewards.append(reward)
    grpo_cfg.reward_functions = normalized_rewards

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
