from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin, get_type_hints

import yaml
from vlm_structgen.core.routing import normalize_route_key
from vlm_structgen.core.utils.logging import get_vlm_logger


@dataclass
class ExperimentConfig:
    name: str = "qwen3vl-exp"
    output_dir: str = "outputs/qwen3vl-exp"
    seed: int = 42


@dataclass
class ModelConfig:
    model_name_or_path: str = "models/Qwen3-VL-2B-Instruct"
    remote_model_name_or_path: str = "Qwen/Qwen3-VL-2B-Instruct"
    trust_remote_code: bool = True
    attn_implementation: str | None = "flash_attention_2"
    freeze_vision_tower: bool = True
    train_projector: bool = False
    vision_name_substrings: list[str] = field(default_factory=lambda: ["visual"])
    projector_name_substrings: list[str] = field(
        default_factory=lambda: ["merger", "projector", "multi_modal_projector"]
    )


@dataclass
class TokenizerConfig:
    # Relative grounding coordinates are normalized to integer bins in [0, num_bins - 1].
    num_bins: int = 1000
    add_eos_token: bool = True


@dataclass
class PromptConfig:
    profile: str | None = None
    system_prompt: str = ""
    system_prompt_template: str | None = None
    user_prompt: str = "Output only valid JSON. No markdown and no extra text."
    user_prompt_template: str | None = None
    # Optional per-route prompt overrides for multi-task training.
    # Key format: opaque route id string, for example "grounding/arrow".
    route_prompts: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class TaskConfig:
    # Optional default route for one-stage inference.
    # Training resolves routes from per-record route in JSONL.
    route: str | None = None
    # LLaMAFactory-style mixed training strategy:
    # - concat
    # - interleave_under
    # - interleave_over
    mix_strategy: str = "interleave_under"
    route_options: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class DataConfig:
    # Dataset registry mode:
    # - registry_path points to a YAML registry file.
    # - train_datasets/val_datasets list dataset ids in that registry.
    # Training paths/routes are resolved strictly from the registry.
    registry_path: str | None = None
    train_datasets: list[str] = field(default_factory=list)
    val_datasets: list[str] = field(default_factory=list)
    # Global pixel budget defaults for training/evaluation collation.
    # Route-specific values in route_pixel_budgets override these defaults.
    min_pixels: int | None = 200704
    max_pixels: int | None = 1048576
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    # Optional route-level pixel budgets for multi-task training.
    # Key format: "task_type/domain_type", for example "grounding/arrow".
    # Value format: {"min_pixels": int | None, "max_pixels": int | None}
    route_pixel_budgets: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class LoraConfig:
    enabled: bool = True
    adapter_type: str = "lora"
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    bias: str = "none"
    use_rslora: bool = False
    lang_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    vis_target_modules: list[str] = field(
        default_factory=lambda: [
            "attn.qkv",
            "attn.proj",
            "mlp.linear_fc1",
            "mlp.linear_fc2",
        ]
    )
    proj_target_modules: list[str] = field(default_factory=list)


@dataclass
class FineTuneConfig:
    mode: str = "lora"


@dataclass
class TrainConfig:
    epochs: int = 3
    per_device_batch_size: int = 1
    grad_accum_steps: int = 8
    gradient_checkpointing: bool = True
    learning_rate: float = 1e-4
    embed_learning_rate: float | None = None
    lm_head_learning_rate: float | None = None
    lora_learning_rate: float | None = None
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0
    bf16: bool = True
    eval_strategy: str = "epoch"
    eval_start_epoch: int = 1
    log_every_steps: int = 10
    eval_every_steps: int = 200
    save_every_steps: int = 200
    save_step_checkpoints: bool = False
    keep_last_n_checkpoints: int = 3
    find_unused_parameters: bool = False


@dataclass
class EvalConfig:
    per_device_batch_size: int = 1
    bucket_by_target_length: bool = True
    max_new_tokens: int = 8192
    num_beams: int = 1
    do_sample: bool = False
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    use_cache: bool = True
    # Primary metric used for best-checkpoint selection.
    best_metric: str = "val/multi_task_score"
    monitor_mode: str = "max"


@dataclass
class LoggingConfig:
    use_wandb: bool = True
    project: str = "vlm_structgen_json"
    run_name: str | None = None
    progress_ncols: int = 88


@dataclass
class CheckpointConfig:
    init_from: str | None = None
    resume_from: str | None = None


@dataclass
class ExperimentRuntimeConfig:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    data: DataConfig = field(default_factory=DataConfig)
    finetune: FineTuneConfig = field(default_factory=FineTuneConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

T = TypeVar("T")


def _unwrap_dataclass_annotation(annotation: Any) -> type[Any] | None:
    if is_dataclass(annotation):
        return annotation
    origin = get_origin(annotation)
    if origin is None:
        return None
    args = [arg for arg in get_args(annotation) if arg is not type(None)]
    if len(args) != 1:
        return None
    candidate = args[0]
    if is_dataclass(candidate):
        return candidate
    return None


def _warn_unknown_config_keys(
    cls: type[Any],
    data: dict[str, Any],
    *,
    path: str,
) -> None:
    if not isinstance(data, dict):
        return

    logger = get_vlm_logger()
    type_hints = get_type_hints(cls)
    field_names = {field_info.name for field_info in fields(cls)}
    unknown_keys = sorted(set(data.keys()) - field_names)
    for key in unknown_keys:
        logger.warning(
            "Unknown config key ignored: %s%s",
            f"{path}." if path else "",
            key,
        )

    for field_info in fields(cls):
        if field_info.name not in data:
            continue
        annotation = type_hints.get(field_info.name, field_info.type)
        nested_cls = _unwrap_dataclass_annotation(annotation)
        if nested_cls is None:
            continue
        nested_data = data[field_info.name]
        if not isinstance(nested_data, dict):
            continue
        next_path = f"{path}.{field_info.name}" if path else field_info.name
        _warn_unknown_config_keys(nested_cls, nested_data, path=next_path)


def _convert_value(value: Any, annotation: Any) -> Any:
    origin = get_origin(annotation)
    if is_dataclass(annotation):
        return _from_dict(annotation, value)
    if origin is list:
        item_type = get_args(annotation)[0]
        return [_convert_value(item, item_type) for item in value]
    if origin is dict:
        key_type, value_type = get_args(annotation)
        return {
            _convert_value(key, key_type): _convert_value(item, value_type)
            for key, item in value.items()
        }
    if origin is tuple:
        item_types = get_args(annotation)
        return tuple(_convert_value(item, t) for item, t in zip(value, item_types))
    if origin is None:
        return value
    if origin is not None:
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(args) == 1:
            return _convert_value(value, args[0])
    return value


def _from_dict(cls: type[T], data: dict[str, Any]) -> T:
    type_hints = get_type_hints(cls)
    kwargs = {}
    for field_info in fields(cls):
        if field_info.name not in data:
            continue
        annotation = type_hints.get(field_info.name, field_info.type)
        kwargs[field_info.name] = _convert_value(data[field_info.name], annotation)
    return cls(**kwargs)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _raise_deprecated_config_keys(yaml_payload: dict[str, Any]) -> None:
    eval_payload = yaml_payload.get("eval")
    if isinstance(eval_payload, dict) and "monitor_metric" in eval_payload:
        raise ValueError(
            "`eval.monitor_metric` has been removed. "
            "Please migrate to `eval.best_metric`."
        )
    if isinstance(eval_payload, dict):
        removed_eval_keys = [
            "bbox_iou_threshold",
            "strict_point_distance_px",
        ]
        used_eval_keys = [key for key in removed_eval_keys if key in eval_payload]
        if used_eval_keys:
            raise ValueError(
                "Legacy eval fields have been removed: "
                f"{used_eval_keys}. "
                "Please configure per-task evaluation thresholds under task.route_options."
            )
    model_payload = yaml_payload.get("model")
    if isinstance(model_payload, dict):
        used_model_pixel_keys = [key for key in ("min_pixels", "max_pixels") if key in model_payload]
        if used_model_pixel_keys:
            raise ValueError(
                "Training pixel budget fields under `model` have been removed: "
                f"{used_model_pixel_keys}. "
                "Please migrate to data.min_pixels/data.max_pixels "
                "and optional data.route_pixel_budgets."
            )
    data_payload = yaml_payload.get("data")
    if isinstance(data_payload, dict):
        deprecated_data_keys = [
            "train_path",
            "val_path",
            "train_route",
            "val_route",
            "train_route_map",
            "val_route_map",
        ]
        used_keys = [key for key in deprecated_data_keys if key in data_payload]
        if used_keys:
            raise ValueError(
                "Legacy data fields have been removed: "
                f"{used_keys}. "
                "Please migrate to data.registry_path + data.train_datasets/data.val_datasets."
            )
    task_payload = yaml_payload.get("task")
    if isinstance(task_payload, dict):
        route_options_payload = task_payload.get("route_options")
        if isinstance(route_options_payload, dict):
            legacy_pixel_budget_routes: dict[str, list[str]] = {}
            for route_key, route_options in route_options_payload.items():
                if not isinstance(route_options, dict):
                    continue
                used_keys = [key for key in ("min_pixels", "max_pixels") if key in route_options]
                if used_keys:
                    legacy_pixel_budget_routes[str(route_key)] = used_keys
            if legacy_pixel_budget_routes:
                raise ValueError(
                    "Per-route pixel budget fields in task.route_options have been removed. "
                    "Please migrate them to data.route_pixel_budgets. "
                    f"Found: {legacy_pixel_budget_routes}"
                )


def load_config(path: str | Path) -> ExperimentRuntimeConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        yaml_payload = yaml.safe_load(handle) or {}
    _resolve_prompt_profile(yaml_payload, config_path)
    _raise_deprecated_config_keys(yaml_payload)
    _warn_unknown_config_keys(ExperimentRuntimeConfig, yaml_payload, path="")
    config = _from_dict(ExperimentRuntimeConfig, yaml_payload)
    _normalize_route_mappings(config)
    return apply_model_scale_tag(config)


def _normalize_route_mappings(config: ExperimentRuntimeConfig) -> None:
    if config.task.route is not None and str(config.task.route).strip():
        config.task.route = normalize_route_key(str(config.task.route))

    normalized_task_options: dict[str, dict[str, Any]] = {}
    for route_key, options in dict(config.task.route_options or {}).items():
        normalized_task_options[normalize_route_key(str(route_key))] = dict(options or {})
    config.task.route_options = normalized_task_options

    normalized_route_prompts: dict[str, dict[str, Any]] = {}
    for route_key, prompt_payload in dict(config.prompt.route_prompts or {}).items():
        normalized_route_prompts[normalize_route_key(str(route_key))] = dict(prompt_payload or {})
    config.prompt.route_prompts = normalized_route_prompts

    normalized_route_pixel_budgets: dict[str, dict[str, Any]] = {}
    for route_key, pixel_budget_payload in dict(config.data.route_pixel_budgets or {}).items():
        normalized_route_pixel_budgets[normalize_route_key(str(route_key))] = dict(pixel_budget_payload or {})
    config.data.route_pixel_budgets = normalized_route_pixel_budgets


def _resolve_prompt_profile(yaml_payload: dict[str, Any], config_path: Path) -> None:
    prompt_payload = yaml_payload.get("prompt")
    if not isinstance(prompt_payload, dict):
        return
    profile = prompt_payload.get("profile")
    if profile:
        profile_prompt_payload = _load_prompt_profile_payload(str(profile), config_path=config_path)
        merged_prompt_payload = dict(profile_prompt_payload)
        for key, value in prompt_payload.items():
            if key == "profile":
                continue
            merged_prompt_payload[key] = value
        merged_prompt_payload["profile"] = str(profile)
        _resolve_route_prompt_profiles(merged_prompt_payload, config_path=config_path)
        yaml_payload["prompt"] = merged_prompt_payload
        return
    _resolve_route_prompt_profiles(prompt_payload, config_path=config_path)
    yaml_payload["prompt"] = prompt_payload


def _resolve_route_prompt_profiles(prompt_payload: dict[str, Any], *, config_path: Path) -> None:
    route_prompts = prompt_payload.get("route_prompts")
    if route_prompts is None:
        return
    if not isinstance(route_prompts, dict):
        raise ValueError("prompt.route_prompts must be a mapping of route -> prompt payload.")

    allowed_keys = {
        "profile",
        "system_prompt",
        "system_prompt_template",
        "user_prompt",
        "user_prompt_template",
    }
    resolved_route_prompts: dict[str, dict[str, Any]] = {}
    for route_key, route_prompt_payload in route_prompts.items():
        if not isinstance(route_prompt_payload, dict):
            raise ValueError(
                "Each prompt.route_prompts item must be a mapping. "
                f"route={route_key!r}, got={type(route_prompt_payload).__name__}."
            )
        unknown_keys = sorted(set(route_prompt_payload.keys()) - allowed_keys)
        if unknown_keys:
            raise ValueError(
                f"Unsupported keys in prompt.route_prompts[{route_key!r}]: {unknown_keys}. "
                f"Supported keys: {sorted(allowed_keys)}"
            )

        profile = route_prompt_payload.get("profile")
        if profile:
            base_prompt_payload = _load_prompt_profile_payload(str(profile), config_path=config_path)
            merged_prompt_payload = dict(base_prompt_payload)
            for key, value in route_prompt_payload.items():
                if key == "profile":
                    continue
                merged_prompt_payload[key] = value
            merged_prompt_payload["profile"] = str(profile)
            resolved_route_prompts[normalize_route_key(str(route_key))] = merged_prompt_payload
        else:
            resolved_route_prompts[normalize_route_key(str(route_key))] = dict(route_prompt_payload)
    prompt_payload["route_prompts"] = resolved_route_prompts


def _load_prompt_profile_payload(profile: str, *, config_path: Path) -> dict[str, Any]:
    profile_path = _resolve_prompt_profile_path(profile, config_path=config_path)
    with profile_path.open("r", encoding="utf-8") as handle:
        profile_yaml = yaml.safe_load(handle) or {}
    prompt_payload = profile_yaml.get("prompt", profile_yaml)
    if not isinstance(prompt_payload, dict):
        raise ValueError(f"Prompt profile must contain a mapping under `prompt`: {profile_path}")
    allowed_keys = {"system_prompt", "system_prompt_template", "user_prompt", "user_prompt_template"}
    unknown_keys = sorted(set(prompt_payload.keys()) - allowed_keys)
    if unknown_keys:
        raise ValueError(
            f"Unsupported keys in prompt profile {profile_path}: {unknown_keys}. "
            f"Supported keys: {sorted(allowed_keys)}"
        )
    return dict(prompt_payload)


def load_prompt_profile_payload(profile: str, *, config_path: str | Path) -> dict[str, Any]:
    return _load_prompt_profile_payload(profile, config_path=Path(config_path))


def _resolve_prompt_profile_path(profile: str, *, config_path: Path) -> Path:
    raw_profile = Path(profile)
    candidate_paths: list[Path] = []
    if raw_profile.is_absolute():
        candidate_paths.append(raw_profile)
    else:
        candidate_paths.append(config_path.parent / raw_profile)
        candidate_paths.append(Path.cwd() / raw_profile)
        candidate_paths.append(Path.cwd() / "configs" / "prompts" / raw_profile)
        candidate_paths.append(Path.cwd() / "configs" / "prompts" / f"{profile}.yaml")
        candidate_paths.append(Path.cwd() / "configs" / "prompts" / f"{profile}.yml")
    for candidate in candidate_paths:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Prompt profile not found: {profile!r}. "
        f"Tried: {[str(candidate) for candidate in candidate_paths]}"
    )


def config_to_dict(config: ExperimentRuntimeConfig) -> dict[str, Any]:
    return asdict(config)


def apply_model_scale_tag(config: ExperimentRuntimeConfig) -> ExperimentRuntimeConfig:
    scale_tag = _extract_model_scale_tag(
        config.model.model_name_or_path,
        fallback=config.model.remote_model_name_or_path,
    )
    if scale_tag is None:
        return config

    if not _contains_standalone_tag(config.experiment.name, scale_tag):
        config.experiment.name = f"{config.experiment.name}-{scale_tag}"

    output_dir = Path(config.experiment.output_dir)
    if output_dir.name != scale_tag:
        config.experiment.output_dir = str(output_dir / scale_tag)

    if config.logging.run_name is not None and not _contains_standalone_tag(config.logging.run_name, scale_tag):
        config.logging.run_name = f"{config.logging.run_name}-{scale_tag}"

    return config


def apply_run_id(
    config: ExperimentRuntimeConfig,
    run_id: str,
    *,
    stage_name: str | None = None,
) -> ExperimentRuntimeConfig:
    normalized_run_id = _normalize_run_component(run_id, field_name="run_id")
    normalized_stage_name = None
    if stage_name is not None:
        normalized_stage_name = _normalize_run_component(stage_name, field_name="stage_name")

    base_experiment_name = config.experiment.name
    base_output_dir = Path(config.experiment.output_dir)
    base_run_name = config.logging.run_name or base_experiment_name

    if normalized_stage_name is None:
        suffix = normalized_run_id
        output_dir = base_output_dir / normalized_run_id
    else:
        suffix = f"{normalized_run_id}-{normalized_stage_name}"
        output_dir = base_output_dir / normalized_run_id / normalized_stage_name

    config.experiment.name = f"{base_experiment_name}-{suffix}"
    config.experiment.output_dir = str(output_dir)
    config.logging.run_name = f"{base_run_name}-{suffix}"
    return config


def _normalize_run_component(value: str, *, field_name: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    normalized = normalized.strip(".-")
    if not normalized:
        raise ValueError(f"{field_name} must contain at least one alphanumeric character.")
    return normalized


def _extract_model_scale_tag(primary: str | None, *, fallback: str | None = None) -> str | None:
    for candidate in (primary, fallback):
        if not candidate:
            continue
        basename = Path(str(candidate)).name.lower()
        match = re.search(r"([0-9]+(?:\.[0-9]+)?b)", basename)
        if match is not None:
            return match.group(1)
    return None


def _contains_standalone_tag(text: str, tag: str) -> bool:
    return re.search(rf"(^|[-_/]){re.escape(tag)}($|[-_/])", text) is not None
