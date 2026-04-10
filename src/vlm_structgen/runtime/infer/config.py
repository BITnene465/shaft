from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from vlm_structgen.core.config import ExperimentRuntimeConfig, _from_dict, load_prompt_profile_payload
from vlm_structgen.core.routing import normalize_route_key
from vlm_structgen.core.utils.checkpoint import load_checkpoint_meta


@dataclass
class InferModelConfig:
    min_pixels: int | None = None
    max_pixels: int | None = None


@dataclass
class InferPromptConfig:
    profile: str | None = None
    system_prompt: str | None = None
    system_prompt_template: str | None = None
    user_prompt: str | None = None
    user_prompt_template: str | None = None


@dataclass
class InferTaskConfig:
    route: str | None = None
    route_options: dict[str, Any] = field(default_factory=dict)


@dataclass
class InferEvalConfig:
    max_new_tokens: int | None = None
    num_beams: int | None = None
    do_sample: bool | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    use_cache: bool | None = None


@dataclass
class InferAppConfig:
    host: str = "127.0.0.1"
    port: int = 7860
    share: bool = False


@dataclass
class OneStageInferenceConfig:
    model: InferModelConfig = field(default_factory=InferModelConfig)
    task: InferTaskConfig = field(default_factory=InferTaskConfig)
    prompt: InferPromptConfig = field(default_factory=InferPromptConfig)
    eval: InferEvalConfig = field(default_factory=InferEvalConfig)
    batch_size: int = 1
    app: InferAppConfig = field(default_factory=InferAppConfig)
    output_dir: str | None = None


@dataclass
class InferenceSettings:
    runtime: ExperimentRuntimeConfig
    lora_adapter_path: str
    device: str | None = None
    batch_size: int = 1
    output_dir: str | None = None
    app: InferAppConfig = field(default_factory=InferAppConfig)


def _find_dotenv_path(explicit_env_file: str | Path | None = None) -> Path | None:
    if explicit_env_file is not None:
        candidate = Path(explicit_env_file).expanduser().resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Env file not found: {candidate}")
        return candidate

    cwd = Path.cwd().resolve()
    for candidate_dir in [cwd, *cwd.parents]:
        dotenv_path = candidate_dir / ".env"
        if dotenv_path.exists():
            return dotenv_path
    return None


def _load_yaml_payload(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_one_stage_inference_config(path: str | Path | None) -> OneStageInferenceConfig:
    payload = _load_yaml_payload(path)
    if path is not None:
        resolve_prompt_profile_for_mapping(payload, config_path=Path(path))
    config = _from_dict(OneStageInferenceConfig, payload)
    if config.task.route is not None and str(config.task.route).strip():
        config.task.route = normalize_route_key(config.task.route)
    return config

def resolve_prompt_profile_for_mapping(mapping: dict[str, Any] | None, *, config_path: Path) -> None:
    if not isinstance(mapping, dict):
        return
    prompt_payload = mapping.get("prompt")
    if not isinstance(prompt_payload, dict):
        return
    profile = prompt_payload.get("profile")
    if not profile:
        return
    resolved_prompt_payload = load_prompt_profile_payload(str(profile), config_path=config_path)
    merged_prompt_payload = dict(resolved_prompt_payload)
    for key, value in prompt_payload.items():
        if key == "profile":
            continue
        merged_prompt_payload[key] = value
    merged_prompt_payload["profile"] = str(profile)
    mapping["prompt"] = merged_prompt_payload


def _extract_runtime_payload_from_checkpoint_meta(checkpoint_path: str | Path) -> dict[str, Any]:
    meta = load_checkpoint_meta(checkpoint_path)
    checkpoint_config = meta.get("config", {})
    runtime_payload: dict[str, Any] = {}
    for section_name in ("model", "tokenizer", "task", "prompt", "finetune", "lora", "eval", "train"):
        section_value = checkpoint_config.get(section_name)
        if isinstance(section_value, dict):
            runtime_payload[section_name] = dict(section_value)
    return runtime_payload


def _build_runtime_from_checkpoint(checkpoint_path: str | Path) -> ExperimentRuntimeConfig:
    runtime = _from_dict(ExperimentRuntimeConfig, _extract_runtime_payload_from_checkpoint_meta(checkpoint_path))
    runtime.train.gradient_checkpointing = False
    return runtime


def _apply_model_overrides(runtime: ExperimentRuntimeConfig, model_cfg: InferModelConfig) -> None:
    if model_cfg.min_pixels is not None:
        runtime.model.min_pixels = model_cfg.min_pixels
    if model_cfg.max_pixels is not None:
        runtime.model.max_pixels = model_cfg.max_pixels


def _apply_prompt_overrides(runtime: ExperimentRuntimeConfig, prompt_cfg: InferPromptConfig) -> None:
    if prompt_cfg.profile is not None:
        runtime.prompt.profile = prompt_cfg.profile
    if prompt_cfg.system_prompt is not None:
        runtime.prompt.system_prompt = prompt_cfg.system_prompt
    if prompt_cfg.system_prompt_template is not None:
        runtime.prompt.system_prompt_template = prompt_cfg.system_prompt_template
    if prompt_cfg.user_prompt is not None:
        runtime.prompt.user_prompt = prompt_cfg.user_prompt
    if prompt_cfg.user_prompt_template is not None:
        runtime.prompt.user_prompt_template = prompt_cfg.user_prompt_template


def _apply_eval_overrides(runtime: ExperimentRuntimeConfig, eval_cfg: InferEvalConfig) -> None:
    if eval_cfg.max_new_tokens is not None:
        runtime.eval.max_new_tokens = eval_cfg.max_new_tokens
    if eval_cfg.num_beams is not None:
        runtime.eval.num_beams = eval_cfg.num_beams
    if eval_cfg.do_sample is not None:
        runtime.eval.do_sample = eval_cfg.do_sample
    if eval_cfg.temperature is not None:
        runtime.eval.temperature = eval_cfg.temperature
    if eval_cfg.top_p is not None:
        runtime.eval.top_p = eval_cfg.top_p
    if eval_cfg.top_k is not None:
        runtime.eval.top_k = eval_cfg.top_k
    if eval_cfg.use_cache is not None:
        runtime.eval.use_cache = eval_cfg.use_cache


def _apply_task_overrides(runtime: ExperimentRuntimeConfig, task_cfg: InferTaskConfig) -> None:
    if task_cfg.route is not None:
        runtime.task.route = normalize_route_key(task_cfg.route)
    if task_cfg.route_options:
        route_key = runtime.task.route
        if not route_key:
            known_routes = sorted(runtime.task.route_options.keys())
            if len(known_routes) == 1:
                route_key = known_routes[0]
                runtime.task.route = route_key
            else:
                raise ValueError(
                    "Infer task route is required when overriding task.route_options. "
                    f"Set task.route explicitly. Known routes from checkpoint: {known_routes}"
                )
        merged_route_options = dict(runtime.task.route_options.get(route_key, {}))
        merged_route_options.update(dict(task_cfg.route_options))
        runtime.task.route_options[route_key] = merged_route_options


def build_runtime_from_one_stage_infer_config(
    checkpoint_path: str | Path,
    infer_config: OneStageInferenceConfig,
) -> ExperimentRuntimeConfig:
    runtime = _build_runtime_from_checkpoint(checkpoint_path)
    _apply_model_overrides(runtime, infer_config.model)
    _apply_task_overrides(runtime, infer_config.task)
    _apply_prompt_overrides(runtime, infer_config.prompt)
    _apply_eval_overrides(runtime, infer_config.eval)
    return runtime


def load_inference_settings(
    *,
    lora_adapter_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    config_path: str | Path | None = None,
    infer_config: OneStageInferenceConfig | None = None,
    env_file: str | Path | None = None,
) -> InferenceSettings:
    dotenv_path = _find_dotenv_path(env_file)
    if dotenv_path is not None:
        load_dotenv(dotenv_path=dotenv_path, override=False)

    effective_infer_config = infer_config or load_one_stage_inference_config(config_path)
    resolved_lora_adapter_path = lora_adapter_path or checkpoint_path
    if resolved_lora_adapter_path:
        runtime = build_runtime_from_one_stage_infer_config(resolved_lora_adapter_path, effective_infer_config)
    else:
        runtime = ExperimentRuntimeConfig()
        _apply_model_overrides(runtime, effective_infer_config.model)
        _apply_task_overrides(runtime, effective_infer_config.task)
        _apply_prompt_overrides(runtime, effective_infer_config.prompt)
        _apply_eval_overrides(runtime, effective_infer_config.eval)
    output_dir = getattr(effective_infer_config, "output_dir", None)
    app = getattr(effective_infer_config, "app", InferAppConfig())
    return InferenceSettings(
        runtime=runtime,
        lora_adapter_path=str(resolved_lora_adapter_path) if resolved_lora_adapter_path else "",
        device=None,
        batch_size=max(int(getattr(effective_infer_config, "batch_size", 1)), 1),
        output_dir=output_dir,
        app=app,
    )
