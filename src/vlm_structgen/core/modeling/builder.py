from __future__ import annotations

import json
import importlib
import importlib.util
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from vlm_structgen.core.config import ExperimentRuntimeConfig

@dataclass
class BuildArtifacts:
    model: torch.nn.Module
    tokenizer: Any
    processor: Any
    trainable_summary: dict[str, int]


def _resolve_model_class():
    transformers_module = importlib.import_module("transformers")
    class_name = "Qwen3VLForConditionalGeneration"
    if not hasattr(transformers_module, class_name):
        raise ImportError(
            "Current transformers installation does not expose Qwen3VLForConditionalGeneration. "
            "Please install the Qwen3-VL compatible transformers version."
        )
    return getattr(transformers_module, class_name)


def _normalize_qwen3vl_text_config(model_config: Any) -> None:
    rope_defaults = {
        "rope_type": "default",
        "mrope_section": [24, 20, 20],
        "mrope_interleaved": True,
    }

    rope_scaling = getattr(model_config, "rope_scaling", None)
    if rope_scaling is None:
        model_config.rope_scaling = dict(rope_defaults)

    text_config = getattr(model_config, "text_config", None)
    if text_config is None:
        return
    rope_scaling = getattr(text_config, "rope_scaling", None)
    if rope_scaling is None:
        text_config.rope_scaling = dict(rope_defaults)


def _normalize_tokenizer_config_for_compatibility(source_dir: Path) -> Path:
    tokenizer_config_path = source_dir / "tokenizer_config.json"
    if not tokenizer_config_path.exists():
        return source_dir

    try:
        tokenizer_config = json.loads(tokenizer_config_path.read_text())
    except json.JSONDecodeError:
        return source_dir

    if not isinstance(tokenizer_config.get("extra_special_tokens"), list):
        return source_dir

    temp_dir = Path(tempfile.mkdtemp(prefix="arrow-vlm-tokenizer-compat-"))
    shutil.copytree(source_dir, temp_dir, dirs_exist_ok=True)
    tokenizer_config.pop("extra_special_tokens", None)
    tokenizer_config_path = temp_dir / "tokenizer_config.json"
    tokenizer_config_path.write_text(json.dumps(tokenizer_config, ensure_ascii=False, indent=2) + "\n")
    return temp_dir


def _build_model_from_config(model_class, model_config, **model_kwargs):
    from_config = getattr(model_class, "from_config", None)
    if callable(from_config):
        return from_config(model_config, **model_kwargs)

    private_from_config = getattr(model_class, "_from_config", None)
    if callable(private_from_config):
        return private_from_config(model_config, **model_kwargs)

    try:
        return model_class(model_config, **model_kwargs)
    except TypeError:
        # Some model constructors may not accept extra kwargs.
        return model_class(model_config)


def _freeze_all_parameters(model: torch.nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False


def _enable_all_parameters(model: torch.nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = True


def _set_requires_grad_by_name(
    model: torch.nn.Module,
    substrings: list[str],
    requires_grad: bool,
) -> None:
    for name, parameter in model.named_parameters():
        if any(substring in name for substring in substrings):
            parameter.requires_grad = requires_grad


def _trainable_summary(model: torch.nn.Module) -> dict[str, int]:
    trainable = 0
    total = 0
    for parameter in model.parameters():
        total += parameter.numel()
        if parameter.requires_grad:
            trainable += parameter.numel()
    return {"trainable_params": trainable, "total_params": total}


def _collect_lora_target_module_names(
    model: torch.nn.Module,
    *,
    include_name_substrings: list[str],
    exclude_name_substrings: list[str] | None,
    suffixes: list[str],
) -> list[str]:
    collected: list[str] = []
    exclude_name_substrings = exclude_name_substrings or []
    for name, module in model.named_modules():
        if not name:
            continue
        if not isinstance(module, torch.nn.Linear):
            continue
        lowered = name.lower()
        if include_name_substrings and not any(substring in lowered for substring in include_name_substrings):
            continue
        if exclude_name_substrings and any(substring in lowered for substring in exclude_name_substrings):
            continue
        if suffixes and not any(name.endswith(suffix) for suffix in suffixes):
            continue
        collected.append(name)
    return sorted(set(collected))


def _resolve_model_source(config: ExperimentRuntimeConfig) -> str:
    local_path = Path(config.model.model_name_or_path)
    if local_path.exists():
        return str(local_path)
    return config.model.remote_model_name_or_path


def _is_local_model_source(model_source: str) -> bool:
    return Path(model_source).exists()


def _resolve_attn_implementation(config: ExperimentRuntimeConfig) -> str | None:
    requested = config.model.attn_implementation
    if not requested:
        return None
    if requested != "flash_attention_2":
        return requested
    if not torch.cuda.is_available():
        print("flash_attention_2 requested but CUDA is unavailable; falling back to sdpa.")
        return "sdpa"
    if importlib.util.find_spec("flash_attn") is None:
        print("flash_attention_2 requested but flash_attn is not installed; falling back to sdpa.")
        return "sdpa"
    return requested


def _sanitize_generation_config(model: torch.nn.Module, config: ExperimentRuntimeConfig) -> None:
    generation_config = getattr(model, "generation_config", None)
    if generation_config is None:
        return
    generation_config.do_sample = config.eval.do_sample
    generation_config.num_beams = config.eval.num_beams
    generation_config.use_cache = config.eval.use_cache
    if config.eval.do_sample:
        generation_config.temperature = config.eval.temperature
        generation_config.top_p = config.eval.top_p
        generation_config.top_k = config.eval.top_k
    else:
        # Greedy / beam search does not consume sampling-only knobs. Clearing them
        # avoids repeated transformers warnings like temperature/top_p/top_k ignored.
        generation_config.temperature = None
        generation_config.top_p = None
        generation_config.top_k = None


def _maybe_enable_gradient_checkpointing(model: torch.nn.Module, config: ExperimentRuntimeConfig) -> None:
    if not config.train.gradient_checkpointing:
        return

    enable_input_require_grads = getattr(model, "enable_input_require_grads", None)
    if callable(enable_input_require_grads):
        enable_input_require_grads()

    gradient_checkpointing_enable = getattr(model, "gradient_checkpointing_enable", None)
    if not callable(gradient_checkpointing_enable):
        raise ValueError(
            "gradient_checkpointing was requested, but the current model does not expose "
            "`gradient_checkpointing_enable()`."
        )

    try:
        gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    except TypeError:
        gradient_checkpointing_enable()
    print("[builder] gradient checkpointing enabled.", flush=True)


def build_model_tokenizer_processor(
    config: ExperimentRuntimeConfig,
) -> BuildArtifacts:
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoProcessor, AutoTokenizer

    if config.finetune.mode not in {"lora", "full"}:
        raise ValueError(
            f"Unsupported finetune.mode={config.finetune.mode!r}. Expected 'lora' or 'full'."
        )
    model_class = _resolve_model_class()
    model_source = _resolve_model_source(config)
    local_files_only = _is_local_model_source(model_source)
    dtype = torch.bfloat16 if config.train.bf16 and torch.cuda.is_available() else None
    model_kwargs = {
        "trust_remote_code": config.model.trust_remote_code,
        "local_files_only": local_files_only,
    }
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    attn_implementation = _resolve_attn_implementation(config)
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    print(f"[builder] loading model from: {model_source}", flush=True)
    model = model_class.from_pretrained(model_source, **model_kwargs)
    print("[builder] loading processor...", flush=True)
    processor = AutoProcessor.from_pretrained(
        model_source,
        trust_remote_code=config.model.trust_remote_code,
        local_files_only=local_files_only,
    )
    print("[builder] resolving tokenizer...", flush=True)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            model_source,
            trust_remote_code=config.model.trust_remote_code,
            local_files_only=local_files_only,
        )
        processor.tokenizer = tokenizer

    model = _finalize_model_for_runtime(model, config)
    summary = _trainable_summary(model)
    return BuildArtifacts(
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        trainable_summary=summary,
    )


def build_model_tokenizer_processor_from_checkpoint(
    config: ExperimentRuntimeConfig,
    *,
    checkpoint_dir: str | Path,
) -> BuildArtifacts:
    from transformers import AutoConfig, AutoProcessor, AutoTokenizer

    checkpoint_dir = Path(checkpoint_dir)

    flat_model_config = checkpoint_dir / "config.json"
    legacy_model_config = checkpoint_dir / "model" / "config.json"
    model_config_path = flat_model_config if flat_model_config.exists() else legacy_model_config
    if not model_config_path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint model config. Tried: {flat_model_config} and {legacy_model_config}"
        )

    has_flat_tokenizer = (checkpoint_dir / "tokenizer_config.json").exists() or (checkpoint_dir / "tokenizer.json").exists()
    has_flat_processor = (checkpoint_dir / "preprocessor_config.json").exists() or (checkpoint_dir / "processor_config.json").exists()
    tokenizer_source = checkpoint_dir if has_flat_tokenizer else (checkpoint_dir / "tokenizer")
    processor_source = checkpoint_dir if has_flat_processor else (checkpoint_dir / "processor")
    tokenizer_source = _normalize_tokenizer_config_for_compatibility(Path(tokenizer_source))
    processor_source = _normalize_tokenizer_config_for_compatibility(Path(processor_source))

    model_class = _resolve_model_class()
    attn_implementation = _resolve_attn_implementation(config)
    model_config = AutoConfig.from_pretrained(model_config_path, trust_remote_code=config.model.trust_remote_code)
    _normalize_qwen3vl_text_config(model_config)
    model_kwargs = {}
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation
    print(f"[builder] constructing model from checkpoint config: {model_config_path}", flush=True)
    model = _build_model_from_config(model_class, model_config, **model_kwargs)

    model_source = _resolve_model_source(config)
    local_files_only = _is_local_model_source(model_source)

    if processor_source.exists():
        print(f"[builder] loading processor from checkpoint: {processor_source}", flush=True)
        processor = AutoProcessor.from_pretrained(
            processor_source,
            trust_remote_code=config.model.trust_remote_code,
            local_files_only=True,
        )
    else:
        print("[builder] checkpoint has no processor dir; falling back to model source.", flush=True)
        processor = AutoProcessor.from_pretrained(
            model_source,
            trust_remote_code=config.model.trust_remote_code,
            local_files_only=local_files_only,
        )

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        if tokenizer_source.exists():
            print(f"[builder] loading tokenizer from checkpoint: {tokenizer_source}", flush=True)
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_source,
                trust_remote_code=config.model.trust_remote_code,
                local_files_only=True,
            )
        else:
            print("[builder] checkpoint has no tokenizer dir; falling back to model source.", flush=True)
            tokenizer = AutoTokenizer.from_pretrained(
                model_source,
                trust_remote_code=config.model.trust_remote_code,
                local_files_only=local_files_only,
            )
        processor.tokenizer = tokenizer

    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(model, "generation_config") and tokenizer.pad_token_id is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    model = _finalize_model_for_runtime(model, config)
    summary = _trainable_summary(model)
    return BuildArtifacts(
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        trainable_summary=summary,
    )


def _finalize_model_for_runtime(model: torch.nn.Module, config: ExperimentRuntimeConfig) -> torch.nn.Module:
    # tokenizer / processor side-effects are managed before calling this helper.
    # This helper configures model trainable/frozen state and generation behavior.
    _sanitize_generation_config(model, config)

    if config.finetune.mode == "lora":
        _freeze_all_parameters(model)
        if not config.lora.enabled:
            raise ValueError("finetune.mode='lora' requires lora.enabled=true.")
        target_modules = list(config.lora.lang_target_modules)
        if not config.model.freeze_vision_tower:
            vis_target_modules = _collect_lora_target_module_names(
                model,
                include_name_substrings=config.model.vision_name_substrings,
                exclude_name_substrings=config.model.projector_name_substrings,
                suffixes=config.lora.vis_target_modules,
            )
            if not vis_target_modules:
                raise ValueError(
                    "freeze_vision_tower=false was requested in LoRA mode, but no visual target modules were found. "
                    "Check model.vision_name_substrings and lora.vis_target_modules."
                )
            target_modules.extend(vis_target_modules)
            print(
                f"[builder] enabling LoRA on {len(vis_target_modules)} visual modules.",
                flush=True,
            )
        if config.model.train_projector:
            proj_target_modules = _collect_lora_target_module_names(
                model,
                include_name_substrings=config.model.projector_name_substrings,
                exclude_name_substrings=None,
                suffixes=config.lora.proj_target_modules,
            )
            if not proj_target_modules:
                raise ValueError(
                    "train_projector=true was requested in LoRA mode, but no projector target modules were found. "
                    "Check model.projector_name_substrings and lora.proj_target_modules."
                )
            target_modules.extend(proj_target_modules)
            print(
                f"[builder] enabling LoRA on {len(proj_target_modules)} projector modules.",
                flush=True,
            )
        lora_config = LoraConfig(
            r=config.lora.r,
            lora_alpha=config.lora.alpha,
            lora_dropout=config.lora.dropout,
            bias=config.lora.bias,
            target_modules=sorted(set(target_modules)),
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
    else:
        _enable_all_parameters(model)

    if config.model.freeze_vision_tower:
        _set_requires_grad_by_name(model, config.model.vision_name_substrings, False)

    # embedding层 和 LM Head 需要训练，否则不会有效果
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    if input_embeddings is not None:
        for parameter in input_embeddings.parameters():
            parameter.requires_grad = True
    if output_embeddings is not None:
        for parameter in output_embeddings.parameters():
            parameter.requires_grad = True

    _maybe_enable_gradient_checkpointing(model, config)
    return model
