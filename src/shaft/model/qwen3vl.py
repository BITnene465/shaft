from __future__ import annotations

import importlib.util

import torch
from transformers import AutoModelForImageTextToText, AutoModelForVision2Seq, AutoProcessor, AutoTokenizer

from shaft.config import RuntimeConfig
from shaft.template import build_template_from_meta, resolve_template_meta

from .finetune import apply_finetune_strategy, make_bnb_4bit_config
from .registry import default_model_groups, register_model
from .types import (
    DefaultPeftPolicy,
    ModelArtifacts,
    ModelCapabilities,
    ModelInfo,
    ModelLoader,
    ModelMeta,
    ProcessorPolicy,
)


def _resolve_dtype(dtype_name: str) -> torch.dtype | str:
    normalized = str(dtype_name).strip().lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    return "auto"


QWEN3VL_META = ModelMeta(
    model_type="qwen3vl",
    family="qwen",
    default_template="qwen3vl",
    model_groups=default_model_groups("qwen3-vl-4b-instruct", "qwen3-vl", template="qwen3vl"),
    capabilities=ModelCapabilities(supports_pixel_budget=True, is_multimodal=True),
    processor_policy=ProcessorPolicy(supports_pixel_budget=True),
    peft_policy=DefaultPeftPolicy(target_modules=["all-linear"]),
)


@register_model(QWEN3VL_META)
class Qwen3VLLoader(ModelLoader):
    def build(self, config: RuntimeConfig, *, model_meta: ModelMeta) -> ModelArtifacts:
        model_name = config.model.model_name_or_path
        resolved_dtype = _resolve_dtype(config.model.torch_dtype)
        finetune = config.model.finetune
        finetune.target_modules = model_meta.resolve_target_modules(finetune.target_modules)
        common_kwargs = {
            "trust_remote_code": bool(config.model.trust_remote_code),
            "dtype": resolved_dtype,
        }
        if config.model.attn_implementation:
            common_kwargs["attn_implementation"] = config.model.attn_implementation
        if finetune.mode == "qlora" and bool(finetune.qlora_load_in_4bit):
            if importlib.util.find_spec("bitsandbytes") is None:
                raise ImportError(
                    "QLoRA with 4bit loading requires bitsandbytes. Install with `uv pip install -e \".[gpu]\"`."
                )
            common_kwargs["quantization_config"] = make_bnb_4bit_config(
                finetune,
                dtype=_resolve_dtype(finetune.qlora_compute_dtype),
            )

        last_err: Exception | None = None
        model = None
        for cls in (AutoModelForImageTextToText, AutoModelForVision2Seq):
            try:
                model = cls.from_pretrained(model_name, **common_kwargs)
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
        if model is None:
            assert last_err is not None
            raise RuntimeError(
                f"Failed to load qwen3vl model from {model_name!r}. "
                "Please verify model path and transformers version."
            ) from last_err

        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=config.model.trust_remote_code)
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=config.model.trust_remote_code)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        model = apply_finetune_strategy(model, finetune)
        model_info = ModelInfo(
            model_type=model_meta.model_type,
            model_dir=str(model_name),
            torch_dtype=resolved_dtype,
            max_model_len=getattr(getattr(model, "config", None), "max_position_embeddings", None),
            is_multimodal=model_meta.capabilities.is_multimodal,
            family=model_meta.family,
        )
        template_meta = resolve_template_meta(
            template_type=config.model.template,
            model_meta=model_meta,
            model_info=model_info,
        )
        template = build_template_from_meta(template_meta)
        return ModelArtifacts(
            model=model,
            tokenizer=tokenizer,
            processor=processor,
            model_meta=model_meta,
            model_info=model_info,
            template=template,
        )
