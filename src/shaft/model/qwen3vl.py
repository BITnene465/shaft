from __future__ import annotations

import importlib.util
import warnings

import torch
from transformers import AutoModelForImageTextToText, AutoModelForVision2Seq, AutoProcessor, AutoTokenizer

from shaft.config import RuntimeConfig

from .finetune import apply_finetune_strategy, make_bnb_4bit_config
from .policies import build_peft_policy, build_processor_policy
from .registry import default_model_groups, register_model
from .types import (
    ModelArtifacts,
    ModelCapabilities,
    ModelLoader,
    ModelMeta,
    ShaftModelAdapter,
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


def _resolve_attn_implementation(attn_implementation: str | None) -> str | None:
    normalized = str(attn_implementation).strip() if attn_implementation is not None else ""
    if not normalized:
        return None
    if normalized != "flash_attention_2":
        return normalized
    if importlib.util.find_spec("flash_attn") is not None:
        return normalized
    warnings.warn(
        "Requested attn_implementation='flash_attention_2' but flash-attn is not installed. "
        "Falling back to the Transformers default attention implementation.",
        stacklevel=2,
    )
    return None


QWEN3VL_META = ModelMeta(
    model_type="qwen3vl",
    family="qwen",
    default_template="qwen3vl",
    model_groups=default_model_groups("qwen3-vl-4b-instruct", "qwen3-vl", template="qwen3vl"),
    capabilities=ModelCapabilities(supports_pixel_budget=True, is_multimodal=True),
    processor_policy=build_processor_policy("pixel_budget"),
    peft_policy=build_peft_policy("all_linear"),
)


@register_model(QWEN3VL_META)
class Qwen3VLLoader(ModelLoader):
    def build(
        self,
        config: RuntimeConfig,
        *,
        model_meta: ModelMeta,
        model_adapter: ShaftModelAdapter,
    ) -> ModelArtifacts:
        model_name = config.model.model_name_or_path
        resolved_dtype = _resolve_dtype(config.model.torch_dtype)
        finetune = config.model.finetune
        finetune.target_modules = model_adapter.resolve_target_modules(finetune.target_modules)
        common_kwargs = {
            "trust_remote_code": bool(config.model.trust_remote_code),
            "dtype": resolved_dtype,
        }
        attn_implementation = _resolve_attn_implementation(config.model.attn_implementation)
        if attn_implementation:
            common_kwargs["attn_implementation"] = attn_implementation
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
        model_info = model_adapter.build_model_info(
            torch_dtype=resolved_dtype,
            max_model_len=getattr(getattr(model, "config", None), "max_position_embeddings", None),
        )
        template = model_adapter.build_template()
        return ModelArtifacts(
            model=model,
            tokenizer=tokenizer,
            processor=processor,
            model_meta=model_meta,
            model_adapter=model_adapter,
            model_info=model_info,
            template=template,
        )
