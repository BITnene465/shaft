from __future__ import annotations

import importlib.util
import warnings

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

try:
    from transformers import AutoModelForVision2Seq
except ImportError:  # Transformers 5.x removed this deprecated alias.
    AutoModelForVision2Seq = None  # type: ignore[assignment]

from shaft.config import RuntimeConfig, resolve_effective_gradient_checkpointing

from .descriptor import ResolvedModelDescriptor
from .finetune import apply_resolved_finetune_plan, make_bnb_4bit_config
from .finetune_plan import build_resolved_finetune_plan
from .objective import QwenVLMoeTrainingObjectivePolicy
from .qwen_inference import QwenVLInferencePolicy
from .policies import build_peft_policy, build_processor_policy
from .registry import default_model_groups, register_model
from .sequence import Qwen3VLMoeSequenceExecutionPolicy, Qwen3VLSequenceExecutionPolicy
from .sharding import ModelShardingPolicy
from .types import (
    ModelArtifacts,
    ModelCapabilities,
    ModelLoader,
    ModelMeta,
    ModelModuleGroups,
    ShaftModelAdapter,
    ShaftSequenceExecutionContract,
)


def _resolve_dtype(dtype_name: str) -> torch.dtype | str:
    normalized = str(dtype_name).strip().lower()
    if normalized == "auto":
        return "auto"
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(
        f"Unsupported torch dtype {dtype_name!r}; expected auto, bf16/bfloat16, "
        "fp16/float16/half, or fp32/float32."
    )


def _resolve_attn_implementation(
    attn_implementation: str | None,
    *,
    required: bool = False,
) -> str | None:
    normalized = str(attn_implementation).strip() if attn_implementation is not None else ""
    if not normalized:
        return None
    if normalized != "flash_attention_2":
        return normalized
    if importlib.util.find_spec("flash_attn") is not None:
        return normalized
    if required:
        raise ImportError(
            "Qwen3VL varlen requested FlashAttention 2, but flash-attn is not "
            "installed; refusing a silent padded-attention fallback."
        )
    warnings.warn(
        "Requested attn_implementation='flash_attention_2' but flash-attn is not installed. "
        "Falling back to the Transformers default attention implementation.",
        stacklevel=2,
    )
    return None


def _is_qwen3vl_dense_descriptor(descriptor: ResolvedModelDescriptor) -> bool:
    architectures = tuple(value.lower() for value in descriptor.architectures)
    return descriptor.hf_model_type == "qwen3_vl" and not any(
        "moe" in value for value in architectures
    )


def _is_qwen3vl_moe_descriptor(descriptor: ResolvedModelDescriptor) -> bool:
    architectures = tuple(value.lower() for value in descriptor.architectures)
    return descriptor.hf_model_type == "qwen3_vl_moe" and (
        not architectures or any("moe" in value for value in architectures)
    )


_QWEN3VL_DENSE_MODULE_GROUPS = ModelModuleGroups(
    language_model=("model.language_model",),
    vision_tower=("model.visual",),
    aligner=("model.visual.merger", "model.visual.deepstack_merger_list"),
    generator=("lm_head",),
)


_QWEN3VL_MOE_MODULE_GROUPS = ModelModuleGroups(
    language_model=("model.language_model",),
    vision_tower=("model.visual",),
    aligner=("model.visual.merger", "model.visual.deepstack_merger_list"),
    generator=("lm_head",),
)


QWEN3VL_META = ModelMeta(
    model_type="qwen3vl",
    family="qwen",
    default_template="qwen3vl",
    hf_model_types=("qwen3_vl", "qwen3_vl_moe"),
    model_groups=(
        *default_model_groups(
            "qwen3-vl-2b-instruct",
            "qwen3-vl-4b-instruct",
            "qwen3-vl",
            name="dense",
            hf_model_types=("qwen3_vl",),
            descriptor_matcher=_is_qwen3vl_dense_descriptor,
            template="qwen3vl",
        ),
        *default_model_groups(
            "qwen3-vl-30b-a3b-instruct",
            "qwen3-vl-30b-a3b",
            name="moe",
            hf_model_types=("qwen3_vl_moe",),
            descriptor_matcher=_is_qwen3vl_moe_descriptor,
            template="qwen3vl",
            module_groups=_QWEN3VL_MOE_MODULE_GROUPS,
            sequence_execution_policy=Qwen3VLMoeSequenceExecutionPolicy(),
            training_objective_policy=QwenVLMoeTrainingObjectivePolicy(),
            peft_policy=build_peft_policy("qwen_vl_moe"),
            default_experts_implementation="grouped_mm",
            sharding_policy=ModelShardingPolicy(
                fsdp_transformer_layer_cls_to_wrap=(
                    "Qwen3VLMoeTextDecoderLayer",
                    "Qwen3VLMoeVisionBlock",
                ),
                supports_fsdp_activation_checkpointing=False,
                supports_deepspeed_zero3=False,
            ),
            requires=("module:transformers.models.qwen3_vl_moe",),
        ),
    ),
    capabilities=ModelCapabilities(is_multimodal=True),
    module_groups=_QWEN3VL_DENSE_MODULE_GROUPS,
    processor_policy=build_processor_policy("qwen_vl"),
    inference_policy=QwenVLInferencePolicy(),
    sequence_execution_policy=Qwen3VLSequenceExecutionPolicy(),
    peft_policy=build_peft_policy("all_linear"),
    sharding_policy=ModelShardingPolicy(
        fsdp_transformer_layer_cls_to_wrap=("Qwen3VLTextDecoderLayer", "Qwen3VLVisionBlock"),
    ),
)


@register_model(QWEN3VL_META)
class Qwen3VLLoader(ModelLoader):
    def build(
        self,
        config: RuntimeConfig,
        *,
        model_meta: ModelMeta,
        model_adapter: ShaftModelAdapter,
        sequence_execution_contract: ShaftSequenceExecutionContract | None = None,
    ) -> ModelArtifacts:
        model_name = config.model.model_name_or_path
        resolved_dtype = _resolve_dtype(config.model.torch_dtype)
        finetune = config.model.finetune
        common_kwargs = {
            "trust_remote_code": bool(config.model.trust_remote_code),
            "dtype": resolved_dtype,
            "revision": config.model.revision,
            "cache_dir": config.model.cache_dir,
            "local_files_only": bool(config.model.local_files_only),
        }
        if config.model.device_map not in (None, ""):
            common_kwargs["device_map"] = config.model.device_map
        execution_contract = sequence_execution_contract
        if execution_contract is None:
            execution_contract = model_adapter.build_sequence_execution_contract(
                layout=config.data.batching.layout,
                device_type="cpu" if bool(config.train.use_cpu) else "cuda",
                attention_implementation=config.model.attn_implementation,
                torch_dtype=config.model.torch_dtype,
                distributed_strategy=config.train.distributed.strategy,
            )
        attn_implementation = _resolve_attn_implementation(
            config.model.attn_implementation,
            required=execution_contract.layout == "varlen",
        )
        if attn_implementation:
            common_kwargs["attn_implementation"] = attn_implementation
        experts_implementation = model_adapter.resolve_experts_implementation(
            config.model.experts_implementation
        )
        if experts_implementation is not None:
            common_kwargs["experts_implementation"] = experts_implementation
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
        model_classes = [AutoModelForImageTextToText]
        if AutoModelForVision2Seq is not None:
            model_classes.append(AutoModelForVision2Seq)
        for cls in model_classes:
            try:
                model = cls.from_pretrained(model_name, **common_kwargs)
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
        if model is None:
            assert last_err is not None
            raise RuntimeError(
                f"Failed to load {model_meta.model_type} model from {model_name!r}. "
                "Please verify model path and transformers version."
            ) from last_err
        if experts_implementation is not None:
            resolved_experts = getattr(model.config, "_experts_implementation", None)
            resolved_text_experts = getattr(
                getattr(model.config, "text_config", None),
                "_experts_implementation",
                None,
            )
            if resolved_experts != experts_implementation or resolved_text_experts != experts_implementation:
                raise RuntimeError(
                    "The loaded MoE model did not retain the requested experts implementation: "
                    f"requested={experts_implementation!r}, root={resolved_experts!r}, "
                    f"text={resolved_text_experts!r}."
                )

        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=config.model.trust_remote_code,
            fix_mistral_regex=False,
            revision=config.model.revision,
            cache_dir=config.model.cache_dir,
            local_files_only=bool(config.model.local_files_only),
        )
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=config.model.trust_remote_code,
                fix_mistral_regex=False,
                revision=config.model.revision,
                cache_dir=config.model.cache_dir,
                local_files_only=bool(config.model.local_files_only),
            )
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        finetune_plan = build_resolved_finetune_plan(model, finetune, model_adapter=model_adapter)
        model = apply_resolved_finetune_plan(
            model,
            finetune_plan,
            finetune=finetune,
            gradient_checkpointing=resolve_effective_gradient_checkpointing(config),
        )
        setattr(model, "_shaft_finetune_plan", finetune_plan)
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
            finetune_plan=finetune_plan,
        )
