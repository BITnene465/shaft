from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .qwen_inference import QwenVLInferencePolicy
from .policies import QwenVLMoePeftPolicy, build_processor_policy
from .qwen3vl import Qwen3VLLoader
from .objective import QwenVLMoeTrainingObjectivePolicy
from .registry import default_model_groups, register_model
from .sequence import Qwen35VLSequenceExecutionPolicy
from .sharding import ModelShardingPolicy
from .types import DefaultPeftPolicy, ModelCapabilities, ModelMeta, ModelModuleGroups
from .descriptor import ResolvedModelDescriptor


def _is_qwen35_dense_descriptor(descriptor: ResolvedModelDescriptor) -> bool:
    architectures = tuple(value.lower() for value in descriptor.architectures)
    return descriptor.hf_model_type == "qwen3_5" and not any(
        "moe" in value for value in architectures
    )


def _is_qwen35_moe_descriptor(descriptor: ResolvedModelDescriptor) -> bool:
    architectures = tuple(value.lower() for value in descriptor.architectures)
    return descriptor.hf_model_type == "qwen3_5_moe" and (
        not architectures or any("moe" in value for value in architectures)
    )


@dataclass(frozen=True)
class Qwen35VLPeftPolicy(DefaultPeftPolicy):
    def validate_training_finetune_config(
        self,
        finetune: Any,
        *,
        model_descriptor: ResolvedModelDescriptor | None = None,
        model_name_or_path: str | None = None,
    ) -> None:
        super().validate_training_finetune_config(
            finetune,
            model_descriptor=model_descriptor,
            model_name_or_path=model_name_or_path,
        )
        quantization_config = (
            None
            if model_descriptor is None
            else model_descriptor.config_value("quantization_config")
        )
        normalized_name = str(model_name_or_path or "").strip().lower()
        if quantization_config or normalized_name.rstrip("/").endswith("-fp8"):
            raise ValueError(
                "Pre-quantized Qwen3.5/3.6 artifacts are inference-only in "
                "Shaft; use an unquantized base checkpoint for training."
            )


@dataclass(frozen=True)
class Qwen35MoePeftPolicy(QwenVLMoePeftPolicy, Qwen35VLPeftPolicy):
    pass


_QWEN35_DENSE_PEFT_POLICY = Qwen35VLPeftPolicy(target_modules=["all-linear"])


_QWEN35_MOE_PEFT_POLICY = Qwen35MoePeftPolicy(
    target_modules=["all-linear"],
    target_parameters=[
        "mlp.experts.gate_up_proj",
        "mlp.experts.down_proj",
        "mlp.gate.weight",
    ],
)


_QWEN35VL_COMMON = dict(
    family="qwen",
    default_template="qwen35vl",
    hf_model_types=("qwen3_5", "qwen3_5_moe"),
    capabilities=ModelCapabilities(is_multimodal=True),
    module_groups=ModelModuleGroups(
        language_model=("model.language_model",),
        vision_tower=("model.visual",),
        aligner=("model.visual.merger", "model.visual.deepstack_merger_list"),
        generator=("lm_head",),
    ),
    processor_policy=build_processor_policy("qwen_vl"),
    inference_policy=QwenVLInferencePolicy(supports_thinking_templates=True),
    sequence_execution_policy=Qwen35VLSequenceExecutionPolicy(),
    peft_policy=_QWEN35_DENSE_PEFT_POLICY,
    requires=("transformers>=5.10.1", "module:transformers.models.qwen3_5"),
)


QWEN35VL_META = ModelMeta(
    model_type="qwen35vl",
    model_groups=(
        *default_model_groups(
            "qwen3.5-27b",
            "qwen3.6-27b",
            "qwen3.6-27b-fp8",
            name="dense",
            hf_model_types=("qwen3_5",),
            descriptor_matcher=_is_qwen35_dense_descriptor,
            template="qwen35vl",
            sharding_policy=ModelShardingPolicy(
                fsdp_transformer_layer_cls_to_wrap=(
                    "Qwen3_5DecoderLayer",
                    "Qwen3_5VisionBlock",
                ),
                supports_fsdp_activation_checkpointing=False,
            ),
        ),
        *default_model_groups(
            "qwen3.5-35b-a3b",
            "qwen3.6-35b-a3b",
            "qwen3.6-35b-a3b-fp8",
            name="moe",
            hf_model_types=("qwen3_5_moe",),
            descriptor_matcher=_is_qwen35_moe_descriptor,
            template="qwen35vl",
            sharding_policy=ModelShardingPolicy(
                fsdp_transformer_layer_cls_to_wrap=(
                    "Qwen3_5MoeDecoderLayer",
                    "Qwen3_5MoeVisionBlock",
                ),
                supports_fsdp_activation_checkpointing=False,
            ),
            training_objective_policy=QwenVLMoeTrainingObjectivePolicy(),
            peft_policy=_QWEN35_MOE_PEFT_POLICY,
            default_experts_implementation="grouped_mm",
            requires=("module:transformers.models.qwen3_5_moe",),
        ),
    ),
    sharding_policy=ModelShardingPolicy(
        fsdp_transformer_layer_cls_to_wrap=(
            "Qwen3_5DecoderLayer",
            "Qwen3_5VisionBlock",
        ),
        supports_fsdp_activation_checkpointing=False,
    ),
    **_QWEN35VL_COMMON,
)


QWEN36VL_META = ModelMeta(
    model_type="qwen36vl",
    model_groups=QWEN35VL_META.model_groups,
    sharding_policy=QWEN35VL_META.sharding_policy,
    **_QWEN35VL_COMMON,
)


@register_model(QWEN35VL_META)
class Qwen35VLLoader(Qwen3VLLoader):
    pass


@register_model(QWEN36VL_META)
class Qwen36VLLoader(Qwen3VLLoader):
    pass
