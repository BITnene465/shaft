from __future__ import annotations

from .policies import build_peft_policy, build_processor_policy
from .qwen3vl import Qwen3VLLoader
from .registry import default_model_groups, register_model
from .sequence import Qwen35VLSequenceExecutionPolicy
from .sharding import ModelShardingPolicy
from .types import ModelCapabilities, ModelMeta, ModelModuleGroups
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
    sequence_execution_policy=Qwen35VLSequenceExecutionPolicy(),
    peft_policy=build_peft_policy("all_linear"),
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
            ),
            requires=("module:transformers.models.qwen3_5_moe",),
        ),
    ),
    sharding_policy=ModelShardingPolicy(
        fsdp_transformer_layer_cls_to_wrap=(
            "Qwen3_5DecoderLayer",
            "Qwen3_5VisionBlock",
        ),
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
