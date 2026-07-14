from __future__ import annotations

from collections.abc import Callable, Iterable

from shaft.plugins import Registry

from .sharding import ModelShardingPolicy
from .types import (
    ModelCapabilities,
    ModelGroup,
    ModelLoader,
    ModelMeta,
    ModelModuleGroups,
    PeftPolicy,
    ProcessorPolicy,
)
from .descriptor import ResolvedModelDescriptor

MODEL_REGISTRY: Registry[ModelMeta] = Registry("model")


def register_model(meta: ModelMeta):
    def _decorator(loader_cls: type[ModelLoader]):
        MODEL_REGISTRY.register(meta.model_type, meta.with_loader(loader_cls()))
        return loader_cls

    return _decorator


def build_model_meta(name: str) -> ModelMeta:
    return MODEL_REGISTRY.get(name)


def default_model_groups(
    *model_ids: str,
    template: str | None = None,
    name: str = "default",
    hf_model_types: Iterable[str] = (),
    capabilities: ModelCapabilities | None = None,
    module_groups: ModelModuleGroups | None = None,
    processor_policy: ProcessorPolicy | None = None,
    peft_policy: PeftPolicy | None = None,
    sharding_policy: ModelShardingPolicy | None = None,
    requires: Iterable[str] = (),
    additional_saved_files: Iterable[str] = (),
    descriptor_matcher: Callable[[ResolvedModelDescriptor], bool] | None = None,
) -> tuple[ModelGroup, ...]:
    cleaned = tuple(str(item).strip() for item in model_ids if str(item).strip())
    return (
        ModelGroup(
            name=str(name).strip() or "default",
            model_ids=cleaned,
            hf_model_types=tuple(
                str(item).strip() for item in hf_model_types if str(item).strip()
            ),
            template=template,
            capabilities=capabilities,
            module_groups=module_groups,
            processor_policy=processor_policy,
            peft_policy=peft_policy,
            sharding_policy=sharding_policy,
            requires=tuple(str(item).strip() for item in requires if str(item).strip()),
            additional_saved_files=tuple(
                str(item).strip() for item in additional_saved_files if str(item).strip()
            ),
            descriptor_matcher=descriptor_matcher,
        ),
    )
