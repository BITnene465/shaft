from __future__ import annotations

from shaft.plugins import Registry

from .types import ModelGroup, ModelLoader, ModelMeta

MODEL_REGISTRY: Registry[ModelMeta] = Registry("model")


def register_model(meta: ModelMeta):
    def _decorator(loader_cls: type[ModelLoader]):
        MODEL_REGISTRY.register(meta.model_type, meta.with_loader(loader_cls()))
        return loader_cls

    return _decorator


def build_model_meta(name: str) -> ModelMeta:
    return MODEL_REGISTRY.get(name)


def default_model_groups(*model_ids: str, template: str | None = None) -> tuple[ModelGroup, ...]:
    cleaned = tuple(str(item).strip() for item in model_ids if str(item).strip())
    return (ModelGroup(name="default", model_ids=cleaned, template=template),)
