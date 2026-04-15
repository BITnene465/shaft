from __future__ import annotations

from shaft.plugins import Registry

from .types import Template, TemplateMeta

TEMPLATE_REGISTRY: Registry[TemplateMeta] = Registry("template")


def register_template(template_meta: TemplateMeta):
    def _decorator(template_cls: type[Template]):
        TEMPLATE_REGISTRY.register(
            template_meta.template_type,
            TemplateMeta(
                template_type=template_meta.template_type,
                template_cls=template_cls,
                default_system=template_meta.default_system,
                stop_words=template_meta.stop_words,
                support_multi_round=template_meta.support_multi_round,
                auto_add_generation_prompt=template_meta.auto_add_generation_prompt,
                response_prefix=template_meta.response_prefix,
                thinking_prefix=template_meta.thinking_prefix,
            ),
        )
        return template_cls

    return _decorator


def build_template_meta(name: str) -> TemplateMeta:
    return TEMPLATE_REGISTRY.get(name)


def resolve_template_meta(
    *,
    template_type: str | None = None,
    model_adapter=None,
    model_meta=None,
    model_info=None,
) -> TemplateMeta:
    resolved = str(template_type).strip().lower() if template_type else None
    if resolved:
        return build_template_meta(resolved)
    if model_adapter is not None:
        resolved = str(getattr(model_adapter, "template_type", "")).strip().lower()
        if resolved:
            return build_template_meta(resolved)
    if model_meta is not None and model_info is not None:
        resolved = getattr(model_meta, "resolve_template_type", lambda _: None)(getattr(model_info, "model_dir", None))
        if resolved:
            return build_template_meta(resolved)
    if model_meta is not None:
        candidates = [str(item).strip().lower() for item in getattr(model_meta, "candidate_templates", ()) if str(item).strip()]
        if len(candidates) == 1:
            return build_template_meta(candidates[0])
        if len(candidates) > 1:
            raise ValueError(f"Multiple candidate templates found for model_type={getattr(model_meta, 'model_type', None)!r}: {candidates}")
    raise ValueError(
        "Unable to resolve template meta automatically. Please provide `model.template` explicitly."
    )


def build_template(name: str) -> Template:
    template_meta = build_template_meta(name)
    if template_meta.template_cls is None:
        raise ValueError(f"Template {name!r} has no template class.")
    return template_meta.template_cls(template_meta)


def build_template_from_meta(template_meta: TemplateMeta) -> Template:
    if template_meta.template_cls is None:
        raise ValueError(f"Template {template_meta.template_type!r} has no template class.")
    return template_meta.template_cls(template_meta)
