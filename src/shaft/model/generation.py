from __future__ import annotations

from typing import Any


def align_model_generation_config(
    target: Any,
    *,
    tokenizer: Any = None,
    max_new_tokens: int | None = None,
    do_sample: bool | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    repetition_penalty: float | None = None,
) -> None:
    generation_config = getattr(target, "generation_config", target)
    if generation_config is None or not hasattr(generation_config, "do_sample"):
        return

    if max_new_tokens is not None:
        generation_config.max_new_tokens = int(max_new_tokens)
    if repetition_penalty is not None:
        generation_config.repetition_penalty = float(repetition_penalty)

    if do_sample is not None:
        generation_config.do_sample = bool(do_sample)
        if bool(do_sample):
            if temperature is not None:
                generation_config.temperature = float(temperature)
            if top_p is not None:
                generation_config.top_p = float(top_p)
            if top_k is not None:
                generation_config.top_k = int(top_k)
        else:
            generation_config.temperature = 1.0
            generation_config.top_p = 1.0
            generation_config.top_k = 50 if top_k is None else int(top_k)

    if tokenizer is not None:
        _align_special_tokens(target, tokenizer)


def _iter_use_cache_targets(model: Any):
    seen: set[int] = set()

    def emit(name: str, obj: Any):
        if obj is None or not hasattr(obj, "use_cache"):
            return
        obj_id = id(obj)
        if obj_id in seen:
            return
        seen.add(obj_id)
        yield name, obj

    yield from emit("config", getattr(model, "config", None))
    yield from emit("generation_config", getattr(model, "generation_config", None))

    root_config = getattr(model, "config", None)
    yield from emit("config.text_config", getattr(root_config, "text_config", None))

    inner_model = getattr(model, "model", None)
    yield from emit("model.config", getattr(inner_model, "config", None))
    language_model = getattr(inner_model, "language_model", None)
    yield from emit("model.language_model.config", getattr(language_model, "config", None))


def _align_special_tokens(target: Any, tokenizer: Any) -> None:
    tokenizer_eos = getattr(tokenizer, "eos_token_id", None)
    tokenizer_bos = getattr(tokenizer, "bos_token_id", None)
    tokenizer_pad = getattr(tokenizer, "pad_token_id", None)
    model_has_generation_config = (
        hasattr(target, "generation_config") and getattr(target, "generation_config") is not None
    )
    generation_config = getattr(target, "generation_config", None) if model_has_generation_config else None
    root_config = getattr(target, "config", target)

    tokenizer_has_new_eos = tokenizer_eos != getattr(root_config, "eos_token_id", None)
    existing_generation_eos = None
    if model_has_generation_config:
        existing_generation_eos = getattr(generation_config, "eos_token_id", None)
        if existing_generation_eos is None:
            tokenizer_has_new_eos |= tokenizer_eos != existing_generation_eos
        elif isinstance(existing_generation_eos, int):
            if tokenizer_eos != existing_generation_eos:
                generation_config.eos_token_id = [existing_generation_eos]
            tokenizer_has_new_eos |= tokenizer_eos != existing_generation_eos
        else:
            existing_generation_eos = list(existing_generation_eos)
            tokenizer_has_new_eos |= tokenizer_eos not in existing_generation_eos
    if tokenizer_has_new_eos:
        root_config.eos_token_id = tokenizer_eos
        if model_has_generation_config:
            eos_tokens = []
            current_generation_eos = getattr(generation_config, "eos_token_id", None)
            if current_generation_eos is not None:
                eos_tokens = list(current_generation_eos)
            generation_config.eos_token_id = [
                token for token in [tokenizer_eos, *eos_tokens] if token is not None
            ]

    tokenizer_has_new_bos = tokenizer_bos != getattr(root_config, "bos_token_id", None)
    if model_has_generation_config:
        tokenizer_has_new_bos |= tokenizer_bos != getattr(generation_config, "bos_token_id", None)
    if tokenizer_has_new_bos:
        root_config.bos_token_id = tokenizer_bos
        if model_has_generation_config:
            generation_config.bos_token_id = tokenizer_bos

    tokenizer_has_new_pad = tokenizer_pad != getattr(root_config, "pad_token_id", None)
    if model_has_generation_config:
        tokenizer_has_new_pad |= tokenizer_pad != getattr(generation_config, "pad_token_id", None)
    if tokenizer_has_new_pad:
        root_config.pad_token_id = tokenizer_pad
        if model_has_generation_config:
            generation_config.pad_token_id = tokenizer_pad


def set_model_use_cache(model: Any, enabled: bool) -> dict[str, bool]:
    previous: dict[str, bool] = {}
    for attr_name, config_obj in _iter_use_cache_targets(model):
        try:
            previous[attr_name] = bool(getattr(config_obj, "use_cache"))
            setattr(config_obj, "use_cache", bool(enabled))
        except Exception:  # noqa: BLE001
            continue
    return previous


def restore_model_use_cache(model: Any, previous: dict[str, bool]) -> None:
    targets = dict(_iter_use_cache_targets(model))
    for attr_name, value in previous.items():
        config_obj = targets.get(attr_name)
        if config_obj is None:
            continue
        try:
            setattr(config_obj, "use_cache", bool(value))
        except Exception:  # noqa: BLE001
            continue
