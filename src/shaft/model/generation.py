from __future__ import annotations

from contextlib import contextmanager
from typing import Any


_TRAINING_USE_CACHE_STATE = "_shaft_training_use_cache_state"


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
    existing_generation_eos: list[int] = []
    if model_has_generation_config:
        raw_generation_eos = getattr(generation_config, "eos_token_id", None)
        if raw_generation_eos is None:
            tokenizer_has_new_eos |= tokenizer_eos is not None
        elif isinstance(raw_generation_eos, int):
            existing_generation_eos = [int(raw_generation_eos)]
            tokenizer_has_new_eos |= tokenizer_eos not in existing_generation_eos
        else:
            existing_generation_eos = [
                int(token) for token in raw_generation_eos if token is not None
            ]
            tokenizer_has_new_eos |= tokenizer_eos not in existing_generation_eos
    if tokenizer_has_new_eos:
        root_config.eos_token_id = tokenizer_eos
        if model_has_generation_config:
            generation_config.eos_token_id = list(
                dict.fromkeys(
                    token
                    for token in [tokenizer_eos, *existing_generation_eos]
                    if token is not None
                )
            )

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


def set_model_use_cache(model: Any, enabled: bool) -> dict[str, Any]:
    previous: dict[str, Any] = {}
    for attr_name, config_obj in _iter_use_cache_targets(model):
        try:
            previous[attr_name] = getattr(config_obj, "use_cache")
            setattr(config_obj, "use_cache", bool(enabled))
        except Exception:  # noqa: BLE001
            continue
    return previous


def restore_model_use_cache(model: Any, previous: dict[str, Any]) -> None:
    targets = dict(_iter_use_cache_targets(model))
    for attr_name, value in previous.items():
        config_obj = targets.get(attr_name)
        if config_obj is None:
            continue
        try:
            setattr(config_obj, "use_cache", value)
        except Exception:  # noqa: BLE001
            continue


def disable_model_cache_for_training(model: Any) -> None:
    """Disable KV caches while retaining the artifact's deployment defaults."""

    existing = getattr(model, _TRAINING_USE_CACHE_STATE, None)
    if existing is None:
        state: list[tuple[Any, Any]] = []
        for _, config_obj in _iter_use_cache_targets(model):
            try:
                state.append((config_obj, getattr(config_obj, "use_cache")))
            except Exception:  # noqa: BLE001
                continue
        existing = tuple(state)
        setattr(model, _TRAINING_USE_CACHE_STATE, existing)
    for config_obj, _ in existing:
        try:
            setattr(config_obj, "use_cache", False)
        except Exception:  # noqa: BLE001
            continue


def _find_training_cache_state(model: Any) -> tuple[tuple[Any, Any], ...] | None:
    pending = [model]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        state = getattr(current, _TRAINING_USE_CACHE_STATE, None)
        if state is not None:
            return state
        for attr_name in ("module", "base_model", "model"):
            child = getattr(current, attr_name, None)
            if child is not None:
                pending.append(child)
    return None


@contextmanager
def export_model_cache(model: Any):
    """Temporarily restore deployment cache defaults while serializing a model."""

    state = _find_training_cache_state(model)
    if state is None:
        yield
        return
    training_state: list[tuple[Any, Any]] = []
    try:
        for config_obj, deployment_value in state:
            training_state.append((config_obj, getattr(config_obj, "use_cache")))
            setattr(config_obj, "use_cache", deployment_value)
        yield
    finally:
        for config_obj, training_value in training_state:
            try:
                setattr(config_obj, "use_cache", training_value)
            except Exception:  # noqa: BLE001
                continue
