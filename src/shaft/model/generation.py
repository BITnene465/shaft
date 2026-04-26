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
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            generation_config.eos_token_id = int(eos_token_id)
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is not None:
            generation_config.pad_token_id = int(pad_token_id)


def set_model_use_cache(model: Any, enabled: bool) -> dict[str, bool]:
    previous: dict[str, bool] = {}
    for attr_name in ("config", "generation_config"):
        config_obj = getattr(model, attr_name, None)
        if config_obj is None or not hasattr(config_obj, "use_cache"):
            continue
        try:
            previous[attr_name] = bool(getattr(config_obj, "use_cache"))
            setattr(config_obj, "use_cache", bool(enabled))
        except Exception:  # noqa: BLE001
            continue
    return previous


def restore_model_use_cache(model: Any, previous: dict[str, bool]) -> None:
    for attr_name, value in previous.items():
        config_obj = getattr(model, attr_name, None)
        if config_obj is None or not hasattr(config_obj, "use_cache"):
            continue
        try:
            setattr(config_obj, "use_cache", bool(value))
        except Exception:  # noqa: BLE001
            continue
