from __future__ import annotations

import torch


def parameter_numel(parameter: torch.nn.Parameter) -> int:
    """Return the logical parameter size, including DeepSpeed placeholders."""

    deepspeed_numel = getattr(parameter, "ds_numel", None)
    if deepspeed_numel is not None:
        return int(deepspeed_numel)
    deepspeed_shape = getattr(parameter, "ds_shape", None)
    if deepspeed_shape is not None:
        total = 1
        for dimension in deepspeed_shape:
            total *= int(dimension)
        return int(total)
    return int(parameter.numel())


def model_parameter_count(
    model: torch.nn.Module,
    *,
    only_trainable: bool = False,
    exclude_embeddings: bool = False,
) -> int:
    """Mirror HF parameter-count semantics without trusting live shard sizes."""

    embedding_parameter_names: set[str] = set()
    if exclude_embeddings:
        embedding_parameter_names = {
            f"{name}.weight"
            for name, module in model.named_modules()
            if isinstance(module, torch.nn.Embedding)
        }

    params_4bit_type: type | None = None
    if bool(getattr(model, "is_loaded_in_4bit", False)):
        import bitsandbytes as bnb

        params_4bit_type = bnb.nn.Params4bit

    total = 0
    for name, parameter in model.named_parameters():
        if exclude_embeddings and name in embedding_parameter_names:
            continue
        if only_trainable and not parameter.requires_grad:
            continue
        count = parameter_numel(parameter)
        if params_4bit_type is not None and isinstance(parameter, params_4bit_type):
            if hasattr(parameter, "element_size"):
                num_bytes = parameter.element_size()
            elif hasattr(parameter, "quant_storage"):
                num_bytes = parameter.quant_storage.itemsize
            else:
                num_bytes = 1
            count *= 2 * int(num_bytes)
        total += count
    return int(total)
