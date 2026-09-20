"""Instance-scoped loss-only forward for the HF dense Qwen3.5 VL model.

Parameters, module names and saved HF architecture remain unchanged. No global
Transformers/Liger monkeypatch: ordinary inference calls retain the upstream forward.
"""

from functools import wraps
from types import MethodType
from typing import Any

import torch

from .types import TrainingObjectivePolicy


class Qwen35VLTrainingObjectivePolicy(TrainingObjectivePolicy):
    def enable_liger_kernels(self, model: Any, *, rms_norm: bool, swiglu: bool) -> None:
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5ForConditionalGeneration,
            Qwen3_5MLP,
            Qwen3_5RMSNorm,
        )

        if type(model) is not Qwen3_5ForConditionalGeneration:
            raise ValueError("Liger kernels currently support native dense Qwen3.5 VL only.")
        from liger_kernel.transformers.monkey_patch import (
            _patch_rms_norm_module,
            _patch_swiglu_module,
        )
        from liger_kernel.transformers.swiglu import LigerQwen3MoeSwiGLUMLP

        # LF patches the classes before loading, including full-attention Q/K norms.
        # Patch the same module instances without changing global HF classes or weights.
        for module in model.modules():
            if rms_norm and isinstance(module, Qwen3_5RMSNorm):
                _patch_rms_norm_module(
                    module, offset=1.0, casting_mode="gemma", in_place=False
                )
            elif swiglu and isinstance(module, Qwen3_5MLP):
                _patch_swiglu_module(module, LigerQwen3MoeSwiGLUMLP)

    def enable_fused_linear_ce(self, model: Any) -> None:
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5CausalLMOutputWithPast,
            Qwen3_5ForConditionalGeneration,
        )

        if type(model) is not Qwen3_5ForConditionalGeneration:
            raise ValueError("Fused linear CE currently supports native dense Qwen3.5 VL only.")
        if type(model.lm_head) is not torch.nn.Linear:
            raise ValueError("Fused linear CE requires an unmodified torch.nn.Linear LM head.")
        original_forward = Qwen3_5ForConditionalGeneration.forward

        @wraps(original_forward)
        def forward(self, *args, shaft_loss=None, **kwargs):
            if shaft_loss is None:
                return original_forward(self, *args, **kwargs)
            selection = kwargs.pop("logits_to_keep", 0)
            if (
                args
                or kwargs.get("labels") is not None
                or not isinstance(selection, int)
                or selection != 0
            ):
                raise ValueError(
                    "Loss-only forward requires keyword inputs without labels/logit slicing."
                )
            kwargs.pop("labels", None)
            kwargs["return_dict"] = True
            outputs = self.model(**kwargs)
            loss = shaft_loss(hidden_states=outputs.last_hidden_state, lm_head=self.lm_head)
            return Qwen3_5CausalLMOutputWithPast(loss=loss, logits=None)

        model.forward = MethodType(forward, model)
