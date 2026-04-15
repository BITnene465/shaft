from __future__ import annotations

import copy

from shaft.plugins import Registry

from .types import DefaultPeftPolicy, PeftPolicy, ProcessorPolicy

PROCESSOR_POLICY_REGISTRY: Registry[ProcessorPolicy] = Registry("model_processor_policy")
PEFT_POLICY_REGISTRY: Registry[PeftPolicy] = Registry("model_peft_policy")


def register_processor_policy(name: str, policy: ProcessorPolicy):
    return PROCESSOR_POLICY_REGISTRY.register(name, policy)


def register_peft_policy(name: str, policy: PeftPolicy):
    return PEFT_POLICY_REGISTRY.register(name, policy)


def build_processor_policy(name: str) -> ProcessorPolicy:
    return copy.deepcopy(PROCESSOR_POLICY_REGISTRY.get(name))


def build_peft_policy(name: str) -> PeftPolicy:
    return copy.deepcopy(PEFT_POLICY_REGISTRY.get(name))


register_processor_policy("pixel_budget", ProcessorPolicy(supports_pixel_budget=True))
register_processor_policy("no_pixel_budget", ProcessorPolicy(supports_pixel_budget=False))

register_peft_policy("all_linear", DefaultPeftPolicy(target_modules=["all-linear"]))
