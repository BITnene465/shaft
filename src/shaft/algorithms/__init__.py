from .base import Algorithm, AlgorithmContext, ShaftTrainerSpec
from .registry import ALGORITHM_REGISTRY

__all__ = [
    "ALGORITHM_REGISTRY",
    "Algorithm",
    "AlgorithmContext",
    "ShaftTrainerSpec",
    "DPOAlgorithm",
    "GRPOAlgorithm",
    "PPOAlgorithm",
    "SFTAlgorithm",
]


def __getattr__(name: str):
    if name == "SFTAlgorithm":
        from .sft import SFTAlgorithm

        return SFTAlgorithm
    if name == "DPOAlgorithm":
        from .dpo import DPOAlgorithm

        return DPOAlgorithm
    if name == "GRPOAlgorithm":
        from .grpo import GRPOAlgorithm

        return GRPOAlgorithm
    if name == "PPOAlgorithm":
        from .ppo import PPOAlgorithm

        return PPOAlgorithm
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
