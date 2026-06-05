from .registry import PIPELINE_REGISTRY

__all__ = [
    "ShaftSFTPipeline",
    "ShaftRLHFPipeline",
    "PIPELINE_REGISTRY",
    "run_rlhf",
    "run_sft",
]


def __getattr__(name: str):
    if name in {"ShaftSFTPipeline", "run_sft"}:
        from .sft import ShaftSFTPipeline, run_sft

        values = {
            "ShaftSFTPipeline": ShaftSFTPipeline,
            "run_sft": run_sft,
        }
        return values[name]
    if name in {"ShaftRLHFPipeline", "run_rlhf"}:
        from .rlhf import ShaftRLHFPipeline, run_rlhf

        values = {
            "ShaftRLHFPipeline": ShaftRLHFPipeline,
            "run_rlhf": run_rlhf,
        }
        return values[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
