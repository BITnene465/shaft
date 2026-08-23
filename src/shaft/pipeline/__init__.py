from .registry import PIPELINE_REGISTRY

__all__ = [
    "ShaftSFTPipeline",
    "ShaftRLPipeline",
    "ShaftOPDPipeline",
    "ShaftOfflineKDPipeline",
    "PIPELINE_REGISTRY",
    "run_rl",
    "run_opd",
    "run_sft",
    "run_offline_kd",
]


def __getattr__(name: str):
    if name in {"ShaftSFTPipeline", "run_sft"}:
        from .sft import ShaftSFTPipeline, run_sft

        values = {
            "ShaftSFTPipeline": ShaftSFTPipeline,
            "run_sft": run_sft,
        }
        return values[name]
    if name in {"ShaftRLPipeline", "run_rl"}:
        from .rl import ShaftRLPipeline, run_rl

        values = {
            "ShaftRLPipeline": ShaftRLPipeline,
            "run_rl": run_rl,
        }
        return values[name]
    if name in {"ShaftOPDPipeline", "run_opd"}:
        from .opd import ShaftOPDPipeline, run_opd

        values = {
            "ShaftOPDPipeline": ShaftOPDPipeline,
            "run_opd": run_opd,
        }
        return values[name]
    if name in {"ShaftOfflineKDPipeline", "run_offline_kd"}:
        from .offline_kd import ShaftOfflineKDPipeline, run_offline_kd

        values = {
            "ShaftOfflineKDPipeline": ShaftOfflineKDPipeline,
            "run_offline_kd": run_offline_kd,
        }
        return values[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
