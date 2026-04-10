from __future__ import annotations

__all__ = [
    "Stage2KeypointInferenceRunner",
    "TwoStageInferenceRunner",
    "draw_prediction",
    "format_prediction_summary",
    "load_two_stage_inference_runner",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(name)
    from . import infer as _infer

    return getattr(_infer, name)
