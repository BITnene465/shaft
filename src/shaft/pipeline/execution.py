from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from shaft.plugins import ExecutionProxy
from shaft.plugins.interceptors import ShaftInterceptorInvocation
from shaft.training.distributed import is_rank_zero
from shaft.training.resume_contract import distributed_training_contract_stage


def prepare_pipeline_call(
    runner: ExecutionProxy,
    *,
    stage: str,
) -> ShaftInterceptorInvocation:
    """Converge rank-local before interceptors before a collective-owning target."""

    manager = runner.interceptor_manager
    normalized_point = str(runner.point).strip().lower()
    invocation = None
    schedule_fingerprint = "none"
    with distributed_training_contract_stage(
        stage=stage,
        fingerprints=lambda: {
            "point": normalized_point,
            "interceptor_schedule": schedule_fingerprint,
        },
    ):
        if not normalized_point:
            raise ValueError("Pipeline interceptor point cannot be empty.")
        if manager is not None:
            schedule_fingerprint = manager.semantic_schedule_fingerprint(
                point=normalized_point
            )
        invocation = runner.prepare()
        prepared_args = invocation.state.get("args")
        prepared_kwargs = invocation.state.get("kwargs")
        if (
            type(prepared_args) is not tuple
            or prepared_args
            or type(prepared_kwargs) is not dict
            or prepared_kwargs
        ):
            raise ValueError(
                "Pipeline before interceptors must preserve the zero-argument call contract."
            )
    assert invocation is not None
    return invocation


def finalize_training_outputs(
    *,
    trainer: Any,
    best_export_dir: str | Path | None,
    save_final_state: bool,
    validate_export: Callable[[Path], None] | None,
    prune_output: Callable[[], None],
) -> None:
    """Save a collective-owned model, then converge rank-local finalization.

    FSDP/DeepSpeed may run collectives inside ``save_model``. It therefore runs
    only after a shared readiness preflight and outside the local status envelope.
    Once that owning API has returned on every rank, export validation, Trainer
    state persistence, and root-layout pruning are rank-local and can safely share
    one failure-convergence stage.
    """

    normalized_best_export_dir: Path | None = None
    with distributed_training_contract_stage(
        stage="post-training-save-preflight",
        fingerprints=lambda: {
            "save_final_model": str(normalized_best_export_dir is not None).lower(),
            "save_final_state": str(save_final_state).lower(),
            "best_export_dir": (
                "disabled"
                if normalized_best_export_dir is None
                else str(normalized_best_export_dir)
            ),
        },
    ):
        if type(save_final_state) is not bool:
            raise TypeError("save_final_state must be a boolean.")
        if best_export_dir is None and validate_export is not None:
            raise ValueError("validate_export requires best_export_dir.")
        if best_export_dir is not None and not callable(validate_export):
            raise TypeError("A final model export requires a validation callback.")
        if not callable(prune_output):
            raise TypeError("prune_output must be callable.")
        if best_export_dir is not None:
            if type(best_export_dir) is str and not best_export_dir.strip():
                raise ValueError("best_export_dir cannot be an empty path.")
            normalized_best_export_dir = Path(best_export_dir).expanduser().resolve(
                strict=False
            )

    save_error: Exception | None = None
    if normalized_best_export_dir is not None:
        try:
            trainer.save_model(output_dir=str(normalized_best_export_dir))
        except Exception as exc:  # noqa: BLE001 - converge after collective owner returns
            save_error = exc

    with distributed_training_contract_stage(
        stage="post-training-finalization",
        fingerprints=lambda: {
            "save_final_model": str(normalized_best_export_dir is not None).lower(),
            "save_final_state": str(save_final_state).lower(),
            "best_export_dir": (
                "disabled"
                if normalized_best_export_dir is None
                else str(normalized_best_export_dir)
            ),
        },
    ):
        if save_error is not None:
            raise save_error
        if is_rank_zero() and validate_export is not None:
            assert normalized_best_export_dir is not None
            validate_export(normalized_best_export_dir)
        if save_final_state:
            trainer.save_state()
        if is_rank_zero():
            prune_output()
