from .batch_planning import (
    BATCHING_RUN_METADATA_FILENAME,
    BOUNDED_BATCHING_CALLBACK_NAME,
    ShaftBoundedBatchingCallback,
    ShaftBatchingMetadataCallback,
    ShaftBatchingRunMetadata,
    batching_run_metadata_path,
    build_bounded_resume_contract_fingerprint,
    build_batching_run_metadata,
    checkpoint_has_bounded_batching_state,
    load_bounded_batching_state,
    load_batching_run_metadata,
    publish_batching_run_metadata,
    write_batching_run_metadata,
)
from .checkpointing import (
    ensure_hf_export_layout,
    resolve_resume_checkpoint,
    validate_resume_checkpoint,
    validate_training_state_policy,
)
from .distributed import (
    barrier_if_distributed,
    destroy_process_group_if_initialized,
    get_rank,
    get_world_size,
    is_distributed,
    is_rank_zero,
)
from .loss import LOSS_REGISTRY, build_loss, register_loss
from .muon import Muon
from .optimizer_mixin import ShaftOptimizerMixin
from .optimizer import OPTIMIZER_REGISTRY, build_optimizer, build_optimizer_and_plan, register_optimizer
from .optimizer_plan import (
    ShaftResolvedOptimizerGroupSummary,
    ShaftResolvedOptimizerPlan,
    ShaftResolvedOptimizerSummary,
    build_resolved_optimizer_plan,
    resolved_optimizer_summary_path,
    summarize_resolved_optimizer_plan,
    write_resolved_optimizer_summary,
)
from .online_eval import ShaftOnlineEvalRunner
from .epoch_interval_callback import ShaftEpochIntervalCallback
from .eval_policy import aggregate_weighted_dataset_values
from .progress_callback import ShaftProgressCallback
from .scheduler import SCHEDULER_REGISTRY, build_scheduler, register_scheduler
from .sft_trainer import ShaftSFTTrainer

__all__ = [
    "LOSS_REGISTRY",
    "BOUNDED_BATCHING_CALLBACK_NAME",
    "BATCHING_RUN_METADATA_FILENAME",
    "OPTIMIZER_REGISTRY",
    "SCHEDULER_REGISTRY",
    "Muon",
    "ShaftEpochIntervalCallback",
    "ShaftBoundedBatchingCallback",
    "ShaftBatchingMetadataCallback",
    "ShaftBatchingRunMetadata",
    "ShaftOnlineEvalRunner",
    "ShaftOptimizerMixin",
    "ShaftProgressCallback",
    "ShaftDPOTrainer",
    "ShaftGRPOTrainer",
    "ShaftResolvedOptimizerGroupSummary",
    "ShaftPPOTrainer",
    "ShaftResolvedOptimizerPlan",
    "ShaftResolvedOptimizerSummary",
    "ShaftSFTTrainer",
    "barrier_if_distributed",
    "batching_run_metadata_path",
    "build_bounded_resume_contract_fingerprint",
    "build_batching_run_metadata",
    "checkpoint_has_bounded_batching_state",
    "destroy_process_group_if_initialized",
    "build_loss",
    "build_optimizer",
    "build_optimizer_and_plan",
    "build_resolved_optimizer_plan",
    "build_scheduler",
    "aggregate_weighted_dataset_values",
    "ensure_hf_export_layout",
    "get_rank",
    "get_world_size",
    "is_distributed",
    "is_rank_zero",
    "load_bounded_batching_state",
    "load_batching_run_metadata",
    "publish_batching_run_metadata",
    "register_optimizer",
    "register_scheduler",
    "register_loss",
    "resolve_resume_checkpoint",
    "resolved_optimizer_summary_path",
    "summarize_resolved_optimizer_plan",
    "validate_resume_checkpoint",
    "validate_training_state_policy",
    "write_resolved_optimizer_summary",
    "write_batching_run_metadata",
]


def __getattr__(name: str):
    if name in {"ShaftDPOTrainer", "ShaftGRPOTrainer", "ShaftPPOTrainer"}:
        from .trl_trainers import ShaftDPOTrainer, ShaftGRPOTrainer, ShaftPPOTrainer

        return {
            "ShaftDPOTrainer": ShaftDPOTrainer,
            "ShaftGRPOTrainer": ShaftGRPOTrainer,
            "ShaftPPOTrainer": ShaftPPOTrainer,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
