from .data import OPDCollator, OPDDataset, OPDRecord, load_jsonl_opd_records
from .execution import (
    OPD_EXECUTION_REGISTRY,
    OPDExecutionPlan,
    OPDExecutionRegistry,
    OPDExecutionRuntime,
    build_opd_execution_runtime,
    resolve_opd_execution_plan,
)
from .loss import (
    OPDLossComponents,
    OPDObjectiveBackend,
    OPDObjectivePlan,
    OPDTeacherDistribution,
    opd_distribution_loss,
    resolve_opd_objective_plan,
)
from .input_abi import (
    ShaftOPDInputABI,
    build_opd_input_abi,
    validate_opd_input_abi_compatibility,
)
from .rollout import (
    HFLocalOPDRolloutBackend,
    OPDRolloutBackend,
    OPDRolloutRequest,
    OPDRolloutResult,
    VLLMOPDRolloutBackend,
)
from .teacher import (
    HTTPRemoteOPDTeacherProvider,
    LocalHFOPDTeacherProvider,
    OPDTeacherProvider,
    OPDTeacherScoreRequest,
)
from .trainer import ShaftOPDTrainer
from . import resume as _resume  # noqa: F401

__all__ = [
    "OPDCollator",
    "OPDDataset",
    "OPD_EXECUTION_REGISTRY",
    "OPDExecutionPlan",
    "OPDExecutionRegistry",
    "OPDExecutionRuntime",
    "OPDRecord",
    "OPDLossComponents",
    "OPDObjectiveBackend",
    "OPDObjectivePlan",
    "OPDTeacherDistribution",
    "OPDRolloutBackend",
    "OPDRolloutRequest",
    "OPDRolloutResult",
    "OPDTeacherProvider",
    "OPDTeacherScoreRequest",
    "HFLocalOPDRolloutBackend",
    "HTTPRemoteOPDTeacherProvider",
    "VLLMOPDRolloutBackend",
    "LocalHFOPDTeacherProvider",
    "ShaftOPDTrainer",
    "ShaftOPDInputABI",
    "build_opd_input_abi",
    "build_opd_execution_runtime",
    "load_jsonl_opd_records",
    "opd_distribution_loss",
    "resolve_opd_execution_plan",
    "resolve_opd_objective_plan",
    "validate_opd_input_abi_compatibility",
]
