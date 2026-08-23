from __future__ import annotations

from typing import Any

from shaft.algorithms import offline_kd as _offline_kd_algorithm  # noqa: F401
from shaft.config import RuntimeConfig
from shaft.offline_kd import (
    OfflineKDArtifactStore,
    OfflineKDCollator,
    OfflineKDDataset,
    build_offline_kd_input_contract,
)
from shaft.opd.input_abi import build_opd_input_abi
from shaft.plugins import ExecutionProxy
from shaft.training.distributed import initialize_process_group_if_needed
from shaft.training.resume_contract import distributed_training_contract_stage

from .execution import prepare_pipeline_call
from .registry import PIPELINE_REGISTRY, register_pipeline
from .sft import ShaftSFTPipeline


@register_pipeline("shaft_offline_kd")
class ShaftOfflineKDPipeline(ShaftSFTPipeline):
    """Offline teacher-forced distribution distillation pipeline."""

    algorithm_name = "offline_kd"
    dataset_cls = OfflineKDDataset
    collator_cls = OfflineKDCollator

    def __init__(self, config: RuntimeConfig):
        super().__init__(config)
        self._artifact_store: OfflineKDArtifactStore | None = None

    def on_model_artifacts(self, artifacts: Any) -> None:
        student_abi = build_opd_input_abi(artifacts)
        store = OfflineKDArtifactStore(
            self.config.offline_kd.artifact_manifest,
            student_input_abi=student_abi,
            student_input_contract=build_offline_kd_input_contract(self.config),
        )
        store.validate_objective(self.config.offline_kd.objective)
        self._artifact_store = store

    def build_data_collator(self, **kwargs: Any) -> Any:
        if self._artifact_store is None:
            raise RuntimeError("Offline KD artifact store was not validated before collation.")
        return self.collator_cls(artifact_store=self._artifact_store, **kwargs)

    def algorithm_trainer_kwargs(self) -> dict[str, Any]:
        return {"offline_kd_config": self.config.offline_kd}

    def input_contract_options(self) -> dict[str, Any]:
        if self._artifact_store is None:
            raise RuntimeError("Offline KD artifact store has not been initialized.")
        return {
            "offline_kd_artifact_id": self._artifact_store.artifact_id,
            "offline_kd_manifest_fingerprint": self._artifact_store.manifest_fingerprint,
            "offline_kd_input_abi_fingerprint": self._artifact_store.input_abi_fingerprint,
            "offline_kd_input_contract_fingerprint": (
                self._artifact_store.input_contract_fingerprint
            ),
        }


def run_offline_kd(config: RuntimeConfig) -> dict[str, Any]:
    initialize_process_group_if_needed(use_cpu=config.train.use_cpu)
    pipeline = None
    with distributed_training_contract_stage(
        stage="runtime-init",
        fingerprints=lambda: {
            "algorithm": "offline_kd",
            "hooks": "\x1f".join(config.plugins.hooks) or "none",
            "interceptors": "\x1f".join(config.plugins.interceptors) or "none",
        },
    ):
        pipeline_cls = PIPELINE_REGISTRY.get("shaft_offline_kd")
        pipeline = pipeline_cls(config)
        pipeline._bootstrap_training_args = pipeline.build_training_args()
        pipeline.initialize_runtime()
    assert pipeline is not None
    assert pipeline.interceptor_manager is not None
    runner = ExecutionProxy(
        point="pipeline.offline_kd.run",
        target=pipeline.run,
        interceptor_manager=pipeline.interceptor_manager,
    )
    try:
        invocation = prepare_pipeline_call(runner, stage="offline-kd-before-interceptors")
        return runner.invoke(invocation)
    except BaseException as exc:
        pipeline.progress_manager.record_failure(str(exc) or type(exc).__name__)
        raise
    finally:
        pipeline.close()
