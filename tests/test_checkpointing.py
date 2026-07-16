from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from tokenizers import AddedToken

import shaft.training.checkpointing as checkpointing_module
import shaft.training.resume_contract as resume_contract_module
from shaft.config import RuntimeConfig
from shaft.data import ShaftBatchPlanningSpec, ShaftBatchPlanningState
from shaft.model.input_identity import stable_artifact_value
from shaft.observability import PROGRESS_SNAPSHOT_FILENAME
from shaft.observability import TRAINING_EFFICIENCY_FILENAME
from shaft.plugins import (
    build_hook_manager,
    build_interceptor_manager,
    hook,
    interceptor,
)
from shaft.training.batch_planning import (
    BATCHING_METADATA_CALLBACK_NAME,
    BATCHING_RUN_METADATA_FILENAME,
    BATCH_PLANNING_CALLBACK_NAME,
    ShaftBatchContract,
    ShaftBatchingMetadataCallback,
    ShaftBatchingRunMetadata,
    ShaftBatchPlanningCallback,
    checkpoint_has_batch_planning_state,
    load_batching_run_metadata,
    load_checkpoint_batching_metadata,
    load_batch_planning_state,
    publish_batching_run_metadata,
    validate_batching_resume_contract,
    validate_batch_planning_resume_contract,
    write_batching_run_metadata,
)
from shaft.training.checkpointing import (
    TRAINING_CHECKPOINT_COMMIT_FILENAME,
    ShaftCheckpointCommitMixin,
    ShaftCheckpointProtocol,
    _validate_shared_callback_schedule,
    commit_training_checkpoint,
    ensure_hf_export_layout,
    revoke_training_checkpoint_commit,
    prune_root_output_layout,
    resolve_best_export_dir,
    resolve_checkpoint_protocol,
    resolve_resume_checkpoint,
    resolve_resume_checkpoint_generation,
    resume_checkpoint_consensus_fingerprints,
    training_checkpoint_is_committed,
    validate_training_checkpoint_commit,
    validate_resume_checkpoint,
    validate_resolved_resume_checkpoint_guard,
    validate_training_state_policy,
)
from shaft.training.input_contract import (
    ShaftTrainInputContract,
    _canonical_value,
    _package_distributions,
    _package_version,
    build_train_input_contract,
    callable_semantic_signature,
    component_semantic_signature,
    validate_train_data_identity_checkpointability,
    validate_train_input_checkpointability,
)
from shaft.training.resume_contract import (
    ShaftTrainingResumeContract,
    build_training_resume_contract as _build_training_resume_contract,
    distributed_training_contract_stage,
)


def _spec(**changes) -> ShaftBatchPlanningSpec:
    values = {
        "data_world_size": 2,
        "buffer_size": 16,
        "per_device_microbatch_size": 1,
        "max_tokens_per_microbatch": 1024,
        "resource_budgets": (("vision_patches", 2048),),
        "seed": 42,
        "sample_schedule_fingerprint": "schedule-v1",
        "cost_fingerprint": "cost-v1",
    }
    values.update(changes)
    return ShaftBatchPlanningSpec(**values)


def _train_input_contract(**changes) -> ShaftTrainInputContract:
    values = {
        "algorithm": "sft",
        "data_execution_fingerprint": "data-v1",
        "data_execution_contract_complete": True,
        "incomplete_reasons": (),
        "train_dataset_signature": "dataset-v1",
        "model_plan_fingerprint": "model-v1",
        "model_adapter_signature": "adapter-v1",
        "processor_signature": "processor-v1",
        "tokenizer_signature": "tokenizer-v1",
        "template_signature": "template-v1",
        "input_builder_signature": "collator-v1",
        "input_policy_version": "sft-input-v1",
        "input_options": (
            ("max_length", 4096),
            ("max_pixels", 1_048_576),
            ("min_pixels", 200_704),
        ),
    }
    values.update(changes)
    return ShaftTrainInputContract(**values)


def _fixed_batch_contract() -> ShaftBatchContract:
    return ShaftBatchContract(
        grouping="none",
        cardinality="fixed",
        packing="none",
        layout="padded",
        per_device_microbatch_size=1,
        data_world_size=1,
        gradient_accumulation_steps=2,
    )


def build_training_resume_contract(**kwargs) -> ShaftTrainingResumeContract:
    algorithm = str(kwargs["config"].algorithm.name).strip().lower()
    input_contract = _train_input_contract(algorithm=algorithm)
    kwargs.setdefault(
        "train_input_contract_fingerprint",
        input_contract.fingerprint,
    )
    kwargs.setdefault(
        "data_execution_fingerprint",
        input_contract.data_execution_fingerprint,
    )
    kwargs.setdefault("model_plan_fingerprint", "fixture-model-plan-v1")
    kwargs.setdefault(
        "resolved_finetune_plan_fingerprint",
        "fixture-finetune-plan-v1",
    )
    kwargs.setdefault(
        "resolved_optimizer_plan_fingerprint",
        "fixture-optimizer-plan-v1",
    )
    return _build_training_resume_contract(**kwargs)


def _resume_training_args(**changes) -> SimpleNamespace:
    values = {
        "max_steps": 100,
        "num_train_epochs": 1.0,
        "gradient_accumulation_steps": 2,
        "full_determinism": True,
        "ddp_static_graph": False,
        "bf16": True,
        "fp16": False,
        "gradient_checkpointing": False,
    }
    values.update(changes)
    return SimpleNamespace(**values)


def _resolved_grpo_args(**changes) -> SimpleNamespace:
    values = {
        "beta": 0.0,
        "num_generations": 2,
        "num_generations_eval": 1,
        "max_completion_length": 256,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": None,
        "repetition_penalty": 1.0,
        "generation_kwargs": None,
        "cache_implementation": None,
        "use_transformers_paged": False,
        "steps_per_generation": 2,
        "num_iterations": 1,
        "generation_batch_size": 4,
        "shuffle_dataset": False,
        "use_vllm": False,
        "vllm_mode": "server",
        "vllm_model_impl": "vllm",
        "vllm_structured_outputs_regex": None,
        "loss_type": "dapo",
        "scale_rewards": "group",
    }
    values.update(changes)
    return SimpleNamespace(**values)


def _resolved_dpo_args(**changes) -> SimpleNamespace:
    values = {
        "disable_dropout": True,
        "pad_token": None,
        "max_length": 1024,
        "truncation_mode": "keep_end",
        "padding_free": False,
        "pad_to_multiple_of": None,
        "precompute_ref_log_probs": False,
        "precompute_ref_batch_size": None,
        "loss_type": "sigmoid",
        "loss_weights": None,
        "ld_alpha": None,
        "f_divergence_type": "reverse_kl",
        "f_alpha_divergence_coef": 1.0,
        "label_smoothing": 0.0,
        "beta": 0.1,
        "use_weighting": False,
        "discopop_tau": 0.05,
        "activation_offloading": False,
        "sync_ref_model": False,
        "ref_model_mixup_alpha": 0.9,
        "ref_model_sync_steps": 64,
    }
    values.update(changes)
    return SimpleNamespace(**values)


def _metadata_with_training_resume_contract(
    contract: ShaftTrainingResumeContract,
) -> ShaftBatchingRunMetadata:
    input_contract = _train_input_contract(algorithm=contract.algorithm)
    return ShaftBatchingRunMetadata(
        grouping="none",
        cardinality="fixed",
        packing="none",
        layout="padded",
        per_device_train_batch_size=1,
        data_world_size=1,
        gradient_accumulation_steps=2,
        min_pixels=None,
        max_pixels=None,
        source_weights=(("fixture", 1.0),),
        media_snapshot_id="fixture-v1",
        sample_execution_fingerprint=input_contract.data_execution_fingerprint,
        train_input_contract=input_contract,
        training_resume_contract=contract,
    )


def _write_metadata_checkpoint(
    path: Path,
    metadata: ShaftBatchingRunMetadata,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    payload = {
        "global_step": 1,
        "stateful_callbacks": {
            BATCHING_METADATA_CALLBACK_NAME: ShaftBatchingMetadataCallback(metadata).state()
        },
    }
    (path / "trainer_state.json").write_text(json.dumps(payload), encoding="utf-8")


class _ContractTokenizerBackend:
    def __init__(self, payload: str) -> None:
        self.payload = payload

    def to_str(self) -> str:
        return self.payload


class _ContractTokenizer:
    def __init__(
        self,
        payload: str,
        vocab: dict[str, int] | None = None,
        *,
        artifact_root: Path | None = None,
        normalizer_mode: str = "default",
        revision: str = "main",
    ) -> None:
        self.backend_tokenizer = _ContractTokenizerBackend(payload)
        self._vocab = dict(vocab or {"fixture": 1})
        self.name_or_path = "" if artifact_root is None else str(artifact_root)
        self.special_tokens_map = {"eos_token": "</s>"}
        self.init_kwargs = {
            "model_max_length": 4096,
            "normalizer_mode": normalizer_mode,
            "revision": revision,
        }
        if artifact_root is not None:
            self.init_kwargs.update(
                {
                    "name_or_path": str(artifact_root),
                    "tokenizer_file": str(artifact_root / "tokenizer.json"),
                    "cache_dir": str(artifact_root.parent / "cache"),
                    "is_local": True,
                    "local_files_only": True,
                }
            )
        self.added_tokens_encoder = {}
        self.eos_token_id = 2
        self.bos_token_id = 1
        self.pad_token_id = 0

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)


class _ContractProcessor:
    def __init__(
        self,
        patch_size: int,
        *,
        artifact_root: Path | None = None,
    ) -> None:
        self.patch_size = patch_size
        self.name_or_path = "" if artifact_root is None else str(artifact_root)

    def to_dict(self) -> dict[str, int]:
        return {"patch_size": self.patch_size}


class _ContractTemplate:
    template_meta = {"template_type": "fixture-v1"}


class _ChangedContractTemplate:
    template_meta = {"template_type": "fixture-v2"}


class _ContractInputBuilder:
    SHAFT_INPUT_POLICY_VERSION = "fixture-input-v1"


class _ChangedContractInputBuilder:
    SHAFT_INPUT_POLICY_VERSION = "fixture-input-v2"


class _ContractDataset:
    def __getitem__(self, index: int) -> dict[str, int]:
        return {"index": index}


class _ChangedContractDataset:
    def __getitem__(self, index: int) -> dict[str, int]:
        return {"changed_index": index}


def _fixture_reward_v1(**kwargs) -> list[float]:
    _ = kwargs
    return [1.0]


def _fixture_reward_v2(**kwargs) -> list[float]:
    _ = kwargs
    return [0.0]


def _fixture_builder_v1(**kwargs):
    return kwargs


def _fixture_builder_v2(**kwargs):
    _ = kwargs
    return None


def _closure_builder(scale: int):
    def builder(**kwargs):
        return int(scale), kwargs

    return builder


def _fixture_codec_helper(raw_text: str):
    return {"patched": raw_text}


def _fixture_reference_model_v1(*, model, finetune_mode):
    _ = finetune_mode
    return model


def _fixture_reference_model_v2(*, model, finetune_mode):
    _ = finetune_mode
    return None if model is None else model


def _fixture_reproducibility_policy_v2():
    return "changed"


class _FixtureMuonV1:
    def step(self):
        return 1


class _FixtureMuonV2:
    def step(self):
        return 2


def _metadata_for_spec(
    spec: ShaftBatchPlanningSpec,
    *,
    gradient_accumulation_steps: int,
    cost_cache_size: int = 0,
) -> ShaftBatchingRunMetadata:
    return ShaftBatchingRunMetadata(
        grouping="bounded_cost",
        cardinality=spec.cardinality,
        packing="none",
        layout="padded",
        per_device_train_batch_size=spec.per_device_microbatch_size,
        data_world_size=spec.data_world_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        min_pixels=None,
        max_pixels=None,
        source_weights=(("fixture", 1.0),),
        media_snapshot_id="fixture-media-v1",
        buffer_size=spec.buffer_size,
        cost_cache_size=cost_cache_size,
        max_tokens_per_microbatch=spec.max_tokens_per_microbatch,
        resource_budgets=spec.resource_budgets,
        planner_spec_fingerprint=spec.fingerprint,
    )


def _write_bounded_trainer_state(
    path: Path,
    *,
    spec: ShaftBatchPlanningSpec,
    state: ShaftBatchPlanningState,
    resume_contract_fingerprint: str = "resume-v1",
    gradient_accumulation_steps: int = 2,
    global_step: int | None = None,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    resolved_global_step = (
        int(state.global_microstep) // int(gradient_accumulation_steps)
        if global_step is None
        else int(global_step)
    )
    payload = {
        "global_step": resolved_global_step,
        "stateful_callbacks": {
            BATCH_PLANNING_CALLBACK_NAME: {
                "args": {
                    "spec": spec.to_dict(),
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                    "resume_contract_fingerprint": resume_contract_fingerprint,
                },
                "attributes": {"planning_state": state.to_dict()},
            },
            BATCHING_METADATA_CALLBACK_NAME: ShaftBatchingMetadataCallback(
                _metadata_for_spec(
                    spec,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                )
            ).state(),
        },
    }
    (path / "trainer_state.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"model")
    _write_exact_resume_artifacts(path, world_size=spec.data_world_size)
    commit_training_checkpoint(
        path,
        world_size=spec.data_world_size,
        requires_grad_scaler=False,
    )


def _write_exact_resume_artifacts(path: Path, *, world_size: int) -> None:
    (path / "optimizer.pt").write_bytes(b"optimizer")
    (path / "scheduler.pt").write_bytes(b"scheduler")
    if int(world_size) <= 1:
        (path / "rng_state.pth").write_bytes(b"rng")
        return
    for rank in range(int(world_size)):
        (path / f"rng_state_{rank}.pth").write_bytes(b"rng")


def test_validate_training_state_policy_requires_eval_for_best_model() -> None:
    cfg = RuntimeConfig()
    cfg.train.load_best_model_at_end = True
    cfg.eval.enabled = False
    with pytest.raises(ValueError):
        validate_training_state_policy(cfg)


def test_validate_training_state_policy_rejects_init_and_resume_together() -> None:
    cfg = RuntimeConfig()
    cfg.train.init_from_checkpoint = "init-checkpoint"
    cfg.train.resume_from_checkpoint = "resume-checkpoint"

    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_training_state_policy(cfg)


def test_validate_training_state_policy_requires_matching_strategies() -> None:
    cfg = RuntimeConfig()
    cfg.train.load_best_model_at_end = True
    cfg.train.save_strategy = "epoch"
    cfg.eval.enabled = True
    cfg.eval.eval_strategy = "steps"
    with pytest.raises(ValueError):
        validate_training_state_policy(cfg)


def test_resolve_resume_checkpoint_picks_last_checkpoint(tmp_path: Path) -> None:
    root = tmp_path / "run"
    for step in (1, 2):
        checkpoint = root / f"checkpoint-{step}"
        _write_full_checkpoint(checkpoint, global_step=step)
        _write_exact_resume_artifacts(checkpoint, world_size=1)
        commit_training_checkpoint(
            checkpoint,
            world_size=1,
            requires_grad_scaler=False,
        )

    assert resolve_resume_checkpoint(
        root,
        protocol=ShaftCheckpointProtocol.COMMITTED_MANIFEST,
    ) == str(root / "checkpoint-2")


def test_typed_resume_resolution_hashes_only_newest_valid_generation_once(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    for step in (1, 2, 3):
        checkpoint = root / f"checkpoint-{step}"
        _write_full_checkpoint(checkpoint, global_step=step)
        _write_exact_resume_artifacts(checkpoint, world_size=1)
        commit_training_checkpoint(
            checkpoint,
            world_size=1,
            requires_grad_scaler=False,
        )

    newest_manifest = json.loads(
        (root / "checkpoint-3" / TRAINING_CHECKPOINT_COMMIT_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    with patch.object(
        checkpointing_module,
        "_sha256",
        wraps=checkpointing_module._sha256,  # noqa: SLF001
    ) as hash_file:
        resolved = resolve_resume_checkpoint_generation(
            root,
            protocol=ShaftCheckpointProtocol.COMMITTED_MANIFEST,
        )
        assert resolved is not None
        assert resolved.path == (root / "checkpoint-3").resolve()
        assert resolved.global_step == 3
        assert hash_file.call_count == len(newest_manifest["artifacts"])

        validate_resume_checkpoint(
            resolved,
            finetune_mode="full",
            protocol=ShaftCheckpointProtocol.COMMITTED_MANIFEST,
        )
        validate_resolved_resume_checkpoint_guard(resolved)
        assert hash_file.call_count == len(newest_manifest["artifacts"])


def test_checkpoint_commit_hashes_each_artifact_once(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-3"
    _write_full_checkpoint(checkpoint, global_step=3)
    _write_exact_resume_artifacts(checkpoint, world_size=1)
    artifact_count = sum(1 for path in checkpoint.rglob("*") if path.is_file())

    with patch.object(
        checkpointing_module,
        "_sha256",
        wraps=checkpointing_module._sha256,  # noqa: SLF001
    ) as hash_file:
        commit_training_checkpoint(
            checkpoint,
            world_size=1,
            requires_grad_scaler=False,
        )

    assert hash_file.call_count == artifact_count


def test_resolved_resume_generation_guard_rejects_late_artifact_mutation(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint-4"
    _write_full_checkpoint(checkpoint, global_step=4)
    _write_exact_resume_artifacts(checkpoint, world_size=1)
    commit_training_checkpoint(
        checkpoint,
        world_size=1,
        requires_grad_scaler=False,
    )
    resolved = resolve_resume_checkpoint_generation(
        checkpoint,
        protocol=ShaftCheckpointProtocol.COMMITTED_MANIFEST,
    )
    assert resolved is not None
    consensus = resume_checkpoint_consensus_fingerprints(
        resolved,
        protocol=ShaftCheckpointProtocol.COMMITTED_MANIFEST,
    )

    replacement = checkpoint / "model.safetensors.replacement"
    replacement.write_bytes(b"other")
    replacement.replace(checkpoint / "model.safetensors")

    assert consensus["resume_global_step"] == "4"
    assert consensus["resume_generation"] == resolved.generation_fingerprint
    with pytest.raises(ValueError, match="changed during startup"):
        validate_resolved_resume_checkpoint_guard(resolved)


def test_committed_resolver_prioritizes_checkpoint_children_over_root_layout(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    _write_full_checkpoint(root, global_step=99)
    child = root / "checkpoint-2"
    _write_full_checkpoint(child, global_step=2)
    _write_exact_resume_artifacts(child, world_size=1)
    commit_training_checkpoint(child, world_size=1, requires_grad_scaler=False)

    assert resolve_resume_checkpoint(
        root,
        protocol=ShaftCheckpointProtocol.COMMITTED_MANIFEST,
    ) == str(child)


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [
        ("ddp", ShaftCheckpointProtocol.COMMITTED_MANIFEST),
        ("fsdp", ShaftCheckpointProtocol.BACKEND_NATIVE),
        ("deepspeed", ShaftCheckpointProtocol.BACKEND_NATIVE),
    ],
)
def test_checkpoint_protocol_routes_by_distributed_strategy(
    strategy: str,
    expected: ShaftCheckpointProtocol,
) -> None:
    assert resolve_checkpoint_protocol(strategy) is expected


def test_callback_schedule_accepts_identical_order_on_every_rank() -> None:
    schedule = (("a", 0), ("b", 0), ("d", 0))

    assert _validate_shared_callback_schedule([schedule, schedule]) == schedule


@pytest.mark.parametrize(
    "status",
    [
        {"ok": "false", "error_type": "OSError", "error": "disk"},
        {"ok": True, "path": "/tmp/metadata", "extra": True},
        {"ok": False, "error_type": "", "error": "disk"},
    ],
)
def test_batching_metadata_publish_rejects_malformed_status_envelope(
    tmp_path: Path,
    status: dict[str, object],
) -> None:
    metadata = _metadata_for_spec(
        _spec(),
        gradient_accumulation_steps=2,
    )
    with (
        patch(
            "shaft.training.batch_planning.is_rank_zero",
            return_value=False,
        ),
        patch(
            "shaft.training.batch_planning.broadcast_object_from_rank_zero",
            return_value=status,
        ),
    ):
        with pytest.raises(RuntimeError, match="malformed status envelope"):
            publish_batching_run_metadata(tmp_path, metadata)


def test_batching_run_metadata_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    metadata = _metadata_for_spec(
        _spec(),
        gradient_accumulation_steps=2,
    )
    target = write_batching_run_metadata(tmp_path, metadata)
    document = target.read_text(encoding="utf-8")
    document = document.replace(
        '"version":',
        '"version": "shadow",\n  "version":',
        1,
    )
    target.write_text(document, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_batching_run_metadata(tmp_path)


def test_checkpoint_batching_metadata_rejects_duplicate_trainer_state_keys(
    tmp_path: Path,
) -> None:
    (tmp_path / "trainer_state.json").write_text(
        '{"global_step": 1, "global_step": 1}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_checkpoint_batching_metadata(tmp_path)


def test_callback_schedule_rejects_conflicting_rank_order() -> None:
    with pytest.raises(RuntimeError, match="identical ordered"):
        _validate_shared_callback_schedule(
            [
                (("a", 0), ("b", 0)),
                (("b", 0), ("a", 0)),
            ]
        )


def test_callback_schedule_rejects_rank_local_subset() -> None:
    with pytest.raises(RuntimeError, match="identical ordered"):
        _validate_shared_callback_schedule(
            [
                (("a", 0), ("a", 1)),
                (("a", 0),),
            ]
        )


@pytest.mark.parametrize(
    "schedule",
    [
        "callback:0",
        (("callback", True),),
        (("callback", "0"),),
        (("callback", 1),),
        (("", 0),),
        (("callback", 0, "extra"),),
    ],
)
def test_callback_schedule_rejects_malformed_tokens(schedule: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        _validate_shared_callback_schedule([schedule])


def test_callback_convergence_rejects_divergent_trainer_control() -> None:
    control = SimpleNamespace(
        should_training_stop=False,
        should_epoch_stop=False,
        should_save=False,
        should_evaluate=False,
        should_log=False,
    )
    local_state = {
        "should_training_stop": False,
        "should_epoch_stop": False,
        "should_save": False,
        "should_evaluate": False,
        "should_log": False,
    }
    peer_state = dict(local_state, should_training_stop=True)
    with patch(
        "shaft.training.checkpointing.all_gather_objects",
        return_value=[
            {"ok": True, "error_type": None, "error": None, "control": local_state},
            {"ok": True, "error_type": None, "error": None, "control": peer_state},
        ],
    ):
        with pytest.raises(RuntimeError, match="divergent TrainerControl"):
            ShaftCheckpointCommitMixin._raise_synchronized_checkpoint_callback_error(
                "test callback",
                None,
                control,
            )


def test_callback_convergence_gathers_malformed_control_before_raising() -> None:
    with patch(
        "shaft.training.checkpointing.all_gather_objects",
        side_effect=lambda value: [value],
    ) as gather:
        with pytest.raises(AttributeError, match="should_training_stop"):
            ShaftCheckpointCommitMixin._raise_synchronized_checkpoint_callback_error(
                "test callback",
                None,
                object(),
            )

    gather.assert_called_once()


def test_checkpoint_convergence_rejects_string_boolean_status() -> None:
    with patch(
        "shaft.training.checkpointing.all_gather_objects",
        return_value=[
            {"ok": "false", "error_type": "OSError", "error": "disk"},
        ],
    ):
        with pytest.raises(RuntimeError, match="malformed status envelope"):
            ShaftCheckpointCommitMixin._raise_synchronized_checkpoint_error(
                "test save",
                None,
            )


def test_callback_convergence_rejects_string_boolean_status() -> None:
    control = SimpleNamespace(
        should_training_stop=False,
        should_epoch_stop=False,
        should_save=False,
        should_evaluate=False,
        should_log=False,
    )
    control_state = dict(vars(control))
    with patch(
        "shaft.training.checkpointing.all_gather_objects",
        return_value=[
            {
                "ok": "false",
                "error_type": "OSError",
                "error": "disk",
                "control": control_state,
            },
        ],
    ):
        with pytest.raises(RuntimeError, match="malformed status envelope"):
            ShaftCheckpointCommitMixin._raise_synchronized_checkpoint_callback_error(
                "test callback",
                None,
                control,
            )


def test_callback_convergence_rejects_string_boolean_control() -> None:
    control = SimpleNamespace(
        should_training_stop=False,
        should_epoch_stop=False,
        should_save="false",
        should_evaluate=False,
        should_log=False,
    )
    with patch(
        "shaft.training.checkpointing.all_gather_objects",
        side_effect=lambda value: [value],
    ) as gather:
        with pytest.raises(TypeError, match="JSON boolean"):
            ShaftCheckpointCommitMixin._raise_synchronized_checkpoint_callback_error(
                "test callback",
                None,
                control,
            )

    gather.assert_called_once()


def test_backend_native_resolver_preserves_backend_owned_checkpoint_selection(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    for step in (1, 2):
        checkpoint = root / f"checkpoint-{step}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "trainer_state.json").write_text(
            json.dumps({"global_step": step}),
            encoding="utf-8",
        )

    latest = root / "checkpoint-2"
    assert resolve_resume_checkpoint(
        root,
        protocol=ShaftCheckpointProtocol.BACKEND_NATIVE,
    ) == str(latest)
    validate_resume_checkpoint(
        latest,
        finetune_mode="full",
        protocol=ShaftCheckpointProtocol.BACKEND_NATIVE,
    )
    with pytest.raises(ValueError, match="committed_manifest"):
        resolve_resume_checkpoint(
            root,
            protocol=ShaftCheckpointProtocol.COMMITTED_MANIFEST,
        )


def test_backend_native_resolver_rejects_root_final_state(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    (root / "trainer_state.json").write_text(
        json.dumps({"global_step": 2}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="checkpoint-<step>"):
        resolve_resume_checkpoint(
            root,
            protocol=ShaftCheckpointProtocol.BACKEND_NATIVE,
        )


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("not-json", "Expecting value"),
        (json.dumps({"global_step": -1}), ">= 0"),
        (json.dumps({"global_step": 3}), "differs"),
    ],
)
def test_backend_native_validation_rejects_invalid_trainer_step(
    tmp_path: Path,
    payload: str,
    message: str,
) -> None:
    checkpoint = tmp_path / "checkpoint-2"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text(payload, encoding="utf-8")

    with pytest.raises((json.JSONDecodeError, ValueError), match=message):
        validate_resume_checkpoint(
            checkpoint,
            finetune_mode="full",
            protocol=ShaftCheckpointProtocol.BACKEND_NATIVE,
        )


def test_backend_native_validation_delegates_conventional_layout_compatibility(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint-2"
    _write_full_checkpoint(checkpoint, global_step=2)

    validate_resume_checkpoint(
        checkpoint,
        finetune_mode="lora",
        protocol=ShaftCheckpointProtocol.BACKEND_NATIVE,
    )


def test_backend_native_protocol_does_not_install_commit_wrapper() -> None:
    class _Handler:
        def on_save(self, *args, **kwargs):
            _ = args, kwargs

    class _BackendTrainerBase:
        def __init__(self) -> None:
            self.is_deepspeed_enabled = True
            self.is_fsdp_enabled = False
            self.callback_handler = _Handler()

    class _BackendTrainer(ShaftCheckpointCommitMixin, _BackendTrainerBase):
        pass

    trainer = _BackendTrainer(shaft_checkpoint_protocol=ShaftCheckpointProtocol.BACKEND_NATIVE)

    assert trainer.callback_handler.on_save.__self__ is trainer.callback_handler
    assert trainer._shaft_checkpoint_protocol is ShaftCheckpointProtocol.BACKEND_NATIVE


def test_backend_native_save_runs_prepare_hook_before_backend_save(
    tmp_path: Path,
) -> None:
    events: list[tuple[str, Path | None]] = []

    class _Handler:
        def on_save(self, *args, **kwargs):
            _ = args, kwargs

    class _BackendTrainerBase:
        def __init__(self) -> None:
            self.is_deepspeed_enabled = False
            self.is_fsdp_enabled = True
            self.callback_handler = _Handler()
            self.state = SimpleNamespace(global_step=4)

        def _get_output_dir(self, trial=None) -> str:
            _ = trial
            return str(tmp_path)

        def _save_checkpoint(self, model, trial) -> None:
            _ = model, trial
            events.append(("backend_save", None))

    class _BackendTrainer(ShaftCheckpointCommitMixin, _BackendTrainerBase):
        def _prepare_shaft_checkpoint_save(self, checkpoint_path: Path) -> None:
            events.append(("prepare", checkpoint_path))

    trainer = _BackendTrainer(shaft_checkpoint_protocol=ShaftCheckpointProtocol.BACKEND_NATIVE)
    trainer._save_checkpoint(model=object(), trial=None)

    assert events == [
        ("prepare", tmp_path / "checkpoint-4"),
        ("backend_save", None),
    ]


def test_resolve_resume_checkpoint_skips_newer_uncommitted_fixed_checkpoint(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    committed = root / "checkpoint-1"
    torn = root / "checkpoint-2"
    _write_full_checkpoint(committed, global_step=1)
    _write_exact_resume_artifacts(committed, world_size=1)
    commit_training_checkpoint(committed, world_size=1, requires_grad_scaler=False)
    _write_full_checkpoint(torn, global_step=2)
    _write_exact_resume_artifacts(torn, world_size=1)

    assert resolve_resume_checkpoint(
        root,
        protocol=ShaftCheckpointProtocol.COMMITTED_MANIFEST,
    ) == str(committed)
    with pytest.raises(ValueError, match="not committed|torn"):
        resolve_resume_checkpoint(
            torn,
            protocol=ShaftCheckpointProtocol.COMMITTED_MANIFEST,
        )


def test_training_checkpoint_commit_rejects_mutated_artifact(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-3"
    _write_full_checkpoint(checkpoint, global_step=3)
    _write_exact_resume_artifacts(checkpoint, world_size=1)
    commit_training_checkpoint(checkpoint, world_size=1, requires_grad_scaler=False)

    assert training_checkpoint_is_committed(checkpoint)
    validate_training_checkpoint_commit(checkpoint)

    (checkpoint / "optimizer.pt").write_bytes(b"")

    assert not training_checkpoint_is_committed(checkpoint)
    with pytest.raises(ValueError, match="artifact"):
        validate_training_checkpoint_commit(checkpoint)


@pytest.mark.parametrize(
    "artifact_name",
    ["model.safetensors", "optimizer.pt", "rng_state.pth"],
)
def test_training_checkpoint_commit_rejects_same_size_required_artifact_mutation(
    tmp_path: Path,
    artifact_name: str,
) -> None:
    checkpoint = tmp_path / "checkpoint-3"
    _write_full_checkpoint(checkpoint, global_step=3)
    _write_exact_resume_artifacts(checkpoint, world_size=1)
    commit_training_checkpoint(checkpoint, world_size=1, requires_grad_scaler=False)

    artifact = checkpoint / artifact_name
    original = artifact.read_bytes()
    artifact.write_bytes(bytes(value ^ 1 for value in original))
    assert artifact.stat().st_size == len(original)

    assert not training_checkpoint_is_committed(checkpoint)
    with pytest.raises(ValueError, match="digest changed"):
        validate_training_checkpoint_commit(checkpoint)


@pytest.mark.parametrize(
    ("path", "value", "error_type", "message"),
    [
        (("global_step",), True, TypeError, "JSON integer"),
        (("world_size",), "1", TypeError, "JSON integer"),
        (
            ("artifacts", "trainer_state.json", "size"),
            1.0,
            TypeError,
            "JSON integer",
        ),
        (
            ("artifacts", "trainer_state.json", "sha256"),
            False,
            TypeError,
            "sha256.*JSON string",
        ),
        (("requires_grad_scaler",), "false", TypeError, "JSON boolean"),
        (("required_artifacts",), {}, TypeError, "JSON list"),
        (("extensions",), [], TypeError, "JSON mapping"),
    ],
)
def test_training_checkpoint_commit_manifest_rejects_json_type_coercion(
    tmp_path: Path,
    path: tuple[str, ...],
    value: object,
    error_type: type[Exception],
    message: str,
) -> None:
    checkpoint = tmp_path / "checkpoint-3"
    _write_full_checkpoint(checkpoint, global_step=3)
    _write_exact_resume_artifacts(checkpoint, world_size=1)
    commit_path = commit_training_checkpoint(
        checkpoint,
        world_size=1,
        requires_grad_scaler=False,
    )
    payload = json.loads(commit_path.read_text(encoding="utf-8"))
    target = payload
    for field_name in path[:-1]:
        target = target[field_name]
    target[path[-1]] = value
    commit_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(error_type, match=message):
        validate_training_checkpoint_commit(checkpoint)


def test_training_checkpoint_commit_manifest_rejects_unknown_schema_fields(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint-3"
    _write_full_checkpoint(checkpoint, global_step=3)
    _write_exact_resume_artifacts(checkpoint, world_size=1)
    commit_path = commit_training_checkpoint(
        checkpoint,
        world_size=1,
        requires_grad_scaler=False,
    )
    payload = json.loads(commit_path.read_text(encoding="utf-8"))
    payload["future_field"] = True
    commit_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="schema differs"):
        validate_training_checkpoint_commit(checkpoint)


@pytest.mark.parametrize(
    "replacement",
    [
        '"global_step": 3, "global_step": 4',
        '"global_step": NaN',
        '"global_step": Infinity',
    ],
)
def test_training_checkpoint_commit_manifest_rejects_ambiguous_json(
    tmp_path: Path,
    replacement: str,
) -> None:
    checkpoint = tmp_path / "checkpoint-3"
    _write_full_checkpoint(checkpoint, global_step=3)
    _write_exact_resume_artifacts(checkpoint, world_size=1)
    commit_path = commit_training_checkpoint(
        checkpoint,
        world_size=1,
        requires_grad_scaler=False,
    )
    document = commit_path.read_text(encoding="utf-8")
    document = document.replace('"global_step": 3', replacement, 1)
    commit_path.write_text(document, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON|non-finite JSON"):
        validate_training_checkpoint_commit(checkpoint)


def test_training_checkpoint_commit_rejects_duplicate_trainer_state_keys(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint-3"
    _write_full_checkpoint(checkpoint, global_step=3)
    _write_exact_resume_artifacts(checkpoint, world_size=1)
    (checkpoint / "trainer_state.json").write_text(
        '{"global_step": 3, "global_step": 3}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate JSON object key"):
        commit_training_checkpoint(
            checkpoint,
            world_size=1,
            requires_grad_scaler=False,
        )


def test_training_checkpoint_commit_rejects_legacy_v1_for_exact_resume(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint-3"
    _write_full_checkpoint(checkpoint, global_step=3)
    _write_exact_resume_artifacts(checkpoint, world_size=1)
    commit_path = commit_training_checkpoint(
        checkpoint,
        world_size=1,
        requires_grad_scaler=False,
    )
    payload = json.loads(commit_path.read_text(encoding="utf-8"))
    payload["version"] = "shaft-training-checkpoint-commit-v1"
    commit_path.write_text(json.dumps(payload), encoding="utf-8")

    assert not training_checkpoint_is_committed(checkpoint)
    with pytest.raises(ValueError, match="Unsupported.*version"):
        validate_training_checkpoint_commit(checkpoint)
    with pytest.raises(ValueError, match="Unsupported.*version"):
        resolve_resume_checkpoint(
            checkpoint,
            protocol=ShaftCheckpointProtocol.COMMITTED_MANIFEST,
        )

    # Commit manifests govern exact continuation only. The unchanged HF model
    # directory remains a valid source of initialization weights.
    ensure_hf_export_layout(checkpoint, finetune_mode="full")


def test_training_checkpoint_commit_requires_declared_grad_scaler_state(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint-3"
    _write_full_checkpoint(checkpoint, global_step=3)
    _write_exact_resume_artifacts(checkpoint, world_size=1)

    with pytest.raises(ValueError, match="scaler.pt"):
        commit_training_checkpoint(
            checkpoint,
            world_size=1,
            requires_grad_scaler=True,
        )


def test_training_checkpoint_commit_accepts_no_scaler_when_not_required(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint-3"
    _write_full_checkpoint(checkpoint, global_step=3)
    _write_exact_resume_artifacts(checkpoint, world_size=1)

    commit_path = commit_training_checkpoint(
        checkpoint,
        world_size=1,
        requires_grad_scaler=False,
    )

    manifest = validate_training_checkpoint_commit(checkpoint)
    assert manifest["requires_grad_scaler"] is False
    assert "scaler.pt" not in manifest["required_artifacts"]
    assert commit_path.is_file()


def test_training_checkpoint_commit_rejects_grad_scaler_flag_coercion(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint-3"
    _write_full_checkpoint(checkpoint, global_step=3)
    _write_exact_resume_artifacts(checkpoint, world_size=1)

    with pytest.raises(TypeError, match="requires_grad_scaler.*JSON boolean"):
        commit_training_checkpoint(
            checkpoint,
            world_size=1,
            requires_grad_scaler=1,  # type: ignore[arg-type]
        )


def test_training_checkpoint_commit_binds_present_grad_scaler_state(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint-3"
    _write_full_checkpoint(checkpoint, global_step=3)
    _write_exact_resume_artifacts(checkpoint, world_size=1)
    scaler = checkpoint / "scaler.pt"
    scaler.write_bytes(b"grad-scaler-state")
    commit_training_checkpoint(checkpoint, world_size=1, requires_grad_scaler=True)

    original = scaler.read_bytes()
    scaler.write_bytes(bytes(value ^ 1 for value in original))
    assert scaler.stat().st_size == len(original)

    assert not training_checkpoint_is_committed(checkpoint)
    with pytest.raises(ValueError, match="digest changed.*scaler.pt"):
        validate_training_checkpoint_commit(checkpoint)


@pytest.mark.parametrize(
    ("scaler", "requires_grad_scaler"),
    [(None, False), (object(), True)],
)
def test_checkpoint_commit_derives_grad_scaler_requirement_from_accelerator(
    tmp_path: Path,
    scaler: object | None,
    requires_grad_scaler: bool,
) -> None:
    class _CommitProbe(ShaftCheckpointCommitMixin):
        def __init__(self) -> None:
            pass

        def is_world_process_zero(self) -> bool:
            return True

    checkpoint = tmp_path / "checkpoint-3"
    trainer = _CommitProbe()
    trainer._shaft_pending_checkpoint_path = checkpoint
    trainer.state = SimpleNamespace(global_step=3, best_model_checkpoint=None)
    trainer.args = SimpleNamespace(
        world_size=1,
        should_save=False,
        save_total_limit=None,
    )
    trainer.accelerator = SimpleNamespace(scaler=scaler)

    with (
        patch.object(
            ShaftCheckpointCommitMixin,
            "_raise_synchronized_checkpoint_error",
        ),
        patch("shaft.training.checkpointing.commit_training_checkpoint") as commit,
    ):
        trainer._commit_and_rotate_shaft_checkpoint()

    commit.assert_called_once_with(
        checkpoint,
        world_size=1,
        requires_grad_scaler=requires_grad_scaler,
    )


def test_training_checkpoint_commit_fsyncs_all_files_and_commit_marker(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint-3"
    _write_full_checkpoint(checkpoint, global_step=3)
    _write_exact_resume_artifacts(checkpoint, world_size=1)

    with patch("shaft.training.checkpointing.os.fsync") as fsync:
        commit_training_checkpoint(checkpoint, world_size=1, requires_grad_scaler=False)

    # Every recorded artifact is fsynced, followed by the manifest temp file
    # and the checkpoint directory after atomic replace.
    assert fsync.call_count >= 7


def test_training_checkpoint_commit_durably_links_nested_artifacts_before_marker(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint-3"
    _write_full_checkpoint(checkpoint, global_step=3)
    _write_exact_resume_artifacts(checkpoint, world_size=1)
    nested_artifact = checkpoint / "plugins" / "state" / "snapshot.json"
    nested_artifact.parent.mkdir(parents=True)
    nested_artifact.write_text('{"cursor": 3}', encoding="utf-8")

    events: list[tuple[str, Path]] = []
    original_fsync_directory = checkpointing_module._fsync_directory
    original_replace = Path.replace

    def tracked_fsync_directory(path: Path) -> None:
        events.append(("directory-fsync", path))
        original_fsync_directory(path)

    def tracked_replace(source: Path, target: Path) -> Path:
        target_path = Path(target)
        if target_path == checkpoint / TRAINING_CHECKPOINT_COMMIT_FILENAME:
            events.append(("marker-replace", target_path))
        return original_replace(source, target)

    with (
        patch.object(
            checkpointing_module,
            "_fsync_directory",
            side_effect=tracked_fsync_directory,
        ),
        patch.object(Path, "replace", new=tracked_replace),
    ):
        commit_training_checkpoint(checkpoint, world_size=1, requires_grad_scaler=False)

    marker_event = (
        "marker-replace",
        checkpoint / TRAINING_CHECKPOINT_COMMIT_FILENAME,
    )
    marker_index = events.index(marker_event)
    assert events[:marker_index] == [
        ("directory-fsync", nested_artifact.parent),
        ("directory-fsync", nested_artifact.parent.parent),
        ("directory-fsync", checkpoint),
    ]
    assert events[marker_index + 1 :] == [("directory-fsync", checkpoint)]

    manifest = validate_training_checkpoint_commit(checkpoint)
    assert manifest["artifacts"]["plugins/state/snapshot.json"] == {
        "size": nested_artifact.stat().st_size,
        "sha256": hashlib.sha256(nested_artifact.read_bytes()).hexdigest(),
    }


@pytest.mark.parametrize("shard_path", [True, 7, ["model-1"], "../escape.bin"])
def test_training_checkpoint_commit_rejects_noncanonical_shard_paths(
    tmp_path: Path,
    shard_path: object,
) -> None:
    checkpoint = tmp_path / "checkpoint-3"
    _write_full_checkpoint(checkpoint, global_step=3)
    (checkpoint / "model.safetensors").unlink()
    (checkpoint / "model-00001-of-00001.safetensors").write_bytes(b"model")
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": shard_path}}),
        encoding="utf-8",
    )
    _write_exact_resume_artifacts(checkpoint, world_size=1)

    with pytest.raises((TypeError, ValueError), match="shard index"):
        commit_training_checkpoint(checkpoint, world_size=1, requires_grad_scaler=False)


def test_training_checkpoint_commit_rejects_duplicate_shard_index_keys(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint-3"
    _write_full_checkpoint(checkpoint, global_step=3)
    (checkpoint / "model.safetensors").unlink()
    (checkpoint / "model-00001-of-00001.safetensors").write_bytes(b"model")
    (checkpoint / "model.safetensors.index.json").write_text(
        '{"weight_map": {"weight": "model-00001-of-00001.safetensors", '
        '"weight": "model-00001-of-00001.safetensors"}}',
        encoding="utf-8",
    )
    _write_exact_resume_artifacts(checkpoint, world_size=1)

    with pytest.raises(ValueError, match="duplicate JSON object key"):
        commit_training_checkpoint(
            checkpoint,
            world_size=1,
            requires_grad_scaler=False,
        )


def test_training_checkpoint_revoke_fsyncs_directory(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-3"
    _write_full_checkpoint(checkpoint, global_step=3)
    _write_exact_resume_artifacts(checkpoint, world_size=1)
    commit_training_checkpoint(checkpoint, world_size=1, requires_grad_scaler=False)

    with patch("shaft.training.checkpointing._fsync_directory") as fsync_dir:
        revoke_training_checkpoint_commit(checkpoint)

    fsync_dir.assert_called_once_with(checkpoint)


def test_training_checkpoint_commit_allows_empty_optional_artifact(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint-3"
    _write_full_checkpoint(checkpoint, global_step=3)
    _write_exact_resume_artifacts(checkpoint, world_size=1)
    (checkpoint / "optional_plugin_marker").touch()

    commit_training_checkpoint(checkpoint, world_size=1, requires_grad_scaler=False)

    manifest = validate_training_checkpoint_commit(checkpoint)
    assert manifest["artifacts"]["optional_plugin_marker"] == {
        "size": 0,
        "sha256": hashlib.sha256(b"").hexdigest(),
    }


def test_training_checkpoint_commit_rejects_same_size_optional_artifact_mutation(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint-3"
    _write_full_checkpoint(checkpoint, global_step=3)
    _write_exact_resume_artifacts(checkpoint, world_size=1)
    optional = checkpoint / "optional_plugin_marker"
    optional.write_bytes(b"optional-state")

    with patch("shaft.training.checkpointing._fsync_file") as fsync_file:
        commit_training_checkpoint(
            checkpoint,
            world_size=1,
            requires_grad_scaler=False,
        )
    fsync_file.assert_any_call(optional)

    original = optional.read_bytes()
    optional.write_bytes(bytes(value ^ 1 for value in original))
    assert optional.stat().st_size == len(original)
    with pytest.raises(ValueError, match="digest changed.*optional_plugin_marker"):
        validate_training_checkpoint_commit(checkpoint)


def test_training_checkpoint_commit_rejects_optional_artifact_symlink(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint-3"
    _write_full_checkpoint(checkpoint, global_step=3)
    _write_exact_resume_artifacts(checkpoint, world_size=1)
    outside = tmp_path / "outside.txt"
    outside.write_text("external", encoding="utf-8")
    (checkpoint / "plugin_artifact").symlink_to(outside)

    with pytest.raises(ValueError, match="must not be symlinks"):
        commit_training_checkpoint(checkpoint, world_size=1, requires_grad_scaler=False)

    assert not (checkpoint / TRAINING_CHECKPOINT_COMMIT_FILENAME).exists()


@pytest.mark.parametrize(
    "artifact_name",
    [
        "model.safetensors",
        "trainer_state.json",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
    ],
)
def test_fixed_resume_skips_newer_checkpoint_with_torn_required_artifact(
    tmp_path: Path,
    artifact_name: str,
) -> None:
    root = tmp_path / "run"
    older = root / "checkpoint-1"
    newer = root / "checkpoint-2"
    for step, checkpoint in ((1, older), (2, newer)):
        _write_full_checkpoint(checkpoint, global_step=step)
        _write_exact_resume_artifacts(checkpoint, world_size=1)
        commit_training_checkpoint(checkpoint, world_size=1, requires_grad_scaler=False)
    (newer / artifact_name).write_bytes(b"")

    assert resolve_resume_checkpoint(
        root,
        protocol=ShaftCheckpointProtocol.COMMITTED_MANIFEST,
    ) == str(older)


def test_legacy_planning_completion_is_not_an_exact_resume_commit(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint-1"
    _write_full_checkpoint(checkpoint, global_step=1)
    _write_exact_resume_artifacts(checkpoint, world_size=1)
    (checkpoint / "shaft_batch_planning_complete.json").write_text(
        "{}",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not committed|torn"):
        resolve_resume_checkpoint(
            checkpoint,
            protocol=ShaftCheckpointProtocol.COMMITTED_MANIFEST,
        )


def test_commit_binds_stable_telemetry_written_by_on_save(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-1"
    _write_full_checkpoint(checkpoint, global_step=1)
    _write_exact_resume_artifacts(checkpoint, world_size=1)
    transaction_name = "shaft_training_efficiency_checkpoint_transaction.json"
    snapshot_name = "shaft_training_efficiency_rank0.json"
    (checkpoint / transaction_name).write_text(
        json.dumps({"state": "committed", "global_step": 1}),
        encoding="utf-8",
    )
    (checkpoint / snapshot_name).write_text(
        json.dumps({"global_step": 1, "frames": []}),
        encoding="utf-8",
    )

    commit_training_checkpoint(checkpoint, world_size=1, requires_grad_scaler=False)

    manifest = validate_training_checkpoint_commit(checkpoint)
    assert manifest["artifacts"][transaction_name]["size"] > 0
    assert manifest["artifacts"][snapshot_name]["size"] > 0
    assert resolve_resume_checkpoint(
        checkpoint,
        protocol=ShaftCheckpointProtocol.COMMITTED_MANIFEST,
    ) == str(checkpoint)


def test_commit_requires_every_rank_rng_artifact(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-1"
    _write_full_checkpoint(checkpoint, global_step=1)
    _write_exact_resume_artifacts(checkpoint, world_size=2)
    (checkpoint / "rng_state_1.pth").unlink()

    with pytest.raises(ValueError, match="rng_state_1.pth"):
        commit_training_checkpoint(checkpoint, world_size=2, requires_grad_scaler=False)


def test_adapter_commit_binds_adapter_weights(tmp_path: Path) -> None:
    checkpoint = tmp_path / "adapter"
    _write_adapter_checkpoint(checkpoint, global_step=1)
    _write_exact_resume_artifacts(checkpoint, world_size=1)
    commit_training_checkpoint(checkpoint, world_size=1, requires_grad_scaler=False)

    (checkpoint / "adapter_model.safetensors").write_bytes(b"")

    assert not training_checkpoint_is_committed(checkpoint)


def test_prune_root_output_layout_preserves_runtime_metadata(tmp_path: Path) -> None:
    root = tmp_path / "run"
    (root / "best").mkdir(parents=True)
    (root / "config.json").write_text("{}", encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"legacy")
    names = (
        "trainer_state.json",
        "shaft_finetune_summary.json",
        "shaft_optimizer_summary.json",
        BATCHING_RUN_METADATA_FILENAME,
        PROGRESS_SNAPSHOT_FILENAME,
        TRAINING_EFFICIENCY_FILENAME,
    )
    for name in names:
        (root / name).write_text("{}", encoding="utf-8")

    prune_root_output_layout(root)

    assert not (root / "config.json").exists()
    assert not (root / "model.safetensors").exists()
    assert all((root / name).is_file() for name in names)


def test_bounded_state_roundtrip_and_resume_validation(tmp_path: Path) -> None:
    spec = _spec()
    state = ShaftBatchPlanningState(
        contract_fingerprint=spec.fingerprint,
        global_microstep=6,
        next_draw_id=12,
        emitted_samples=12,
    )
    _write_bounded_trainer_state(tmp_path, spec=spec, state=state)

    assert (
        load_batch_planning_state(
            tmp_path,
            expected_spec=spec,
            expected_global_step=3,
            gradient_accumulation_steps=2,
            expected_resume_contract_fingerprint="resume-v1",
        )
        == state
    )


def test_completion_manifest_binds_the_committed_cursor(tmp_path: Path) -> None:
    spec = _spec(
        cardinality="token_budget",
        per_device_microbatch_size=2,
    )
    committed = ShaftBatchPlanningState(
        contract_fingerprint=spec.fingerprint,
        global_microstep=2,
        next_draw_id=4,
        emitted_samples=4,
    )
    _write_bounded_trainer_state(tmp_path, spec=spec, state=committed)
    replacement = ShaftBatchPlanningState(
        contract_fingerprint=spec.fingerprint,
        global_microstep=2,
        next_draw_id=8,
        emitted_samples=8,
    )
    trainer_state_path = tmp_path / "trainer_state.json"
    trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    trainer_state["stateful_callbacks"][BATCH_PLANNING_CALLBACK_NAME]["attributes"][
        "planning_state"
    ] = replacement.to_dict()
    trainer_state_path.write_text(json.dumps(trainer_state), encoding="utf-8")

    assert not checkpoint_has_batch_planning_state(tmp_path)


def test_bounded_resume_rejects_contract_or_optimizer_boundary_drift(
    tmp_path: Path,
) -> None:
    spec = _spec()
    state = ShaftBatchPlanningState(
        contract_fingerprint=spec.fingerprint,
        global_microstep=4,
        next_draw_id=8,
        emitted_samples=8,
    )
    _write_bounded_trainer_state(tmp_path, spec=spec, state=state)

    with pytest.raises(ValueError, match="changed fields.*buffer_size"):
        load_batch_planning_state(
            tmp_path,
            expected_spec=_spec(buffer_size=32),
            expected_global_step=2,
            gradient_accumulation_steps=2,
            expected_resume_contract_fingerprint="resume-v1",
        )
    with pytest.raises(ValueError, match="changed fields.*cost_fingerprint"):
        load_batch_planning_state(
            tmp_path,
            expected_spec=_spec(cost_fingerprint="cost-v2"),
            expected_global_step=2,
            gradient_accumulation_steps=2,
            expected_resume_contract_fingerprint="resume-v1",
        )
    with pytest.raises(ValueError, match="global_step differs"):
        load_batch_planning_state(
            tmp_path,
            expected_spec=spec,
            expected_global_step=3,
            gradient_accumulation_steps=2,
            expected_resume_contract_fingerprint="resume-v1",
        )

    with pytest.raises(ValueError, match="training contract changed"):
        load_batch_planning_state(
            tmp_path,
            expected_spec=spec,
            expected_global_step=2,
            gradient_accumulation_steps=2,
            expected_resume_contract_fingerprint="changed-resume",
        )


def test_bounded_callback_saves_only_committed_step_state(tmp_path: Path) -> None:
    spec = _spec()
    committed = ShaftBatchPlanningState(
        contract_fingerprint=spec.fingerprint,
        global_microstep=4,
        next_draw_id=8,
        emitted_samples=8,
    )

    class _Sampler:
        committed_state = committed

        def commit_global_microstep(self, global_microstep):
            assert global_microstep == 4
            return self.committed_state

    callback = ShaftBatchPlanningCallback(
        _Sampler(),
        spec,
        gradient_accumulation_steps=2,
        resume_contract_fingerprint="resume-v1",
    )
    control = object()
    state = SimpleNamespace(
        global_step=2,
        max_steps=4,
        epoch=2.0,
        num_train_epochs=4,
        is_world_process_zero=True,
    )
    callback.on_step_end(SimpleNamespace(), state, control)
    assert state.epoch == 0.5
    assert state.num_train_epochs == 1
    checkpoint = tmp_path / "checkpoint-2"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text(
        json.dumps(
            {
                "global_step": 2,
                "stateful_callbacks": {
                    BATCH_PLANNING_CALLBACK_NAME: callback.state(),
                    BATCHING_METADATA_CALLBACK_NAME: ShaftBatchingMetadataCallback(
                        _metadata_for_spec(
                            spec,
                            gradient_accumulation_steps=2,
                        )
                    ).state(),
                },
            }
        ),
        encoding="utf-8",
    )
    _write_exact_resume_artifacts(checkpoint, world_size=spec.data_world_size)
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"model")
    commit_training_checkpoint(
        checkpoint,
        world_size=spec.data_world_size,
        requires_grad_scaler=False,
    )

    assert (
        load_batch_planning_state(
            tmp_path / "checkpoint-2",
            expected_spec=spec,
            expected_global_step=2,
            gradient_accumulation_steps=2,
            expected_resume_contract_fingerprint="resume-v1",
        )
        == committed
    )
    assert checkpoint_has_batch_planning_state(checkpoint)


def test_batching_run_metadata_roundtrip(tmp_path: Path) -> None:
    metadata = ShaftBatchingRunMetadata(
        grouping="bounded_cost",
        cardinality="fixed",
        packing="none",
        layout="padded",
        per_device_train_batch_size=1,
        data_world_size=2,
        gradient_accumulation_steps=2,
        min_pixels=200704,
        max_pixels=2_000_000,
        source_weights=(("a", 2.0), ("b", 1.0)),
        media_snapshot_id="banana-media-v1",
        buffer_size=64,
        cost_cache_size=65536,
        max_tokens_per_microbatch=10000,
        resource_budgets=(("vision_patches", 16384),),
        planner_spec_fingerprint="planner-v1",
    )
    assert metadata.to_dict()["batch_contract"] == metadata.batch_contract.to_dict()
    write_batching_run_metadata(tmp_path, metadata)
    assert load_batching_run_metadata(tmp_path) == metadata


@pytest.mark.parametrize("field_name", ["version", "global_pack_count"])
def test_batching_run_metadata_rejects_missing_canonical_fields(
    field_name: str,
) -> None:
    metadata = ShaftBatchingRunMetadata(
        grouping="none",
        cardinality="fixed",
        packing="none",
        layout="padded",
        per_device_train_batch_size=1,
        data_world_size=1,
        gradient_accumulation_steps=1,
        min_pixels=None,
        max_pixels=None,
        source_weights=(("fixture", 1.0),),
    )
    payload = metadata.to_dict()
    payload.pop(field_name)

    with pytest.raises(ValueError, match="schema differs"):
        ShaftBatchingRunMetadata.from_dict(payload)


def test_batching_run_metadata_rejects_unknown_and_nested_unknown_fields() -> None:
    input_contract = _train_input_contract()
    metadata = ShaftBatchingRunMetadata(
        grouping="none",
        cardinality="fixed",
        packing="none",
        layout="padded",
        per_device_train_batch_size=1,
        data_world_size=1,
        gradient_accumulation_steps=1,
        min_pixels=None,
        max_pixels=None,
        source_weights=(("fixture", 1.0),),
        sample_execution_fingerprint=input_contract.data_execution_fingerprint,
        train_input_contract=input_contract,
    )

    top_level = metadata.to_dict()
    top_level["future_semantic"] = True
    with pytest.raises(ValueError, match="schema differs"):
        ShaftBatchingRunMetadata.from_dict(top_level)

    nested_batch = metadata.to_dict()
    nested_batch["batch_contract"]["future_semantic"] = True
    with pytest.raises(ValueError, match="schema differs"):
        ShaftBatchingRunMetadata.from_dict(nested_batch)

    nested_input = metadata.to_dict()
    nested_input["train_input_contract"]["future_semantic"] = True
    with pytest.raises(ValueError, match="schema differs"):
        ShaftBatchingRunMetadata.from_dict(nested_input)


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("per_device_train_batch_size", "1"),
        ("data_world_size", True),
        ("min_pixels", False),
        ("global_pack_count", "1"),
        ("local_pack_count_range", (1, 1)),
    ],
)
def test_batching_run_metadata_rejects_noncanonical_json_types(
    field_name: str,
    invalid_value: object,
) -> None:
    metadata = ShaftBatchingRunMetadata(
        grouping="none",
        cardinality="fixed",
        packing="none",
        layout="padded",
        per_device_train_batch_size=1,
        data_world_size=1,
        gradient_accumulation_steps=1,
        min_pixels=None,
        max_pixels=None,
        source_weights=(("fixture", 1.0),),
    )
    payload = metadata.to_dict()
    payload[field_name] = invalid_value

    with pytest.raises(TypeError, match="JSON"):
        ShaftBatchingRunMetadata.from_dict(payload)


@pytest.mark.parametrize("invalid_weight", ["1.0", True, float("inf"), float("nan")])
def test_batching_run_metadata_rejects_invalid_source_weights(
    invalid_weight: object,
) -> None:
    metadata = ShaftBatchingRunMetadata(
        grouping="none",
        cardinality="fixed",
        packing="none",
        layout="padded",
        per_device_train_batch_size=1,
        data_world_size=1,
        gradient_accumulation_steps=1,
        min_pixels=None,
        max_pixels=None,
        source_weights=(("fixture", 1.0),),
    )
    payload = metadata.to_dict()
    payload["source_weights"]["fixture"] = invalid_weight

    with pytest.raises((TypeError, ValueError), match="source_weights"):
        ShaftBatchingRunMetadata.from_dict(payload)


def test_batching_run_metadata_rejects_missing_canonical_contract(
    tmp_path: Path,
) -> None:
    metadata = ShaftBatchingRunMetadata(
        grouping="none",
        cardinality="fixed",
        packing="none",
        layout="padded",
        per_device_train_batch_size=1,
        data_world_size=1,
        gradient_accumulation_steps=1,
        min_pixels=None,
        max_pixels=None,
        source_weights=(("a", 1.0),),
    )
    payload = metadata.to_dict()
    payload.pop("batch_contract")
    (tmp_path / BATCHING_RUN_METADATA_FILENAME).write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="schema differs.*batch_contract"):
        load_batching_run_metadata(tmp_path)


def test_batch_contract_canonical_roundtrip_and_fingerprint() -> None:
    contract = ShaftBatchContract(
        grouping="none",
        cardinality="fixed",
        packing="none",
        layout="padded",
        per_device_microbatch_size=2,
        data_world_size=4,
        gradient_accumulation_steps=3,
    )

    assert ShaftBatchContract.from_dict(contract.to_dict()) == contract
    assert ShaftBatchContract.from_dict(contract.to_dict()).fingerprint == contract.fingerprint
    changed = ShaftBatchContract(
        grouping="none",
        cardinality="fixed",
        packing="none",
        layout="padded",
        per_device_microbatch_size=1,
        data_world_size=4,
        gradient_accumulation_steps=3,
    )
    assert changed.fingerprint != contract.fingerprint


@pytest.mark.parametrize("field_name", ["version", "layout"])
def test_batch_contract_rejects_missing_canonical_fields(field_name: str) -> None:
    payload = _fixed_batch_contract().to_dict()
    payload.pop(field_name)

    with pytest.raises(ValueError, match="schema differs"):
        ShaftBatchContract.from_dict(payload)


def test_batch_contract_rejects_unknown_field() -> None:
    payload = _fixed_batch_contract().to_dict()
    payload["future_semantic"] = True

    with pytest.raises(ValueError, match="schema differs"):
        ShaftBatchContract.from_dict(payload)


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("grouping", 1),
        ("per_device_microbatch_size", "1"),
        ("data_world_size", True),
        ("buffer_size", False),
    ],
)
def test_batch_contract_rejects_noncanonical_json_types(
    field_name: str,
    invalid_value: object,
) -> None:
    payload = _fixed_batch_contract().to_dict()
    payload[field_name] = invalid_value

    with pytest.raises(TypeError, match="JSON"):
        ShaftBatchContract.from_dict(payload)


def test_train_input_contract_binds_components_and_input_options() -> None:
    def build(
        *,
        tokenizer_payload: str = "tokenizer-v1",
        processor_patch_size: int = 16,
        template: object | None = None,
        input_builder: type[object] = _ContractInputBuilder,
        train_dataset_type: type[object] = _ContractDataset,
        max_length: int = 4096,
    ) -> ShaftTrainInputContract:
        return build_train_input_contract(
            algorithm="sft",
            data_execution_fingerprint="data-v1",
            data_execution_contract_complete=True,
            train_dataset_type=train_dataset_type,
            model_plan_fingerprint="model-plan-v1",
            model_adapter=SimpleNamespace(model_type="fixture"),
            processor=_ContractProcessor(processor_patch_size),
            tokenizer=_ContractTokenizer(tokenizer_payload),
            template=template or _ContractTemplate(),
            input_builder=input_builder,
            input_options={
                "min_pixels": 200_704,
                "max_pixels": 1_048_576,
                "max_length": max_length,
                "add_eos_token": True,
            },
        )

    base = build()
    assert base.exact_resume_safe is True
    assert ShaftTrainInputContract.from_dict(base.to_dict()) == base
    invalid_payload = base.to_dict()
    invalid_payload["data_execution_contract_complete"] = "false"
    with pytest.raises(TypeError, match="JSON boolean"):
        ShaftTrainInputContract.from_dict(invalid_payload)
    assert (
        len(
            {
                base.fingerprint,
                build(tokenizer_payload="tokenizer-v2").fingerprint,
                build(processor_patch_size=32).fingerprint,
                build(template=_ChangedContractTemplate()).fingerprint,
                build(input_builder=_ChangedContractInputBuilder).fingerprint,
                build(train_dataset_type=_ChangedContractDataset).fingerprint,
                build(max_length=2048).fingerprint,
            }
        )
        == 7
    )


def test_train_input_contract_ignores_content_artifact_relocation(
    tmp_path: Path,
) -> None:
    def build(
        root: Path,
        *,
        tokenizer_payload: str = "same-backend-artifact",
        normalizer_mode: str = "default",
        revision: str = "main",
    ) -> ShaftTrainInputContract:
        return build_train_input_contract(
            algorithm="sft",
            data_execution_fingerprint="data-v1",
            data_execution_contract_complete=True,
            train_dataset_type=_ContractDataset,
            model_plan_fingerprint="same-content-addressed-model-plan",
            model_adapter=SimpleNamespace(
                model_type="fixture",
                family="fixture",
                model_name_or_path=str(root),
                template_type="fixture",
                group_name="fixture",
            ),
            processor=_ContractProcessor(16, artifact_root=root),
            tokenizer=_ContractTokenizer(
                tokenizer_payload,
                artifact_root=root,
                normalizer_mode=normalizer_mode,
                revision=revision,
            ),
            template=_ContractTemplate(),
            input_builder=_ContractInputBuilder,
            input_options={"max_length": 4096},
        )

    first = build(tmp_path / "first" / "model")
    relocated = build(tmp_path / "second" / "model")

    assert first.model_adapter_signature == relocated.model_adapter_signature
    assert first.processor_signature == relocated.processor_signature
    assert first.tokenizer_signature == relocated.tokenizer_signature
    assert first.fingerprint == relocated.fingerprint

    assert (
        build(
            tmp_path / "second" / "model",
            tokenizer_payload="changed-backend-artifact",
        ).tokenizer_signature
        != first.tokenizer_signature
    )
    assert (
        build(
            tmp_path / "second" / "model",
            normalizer_mode="changed",
        ).tokenizer_signature
        != first.tokenizer_signature
    )
    assert (
        build(
            tmp_path / "second" / "model",
            revision="other-immutable-revision",
        ).tokenizer_signature
        != first.tokenizer_signature
    )


@pytest.mark.parametrize("field_name", ["version", "tokenizer_signature"])
def test_train_input_contract_rejects_missing_canonical_fields(
    field_name: str,
) -> None:
    payload = _train_input_contract().to_dict()
    payload.pop(field_name)

    with pytest.raises(ValueError, match="schema differs"):
        ShaftTrainInputContract.from_dict(payload)


def test_train_input_contract_rejects_unknown_field() -> None:
    payload = _train_input_contract().to_dict()
    payload["future_semantic"] = True

    with pytest.raises(ValueError, match="schema differs"):
        ShaftTrainInputContract.from_dict(payload)


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("algorithm", 7),
        ("data_execution_fingerprint", True),
        ("incomplete_reasons", "oops"),
        ("input_options", []),
    ],
)
def test_train_input_contract_rejects_noncanonical_json_types(
    field_name: str,
    invalid_value: object,
) -> None:
    payload = _train_input_contract().to_dict()
    payload[field_name] = invalid_value

    with pytest.raises(TypeError, match="JSON"):
        ShaftTrainInputContract.from_dict(payload)


def test_train_input_contract_rejects_non_string_reason_and_nonfinite_option() -> None:
    invalid_reason = _train_input_contract().to_dict()
    invalid_reason["incomplete_reasons"] = [1]
    with pytest.raises(TypeError, match="incomplete_reasons"):
        ShaftTrainInputContract.from_dict(invalid_reason)

    nonfinite_option = _train_input_contract().to_dict()
    nonfinite_option["input_options"]["loss_weight"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        ShaftTrainInputContract.from_dict(nonfinite_option)


def test_train_input_contract_binds_effective_dataset_runtime_method() -> None:
    def build() -> ShaftTrainInputContract:
        return build_train_input_contract(
            algorithm="sft",
            data_execution_fingerprint="data-v1",
            data_execution_contract_complete=True,
            train_dataset_type=_ContractDataset,
            model_plan_fingerprint="model-v1",
            model_adapter=SimpleNamespace(model_type="fixture"),
            processor=_ContractProcessor(16),
            tokenizer=_ContractTokenizer("tokenizer-v1"),
            template=_ContractTemplate(),
            input_builder=_ContractInputBuilder,
            input_options={"max_length": 4096},
        )

    base = build()
    with patch.object(
        _ContractDataset,
        "__getitem__",
        _ChangedContractDataset.__getitem__,
    ):
        changed = build()

    assert base.train_dataset_signature != changed.train_dataset_signature
    assert base.fingerprint != changed.fingerprint


def test_train_input_contract_binds_tokenizer_runtime_package_version() -> None:
    class _VersionedTokenizer(_ContractTokenizer):
        pass

    _VersionedTokenizer.__module__ = "fixture_tokenizer_runtime"

    def build(version: str) -> ShaftTrainInputContract:
        with (
            patch(
                "shaft.training.input_contract._package_distributions",
                return_value={"fixture_tokenizer_runtime": ["fixture-tokenizer"]},
            ),
            patch(
                "shaft.training.input_contract.importlib_metadata.version",
                return_value=version,
            ),
        ):
            _package_version.cache_clear()
            return build_train_input_contract(
                algorithm="sft",
                data_execution_fingerprint="data-v1",
                data_execution_contract_complete=True,
                train_dataset_type=_ContractDataset,
                model_plan_fingerprint="model-v1",
                model_adapter=SimpleNamespace(model_type="fixture"),
                processor=_ContractProcessor(16),
                tokenizer=_VersionedTokenizer("same-backend-artifact"),
                template=_ContractTemplate(),
                input_builder=_ContractInputBuilder,
                input_options={"max_length": 4096},
            )

    try:
        first = build("1.0")
        second = build("2.0")
    finally:
        _package_version.cache_clear()
        _package_distributions.cache_clear()

    assert first.tokenizer_signature != second.tokenizer_signature


def test_train_input_contract_binds_tokenizer_base_vocabulary() -> None:
    def build(vocab: dict[str, int]) -> ShaftTrainInputContract:
        return build_train_input_contract(
            algorithm="sft",
            data_execution_fingerprint="data-v1",
            data_execution_contract_complete=True,
            train_dataset_type=_ContractDataset,
            model_plan_fingerprint="model-v1",
            model_adapter=SimpleNamespace(model_type="fixture"),
            processor=_ContractProcessor(16),
            tokenizer=_ContractTokenizer("same-backend", vocab=vocab),
            template=_ContractTemplate(),
            input_builder=_ContractInputBuilder,
            input_options={"max_length": 4096},
        )

    assert build({"a": 1}).tokenizer_signature != build({"b": 1}).tokenizer_signature


class _ContractMode(Enum):
    FIRST = "first"
    SECOND = "second"


def test_train_input_canonicalization_preserves_semantic_scalar_values() -> None:
    assert _canonical_value(_ContractMode.FIRST) != _canonical_value(_ContractMode.SECOND)
    assert _canonical_value(torch.float16) != _canonical_value(torch.bfloat16)
    assert _canonical_value(np.dtype("int32")) != _canonical_value(np.dtype("int64"))
    assert _canonical_value(np.int64(7)) == 7


def test_train_input_canonicalization_binds_added_token_semantics() -> None:
    base = AddedToken(
        "<fixture>",
        single_word=True,
        lstrip=False,
        rstrip=True,
        normalized=False,
        special=True,
    )
    repeated = AddedToken(
        "<fixture>",
        single_word=True,
        lstrip=False,
        rstrip=True,
        normalized=False,
        special=True,
    )
    changed = AddedToken(
        "<fixture>",
        single_word=True,
        lstrip=True,
        rstrip=True,
        normalized=False,
        special=True,
    )
    unresolved_types: set[str] = set()

    canonical = _canonical_value(base, unresolved_types=unresolved_types)

    assert canonical == _canonical_value(repeated, unresolved_types=set())
    assert canonical != _canonical_value(changed, unresolved_types=set())
    assert stable_artifact_value(base) == stable_artifact_value(repeated)
    assert stable_artifact_value(base) != stable_artifact_value(changed)
    assert unresolved_types == set()


def test_unknown_input_option_marks_contract_incomplete() -> None:
    contract = build_train_input_contract(
        algorithm="sft",
        data_execution_fingerprint="data-v1",
        data_execution_contract_complete=True,
        train_dataset_type=_ContractDataset,
        model_plan_fingerprint="model-v1",
        model_adapter=SimpleNamespace(model_type="fixture"),
        processor=_ContractProcessor(16),
        tokenizer=_ContractTokenizer("tokenizer-v1"),
        template=_ContractTemplate(),
        input_builder=_ContractInputBuilder,
        input_options={"opaque": object()},
    )

    assert contract.exact_resume_safe is False
    assert any(
        reason.startswith("unresolved_input_option_type:builtins.object")
        for reason in contract.incomplete_reasons
    )
    with pytest.raises(ValueError, match="identity is incomplete"):
        validate_train_input_checkpointability(contract, save_strategy="steps")


def test_train_input_contract_rejects_string_data_completeness() -> None:
    with pytest.raises(TypeError, match="must be a boolean"):
        build_train_input_contract(
            algorithm="sft",
            data_execution_fingerprint="data-v1",
            data_execution_contract_complete="false",  # type: ignore[arg-type]
            train_dataset_type=_ContractDataset,
            model_plan_fingerprint="model-v1",
            model_adapter=SimpleNamespace(model_type="fixture"),
            processor=_ContractProcessor(16),
            tokenizer=_ContractTokenizer("tokenizer-v1"),
            template=_ContractTemplate(),
            input_builder=_ContractInputBuilder,
            input_options={},
        )


def test_train_input_contract_rejects_contradictory_data_identity() -> None:
    with pytest.raises(
        ValueError,
        match="complete data execution contract cannot declare incomplete reasons",
    ):
        build_train_input_contract(
            algorithm="sft",
            data_execution_fingerprint="data-v1",
            data_execution_contract_complete=True,
            data_execution_incomplete_reasons=("unversioned_online_transform",),
            train_dataset_type=_ContractDataset,
            model_plan_fingerprint="model-v1",
            model_adapter=SimpleNamespace(model_type="fixture"),
            processor=_ContractProcessor(patch_size=16),
            tokenizer=_ContractTokenizer("tokenizer-v1"),
            template=_ContractTemplate(),
            input_builder=_ContractInputBuilder,
            input_options={"max_length": 4096},
        )


def test_exact_resume_rejects_training_input_contract_drift(
    tmp_path: Path,
) -> None:
    input_contract = _train_input_contract()
    with pytest.raises(ValueError, match="sample execution fingerprint differs"):
        ShaftBatchingRunMetadata(
            grouping="none",
            cardinality="fixed",
            packing="none",
            layout="padded",
            per_device_train_batch_size=1,
            data_world_size=1,
            gradient_accumulation_steps=1,
            min_pixels=None,
            max_pixels=None,
            source_weights=(("a", 1.0),),
            sample_execution_fingerprint="different-data",
            train_input_contract=input_contract,
        )
    metadata = ShaftBatchingRunMetadata(
        grouping="none",
        cardinality="fixed",
        packing="none",
        layout="padded",
        per_device_train_batch_size=1,
        data_world_size=1,
        gradient_accumulation_steps=1,
        min_pixels=200_704,
        max_pixels=1_048_576,
        source_weights=(("a", 1.0),),
        sample_execution_fingerprint="data-v1",
        train_input_contract=input_contract,
    )
    callback = ShaftBatchingMetadataCallback(metadata)
    (tmp_path / "trainer_state.json").write_text(
        json.dumps(
            {
                "stateful_callbacks": {
                    BATCHING_METADATA_CALLBACK_NAME: callback.state(),
                }
            }
        ),
        encoding="utf-8",
    )

    assert (
        validate_batching_resume_contract(
            tmp_path,
            expected_contract=metadata.batch_contract,
            expected_sample_execution_fingerprint="data-v1",
            expected_train_input_contract=input_contract,
        )
        == metadata
    )

    changed_options = tuple(
        (name, 2048 if name == "max_length" else value)
        for name, value in input_contract.input_options
    )
    changed = _train_input_contract(input_options=changed_options)
    with pytest.raises(ValueError, match="Training input contract changed"):
        validate_batching_resume_contract(
            tmp_path,
            expected_contract=metadata.batch_contract,
            expected_sample_execution_fingerprint="data-v1",
            expected_train_input_contract=changed,
        )


def test_exact_resume_rejects_unversioned_transform_contract(
    tmp_path: Path,
) -> None:
    incomplete = _train_input_contract(
        data_execution_contract_complete=False,
        incomplete_reasons=("unversioned_online_transform",),
    )
    metadata = ShaftBatchingRunMetadata(
        grouping="none",
        cardinality="fixed",
        packing="none",
        layout="padded",
        per_device_train_batch_size=1,
        data_world_size=1,
        gradient_accumulation_steps=1,
        min_pixels=None,
        max_pixels=None,
        source_weights=(("a", 1.0),),
        sample_execution_fingerprint="data-v1",
        train_input_contract=incomplete,
    )
    callback = ShaftBatchingMetadataCallback(metadata)
    (tmp_path / "trainer_state.json").write_text(
        json.dumps(
            {
                "stateful_callbacks": {
                    BATCHING_METADATA_CALLBACK_NAME: callback.state(),
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="training input identity"):
        validate_batching_resume_contract(
            tmp_path,
            expected_contract=metadata.batch_contract,
            expected_train_input_contract=incomplete,
        )


def test_fresh_run_rejects_incomplete_input_contract_when_checkpointing() -> None:
    incomplete = _train_input_contract(
        data_execution_contract_complete=False,
        incomplete_reasons=("unversioned_online_transform",),
    )

    validate_train_input_checkpointability(incomplete, save_strategy="no")
    with pytest.raises(ValueError, match="Checkpointing requires"):
        validate_train_input_checkpointability(incomplete, save_strategy="steps")
    with pytest.raises(ValueError, match="Checkpointing requires"):
        validate_train_input_checkpointability(incomplete, save_strategy="epoch")


def test_incomplete_data_identity_fails_before_full_input_contract_build() -> None:
    validate_train_data_identity_checkpointability(
        data_execution_contract_complete=False,
        incomplete_reasons=("missing_media_snapshot_id",),
        train_dataset_type=_ContractDataset,
        save_strategy="no",
        resume_requested=False,
    )
    with pytest.raises(ValueError, match="before model loading"):
        validate_train_data_identity_checkpointability(
            data_execution_contract_complete=False,
            incomplete_reasons=("missing_media_snapshot_id",),
            train_dataset_type=_ContractDataset,
            save_strategy="steps",
            resume_requested=False,
        )
    with pytest.raises(ValueError, match="Exact resume requires"):
        validate_train_data_identity_checkpointability(
            data_execution_contract_complete=False,
            incomplete_reasons=("unversioned_online_transform",),
            train_dataset_type=_ContractDataset,
            save_strategy="no",
            resume_requested=True,
        )


def test_resume_preflight_rejects_checkpoint_without_input_contract(
    tmp_path: Path,
) -> None:
    metadata = ShaftBatchingRunMetadata(
        grouping="none",
        cardinality="fixed",
        packing="none",
        layout="padded",
        per_device_train_batch_size=1,
        data_world_size=1,
        gradient_accumulation_steps=1,
        min_pixels=None,
        max_pixels=None,
        source_weights=(("a", 1.0),),
    )
    callback = ShaftBatchingMetadataCallback(metadata)
    (tmp_path / "trainer_state.json").write_text(
        json.dumps(
            {
                "stateful_callbacks": {
                    BATCHING_METADATA_CALLBACK_NAME: callback.state(),
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="predates the complete training input"):
        validate_batching_resume_contract(
            tmp_path,
            expected_contract=metadata.batch_contract,
            require_train_input_contract_payload=True,
        )


def test_length_greedy_varlen_batch_contract_roundtrip() -> None:
    contract = ShaftBatchContract(
        grouping="length",
        cardinality="fixed",
        packing="greedy",
        layout="varlen",
        per_device_microbatch_size=2,
        data_world_size=8,
        gradient_accumulation_steps=4,
        buffer_size=64,
        max_sequence_length=10_000,
        resource_budgets=(("vision_patches", 16_384),),
    )

    restored = ShaftBatchContract.from_dict(contract.to_dict())

    assert restored == contract
    assert restored.is_planned is True
    assert restored.is_bounded is False
    assert restored.local_token_capacity == 20_000


def test_length_batching_metadata_uses_unified_planner_fingerprint() -> None:
    spec = ShaftBatchPlanningSpec(
        grouping="length",
        cardinality="fixed",
        packing="greedy",
        layout="varlen",
        max_sequence_length=128,
        data_world_size=2,
        buffer_size=16,
        per_device_microbatch_size=2,
        max_tokens_per_microbatch=256,
        resource_budgets=(("vision_patches", 4096),),
        seed=42,
        sample_schedule_fingerprint="schedule-v1",
        cost_fingerprint="cost-v1",
    )
    metadata = ShaftBatchingRunMetadata(
        grouping="length",
        cardinality="fixed",
        packing="greedy",
        layout="varlen",
        per_device_train_batch_size=2,
        data_world_size=2,
        gradient_accumulation_steps=4,
        min_pixels=None,
        max_pixels=None,
        source_weights=(("fixture", 1.0),),
        media_snapshot_id="fixture-media-v1",
        buffer_size=16,
        cost_cache_size=32,
        max_sequence_length=128,
        resource_budgets=(("vision_patches", 4096),),
        planner_spec_fingerprint=spec.fingerprint,
    )

    assert metadata.batch_contract.grouping == "length"
    assert metadata.batch_contract.local_token_capacity == 256


def test_cost_cache_size_is_not_part_of_exact_batch_contract() -> None:
    spec = _spec()
    small_cache = _metadata_for_spec(
        spec,
        gradient_accumulation_steps=2,
        cost_cache_size=0,
    )
    large_cache = _metadata_for_spec(
        spec,
        gradient_accumulation_steps=2,
        cost_cache_size=65536,
    )

    assert small_cache.batch_contract == large_cache.batch_contract
    assert small_cache.batch_contract_fingerprint == large_cache.batch_contract_fingerprint


def test_token_budget_metadata_reports_pack_count_ranges(tmp_path: Path) -> None:
    metadata = ShaftBatchingRunMetadata(
        grouping="bounded_cost",
        cardinality="token_budget",
        packing="none",
        layout="padded",
        per_device_train_batch_size=2,
        data_world_size=8,
        gradient_accumulation_steps=4,
        min_pixels=200704,
        max_pixels=4_000_000,
        source_weights=(("a", 1.0),),
        media_snapshot_id="media-v1",
        buffer_size=64,
        cost_cache_size=65536,
        max_tokens_per_microbatch=10000,
        resource_budgets=(("vision_patches", 16384),),
        planner_spec_fingerprint="planner-v3",
    )

    payload = metadata.to_dict()

    assert payload["local_pack_count_range"] == [1, 2]
    assert payload["global_pack_count_range"] == [8, 16]
    assert payload["optimizer_pack_count_range"] == [32, 64]
    assert payload["global_pack_count"] is None
    assert payload["optimizer_pack_count"] is None
    write_batching_run_metadata(tmp_path, metadata)
    assert load_batching_run_metadata(tmp_path) == metadata


@pytest.mark.parametrize(
    "field_name",
    ["global_pack_count", "optimizer_pack_count"],
)
def test_batching_run_metadata_rejects_tampered_derived_counts(
    tmp_path: Path,
    field_name: str,
) -> None:
    metadata = ShaftBatchingRunMetadata(
        grouping="none",
        cardinality="fixed",
        packing="none",
        layout="padded",
        per_device_train_batch_size=2,
        data_world_size=4,
        gradient_accumulation_steps=3,
        min_pixels=None,
        max_pixels=None,
        source_weights=(("a", 1.0),),
    )
    payload = metadata.to_dict()
    payload[field_name] += 1
    (tmp_path / BATCHING_RUN_METADATA_FILENAME).write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=field_name):
        load_batching_run_metadata(tmp_path)


def test_batching_run_metadata_rejects_tampered_token_budget_range(
    tmp_path: Path,
) -> None:
    metadata = ShaftBatchingRunMetadata(
        grouping="bounded_cost",
        cardinality="token_budget",
        packing="none",
        layout="padded",
        per_device_train_batch_size=2,
        data_world_size=4,
        gradient_accumulation_steps=3,
        min_pixels=None,
        max_pixels=None,
        source_weights=(("a", 1.0),),
        media_snapshot_id="media-v1",
        buffer_size=8,
        cost_cache_size=32,
        max_tokens_per_microbatch=512,
        planner_spec_fingerprint="planner-v3",
    )
    payload = metadata.to_dict()
    payload["optimizer_pack_count_range"] = [12, 25]
    (tmp_path / BATCHING_RUN_METADATA_FILENAME).write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="optimizer_pack_count_range"):
        load_batching_run_metadata(tmp_path)


def test_batching_run_metadata_rejects_missing_batch_contract_fingerprint(
    tmp_path: Path,
) -> None:
    metadata = ShaftBatchingRunMetadata(
        grouping="none",
        cardinality="fixed",
        packing="none",
        layout="padded",
        per_device_train_batch_size=1,
        data_world_size=1,
        gradient_accumulation_steps=1,
        min_pixels=None,
        max_pixels=None,
        source_weights=(("a", 1.0),),
    )
    payload = metadata.to_dict()
    payload.pop("batch_contract_fingerprint")
    (tmp_path / BATCHING_RUN_METADATA_FILENAME).write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="schema differs.*batch_contract_fingerprint"):
        load_batching_run_metadata(tmp_path)


def test_batching_run_metadata_reuses_executable_batch_contract_validation() -> None:
    with pytest.raises(
        ValueError,
        match="Non-bounded ShaftBatchContract cannot carry bounded planner fields",
    ):
        ShaftBatchingRunMetadata(
            grouping="none",
            cardinality="fixed",
            packing="none",
            layout="padded",
            per_device_train_batch_size=1,
            data_world_size=1,
            gradient_accumulation_steps=1,
            min_pixels=None,
            max_pixels=None,
            source_weights=(("a", 1.0),),
            buffer_size=64,
        )


def test_batching_metadata_callback_publishes_wandb_config(monkeypatch) -> None:
    updates = []
    run = SimpleNamespace(
        config=SimpleNamespace(
            update=lambda payload, allow_val_change: updates.append((payload, allow_val_change))
        )
    )
    monkeypatch.setitem(__import__("sys").modules, "wandb", SimpleNamespace(run=run))
    metadata = ShaftBatchingRunMetadata(
        grouping="none",
        cardinality="fixed",
        packing="none",
        layout="padded",
        per_device_train_batch_size=1,
        data_world_size=1,
        gradient_accumulation_steps=1,
        min_pixels=None,
        max_pixels=None,
        source_weights=(("a", 1.0),),
    )
    callback = ShaftBatchingMetadataCallback(metadata)
    callback.on_train_begin(
        SimpleNamespace(report_to=["wandb"]),
        SimpleNamespace(is_world_process_zero=True),
        object(),
    )
    assert updates == [({"shaft_batching": metadata.to_dict()}, True)]
    assert callback.state() == {
        "args": {"metadata": metadata.to_dict()},
        "attributes": {},
    }


def test_batching_metadata_callback_exportable_state_roundtrip() -> None:
    metadata = ShaftBatchingRunMetadata(
        grouping="none",
        cardinality="fixed",
        packing="none",
        layout="padded",
        per_device_train_batch_size=2,
        data_world_size=4,
        gradient_accumulation_steps=3,
        min_pixels=None,
        max_pixels=None,
        source_weights=(("a", 1.0),),
    )
    original = ShaftBatchingMetadataCallback(metadata)

    restored = ShaftBatchingMetadataCallback.from_state(original.state())

    assert restored.metadata == metadata
    assert restored.state() == original.state()


def test_checkpoint_batch_contract_roundtrip_and_resume_drift_rejection(
    tmp_path: Path,
) -> None:
    metadata = ShaftBatchingRunMetadata(
        grouping="none",
        cardinality="fixed",
        packing="none",
        layout="padded",
        per_device_train_batch_size=2,
        data_world_size=1,
        gradient_accumulation_steps=4,
        min_pixels=None,
        max_pixels=None,
        source_weights=(("a", 1.0),),
        sample_execution_fingerprint="sample-v1",
    )
    callback = ShaftBatchingMetadataCallback(metadata)
    (tmp_path / "trainer_state.json").write_text(
        json.dumps(
            {
                "stateful_callbacks": {
                    BATCHING_METADATA_CALLBACK_NAME: callback.state(),
                }
            }
        ),
        encoding="utf-8",
    )

    assert load_checkpoint_batching_metadata(tmp_path) == metadata
    assert (
        validate_batching_resume_contract(
            tmp_path,
            expected_contract=metadata.batch_contract,
        )
        == metadata
    )
    assert (
        validate_batching_resume_contract(
            tmp_path,
            expected_contract=metadata.batch_contract,
            expected_sample_execution_fingerprint="sample-v1",
        )
        == metadata
    )
    with pytest.raises(ValueError, match="sample execution changed"):
        validate_batching_resume_contract(
            tmp_path,
            expected_contract=metadata.batch_contract,
            expected_sample_execution_fingerprint="sample-v2",
        )
    changed_contract = ShaftBatchContract(
        grouping="none",
        cardinality="fixed",
        packing="none",
        layout="padded",
        per_device_microbatch_size=1,
        data_world_size=1,
        gradient_accumulation_steps=4,
    )
    with pytest.raises(ValueError, match="Training batch contract changed"):
        validate_batching_resume_contract(
            tmp_path,
            expected_contract=changed_contract,
        )


def test_bounded_completion_requires_canonical_metadata_callback(
    tmp_path: Path,
) -> None:
    spec = _spec()
    state = ShaftBatchPlanningState(
        contract_fingerprint=spec.fingerprint,
        global_microstep=2,
        next_draw_id=4,
        emitted_samples=4,
    )
    _write_bounded_trainer_state(tmp_path, spec=spec, state=state)
    trainer_state_path = tmp_path / "trainer_state.json"
    payload = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    payload["stateful_callbacks"].pop(BATCHING_METADATA_CALLBACK_NAME)
    trainer_state_path.write_text(json.dumps(payload), encoding="utf-8")
    (tmp_path / TRAINING_CHECKPOINT_COMMIT_FILENAME).unlink()

    with pytest.raises(ValueError, match=BATCHING_METADATA_CALLBACK_NAME):
        commit_training_checkpoint(
            tmp_path,
            world_size=spec.data_world_size,
            requires_grad_scaler=False,
        )
    assert not checkpoint_has_batch_planning_state(tmp_path)


def test_bounded_completion_rejects_planner_metadata_drift(tmp_path: Path) -> None:
    spec = _spec()
    state = ShaftBatchPlanningState(
        contract_fingerprint=spec.fingerprint,
        global_microstep=2,
        next_draw_id=4,
        emitted_samples=4,
    )
    _write_bounded_trainer_state(tmp_path, spec=spec, state=state)
    trainer_state_path = tmp_path / "trainer_state.json"
    payload = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    metadata = payload["stateful_callbacks"][BATCHING_METADATA_CALLBACK_NAME]["args"]["metadata"]
    metadata["planner_spec_fingerprint"] = "different-planner"
    trainer_state_path.write_text(json.dumps(payload), encoding="utf-8")

    assert not checkpoint_has_batch_planning_state(tmp_path)


def test_bounded_training_contract_rejects_optimizer_schedule_drift(
    tmp_path: Path,
) -> None:
    spec = _spec()
    state = ShaftBatchPlanningState(
        contract_fingerprint=spec.fingerprint,
        global_microstep=2,
        next_draw_id=4,
        emitted_samples=4,
    )
    _write_bounded_trainer_state(
        tmp_path,
        spec=spec,
        state=state,
        resume_contract_fingerprint="resume-v1",
    )

    with pytest.raises(ValueError, match="training contract changed"):
        validate_batch_planning_resume_contract(
            tmp_path,
            expected_resume_contract_fingerprint="resume-v2",
        )


def _write_full_checkpoint(
    path: Path,
    *,
    trainer_state: bool = True,
    global_step: int | None = None,
) -> None:
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"model")
    if trainer_state:
        inferred_step = (
            int(path.name.rsplit("-", 1)[1])
            if global_step is None and path.name.startswith("checkpoint-")
            else int(global_step or 0)
        )
        (path / "trainer_state.json").write_text(
            json.dumps({"global_step": inferred_step}),
            encoding="utf-8",
        )


def _write_adapter_checkpoint(
    path: Path,
    *,
    trainer_state: bool = True,
    global_step: int = 0,
) -> None:
    path.mkdir(parents=True)
    (path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (path / "adapter_model.safetensors").write_bytes(b"adapter")
    if trainer_state:
        (path / "trainer_state.json").write_text(
            json.dumps({"global_step": int(global_step)}),
            encoding="utf-8",
        )


def test_bounded_resume_resolver_skips_newer_incomplete_checkpoint(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    complete = root / "checkpoint-1"
    incomplete = root / "checkpoint-2"
    _write_full_checkpoint(complete)
    _write_full_checkpoint(incomplete)
    spec = _spec()
    _write_bounded_trainer_state(
        complete,
        spec=spec,
        state=ShaftBatchPlanningState(
            contract_fingerprint=spec.fingerprint,
            global_microstep=2,
            next_draw_id=4,
            emitted_samples=4,
        ),
    )

    assert resolve_resume_checkpoint(
        root,
        protocol=ShaftCheckpointProtocol.COMMITTED_MANIFEST,
        require_planning_state=True,
    ) == str(complete)


@pytest.mark.parametrize(
    "corruption",
    [
        "missing_completion",
        "missing_peer_rng",
        "empty_optimizer",
        "step_misalignment",
        "empty_resume_contract",
    ],
)
def test_bounded_resume_resolver_skips_newer_internally_incomplete_checkpoint(
    tmp_path: Path,
    corruption: str,
) -> None:
    root = tmp_path / "run"
    older = root / "checkpoint-1"
    newer = root / "checkpoint-2"
    _write_full_checkpoint(older)
    _write_full_checkpoint(newer)
    spec = _spec()
    older_state = ShaftBatchPlanningState(
        contract_fingerprint=spec.fingerprint,
        global_microstep=2,
        next_draw_id=4,
        emitted_samples=4,
    )
    newer_state = ShaftBatchPlanningState(
        contract_fingerprint=spec.fingerprint,
        global_microstep=4,
        next_draw_id=8,
        emitted_samples=8,
    )
    _write_bounded_trainer_state(older, spec=spec, state=older_state)
    _write_bounded_trainer_state(newer, spec=spec, state=newer_state)

    if corruption == "missing_completion":
        (newer / TRAINING_CHECKPOINT_COMMIT_FILENAME).unlink()
    elif corruption == "missing_peer_rng":
        (newer / "rng_state_1.pth").unlink()
    elif corruption == "empty_optimizer":
        (newer / "optimizer.pt").write_bytes(b"")
    else:
        payload = json.loads((newer / "trainer_state.json").read_text(encoding="utf-8"))
        if corruption == "step_misalignment":
            payload["global_step"] = 3
        else:
            payload["stateful_callbacks"][BATCH_PLANNING_CALLBACK_NAME]["args"][
                "resume_contract_fingerprint"
            ] = ""
        (newer / "trainer_state.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )

    assert not checkpoint_has_batch_planning_state(newer)
    assert resolve_resume_checkpoint(
        root,
        protocol=ShaftCheckpointProtocol.COMMITTED_MANIFEST,
        require_planning_state=True,
    ) == str(older)


def test_ensure_hf_export_layout_and_best_dir(tmp_path: Path) -> None:
    full = tmp_path / "full"
    _write_full_checkpoint(full, trainer_state=False)
    ensure_hf_export_layout(full, finetune_mode="full")
    assert resolve_best_export_dir(tmp_path) == tmp_path / "best"


def test_ensure_hf_export_layout_validates_model_specific_files(tmp_path: Path) -> None:
    full = tmp_path / "full"
    _write_full_checkpoint(full, trainer_state=False)
    model_meta = SimpleNamespace(required_saved_files=lambda: ("processor_config.json",))
    with pytest.raises(ValueError, match="Missing additional saved files"):
        ensure_hf_export_layout(full, finetune_mode="full", model_meta=model_meta)


@pytest.mark.parametrize("mode", ["lora", "dora", "qlora"])
def test_validate_resume_checkpoint_accepts_matching_adapter_mode(
    tmp_path: Path,
    mode: str,
) -> None:
    checkpoint = tmp_path / mode
    _write_adapter_checkpoint(checkpoint)
    _write_exact_resume_artifacts(checkpoint, world_size=1)
    commit_training_checkpoint(checkpoint, world_size=1, requires_grad_scaler=False)
    validate_resume_checkpoint(
        checkpoint,
        finetune_mode=mode,
        protocol=ShaftCheckpointProtocol.COMMITTED_MANIFEST,
    )


def test_validate_resume_checkpoint_rejects_mismatched_or_missing_state(
    tmp_path: Path,
) -> None:
    full = tmp_path / "full"
    _write_full_checkpoint(full)
    _write_exact_resume_artifacts(full, world_size=1)
    commit_training_checkpoint(full, world_size=1, requires_grad_scaler=False)
    with pytest.raises(ValueError, match="adapter"):
        validate_resume_checkpoint(
            full,
            finetune_mode="lora",
            protocol=ShaftCheckpointProtocol.COMMITTED_MANIFEST,
        )

    export = tmp_path / "export"
    _write_full_checkpoint(export, trainer_state=False)
    with pytest.raises(ValueError, match="trainer_state"):
        validate_resume_checkpoint(
            export,
            finetune_mode="full",
            protocol=ShaftCheckpointProtocol.COMMITTED_MANIFEST,
        )


def test_training_resume_contract_round_trips_canonical_payload() -> None:
    config = RuntimeConfig()
    config.algorithm.name = "sft"
    batch_contract = _fixed_batch_contract()
    contract = build_training_resume_contract(
        config=config,
        training_args=_resume_training_args(),
        batch_contract_fingerprint=batch_contract.fingerprint,
    )

    assert ShaftTrainingResumeContract.from_dict(contract.to_dict()) == contract
    assert len(contract.fingerprint) == 64
    assert contract.train_input_contract_fingerprint == _train_input_contract().fingerprint
    assert contract.data_execution_fingerprint == "data-v1"
    assert contract.to_dict()["objective"] == {
        "ignore_index": -100,
        "loss_name": "auto",
        "loss_scale": "default",
    }


def test_training_resume_contract_composes_input_and_data_execution_identity() -> None:
    config = RuntimeConfig()
    batch_fingerprint = _fixed_batch_contract().fingerprint
    original = build_training_resume_contract(
        config=config,
        training_args=_resume_training_args(),
        batch_contract_fingerprint=batch_fingerprint,
    )
    changed_input = build_training_resume_contract(
        config=config,
        training_args=_resume_training_args(),
        batch_contract_fingerprint=batch_fingerprint,
        train_input_contract_fingerprint="changed-input",
    )
    changed_data = build_training_resume_contract(
        config=config,
        training_args=_resume_training_args(),
        batch_contract_fingerprint=batch_fingerprint,
        data_execution_fingerprint="changed-data",
    )

    assert original.fingerprint != changed_input.fingerprint
    assert original.fingerprint != changed_data.fingerprint


def test_batching_metadata_rejects_resume_root_with_different_input_identity() -> None:
    contract = build_training_resume_contract(
        config=RuntimeConfig(),
        training_args=_resume_training_args(),
        batch_contract_fingerprint=_fixed_batch_contract().fingerprint,
        train_input_contract_fingerprint="different-input",
    )

    with pytest.raises(ValueError, match="different training input contract"):
        _metadata_with_training_resume_contract(contract)


@pytest.mark.parametrize(
    ("section", "field_name"),
    [
        (None, "unknown_top_level"),
        ("duration", "unknown_duration"),
        ("optimizer", "unknown_optimizer"),
        ("scheduler", "unknown_scheduler"),
    ],
)
def test_training_resume_contract_rejects_unknown_schema_fields(
    section: str | None,
    field_name: str,
) -> None:
    contract = build_training_resume_contract(
        config=RuntimeConfig(),
        training_args=_resume_training_args(),
        batch_contract_fingerprint=_fixed_batch_contract().fingerprint,
    )
    payload = contract.to_dict()
    target = payload if section is None else payload[section]
    target[field_name] = 1
    with pytest.raises(ValueError, match="schema differs"):
        ShaftTrainingResumeContract.from_dict(payload)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("seed",), True),
        (("data_seed",), "42"),
        (("gradient_accumulation_steps",), False),
        (("duration", "resolved_max_steps"), True),
        (("duration", "value"), "100"),
        (("optimizer", "learning_rate"), False),
        (("scheduler", "warmup_ratio"), "0.1"),
    ],
)
def test_training_resume_contract_rejects_non_json_numeric_types(
    path: tuple[str, ...],
    value: object,
) -> None:
    contract = build_training_resume_contract(
        config=RuntimeConfig(),
        training_args=_resume_training_args(),
        batch_contract_fingerprint=_fixed_batch_contract().fingerprint,
    )
    payload = contract.to_dict()
    target = payload
    for field_name in path[:-1]:
        target = target[field_name]
    target[path[-1]] = value
    with pytest.raises(TypeError, match="JSON (integer|number)"):
        ShaftTrainingResumeContract.from_dict(payload)


@pytest.mark.parametrize(
    ("section", "invalid_key"),
    [
        (None, 7),
        ("execution", 9),
        ("implementation", False),
        ("objective", 3.5),
        ("optimizer.param_group_lrs", 11),
    ],
)
def test_training_resume_contract_rejects_non_string_json_keys(
    section: str | None,
    invalid_key: object,
) -> None:
    contract = build_training_resume_contract(
        config=RuntimeConfig(),
        training_args=_resume_training_args(),
        batch_contract_fingerprint=_fixed_batch_contract().fingerprint,
    )
    payload = contract.to_dict()
    if section is None:
        target = payload
    elif section == "optimizer.param_group_lrs":
        target = payload["optimizer"]["param_group_lrs"]
    else:
        target = payload[section]
    target[invalid_key] = "invalid"

    with pytest.raises(TypeError, match="keys must be JSON strings"):
        ShaftTrainingResumeContract.from_dict(payload)


class _ResumeOpaqueMode(Enum):
    FIRST = "first"


@pytest.mark.parametrize(
    "value",
    [
        (1, 2),
        {1, 2},
        Path("opaque"),
        _ResumeOpaqueMode.FIRST,
    ],
)
def test_training_resume_contract_rejects_python_only_nested_values(
    value: object,
) -> None:
    contract = build_training_resume_contract(
        config=RuntimeConfig(),
        training_args=_resume_training_args(),
        batch_contract_fingerprint=_fixed_batch_contract().fingerprint,
    )
    payload = contract.to_dict()
    payload["objective"]["opaque"] = value

    with pytest.raises(TypeError, match="canonical JSON"):
        ShaftTrainingResumeContract.from_dict(payload)


def test_distributed_training_stage_rejects_malformed_peer_envelope() -> None:
    malformed = {
        "ok": "false",
        "error_type": None,
        "error": None,
        "fingerprints": {"training": "same"},
    }
    with patch(
        "shaft.training.resume_contract.all_gather_objects",
        return_value=[malformed],
    ):
        with pytest.raises(RuntimeError, match="malformed status envelope"):
            with distributed_training_contract_stage(
                stage="malformed-fixture",
                fingerprints=lambda: {"training": "same"},
            ):
                pass


@pytest.mark.parametrize(
    "fingerprints",
    [
        {"training": True},
        {"training": 1},
        {1: "training"},
        {"": "training"},
        {"training": ""},
    ],
)
def test_distributed_training_stage_rejects_local_fingerprint_coercion(
    fingerprints: dict[object, object],
) -> None:
    with pytest.raises(TypeError, match="non-empty strings"):
        with distributed_training_contract_stage(
            stage="malformed-local-fingerprint",
            fingerprints=lambda: fingerprints,  # type: ignore[arg-type,return-value]
        ):
            pass


def test_dpo_beta_drift_is_rejected_before_exact_resume(tmp_path: Path) -> None:
    config = RuntimeConfig()
    config.algorithm.name = "dpo"
    config.model.finetune.mode = "full"
    batch_contract = _fixed_batch_contract()
    original = build_training_resume_contract(
        config=config,
        training_args=_resume_training_args(),
        batch_contract_fingerprint=batch_contract.fingerprint,
        resolved_dpo_args=_resolved_dpo_args(),
    )
    checkpoint = tmp_path / "checkpoint-1"
    _write_metadata_checkpoint(
        checkpoint,
        _metadata_with_training_resume_contract(original),
    )

    config.rlhf.dpo.beta = 0.25
    changed = build_training_resume_contract(
        config=config,
        training_args=_resume_training_args(),
        batch_contract_fingerprint=batch_contract.fingerprint,
        resolved_dpo_args=_resolved_dpo_args(beta=0.25),
    )

    with pytest.raises(ValueError, match=r"Training resume contract.*objective"):
        validate_batching_resume_contract(
            checkpoint,
            expected_contract=batch_contract,
            expected_training_resume_contract=changed,
        )


@pytest.mark.parametrize(
    ("field_name", "changed_value"),
    [("padding_free", True), ("f_divergence_type", "js_divergence")],
)
def test_dpo_resolved_trl_objective_drift_is_rejected(
    tmp_path: Path,
    field_name: str,
    changed_value: object,
) -> None:
    config = RuntimeConfig()
    config.algorithm.name = "dpo"
    batch_contract = _fixed_batch_contract()
    original = build_training_resume_contract(
        config=config,
        training_args=_resume_training_args(),
        batch_contract_fingerprint=batch_contract.fingerprint,
        resolved_dpo_args=_resolved_dpo_args(),
    )
    checkpoint = tmp_path / field_name / "checkpoint-1"
    _write_metadata_checkpoint(
        checkpoint,
        _metadata_with_training_resume_contract(original),
    )
    changed = build_training_resume_contract(
        config=config,
        training_args=_resume_training_args(),
        batch_contract_fingerprint=batch_contract.fingerprint,
        resolved_dpo_args=_resolved_dpo_args(**{field_name: changed_value}),
    )

    with pytest.raises(ValueError, match=r"Training resume contract.*objective"):
        validate_batching_resume_contract(
            checkpoint,
            expected_contract=batch_contract,
            expected_training_resume_contract=changed,
        )


@pytest.mark.parametrize(
    ("field_name", "changed_value", "changed_section"),
    [
        ("optimizer_name", "muon", "optimizer"),
        ("scheduler_name", "linear", "scheduler"),
        ("duration", 120, "duration"),
    ],
)
def test_fixed_sft_training_trajectory_drift_is_rejected(
    tmp_path: Path,
    field_name: str,
    changed_value: object,
    changed_section: str,
) -> None:
    config = RuntimeConfig()
    config.algorithm.name = "sft"
    batch_contract = _fixed_batch_contract()
    original = build_training_resume_contract(
        config=config,
        training_args=_resume_training_args(),
        batch_contract_fingerprint=batch_contract.fingerprint,
    )
    checkpoint = tmp_path / field_name / "checkpoint-1"
    _write_metadata_checkpoint(
        checkpoint,
        _metadata_with_training_resume_contract(original),
    )

    changed_args = _resume_training_args()
    if field_name == "duration":
        config.train.duration.value = changed_value
        changed_args.max_steps = int(changed_value)
    else:
        setattr(config.train, field_name, changed_value)
    changed = build_training_resume_contract(
        config=config,
        training_args=changed_args,
        batch_contract_fingerprint=batch_contract.fingerprint,
    )

    with pytest.raises(ValueError, match=changed_section):
        validate_batching_resume_contract(
            checkpoint,
            expected_contract=batch_contract,
            expected_training_resume_contract=changed,
        )


def test_grpo_reward_and_update_cadence_drift_are_rejected(tmp_path: Path) -> None:
    config = RuntimeConfig()
    config.algorithm.name = "grpo"
    batch_contract = _fixed_batch_contract()
    original = build_training_resume_contract(
        config=config,
        training_args=_resume_training_args(),
        batch_contract_fingerprint=batch_contract.fingerprint,
        resolved_grpo_args=_resolved_grpo_args(),
    )
    checkpoint = tmp_path / "checkpoint-1"
    _write_metadata_checkpoint(
        checkpoint,
        _metadata_with_training_resume_contract(original),
    )

    config.rlhf.grpo.reward_functions[0].weight = 0.5
    changed = build_training_resume_contract(
        config=config,
        training_args=_resume_training_args(),
        batch_contract_fingerprint=batch_contract.fingerprint,
        resolved_grpo_args=_resolved_grpo_args(steps_per_generation=4),
    )

    with pytest.raises(ValueError, match=r"Training resume contract.*objective"):
        validate_batching_resume_contract(
            checkpoint,
            expected_contract=batch_contract,
            expected_training_resume_contract=changed,
        )


@pytest.mark.parametrize(
    ("field_name", "changed_value"),
    [("loss_type", "grpo"), ("scale_rewards", "batch")],
)
def test_grpo_resolved_trl_loss_semantics_drift_is_rejected(
    tmp_path: Path,
    field_name: str,
    changed_value: object,
) -> None:
    config = RuntimeConfig()
    config.algorithm.name = "grpo"
    batch_contract = _fixed_batch_contract()
    original = build_training_resume_contract(
        config=config,
        training_args=_resume_training_args(),
        batch_contract_fingerprint=batch_contract.fingerprint,
        resolved_grpo_args=_resolved_grpo_args(),
    )
    checkpoint = tmp_path / field_name / "checkpoint-1"
    _write_metadata_checkpoint(
        checkpoint,
        _metadata_with_training_resume_contract(original),
    )
    changed = build_training_resume_contract(
        config=config,
        training_args=_resume_training_args(),
        batch_contract_fingerprint=batch_contract.fingerprint,
        resolved_grpo_args=_resolved_grpo_args(**{field_name: changed_value}),
    )

    with pytest.raises(ValueError, match=r"Training resume contract.*objective"):
        validate_batching_resume_contract(
            checkpoint,
            expected_contract=batch_contract,
            expected_training_resume_contract=changed,
        )


def test_grpo_eval_only_generation_count_does_not_change_training_contract() -> None:
    config = RuntimeConfig()
    config.algorithm.name = "grpo"
    batch_contract = _fixed_batch_contract()
    original = build_training_resume_contract(
        config=config,
        training_args=_resume_training_args(),
        batch_contract_fingerprint=batch_contract.fingerprint,
        resolved_grpo_args=_resolved_grpo_args(num_generations_eval=1),
    )
    changed = build_training_resume_contract(
        config=config,
        training_args=_resume_training_args(),
        batch_contract_fingerprint=batch_contract.fingerprint,
        resolved_grpo_args=_resolved_grpo_args(num_generations_eval=8),
    )

    assert original.fingerprint == changed.fingerprint


def test_grpo_reward_implementation_drift_changes_contract() -> None:
    config = RuntimeConfig()
    config.algorithm.name = "grpo"
    batch_contract = _fixed_batch_contract()
    with patch(
        "shaft.algorithms.grpo_rewards.GRPO_REWARD_REGISTRY.get",
        return_value=_fixture_reward_v1,
    ):
        original = build_training_resume_contract(
            config=config,
            training_args=_resume_training_args(),
            batch_contract_fingerprint=batch_contract.fingerprint,
            resolved_grpo_args=_resolved_grpo_args(),
        )
    with patch(
        "shaft.algorithms.grpo_rewards.GRPO_REWARD_REGISTRY.get",
        return_value=_fixture_reward_v2,
    ):
        changed = build_training_resume_contract(
            config=config,
            training_args=_resume_training_args(),
            batch_contract_fingerprint=batch_contract.fingerprint,
            resolved_grpo_args=_resolved_grpo_args(),
        )

    assert original.fingerprint != changed.fingerprint


def test_training_resume_contract_binds_use_cpu_and_fsdp_runtime() -> None:
    config = RuntimeConfig()
    config.algorithm.name = "sft"
    config.train.distributed.strategy = "fsdp"
    batch_contract = _fixed_batch_contract()
    original = build_training_resume_contract(
        config=config,
        training_args=_resume_training_args(
            use_cpu=False,
            fsdp=True,
            fsdp_config={"use_orig_params": True},
        ),
        batch_contract_fingerprint=batch_contract.fingerprint,
    )

    config.train.use_cpu = True
    config.train.distributed.fsdp.use_orig_params = False
    changed = build_training_resume_contract(
        config=config,
        training_args=_resume_training_args(
            use_cpu=True,
            fsdp=True,
            fsdp_config={"use_orig_params": False},
        ),
        batch_contract_fingerprint=batch_contract.fingerprint,
    )

    assert original.fingerprint != changed.fingerprint
    assert original.to_dict()["execution"] != changed.to_dict()["execution"]


def test_training_resume_contract_binds_resolved_ddp_static_graph(tmp_path: Path) -> None:
    config = RuntimeConfig()
    config.algorithm.name = "sft"
    batch_contract = _fixed_batch_contract()
    original = build_training_resume_contract(
        config=config,
        training_args=_resume_training_args(ddp_static_graph=False),
        batch_contract_fingerprint=batch_contract.fingerprint,
    )

    config.train.distributed.ddp.static_graph = True
    changed = build_training_resume_contract(
        config=config,
        training_args=_resume_training_args(ddp_static_graph=True),
        batch_contract_fingerprint=batch_contract.fingerprint,
    )

    assert original.fingerprint != changed.fingerprint
    assert original.to_dict()["execution"]["ddp_static_graph"] is False
    assert changed.to_dict()["execution"]["ddp_static_graph"] is True
    checkpoint = tmp_path / "checkpoint-1"
    _write_metadata_checkpoint(
        checkpoint,
        _metadata_with_training_resume_contract(original),
    )
    with pytest.raises(ValueError, match=r"Training resume contract.*execution"):
        validate_batching_resume_contract(
            checkpoint,
            expected_contract=batch_contract,
            expected_training_resume_contract=changed,
        )


def test_training_resume_contract_binds_complete_finetune_config() -> None:
    config = RuntimeConfig()
    config.algorithm.name = "sft"
    config.model.finetune.mode = "lora"
    batch_contract = _fixed_batch_contract()
    original = build_training_resume_contract(
        config=config,
        training_args=_resume_training_args(),
        batch_contract_fingerprint=batch_contract.fingerprint,
    )

    config.model.finetune.lora_alpha = 64
    config.model.finetune.lora_dropout = 0.1
    config.model.finetune.freeze.regex = r"^model\.visual\."
    changed = build_training_resume_contract(
        config=config,
        training_args=_resume_training_args(),
        batch_contract_fingerprint=batch_contract.fingerprint,
    )

    assert original.fingerprint != changed.fingerprint
    assert (
        original.to_dict()["execution"]["finetune_config"]
        != changed.to_dict()["execution"]["finetune_config"]
    )


def test_training_resume_contract_binds_effective_model_execution() -> None:
    config = RuntimeConfig()
    config.algorithm.name = "sft"
    batch_contract = _fixed_batch_contract()
    original = build_training_resume_contract(
        config=config,
        training_args=_resume_training_args(torch_compile=False),
        batch_contract_fingerprint=batch_contract.fingerprint,
    )

    config.model.torch_dtype = "float32"
    config.model.attn_implementation = "sdpa"
    config.model.device_map = "auto"
    changed = build_training_resume_contract(
        config=config,
        training_args=_resume_training_args(
            torch_compile=True,
            torch_compile_backend="inductor",
            torch_compile_mode="default",
        ),
        batch_contract_fingerprint=batch_contract.fingerprint,
    )

    assert original.fingerprint != changed.fingerprint
    assert (
        original.to_dict()["execution"]["model_execution"]
        != changed.to_dict()["execution"]["model_execution"]
    )


def test_training_resume_contract_binds_best_model_selection_cadence() -> None:
    config = RuntimeConfig()
    config.train.load_best_model_at_end = True
    config.eval.enabled = True
    config.eval.eval_strategy = "steps"
    config.train.save_strategy = "steps"
    batch_contract = _fixed_batch_contract()
    original = build_training_resume_contract(
        config=config,
        training_args=_resume_training_args(load_best_model_at_end=True),
        batch_contract_fingerprint=batch_contract.fingerprint,
    )

    config.eval.metric_for_best_model = "eval_final_score"
    config.eval.greater_is_better = True
    config.eval.eval_steps = 17
    config.train.save_steps = 17
    changed = build_training_resume_contract(
        config=config,
        training_args=_resume_training_args(load_best_model_at_end=True),
        batch_contract_fingerprint=batch_contract.fingerprint,
    )

    assert original.fingerprint != changed.fingerprint
    assert (
        original.to_dict()["execution"]["final_selection"]
        != changed.to_dict()["execution"]["final_selection"]
    )


def test_training_resume_contract_binds_deepspeed_file_contents(
    tmp_path: Path,
) -> None:
    config = RuntimeConfig()
    config.algorithm.name = "sft"
    config.train.distributed.strategy = "deepspeed"
    deepspeed_path = tmp_path / "deepspeed.json"
    deepspeed_path.write_text('{"zero_optimization": {"stage": 2}}', encoding="utf-8")
    config.train.distributed.deepspeed.config_path = str(deepspeed_path)
    batch_contract = _fixed_batch_contract()
    original = build_training_resume_contract(
        config=config,
        training_args=_resume_training_args(deepspeed=str(deepspeed_path)),
        batch_contract_fingerprint=batch_contract.fingerprint,
    )

    deepspeed_path.write_text('{"zero_optimization": {"stage": 3}}', encoding="utf-8")
    changed = build_training_resume_contract(
        config=config,
        training_args=_resume_training_args(deepspeed=str(deepspeed_path)),
        batch_contract_fingerprint=batch_contract.fingerprint,
    )

    assert original.fingerprint != changed.fingerprint


@pytest.mark.parametrize(
    "registry_get",
    [
        "shaft.training.optimizer.OPTIMIZER_REGISTRY.get",
        "shaft.training.scheduler.SCHEDULER_REGISTRY.get",
    ],
)
def test_training_resume_contract_binds_optimizer_and_scheduler_implementations(
    registry_get: str,
) -> None:
    config = RuntimeConfig()
    config.algorithm.name = "sft"
    batch_contract = _fixed_batch_contract()
    with patch(registry_get, return_value=_fixture_builder_v1):
        original = build_training_resume_contract(
            config=config,
            training_args=_resume_training_args(),
            batch_contract_fingerprint=batch_contract.fingerprint,
        )
    with patch(registry_get, return_value=_fixture_builder_v2):
        changed = build_training_resume_contract(
            config=config,
            training_args=_resume_training_args(),
            batch_contract_fingerprint=batch_contract.fingerprint,
        )

    assert original.fingerprint != changed.fingerprint


def test_training_resume_contract_binds_muon_transitive_implementation() -> None:
    config = RuntimeConfig()
    config.train.optimizer_name = "muon"
    batch_contract = _fixed_batch_contract()
    with patch("shaft.training.optimizer.Muon", _FixtureMuonV1):
        original = build_training_resume_contract(
            config=config,
            training_args=_resume_training_args(),
            batch_contract_fingerprint=batch_contract.fingerprint,
        )
    with patch("shaft.training.optimizer.Muon", _FixtureMuonV2):
        changed = build_training_resume_contract(
            config=config,
            training_args=_resume_training_args(),
            batch_contract_fingerprint=batch_contract.fingerprint,
        )

    assert original.fingerprint != changed.fingerprint


def test_training_resume_contract_accepts_default_sft_runtime() -> None:
    config = RuntimeConfig()

    contract = build_training_resume_contract(
        config=config,
        training_args=_resume_training_args(),
        batch_contract_fingerprint=_fixed_batch_contract().fingerprint,
    )

    assert contract.algorithm == "sft"
    assert len(contract.fingerprint) == 64


def test_training_resume_contract_binds_algorithm_helper_implementation() -> None:
    config = RuntimeConfig()
    config.algorithm.name = "dpo"
    batch_contract = _fixed_batch_contract()
    with patch(
        "shaft.algorithms.dpo.build_reference_model",
        _fixture_reference_model_v1,
    ):
        original = build_training_resume_contract(
            config=config,
            training_args=_resume_training_args(),
            batch_contract_fingerprint=batch_contract.fingerprint,
            resolved_dpo_args=_resolved_dpo_args(),
        )
    with patch(
        "shaft.algorithms.dpo.build_reference_model",
        _fixture_reference_model_v2,
    ):
        changed = build_training_resume_contract(
            config=config,
            training_args=_resume_training_args(),
            batch_contract_fingerprint=batch_contract.fingerprint,
            resolved_dpo_args=_resolved_dpo_args(),
        )

    assert original.fingerprint != changed.fingerprint


def test_callable_semantic_signature_binds_closure_values() -> None:
    assert callable_semantic_signature(
        _closure_builder(1),
        role="fixture_builder",
    ) != callable_semantic_signature(
        _closure_builder(9),
        role="fixture_builder",
    )


def test_component_semantic_signature_handles_self_referential_state() -> None:
    policy = SimpleNamespace(mode="strict")
    policy.self = policy

    original = component_semantic_signature(policy, role="self_referential_policy")
    repeated = component_semantic_signature(policy, role="self_referential_policy")
    policy.mode = "relaxed"
    changed = component_semantic_signature(policy, role="self_referential_policy")

    assert original == repeated
    assert original != changed


def test_training_resume_contract_binds_codec_helper_implementation() -> None:
    config = RuntimeConfig()
    config.algorithm.name = "grpo"
    batch_contract = _fixed_batch_contract()
    original = build_training_resume_contract(
        config=config,
        training_args=_resume_training_args(),
        batch_contract_fingerprint=batch_contract.fingerprint,
        resolved_grpo_args=_resolved_grpo_args(),
    )
    with patch(
        "shaft.codec.json._decode_json_lenient",
        _fixture_codec_helper,
    ):
        changed = build_training_resume_contract(
            config=config,
            training_args=_resume_training_args(),
            batch_contract_fingerprint=batch_contract.fingerprint,
            resolved_grpo_args=_resolved_grpo_args(),
        )

    assert original.fingerprint != changed.fingerprint


def test_training_resume_contract_binds_rng_isolation_policy() -> None:
    config = RuntimeConfig()
    batch_contract = _fixed_batch_contract()
    original = build_training_resume_contract(
        config=config,
        training_args=_resume_training_args(),
        batch_contract_fingerprint=batch_contract.fingerprint,
    )
    with patch(
        "shaft.training.reproducibility.preserve_training_rng_state",
        _fixture_reproducibility_policy_v2,
    ):
        changed = build_training_resume_contract(
            config=config,
            training_args=_resume_training_args(),
            batch_contract_fingerprint=batch_contract.fingerprint,
        )

    assert original.fingerprint != changed.fingerprint


def test_training_resume_contract_binds_hook_order_state_and_algorithm_params() -> None:
    config = RuntimeConfig()
    config.algorithm.name = "sft"
    config.plugins.hooks = ["fixture"]
    config.algorithm.params = {"trajectory_scale": 1}
    batch_contract = _fixed_batch_contract()
    with patch(
        "shaft.plugins.hooks.HOOK_REGISTRY.create",
        return_value=SimpleNamespace(
            name="fixture",
            before_step_fn=_closure_builder(1),
            shaft_trajectory_neutral=True,
        ),
    ) as create_hook:
        hook_manager = build_hook_manager(config.plugins.hooks)
        original = build_training_resume_contract(
            config=config,
            training_args=_resume_training_args(),
            batch_contract_fingerprint=batch_contract.fingerprint,
            hook_instances=hook_manager.hooks,
        )
        assert create_hook.call_count == 1

    config.algorithm.params = {"trajectory_scale": 9}
    with patch(
        "shaft.plugins.hooks.HOOK_REGISTRY.create",
        return_value=SimpleNamespace(
            name="fixture",
            before_step_fn=_closure_builder(9),
            shaft_trajectory_neutral=True,
        ),
    ) as create_hook:
        hook_manager = build_hook_manager(config.plugins.hooks)
        changed = build_training_resume_contract(
            config=config,
            training_args=_resume_training_args(),
            batch_contract_fingerprint=batch_contract.fingerprint,
            hook_instances=hook_manager.hooks,
        )
        assert create_hook.call_count == 1

    assert original.fingerprint != changed.fingerprint


def test_training_resume_contract_handles_self_referential_plugin_state() -> None:
    @dataclass
    class _Observer:
        name: str = "self-referential"
        shaft_trajectory_neutral: bool = True
        parent: object | None = None

    observer = _Observer()
    observer.parent = observer
    config = RuntimeConfig()
    config.train.save_strategy = "steps"
    config.plugins.hooks = [observer.name]

    original = build_training_resume_contract(
        config=config,
        training_args=_resume_training_args(),
        batch_contract_fingerprint=_fixed_batch_contract().fingerprint,
        hook_instances=[observer],
    )
    repeated = build_training_resume_contract(
        config=config,
        training_args=_resume_training_args(),
        batch_contract_fingerprint=_fixed_batch_contract().fingerprint,
        hook_instances=[observer],
    )

    assert original.fingerprint == repeated.fingerprint


def test_plugin_identity_has_one_bounded_state_truth_source() -> None:
    observer = SimpleNamespace(
        name="cached-observer",
        shaft_trajectory_neutral=True,
        cache={f"item-{index}": index for index in range(50_000)},
    )

    original = resume_contract_module._plugin_instance_identity(  # noqa: SLF001
        observer,
        role="training_hook:cached-observer",
    )
    observer.cache["item-49"] = -1
    changed = resume_contract_module._plugin_instance_identity(  # noqa: SLF001
        observer,
        role="training_hook:cached-observer",
    )

    assert set(original) == {"implementation"}
    assert original != changed


def test_checkpointable_contract_rejects_plugin_without_neutrality_marker() -> None:
    config = RuntimeConfig()
    config.train.save_strategy = "steps"
    config.plugins.hooks = ["fixture"]
    instance = SimpleNamespace(name="fixture", counter=0)

    with pytest.raises(ValueError, match="shaft_trajectory_neutral=True"):
        build_training_resume_contract(
            config=config,
            training_args=_resume_training_args(),
            batch_contract_fingerprint=_fixed_batch_contract().fingerprint,
            hook_instances=[instance],
        )


def test_decorated_observer_plugins_flow_through_managers_into_resume_contract() -> None:
    hook_name = "resume_contract_neutral_hook"
    interceptor_name = "resume_contract_neutral_interceptor"

    @hook(
        "after_step",
        name=hook_name,
        trajectory_neutral=True,
    )
    def _neutral_hook(state: dict) -> None:
        _ = state

    @interceptor(
        "pipeline.sft.run",
        name=interceptor_name,
        trajectory_neutral=True,
    )
    def _neutral_interceptor(state: dict) -> None:
        _ = state

    config = RuntimeConfig()
    config.train.save_strategy = "steps"
    config.plugins.hooks = [hook_name]
    config.plugins.interceptors = [interceptor_name]
    hook_manager = build_hook_manager(config.plugins.hooks)
    interceptor_manager = build_interceptor_manager(config.plugins.interceptors)

    contract = build_training_resume_contract(
        config=config,
        training_args=_resume_training_args(),
        batch_contract_fingerprint=_fixed_batch_contract().fingerprint,
        hook_instances=hook_manager.hooks,
        interceptor_instances=interceptor_manager.interceptors,
    )

    plugins = contract.to_dict()["implementation"]["plugins"]
    assert [item["name"] for item in plugins["hooks"]] == [hook_name]
    assert [item["name"] for item in plugins["interceptors"]] == [interceptor_name]


def test_resume_contract_rejects_string_plugin_neutrality_marker() -> None:
    config = RuntimeConfig()
    config.train.save_strategy = "steps"
    config.plugins.hooks = ["malformed-neutral"]
    instance = SimpleNamespace(
        name="malformed-neutral",
        shaft_trajectory_neutral="false",
    )

    with pytest.raises(TypeError, match="shaft_trajectory_neutral.*boolean"):
        build_training_resume_contract(
            config=config,
            training_args=_resume_training_args(),
            batch_contract_fingerprint=_fixed_batch_contract().fingerprint,
            hook_instances=[instance],
        )


def test_builtin_observer_plugin_is_exact_resume_safe() -> None:
    config = RuntimeConfig()
    config.train.save_strategy = "steps"
    config.plugins.hooks = ["log_before_step"]
    manager = build_hook_manager(config.plugins.hooks)

    contract = build_training_resume_contract(
        config=config,
        training_args=_resume_training_args(),
        batch_contract_fingerprint=_fixed_batch_contract().fingerprint,
        hook_instances=manager.hooks,
    )

    assert contract.to_dict()["implementation"]["plugins"]["hooks"][0]["name"] == "log_before_step"


def test_training_resume_contract_binds_selective_runtime_versions() -> None:
    config = RuntimeConfig()
    config.algorithm.name = "dpo"
    batch_contract = _fixed_batch_contract()

    def version_v1(name: str) -> str:
        return f"{name}-1"

    def version_with_trl_drift(name: str) -> str:
        return "trl-2" if name == "trl" else f"{name}-1"

    with patch(
        "shaft.training.resume_contract._runtime_package_version",
        side_effect=version_v1,
    ):
        original = build_training_resume_contract(
            config=config,
            training_args=_resume_training_args(),
            batch_contract_fingerprint=batch_contract.fingerprint,
            resolved_dpo_args=_resolved_dpo_args(),
        )
    with patch(
        "shaft.training.resume_contract._runtime_package_version",
        side_effect=version_with_trl_drift,
    ):
        changed = build_training_resume_contract(
            config=config,
            training_args=_resume_training_args(),
            batch_contract_fingerprint=batch_contract.fingerprint,
            resolved_dpo_args=_resolved_dpo_args(),
        )

    assert original.fingerprint != changed.fingerprint


def test_training_resume_contract_derives_model_sequence_runtime_dependencies() -> None:
    config = RuntimeConfig()
    batch_contract = _fixed_batch_contract()
    capabilities = (
        "flash-attn=2.8.3",
        "flash-linear-attention=0.4.0",
        "causal-conv1d=1.5.0",
    )

    with patch(
        "shaft.training.resume_contract._runtime_package_version",
        side_effect=lambda name: f"{name}-v1",
    ):
        original = build_training_resume_contract(
            config=config,
            training_args=_resume_training_args(),
            batch_contract_fingerprint=batch_contract.fingerprint,
            sequence_execution_capabilities=capabilities,
        )
    with patch(
        "shaft.training.resume_contract._runtime_package_version",
        side_effect=lambda name: (
            "flash-linear-attention-v2" if name == "flash-linear-attention" else f"{name}-v1"
        ),
    ):
        changed = build_training_resume_contract(
            config=config,
            training_args=_resume_training_args(),
            batch_contract_fingerprint=batch_contract.fingerprint,
            sequence_execution_capabilities=capabilities,
        )

    assert original.fingerprint != changed.fingerprint


def test_exact_resume_rejects_checkpoint_without_training_resume_contract(
    tmp_path: Path,
) -> None:
    batch_contract = _fixed_batch_contract()
    metadata = _metadata_with_training_resume_contract(
        build_training_resume_contract(
            config=RuntimeConfig(),
            training_args=_resume_training_args(),
            batch_contract_fingerprint=batch_contract.fingerprint,
        )
    )
    legacy_payload = metadata.to_dict()
    legacy_payload.pop("training_resume_contract")
    legacy_payload.pop("training_resume_contract_fingerprint")
    checkpoint = tmp_path / "checkpoint-1"
    checkpoint.mkdir()
    callback_state = ShaftBatchingMetadataCallback(metadata).state()
    callback_state["args"]["metadata"] = legacy_payload
    (checkpoint / "trainer_state.json").write_text(
        json.dumps(
            {
                "global_step": 1,
                "stateful_callbacks": {
                    BATCHING_METADATA_CALLBACK_NAME: callback_state,
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="schema differs.*training_resume_contract",
    ):
        validate_batching_resume_contract(
            checkpoint,
            expected_contract=batch_contract,
            require_training_resume_contract_payload=True,
        )
