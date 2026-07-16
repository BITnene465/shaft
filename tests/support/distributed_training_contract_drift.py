from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sys
from unittest.mock import patch

import torch
import torch.distributed as dist

from shaft.training.resume_contract import (
    converge_training_contract_fingerprints,
    distributed_training_contract_stage,
)


def _load_shared_sft_config(path: Path):
    from shaft.config import load_config
    from tests.support.pipeline import write_sft_pipeline_config

    if dist.get_rank() == 0:
        path.mkdir(parents=True, exist_ok=True)
        config = write_sft_pipeline_config(path)
        source = config.data.datasets[0]
        train_path = Path(source.train_path or source.train_paths[0])
        sample = train_path.read_text(encoding="utf-8")
        train_path.write_text(sample * 8, encoding="utf-8")
    dist.barrier()
    config = load_config(path / "config.yaml")
    config.model.model_type = "smoke_vlm"
    config.model.model_name_or_path = "models/Smoke-VLM"
    config.data.media_snapshot_id = "distributed-setup-fixture-v1"
    config.data.num_workers = 0
    config.data.persistent_workers = False
    config.train.efficiency.enabled = False
    return config


def _expect_sft_pipeline_setup_failure(
    root: Path,
    *,
    mode: str,
    expected_message: str,
) -> None:
    import shaft.pipeline.sft as sft_module
    from shaft.pipeline import run_sft
    from tests.support.pipeline import (
        FakePipelineTrainer,
        build_fake_model_artifacts,
    )

    config = _load_shared_sft_config(root / mode)
    original_build_metadata = sft_module.build_batching_run_metadata
    original_collator = sft_module.SFTCollator
    original_stage = sft_module.distributed_training_contract_stage
    status_stage_depth = 0

    @contextmanager
    def _tracked_status_stage(*args, **kwargs):
        nonlocal status_stage_depth
        status_stage_depth += 1
        try:
            with original_stage(*args, **kwargs):
                yield
        finally:
            status_stage_depth -= 1

    class _CollectiveTrainer(FakePipelineTrainer):
        constructed = False

        def __init__(self, **kwargs):
            type(self).constructed = True
            if status_stage_depth:
                raise AssertionError("Trainer constructor ran inside a status envelope.")
            ready = torch.tensor([1], dtype=torch.int64)
            dist.all_reduce(ready)
            super().__init__(**kwargs)

    def _fake_build_model(*args, **kwargs):
        _ = args, kwargs
        if status_stage_depth:
            raise AssertionError("Model loader ran inside a status envelope.")
        ready = torch.tensor([1], dtype=torch.int64)
        dist.all_reduce(ready)
        return build_fake_model_artifacts()

    def _build_metadata_with_rank_failure(*args, **kwargs):
        if mode == "metadata" and dist.get_rank() == 1:
            raise ValueError("intentional rank-one metadata build failure")
        return original_build_metadata(*args, **kwargs)

    def _collator_with_rank_failure(*args, **kwargs):
        if mode == "trainer_input" and dist.get_rank() == 1:
            raise ValueError("intentional rank-one trainer input failure")
        return original_collator(*args, **kwargs)

    _collator_with_rank_failure.SHAFT_INPUT_POLICY_VERSION = (  # type: ignore[attr-defined]
        original_collator.SHAFT_INPUT_POLICY_VERSION
    )

    failure_message: str | None = None
    with (
        patch(
            "shaft.pipeline.sft.build_model_tokenizer_processor",
            _fake_build_model,
        ),
        patch(
            "shaft.pipeline.sft.build_batching_run_metadata",
            _build_metadata_with_rank_failure,
        ),
        patch("shaft.pipeline.sft.SFTCollator", _collator_with_rank_failure),
        patch(
            "shaft.pipeline.sft.distributed_training_contract_stage",
            _tracked_status_stage,
        ),
        patch("shaft.algorithms.sft.ShaftSFTTrainer", _CollectiveTrainer),
    ):
        try:
            run_sft(config)
        except RuntimeError as exc:
            failure_message = str(exc)
            if expected_message not in failure_message:
                raise
        else:
            raise AssertionError(f"SFT {mode} rank-local failure was accepted.")

    if _CollectiveTrainer.constructed:
        raise AssertionError(f"SFT trainer constructor ran after {mode} setup failed.")
    peer_messages: list[str | None] = [None] * dist.get_world_size()
    dist.all_gather_object(peer_messages, failure_message)
    if any(message != peer_messages[0] for message in peer_messages[1:]):
        raise AssertionError(f"SFT {mode} failure differed by rank: {peer_messages!r}.")


def _expect_config_preflight_failure_before_efficiency_collective(root: Path) -> None:
    import shaft.pipeline.sft as sft_module

    config = _load_shared_sft_config(root / "config-preflight")
    original_initialize_runtime = sft_module.ShaftSFTPipeline.initialize_runtime
    invalidation_reached = False

    def _rank_local_initialize_runtime(self) -> None:
        if dist.get_rank() == 1:
            raise ValueError("intentional rank-one config preflight failure")
        original_initialize_runtime(self)

    def _record_invalidation(_output_dir) -> None:
        nonlocal invalidation_reached
        invalidation_reached = True

    pipeline = sft_module.ShaftSFTPipeline(config)
    failure_message: str | None = None
    try:
        with (
            patch.object(
                sft_module.ShaftSFTPipeline,
                "initialize_runtime",
                _rank_local_initialize_runtime,
            ),
            patch(
                "shaft.pipeline.sft.invalidate_training_efficiency_summary",
                _record_invalidation,
            ),
        ):
            try:
                pipeline.run()
            except RuntimeError as exc:
                failure_message = str(exc)
                if "intentional rank-one config preflight failure" not in failure_message:
                    raise
            else:
                raise AssertionError("Rank-local config preflight failure was accepted.")
    finally:
        pipeline.close()

    reached_by_rank: list[bool | None] = [None] * dist.get_world_size()
    dist.all_gather_object(reached_by_rank, invalidation_reached)
    if any(reached is not False for reached in reached_by_rank):
        raise AssertionError(
            "Efficiency invalidation ran before config-preflight consensus: "
            f"{reached_by_rank!r}."
        )
    peer_messages: list[str | None] = [None] * dist.get_world_size()
    dist.all_gather_object(peer_messages, failure_message)
    if any(message != peer_messages[0] for message in peer_messages[1:]):
        raise AssertionError(
            f"Config-preflight failure differed by rank: {peer_messages!r}."
        )


def _expect_collective_owning_model_and_trainer_are_not_nested(root: Path) -> None:
    import shaft.pipeline.sft as sft_module
    from shaft.pipeline import run_sft
    from tests.support.pipeline import FakePipelineTrainer, build_fake_model_artifacts

    config = _load_shared_sft_config(root / "collective-owners")
    original_stage = sft_module.distributed_training_contract_stage
    status_stage_depth = 0

    @contextmanager
    def _tracked_status_stage(*args, **kwargs):
        nonlocal status_stage_depth
        status_stage_depth += 1
        try:
            with original_stage(*args, **kwargs):
                yield
        finally:
            status_stage_depth -= 1

    def _fake_build_model(*args, **kwargs):
        _ = args, kwargs
        if status_stage_depth:
            raise AssertionError("Model loader ran inside a status envelope.")
        ready = torch.tensor([1], dtype=torch.int64)
        dist.all_reduce(ready)
        return build_fake_model_artifacts()

    class _CollectiveTrainer(FakePipelineTrainer):
        def __init__(self, **kwargs):
            if status_stage_depth:
                raise AssertionError("Trainer constructor ran inside a status envelope.")
            ready = torch.tensor([1], dtype=torch.int64)
            dist.all_reduce(ready)
            super().__init__(**kwargs)

    with (
        patch(
            "shaft.pipeline.sft.distributed_training_contract_stage",
            _tracked_status_stage,
        ),
        patch(
            "shaft.pipeline.sft.build_model_tokenizer_processor",
            _fake_build_model,
        ),
        patch("shaft.algorithms.sft.ShaftSFTTrainer", _CollectiveTrainer),
    ):
        metrics = run_sft(config)
    if "train_loss" not in metrics:
        raise AssertionError("Collective-owning SFT smoke did not complete training.")


def _expect_model_build_local_phase_failure(
    root: Path,
    *,
    phase: str,
) -> None:
    import shaft.model.builder as model_builder
    import shaft.pipeline.sft as sft_module
    from shaft.pipeline import run_sft
    from tests.support.pipeline import FakePipelineTrainer, build_fake_model_artifacts

    if phase not in {"prepare", "finalize"}:
        raise AssertionError(f"Unsupported model build phase: {phase!r}.")
    config = _load_shared_sft_config(root / f"model-{phase}-failure")
    original_prepare = model_builder.prepare_model_build
    original_finalize = model_builder.finalize_model_build
    original_stage = sft_module.distributed_training_contract_stage
    status_stage_depth = 0
    raw_loader_calls = 0

    @contextmanager
    def _tracked_status_stage(*args, **kwargs):
        nonlocal status_stage_depth
        status_stage_depth += 1
        try:
            with original_stage(*args, **kwargs):
                yield
        finally:
            status_stage_depth -= 1

    def _prepare_with_rank_failure(*args, **kwargs):
        if phase == "prepare" and dist.get_rank() == 1:
            raise ValueError("intentional rank-one model prepare failure")
        return original_prepare(*args, **kwargs)

    def _collective_raw_loader(prepared):
        nonlocal raw_loader_calls
        _ = prepared
        if status_stage_depth:
            raise AssertionError("Raw model loader ran inside a status envelope.")
        raw_loader_calls += 1
        ready = torch.tensor([1], dtype=torch.int64)
        dist.all_reduce(ready)
        return build_fake_model_artifacts()

    def _finalize_with_rank_failure(prepared, artifacts):
        if phase == "finalize" and dist.get_rank() == 1:
            raise ValueError("intentional rank-one model finalize failure")
        return original_finalize(prepared, artifacts)

    class _ForbiddenTrainer(FakePipelineTrainer):
        constructed = False

        def __init__(self, **kwargs):
            type(self).constructed = True
            super().__init__(**kwargs)

    expected_message = f"intentional rank-one model {phase} failure"
    failure_message: str | None = None
    with (
        patch(
            "shaft.pipeline.sft.distributed_training_contract_stage",
            _tracked_status_stage,
        ),
        patch(
            "shaft.model.builder.prepare_model_build",
            _prepare_with_rank_failure,
        ),
        patch(
            "shaft.model.builder.invoke_model_loader",
            _collective_raw_loader,
        ),
        patch(
            "shaft.model.builder.finalize_model_build",
            _finalize_with_rank_failure,
        ),
        patch("shaft.algorithms.sft.ShaftSFTTrainer", _ForbiddenTrainer),
    ):
        try:
            run_sft(config)
        except RuntimeError as exc:
            failure_message = str(exc)
            if expected_message not in failure_message:
                raise
        else:
            raise AssertionError(f"Rank-local model {phase} failure was accepted.")

    if _ForbiddenTrainer.constructed:
        raise AssertionError(f"Trainer ran after model {phase} failed.")
    raw_calls_by_rank: list[int | None] = [None] * dist.get_world_size()
    dist.all_gather_object(raw_calls_by_rank, raw_loader_calls)
    expected_raw_calls = 0 if phase == "prepare" else 1
    if any(count != expected_raw_calls for count in raw_calls_by_rank):
        raise AssertionError(
            f"Unexpected raw-loader calls after model {phase} failure: "
            f"{raw_calls_by_rank!r}."
        )
    peer_messages: list[str | None] = [None] * dist.get_world_size()
    dist.all_gather_object(peer_messages, failure_message)
    if any(message != peer_messages[0] for message in peer_messages[1:]):
        raise AssertionError(
            f"Model {phase} failure differed by rank: {peer_messages!r}."
        )


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: distributed_training_contract_drift.py OUTPUT_DIR")
    output_dir = Path(sys.argv[1])
    dist.init_process_group("gloo")
    try:
        rank = dist.get_rank()
        try:
            converge_training_contract_fingerprints(
                stage="test-pre-model",
                fingerprints={
                    "batch": "same-batch",
                    "model_plan": f"rank-{rank}-model-plan",
                },
            )
        except ValueError as exc:
            if "differs across ranks" not in str(exc):
                raise
        else:
            raise AssertionError("Rank-divergent training contracts were accepted.")

        failure_message: str | None = None
        try:
            with distributed_training_contract_stage(
                stage="test-builder-failure",
                fingerprints=lambda: {"training": "same-training"},
            ):
                if rank == 1:
                    raise ValueError("intentional rank-one builder failure")
        except RuntimeError as exc:
            failure_message = str(exc)
            if "intentional rank-one builder failure" not in failure_message:
                raise
        else:
            raise AssertionError("Rank-local builder failure was accepted.")

        peer_messages: list[str | None] = [None] * dist.get_world_size()
        dist.all_gather_object(peer_messages, failure_message)
        if any(message != peer_messages[0] for message in peer_messages[1:]):
            raise AssertionError(f"Distributed stage failure differed by rank: {peer_messages!r}.")
        _expect_sft_pipeline_setup_failure(
            output_dir,
            mode="metadata",
            expected_message="intentional rank-one metadata build failure",
        )
        _expect_sft_pipeline_setup_failure(
            output_dir,
            mode="trainer_input",
            expected_message="intentional rank-one trainer input failure",
        )
        _expect_config_preflight_failure_before_efficiency_collective(output_dir)
        _expect_collective_owning_model_and_trainer_are_not_nested(output_dir)
        _expect_model_build_local_phase_failure(output_dir, phase="prepare")
        _expect_model_build_local_phase_failure(output_dir, phase="finalize")
        dist.barrier()
        if rank == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "contract_drift_rejected.txt").write_text(
                "ok\n",
                encoding="utf-8",
            )
            (output_dir / "builder_failure_rejected.txt").write_text(
                "ok\n",
                encoding="utf-8",
            )
            (output_dir / "pipeline_setup_failure_rejected.txt").write_text(
                "ok\n",
                encoding="utf-8",
            )
            (output_dir / "collective_owner_boundaries_verified.txt").write_text(
                "ok\n",
                encoding="utf-8",
            )
            (output_dir / "model_build_phase_failures_converged.txt").write_text(
                "ok\n",
                encoding="utf-8",
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
