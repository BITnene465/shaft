from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
import sys
from unittest.mock import patch

import torch
import torch.distributed as dist


def _load_config(root: Path, algorithm: str):
    from shaft.config import load_config
    from tests.support.rlhf import write_dpo_config, write_grpo_config, write_ppo_config

    config_names = {
        "dpo": "config_dpo.yaml",
        "grpo": "config_grpo.yaml",
        "ppo": "config_ppo.yaml",
    }
    if dist.get_rank() == 0:
        root.mkdir(parents=True, exist_ok=True)
        if algorithm == "dpo":
            path = write_dpo_config(root)
        elif algorithm == "grpo":
            path = write_grpo_config(root, sample_count=8)
        else:
            path = write_ppo_config(root)
        if algorithm in {"dpo", "ppo"}:
            train_path = root / f"train_{algorithm}.jsonl"
            row = train_path.read_text(encoding="utf-8")
            train_path.write_text(row * 8, encoding="utf-8")
        if path.name != config_names[algorithm]:
            raise AssertionError(f"Unexpected {algorithm} config path: {path}.")
    dist.barrier()
    config = load_config(root / config_names[algorithm])
    config.data.num_workers = 0
    config.data.persistent_workers = False
    if algorithm == "ppo":
        config.rlhf.ppo.reward_model_mode = "copy_backbone"
    return config


def _exercise_trainer_boundary(
    root: Path,
    algorithm: str,
    *,
    mode: str,
) -> None:
    import shaft.pipeline.rlhf as rlhf_module
    from shaft.algorithms import DPOAlgorithm, GRPOAlgorithm, PPOAlgorithm
    from shaft.pipeline import run_rlhf
    from tests.support.pipeline import FakePipelineTrainer, build_fake_model_artifacts

    algorithm_types = {
        "dpo": DPOAlgorithm,
        "grpo": GRPOAlgorithm,
        "ppo": PPOAlgorithm,
    }
    trainer_targets = {
        "dpo": "shaft.algorithms.dpo.ShaftDPOTrainer",
        "grpo": "shaft.algorithms.grpo.ShaftGRPOTrainer",
        "ppo": "shaft.algorithms.ppo.ShaftPPOTrainer",
    }
    if mode not in {
        "prepare-failure",
        "spec-drift",
        "implementation-drift",
        "constructor-boundary",
    }:
        raise AssertionError(f"Unsupported trainer-boundary mode: {mode!r}.")
    config = _load_config(root / mode / algorithm, algorithm)
    algorithm_type = algorithm_types[algorithm]
    original_prepare = algorithm_type.prepare_trainer
    original_stage = rlhf_module.distributed_training_contract_stage
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
        artifacts = build_fake_model_artifacts()
        artifacts.model.config.text_config.hidden_size = 1
        return artifacts

    def _prepare_with_rank_failure(self, *, context, **kwargs):
        spec = original_prepare(self, context=context, **kwargs)
        if mode == "prepare-failure" and dist.get_rank() == 1:
            raise ValueError(f"intentional rank-one {algorithm} trainer prepare failure")
        if mode == "spec-drift" and dist.get_rank() == 1:
            contract = dict(spec.contract)
            contract["test_rank_drift"] = 1
            return replace(spec, contract=contract)
        if mode == "implementation-drift" and dist.get_rank() == 1:
            if (
                _DivergentTrainer.__module__ != spec.trainer_cls.__module__
                or _DivergentTrainer.__qualname__ != spec.trainer_cls.__qualname__
            ):
                raise AssertionError("Trainer implementation-drift fixture changed its type name.")
            return replace(spec, trainer_cls=_DivergentTrainer)
        return spec

    class _CollectiveTrainer(FakePipelineTrainer):
        constructed = False

        def __init__(self, **kwargs):
            type(self).constructed = True
            if status_stage_depth:
                raise AssertionError("RLHF Trainer constructor ran inside a status envelope.")
            ready = torch.tensor([1], dtype=torch.int64)
            dist.all_reduce(ready)
            super().__init__(**kwargs)

    class _DivergentTrainer(_CollectiveTrainer):
        def __init__(self, **kwargs):
            raise AssertionError(
                "Rank-local same-name Trainer replacement reached its constructor."
            )

    _DivergentTrainer.__module__ = _CollectiveTrainer.__module__
    _DivergentTrainer.__qualname__ = _CollectiveTrainer.__qualname__

    expected_message = (
        f"intentional rank-one {algorithm} trainer prepare failure"
        if mode == "prepare-failure"
        else "Distributed trainer-prepare contract differs across ranks"
    )
    failure_message: str | None = None
    metrics = None
    with (
        patch(
            "shaft.pipeline.rlhf.distributed_training_contract_stage",
            _tracked_status_stage,
        ),
        patch(
            "shaft.pipeline.rlhf.build_model_tokenizer_processor",
            _fake_build_model,
        ),
        patch.object(algorithm_type, "prepare_trainer", _prepare_with_rank_failure),
        patch(trainer_targets[algorithm], _CollectiveTrainer),
    ):
        try:
            metrics = run_rlhf(config)
        except (RuntimeError, ValueError) as exc:
            if mode == "constructor-boundary":
                raise
            failure_message = str(exc)
            if expected_message not in failure_message:
                raise
        else:
            if mode != "constructor-boundary":
                raise AssertionError(f"{algorithm} trainer {mode} fault was accepted.")

    constructed_by_rank: list[bool | None] = [None] * dist.get_world_size()
    dist.all_gather_object(constructed_by_rank, _CollectiveTrainer.constructed)
    expected_constructed = mode == "constructor-boundary"
    if any(constructed is not expected_constructed for constructed in constructed_by_rank):
        raise AssertionError(
            f"Unexpected {algorithm} Trainer construction state: {constructed_by_rank!r}."
        )
    if mode == "constructor-boundary" and "train_loss" not in dict(metrics or {}):
        raise AssertionError(f"Collective-owning {algorithm} Trainer smoke did not train.")
    peer_messages: list[str | None] = [None] * dist.get_world_size()
    dist.all_gather_object(peer_messages, failure_message)
    if any(message != peer_messages[0] for message in peer_messages[1:]):
        raise AssertionError(
            f"{algorithm} trainer prepare failure differed by rank: {peer_messages!r}."
        )


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: distributed_rlhf_trainer_prepare_fault.py OUTPUT_DIR")
    output_dir = Path(sys.argv[1])
    dist.init_process_group("gloo")
    try:
        for algorithm in ("dpo", "grpo", "ppo"):
            _exercise_trainer_boundary(
                output_dir,
                algorithm,
                mode="prepare-failure",
            )
        _exercise_trainer_boundary(output_dir, "ppo", mode="spec-drift")
        _exercise_trainer_boundary(output_dir, "ppo", mode="implementation-drift")
        for algorithm in ("dpo", "grpo", "ppo"):
            _exercise_trainer_boundary(
                output_dir,
                algorithm,
                mode="constructor-boundary",
            )
        dist.barrier()
        if dist.get_rank() == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "rlhf_trainer_boundaries_verified.txt").write_text(
                "ok\n",
                encoding="utf-8",
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
