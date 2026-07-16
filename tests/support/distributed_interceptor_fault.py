from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import patch

import torch
import torch.distributed as dist

from shaft.plugins import interceptor
from shaft.plugins import ExecutionProxy
from shaft.plugins.interceptors import FunctionInterceptor, InterceptorManager
from shaft.pipeline.execution import prepare_pipeline_call


def _load_config(path: Path):
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
    config.data.media_snapshot_id = "distributed-interceptor-fixture-v1"
    config.data.num_workers = 0
    config.data.persistent_workers = False
    config.train.efficiency.enabled = False
    return config


def _expect_before_interceptor_failure(root: Path, *, algorithm: str) -> None:
    config = _load_config(root / algorithm)
    interceptor_name = f"distributed_{algorithm}_rank_one_before_failure"
    if algorithm == "sft":
        import shaft.pipeline.sft as pipeline_module

        entrypoint = pipeline_module.run_sft
        pipeline_class = pipeline_module.ShaftSFTPipeline
        point = "pipeline.sft.run"
        expected_stage = "sft-before-interceptors"
    elif algorithm == "rlhf":
        import shaft.pipeline.rlhf as pipeline_module

        config.algorithm.name = "dpo"
        entrypoint = pipeline_module.run_rlhf
        pipeline_class = pipeline_module.ShaftRLHFPipeline
        point = "pipeline.rlhf.run"
        expected_stage = "rlhf-before-interceptors"
    else:
        raise AssertionError(f"Unsupported test algorithm: {algorithm!r}.")

    @interceptor(point, phase="before", name=interceptor_name)
    def _fail_on_rank_one(_state: dict) -> None:
        if dist.get_rank() == 1:
            raise ValueError(f"intentional rank-one {algorithm} before interceptor failure")

    config.plugins.interceptors = [interceptor_name]
    body_reached = False

    def _collective_pipeline_body(_self) -> dict[str, float]:
        nonlocal body_reached
        body_reached = True
        ready = torch.tensor([1], dtype=torch.int64)
        dist.all_reduce(ready)
        return {"train_loss": 0.0}

    failure_message: str | None = None
    with patch.object(pipeline_class, "run", _collective_pipeline_body):
        try:
            entrypoint(config)
        except RuntimeError as exc:
            failure_message = str(exc)
            expected_failure = (
                f"intentional rank-one {algorithm} before interceptor failure"
            )
            if expected_stage not in failure_message or expected_failure not in failure_message:
                raise
        else:
            raise AssertionError(
                f"Rank-local {algorithm} before interceptor failure was accepted."
            )

    reached_by_rank: list[bool | None] = [None] * dist.get_world_size()
    dist.all_gather_object(reached_by_rank, body_reached)
    if any(reached is not False for reached in reached_by_rank):
        raise AssertionError(
            f"{algorithm} pipeline body ran before interceptor consensus: "
            f"{reached_by_rank!r}."
        )
    peer_messages: list[str | None] = [None] * dist.get_world_size()
    dist.all_gather_object(peer_messages, failure_message)
    if any(message != peer_messages[0] for message in peer_messages[1:]):
        raise AssertionError(
            f"{algorithm} before interceptor failure differed by rank: "
            f"{peer_messages!r}."
        )


def _expect_same_name_implementation_drift_before_body() -> None:
    if dist.get_rank() == 0:

        def _same_name(_state: dict) -> None:
            return None

    else:

        def _same_name(state: dict) -> None:
            state["rank_one_implementation"] = True

    manager = InterceptorManager(
        interceptors=[
            FunctionInterceptor(
                name="same-name-interceptor",
                point="pipeline.schedule-drift.run",
                phase="before",
                order=10,
                fn=_same_name,
                shaft_trajectory_neutral=True,
            )
        ]
    )
    body_calls = 0

    def _body() -> None:
        nonlocal body_calls
        body_calls += 1

    runner = ExecutionProxy(
        point="pipeline.schedule-drift.run",
        target=_body,
        interceptor_manager=manager,
    )
    try:
        invocation = prepare_pipeline_call(
            runner,
            stage="same-name-interceptor-schedule",
        )
        runner.invoke(invocation)
    except ValueError as exc:
        failure = str(exc)
    else:  # pragma: no cover - semantic schedule drift must be rejected
        raise AssertionError("same-name interceptor implementation drift was accepted")

    if "same-name-interceptor-schedule" not in failure:
        raise AssertionError(f"unexpected interceptor schedule failure: {failure!r}")
    peer_body_calls: list[int | None] = [None] * dist.get_world_size()
    dist.all_gather_object(peer_body_calls, body_calls)
    if any(count != 0 for count in peer_body_calls):
        raise AssertionError(
            "pipeline body ran before interceptor schedule consensus: "
            f"{peer_body_calls!r}"
        )
    dist.barrier()


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: distributed_interceptor_fault.py OUTPUT_DIR")
    output_dir = Path(sys.argv[1])
    dist.init_process_group("gloo")
    try:
        _expect_before_interceptor_failure(output_dir, algorithm="sft")
        _expect_before_interceptor_failure(output_dir, algorithm="rlhf")
        _expect_same_name_implementation_drift_before_body()
        dist.barrier()
        if dist.get_rank() == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "before_interceptor_failure_rejected.txt").write_text(
                "ok\n",
                encoding="utf-8",
            )
            (output_dir / "same_name_schedule_drift_rejected.txt").write_text(
                "ok\n",
                encoding="utf-8",
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
