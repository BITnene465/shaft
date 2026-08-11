from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
from urllib.request import urlopen

import numpy as np
import pytest
import torch
import yaml
from torch.distributed.tensor import DTensor
from PIL import Image
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor, GenerationConfig
from peft import PeftModel, get_peft_model_state_dict

from shaft.export import validate_hf_artifact
from shaft.infer import (
    InferEngineConfig,
    InferGenerationConfig,
    ShaftInferEngine,
    ShaftInferRequest,
)
from shaft.data import (
    DPOCollator,
    SFTCollator,
    SFTDataset,
    SFTRecord,
    ShaftBatchPlanner,
    ShaftBatchPlanningSpec,
    ShaftSamplePlan,
    ShaftSFTSampleCostProvider,
)
from shaft.model import MODEL_REGISTRY, build_model_meta
from shaft.observability import TRAINING_EFFICIENCY_SCHEMA_VERSION
from shaft.template import ShaftChatRenderer, build_template
from shaft.training import (
    checkpoint_has_batch_planning_state,
    validate_training_checkpoint_commit,
)
from tests.support.qwen_training_gate import (
    prepare_qwen_training_dataset,
    prepare_tiny_qwen3vl_artifact,
    prepare_tiny_qwen3vl_moe_training_assets,
    prepare_tiny_qwen35_training_assets,
    write_qwen_training_gate_config,
)
from tests.support.opd import write_qwen3vl_opd_config


class _CountingProcessor:
    def __init__(self, wrapped) -> None:
        self.wrapped = wrapped
        self.tokenizer = wrapped.tokenizer
        self.call_count = 0

    def __getattr__(self, name):
        return getattr(self.wrapped, name)

    def __call__(self, *args, **kwargs):
        self.call_count += 1
        return self.wrapped(*args, **kwargs)


def _terminate_qwen_training_gate(process: subprocess.Popen[str]) -> tuple[str, str]:
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        return process.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return process.communicate(timeout=30)


def _run_qwen_training_gate(
    repo_root: Path,
    config_path: Path,
    *,
    cpu_only: bool = False,
    timeout_seconds: int = 600,
    world_size: int = 2,
) -> None:
    if int(world_size) <= 0:
        raise ValueError("Qwen training gate world_size must be > 0.")
    env = {**os.environ, "OMP_NUM_THREADS": "1"}
    if cpu_only:
        env["CUDA_VISIBLE_DEVICES"] = ""
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            f"--nproc_per_node={int(world_size)}",
            "scripts/train.py",
            "sft",
            "--config",
            str(config_path),
        ],
        cwd=repo_root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        stdout, stderr = _terminate_qwen_training_gate(process)
        raise AssertionError(
            "Qwen training release gate timed out and its process group was terminated.\n"
            f"stdout:\n{stdout}\nstderr:\n{stderr}"
        ) from exc
    except BaseException:
        _terminate_qwen_training_gate(process)
        raise
    if process.returncode != 0:
        raise AssertionError(
            f"Qwen training release gate failed.\nstdout:\n{stdout}\nstderr:\n{stderr}"
        )


def _run_qwen_opd_cpu_gate(repo_root: Path, config_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/train.py",
            "opd",
            "--config",
            str(config_path),
        ],
        cwd=repo_root,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"},
        text=True,
        capture_output=True,
        timeout=600,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(
            "Qwen3VL OPD CPU gate failed.\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )


def _run_qwen_opd_cuda_gate(
    repo_root: Path,
    config_path: Path,
    *,
    world_size: int,
    timeout_seconds: int = 1200,
) -> None:
    if int(world_size) not in {1, 2}:
        raise ValueError("The Qwen3VL OPD release gate supports world_size 1 or 2.")
    raw_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw_visible_devices is None:
        visible_devices = [str(index) for index in range(torch.cuda.device_count())]
    else:
        visible_devices = [
            device.strip()
            for device in raw_visible_devices.split(",")
            if device.strip()
        ]
    if len(visible_devices) < int(world_size):
        raise RuntimeError(
            "The Qwen3VL OPD CUDA gate does not have enough visible devices: "
            f"requested={world_size}, visible={visible_devices}."
        )
    command = [sys.executable]
    if int(world_size) == 2:
        command.extend(
            (
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nnodes=1",
                "--nproc_per_node=2",
            )
        )
    command.extend(("scripts/train.py", "opd", "--config", str(config_path)))
    process = subprocess.Popen(
        command,
        cwd=repo_root,
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": ",".join(visible_devices[: int(world_size)]),
            "OMP_NUM_THREADS": "1",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=int(timeout_seconds))
    except subprocess.TimeoutExpired as exc:
        stdout, stderr = _terminate_qwen_training_gate(process)
        raise AssertionError(
            "Qwen3VL OPD CUDA gate timed out and its process group was terminated.\n"
            f"stdout:\n{stdout}\nstderr:\n{stderr}"
        ) from exc
    except BaseException:
        _terminate_qwen_training_gate(process)
        raise
    if process.returncode != 0:
        raise AssertionError(
            "Qwen3VL OPD CUDA gate failed.\n"
            f"stdout:\n{stdout}\nstderr:\n{stderr}"
        )


def _assert_nested_state_equal(left, right, *, path: str = "root") -> None:
    if isinstance(left, DTensor) or isinstance(right, DTensor):
        assert isinstance(left, DTensor) and isinstance(right, DTensor), path
        assert left.dtype == right.dtype, path
        assert tuple(left.shape) == tuple(right.shape), path
        assert tuple(left.stride()) == tuple(right.stride()), path
        assert left.placements == right.placements, path
        assert left.device_mesh.device_type == right.device_mesh.device_type, path
        assert torch.equal(left.device_mesh.mesh, right.device_mesh.mesh), path
        _assert_nested_state_equal(
            left.to_local(),
            right.to_local(),
            path=f"{path}.local",
        )
        return
    if torch.is_tensor(left) or torch.is_tensor(right):
        assert torch.is_tensor(left) and torch.is_tensor(right), path
        assert left.dtype == right.dtype, path
        assert tuple(left.shape) == tuple(right.shape), path
        assert torch.equal(left.cpu(), right.cpu()), path
        return
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        assert isinstance(left, np.ndarray) and isinstance(right, np.ndarray), path
        np.testing.assert_array_equal(left, right, err_msg=path)
        return
    if isinstance(left, dict) or isinstance(right, dict):
        assert isinstance(left, dict) and isinstance(right, dict), path
        assert set(left) == set(right), path
        for key in sorted(left, key=str):
            _assert_nested_state_equal(left[key], right[key], path=f"{path}.{key}")
        return
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        assert isinstance(left, (list, tuple)) and isinstance(
            right,
            (list, tuple),
        ), path
        assert len(left) == len(right), path
        for index, (left_value, right_value) in enumerate(zip(left, right, strict=True)):
            _assert_nested_state_equal(
                left_value,
                right_value,
                path=f"{path}[{index}]",
            )
        return
    if isinstance(getattr(left, "__dict__", None), dict) or isinstance(
        getattr(right, "__dict__", None),
        dict,
    ):
        assert type(left) is type(right), path
        assert isinstance(getattr(left, "__dict__", None), dict), path
        assert isinstance(getattr(right, "__dict__", None), dict), path
        _assert_nested_state_equal(
            left.__dict__,
            right.__dict__,
            path=f"{path}.__dict__",
        )
        return
    assert left == right, path


def _normalized_trainer_state(checkpoint_dir: Path) -> dict:
    payload = json.loads((checkpoint_dir / "trainer_state.json").read_text(encoding="utf-8"))
    best_checkpoint = payload.get("best_model_checkpoint")
    if best_checkpoint:
        payload["best_model_checkpoint"] = Path(best_checkpoint).name
    timing_keys = (
        "runtime",
        "samples_per_second",
        "steps_per_second",
        "jit_compilation_time",
    )
    payload["log_history"] = [
        {
            key: value
            for key, value in entry.items()
            if not key.startswith("efficiency/")
            and not any(key.endswith(suffix) for suffix in timing_keys)
        }
        for entry in payload.get("log_history", [])
    ]
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_checkpoint_state_equivalent(
    fresh_checkpoint: Path,
    resumed_checkpoint: Path,
    *,
    weight_filename: str,
    efficiency_expected: bool = True,
) -> None:
    fresh_commit = validate_training_checkpoint_commit(fresh_checkpoint)
    resumed_commit = validate_training_checkpoint_commit(resumed_checkpoint)
    fresh_planning = fresh_commit["extensions"].get("batch_planning")
    resumed_planning = resumed_commit["extensions"].get("batch_planning")
    assert (fresh_planning is not None) == (resumed_planning is not None)
    if fresh_planning is not None:
        assert checkpoint_has_batch_planning_state(fresh_checkpoint)
        assert checkpoint_has_batch_planning_state(resumed_checkpoint)
        assert fresh_planning == resumed_planning
    else:
        assert not checkpoint_has_batch_planning_state(fresh_checkpoint)
        assert not checkpoint_has_batch_planning_state(resumed_checkpoint)
    assert _file_sha256(fresh_checkpoint / weight_filename) == _file_sha256(
        resumed_checkpoint / weight_filename
    )

    for filename in ("optimizer.pt", "scheduler.pt"):
        _assert_nested_state_equal(
            torch.load(fresh_checkpoint / filename, map_location="cpu", weights_only=True),
            torch.load(resumed_checkpoint / filename, map_location="cpu", weights_only=True),
            path=filename,
        )
    fresh_scaler = fresh_checkpoint / "scaler.pt"
    resumed_scaler = resumed_checkpoint / "scaler.pt"
    assert fresh_scaler.is_file() == resumed_scaler.is_file()
    if fresh_scaler.is_file():
        _assert_nested_state_equal(
            torch.load(fresh_scaler, map_location="cpu", weights_only=False),
            torch.load(resumed_scaler, map_location="cpu", weights_only=False),
            path="scaler.pt",
        )
    fresh_rng_files = tuple(sorted(path.name for path in fresh_checkpoint.glob("rng_state*.pth")))
    resumed_rng_files = tuple(
        sorted(path.name for path in resumed_checkpoint.glob("rng_state*.pth"))
    )
    assert fresh_rng_files == resumed_rng_files
    assert fresh_rng_files
    for filename in fresh_rng_files:
        _assert_nested_state_equal(
            torch.load(fresh_checkpoint / filename, map_location="cpu", weights_only=False),
            torch.load(resumed_checkpoint / filename, map_location="cpu", weights_only=False),
            path=filename,
        )
    assert _normalized_trainer_state(fresh_checkpoint) == _normalized_trainer_state(
        resumed_checkpoint
    )
    if efficiency_expected:
        assert _restore_efficiency_snapshot_set(fresh_checkpoint) == (
            _restore_efficiency_snapshot_set(resumed_checkpoint)
        )
    else:
        for checkpoint in (fresh_checkpoint, resumed_checkpoint):
            assert not tuple(checkpoint.glob("shaft_training_efficiency_rank*.json"))
            assert not (checkpoint / "shaft_training_efficiency_snapshot_set.json").exists()


def _assert_backend_native_checkpoint_state_equivalent(
    fresh_checkpoint: Path,
    resumed_checkpoint: Path,
    *,
    distributed_strategy: str,
    weight_filename: str,
) -> None:
    strategy = str(distributed_strategy).strip().lower()
    assert strategy in {"fsdp", "deepspeed"}
    assert _file_sha256(fresh_checkpoint / weight_filename) == _file_sha256(
        resumed_checkpoint / weight_filename
    )
    assert _normalized_trainer_state(fresh_checkpoint) == _normalized_trainer_state(
        resumed_checkpoint
    )

    for filename in ("scheduler.pt", "rng_state_0.pth", "rng_state_1.pth"):
        _assert_nested_state_equal(
            torch.load(
                fresh_checkpoint / filename,
                map_location="cpu",
                weights_only=False,
            ),
            torch.load(
                resumed_checkpoint / filename,
                map_location="cpu",
                weights_only=False,
            ),
            path=filename,
        )
    fresh_scaler = fresh_checkpoint / "scaler.pt"
    resumed_scaler = resumed_checkpoint / "scaler.pt"
    assert fresh_scaler.is_file() == resumed_scaler.is_file()
    if fresh_scaler.is_file():
        _assert_nested_state_equal(
            torch.load(fresh_scaler, map_location="cpu", weights_only=False),
            torch.load(resumed_scaler, map_location="cpu", weights_only=False),
            path="scaler.pt",
        )
    assert _restore_efficiency_snapshot_set(fresh_checkpoint) == (
        _restore_efficiency_snapshot_set(resumed_checkpoint)
    )

    if strategy == "fsdp":
        for checkpoint in (fresh_checkpoint, resumed_checkpoint):
            assert (checkpoint / "pytorch_model_fsdp.bin").is_file()
            assert (checkpoint / "optimizer.bin").is_file()
        if weight_filename != "adapter_model.safetensors":
            _assert_nested_state_equal(
                torch.load(
                    fresh_checkpoint / "pytorch_model_fsdp.bin",
                    map_location="cpu",
                    weights_only=False,
                ),
                torch.load(
                    resumed_checkpoint / "pytorch_model_fsdp.bin",
                    map_location="cpu",
                    weights_only=False,
                ),
                path="pytorch_model_fsdp.bin",
            )
        _assert_nested_state_equal(
            torch.load(
                fresh_checkpoint / "optimizer.bin",
                map_location="cpu",
                weights_only=False,
            ),
            torch.load(
                resumed_checkpoint / "optimizer.bin",
                map_location="cpu",
                weights_only=False,
            ),
            path="optimizer.bin",
        )
        return

    fresh_generation = (fresh_checkpoint / "latest").read_text(encoding="utf-8").strip()
    resumed_generation = (resumed_checkpoint / "latest").read_text(encoding="utf-8").strip()
    assert fresh_generation == resumed_generation
    fresh_shard_dir = fresh_checkpoint / fresh_generation
    resumed_shard_dir = resumed_checkpoint / resumed_generation
    fresh_model_shards = tuple(
        sorted(path.name for path in fresh_shard_dir.glob("*_model_states.pt"))
    )
    fresh_optimizer_shards = tuple(
        sorted(path.name for path in fresh_shard_dir.glob("*_optim_states.pt"))
    )
    resumed_model_shards = tuple(
        sorted(path.name for path in resumed_shard_dir.glob("*_model_states.pt"))
    )
    resumed_optimizer_shards = tuple(
        sorted(path.name for path in resumed_shard_dir.glob("*_optim_states.pt"))
    )
    assert len(fresh_model_shards) == 2
    assert len(fresh_optimizer_shards) == 2
    assert fresh_model_shards == resumed_model_shards
    assert fresh_optimizer_shards == resumed_optimizer_shards

    model_state_keys = (
        "module",
        "buffer_names",
        "param_shapes",
        "frozen_param_shapes",
        "shared_params",
        "frozen_param_fragments",
        "lr_scheduler",
        "skipped_steps",
        "global_steps",
        "global_samples",
        "dp_world_size",
        "mp_world_size",
    )
    for filename in fresh_model_shards:
        fresh_state = torch.load(
            fresh_shard_dir / filename,
            map_location="cpu",
            weights_only=False,
        )
        resumed_state = torch.load(
            resumed_shard_dir / filename,
            map_location="cpu",
            weights_only=False,
        )
        for key in model_state_keys:
            assert key in fresh_state and key in resumed_state, (filename, key)
            _assert_nested_state_equal(
                fresh_state[key],
                resumed_state[key],
                path=f"{filename}.{key}",
            )
    for filename in fresh_optimizer_shards:
        fresh_state = torch.load(
            fresh_shard_dir / filename,
            map_location="cpu",
            weights_only=False,
        )
        resumed_state = torch.load(
            resumed_shard_dir / filename,
            map_location="cpu",
            weights_only=False,
        )
        _assert_nested_state_equal(
            fresh_state["optimizer_state_dict"],
            resumed_state["optimizer_state_dict"],
            path=f"{filename}.optimizer_state_dict",
        )


def _assert_lora_adapter_has_learned_update(adapter_path: Path) -> None:
    with safe_open(adapter_path, framework="pt", device="cpu") as tensors:
        lora_b_names = [name for name in tensors.keys() if "lora_B" in name]
        assert lora_b_names
        assert any(bool(torch.count_nonzero(tensors.get_tensor(name))) for name in lora_b_names)


def _assert_full_moe_router_and_experts_updated(
    base_weights: Path,
    trained_weights: Path,
) -> None:
    roles = {
        "router": (".mlp.gate.weight",),
        "routed_gate": (".mlp.experts.", ".gate_proj.weight"),
        "routed_up": (".mlp.experts.", ".up_proj.weight"),
        "routed_down": (".mlp.experts.", ".down_proj.weight"),
    }
    with safe_open(base_weights, framework="pt", device="cpu") as base_handle:
        base_keys = tuple(base_handle.keys())
        with safe_open(trained_weights, framework="pt", device="cpu") as trained_handle:
            trained_keys = set(trained_handle.keys())
            for role, markers in roles.items():
                matching = [key for key in base_keys if all(marker in key for marker in markers)]
                assert matching, role
                assert all(key in trained_keys for key in matching), role
                assert any(
                    not torch.equal(
                        base_handle.get_tensor(key),
                        trained_handle.get_tensor(key),
                    )
                    for key in matching
                ), role


def _assert_fused_moe_router_and_experts_updated(
    base_weights: Path,
    trained_weights: Path,
) -> None:
    roles = {
        "router": (".mlp.gate.weight",),
        "fused_gate_up": (".mlp.experts.gate_up_proj",),
        "fused_down": (".mlp.experts.down_proj",),
    }
    with safe_open(base_weights, framework="pt", device="cpu") as base_handle:
        base_keys = tuple(base_handle.keys())
        with safe_open(trained_weights, framework="pt", device="cpu") as trained_handle:
            trained_keys = set(trained_handle.keys())
            for role, markers in roles.items():
                matching = [key for key in base_keys if all(marker in key for marker in markers)]
                assert matching, role
                assert all(key in trained_keys for key in matching), role
                assert any(
                    not torch.equal(
                        base_handle.get_tensor(key),
                        trained_handle.get_tensor(key),
                    )
                    for key in matching
                ), role


def _assert_router_auxiliary_loss_was_logged(
    checkpoint: Path,
    *,
    expected_coefficient: float | None = None,
) -> None:
    history = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))[
        "log_history"
    ]
    values = [
        float(entry["aux/router_aux_loss"]) for entry in history if "aux/router_aux_loss" in entry
    ]
    weighted = [
        float(entry["aux/router_aux_loss_weighted"])
        for entry in history
        if "aux/router_aux_loss_weighted" in entry
    ]
    assert values and weighted
    assert all(np.isfinite(value) and value > 0.0 for value in values)
    assert all(np.isfinite(value) and value > 0.0 for value in weighted)
    if expected_coefficient is not None:
        assert len(values) == len(weighted)
        for raw_value, weighted_value in zip(values, weighted, strict=True):
            assert weighted_value == pytest.approx(
                raw_value * float(expected_coefficient),
                rel=1e-5,
                abs=1e-8,
            )


def _assert_finite_training_metrics(checkpoint: Path) -> None:
    history = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))[
        "log_history"
    ]
    for name in ("loss", "grad_norm"):
        values = [float(entry[name]) for entry in history if name in entry]
        assert values, name
        assert all(np.isfinite(value) for value in values), name


def _assert_resumed_root_train_loss_matches_global_checkpoint_window(
    *,
    start_checkpoint: Path,
    end_checkpoint: Path,
    resumed_output: Path,
) -> None:
    def cumulative_rank_losses(checkpoint: Path) -> list[float]:
        state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
        callback = state["stateful_callbacks"]["ShaftSFTReportingStateCallback"]
        snapshot = callback["attributes"]["snapshot"]
        return [
            float(rank["total_loss_scalar"]) + float(rank["tr_loss"]) for rank in snapshot["ranks"]
        ]

    start = cumulative_rank_losses(start_checkpoint)
    end = cumulative_rank_losses(end_checkpoint)
    assert len(start) == len(end) == 2
    assert start[0] != start[1], "The gate must expose rank-local pending loss skew."
    start_step = int(start_checkpoint.name.removeprefix("checkpoint-"))
    end_step = int(end_checkpoint.name.removeprefix("checkpoint-"))
    expected = (float(np.mean(end)) - float(np.mean(start))) / (end_step - start_step)
    root_state = json.loads((resumed_output / "trainer_state.json").read_text(encoding="utf-8"))
    final_entry = next(
        entry for entry in reversed(root_state["log_history"]) if "train_loss" in entry
    )
    assert float(final_entry["train_loss"]) == pytest.approx(expected)


def _assert_moe_lora_router_and_experts_updated(adapter_path: Path) -> None:
    markers = {
        "router": ".mlp.gate.lora_B.",
        "routed_expert_parameter": ".mlp.experts.lora_B.",
        "nested_routed_expert_parameter": ".mlp.experts.base_layer.lora_B.",
    }
    with safe_open(adapter_path, framework="pt", device="cpu") as tensors:
        keys = tuple(tensors.keys())
        for role, marker in markers.items():
            matching = [name for name in keys if marker in name and "lora_B" in name]
            assert matching, role
            assert any(bool(torch.count_nonzero(tensors.get_tensor(name))) for name in matching), (
                role
            )
        ordinary_modules = [
            name
            for name in keys
            if "lora_B" in name and ".mlp.gate.lora_B." not in name and ".mlp.experts" not in name
        ]
        assert ordinary_modules
        assert any(bool(torch.count_nonzero(tensors.get_tensor(name))) for name in ordinary_modules)

    adapter_config = json.loads(
        (adapter_path.parent / "adapter_config.json").read_text(encoding="utf-8")
    )
    target_parameters = tuple(adapter_config.get("target_parameters") or ())
    for suffix in (
        "mlp.experts.gate_up_proj",
        "mlp.experts.down_proj",
        "mlp.gate.weight",
    ):
        assert any(name.endswith(suffix) for name in target_parameters), suffix


def _assert_qwen3vl_moe_lora_roles_updated(
    adapter_path: Path,
    *,
    language_layers: int,
    vision_blocks: int,
) -> None:
    _assert_moe_lora_router_and_experts_updated(adapter_path)
    with safe_open(adapter_path, framework="pt", device="cpu") as tensors:
        keys = tuple(tensors.keys())
        lora_b_keys = tuple(name for name in keys if "lora_B" in name)
        assert lora_b_keys
        assert all(bool(torch.count_nonzero(tensors.get_tensor(name))) for name in lora_b_keys)
        for layer_index in range(language_layers):
            layer_marker = f".language_model.layers.{layer_index}."
            for role, marker in {
                "router": ".mlp.gate.lora_B.",
                "fused_gate_up": ".mlp.experts.lora_B.",
                "fused_down": ".mlp.experts.base_layer.lora_B.",
                "language_attention": ".self_attn.",
            }.items():
                matching = [name for name in lora_b_keys if layer_marker in name and marker in name]
                assert matching, (layer_index, role)
        for block_index in range(vision_blocks):
            assert any(f".visual.blocks.{block_index}." in name for name in lora_b_keys), (
                block_index,
                "vision_tower",
            )


def _assert_adapter_tensors_changed(before_path: Path, after_path: Path) -> None:
    with (
        safe_open(before_path, framework="pt", device="cpu") as before,
        safe_open(after_path, framework="pt", device="cpu") as after,
    ):
        before_keys = tuple(before.keys())
        assert tuple(after.keys()) == before_keys
        unchanged = [
            name
            for name in before_keys
            if torch.equal(before.get_tensor(name), after.get_tensor(name))
        ]
    assert not unchanged, unchanged[:8]


def _restore_efficiency_snapshot_set(checkpoint_dir: Path) -> dict:
    global_step = int(checkpoint_dir.name.rsplit("-", 1)[-1])
    transaction_path = (
        checkpoint_dir / "shaft_training_efficiency_checkpoint_transaction.json"
    )
    transaction = json.loads(transaction_path.read_text(encoding="utf-8"))
    world_size = int(transaction.get("world_size", 0))
    if world_size <= 0:
        raise AssertionError(
            "Efficiency checkpoint transaction has an invalid world_size: "
            f"{transaction!r}."
        )
    with tempfile.TemporaryDirectory(prefix="shaft-efficiency-restore-") as directory:
        output_path = Path(directory) / "restored.json"
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nnodes=1",
                f"--nproc_per_node={world_size}",
                "tests/support/distributed_efficiency_checkpoint_validate.py",
                str(checkpoint_dir),
                str(global_step),
                str(output_path),
            ],
            cwd=Path(__file__).resolve().parents[1],
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"},
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
        )
        if completed.returncode != 0:
            raise AssertionError(
                "Efficiency checkpoint restore validation failed.\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
        return json.loads(output_path.read_text(encoding="utf-8"))


def _assert_full_hf_export_reloads(
    export_dir: Path,
    *,
    source_model_dir: Path,
    expected_type: str,
    device: str | torch.device | None = None,
) -> None:
    processor = AutoProcessor.from_pretrained(
        export_dir,
        local_files_only=True,
        trust_remote_code=True,
        fix_mistral_regex=False,
    )
    selected_device = device
    if selected_device is None:
        selected_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    resolved_device = torch.device(selected_device)
    load_kwargs = {
        "local_files_only": True,
        "dtype": torch.bfloat16 if resolved_device.type == "cuda" else torch.float32,
    }
    if resolved_device.type == "cuda":
        load_kwargs["device_map"] = {"": resolved_device.index or 0}
    model = AutoModelForImageTextToText.from_pretrained(export_dir, **load_kwargs)
    assert type(model).__name__ == expected_type
    source_config = AutoConfig.from_pretrained(source_model_dir, local_files_only=True)
    assert hasattr(model.config, "use_cache") == hasattr(source_config, "use_cache")
    if hasattr(source_config, "use_cache"):
        assert model.config.use_cache == source_config.use_cache
    text_config = getattr(model.config, "text_config", None)
    source_text_config = getattr(source_config, "text_config", None)
    assert text_config is not None and source_text_config is not None
    assert text_config.use_cache == source_text_config.use_cache
    source_generation_config = GenerationConfig.from_pretrained(
        source_model_dir,
        local_files_only=True,
    )
    assert model.generation_config.use_cache == source_generation_config.use_cache
    assert next(model.parameters()).device.type == resolved_device.type
    inputs = {
        key: value.to(resolved_device) if torch.is_tensor(value) else value
        for key, value in processor(text=["hello"], return_tensors="pt").items()
    }
    with torch.no_grad():
        logits = model(**inputs).logits
    assert torch.isfinite(logits).all()
    del logits, inputs, model, processor
    if resolved_device.type == "cuda":
        torch.cuda.empty_cache()


def _assert_standard_qwen_peft_export_reloads(
    *,
    base_model_path: Path,
    export_dir: Path,
    experts_implementation: str | None = None,
    expected_router_layers: int | None = None,
    expected_num_experts: int | None = None,
    expected_dtype: torch.dtype | None = None,
    image_path: Path | None = None,
) -> None:
    processor = AutoProcessor.from_pretrained(
        export_dir,
        local_files_only=True,
        trust_remote_code=True,
        fix_mistral_regex=False,
    )
    load_kwargs = {
        "local_files_only": True,
        "dtype": torch.bfloat16,
        "device_map": {"": 0},
    }
    if experts_implementation is not None:
        load_kwargs["experts_implementation"] = experts_implementation
    base_model = AutoModelForImageTextToText.from_pretrained(
        base_model_path,
        **load_kwargs,
    )
    model = PeftModel.from_pretrained(base_model, export_dir, is_trainable=False)
    loaded_adapter_state = get_peft_model_state_dict(
        model,
        adapter_name="default",
    )
    with safe_open(
        export_dir / "adapter_model.safetensors",
        framework="pt",
        device="cpu",
    ) as saved_adapter:
        saved_keys = set(saved_adapter.keys())
        assert set(loaded_adapter_state) == saved_keys
        for name in sorted(saved_keys):
            torch.testing.assert_close(
                loaded_adapter_state[name].detach().cpu(),
                saved_adapter.get_tensor(name),
                rtol=0.0,
                atol=0.0,
            )
    if image_path is None:
        processed = processor(text=["hello"], return_tensors="pt")
    else:
        rendered = processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Describe the image briefly."},
                    ],
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        with Image.open(image_path) as opened:
            processed = processor(
                text=[rendered],
                images=[opened.convert("RGB")],
                return_tensors="pt",
            )
    inputs = {
        key: value.to("cuda:0") if torch.is_tensor(value) else value
        for key, value in processed.items()
    }
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_router_logits=expected_router_layers is not None,
            use_cache=False,
        )
        logits = outputs.logits
    assert torch.isfinite(logits).all()
    if expected_dtype is not None:
        assert logits.dtype == expected_dtype
    if expected_router_layers is not None:
        assert expected_num_experts is not None
        assert isinstance(outputs.router_logits, tuple)
        assert len(outputs.router_logits) == expected_router_layers
        assert all(
            tensor.ndim == 2
            and int(tensor.shape[-1]) == expected_num_experts
            and torch.isfinite(tensor).all()
            for tensor in outputs.router_logits
        )
        assert torch.is_tensor(outputs.aux_loss)
        assert torch.isfinite(outputs.aux_loss)
    if experts_implementation is not None:
        assert model.base_model.model.config._experts_implementation == experts_implementation
        assert (
            model.base_model.model.config.text_config._experts_implementation
            == experts_implementation
        )
    del (
        logits,
        outputs,
        inputs,
        processed,
        loaded_adapter_state,
        model,
        base_model,
        processor,
    )
    torch.cuda.empty_cache()


def _validate_qwen_peft_export(
    repo_root: Path,
    *,
    export_path: Path,
    model_type: str,
    model_path: Path,
) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/export.py",
            "validate",
            "--path",
            str(export_path),
            "--finetune-mode",
            "lora",
            "--model-type",
            model_type,
            "--model-name-or-path",
            str(model_path),
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(
            "Qwen PEFT export validation failed.\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )


@pytest.mark.integration
@pytest.mark.parametrize(
    ("model_type", "template_name", "model_path"),
    [
        ("qwen3vl", "qwen3vl", Path("models/Qwen3-VL-4B-Instruct")),
        ("qwen36vl", "qwen35vl", Path("models/Qwen3.6-27B")),
    ],
)
def test_qwen_vl_runtime_cost_matches_real_processor_and_collator(
    tmp_path: Path,
    model_type: str,
    template_name: str,
    model_path: Path,
) -> None:
    if not model_path.exists():
        pytest.skip(f"Model path not found: {model_path}")
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        fix_mistral_regex=False,
    )
    tokenizer = processor.tokenizer
    model_adapter = build_model_meta(model_type).resolve_adapter(model_name_or_path=str(model_path))
    template = build_template(template_name)

    for index, image_size in enumerate(((64, 64), (128, 512))):
        image_path = tmp_path / f"cost-{index}.png"
        Image.new("RGB", image_size, color=(index * 20, 30, 40)).save(image_path)
        messages = None
        if index == 1:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "first question"},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": "first answer"}]},
                {"role": "user", "content": [{"type": "text", "text": "second question"}]},
            ]
        plan = ShaftSamplePlan(
            {"integration": 1},
            {"integration": 1.0},
            strategy="concat",
            shuffle=False,
        )
        dataset = SFTDataset(
            {
                "integration": [
                    SFTRecord(
                        image_path=str(image_path),
                        target_text="one two three four",
                        dataset_name="integration",
                        messages=messages,
                        user_prompt="Return a compact answer.",
                    )
                ]
            },
            sample_plan=plan,
            media_snapshot_id="integration-media-v1",
        )
        ref = plan.ref_at(0)
        common_kwargs = {
            "model_adapter": model_adapter,
            "template": template,
            "processor": processor,
            "tokenizer": tokenizer,
            "min_pixels": 16384,
            "max_pixels": 262144,
            "max_length": 256,
            "add_eos_token": True,
            "loss_scale_name": "default",
        }
        estimated = ShaftSFTSampleCostProvider(
            dataset=dataset,
            **common_kwargs,
        )(ref)
        actual = SFTCollator(
            include_targets_in_inputs=True,
            **common_kwargs,
        )([dataset[ref]])
        shifted_valid = actual["labels"][:, 1:].ne(-100)
        actual_loss_weight = (
            float(actual["loss_scale"][:, 1:][shifted_valid].sum())
            if "loss_scale" in actual
            else float(shifted_valid.sum())
        )

        assert estimated.llm_tokens == int(actual["attention_mask"].sum())
        assert estimated.supervised_tokens == int(shifted_valid.sum())
        assert estimated.loss_weight_sum == pytest.approx(actual_loss_weight)
        assert estimated.vision_patches == int(actual["image_grid_thw"].prod(dim=-1).sum())


@pytest.mark.integration
def test_qwen3vl_bounded_planner_hard_caps_match_heterogeneous_real_batches(
    tmp_path: Path,
) -> None:
    model_path = Path("models/Qwen3-VL-4B-Instruct")
    if not model_path.exists():
        pytest.skip(f"Model path not found: {model_path}")
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        fix_mistral_regex=False,
    )
    model_adapter = build_model_meta("qwen3vl").resolve_adapter(model_name_or_path=str(model_path))
    template = build_template("qwen3vl")
    records: list[SFTRecord] = []
    for index, (image_size, prompt_length) in enumerate(
        (
            ((64, 64), 8),
            ((128, 256), 80),
            ((256, 128), 24),
            ((64, 512), 160),
        )
    ):
        image_path = tmp_path / f"heterogeneous-{index}.png"
        Image.new("RGB", image_size, color=(index * 30, 40, 50)).save(image_path)
        records.append(
            SFTRecord(
                image_path=str(image_path),
                target_text=f"answer-{index}",
                dataset_name="integration",
                sample_id=f"sample-{index}",
                user_prompt="x" * prompt_length,
            )
        )
    plan = ShaftSamplePlan(
        {"integration": len(records)},
        {"integration": 1.0},
        strategy="concat",
        shuffle=False,
    )
    dataset = SFTDataset(
        {"integration": records},
        sample_plan=plan,
        media_snapshot_id="integration-media-v1",
    )
    common_kwargs = {
        "model_adapter": model_adapter,
        "template": template,
        "processor": processor,
        "tokenizer": processor.tokenizer,
        "min_pixels": 16384,
        "max_pixels": 262144,
        "max_length": 512,
        "add_eos_token": True,
        "loss_scale_name": "default",
    }
    cost_provider = ShaftSFTSampleCostProvider(
        dataset=dataset,
        **common_kwargs,
    )
    costs = tuple(cost_provider(plan.ref_at(index)) for index in range(len(plan)))
    max_tokens_per_microbatch = 2 * max(cost.llm_tokens for cost in costs)
    vision_patch_budget = sum(cost.vision_patches for cost in costs)
    planning_spec = ShaftBatchPlanningSpec(
        data_world_size=2,
        buffer_size=4,
        per_device_microbatch_size=2,
        max_tokens_per_microbatch=max_tokens_per_microbatch,
        resource_budgets=(("vision_patches", vision_patch_budget),),
        seed=7,
        sample_schedule_fingerprint=plan.schedule.fingerprint,
        cost_fingerprint=cost_provider.fingerprint,
    )
    planner = ShaftBatchPlanner(
        schedule=plan.schedule,
        cost_provider=cost_provider,
        spec=planning_spec,
    )
    collator = SFTCollator(include_targets_in_inputs=True, **common_kwargs)

    microstep = planner.next_global_microbatch()
    local_batches = microstep.rank_microbatches
    assert len(local_batches) == 2
    assert all(batch.sample_refs for batch in local_batches)
    for local_batch in local_batches:
        actual = collator([dataset[ref] for ref in local_batch.sample_refs])
        assert actual["input_ids"].numel() == local_batch.padded_llm_tokens
        assert int(actual["image_grid_thw"].prod(dim=-1).sum()) == (local_batch.vision_patches)
        assert len(local_batch.sample_refs) == planning_spec.per_device_microbatch_size
        assert local_batch.padded_llm_tokens <= planning_spec.max_tokens_per_microbatch
        assert local_batch.vision_patches <= int(planning_spec.resource_budget("vision_patches"))


@pytest.mark.integration
@pytest.mark.parametrize(
    ("model_type", "template_name", "model_path"),
    [
        ("qwen3vl", "qwen3vl", Path("models/Qwen3-VL-4B-Instruct")),
        ("qwen36vl", "qwen35vl", Path("models/Qwen3.6-27B")),
        ("qwen36vl", "qwen35vl_thinking", Path("models/Qwen3.6-27B")),
    ],
)
def test_qwen_vl_sft_multiround_supervision_uses_one_processor_call(
    tmp_path: Path,
    model_type: str,
    template_name: str,
    model_path: Path,
) -> None:
    if not model_path.exists():
        pytest.skip(f"Model path not found: {model_path}")
    if not MODEL_REGISTRY.has(model_type):
        pytest.skip(f"Model adapter is not registered: {model_type}")

    image = Image.new("RGB", (64, 64), color=(240, 240, 240))
    base_processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        fix_mistral_regex=False,
    )
    model_adapter = build_model_meta(model_type).resolve_adapter(model_name_or_path=str(model_path))
    template = build_template(template_name)
    item = {
        "dataset_name": "integration",
        "sample_id": "multi-round",
        "image_path": str(tmp_path / "image.png"),
        "image": image,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": "first"}],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "answer one"}]},
            {"role": "user", "content": [{"type": "text", "text": "second"}]},
        ],
        "target_text": "answer two",
        "system_prompt": "",
        "user_prompt": "",
        "extra": {},
    }

    outputs = {}
    processors = {}
    for loss_scale_name in ("default", "last_round"):
        processor = _CountingProcessor(base_processor)
        processors[loss_scale_name] = processor
        collator = SFTCollator(
            model_adapter=model_adapter,
            template=template,
            processor=processor,
            tokenizer=base_processor.tokenizer,
            loss_scale_name=loss_scale_name,
        )
        outputs[loss_scale_name] = collator([item])

    assert processors["default"].call_count == 1
    assert processors["last_round"].call_count == 1
    assert torch.equal(outputs["default"]["input_ids"], outputs["last_round"]["input_ids"])
    assert torch.equal(outputs["default"]["pixel_values"], outputs["last_round"]["pixel_values"])
    assert int(outputs["default"]["labels"].ne(-100).sum()) > int(
        outputs["last_round"]["labels"].ne(-100).sum()
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    ("model_type", "template_name", "model_path"),
    [
        ("qwen3vl", "qwen3vl", Path("models/Qwen3-VL-4B-Instruct")),
        ("qwen36vl", "qwen35vl", Path("models/Qwen3.6-27B")),
    ],
)
def test_qwen_vl_dpo_reuses_one_processed_prompt_for_both_completions(
    tmp_path: Path,
    model_type: str,
    template_name: str,
    model_path: Path,
) -> None:
    if not model_path.exists():
        pytest.skip(f"Model path not found: {model_path}")
    if not MODEL_REGISTRY.has(model_type):
        pytest.skip(f"Model adapter is not registered: {model_type}")

    image = Image.new("RGB", (64, 64), color=(240, 240, 240))
    base_processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        fix_mistral_regex=False,
    )
    processor = _CountingProcessor(base_processor)
    model_adapter = build_model_meta(model_type).resolve_adapter(model_name_or_path=str(model_path))
    collator = DPOCollator(
        model_adapter=model_adapter,
        template=build_template(template_name),
        processor=processor,
        tokenizer=base_processor.tokenizer,
    )
    item = {
        "dataset_name": "integration",
        "sample_id": "dpo-multi-round",
        "image_path": str(tmp_path / "image.png"),
        "image": image,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": "first"}],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "answer one"}]},
            {"role": "user", "content": [{"type": "text", "text": "second"}]},
        ],
        "chosen_text": "preferred answer",
        "rejected_text": "rejected answer",
        "system_prompt": "",
        "user_prompt": "",
        "extra": {},
    }

    output = collator([item])

    assert processor.call_count == 1
    assert output["input_ids"].shape[0] == 2
    assert output["completion_mask"].shape == output["input_ids"].shape
    assert output["image_grid_thw"].shape[0] == 2
    midpoint = output["pixel_values"].shape[0] // 2
    assert midpoint > 0
    assert torch.equal(output["pixel_values"][:midpoint], output["pixel_values"][midpoint:])


@pytest.mark.integration
@pytest.mark.manual
def test_qwen3vl_standard_model_load_and_chat() -> None:
    model_path = Path("models/Qwen3-VL-4B-Instruct")
    if not model_path.exists():
        pytest.skip(f"Model path not found: {model_path}")
    if not MODEL_REGISTRY.has("qwen3vl"):
        pytest.skip("qwen3vl model adapter is not registered in current runtime.")

    image_path = Path(__file__).parent.parent / "temp" / "unit_smoke_image.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    if not image_path.exists():
        Image.new("RGB", (32, 32), color=(240, 240, 240)).save(image_path)

    engine = ShaftInferEngine.from_engine_config(
        InferEngineConfig(
            model_type="qwen3vl",
            model_name_or_path=str(model_path),
            template="qwen3vl",
            device="cpu",
            attn_implementation=None,
            torch_dtype="float32",
            generation=InferGenerationConfig(
                max_new_tokens=32,
                do_sample=False,
            ),
        )
    )

    response = engine.run(
        ShaftInferRequest(
            image_path=str(image_path),
            system_prompt="You are an accurate image description assistant.",
            user_prompt="请只回答：图片里有一张桌子。",
        )
    )

    assert isinstance(response.text, str)
    assert isinstance(response.output_ids, list)
    assert response.text.strip() != ""


@pytest.mark.integration
@pytest.mark.manual
def test_qwen36vl_processor_template_disables_thinking_by_default() -> None:
    model_path = Path("models/Qwen3.6-27B")
    required_files = [
        "config.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
    ]
    missing_files = [name for name in required_files if not (model_path / name).exists()]
    if missing_files:
        pytest.skip(f"Qwen3.6 model path is incomplete: missing {missing_files}")
    if importlib.util.find_spec("transformers.models.qwen3_5") is None:
        pytest.skip("Current Transformers build does not include qwen3_5 support.")
    if not MODEL_REGISTRY.has("qwen36vl"):
        pytest.skip("qwen36vl model adapter is not registered in current runtime.")

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        fix_mistral_regex=False,
    )
    template = build_template("qwen35vl")
    rendered = template.apply_chat_template(
        renderer=ShaftChatRenderer.from_components(
            processor=processor,
            tokenizer=getattr(processor, "tokenizer", None),
        ),
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Return compact JSON only."},
                ],
            }
        ],
    )

    assert "<|im_start|>assistant" in rendered
    assert "<think>\n\n</think>" in rendered


@pytest.mark.integration
def test_qwen35_qwen36_moe_cpu_train_save_exact_resume_and_hf_reload(
    tmp_path: Path,
    repo_root: Path,
) -> None:
    processor_source = Path("models/Qwen3.6-27B")
    if not (processor_source / "preprocessor_config.json").is_file():
        pytest.skip(f"Qwen3.6 processor assets not found: {processor_source}")

    model_dir, dataset_path = prepare_tiny_qwen35_training_assets(
        tmp_path,
        processor_source=processor_source,
        moe=True,
        attention_implementation="eager",
        layer_types=("full_attention", "full_attention"),
    )
    fresh_output = tmp_path / "qwen35-moe-cpu-fresh"
    fresh_config = write_qwen_training_gate_config(
        tmp_path / "qwen35-moe-cpu-fresh.yaml",
        model_type="qwen35vl",
        model_dir=model_dir,
        dataset_path=dataset_path,
        output_dir=fresh_output,
        layout="padded",
        packing="none",
        steps=2,
        save_steps=1,
        use_cpu=True,
        attention_implementation="eager",
        torch_dtype="float32",
        logging_steps=2,
    )
    _run_qwen_training_gate(repo_root, fresh_config, cpu_only=True)

    resumed_output = tmp_path / "qwen35-moe-cpu-resumed"
    resumed_config = write_qwen_training_gate_config(
        tmp_path / "qwen35-moe-cpu-resumed.yaml",
        model_type="qwen35vl",
        model_dir=model_dir,
        dataset_path=dataset_path,
        output_dir=resumed_output,
        layout="padded",
        packing="none",
        steps=2,
        save_steps=1,
        resume_from_checkpoint=fresh_output / "checkpoint-1",
        use_cpu=True,
        attention_implementation="eager",
        torch_dtype="float32",
        logging_steps=2,
    )
    _run_qwen_training_gate(repo_root, resumed_config, cpu_only=True)

    _assert_checkpoint_state_equivalent(
        fresh_output / "checkpoint-2",
        resumed_output / "checkpoint-2",
        weight_filename="model.safetensors",
    )
    _assert_full_moe_router_and_experts_updated(
        model_dir / "model.safetensors",
        fresh_output / "checkpoint-2" / "model.safetensors",
    )
    _assert_router_auxiliary_loss_was_logged(fresh_output / "checkpoint-2")
    _assert_router_auxiliary_loss_was_logged(resumed_output / "checkpoint-2")
    _assert_resumed_root_train_loss_matches_global_checkpoint_window(
        start_checkpoint=fresh_output / "checkpoint-1",
        end_checkpoint=fresh_output / "checkpoint-2",
        resumed_output=resumed_output,
    )
    assert _file_sha256(fresh_output / "best" / "model.safetensors") == (
        _file_sha256(resumed_output / "best" / "model.safetensors")
    )
    for run_dir in (fresh_output, resumed_output):
        _assert_full_hf_export_reloads(
            run_dir / "best",
            source_model_dir=model_dir,
            expected_type="Qwen3_5MoeForConditionalGeneration",
            device="cpu",
        )
        layout = validate_hf_artifact(
            run_dir / "best",
            finetune_mode="full",
            model_type="qwen36vl",
            model_name_or_path=str(run_dir / "best"),
            local_files_only=True,
        )
        assert layout.kind == "full"


@pytest.mark.integration
def test_qwen3vl_opd_cpu_multimodal_rollout_backward_and_hf_reload(
    tmp_path: Path,
    repo_root: Path,
) -> None:
    processor_source = Path("models/Qwen3-VL-2B-Instruct")
    if not (processor_source / "preprocessor_config.json").is_file():
        pytest.skip(f"Qwen3VL processor assets not found: {processor_source}")

    student_dir = prepare_tiny_qwen3vl_artifact(
        tmp_path,
        processor_source=processor_source,
        name="tiny-qwen3vl-opd-student",
        seed=101,
    )
    teacher_dir = prepare_tiny_qwen3vl_artifact(
        tmp_path,
        processor_source=processor_source,
        name="tiny-qwen3vl-opd-teacher",
        seed=202,
    )
    output_dir = tmp_path / "qwen3vl-opd-output"
    config_path = write_qwen3vl_opd_config(
        tmp_path,
        student_dir=student_dir,
        teacher_dir=teacher_dir,
        output_dir=output_dir,
        train_steps=2,
        image_count=2,
        do_sample=True,
    )
    teacher_hash_before = _file_sha256(teacher_dir / "model.safetensors")
    _run_qwen_opd_cpu_gate(repo_root, config_path)

    checkpoint = output_dir / "checkpoint-2"
    validate_training_checkpoint_commit(checkpoint)
    assert _file_sha256(teacher_dir / "model.safetensors") == teacher_hash_before
    with (
        safe_open(
            student_dir / "model.safetensors",
            framework="pt",
            device="cpu",
        ) as initial,
        safe_open(
            output_dir / "best" / "model.safetensors",
            framework="pt",
            device="cpu",
        ) as trained,
    ):
        shared_keys = sorted(set(initial.keys()) & set(trained.keys()))
        assert shared_keys
        assert any(
            not torch.equal(initial.get_tensor(name), trained.get_tensor(name))
            for name in shared_keys
        )
    _assert_finite_training_metrics(checkpoint)
    _assert_full_hf_export_reloads(
        output_dir / "best",
        source_model_dir=student_dir,
        expected_type="Qwen3VLForConditionalGeneration",
        device="cpu",
    )

    resumed_output_dir = tmp_path / "qwen3vl-opd-resumed"
    resumed_config_path = write_qwen3vl_opd_config(
        tmp_path,
        student_dir=student_dir,
        teacher_dir=teacher_dir,
        output_dir=resumed_output_dir,
        train_steps=2,
        image_count=2,
        do_sample=True,
        resume_from_checkpoint=output_dir / "checkpoint-1",
    )
    _run_qwen_opd_cpu_gate(repo_root, resumed_config_path)
    resumed_checkpoint = resumed_output_dir / "checkpoint-2"
    validate_training_checkpoint_commit(resumed_checkpoint)
    assert _file_sha256(checkpoint / "model.safetensors") == _file_sha256(
        resumed_checkpoint / "model.safetensors"
    )
    for filename in ("optimizer.pt", "scheduler.pt"):
        _assert_nested_state_equal(
            torch.load(checkpoint / filename, map_location="cpu", weights_only=True),
            torch.load(
                resumed_checkpoint / filename,
                map_location="cpu",
                weights_only=True,
            ),
            path=filename,
        )
    expected_rng = tuple(sorted(path.name for path in checkpoint.glob("rng_state*.pth")))
    actual_rng = tuple(sorted(path.name for path in resumed_checkpoint.glob("rng_state*.pth")))
    assert expected_rng == actual_rng
    assert expected_rng
    for filename in expected_rng:
        _assert_nested_state_equal(
            torch.load(checkpoint / filename, map_location="cpu", weights_only=False),
            torch.load(
                resumed_checkpoint / filename,
                map_location="cpu",
                weights_only=False,
            ),
            path=filename,
        )
    assert _normalized_trainer_state(checkpoint) == _normalized_trainer_state(resumed_checkpoint)
    assert _file_sha256(output_dir / "best" / "model.safetensors") == _file_sha256(
        resumed_output_dir / "best" / "model.safetensors"
    )


@pytest.mark.integration
@pytest.mark.manual
def test_qwen3vl_opd_release_weights_single_and_two_rank_exact_resume_gate(
    tmp_path: Path,
    repo_root: Path,
) -> None:
    if os.environ.get("SHAFT_RUN_QWEN_OPD_RELEASE_GATE") != "1":
        pytest.skip("Set SHAFT_RUN_QWEN_OPD_RELEASE_GATE=1 to run the OPD CUDA gate.")
    if torch.cuda.device_count() < 2:
        pytest.skip("The Qwen3VL OPD release gate requires two visible CUDA devices.")
    student_path = Path("models/Qwen3-VL-2B-Instruct").resolve()
    teacher_path = Path("models/Qwen3-VL-4B-Instruct").resolve()
    for role, model_path in (("student", student_path), ("teacher", teacher_path)):
        if not (model_path / "config.json").is_file():
            pytest.skip(f"Qwen3VL OPD {role} assets not found: {model_path}")

    common = {
        "student_dir": student_path,
        "teacher_dir": teacher_path,
        "do_sample": True,
        "train_size": 8,
        "finetune_mode": "lora",
        "use_cpu": False,
        "torch_dtype": "bfloat16",
        "training_precision": "bf16",
        "attention_implementation": "eager",
        "learning_rate": 1.0e-4,
    }

    single_output = tmp_path / "qwen3vl-opd-release-single"
    single_config = write_qwen3vl_opd_config(
        tmp_path,
        output_dir=single_output,
        train_steps=1,
        image_count=2,
        gradient_accumulation_steps=1,
        **common,
    )
    _run_qwen_opd_cuda_gate(repo_root, single_config, world_size=1)
    single_checkpoint = single_output / "checkpoint-1"
    validate_training_checkpoint_commit(single_checkpoint)
    _assert_finite_training_metrics(single_checkpoint)
    with safe_open(
        single_checkpoint / "adapter_model.safetensors",
        framework="pt",
        device="cpu",
    ) as adapter:
        lora_b_keys = [name for name in adapter.keys() if "lora_B" in name]
        assert lora_b_keys
        assert any(
            bool(torch.count_nonzero(adapter.get_tensor(name))) for name in lora_b_keys
        )
    _validate_qwen_peft_export(
        repo_root,
        export_path=single_output / "best",
        model_type="qwen3vl",
        model_path=student_path,
    )

    fresh_output = tmp_path / "qwen3vl-opd-release-ddp-fresh"
    fresh_config = write_qwen3vl_opd_config(
        tmp_path,
        output_dir=fresh_output,
        train_steps=2,
        image_count=1,
        gradient_accumulation_steps=2,
        **common,
    )
    _run_qwen_opd_cuda_gate(repo_root, fresh_config, world_size=2)

    resumed_output = tmp_path / "qwen3vl-opd-release-ddp-resumed"
    resumed_config = write_qwen3vl_opd_config(
        tmp_path,
        output_dir=resumed_output,
        train_steps=2,
        image_count=1,
        gradient_accumulation_steps=2,
        resume_from_checkpoint=fresh_output / "checkpoint-1",
        **common,
    )
    _run_qwen_opd_cuda_gate(repo_root, resumed_config, world_size=2)

    _assert_checkpoint_state_equivalent(
        fresh_output / "checkpoint-2",
        resumed_output / "checkpoint-2",
        weight_filename="adapter_model.safetensors",
        efficiency_expected=False,
    )
    assert _file_sha256(fresh_output / "best" / "adapter_model.safetensors") == (
        _file_sha256(resumed_output / "best" / "adapter_model.safetensors")
    )
    for run_dir in (fresh_output, resumed_output):
        _assert_finite_training_metrics(run_dir / "checkpoint-2")
        _validate_qwen_peft_export(
            repo_root,
            export_path=run_dir / "best",
            model_type="qwen3vl",
            model_path=student_path,
        )


@pytest.mark.integration
@pytest.mark.manual
@pytest.mark.gpu
def test_qwen3vl_opd_vllm_server_rollout_release_gate(
    tmp_path: Path,
    repo_root: Path,
) -> None:
    if os.environ.get("SHAFT_RUN_QWEN_OPD_VLLM_GATE") != "1":
        pytest.skip("Set SHAFT_RUN_QWEN_OPD_VLLM_GATE=1 to run the OPD vLLM gate.")
    training_world_size = int(os.environ.get("SHAFT_OPD_VLLM_TRAIN_WORLD_SIZE", "1"))
    if training_world_size <= 0:
        raise ValueError("SHAFT_OPD_VLLM_TRAIN_WORLD_SIZE must be > 0.")
    training_gpus = [
        value.strip()
        for value in os.environ.get(
            "SHAFT_OPD_VLLM_TRAIN_GPUS",
            ",".join(str(index) for index in range(training_world_size)),
        ).split(",")
        if value.strip()
    ]
    server_gpu = os.environ.get(
        "SHAFT_OPD_VLLM_SERVER_GPU",
        str(training_world_size),
    ).strip()
    if len(training_gpus) != training_world_size:
        raise ValueError(
            "SHAFT_OPD_VLLM_TRAIN_GPUS cardinality must match "
            "SHAFT_OPD_VLLM_TRAIN_WORLD_SIZE."
        )
    if len(set(training_gpus)) != len(training_gpus) or server_gpu in training_gpus:
        raise ValueError("OPD vLLM training and server GPU assignments must be disjoint.")
    numeric_gpus = [int(value) for value in (*training_gpus, server_gpu)]
    if any(index < 0 or index >= torch.cuda.device_count() for index in numeric_gpus):
        pytest.skip(
            "The OPD vLLM gate does not have all requested CUDA devices: "
            f"training={training_gpus}, server={server_gpu}."
        )
    student_path = Path("models/Qwen3-VL-2B-Instruct").resolve()
    teacher_path = Path("models/Qwen3-VL-4B-Instruct").resolve()
    for model_path in (student_path, teacher_path):
        if not (model_path / "config.json").is_file():
            pytest.skip(f"Qwen3VL OPD vLLM asset not found: {model_path}")

    def reserve_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            return int(listener.getsockname()[1])

    server_port = reserve_port()
    group_port = reserve_port()
    while group_port == server_port:
        group_port = reserve_port()
    server = subprocess.Popen(
        [
            str(Path(sys.executable).with_name("trl")),
            "vllm-serve",
            "--model",
            str(student_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(server_port),
            "--gpu-memory-utilization",
            "0.5",
            "--dtype",
            "bfloat16",
            "--max-model-len",
            "512",
            "--enforce-eager",
            "--log-level",
            "warning",
        ],
        cwd=repo_root,
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": server_gpu,
            "OMP_NUM_THREADS": "1",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 300
        while True:
            if server.poll() is not None:
                stdout, stderr = server.communicate()
                raise AssertionError(
                    "TRL vLLM server exited during startup.\n"
                    f"stdout:\n{stdout}\nstderr:\n{stderr}"
                )
            try:
                with urlopen(  # noqa: S310 - loopback test endpoint
                    f"http://127.0.0.1:{server_port}/health/",
                    timeout=2,
                ) as response:
                    if response.status == 200:
                        break
            except OSError:
                pass
            if time.monotonic() >= deadline:
                raise AssertionError("TRL vLLM server did not become healthy within 300s.")
            time.sleep(1)

        output_dir = tmp_path / "qwen3vl-opd-vllm"
        config_path = write_qwen3vl_opd_config(
            tmp_path,
            student_dir=student_path,
            teacher_dir=teacher_path,
            output_dir=output_dir,
            train_steps=1,
            image_count=1,
            do_sample=True,
            train_size=max(2, training_world_size),
            finetune_mode="lora",
            use_cpu=False,
            torch_dtype="bfloat16",
            training_precision="bf16",
            learning_rate=1.0e-4,
        )
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        payload["train"]["efficiency"]["enabled"] = True
        payload["opd"]["rollout"]["backend"] = "vllm"
        payload["opd"]["rollout"]["vllm"] = {
            "mode": "server",
            "server_host": "127.0.0.1",
            "server_port": server_port,
            "group_port": group_port,
            "server_timeout": 300.0,
            "max_model_length": 512,
            "max_num_seqs": 2,
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.5,
        }
        config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(training_gpus)
        env["OMP_NUM_THREADS"] = "1"
        probe_path = tmp_path / "qwen3vl-opd-vllm-probe.json"
        command = [sys.executable]
        if training_world_size > 1:
            command.extend(
                (
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    "--nnodes=1",
                    f"--nproc_per_node={training_world_size}",
                )
            )
        command.extend(
            (
                "tests/support/distributed_opd_train.py",
                str(config_path),
                str(probe_path),
            )
        )
        completed = subprocess.run(
            command,
            cwd=repo_root,
            env=env,
            text=True,
            capture_output=True,
            timeout=1200,
            check=False,
        )
        if completed.returncode != 0:
            raise AssertionError(
                "Qwen3VL OPD vLLM gate failed.\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
        probe = json.loads(probe_path.read_text(encoding="utf-8"))
        expected_probe = {
            "all_ranks_student_changed": True,
            "all_ranks_teacher_unchanged": True,
            "teacher_model_loaded": True,
            "teacher_provider": "hf_local",
            "world_size": training_world_size,
        }
        assert {key: probe[key] for key in expected_probe} == expected_probe
        assert probe["minimum_rank_student_max_abs_delta"] > 0
        assert len(probe["rank_traces"]) == training_world_size
        telemetry = json.loads(
            (output_dir / "shaft_opd_telemetry.json").read_text(encoding="utf-8")
        )
        assert telemetry["world_size"] == training_world_size
        assert len(telemetry["rank_frames"]) == training_world_size
        assert all(len(rank_frames) == 1 for rank_frames in telemetry["rank_frames"])
        assert all(
            frame["global_step"] == 1
            and frame["update_applied"]
            and frame["rollout_weight_sync_seconds"] > 0
            and frame["rollout_generate_seconds"] > 0
            and frame["completion_tokens"] > 0
            and frame["device_optimizer_frame_seconds"] is not None
            for rank_frames in telemetry["rank_frames"]
            for frame in rank_frames
        )
    finally:
        try:
            os.killpg(server.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            server.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(server.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            server.communicate(timeout=30)


@pytest.mark.integration
def test_qwen3vl_moe_cpu_train_save_exact_resume_and_hf_reload(
    tmp_path: Path,
    repo_root: Path,
) -> None:
    processor_source = Path("models/Qwen3-VL-4B-Instruct")
    if not (processor_source / "preprocessor_config.json").is_file():
        pytest.skip(f"Qwen3VL processor assets not found: {processor_source}")

    model_dir, dataset_path = prepare_tiny_qwen3vl_moe_training_assets(
        tmp_path,
        processor_source=processor_source,
    )
    fresh_output = tmp_path / "qwen3vl-moe-cpu-fresh"
    fresh_config = write_qwen_training_gate_config(
        tmp_path / "qwen3vl-moe-cpu-fresh.yaml",
        model_type="qwen3vl",
        model_dir=model_dir,
        dataset_path=dataset_path,
        output_dir=fresh_output,
        layout="padded",
        packing="none",
        steps=2,
        save_steps=1,
        use_cpu=True,
        attention_implementation="eager",
        experts_implementation="grouped_mm",
        torch_dtype="float32",
        logging_steps=1,
        router_aux_loss_weight=0.002,
        warmup_ratio=0.0,
    )
    _run_qwen_training_gate(repo_root, fresh_config, cpu_only=True)

    resumed_output = tmp_path / "qwen3vl-moe-cpu-resumed"
    resumed_config = write_qwen_training_gate_config(
        tmp_path / "qwen3vl-moe-cpu-resumed.yaml",
        model_type="qwen3vl",
        model_dir=model_dir,
        dataset_path=dataset_path,
        output_dir=resumed_output,
        layout="padded",
        packing="none",
        steps=2,
        save_steps=1,
        resume_from_checkpoint=fresh_output / "checkpoint-1",
        use_cpu=True,
        attention_implementation="eager",
        experts_implementation="grouped_mm",
        torch_dtype="float32",
        logging_steps=1,
        router_aux_loss_weight=0.002,
        warmup_ratio=0.0,
    )
    _run_qwen_training_gate(repo_root, resumed_config, cpu_only=True)

    _assert_checkpoint_state_equivalent(
        fresh_output / "checkpoint-2",
        resumed_output / "checkpoint-2",
        weight_filename="model.safetensors",
    )
    _assert_fused_moe_router_and_experts_updated(
        model_dir / "model.safetensors",
        fresh_output / "checkpoint-2" / "model.safetensors",
    )
    for run_dir in (fresh_output, resumed_output):
        checkpoint = run_dir / "checkpoint-2"
        _assert_router_auxiliary_loss_was_logged(
            checkpoint,
            expected_coefficient=0.002,
        )
        _assert_finite_training_metrics(checkpoint)
        _assert_full_hf_export_reloads(
            run_dir / "best",
            source_model_dir=model_dir,
            expected_type="Qwen3VLMoeForConditionalGeneration",
            device="cpu",
        )
        layout = validate_hf_artifact(
            run_dir / "best",
            finetune_mode="full",
            model_type="qwen3vl",
            model_name_or_path=str(run_dir / "best"),
            local_files_only=True,
        )
        assert layout.kind == "full"


@pytest.mark.integration
@pytest.mark.manual
def test_qwen35_qwen36_two_rank_train_save_and_exact_resume_release_gate(
    tmp_path: Path,
    repo_root: Path,
) -> None:
    if os.environ.get("SHAFT_RUN_QWEN_TRAIN_RELEASE_GATE") != "1":
        pytest.skip("Set SHAFT_RUN_QWEN_TRAIN_RELEASE_GATE=1 to run the CUDA release gate.")
    if torch.cuda.device_count() < 2:
        pytest.skip("The Qwen training release gate requires two visible CUDA devices.")
    processor_source = Path("models/Qwen3.6-27B")
    if not (processor_source / "preprocessor_config.json").is_file():
        pytest.skip(f"Qwen3.6 processor assets not found: {processor_source}")

    model_dir, dataset_path = prepare_tiny_qwen35_training_assets(
        tmp_path,
        processor_source=processor_source,
    )
    qwen35_output = tmp_path / "qwen35-fixed-fresh"
    qwen35_config = write_qwen_training_gate_config(
        tmp_path / "qwen35-fixed.yaml",
        model_type="qwen35vl",
        model_dir=model_dir,
        dataset_path=dataset_path,
        output_dir=qwen35_output,
        layout="padded",
        packing="none",
        steps=2,
        save_steps=1,
    )
    _run_qwen_training_gate(repo_root, qwen35_config)
    qwen35_resumed_output = tmp_path / "qwen35-fixed-resumed"
    qwen35_resumed_config = write_qwen_training_gate_config(
        tmp_path / "qwen35-fixed-resumed.yaml",
        model_type="qwen35vl",
        model_dir=model_dir,
        dataset_path=dataset_path,
        output_dir=qwen35_resumed_output,
        layout="padded",
        packing="none",
        steps=2,
        save_steps=1,
        resume_from_checkpoint=qwen35_output / "checkpoint-1",
    )
    _run_qwen_training_gate(repo_root, qwen35_resumed_config)

    fresh_output = tmp_path / "qwen36-fresh"
    fresh_config = write_qwen_training_gate_config(
        tmp_path / "qwen36-fresh.yaml",
        model_type="qwen36vl",
        model_dir=model_dir,
        dataset_path=dataset_path,
        output_dir=fresh_output,
        layout="varlen",
        packing="greedy",
        steps=2,
        save_steps=1,
    )
    _run_qwen_training_gate(repo_root, fresh_config)

    resumed_output = tmp_path / "qwen36-resumed"
    resumed_config = write_qwen_training_gate_config(
        tmp_path / "qwen36-resumed.yaml",
        model_type="qwen36vl",
        model_dir=model_dir,
        dataset_path=dataset_path,
        output_dir=resumed_output,
        layout="varlen",
        packing="greedy",
        steps=2,
        save_steps=1,
        resume_from_checkpoint=fresh_output / "checkpoint-1",
    )
    _run_qwen_training_gate(repo_root, resumed_config)

    _assert_checkpoint_state_equivalent(
        qwen35_output / "checkpoint-2",
        qwen35_resumed_output / "checkpoint-2",
        weight_filename="model.safetensors",
    )
    _assert_checkpoint_state_equivalent(
        fresh_output / "checkpoint-2",
        resumed_output / "checkpoint-2",
        weight_filename="model.safetensors",
    )
    assert _file_sha256(qwen35_output / "best" / "model.safetensors") == (
        _file_sha256(qwen35_resumed_output / "best" / "model.safetensors")
    )
    assert _file_sha256(fresh_output / "best" / "model.safetensors") == (
        _file_sha256(resumed_output / "best" / "model.safetensors")
    )
    for run_dir in (
        qwen35_output,
        qwen35_resumed_output,
        fresh_output,
        resumed_output,
    ):
        _assert_full_hf_export_reloads(
            run_dir / "best",
            source_model_dir=model_dir,
            expected_type="Qwen3_5ForConditionalGeneration",
        )

    for output_dir, expected_steps in (
        (qwen35_output, 2),
        (qwen35_resumed_output, 2),
        (fresh_output, 2),
        (resumed_output, 2),
    ):
        summary = json.loads(
            (output_dir / "shaft_training_efficiency.json").read_text(encoding="utf-8")
        )
        assert summary["schema_version"] == TRAINING_EFFICIENCY_SCHEMA_VERSION
        assert summary["complete_history"] is True
        assert summary["aggregate"]["optimizer_steps"] == expected_steps
        assert summary["aggregate"]["update_applied_steps"] > 0
        assert summary["aggregate"]["device_timing_steps"] == expected_steps
        assert summary["aggregate"]["device_training_seconds"] > 0
        assert (
            summary["aggregate"]["critical_path_seconds"]
            >= summary["aggregate"]["device_training_seconds"]
        )

    fresh_contract = json.loads(
        (fresh_output / "shaft_training_efficiency.json").read_text(encoding="utf-8")
    )["contract"]
    resumed_contract = json.loads(
        (resumed_output / "shaft_training_efficiency.json").read_text(encoding="utf-8")
    )["contract"]
    assert fresh_contract == resumed_contract


@pytest.mark.integration
@pytest.mark.manual
def test_qwen35_qwen36_moe_two_rank_train_save_and_exact_resume_release_gate(
    tmp_path: Path,
    repo_root: Path,
) -> None:
    if os.environ.get("SHAFT_RUN_QWEN_TRAIN_RELEASE_GATE") != "1":
        pytest.skip("Set SHAFT_RUN_QWEN_TRAIN_RELEASE_GATE=1 to run the CUDA release gate.")
    if torch.cuda.device_count() < 2:
        pytest.skip("The Qwen training release gate requires two visible CUDA devices.")
    processor_source = Path("models/Qwen3.6-27B")
    if not (processor_source / "preprocessor_config.json").is_file():
        pytest.skip(f"Qwen3.6 processor assets not found: {processor_source}")

    model_dir, dataset_path = prepare_tiny_qwen35_training_assets(
        tmp_path,
        processor_source=processor_source,
        moe=True,
    )
    qwen35_output = tmp_path / "qwen35-moe-fixed-fresh"
    qwen35_config = write_qwen_training_gate_config(
        tmp_path / "qwen35-moe-fixed.yaml",
        model_type="qwen35vl",
        model_dir=model_dir,
        dataset_path=dataset_path,
        output_dir=qwen35_output,
        layout="padded",
        packing="none",
        steps=2,
        save_steps=1,
        finetune_mode="lora",
        target_parameters=("auto",),
        logging_steps=2,
    )
    _run_qwen_training_gate(repo_root, qwen35_config)
    qwen35_resumed_output = tmp_path / "qwen35-moe-fixed-resumed"
    qwen35_resumed_config = write_qwen_training_gate_config(
        tmp_path / "qwen35-moe-fixed-resumed.yaml",
        model_type="qwen35vl",
        model_dir=model_dir,
        dataset_path=dataset_path,
        output_dir=qwen35_resumed_output,
        layout="padded",
        packing="none",
        steps=2,
        save_steps=1,
        resume_from_checkpoint=qwen35_output / "checkpoint-1",
        finetune_mode="lora",
        target_parameters=("auto",),
        logging_steps=2,
    )
    _run_qwen_training_gate(repo_root, qwen35_resumed_config)

    fresh_output = tmp_path / "qwen36-moe-fresh"
    fresh_config = write_qwen_training_gate_config(
        tmp_path / "qwen36-moe-fresh.yaml",
        model_type="qwen36vl",
        model_dir=model_dir,
        dataset_path=dataset_path,
        output_dir=fresh_output,
        layout="varlen",
        packing="greedy",
        steps=2,
        save_steps=1,
        logging_steps=2,
    )
    _run_qwen_training_gate(repo_root, fresh_config)

    resumed_output = tmp_path / "qwen36-moe-resumed"
    resumed_config = write_qwen_training_gate_config(
        tmp_path / "qwen36-moe-resumed.yaml",
        model_type="qwen36vl",
        model_dir=model_dir,
        dataset_path=dataset_path,
        output_dir=resumed_output,
        layout="varlen",
        packing="greedy",
        steps=2,
        save_steps=1,
        resume_from_checkpoint=fresh_output / "checkpoint-1",
        logging_steps=2,
    )
    _run_qwen_training_gate(repo_root, resumed_config)

    _assert_checkpoint_state_equivalent(
        qwen35_output / "checkpoint-2",
        qwen35_resumed_output / "checkpoint-2",
        weight_filename="adapter_model.safetensors",
    )
    _assert_checkpoint_state_equivalent(
        fresh_output / "checkpoint-2",
        resumed_output / "checkpoint-2",
        weight_filename="model.safetensors",
    )
    assert _file_sha256(qwen35_output / "best" / "adapter_model.safetensors") == _file_sha256(
        qwen35_resumed_output / "best" / "adapter_model.safetensors"
    )
    for run_dir in (qwen35_output, qwen35_resumed_output):
        _validate_qwen_peft_export(
            repo_root,
            export_path=run_dir / "best",
            model_type="qwen35vl",
            model_path=model_dir,
        )
        _assert_standard_qwen_peft_export_reloads(
            base_model_path=model_dir,
            export_dir=run_dir / "best",
        )
        _assert_moe_lora_router_and_experts_updated(run_dir / "best" / "adapter_model.safetensors")
    _assert_router_auxiliary_loss_was_logged(qwen35_output / "checkpoint-2")
    _assert_router_auxiliary_loss_was_logged(qwen35_resumed_output / "checkpoint-2")
    assert _file_sha256(fresh_output / "best" / "model.safetensors") == (
        _file_sha256(resumed_output / "best" / "model.safetensors")
    )
    for run_dir in (fresh_output, resumed_output):
        _assert_full_hf_export_reloads(
            run_dir / "best",
            source_model_dir=model_dir,
            expected_type="Qwen3_5MoeForConditionalGeneration",
        )

    for output_dir, expected_steps in (
        (qwen35_output, 2),
        (qwen35_resumed_output, 2),
        (fresh_output, 2),
        (resumed_output, 2),
    ):
        summary = json.loads(
            (output_dir / "shaft_training_efficiency.json").read_text(encoding="utf-8")
        )
        assert summary["complete_history"] is True
        assert summary["aggregate"]["optimizer_steps"] == expected_steps
        assert summary["aggregate"]["update_applied_steps"] > 0
        assert summary["aggregate"]["device_timing_steps"] == expected_steps
        assert summary["contract"]["model_plan_fingerprint"]

    fresh_contract = json.loads(
        (fresh_output / "shaft_training_efficiency.json").read_text(encoding="utf-8")
    )["contract"]
    resumed_contract = json.loads(
        (resumed_output / "shaft_training_efficiency.json").read_text(encoding="utf-8")
    )["contract"]
    assert fresh_contract == resumed_contract


@pytest.mark.integration
@pytest.mark.manual
@pytest.mark.parametrize("distributed_strategy", ["fsdp", "deepspeed"])
def test_qwen35_moe_two_rank_sharded_backend_release_gate(
    tmp_path: Path,
    repo_root: Path,
    distributed_strategy: str,
) -> None:
    if os.environ.get("SHAFT_RUN_QWEN_TRAIN_RELEASE_GATE") != "1":
        pytest.skip("Set SHAFT_RUN_QWEN_TRAIN_RELEASE_GATE=1 to run the CUDA release gate.")
    if torch.cuda.device_count() < 2:
        pytest.skip("The Qwen training release gate requires two visible CUDA devices.")
    processor_source = Path("models/Qwen3.6-27B")
    if not (processor_source / "preprocessor_config.json").is_file():
        pytest.skip(f"Qwen3.6 processor assets not found: {processor_source}")

    model_dir, dataset_path = prepare_tiny_qwen35_training_assets(
        tmp_path,
        processor_source=processor_source,
        moe=True,
    )
    finetune_mode = "lora" if distributed_strategy == "fsdp" else "full"
    target_parameters = ("auto",) if finetune_mode == "lora" else ()
    output_dir = tmp_path / f"qwen35-moe-{distributed_strategy}"
    config_path = write_qwen_training_gate_config(
        tmp_path / f"qwen35-moe-{distributed_strategy}.yaml",
        model_type="qwen35vl",
        model_dir=model_dir,
        dataset_path=dataset_path,
        output_dir=output_dir,
        layout="padded",
        packing="none",
        steps=2,
        save_steps=1,
        finetune_mode=finetune_mode,
        target_parameters=target_parameters,
        distributed_strategy=distributed_strategy,
        gradient_accumulation_steps=2,
        gradient_checkpointing=distributed_strategy == "fsdp",
        logging_steps=2,
        warmup_ratio=0.0,
        bounded_cost_grouping=True,
    )
    _run_qwen_training_gate(repo_root, config_path)

    resumed_output = tmp_path / f"qwen35-moe-{distributed_strategy}-resumed"
    resumed_config = write_qwen_training_gate_config(
        tmp_path / f"qwen35-moe-{distributed_strategy}-resumed.yaml",
        model_type="qwen35vl",
        model_dir=model_dir,
        dataset_path=dataset_path,
        output_dir=resumed_output,
        layout="padded",
        packing="none",
        steps=2,
        save_steps=1,
        resume_from_checkpoint=output_dir / "checkpoint-1",
        finetune_mode=finetune_mode,
        target_parameters=target_parameters,
        distributed_strategy=distributed_strategy,
        gradient_accumulation_steps=2,
        gradient_checkpointing=distributed_strategy == "fsdp",
        logging_steps=2,
        warmup_ratio=0.0,
        bounded_cost_grouping=True,
    )
    _run_qwen_training_gate(repo_root, resumed_config)

    weight_filename = (
        "adapter_model.safetensors" if finetune_mode == "lora" else "model.safetensors"
    )
    _assert_backend_native_checkpoint_state_equivalent(
        output_dir / "checkpoint-2",
        resumed_output / "checkpoint-2",
        distributed_strategy=distributed_strategy,
        weight_filename=weight_filename,
    )
    if finetune_mode == "lora":
        _assert_moe_lora_router_and_experts_updated(output_dir / "checkpoint-1" / weight_filename)
        assert _file_sha256(output_dir / "checkpoint-1" / weight_filename) != (
            _file_sha256(output_dir / "checkpoint-2" / weight_filename)
        )
    assert _file_sha256(output_dir / "best" / weight_filename) == _file_sha256(
        resumed_output / "best" / weight_filename
    )

    for run_dir in (output_dir, resumed_output):
        assert (run_dir / "checkpoint-2").is_dir()
        trainer_state = json.loads(
            (run_dir / "checkpoint-2" / "trainer_state.json").read_text(encoding="utf-8")
        )
        # Eight samples / (2 ranks * local batch 2) yield two microbatches per
        # rank; GA=2 therefore reaches optimizer step 2 at exactly one epoch.
        assert float(trainer_state["epoch"]) == pytest.approx(1.0)
        assert float(trainer_state["total_flos"]) > 0.0
        efficiency = json.loads(
            (run_dir / "shaft_training_efficiency.json").read_text(encoding="utf-8")
        )
        assert efficiency["contract"]["gradient_accumulation_steps"] == 2
        _assert_router_auxiliary_loss_was_logged(run_dir / "checkpoint-2")
        if finetune_mode == "lora":
            _validate_qwen_peft_export(
                repo_root,
                export_path=run_dir / "best",
                model_type="qwen35vl",
                model_path=model_dir,
            )
            _assert_moe_lora_router_and_experts_updated(
                run_dir / "best" / "adapter_model.safetensors"
            )
        else:
            _assert_full_moe_router_and_experts_updated(
                model_dir / "model.safetensors",
                run_dir / "best" / "model.safetensors",
            )
            _assert_full_hf_export_reloads(
                run_dir / "best",
                source_model_dir=model_dir,
                expected_type="Qwen3_5MoeForConditionalGeneration",
            )

    if finetune_mode == "lora":
        for run_dir in (output_dir, resumed_output):
            _assert_standard_qwen_peft_export_reloads(
                base_model_path=model_dir,
                export_dir=run_dir / "best",
            )


@pytest.mark.integration
@pytest.mark.manual
def test_qwen3vl_30b_moe_two_rank_fsdp_lora_release_gate(
    tmp_path: Path,
    repo_root: Path,
) -> None:
    if os.environ.get("SHAFT_RUN_QWEN3VL_MOE_30B_GATE") != "1":
        pytest.skip("Set SHAFT_RUN_QWEN3VL_MOE_30B_GATE=1 to run the real 30B MoE SFT gate.")
    if torch.cuda.device_count() < 2:
        pytest.skip("The Qwen3VL 30B MoE gate requires two visible CUDA devices.")
    model_path = Path("models/Qwen3-VL-30B-A3B-Instruct").resolve()
    required_files = (
        "config.json",
        "model.safetensors.index.json",
        "preprocessor_config.json",
        "tokenizer.json",
    )
    missing = [name for name in required_files if not (model_path / name).is_file()]
    if missing:
        pytest.skip(f"Qwen3VL 30B MoE assets are incomplete: missing={missing}")
    model_config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    language_layers = int(model_config["text_config"]["num_hidden_layers"])
    vision_blocks = int(model_config["vision_config"]["depth"])
    num_experts = int(model_config["text_config"]["num_experts"])

    dataset_path = prepare_qwen_training_dataset(tmp_path)
    fresh_output = tmp_path / "qwen3vl-30b-moe-fsdp-fresh"
    fresh_config = write_qwen_training_gate_config(
        tmp_path / "qwen3vl-30b-moe-fsdp-fresh.yaml",
        model_type="qwen3vl",
        model_dir=model_path,
        dataset_path=dataset_path,
        output_dir=fresh_output,
        layout="padded",
        packing="none",
        steps=2,
        save_steps=1,
        finetune_mode="lora",
        target_parameters=("auto",),
        attention_implementation="flash_attention_2",
        experts_implementation="grouped_mm",
        distributed_strategy="fsdp",
        gradient_checkpointing=True,
        per_device_train_batch_size=1,
        num_workers=0,
        router_aux_loss_weight=0.002,
        warmup_ratio=0.0,
    )
    _run_qwen_training_gate(
        repo_root,
        fresh_config,
        timeout_seconds=3600,
    )

    resumed_output = tmp_path / "qwen3vl-30b-moe-fsdp-resumed"
    resumed_config = write_qwen_training_gate_config(
        tmp_path / "qwen3vl-30b-moe-fsdp-resumed.yaml",
        model_type="qwen3vl",
        model_dir=model_path,
        dataset_path=dataset_path,
        output_dir=resumed_output,
        layout="padded",
        packing="none",
        steps=2,
        save_steps=1,
        resume_from_checkpoint=fresh_output / "checkpoint-1",
        finetune_mode="lora",
        target_parameters=("auto",),
        attention_implementation="flash_attention_2",
        experts_implementation="grouped_mm",
        distributed_strategy="fsdp",
        gradient_checkpointing=True,
        per_device_train_batch_size=1,
        num_workers=0,
        router_aux_loss_weight=0.002,
        warmup_ratio=0.0,
    )
    _run_qwen_training_gate(
        repo_root,
        resumed_config,
        timeout_seconds=3600,
    )

    weight_filename = "adapter_model.safetensors"
    _assert_backend_native_checkpoint_state_equivalent(
        fresh_output / "checkpoint-2",
        resumed_output / "checkpoint-2",
        distributed_strategy="fsdp",
        weight_filename=weight_filename,
    )
    assert _file_sha256(fresh_output / "best" / weight_filename) == _file_sha256(
        resumed_output / "best" / weight_filename
    )
    _assert_qwen3vl_moe_lora_roles_updated(
        fresh_output / "checkpoint-1" / weight_filename,
        language_layers=language_layers,
        vision_blocks=vision_blocks,
    )
    assert _file_sha256(fresh_output / "checkpoint-1" / weight_filename) != (
        _file_sha256(fresh_output / "checkpoint-2" / weight_filename)
    )
    _assert_adapter_tensors_changed(
        fresh_output / "checkpoint-1" / weight_filename,
        fresh_output / "checkpoint-2" / weight_filename,
    )
    for run_dir in (fresh_output, resumed_output):
        checkpoint = run_dir / "checkpoint-2"
        _assert_router_auxiliary_loss_was_logged(
            checkpoint,
            expected_coefficient=0.002,
        )
        _assert_finite_training_metrics(checkpoint)
        _assert_qwen3vl_moe_lora_roles_updated(
            run_dir / "best" / weight_filename,
            language_layers=language_layers,
            vision_blocks=vision_blocks,
        )
        efficiency = json.loads(
            (run_dir / "shaft_training_efficiency.json").read_text(encoding="utf-8")
        )
        finetune_summary = json.loads(
            (run_dir / "shaft_finetune_summary.json").read_text(encoding="utf-8")
        )
        optimizer_summary = json.loads(
            (run_dir / "shaft_optimizer_summary.json").read_text(encoding="utf-8")
        )
        adapter_path = run_dir / "best" / weight_filename
        with safe_open(adapter_path, framework="pt", device="cpu") as adapter_state:
            adapter_keys = tuple(adapter_state.keys())
            adapter_numel = sum(
                int(np.prod(adapter_state.get_slice(name).get_shape())) for name in adapter_keys
            )
        trainable_params = int(finetune_summary["trainable_params"])
        assert adapter_numel == trainable_params
        assert int(optimizer_summary["total_trainable_params"]) == trainable_params
        assert sum(int(group["num_tensors"]) for group in optimizer_summary["groups"]) == len(
            adapter_keys
        )
        assert int(finetune_summary["total_params"]) == (
            trainable_params + int(finetune_summary["frozen_params"])
        )
        assert int(finetune_summary["frozen_params"]) > int(finetune_summary["trainable_params"])
        assert all("lora_" in name for name in finetune_summary["sample_trainable_parameters"])
        assert all("lora_" not in name for name in finetune_summary["sample_frozen_parameters"])
        assert efficiency["aggregate"]["optimizer_steps"] == 2
        assert efficiency["aggregate"]["update_applied_steps"] == 2
        assert efficiency["contract"]["torch_dtype"] == "bfloat16"
        assert int(efficiency["peak_device_memory_allocated_bytes"]) > 0
        assert int(efficiency["peak_device_memory_reserved_bytes"]) > 0
        training_args = torch.load(
            checkpoint / "training_args.bin",
            map_location="cpu",
            weights_only=False,
        )
        assert training_args.bf16 is True
        assert training_args.fp16 is False
        assert training_args.warmup_steps == 0
        _validate_qwen_peft_export(
            repo_root,
            export_path=run_dir / "best",
            model_type="qwen3vl",
            model_path=model_path,
        )

    _assert_standard_qwen_peft_export_reloads(
        base_model_path=model_path,
        export_dir=fresh_output / "best",
        experts_implementation="grouped_mm",
        expected_router_layers=language_layers,
        expected_num_experts=num_experts,
        expected_dtype=torch.bfloat16,
        image_path=tmp_path / "image.png",
    )


@pytest.mark.integration
@pytest.mark.manual
def test_qwen3vl_2b_two_rank_fp16_padded_lora_exact_resume_release_gate(
    tmp_path: Path,
    repo_root: Path,
) -> None:
    if os.environ.get("SHAFT_RUN_QWEN_TRAIN_RELEASE_GATE") != "1":
        pytest.skip("Set SHAFT_RUN_QWEN_TRAIN_RELEASE_GATE=1 to run the CUDA release gate.")
    if torch.cuda.device_count() < 2:
        pytest.skip("The Qwen training release gate requires two visible CUDA devices.")
    model_path = Path("models/Qwen3-VL-2B-Instruct").resolve()
    if not (model_path / "config.json").is_file():
        pytest.skip(f"Qwen3VL model assets not found: {model_path}")

    dataset_path = prepare_qwen_training_dataset(tmp_path)
    fresh_output = tmp_path / "qwen3vl-2b-fp16-padded-fresh"
    fresh_config = write_qwen_training_gate_config(
        tmp_path / "qwen3vl-2b-fp16-padded-fresh.yaml",
        model_type="qwen3vl",
        model_dir=model_path,
        dataset_path=dataset_path,
        output_dir=fresh_output,
        layout="padded",
        packing="none",
        steps=8,
        save_steps=4,
        finetune_mode="lora",
        torch_dtype="float32",
        training_precision="fp16",
    )
    _run_qwen_training_gate(repo_root, fresh_config)

    resumed_output = tmp_path / "qwen3vl-2b-fp16-padded-resumed"
    resumed_config = write_qwen_training_gate_config(
        tmp_path / "qwen3vl-2b-fp16-padded-resumed.yaml",
        model_type="qwen3vl",
        model_dir=model_path,
        dataset_path=dataset_path,
        output_dir=resumed_output,
        layout="padded",
        packing="none",
        steps=8,
        save_steps=4,
        resume_from_checkpoint=fresh_output / "checkpoint-4",
        finetune_mode="lora",
        torch_dtype="float32",
        training_precision="fp16",
    )
    _run_qwen_training_gate(repo_root, resumed_config)

    fresh_checkpoint = fresh_output / "checkpoint-8"
    resumed_checkpoint = resumed_output / "checkpoint-8"
    _assert_checkpoint_state_equivalent(
        fresh_checkpoint,
        resumed_checkpoint,
        weight_filename="adapter_model.safetensors",
    )
    assert (fresh_checkpoint / "scaler.pt").is_file()
    assert (resumed_checkpoint / "scaler.pt").is_file()
    assert _file_sha256(fresh_output / "best" / "adapter_model.safetensors") == _file_sha256(
        resumed_output / "best" / "adapter_model.safetensors"
    )
    for run_dir in (fresh_output, resumed_output):
        _validate_qwen_peft_export(
            repo_root,
            export_path=run_dir / "best",
            model_type="qwen3vl",
            model_path=model_path,
        )
        _assert_standard_qwen_peft_export_reloads(
            base_model_path=model_path,
            export_dir=run_dir / "best",
        )
        _assert_lora_adapter_has_learned_update(run_dir / "best" / "adapter_model.safetensors")


@pytest.mark.integration
@pytest.mark.manual
def test_qwen3vl_multi_rank_lora_varlen_and_export_release_gate(
    tmp_path: Path,
    repo_root: Path,
) -> None:
    if os.environ.get("SHAFT_RUN_QWEN_TRAIN_RELEASE_GATE") != "1":
        pytest.skip("Set SHAFT_RUN_QWEN_TRAIN_RELEASE_GATE=1 to run the CUDA release gate.")
    world_size = int(os.environ.get("SHAFT_QWEN_PACKING_WORLD_SIZE", "2"))
    if world_size < 2:
        raise ValueError("SHAFT_QWEN_PACKING_WORLD_SIZE must be >= 2.")
    if torch.cuda.device_count() < world_size:
        pytest.skip(
            f"The Qwen packing release gate requires {world_size} visible CUDA devices."
        )
    model_path = Path("models/Qwen3-VL-4B-Instruct").resolve()
    if not (model_path / "config.json").is_file():
        pytest.skip(f"Qwen3VL model assets not found: {model_path}")

    dataset_path = prepare_qwen_training_dataset(tmp_path)
    output_dir = tmp_path / "qwen3vl-lora-varlen-fresh"
    config_path = write_qwen_training_gate_config(
        tmp_path / "qwen3vl-lora-varlen.yaml",
        model_type="qwen3vl",
        model_dir=model_path,
        dataset_path=dataset_path,
        output_dir=output_dir,
        layout="varlen",
        packing="greedy",
        steps=2,
        save_steps=1,
        finetune_mode="lora",
    )
    _run_qwen_training_gate(repo_root, config_path, world_size=world_size)

    resumed_output = tmp_path / "qwen3vl-lora-varlen-resumed"
    resumed_config = write_qwen_training_gate_config(
        tmp_path / "qwen3vl-lora-varlen-resumed.yaml",
        model_type="qwen3vl",
        model_dir=model_path,
        dataset_path=dataset_path,
        output_dir=resumed_output,
        layout="varlen",
        packing="greedy",
        steps=2,
        save_steps=1,
        resume_from_checkpoint=output_dir / "checkpoint-1",
        finetune_mode="lora",
    )
    _run_qwen_training_gate(repo_root, resumed_config, world_size=world_size)

    _assert_checkpoint_state_equivalent(
        output_dir / "checkpoint-2",
        resumed_output / "checkpoint-2",
        weight_filename="adapter_model.safetensors",
    )

    export_path = output_dir / "best"
    assert _file_sha256(export_path / "adapter_model.safetensors") == _file_sha256(
        resumed_output / "best" / "adapter_model.safetensors"
    )
    for run_dir in (output_dir, resumed_output):
        _validate_qwen_peft_export(
            repo_root,
            export_path=run_dir / "best",
            model_type="qwen3vl",
            model_path=model_path,
        )
        _assert_standard_qwen_peft_export_reloads(
            base_model_path=model_path,
            export_dir=run_dir / "best",
        )
    summary = json.loads(
        (output_dir / "shaft_training_efficiency.json").read_text(encoding="utf-8")
    )
    assert summary["schema_version"] == TRAINING_EFFICIENCY_SCHEMA_VERSION
    assert summary["complete_history"] is True
    assert summary["aggregate"]["optimizer_steps"] == 2
    assert summary["aggregate"]["update_applied_steps"] > 0
    assert summary["aggregate"]["device_timing_steps"] == 2
    assert summary["aggregate"]["logical_segments"] > summary["aggregate"]["physical_packs"]
    assert (export_path / "adapter_config.json").is_file()
    assert (export_path / "adapter_model.safetensors").is_file()
    _assert_lora_adapter_has_learned_update(export_path / "adapter_model.safetensors")

    reload_output = tmp_path / "qwen3vl-lora-varlen-reloaded"
    reload_config = write_qwen_training_gate_config(
        tmp_path / "qwen3vl-lora-varlen-reloaded.yaml",
        model_type="qwen3vl",
        model_dir=model_path,
        dataset_path=dataset_path,
        output_dir=reload_output,
        layout="varlen",
        packing="greedy",
        steps=1,
        save_steps=None,
        init_from_checkpoint=export_path,
        finetune_mode="lora",
    )
    _run_qwen_training_gate(repo_root, reload_config, world_size=world_size)
    assert (reload_output / "best" / "adapter_model.safetensors").is_file()
