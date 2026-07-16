from __future__ import annotations

import errno
import fcntl
import json
import logging
import os
from pathlib import Path
import pty
import re
import select
import struct
import subprocess
import sys
import termios
import time

import pytest
import torch
from safetensors.torch import load_file

from shaft.config import load_config
from shaft.model.smoke_vlm import SmokeProcessor
from shaft.observability import PROGRESS_SNAPSHOT_FILENAME
from shaft.pipeline import run_sft
from shaft.training.batch_planning import BATCH_PLANNING_CALLBACK_NAME
from tests.support.configs import write_sft_smoke_config


pytestmark = pytest.mark.smoke


def _set_terminal_size(fd: int, *, columns: int) -> None:
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", 24, columns, 0, 0))


def _run_cli_in_pty(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    columns: int,
    timeout: float,
) -> tuple[int, bytes]:
    master_fd, slave_fd = pty.openpty()
    _set_terminal_size(slave_fd, columns=columns)
    process: subprocess.Popen[bytes] | None = None
    chunks: list[bytes] = []
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=slave_fd,
            stderr=slave_fd,
            close_fds=True,
        )
        os.close(slave_fd)
        slave_fd = -1
        deadline = time.monotonic() + timeout
        exited_at: float | None = None
        while True:
            if time.monotonic() >= deadline:
                process.kill()
                raise TimeoutError(f"PTY command timed out after {timeout}s: {command}")
            ready, _, _ = select.select([master_fd], [], [], 0.05)
            if ready:
                try:
                    chunk = os.read(master_fd, 65_536)
                except OSError as exc:
                    if exc.errno != errno.EIO:
                        raise
                    chunk = b""
                if chunk:
                    chunks.append(chunk)
                    continue
                return process.wait(timeout=5), b"".join(chunks)
            returncode = process.poll()
            if returncode is not None:
                exited_at = time.monotonic() if exited_at is None else exited_at
                if time.monotonic() - exited_at >= 0.2:
                    return returncode, b"".join(chunks)
    finally:
        if slave_fd >= 0:
            os.close(slave_fd)
        os.close(master_fd)
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def _replay_terminal_lines(payload: str) -> list[str]:
    """Replay the terminal controls emitted by the progress sink.

    This intentionally implements only the controls that Shaft owns: carriage
    return, newline, SGR styling, and CSI 2K erase-line.  Replaying the final
    screen catches a progress renderer that accidentally appends a newline on
    every refresh, which raw-frame assertions cannot detect.
    """

    lines: list[list[str]] = [[]]
    row = 0
    column = 0
    index = 0
    while index < len(payload):
        char = payload[index]
        if char == "\r":
            column = 0
            index += 1
            continue
        if char == "\n":
            row += 1
            column = 0
            if row == len(lines):
                lines.append([])
            index += 1
            continue
        if char == "\x1b" and index + 1 < len(payload) and payload[index + 1] == "[":
            match = re.match(r"\x1b\[([0-9;?]*)([@-~])", payload[index:])
            if match is not None:
                parameters, command = match.groups()
                if command == "K" and parameters in {"2", "02"}:
                    lines[row].clear()
                index += len(match.group(0))
                continue
        line = lines[row]
        if column < len(line):
            line[column] = char
        else:
            line.extend(" " for _ in range(column - len(line)))
            line.append(char)
        column += 1
        index += 1
    return ["".join(line).rstrip() for line in lines]


def _run_distributed_progress_console(
    repo_root: Path,
    sync_dir: Path,
    *,
    timeout: int = 60,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = "1"
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc_per_node=2",
            "tests/support/distributed_progress_console.py",
            str(sync_dir),
        ],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def _run_mode(
    tmp_path: Path, mode: str, *, online_eval: bool = False
) -> tuple[Path, dict[str, float]]:
    cfg_path = write_sft_smoke_config(tmp_path, finetune_mode=mode, online_eval=online_eval)
    cfg = load_config(cfg_path)
    metrics = run_sft(cfg)
    assert "train_loss" in metrics
    assert "epoch" in metrics
    return cfg_path, metrics


def test_smoke_full(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    processor_batch_sizes: list[int] = []
    original_call = SmokeProcessor.__call__

    def _counting_call(self, *args, **kwargs):
        processor_batch_sizes.append(len(kwargs["text"]))
        return original_call(self, *args, **kwargs)

    monkeypatch.setattr(SmokeProcessor, "__call__", _counting_call)
    _run_mode(tmp_path, "full")
    assert processor_batch_sizes == [1, 1]


def test_smoke_lora(tmp_path: Path) -> None:
    _run_mode(tmp_path, "lora")


def test_smoke_dora(tmp_path: Path) -> None:
    _run_mode(tmp_path, "dora")


def test_smoke_qlora(tmp_path: Path) -> None:
    _run_mode(tmp_path, "qlora")


def test_cli_non_tty_progress_is_sparse_and_persists_final_state(
    tmp_path: Path,
    repo_root: Path,
) -> None:
    cfg_path = write_sft_smoke_config(
        tmp_path,
        finetune_mode="full",
        train_size=2,
        val_size=1,
        train_steps=1,
    )
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/train.py",
            "sft",
            "--config",
            str(cfg_path),
        ],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    output = f"{completed.stdout}\n{completed.stderr}"
    assert completed.returncode == 0, output
    assert "\r" not in output
    assert "\x1b[" not in output
    progress_lines = [line for line in output.splitlines() if "progress " in line]
    assert any("progress data started" in line for line in progress_lines)
    assert any("progress model succeeded" in line for line in progress_lines)
    assert any("progress train succeeded" in line for line in progress_lines)
    assert len(progress_lines) <= 12

    config = load_config(cfg_path)
    snapshot = json.loads(
        (Path(config.experiment.output_dir) / PROGRESS_SNAPSHOT_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    assert snapshot["status"] == "succeeded"
    assert snapshot["active_task_id"] is None
    assert snapshot["tasks"]["train"]["current"] == 1
    assert snapshot["tasks"]["train"]["total"] == 1


def test_cli_forced_interactive_progress_uses_compact_single_line_contract(
    tmp_path: Path,
    repo_root: Path,
) -> None:
    cfg_path = write_sft_smoke_config(
        tmp_path,
        finetune_mode="full",
        train_size=4,
        val_size=1,
        train_steps=2,
    )
    cfg_path.write_text(
        cfg_path.read_text(encoding="utf-8")
        + "\nprogress:\n"
        + "  enabled: true\n"
        + "  display: interactive\n"
        + "  width: 72\n"
        + "  refresh_interval: 0.01\n"
        + "  leave_completed: false\n"
        + "  persist: true\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/train.py",
            "sft",
            "--config",
            str(cfg_path),
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        timeout=120,
        check=False,
    )
    output = completed.stdout + completed.stderr
    decoded = output.decode("utf-8", errors="replace")

    assert completed.returncode == 0, decoded
    frames = [segment.rstrip() for segment in decoded.split("\r") if segment.startswith("train ")]
    assert frames
    assert any("━" in frame and "─" in frame for frame in frames)
    assert "▏" not in decoded and "·" not in decoded
    assert any("s/it" in frame or "it/s" in frame for frame in frames)
    assert any("lr " in frame for frame in frames)
    assert "[----------]" not in decoded
    assert "progress train" not in decoded


def test_cli_auto_progress_uses_real_pty_color_and_bounded_frames(
    tmp_path: Path,
    repo_root: Path,
) -> None:
    cfg_path = write_sft_smoke_config(
        tmp_path,
        finetune_mode="full",
        train_size=4,
        val_size=1,
        train_steps=2,
    )
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["TERM"] = "xterm-256color"
    env.pop("NO_COLOR", None)
    env.pop("CLICOLOR", None)
    returncode, output = _run_cli_in_pty(
        [
            sys.executable,
            "scripts/train.py",
            "sft",
            "--config",
            str(cfg_path),
        ],
        cwd=repo_root,
        env=env,
        columns=72,
        timeout=120,
    )
    decoded = output.decode("utf-8", errors="replace")

    assert returncode == 0, decoded
    assert "\r\x1b[2K" in decoded
    assert "\x1b[1;36mtrain\x1b[0m" in decoded
    assert "progress train" not in decoded
    assert "▏" not in decoded and "·" not in decoded

    progress_frames: list[str] = []
    for segment in decoded.split("\r\x1b[2K")[1:]:
        frame = segment.split("\r", maxsplit=1)[0].split("\n", maxsplit=1)[0]
        plain = re.sub(r"\x1b\[[0-9;]*m", "", frame)
        if plain.startswith("train "):
            progress_frames.append(plain)

    assert progress_frames
    assert all(len(frame) <= 72 for frame in progress_frames)
    assert any("━" in frame or "╸" in frame for frame in progress_frames)
    assert any("tok " in frame for frame in progress_frames)
    terminal_lines = _replay_terminal_lines(decoded)
    completed_lines = [line for line in terminal_lines if line.startswith("train ")]
    assert len(completed_lines) == 1, terminal_lines
    assert "100%" in completed_lines[0] and "2/2" in completed_lines[0]


def test_required_smoke_keeps_distributed_progress_on_rank_zero(
    tmp_path: Path,
    repo_root: Path,
) -> None:
    completed = _run_distributed_progress_console(repo_root, tmp_path / "progress-sync")
    output = f"{completed.stdout}\n{completed.stderr}"
    if (
        completed.returncode != 0
        and "Operation not permitted" in output
        and "RendezvousConnectionError" in output
    ):
        pytest.skip("torchrun rendezvous is blocked by current sandbox/network policy.")

    assert completed.returncode == 0, output
    assert output.count("rank-zero-warning") == 1
    assert "rank-one-warning-must-be-hidden" not in output
    frames = [line.rstrip() for line in output.splitlines() if line.startswith("train ")]
    assert any("0%" in frame and "0/10k" in frame for frame in frames)
    assert any("0.01%" in frame and "1/10k" in frame for frame in frames)
    assert any("━" in frame or "╸" in frame for frame in frames)
    warning_end = output.index("rank-zero-warning") + len("rank-zero-warning")
    assert output[warning_end:].lstrip().startswith("train")


def test_bounded_cost_grouping_keeps_fixed_per_device_microbatches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg_path = write_sft_smoke_config(
        tmp_path,
        finetune_mode="full",
        train_size=6,
        val_size=1,
        bounded_cost_grouping=True,
        bounded_max_tokens_per_microbatch=2048,
        per_device_train_batch_size=3,
        gradient_accumulation_steps=2,
    )
    train_path = tmp_path / "train.jsonl"
    rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines()]
    prompt_lengths = [300, 1, 1, 1, 1, 10]
    for row, prompt_length in zip(rows, prompt_lengths, strict=True):
        row["user_prompt"] = "x" * prompt_length
    train_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    config = load_config(cfg_path)
    config.eval.enabled = False

    processor_batch_sizes: list[int] = []
    original_call = SmokeProcessor.__call__

    def _counting_call(self, *args, **kwargs):
        processor_batch_sizes.append(len(kwargs["text"]))
        return original_call(self, *args, **kwargs)

    monkeypatch.setattr(SmokeProcessor, "__call__", _counting_call)

    metrics = run_sft(config)

    assert metrics["train_loss"] >= 0
    train_batch_sizes = processor_batch_sizes[-2:]
    assert sum(train_batch_sizes) == 6
    assert train_batch_sizes == [3, 3]


def test_token_budget_uses_real_batch_sizes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    cfg_path = write_sft_smoke_config(
        tmp_path,
        finetune_mode="full",
        train_size=3,
        val_size=1,
        bounded_cost_grouping=True,
        bounded_cardinality="token_budget",
        bounded_max_tokens_per_microbatch=450,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
    )
    train_path = tmp_path / "train.jsonl"
    rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines()]
    for row, prompt_length in zip(rows, [300, 1, 1], strict=True):
        row["user_prompt"] = "x" * prompt_length
    train_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    config = load_config(cfg_path)
    config.eval.enabled = False

    processor_batch_sizes: list[int] = []
    original_call = SmokeProcessor.__call__

    def _counting_call(self, *args, **kwargs):
        processor_batch_sizes.append(len(kwargs["text"]))
        return original_call(self, *args, **kwargs)

    monkeypatch.setattr(SmokeProcessor, "__call__", _counting_call)

    metrics = run_sft(config)

    assert metrics["train_loss"] >= 0
    assert processor_batch_sizes[-2:] == [1, 2]
    assert metrics["train_samples_per_second"] == pytest.approx(
        round(3 / metrics["train_runtime"], 3)
    )
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "[train-batch] local_packs=1..2 global_packs=1..2 "
        "optimizer_packs=2..4 per_device_train_batch_size=2" in message
        for message in messages
    )
    assert not any(
        "Instantaneous batch size per device" in message
        or "Total train batch size (w. parallel, distributed & accumulation)" in message
        for message in messages
    )


def test_bounded_cost_smoke_supports_persistent_workers(tmp_path: Path) -> None:
    cfg_path = write_sft_smoke_config(
        tmp_path,
        finetune_mode="full",
        train_size=6,
        val_size=1,
        bounded_cost_grouping=True,
        per_device_train_batch_size=3,
        gradient_accumulation_steps=2,
    )
    config = load_config(cfg_path)
    config.eval.enabled = False
    config.data.num_workers = 2
    config.data.prefetch_factor = 2
    config.data.persistent_workers = True

    metrics = run_sft(config)

    assert metrics["train_loss"] >= 0
    assert metrics["train_steps_per_second"] > 0


def test_smoke_online_eval_canary(tmp_path: Path) -> None:
    cfg_path, _ = _run_mode(tmp_path, "full", online_eval=True)
    cfg = load_config(cfg_path)
    trainer_state_path = Path(cfg.experiment.output_dir) / "checkpoint-1" / "trainer_state.json"
    assert trainer_state_path.exists()
    trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    assert float(trainer_state["best_metric"]) == 1.0
    assert str(trainer_state["best_model_checkpoint"]).endswith("checkpoint-1")


def test_bounded_exact_resume_and_contract_guard(tmp_path: Path) -> None:
    cfg_path = write_sft_smoke_config(
        tmp_path,
        finetune_mode="full",
        train_size=12,
        val_size=1,
        bounded_cost_grouping=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
    )
    initial = load_config(cfg_path)
    initial.eval.enabled = False
    initial.train.duration.value = 2
    initial.train.save_strategy = "steps"
    initial.train.save_steps = 1
    initial.train.save_total_limit = 2
    initial.train.save_final_state = True

    metrics = run_sft(initial)
    checkpoint_one = Path(initial.experiment.output_dir) / "checkpoint-1"
    uninterrupted_checkpoint = Path(initial.experiment.output_dir) / "checkpoint-2"

    assert metrics["train_loss"] >= 0
    assert _bounded_callback_payload(checkpoint_one)

    resumed = load_config(cfg_path)
    resumed.eval.enabled = False
    resumed.train.duration.value = 2
    resumed.experiment.output_dir = str(tmp_path / "bounded-resumed")
    resumed.train.save_strategy = "steps"
    resumed.train.save_steps = 1
    resumed.train.save_total_limit = 2
    resumed.train.save_final_state = True
    resumed.train.resume_from_checkpoint = str(checkpoint_one)

    resumed_metrics = run_sft(resumed)
    resumed_checkpoint = Path(resumed.experiment.output_dir) / "checkpoint-2"

    assert resumed_metrics["train_loss"] >= 0
    _assert_checkpoint_training_state_equal(
        uninterrupted_checkpoint,
        resumed_checkpoint,
    )
    assert _bounded_callback_payload(uninterrupted_checkpoint) == _bounded_callback_payload(
        resumed_checkpoint
    )

    changed_contract = load_config(cfg_path)
    changed_contract.eval.enabled = False
    changed_contract.train.duration.value = 2
    changed_contract.data.batching.buffer_size = 16
    changed_contract.train.resume_from_checkpoint = str(checkpoint_one)
    with pytest.raises(ValueError, match="changed fields.*buffer_size"):
        run_sft(changed_contract)

    changed_duration = load_config(cfg_path)
    changed_duration.eval.enabled = False
    changed_duration.train.duration.value = 3
    changed_duration.train.resume_from_checkpoint = str(checkpoint_one)
    with pytest.raises(ValueError, match="(?i)training resume contract changed"):
        run_sft(changed_duration)


def test_token_budget_exact_resume_preserves_variable_draw_cursor(
    tmp_path: Path,
) -> None:
    cfg_path = write_sft_smoke_config(
        tmp_path,
        finetune_mode="full",
        train_size=12,
        val_size=1,
        bounded_cost_grouping=True,
        bounded_cardinality="token_budget",
        bounded_max_tokens_per_microbatch=450,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        train_steps=2,
        save_steps=1,
    )
    train_path = tmp_path / "train.jsonl"
    rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["user_prompt"] = "x" * 300
    train_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    initial = load_config(cfg_path)
    initial.eval.enabled = False
    initial.train.save_total_limit = 2
    initial.train.save_final_state = True
    run_sft(initial)
    checkpoint_one = Path(initial.experiment.output_dir) / "checkpoint-1"
    uninterrupted = Path(initial.experiment.output_dir) / "checkpoint-2"

    resumed = load_config(cfg_path)
    resumed.eval.enabled = False
    resumed.experiment.output_dir = str(tmp_path / "token-budget-resumed")
    resumed.train.save_total_limit = 2
    resumed.train.save_final_state = True
    resumed.train.resume_from_checkpoint = str(checkpoint_one)
    run_sft(resumed)
    resumed_checkpoint = Path(resumed.experiment.output_dir) / "checkpoint-2"

    _assert_checkpoint_training_state_equal(uninterrupted, resumed_checkpoint)
    uninterrupted_callback = _bounded_callback_payload(uninterrupted)
    resumed_callback = _bounded_callback_payload(resumed_checkpoint)
    assert uninterrupted_callback == resumed_callback
    assert uninterrupted_callback["args"]["spec"]["cardinality"] == "token_budget"


def test_bounded_resume_rejects_changed_source_execution_contract(tmp_path: Path) -> None:
    cfg_path = write_sft_smoke_config(
        tmp_path,
        finetune_mode="full",
        train_size=4,
        val_size=1,
        bounded_cost_grouping=True,
        gradient_accumulation_steps=1,
        train_steps=2,
        save_steps=1,
    )
    initial = load_config(cfg_path)
    initial.eval.enabled = False
    initial.train.save_total_limit = 2
    run_sft(initial)
    checkpoint = Path(initial.experiment.output_dir) / "checkpoint-1"

    train_path = tmp_path / "train.jsonl"
    rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        row["target_text"] = "changed-supervision-" * 20
    train_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    resumed = load_config(cfg_path)
    resumed.eval.enabled = False
    resumed.train.resume_from_checkpoint = str(checkpoint)

    with pytest.raises(ValueError, match="sample execution changed"):
        run_sft(resumed)


def _assert_checkpoint_training_state_equal(expected: Path, actual: Path) -> None:
    expected_model = load_file(str(expected / "model.safetensors"))
    actual_model = load_file(str(actual / "model.safetensors"))
    assert expected_model.keys() == actual_model.keys()
    for name in expected_model:
        assert torch.equal(expected_model[name], actual_model[name]), name

    expected_optimizer = torch.load(
        expected / "optimizer.pt",
        map_location="cpu",
        weights_only=True,
    )
    actual_optimizer = torch.load(
        actual / "optimizer.pt",
        map_location="cpu",
        weights_only=True,
    )
    _assert_nested_state_equal(expected_optimizer, actual_optimizer)

    expected_scheduler = torch.load(
        expected / "scheduler.pt",
        map_location="cpu",
        weights_only=True,
    )
    actual_scheduler = torch.load(
        actual / "scheduler.pt",
        map_location="cpu",
        weights_only=True,
    )
    _assert_nested_state_equal(expected_scheduler, actual_scheduler)


def _bounded_callback_payload(checkpoint: Path) -> dict:
    trainer_state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    return trainer_state["stateful_callbacks"][BATCH_PLANNING_CALLBACK_NAME]


def _assert_nested_state_equal(expected, actual) -> None:
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert torch.equal(expected, actual)
        return
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert expected.keys() == actual.keys()
        for key in expected:
            _assert_nested_state_equal(expected[key], actual[key])
        return
    if isinstance(expected, (list, tuple)):
        assert type(expected) is type(actual)
        assert len(expected) == len(actual)
        for expected_item, actual_item in zip(expected, actual, strict=True):
            _assert_nested_state_equal(expected_item, actual_item)
        return
    assert expected == actual
