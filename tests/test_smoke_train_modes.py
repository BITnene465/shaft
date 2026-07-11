from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from PIL import Image
from safetensors.torch import load_file

from shaft.config import load_config
from shaft.model.smoke_vlm import SmokeProcessor
from shaft.pipeline import run_sft
from tests.support.configs import write_sft_smoke_config


pytestmark = pytest.mark.smoke


def _run_mode(tmp_path: Path, mode: str, *, online_eval: bool = False) -> tuple[Path, dict[str, float]]:
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


def test_dynamic_cost_aware_smoke_uses_variable_microbatches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg_path = write_sft_smoke_config(
        tmp_path,
        finetune_mode="full",
        train_size=6,
        val_size=1,
        dynamic_cost_aware=True,
        dynamic_target_samples=6,
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
    assert processor_batch_sizes[-2:] == [1, 5]


def test_dynamic_cost_aware_smoke_supports_persistent_workers(tmp_path: Path) -> None:
    cfg_path = write_sft_smoke_config(
        tmp_path,
        finetune_mode="full",
        train_size=6,
        val_size=1,
        dynamic_cost_aware=True,
        dynamic_target_samples=6,
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


def test_cost_aware_same_horizon_checkpoint_resume_and_extension_guard(
    tmp_path: Path,
) -> None:
    cfg_path = write_sft_smoke_config(
        tmp_path,
        finetune_mode="full",
        train_size=4,
        val_size=2,
        cost_aware=True,
        per_device_train_batch_size=2,
    )
    initial = load_config(cfg_path)
    initial.train.duration.value = 2
    initial.train.save_strategy = "steps"
    initial.train.save_steps = 1
    initial.train.save_total_limit = 2
    initial.train.save_final_state = True

    first_metrics = run_sft(initial)
    checkpoint_one = Path(initial.experiment.output_dir) / "checkpoint-1"
    uninterrupted_checkpoint = Path(initial.experiment.output_dir) / "checkpoint-2"
    assert first_metrics["train_loss"] >= 0
    assert (checkpoint_one / "trainer_state.json").is_file()
    assert (checkpoint_one / "shaft_batch_planning_signature.json").is_file()
    assert uninterrupted_checkpoint.is_dir()

    resumed = load_config(cfg_path)
    resumed.train.duration.value = 2
    resumed.experiment.output_dir = str(tmp_path / "resumed_outputs")
    resumed.train.save_strategy = "steps"
    resumed.train.save_steps = 1
    resumed.train.save_total_limit = 2
    resumed.train.save_final_state = True
    resumed.train.resume_from_checkpoint = str(checkpoint_one)
    resumed_metrics = run_sft(resumed)
    resumed_checkpoint = Path(resumed.experiment.output_dir) / "checkpoint-2"

    assert resumed_metrics["train_loss"] >= 0
    root_state = json.loads(
        (Path(resumed.experiment.output_dir) / "trainer_state.json").read_text(
            encoding="utf-8"
        )
    )
    assert int(root_state["global_step"]) == 2
    _assert_checkpoint_training_state_equal(
        uninterrupted_checkpoint,
        resumed_checkpoint,
    )

    extended = load_config(cfg_path)
    extended.train.duration.value = 3
    extended.train.resume_from_checkpoint = str(checkpoint_one)
    with pytest.raises(ValueError, match="resume planning geometry changed"):
        run_sft(extended)


def test_dynamic_cost_aware_exact_resume_and_extension_guard(tmp_path: Path) -> None:
    cfg_path = write_sft_smoke_config(
        tmp_path,
        finetune_mode="full",
        train_size=12,
        val_size=1,
        dynamic_cost_aware=True,
        dynamic_target_samples=6,
        per_device_train_batch_size=3,
        gradient_accumulation_steps=2,
    )
    initial = load_config(cfg_path)
    initial.eval.enabled = False
    initial.train.duration.value = 2
    initial.train.save_strategy = "steps"
    initial.train.save_steps = 1
    initial.train.save_total_limit = 2
    initial.train.save_final_state = True

    first_metrics = run_sft(initial)
    checkpoint_one = Path(initial.experiment.output_dir) / "checkpoint-1"
    uninterrupted_checkpoint = Path(initial.experiment.output_dir) / "checkpoint-2"

    assert first_metrics["train_loss"] >= 0
    assert (checkpoint_one / "shaft_batch_planning_signature.json").is_file()

    resumed = load_config(cfg_path)
    resumed.eval.enabled = False
    resumed.train.duration.value = 2
    resumed.experiment.output_dir = str(tmp_path / "dynamic-resumed-outputs")
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

    extended = load_config(cfg_path)
    extended.eval.enabled = False
    extended.train.duration.value = 3
    extended.train.resume_from_checkpoint = str(checkpoint_one)
    with pytest.raises(ValueError, match="resume planning geometry changed"):
        run_sft(extended)


def test_dynamic_token_target_executes_variable_counts_and_exact_resume(
    tmp_path: Path,
) -> None:
    cfg_path = write_sft_smoke_config(
        tmp_path,
        finetune_mode="full",
        train_size=12,
        val_size=1,
        dynamic_cost_aware=True,
        dynamic_target_supervised_tokens=10,
        dynamic_max_samples_per_microbatch=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
    )
    train_path = tmp_path / "train.jsonl"
    rows = [
        json.loads(line)
        for line in train_path.read_text(encoding="utf-8").splitlines()
    ]
    target_texts = ["x" * 8, "a", "b", "c", "y" * 8, "d"] * 2
    for row, target_text in zip(rows, target_texts, strict=True):
        row["target_text"] = target_text
    train_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    initial = load_config(cfg_path)
    initial.eval.enabled = False
    initial.train.duration.value = 2
    initial.train.save_strategy = "steps"
    initial.train.save_steps = 1
    initial.train.save_total_limit = 2
    initial.train.save_final_state = True

    uninterrupted_metrics = run_sft(initial)
    checkpoint_one = Path(initial.experiment.output_dir) / "checkpoint-1"
    uninterrupted_checkpoint = Path(initial.experiment.output_dir) / "checkpoint-2"
    uninterrupted_state = json.loads(
        (uninterrupted_checkpoint / "trainer_state.json").read_text(encoding="utf-8")
    )
    step_two_loss = next(
        float(entry["loss"])
        for entry in uninterrupted_state["log_history"]
        if int(entry.get("step", -1)) == 2 and "loss" in entry
    )
    assert uninterrupted_metrics["train_samples_per_second"] * uninterrupted_metrics[
        "train_runtime"
    ] == pytest.approx(5, abs=0.1)
    assert uninterrupted_metrics["train_steps_per_second"] * uninterrupted_metrics[
        "train_runtime"
    ] == pytest.approx(2, abs=0.1)

    resumed = load_config(cfg_path)
    resumed.eval.enabled = False
    resumed.train.duration.value = 2
    resumed.experiment.output_dir = str(tmp_path / "dynamic-token-resumed")
    resumed.train.save_strategy = "steps"
    resumed.train.save_steps = 1
    resumed.train.save_total_limit = 2
    resumed.train.save_final_state = True
    resumed.train.resume_from_checkpoint = str(checkpoint_one)

    resumed_metrics = run_sft(resumed)
    resumed_checkpoint = Path(resumed.experiment.output_dir) / "checkpoint-2"
    assert resumed_metrics["train_samples_per_second"] * resumed_metrics[
        "train_runtime"
    ] == pytest.approx(3, abs=0.1)
    assert resumed_metrics["train_steps_per_second"] * resumed_metrics[
        "train_runtime"
    ] == pytest.approx(1, abs=0.1)
    assert resumed_metrics["train_loss"] == pytest.approx(step_two_loss, abs=1e-6)
    _assert_checkpoint_training_state_equal(
        uninterrupted_checkpoint,
        resumed_checkpoint,
    )


def test_cost_aware_resume_rejects_in_place_image_dimension_change(
    tmp_path: Path,
) -> None:
    cfg_path = write_sft_smoke_config(
        tmp_path,
        finetune_mode="full",
        train_size=2,
        val_size=1,
        cost_aware=True,
        per_device_train_batch_size=1,
    )
    initial = load_config(cfg_path)
    initial.train.save_strategy = "steps"
    initial.train.save_steps = 1
    run_sft(initial)
    checkpoint = Path(initial.experiment.output_dir) / "checkpoint-1"

    Image.new("RGB", (16, 8), color=(0, 0, 0)).save(tmp_path / "image.png")
    resumed = load_config(cfg_path)
    resumed.train.resume_from_checkpoint = str(checkpoint)

    with pytest.raises(ValueError, match="cost_fingerprint"):
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
