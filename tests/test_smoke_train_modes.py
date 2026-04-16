from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from shaft.config import load_config
from shaft.pipeline import run_sft


def _write_smoke_data(base_dir: Path) -> tuple[Path, Path]:
    image_path = base_dir / "image.png"
    Image.new("RGB", (8, 8), color=(0, 0, 0)).save(image_path)
    train_jsonl = base_dir / "train.jsonl"
    val_jsonl = base_dir / "val.jsonl"
    for path, size in ((train_jsonl, 2), (val_jsonl, 1)):
        with path.open("w", encoding="utf-8") as handle:
            for idx in range(size):
                row = {
                    "image_path": str(image_path),
                    "sample_id": f"s{idx}",
                    "target_text": "{\"ok\":1}",
                    "user_prompt": "return json",
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return train_jsonl, val_jsonl


def _write_smoke_config(base_dir: Path, mode: str, *, online_eval: bool = False) -> Path:
    train_jsonl, val_jsonl = _write_smoke_data(base_dir)
    output_dir = base_dir / "outputs"
    cfg = base_dir / f"config_{mode}.yaml"
    train_block = """
train:
  epochs: 1
  max_steps: 1
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 1
  learning_rate: 1.0e-3
  optimizer_name: adamw_torch
  scheduler_name: linear
  loss_name: auto
  logging_steps: 1
  save_strategy: no
  report_to: ["none"]
  load_best_model_at_end: false
  save_final_model: false
  save_final_state: false
  bf16: false
  use_cpu: true
"""
    eval_block = """
eval:
  enabled: true
  eval_strategy: steps
  eval_steps: 1
  per_device_eval_batch_size: 1
"""
    if online_eval:
        train_block = """
train:
  epochs: 1
  max_steps: 1
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 1
  learning_rate: 1.0e-3
  optimizer_name: adamw_torch
  scheduler_name: linear
  loss_name: auto
  logging_steps: 1
  save_strategy: steps
  save_steps: 1
  save_total_limit: 1
  report_to: ["none"]
  load_best_model_at_end: true
  save_final_model: false
  save_final_state: false
  bf16: false
  use_cpu: true
"""
        eval_block = """
eval:
  enabled: true
  eval_strategy: steps
  eval_steps: 1
  per_device_eval_batch_size: 1
  online_metrics_enabled: true
  metric_for_best_model: eval_final_score
  greater_is_better: true
  datasets:
    smoke_ds:
      prediction_codec: text
      target_adapter: target_text
      metrics:
        - name: parse_success
      primary_metric: parse_success
      normalizer:
        type: identity
      weight: 1.0
"""
    cfg.write_text(
        f"""
experiment:
  name: smoke-{mode}
  output_dir: {output_dir}
  seed: 7
model:
  model_type: smoke_vlm
  finetune:
    mode: {mode}
    target_modules: ["all-linear"]
    qlora_load_in_4bit: false
algorithm:
  name: sft
data:
  datasets:
    - dataset_name: smoke_ds
      train_path: {train_jsonl}
      val_path: {val_jsonl}
  num_workers: 0
  persistent_workers: false
  pin_memory: false
  min_pixels:
  max_pixels:
{train_block}
{eval_block}
""",
        encoding="utf-8",
    )
    return cfg


def _run_mode(tmp_path: Path, mode: str, *, online_eval: bool = False) -> tuple[Path, dict[str, float]]:
    cfg_path = _write_smoke_config(tmp_path, mode, online_eval=online_eval)
    cfg = load_config(cfg_path)
    metrics = run_sft(cfg)
    assert "train_loss" in metrics
    assert "epoch" in metrics
    return cfg_path, metrics


def test_smoke_full(tmp_path: Path) -> None:
    _run_mode(tmp_path, "full")


def test_smoke_lora(tmp_path: Path) -> None:
    _run_mode(tmp_path, "lora")


def test_smoke_dora(tmp_path: Path) -> None:
    _run_mode(tmp_path, "dora")


def test_smoke_qlora(tmp_path: Path) -> None:
    _run_mode(tmp_path, "qlora")


def test_smoke_online_eval_canary(tmp_path: Path) -> None:
    cfg_path, _ = _run_mode(tmp_path, "full", online_eval=True)
    cfg = load_config(cfg_path)
    trainer_state_path = Path(cfg.experiment.output_dir) / "checkpoint-1" / "trainer_state.json"
    assert trainer_state_path.exists()
    trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    assert float(trainer_state["best_metric"]) == 1.0
    assert str(trainer_state["best_model_checkpoint"]).endswith("checkpoint-1")
