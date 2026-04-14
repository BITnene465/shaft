from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from shaft.config import load_config
from shaft.pipeline import run_train


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


def _write_smoke_config(base_dir: Path, mode: str) -> Path:
    train_jsonl, val_jsonl = _write_smoke_data(base_dir)
    output_dir = base_dir / "outputs"
    cfg = base_dir / f"config_{mode}.yaml"
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
    - name: smoke_ds
      train_path: {train_jsonl}
      val_path: {val_jsonl}
  num_workers: 0
  persistent_workers: false
  pin_memory: false
  min_pixels:
  max_pixels:
sft:
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
    bf16: false
    use_cpu: true
  eval:
    enabled: true
    eval_strategy: steps
    eval_steps: 1
    per_device_eval_batch_size: 1
""",
        encoding="utf-8",
    )
    return cfg


def _run_mode(tmp_path: Path, mode: str) -> None:
    cfg_path = _write_smoke_config(tmp_path, mode)
    cfg = load_config(cfg_path)
    metrics = run_train(cfg)
    assert "train_loss" in metrics
    assert "epoch" in metrics


def test_smoke_full(tmp_path: Path) -> None:
    _run_mode(tmp_path, "full")


def test_smoke_lora(tmp_path: Path) -> None:
    _run_mode(tmp_path, "lora")


def test_smoke_dora(tmp_path: Path) -> None:
    _run_mode(tmp_path, "dora")


def test_smoke_qlora(tmp_path: Path) -> None:
    _run_mode(tmp_path, "qlora")
