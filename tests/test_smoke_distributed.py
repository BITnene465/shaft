from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image


def _write_data_and_config(base_dir: Path) -> Path:
    image_path = base_dir / "image.png"
    Image.new("RGB", (8, 8), color=(0, 0, 0)).save(image_path)
    train_jsonl = base_dir / "train.jsonl"
    val_jsonl = base_dir / "val.jsonl"
    for path, size in ((train_jsonl, 4), (val_jsonl, 2)):
        with path.open("w", encoding="utf-8") as handle:
            for idx in range(size):
                row = {
                    "image_path": str(image_path),
                    "sample_id": f"d{idx}",
                    "target_text": "{\"ok\":1}",
                    "user_prompt": "return json",
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    cfg = base_dir / "ddp_smoke.yaml"
    cfg.write_text(
        f"""
experiment:
  name: ddp-smoke
  output_dir: {base_dir / "outputs"}
  seed: 9
model:
  model_type: smoke_vlm
  finetune:
    mode: full
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
eval:
  enabled: true
  eval_strategy: steps
  eval_steps: 1
  per_device_eval_batch_size: 1
""",
        encoding="utf-8",
    )
    return cfg


def test_torchrun_train_eval_smoke(tmp_path: Path) -> None:
    torchrun_bin = shutil.which("torchrun")
    if not torchrun_bin:
        pytest.skip("torchrun is not available in current environment.")

    cfg_path = _write_data_and_config(tmp_path)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = "1"
    command = [
        torchrun_bin,
        "--standalone",
        "--nnodes=1",
        "--nproc_per_node=2",
        "scripts/train.py",
        "sft",
        "--config",
        str(cfg_path),
        "--max-steps",
        "1",
    ]
    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr or ""
        if "Operation not permitted" in stderr and "RendezvousConnectionError" in stderr:
            pytest.skip("torchrun rendezvous is blocked by current sandbox/network policy.")
        raise AssertionError(
            f"torchrun smoke failed (code={completed.returncode}).\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
