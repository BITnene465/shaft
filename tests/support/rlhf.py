from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


def write_common_image(base_dir: Path) -> Path:
    image_path = base_dir / "image.png"
    Image.new("RGB", (8, 8), color=(0, 0, 0)).save(image_path)
    return image_path


def write_dpo_config(base_dir: Path) -> Path:
    image_path = write_common_image(base_dir)
    train_jsonl = base_dir / "train_dpo.jsonl"
    val_jsonl = base_dir / "val_dpo.jsonl"
    row = {
        "image_path": str(image_path),
        "chosen_text": "{\"ok\":1}",
        "rejected_text": "{\"ok\":0}",
        "user_prompt": "return json",
    }
    train_jsonl.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    val_jsonl.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    cfg = base_dir / "config_dpo.yaml"
    cfg.write_text(
        f"""
experiment:
  name: smoke-dpo
  output_dir: {base_dir}/outputs_dpo
  seed: 7
model:
  model_type: smoke_vlm
  finetune:
    mode: lora
    target_modules: ["all-linear"]
algorithm:
  name: dpo
data:
  datasets:
    - dataset_name: dpo_ds
      source_type: jsonl_dpo
      train_path: {train_jsonl}
      val_path: {val_jsonl}
  num_workers: 0
  persistent_workers: false
  pin_memory: false
  min_pixels:
  max_pixels:
train:
  duration:
    unit: steps
    value: 1
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
  enabled: false
rlhf:
  enabled: true
  dpo:
    precompute_ref_log_probs: false
""",
        encoding="utf-8",
    )
    return cfg


def write_ppo_config(base_dir: Path) -> Path:
    image_path = write_common_image(base_dir)
    train_jsonl = base_dir / "train_ppo.jsonl"
    val_jsonl = base_dir / "val_ppo.jsonl"
    row = {
        "image_path": str(image_path),
        "prompt": "return json",
        "user_prompt": "return json",
    }
    train_jsonl.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    val_jsonl.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    cfg = base_dir / "config_ppo.yaml"
    cfg.write_text(
        f"""
experiment:
  name: smoke-ppo
  output_dir: {base_dir}/outputs_ppo
  seed: 7
model:
  model_type: smoke_vlm
  finetune:
    mode: lora
    target_modules: ["all-linear"]
algorithm:
  name: ppo
data:
  datasets:
    - dataset_name: ppo_ds
      source_type: jsonl_ppo
      train_path: {train_jsonl}
      val_path: {val_jsonl}
  num_workers: 0
  persistent_workers: false
  pin_memory: false
  min_pixels:
  max_pixels:
train:
  duration:
    unit: steps
    value: 1
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
  enabled: false
rlhf:
  enabled: true
  ppo:
    response_length: 4
    num_ppo_epochs: 1
    num_mini_batches: 1
    local_rollout_forward_batch_size: 1
    num_sample_generations: 0
    allow_untrained_reward_model: true
    allow_text_only_multimodal_ppo: true
""",
        encoding="utf-8",
    )
    return cfg


def write_grpo_config(base_dir: Path) -> Path:
    image_path = write_common_image(base_dir)
    train_jsonl = base_dir / "train_grpo.jsonl"
    val_jsonl = base_dir / "val_grpo.jsonl"
    row = {
        "image_path": str(image_path),
        "target_text": "{\"ok\":1}",
        "user_prompt": "return json",
    }
    train_jsonl.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    val_jsonl.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    cfg = base_dir / "config_grpo.yaml"
    cfg.write_text(
        f"""
experiment:
  name: smoke-grpo
  output_dir: {base_dir}/outputs_grpo
  seed: 7
model:
  model_type: smoke_vlm
  finetune:
    mode: lora
    target_modules: ["all-linear"]
algorithm:
  name: grpo
data:
  datasets:
    - dataset_name: grpo_ds
      source_type: jsonl_sft
      train_path: {train_jsonl}
      val_path: {val_jsonl}
  num_workers: 0
  persistent_workers: false
  pin_memory: false
  min_pixels:
  max_pixels:
train:
  duration:
    unit: steps
    value: 1
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
  enabled: false
rlhf:
  enabled: true
  grpo:
    num_generations: 2
    max_completion_length: 8
    reward_functions:
      - name: exact_match
        codec: json_any
""",
        encoding="utf-8",
    )
    return cfg
