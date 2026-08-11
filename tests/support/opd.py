from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


def write_opd_config(
    base_dir: Path,
    *,
    train_steps: int = 1,
    save_steps: int | None = None,
    save_final_model: bool = False,
    do_sample: bool = False,
    train_size: int = 2,
    gradient_accumulation_steps: int = 1,
    finetune_mode: str = "full",
    use_cpu: bool = True,
    distributed_strategy: str = "ddp",
) -> Path:
    image_path = base_dir / "opd_image.png"
    Image.new("RGB", (8, 8), color=(17, 31, 47)).save(image_path)
    train_jsonl = base_dir / "train_opd.jsonl"
    rows = [
        {
            "image_path": str(image_path),
            "sample_id": f"opd-{index}",
            "user_prompt": f"describe image {index}",
        }
        for index in range(int(train_size))
    ]
    train_jsonl.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    if save_steps is None:
        save_block = "  save_strategy: no\n"
    else:
        save_block = (
            f"  save_strategy: steps\n  save_steps: {int(save_steps)}\n  save_total_limit: 4\n"
        )
    strategy = str(distributed_strategy).strip().lower()
    if strategy == "ddp":
        distributed_block = "  distributed:\n    strategy: ddp\n"
    elif strategy == "fsdp":
        distributed_block = (
            "  distributed:\n"
            "    strategy: fsdp\n"
            "    fsdp:\n"
            "      sharding_strategy: full_shard\n"
            "      auto_wrap_policy: size\n"
            "      min_num_params: 1\n"
            "      activation_checkpointing: false\n"
            "      state_dict_type: full_state_dict\n"
            "      use_orig_params: true\n"
            "      sync_module_states: true\n"
        )
    elif strategy == "deepspeed":
        distributed_block = (
            "  distributed:\n"
            "    strategy: deepspeed\n"
            "    deepspeed:\n"
            "      config:\n"
            "        train_micro_batch_size_per_gpu: auto\n"
            "        gradient_accumulation_steps: auto\n"
            "        train_batch_size: auto\n"
            "        zero_optimization:\n"
            "          stage: 3\n"
            "          stage3_gather_16bit_weights_on_model_save: true\n"
        )
    else:
        raise ValueError("distributed_strategy must be 'ddp', 'fsdp', or 'deepspeed'.")
    config_path = base_dir / "config_opd.yaml"
    config_path.write_text(
        f"""
experiment:
  name: smoke-opd
  output_dir: {base_dir / "outputs_opd"}
  seed: 7
model:
  model_type: smoke_vlm
  model_name_or_path: smoke-student
  torch_dtype: float32
  finetune:
    mode: {str(finetune_mode)}
algorithm:
  name: opd
data:
  max_length: 96
  batching:
    grouping: none
    cardinality: fixed
    packing:
      mode: none
    layout: padded
  datasets:
    - dataset_name: opd_ds
      source_type: jsonl_opd
      train_path: {train_jsonl}
      use_for_eval: false
  num_workers: 0
  media_snapshot_id: smoke-opd-fixture-v1
  persistent_workers: false
  pin_memory: false
  min_pixels:
  max_pixels:
train:
  duration:
    unit: steps
    value: {int(train_steps)}
  per_device_train_batch_size: 1
  gradient_accumulation_steps: {int(gradient_accumulation_steps)}
  learning_rate: 1.0e-3
  optimizer_name: adamw_torch
  scheduler_name: linear
  loss_name: auto
  loss_scale: default
  logging_steps: 1
{save_block}  report_to: ["none"]
  load_best_model_at_end: false
  save_final_model: {str(bool(save_final_model)).lower()}
  save_final_state: false
  efficiency:
    enabled: false
  bf16: false
  fp16: false
  use_cpu: {str(bool(use_cpu)).lower()}
{distributed_block.rstrip()}
eval:
  enabled: false
opd:
  teacher:
    model_type: smoke_vlm
    model_name_or_path: smoke-teacher
    torch_dtype: float32
    local_files_only: true
    trust_remote_code: false
  rollout:
    max_new_tokens: 4
    do_sample: {str(bool(do_sample)).lower()}
    temperature: 1.0
    top_p: 1.0
    top_k: 0
    repetition_penalty: 1.0
  objective:
    divergence: reverse_kl
    temperature: 1.0
""",
        encoding="utf-8",
    )
    return config_path


def write_qwen3vl_opd_config(
    base_dir: Path,
    *,
    student_dir: Path,
    teacher_dir: Path,
    output_dir: Path,
    train_steps: int = 1,
    image_count: int = 1,
    do_sample: bool = False,
    resume_from_checkpoint: Path | None = None,
    train_size: int | None = None,
    gradient_accumulation_steps: int = 1,
    finetune_mode: str = "full",
    use_cpu: bool = True,
    torch_dtype: str = "float32",
    training_precision: str = "fp32",
    attention_implementation: str = "eager",
    learning_rate: float = 1.0e-3,
) -> Path:
    if int(image_count) <= 0:
        raise ValueError("image_count must be > 0.")
    if int(gradient_accumulation_steps) <= 0:
        raise ValueError("gradient_accumulation_steps must be > 0.")
    normalized_finetune_mode = str(finetune_mode).strip().lower()
    if normalized_finetune_mode == "full":
        finetune_block = "    mode: full\n    target_modules: [auto]\n"
    elif normalized_finetune_mode == "lora":
        finetune_block = (
            "    mode: lora\n"
            "    target_modules: [auto]\n"
            "    lora_r: 8\n"
            "    lora_alpha: 16\n"
            "    lora_dropout: 0.0\n"
        )
    else:
        raise ValueError("Qwen3VL OPD gates support finetune_mode='full' or 'lora'.")
    normalized_precision = str(training_precision).strip().lower()
    if normalized_precision not in {"bf16", "fp16", "fp32"}:
        raise ValueError("training_precision must be 'bf16', 'fp16', or 'fp32'.")
    bf16 = normalized_precision == "bf16"
    fp16 = normalized_precision == "fp16"
    image_paths: list[Path] = []
    for image_index in range(int(image_count)):
        image_path = base_dir / f"qwen_opd_image_{image_index}.png"
        if not image_path.is_file():
            Image.new(
                "RGB",
                (256, 256),
                color=(17 + image_index, 31, 47),
            ).save(image_path)
        image_paths.append(image_path)
    train_jsonl = base_dir / "qwen_opd.jsonl"
    if len(image_paths) == 1:
        input_fields = {
            "image_path": str(image_paths[0]),
            "user_prompt": "Describe the dominant color in one word.",
        }
    else:
        content: list[dict[str, str]] = []
        for image_index in range(len(image_paths)):
            content.extend(
                (
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": f"Image {image_index + 1} appears before the next image. ",
                    },
                )
            )
        content.append({"type": "text", "text": "Name the shared dominant color."})
        input_fields = {
            "images": [str(path) for path in image_paths],
            "messages": [{"role": "user", "content": content}],
        }
    resolved_train_size = (
        max(2, int(train_steps)) if train_size is None else int(train_size)
    )
    if resolved_train_size <= 0:
        raise ValueError("train_size must be > 0.")
    train_payload = "".join(
        json.dumps(
            {
                "sample_id": f"qwen-opd-{index}",
                **input_fields,
            },
            ensure_ascii=False,
        )
        + "\n"
        for index in range(resolved_train_size)
    )
    if not train_jsonl.is_file() or train_jsonl.read_text(encoding="utf-8") != train_payload:
        train_jsonl.write_text(train_payload, encoding="utf-8")
    config_path = base_dir / f"{output_dir.name}.yaml"
    resume_line = (
        ""
        if resume_from_checkpoint is None
        else f"  resume_from_checkpoint: {resume_from_checkpoint}\n"
    )
    config_path.write_text(
        f"""
experiment:
  name: qwen3vl-opd-cpu-gate
  output_dir: {output_dir}
  seed: 29
model:
  model_type: qwen3vl
  model_name_or_path: {student_dir}
  local_files_only: true
  trust_remote_code: false
  attn_implementation: {attention_implementation}
  torch_dtype: {torch_dtype}
  finetune:
{finetune_block.rstrip()}
algorithm:
  name: opd
data:
  max_length: 240
  batching:
    grouping: none
    cardinality: fixed
    packing:
      mode: none
    layout: padded
  datasets:
    - dataset_name: qwen_opd
      source_type: jsonl_opd
      train_path: {train_jsonl}
      use_for_eval: false
  num_workers: 0
  media_snapshot_id: qwen-opd-cpu-gate-v1
  persistent_workers: false
  pin_memory: false
  min_pixels: 65536
  max_pixels: 65536
train:
  duration:
    unit: steps
    value: {int(train_steps)}
  per_device_train_batch_size: 1
  gradient_accumulation_steps: {int(gradient_accumulation_steps)}
  full_determinism: true
  learning_rate: {float(learning_rate)}
  optimizer_name: adamw_torch
  scheduler_name: linear
  warmup_ratio: 0.0
  loss_name: auto
  loss_scale: default
  logging_steps: 1
  save_strategy: steps
  save_steps: 1
  save_total_limit: 2
  report_to: ["none"]
  load_best_model_at_end: false
  save_final_model: true
  save_final_state: false
  efficiency:
    enabled: false
{resume_line.rstrip()}
  bf16: {str(bf16).lower()}
  fp16: {str(fp16).lower()}
  use_cpu: {str(bool(use_cpu)).lower()}
eval:
  enabled: false
opd:
  teacher:
    model_type: qwen3vl
    model_name_or_path: {teacher_dir}
    local_files_only: true
    trust_remote_code: false
    attn_implementation: {attention_implementation}
    torch_dtype: {torch_dtype}
  rollout:
    max_new_tokens: 2
    do_sample: {str(bool(do_sample)).lower()}
    temperature: 1.0
    top_p: 1.0
    top_k: 0
    repetition_penalty: 1.0
  objective:
    divergence: reverse_kl
    temperature: 1.0
""",
        encoding="utf-8",
    )
    return config_path
