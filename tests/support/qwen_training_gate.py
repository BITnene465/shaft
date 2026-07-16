from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen3_5Config,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5MoeConfig,
    Qwen3_5MoeForConditionalGeneration,
)


def prepare_qwen_training_dataset(root: Path) -> Path:
    image_path = root / "image.png"
    Image.new("RGB", (256, 256), color=(20, 80, 160)).save(image_path)
    dataset_path = root / "train.jsonl"
    dataset_path.write_text(
        "".join(
            json.dumps(
                {
                    "image_path": str(image_path),
                    "sample_id": f"sample-{index}",
                    "user_prompt": "Return a short JSON object describing the image.",
                    "target_text": json.dumps(
                        {"color": "blue", "id": index},
                        separators=(",", ":"),
                    ),
                },
                ensure_ascii=False,
            )
            + "\n"
            for index in range(8)
        ),
        encoding="utf-8",
    )
    return dataset_path


def prepare_tiny_qwen35_training_assets(
    root: Path,
    *,
    processor_source: Path,
    moe: bool = False,
    attention_implementation: str = "flash_attention_2",
    layer_types: tuple[str, ...] = ("linear_attention", "full_attention"),
) -> tuple[Path, Path]:
    model_dir = root / ("tiny-qwen35-moe" if moe else "tiny-qwen35-dense")
    model_dir.mkdir(parents=True, exist_ok=True)
    text_config = {
        "vocab_size": 248320,
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "linear_key_head_dim": 16,
        "linear_value_head_dim": 16,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "linear_conv_kernel_dim": 4,
        "layer_types": list(layer_types),
        "max_position_embeddings": 512,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10_000.0,
            "mrope_section": [2, 1, 1],
            "mrope_interleaved": True,
        },
        "use_cache": False,
        "_attn_implementation": str(attention_implementation),
    }
    if moe:
        text_config.update(
            {
                "moe_intermediate_size": 32,
                "shared_expert_intermediate_size": 32,
                "num_experts": 4,
                "num_experts_per_tok": 2,
            }
        )
        config_cls = Qwen3_5MoeConfig
        model_cls = Qwen3_5MoeForConditionalGeneration
    else:
        text_config["intermediate_size"] = 128
        config_cls = Qwen3_5Config
        model_cls = Qwen3_5ForConditionalGeneration
    config = config_cls(
        text_config=text_config,
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 4,
            "in_channels": 3,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "out_hidden_size": 64,
            "num_position_embeddings": 256,
            "_attn_implementation": str(attention_implementation),
        },
        image_token_id=248056,
        video_token_id=248057,
        vision_start_token_id=248053,
        vision_end_token_id=248054,
    )
    config._attn_implementation = str(attention_implementation)
    model_cls(config).save_pretrained(model_dir)
    processor = AutoProcessor.from_pretrained(
        processor_source,
        trust_remote_code=False,
        fix_mistral_regex=False,
    )
    processor.save_pretrained(model_dir)
    dataset_path = prepare_qwen_training_dataset(root)
    return model_dir, dataset_path


def write_qwen_training_gate_config(
    path: Path,
    *,
    model_type: str,
    model_dir: Path,
    dataset_path: Path,
    output_dir: Path,
    layout: str,
    packing: str,
    steps: int,
    save_steps: int | None,
    resume_from_checkpoint: Path | None = None,
    init_from_checkpoint: Path | None = None,
    finetune_mode: str = "full",
    use_cpu: bool = False,
    attention_implementation: str = "flash_attention_2",
    torch_dtype: str = "bfloat16",
) -> Path:
    grouping = "length" if layout == "varlen" else "none"
    planning_lines = (
        "    buffer_size: 8\n"
        "    cost_cache_size: 32\n"
        "    resource_budgets:\n"
        "      vision_patches: 1024\n"
        if grouping == "length"
        else ""
    )
    save_strategy = "steps" if save_steps is not None else "no"
    save_lines = (
        f"  save_steps: {int(save_steps)}\n  save_total_limit: 2\n"
        if save_steps is not None
        else ""
    )
    resume_line = (
        ""
        if resume_from_checkpoint is None
        else f"  resume_from_checkpoint: {resume_from_checkpoint}\n"
    )
    init_line = (
        ""
        if init_from_checkpoint is None
        else f"  init_from_checkpoint: {init_from_checkpoint}\n"
    )
    if finetune_mode == "full":
        finetune_lines = "    mode: full\n    target_modules: [auto]\n"
    elif finetune_mode == "lora":
        finetune_lines = (
            "    mode: lora\n"
            "    target_modules: [auto]\n"
            "    lora_r: 8\n"
            "    lora_alpha: 16\n"
            "    lora_dropout: 0.0\n"
        )
    else:
        raise ValueError(f"Unsupported release-gate finetune mode: {finetune_mode!r}")
    content = f"""experiment:
  name: {model_type}-{layout}-release-gate
  output_dir: {output_dir}
  seed: 17
model:
  model_type: {model_type}
  model_name_or_path: {model_dir}
  trust_remote_code: false
  local_files_only: true
  attn_implementation: {attention_implementation}
  torch_dtype: {torch_dtype}
  finetune:
{finetune_lines}algorithm:
  name: sft
  params: {{}}
data:
  batching:
    grouping: {grouping}
    cardinality: fixed
    packing:
      mode: {packing}
    layout: {layout}
{planning_lines}  datasets:
    - dataset_name: tiny
      train_path: {dataset_path}
      enabled: true
      use_for_eval: false
      weight: 1.0
  schedule:
    mixing: concat
    shuffle: true
  media_snapshot_id: qwen-training-release-gate-v1
  num_workers: {0 if use_cpu else 1}
  prefetch_factor: {2 if not use_cpu else 'null'}
  pin_memory: {str(not use_cpu).lower()}
  persistent_workers: {str(not use_cpu).lower()}
  min_pixels: 65536
  max_pixels: 65536
  max_length: 256
  add_eos_token: true
train:
  duration:
    unit: steps
    value: {steps}
  per_device_train_batch_size: {1 if layout == "varlen" else 2}
  gradient_accumulation_steps: 1
  gradient_checkpointing: false
  full_determinism: true
  learning_rate: 1.0e-4
  optimizer_name: adamw_torch
  scheduler_name: cosine
  loss_name: auto
  loss_scale: default
  bf16: {str(torch_dtype == "bfloat16").lower()}
  use_cpu: {str(use_cpu).lower()}
  logging_steps: 1
  save_strategy: {save_strategy}
{save_lines}  load_best_model_at_end: false
  save_final_model: true
  save_final_state: true
  ddp_find_unused_parameters: false
  distributed:
    strategy: ddp
    ddp:
      static_graph: true
  report_to: [none]
  efficiency:
    enabled: true
    device_timing: auto
    persist: true
{resume_line}{init_line}eval:
  enabled: false
  eval_strategy: no
  loss_metrics_enabled: false
  online_metrics_enabled: false
  datasets: {{}}
progress:
  enabled: true
  display: plain
  persist: true
logging:
  level: INFO
  fmt: text
  rank_zero_only: true
"""
    path.write_text(content, encoding="utf-8")
    return path
