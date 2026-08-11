from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import torch
from transformers import (
    AutoProcessor,
    Qwen3VLConfig,
    Qwen3VLForConditionalGeneration,
    Qwen3_5Config,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5MoeConfig,
    Qwen3_5MoeForConditionalGeneration,
    Qwen3VLMoeConfig,
    Qwen3VLMoeForConditionalGeneration,
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
    max_position_embeddings: int = 512,
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
        "max_position_embeddings": int(max_position_embeddings),
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10_000.0,
            "mrope_section": [2, 1, 1],
            "mrope_interleaved": True,
        },
        "use_cache": True,
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


def prepare_tiny_qwen3vl_moe_training_assets(
    root: Path,
    *,
    processor_source: Path,
) -> tuple[Path, Path]:
    model_dir = root / "tiny-qwen3vl-moe"
    model_dir.mkdir(parents=True, exist_ok=True)
    config = Qwen3VLMoeConfig(
        text_config={
            "vocab_size": 151936,
            "hidden_size": 32,
            "intermediate_size": 64,
            "moe_intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "max_position_embeddings": 512,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10_000.0,
                "mrope_section": [2, 1, 1],
                "mrope_interleaved": True,
            },
            "use_cache": True,
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 4,
            "in_channels": 3,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "out_hidden_size": 32,
            "num_position_embeddings": 256,
            "deepstack_visual_indexes": [0],
        },
        image_token_id=151655,
        video_token_id=151656,
        vision_start_token_id=151652,
        vision_end_token_id=151653,
    )
    Qwen3VLMoeForConditionalGeneration(config).save_pretrained(model_dir)
    processor = AutoProcessor.from_pretrained(
        processor_source,
        trust_remote_code=False,
        fix_mistral_regex=False,
    )
    processor.save_pretrained(model_dir)
    dataset_path = prepare_qwen_training_dataset(root)
    return model_dir, dataset_path


def prepare_tiny_qwen3vl_artifact(
    root: Path,
    *,
    processor_source: Path,
    name: str,
    seed: int,
) -> Path:
    """Build a real Qwen3VL-class artifact small enough for CPU integration gates."""

    model_dir = root / name
    model_dir.mkdir(parents=True, exist_ok=True)
    config = Qwen3VLConfig(
        text_config={
            "vocab_size": 151936,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "max_position_embeddings": 256,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10_000.0,
                "mrope_section": [2, 1, 1],
                "mrope_interleaved": True,
            },
            "attention_dropout": 0.0,
            "use_cache": True,
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 4,
            "in_channels": 3,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "out_hidden_size": 32,
            "deepstack_visual_indexes": [0],
            "num_position_embeddings": 256,
        },
        image_token_id=151655,
        video_token_id=151656,
        vision_start_token_id=151652,
        vision_end_token_id=151653,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        Qwen3VLForConditionalGeneration(config).save_pretrained(model_dir)
    processor = AutoProcessor.from_pretrained(
        processor_source,
        trust_remote_code=False,
        fix_mistral_regex=False,
    )
    processor.save_pretrained(model_dir)
    return model_dir


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
    target_parameters: tuple[str, ...] = (),
    use_cpu: bool = False,
    attention_implementation: str = "flash_attention_2",
    torch_dtype: str = "bfloat16",
    training_precision: str = "bf16",
    distributed_strategy: str = "ddp",
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = False,
    logging_steps: int = 1,
    per_device_train_batch_size: int | None = None,
    num_workers: int | None = None,
    experts_implementation: str | None = None,
    router_aux_loss_weight: float | None = None,
    warmup_ratio: float | None = None,
    bounded_cost_grouping: bool = False,
    bounded_max_tokens_per_microbatch: int = 512,
    bounded_vision_patches: int = 1024,
    data_max_length: int = 256,
    min_pixels: int = 65536,
    max_pixels: int = 65536,
) -> Path:
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be > 0.")
    if logging_steps <= 0:
        raise ValueError("logging_steps must be > 0.")
    if bounded_cost_grouping and (layout != "padded" or packing != "none"):
        raise ValueError("bounded_cost_grouping requires layout='padded' and packing='none'.")
    grouping = (
        "bounded_cost" if bounded_cost_grouping else "length" if layout == "varlen" else "none"
    )
    planning_lines = (
        "    buffer_size: 8\n"
        "    cost_cache_size: 32\n"
        f"    max_tokens_per_microbatch: {int(bounded_max_tokens_per_microbatch)}\n"
        "    resource_budgets:\n"
        f"      vision_patches: {int(bounded_vision_patches)}\n"
        if grouping == "bounded_cost"
        else "    buffer_size: 8\n"
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
        "" if init_from_checkpoint is None else f"  init_from_checkpoint: {init_from_checkpoint}\n"
    )
    if finetune_mode == "full":
        finetune_lines = "    mode: full\n    target_modules: [auto]\n"
    elif finetune_mode == "lora":
        target_parameters_line = (
            "    target_parameters: ["
            + ", ".join(str(value) for value in target_parameters)
            + "]\n"
            if target_parameters
            else ""
        )
        finetune_lines = (
            "    mode: lora\n"
            "    target_modules: [auto]\n"
            f"{target_parameters_line}"
            "    lora_r: 8\n"
            "    lora_alpha: 16\n"
            "    lora_dropout: 0.0\n"
        )
    else:
        raise ValueError(f"Unsupported release-gate finetune mode: {finetune_mode!r}")
    normalized_precision = str(training_precision).strip().lower()
    if normalized_precision not in {"bf16", "fp16", "fp32"}:
        raise ValueError("training_precision must be one of 'bf16', 'fp16', or 'fp32'.")
    bf16 = normalized_precision == "bf16"
    fp16 = normalized_precision == "fp16"
    if per_device_train_batch_size is None:
        resolved_batch_size = 1 if layout == "varlen" else 2
    else:
        resolved_batch_size = int(per_device_train_batch_size)
    if resolved_batch_size <= 0:
        raise ValueError("per_device_train_batch_size must be > 0.")
    resolved_num_workers = (0 if use_cpu else 1) if num_workers is None else int(num_workers)
    if resolved_num_workers < 0:
        raise ValueError("num_workers must be >= 0.")
    workers_enabled = resolved_num_workers > 0
    experts_line = (
        ""
        if experts_implementation is None
        else f"  experts_implementation: {experts_implementation}\n"
    )
    algorithm_params = (
        "{}"
        if router_aux_loss_weight is None
        else "\n    auxiliary_loss_weights:\n"
        f"      router_aux_loss: {float(router_aux_loss_weight)}"
    )
    warmup_line = "" if warmup_ratio is None else f"  warmup_ratio: {float(warmup_ratio)}\n"
    normalized_strategy = str(distributed_strategy).strip().lower()
    if normalized_strategy == "ddp":
        distributed_lines = "    strategy: ddp\n    ddp:\n      static_graph: true\n"
    elif normalized_strategy == "fsdp":
        distributed_lines = (
            "    strategy: fsdp\n"
            "    fsdp:\n"
            "      sharding_strategy: full_shard\n"
            "      auto_wrap_policy: transformer\n"
            "      transformer_layer_cls_to_wrap: [auto]\n"
            "      activation_checkpointing: false\n"
            "      state_dict_type: full_state_dict\n"
            "      use_orig_params: true\n"
            "      sync_module_states: true\n"
        )
    elif normalized_strategy == "deepspeed":
        distributed_lines = (
            "    strategy: deepspeed\n"
            "    deepspeed:\n"
            "      config:\n"
            "        bf16:\n"
            "          enabled: true\n"
            "        train_micro_batch_size_per_gpu: auto\n"
            "        gradient_accumulation_steps: auto\n"
            "        train_batch_size: auto\n"
            "        zero_optimization:\n"
            "          stage: 3\n"
            "          stage3_gather_16bit_weights_on_model_save: true\n"
        )
    else:
        raise ValueError("distributed_strategy must be one of 'ddp', 'fsdp', or 'deepspeed'.")
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
{experts_line.rstrip()}
  torch_dtype: {torch_dtype}
  finetune:
{finetune_lines}algorithm:
  name: sft
  params: {algorithm_params}
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
  num_workers: {resolved_num_workers}
  prefetch_factor: {2 if workers_enabled else "null"}
  pin_memory: {str(not use_cpu).lower()}
  persistent_workers: {str(workers_enabled).lower()}
  min_pixels: {int(min_pixels)}
  max_pixels: {int(max_pixels)}
  max_length: {int(data_max_length)}
  add_eos_token: true
train:
  duration:
    unit: steps
    value: {steps}
  per_device_train_batch_size: {resolved_batch_size}
  gradient_accumulation_steps: {gradient_accumulation_steps}
  gradient_checkpointing: {str(gradient_checkpointing).lower()}
  full_determinism: true
  learning_rate: 1.0e-4
  optimizer_name: adamw_torch
  scheduler_name: cosine
{warmup_line.rstrip()}
  loss_name: auto
  loss_scale: default
  bf16: {str(bf16).lower()}
  fp16: {str(fp16).lower()}
  use_cpu: {str(use_cpu).lower()}
  logging_steps: {logging_steps}
  save_strategy: {save_strategy}
{save_lines}  load_best_model_at_end: false
  save_final_model: true
  save_final_state: true
  ddp_find_unused_parameters: false
  distributed:
{distributed_lines.rstrip()}
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
