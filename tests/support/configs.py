from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import yaml

from shaft.config import RuntimeConfig, load_config


def write_config_yaml(
    base_dir: Path,
    payload: str,
    *,
    filename: str = "config.yaml",
    ensure_explicit_batching: bool = True,
) -> Path:
    config_path = base_dir / filename
    config_path.parent.mkdir(parents=True, exist_ok=True)
    if ensure_explicit_batching:
        parsed = yaml.safe_load(payload) or {}
        data = parsed.setdefault("data", {})
        batching = data.setdefault("batching", {})
        batching.setdefault("grouping", "none")
        batching.setdefault("cardinality", "fixed")
        batching.setdefault("packing", {"mode": "none"})
        batching.setdefault("layout", "padded")
        payload = yaml.safe_dump(parsed, sort_keys=False, allow_unicode=True)
    config_path.write_text(payload, encoding="utf-8")
    return config_path


def load_config_from_yaml(base_dir: Path, payload: str, *, filename: str = "config.yaml") -> RuntimeConfig:
    return load_config(write_config_yaml(base_dir, payload, filename=filename))


def write_smoke_jsonl_dataset(
    base_dir: Path,
    *,
    train_size: int = 2,
    val_size: int = 1,
    image_name: str = "image.png",
) -> tuple[Path, Path]:
    image_path = base_dir / image_name
    Image.new("RGB", (8, 8), color=(0, 0, 0)).save(image_path)
    train_jsonl = base_dir / "train.jsonl"
    val_jsonl = base_dir / "val.jsonl"
    for path, size in ((train_jsonl, train_size), (val_jsonl, val_size)):
        with path.open("w", encoding="utf-8") as handle:
            for idx in range(size):
                row = {
                    "image_path": str(image_path),
                    "sample_id": f"s{idx}",
                    "target_text": json.dumps({"ok": idx}, separators=(",", ":")),
                    "user_prompt": "return json",
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return train_jsonl, val_jsonl


def write_sft_smoke_config(
    base_dir: Path,
    *,
    finetune_mode: str = "full",
    output_name: str = "outputs",
    train_size: int = 2,
    val_size: int = 1,
    online_eval: bool = False,
    distributed: bool = False,
    bounded_cost_grouping: bool = False,
    bounded_cardinality: str = "fixed",
    bounded_max_tokens_per_microbatch: int = 512,
    schedule_mixing: str = "concat",
    schedule_shuffle: bool = False,
    secondary_train_size: int = 0,
    secondary_weight: float = 3.0,
    num_workers: int = 0,
    persistent_workers: bool = False,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 1,
    train_steps: int = 1,
    save_steps: int | None = None,
    save_total_limit: int | None = None,
) -> Path:
    train_jsonl, val_jsonl = write_smoke_jsonl_dataset(
        base_dir,
        train_size=train_size,
        val_size=val_size,
    )
    cfg = base_dir / f"sft_{finetune_mode}_smoke.yaml"
    train_block = _sft_train_block(
        online_eval=online_eval,
        train_steps=train_steps,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    eval_block = _sft_eval_block(online_eval=online_eval)
    target_modules = '    target_modules: ["all-linear"]\n' if not distributed else ""
    dataset_block = (
        "    - dataset_name: smoke_ds\n"
        f"      train_path: {train_jsonl}\n"
        f"      val_path: {val_jsonl}\n"
    )
    if int(secondary_train_size) > 0:
        secondary_dir = base_dir / "secondary_source"
        secondary_dir.mkdir(parents=True, exist_ok=True)
        secondary_train, secondary_val = write_smoke_jsonl_dataset(
            secondary_dir,
            train_size=int(secondary_train_size),
            val_size=1,
        )
        dataset_block += (
            "    - dataset_name: smoke_secondary\n"
            f"      train_path: {secondary_train}\n"
            f"      val_path: {secondary_val}\n"
            f"      weight: {float(secondary_weight)}\n"
            "      use_for_eval: false\n"
        )
    if bounded_cost_grouping:
        buffer_size = max(8, 2 if distributed else 1)
        batching_block = (
            "  batching:\n"
            "    grouping: bounded_cost\n"
            f"    cardinality: {bounded_cardinality}\n"
            "    packing:\n"
            "      mode: none\n"
            "    layout: padded\n"
            f"    buffer_size: {buffer_size}\n"
            "    cost_cache_size: 32\n"
            f"    max_tokens_per_microbatch: {bounded_max_tokens_per_microbatch}\n"
            "  schedule:\n"
            f"    mixing: {str(schedule_mixing).strip().lower()}\n"
            f"    shuffle: {str(bool(schedule_shuffle)).lower()}\n"
        )
    else:
        batching_block = (
            "  batching:\n"
            "    grouping: none\n"
            "    cardinality: fixed\n"
            "    packing:\n"
            "      mode: none\n"
            "    layout: padded\n"
        )
    cfg.write_text(
        f"""
experiment:
  name: smoke-{finetune_mode}
  output_dir: {base_dir / output_name}
  seed: 7
model:
  model_type: smoke_vlm
  finetune:
    mode: {finetune_mode}
{target_modules}    qlora_load_in_4bit: false
algorithm:
  name: sft
data:
{batching_block}  datasets:
{dataset_block.rstrip()}
  num_workers: {int(num_workers)}
  media_snapshot_id: smoke-fixture-v1
  persistent_workers: {str(bool(persistent_workers)).lower()}
  pin_memory: false
  min_pixels:
  max_pixels:
{train_block}
{eval_block}
""",
        encoding="utf-8",
    )
    return cfg


def _sft_train_block(
    *,
    online_eval: bool,
    train_steps: int,
    save_steps: int | None,
    save_total_limit: int | None,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
) -> str:
    if online_eval:
        resolved_save_steps = 1 if save_steps is None else int(save_steps)
        resolved_save_limit = 1 if save_total_limit is None else int(save_total_limit)
        save_block = (
            "  save_strategy: steps\n"
            f"  save_steps: {resolved_save_steps}\n"
            f"  save_total_limit: {resolved_save_limit}\n"
            "  load_best_model_at_end: true\n"
        )
    elif save_steps is not None:
        resolved_save_limit = 2 if save_total_limit is None else int(save_total_limit)
        save_block = (
            "  save_strategy: steps\n"
            f"  save_steps: {int(save_steps)}\n"
            f"  save_total_limit: {resolved_save_limit}\n"
            "  load_best_model_at_end: false\n"
        )
    else:
        save_block = "  save_strategy: no\n  load_best_model_at_end: false\n"
    return f"""
train:
  duration:
    unit: steps
    value: {int(train_steps)}
  per_device_train_batch_size: {int(per_device_train_batch_size)}
  gradient_accumulation_steps: {int(gradient_accumulation_steps)}
  learning_rate: 1.0e-3
  optimizer_name: adamw_torch
  scheduler_name: linear
  loss_name: auto
  logging_steps: 1
{save_block}  report_to: ["none"]
  save_final_model: false
  save_final_state: false
  bf16: false
  use_cpu: true
"""


def _sft_eval_block(*, online_eval: bool) -> str:
    if not online_eval:
        return """
eval:
  enabled: true
  eval_strategy: steps
  eval_steps: 1
  per_device_eval_batch_size: 1
"""
    return """
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
