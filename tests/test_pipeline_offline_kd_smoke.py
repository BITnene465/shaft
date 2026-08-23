from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image
from safetensors.torch import save_file
import torch

from shaft.config import load_config
from shaft.data import SFTCollator
from shaft.model import build_model_tokenizer_processor
from shaft.offline_kd import (
    ARTIFACT_VERSION,
    build_offline_kd_input_contract,
    offline_kd_artifact_identity,
)
from shaft.opd.input_abi import build_opd_input_abi
from shaft.pipeline import run_offline_kd


def test_offline_kd_cpu_smoke_runs_without_teacher_or_rollout(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (8, 8), color=(11, 23, 47)).save(image_path)
    train_path = tmp_path / "train.jsonl"
    manifest_path = tmp_path / "artifact" / "manifest.json"
    manifest_path.parent.mkdir()
    config_path = tmp_path / "offline_kd.yaml"
    config_path.write_text(
        f"""
experiment:
  name: offline-kd-smoke
  output_dir: {tmp_path / 'outputs'}
  seed: 7
model:
  model_type: smoke_vlm
  model_name_or_path: smoke-student
  torch_dtype: float32
  finetune: {{mode: full}}
algorithm: {{name: offline_kd}}
data:
  max_length: 64
  batching:
    grouping: none
    cardinality: fixed
    packing: {{mode: none}}
    layout: padded
  datasets:
    - dataset_name: kd
      source_type: jsonl_offline_kd
      train_path: {train_path}
      use_for_eval: false
  num_workers: 0
  persistent_workers: false
  pin_memory: false
  media_snapshot_id: offline-kd-smoke-v1
train:
  duration: {{unit: steps, value: 1}}
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 1
  learning_rate: 1.0e-3
  optimizer_name: adamw_torch
  scheduler_name: linear
  loss_name: auto
  loss_scale: default
  logging_steps: 1
  save_strategy: no
  report_to: [none]
  save_final_model: false
  save_final_state: false
  load_best_model_at_end: false
  efficiency: {{enabled: false}}
  bf16: false
  fp16: false
  use_cpu: true
  distributed: {{strategy: ddp}}
eval: {{enabled: false}}
offline_kd:
  artifact_manifest: {manifest_path}
  objective:
    mode: dense_logits
    divergence: forward_kl
    temperature: 1.0
  loss:
    ce_weight: 0.5
    kd_weight: 0.5
""",
        encoding="utf-8",
    )
    config = load_config(config_path)
    artifacts = build_model_tokenizer_processor(config)
    collator = SFTCollator(
        model_adapter=artifacts.model_adapter,
        template=artifacts.template,
        processor=artifacts.processor,
        tokenizer=artifacts.tokenizer,
        max_length=64,
        layout="padded",
        packing_mode="none",
    )
    item = {
        "dataset_name": "kd",
        "sample_id": "one",
        "image_path": str(image_path),
        "image_paths": (str(image_path),),
        "image": Image.open(image_path).convert("RGB"),
        "images": (Image.open(image_path).convert("RGB"),),
        "target_text": "blue",
        "target_reasoning_content": None,
        "messages": None,
        "system_prompt": "",
        "user_prompt": "name the color",
        "prompt_args": {},
        "extra": {},
    }
    prepared = collator([item])
    completion_ids = prepared["labels"][0]
    completion_ids = completion_ids[completion_ids.ne(-100)].to(dtype=torch.long)
    input_ids = prepared["input_ids"][0][prepared["attention_mask"][0].bool()].to(
        dtype=torch.long
    )
    vocab_size = build_opd_input_abi(artifacts).logits_vocab_size
    shard_path = manifest_path.parent / "teacher-00001.safetensors"
    image_digest = hashlib.sha256(image_path.read_bytes()).digest()
    media_hasher = hashlib.sha256()
    media_hasher.update(len(image_digest).to_bytes(4, "big"))
    media_hasher.update(image_digest)
    save_file(
        {
            "row_offsets": torch.tensor([0, completion_ids.numel()], dtype=torch.long),
            "completion_token_ids": completion_ids,
            "input_row_offsets": torch.tensor([0, input_ids.numel()], dtype=torch.long),
            "input_token_ids": input_ids,
            "media_sha256": torch.tensor(
                [list(media_hasher.digest())], dtype=torch.uint8
            ),
            "dense_logits": torch.zeros((completion_ids.numel(), vocab_size)),
        },
        str(shard_path),
    )
    teacher = {
        "model": "smoke-teacher",
        "checkpoint_fingerprint": "b" * 64,
    }
    input_abi = build_opd_input_abi(artifacts).to_dict()
    input_contract = build_offline_kd_input_contract(config).to_dict()
    distribution = {
        "mode": "dense_logits",
        "temperature": None,
        "top_k": None,
        "vocab_size": vocab_size,
    }
    build = {
        "source_fingerprint": "c" * 64,
        "denylist_fingerprint": "d" * 64,
    }
    artifact_id = offline_kd_artifact_identity(
        teacher=teacher,
        input_abi=input_abi,
        input_contract=input_contract,
        distribution=distribution,
        build=build,
    )
    manifest_path.write_text(
        json.dumps(
            {
                "version": ARTIFACT_VERSION,
                "artifact_id": artifact_id,
                "teacher": teacher,
                "input_abi": input_abi,
                "input_contract": input_contract,
                "distribution": distribution,
                "build": build,
                "shards": {
                    shard_path.name: hashlib.sha256(shard_path.read_bytes()).hexdigest()
                },
            }
        ),
        encoding="utf-8",
    )
    train_path.write_text(
        json.dumps(
            {
                "image_path": str(image_path),
                "sample_id": "one",
                "user_prompt": "name the color",
                "target_text": "blue",
                "distillation_ref": {
                    "artifact_id": artifact_id,
                    "shard": shard_path.name,
                    "row": 0,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    metrics = run_offline_kd(load_config(config_path))

    assert "train_loss" in metrics
