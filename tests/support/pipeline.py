from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from shaft.config import FinetuneConfig, RuntimeConfig, load_config
from shaft.model import build_model_meta
from shaft.model.finetune_plan import build_resolved_finetune_plan
from shaft.template import build_template


def fsdp_enabled(value) -> bool:
    if isinstance(value, bool):
        return value
    return bool(value)


def fsdp_option_values(value) -> list[str]:
    if isinstance(value, bool):
        return []
    return [getattr(option, "value", str(option)) for option in value]


def write_sft_pipeline_config(
    tmp_path: Path,
    *,
    hooks: list[str] | None = None,
    loss_scale: str = "default",
) -> RuntimeConfig:
    train_jsonl = tmp_path / "train.jsonl"
    val_jsonl = tmp_path / "val.jsonl"
    image = tmp_path / "img.png"

    Image.new("RGB", (8, 8), color=(0, 0, 0)).save(image)
    train_jsonl.write_text(
        f'{{"image_path":"{image}","target_text":"{{}}","user_prompt":"x"}}\n',
        encoding="utf-8",
    )
    val_jsonl.write_text(
        f'{{"image_path":"{image}","target_text":"{{}}","user_prompt":"x"}}\n',
        encoding="utf-8",
    )
    hooks_yaml = f"  hooks: {hooks}\n" if hooks is not None else ""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        f"""
experiment:
  name: test
  output_dir: {tmp_path}/out
data:
  batching:
    grouping: none
    cardinality: fixed
    packing:
      mode: none
    layout: padded
  datasets:
    - dataset_name: ds
      train_path: {train_jsonl}
      val_path: {val_jsonl}
algorithm:
  name: sft
plugins:
{hooks_yaml if hooks_yaml else '  hooks: []'}
train:
  duration:
    unit: steps
    value: 1
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 1
  learning_rate: 1.0e-5
  loss_scale: {loss_scale}
  use_cpu: true
  report_to: ["none"]
  load_best_model_at_end: false
  save_final_model: false
  save_final_state: false
eval:
  enabled: false
""",
        encoding="utf-8",
    )
    return load_config(cfg_path)


class FakePipelineTokenizer:
    eos_token_id = 2
    pad_token_id = 0
    bos_token_id = None
    eos_token = "</s>"

    def __call__(self, texts, add_special_tokens=False, return_attention_mask=False):
        _ = add_special_tokens, return_attention_mask
        return {"input_ids": [[1] for _ in texts]}


class FakePipelineProcessor:
    tokenizer = FakePipelineTokenizer()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        _ = messages, tokenize, add_generation_prompt
        return "prompt"

    def __call__(self, text, images, padding=True, return_tensors="pt", **kwargs):
        _ = text, images, padding, return_tensors, kwargs
        return {
            "input_ids": torch.tensor([[1], [1]], dtype=torch.long),
            "attention_mask": torch.tensor([[1], [1]], dtype=torch.long),
            "pixel_values": torch.randn(2, 3, 2, 2),
        }


class FakePipelineModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            use_cache=False,
            eos_token_id=99,
            bos_token_id=98,
            pad_token_id=97,
            text_config=SimpleNamespace(eos_token_id=99, bos_token_id=98, pad_token_id=97),
        )
        self.generation_config = SimpleNamespace(
            use_cache=False,
            max_new_tokens=32,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            repetition_penalty=1.0,
            eos_token_id=99,
            bos_token_id=98,
            pad_token_id=97,
        )

    def forward(self, **kwargs):
        _ = kwargs
        return type("Out", (), {"loss": torch.tensor(0.1)})


class FakePipelineTrainResult:
    metrics = {"train_loss": 0.1}


class FakePipelineTrainer:
    last_kwargs = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        type(self).last_kwargs = kwargs

    def train(self, resume_from_checkpoint=None):
        _ = resume_from_checkpoint
        return FakePipelineTrainResult()

    def save_model(self, *args, **kwargs):
        _ = args, kwargs
        return None

    def save_state(self):
        return None


def build_fake_model_artifacts(
    *,
    model: FakePipelineModel | None = None,
    include_finetune_plan: bool = False,
):
    adapter = build_model_meta("smoke_vlm").resolve_adapter(model_name_or_path="models/Smoke-VLM")
    fake_model = model or FakePipelineModel()
    values = {
        "model": fake_model,
        "tokenizer": FakePipelineTokenizer(),
        "processor": FakePipelineProcessor(),
        "model_meta": build_model_meta("smoke_vlm"),
        "model_adapter": adapter,
        "template": build_template("smoke_vlm"),
    }
    if include_finetune_plan:
        values["finetune_plan"] = build_resolved_finetune_plan(
            FakePipelineModel(),
            FinetuneConfig(mode="full"),
            model_adapter=adapter,
        )
    return type("Artifacts", (), values)()
