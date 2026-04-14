from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers.generation import GenerationMixin
from transformers import PreTrainedModel, PretrainedConfig
from transformers.processing_utils import ProcessorMixin
from transformers.modeling_outputs import CausalLMOutput

from shaft.config import RuntimeConfig
from shaft.template import build_template_from_meta, resolve_template_meta

from .finetune import apply_finetune_strategy
from .registry import default_model_groups, register_model
from .types import (
    DefaultPeftPolicy,
    ModelArtifacts,
    ModelCapabilities,
    ModelInfo,
    ModelLoader,
    ModelMeta,
    ProcessorPolicy,
)


class SmokeVLMConfig(PretrainedConfig):
    model_type = "smoke_vlm"

    def __init__(self, vocab_size: int = 128, hidden_size: int = 32, **kwargs: Any):
        super().__init__(**kwargs)
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.tie_word_embeddings = False
        # Required by generation cache helpers in newer transformers.
        self.num_hidden_layers = 1


class SmokeVLMModel(PreTrainedModel, GenerationMixin):
    config_class = SmokeVLMConfig

    def __init__(self, config: SmokeVLMConfig):
        super().__init__(config)
        self.embed_tokens = torch.nn.Embedding(config.vocab_size, config.hidden_size)
        self.proj = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **_: Any,
    ) -> CausalLMOutput:
        del attention_mask, pixel_values
        if return_dict is None:
            return_dict = True
        if input_ids is None:
            raise ValueError("input_ids is required.")
        hidden = self.embed_tokens(input_ids)
        hidden = torch.tanh(self.proj(hidden))
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
            )
        hidden_states = (hidden,) if output_hidden_states else None
        output = CausalLMOutput(loss=loss, logits=logits, hidden_states=hidden_states)
        if return_dict:
            return output
        return output.to_tuple()

    def prepare_inputs_for_generation(self, input_ids: torch.Tensor, **kwargs: Any):
        return {"input_ids": input_ids, **kwargs}

@dataclass
class SmokeTokenizer:
    vocab_size: int = 128
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token: str = "<pad>"
    bos_token: str = "<s>"
    eos_token: str = "</s>"

    def _encode(self, text: str) -> list[int]:
        ids = [3 + (ord(ch) % max(self.vocab_size - 4, 1)) for ch in text[:16]]
        return ids or [3]

    def __call__(self, texts, add_special_tokens: bool = False, return_attention_mask: bool = False):
        _ = add_special_tokens, return_attention_mask
        if isinstance(texts, str):
            texts = [texts]
        return {"input_ids": [self._encode(str(text)) for text in texts]}

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        ids = list(token_ids)
        chars = []
        for idx in ids:
            val = int(idx)
            if skip_special_tokens and val in {self.pad_token_id, self.eos_token_id}:
                continue
            # Keep decoding deterministic and printable for smoke tests.
            chars.append(chr(32 + (val % 95)))
        return "".join(chars)

    def batch_decode(self, sequences, skip_special_tokens: bool = True) -> list[str]:
        return [self.decode(seq, skip_special_tokens=skip_special_tokens) for seq in sequences]

    def apply_chat_template(self, messages: list[dict[str, Any]], tokenize: bool = False, add_generation_prompt: bool = True):
        _ = tokenize, add_generation_prompt
        rendered = []
        for message in messages:
            role = str(message.get("role", "user"))
            content = message.get("content", [])
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(str(item.get("text", "")))
                rendered.append(f"{role}:{' '.join(text_parts)}")
            else:
                rendered.append(f"{role}:{str(content)}")
        return "\n".join(rendered)

    def save_pretrained(self, output_dir: str | Path):
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        payload = {
            "vocab_size": self.vocab_size,
            "pad_token_id": self.pad_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "pad_token": self.pad_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
        }
        (target / "smoke_tokenizer.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return [str(target)]


class SmokeProcessor(ProcessorMixin):
    attributes = ["tokenizer"]
    tokenizer_class = "AutoTokenizer"

    def __init__(self, tokenizer: SmokeTokenizer):
        self.tokenizer = tokenizer

    @property
    def pad_token_id(self) -> int:
        return int(self.tokenizer.pad_token_id)

    @property
    def eos_token_id(self) -> int:
        return int(self.tokenizer.eos_token_id)

    def batch_decode(self, sequences, skip_special_tokens: bool = True) -> list[str]:
        return self.tokenizer.batch_decode(sequences, skip_special_tokens=skip_special_tokens)

    def apply_chat_template(self, messages: list[dict[str, Any]], tokenize: bool = False, add_generation_prompt: bool = True):
        return self.tokenizer.apply_chat_template(messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt)

    def __call__(
        self,
        *,
        text: list[str],
        images: list[Any],
        padding: bool = True,
        return_tensors: str = "pt",
        **kwargs: Any,
    ):
        _ = padding, return_tensors, kwargs
        tokenized = self.tokenizer(text, add_special_tokens=False, return_attention_mask=False)["input_ids"]
        max_len = max(len(ids) for ids in tokenized)
        input_ids = []
        attention_mask = []
        for ids in tokenized:
            row = ids + [self.tokenizer.pad_token_id] * (max_len - len(ids))
            input_ids.append(row)
            attention_mask.append([1] * len(ids) + [0] * (max_len - len(ids)))
        pixel_values = torch.zeros((len(images), 3, 4, 4), dtype=torch.float32)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "pixel_values": pixel_values,
        }

    def save_pretrained(self, output_dir: str | Path):
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        (target / "smoke_processor.json").write_text("{\"type\":\"smoke_processor\"}", encoding="utf-8")
        self.tokenizer.save_pretrained(target)
        return [str(target)]


SMOKE_VLM_META = ModelMeta(
    model_type="smoke_vlm",
    family="smoke",
    default_template="smoke_vlm",
    model_groups=default_model_groups("smoke-vlm", "models/smoke-vlm", template="smoke_vlm"),
    capabilities=ModelCapabilities(supports_pixel_budget=False, is_multimodal=True),
    processor_policy=ProcessorPolicy(supports_pixel_budget=False),
    peft_policy=DefaultPeftPolicy(target_modules=["all-linear"]),
    additional_saved_files=("smoke_tokenizer.json", "smoke_processor.json"),
)


@register_model(SMOKE_VLM_META)
class SmokeVLMLoader(ModelLoader):
    def build(self, config: RuntimeConfig, *, model_meta: ModelMeta) -> ModelArtifacts:
        config.model.finetune.target_modules = model_meta.resolve_target_modules(config.model.finetune.target_modules)
        model = SmokeVLMModel(SmokeVLMConfig())
        model.name_or_path = str(config.model.model_name_or_path)
        model.config._name_or_path = str(config.model.model_name_or_path)
        model = apply_finetune_strategy(model, config.model.finetune)
        tokenizer = SmokeTokenizer()
        processor = SmokeProcessor(tokenizer=tokenizer)
        model_info = ModelInfo(
            model_type=model_meta.model_type,
            model_dir=str(config.model.model_name_or_path),
            torch_dtype=config.model.torch_dtype,
            max_model_len=128,
            is_multimodal=model_meta.capabilities.is_multimodal,
            family=model_meta.family,
        )
        template_meta = resolve_template_meta(
            template_type=config.model.template,
            model_meta=model_meta,
            model_info=model_info,
        )
        template = build_template_from_meta(template_meta)
        return ModelArtifacts(
            model=model,
            tokenizer=tokenizer,
            processor=processor,
            model_meta=model_meta,
            model_info=model_info,
            template=template,
        )
