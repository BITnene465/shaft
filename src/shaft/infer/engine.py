from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from shaft.config import RuntimeConfig
from shaft.model import build_model_tokenizer_processor

from .schema import InferGenerationConfig, InferModelConfig


@dataclass
class InferRequest:
    image_path: str
    user_prompt: str = ""
    system_prompt: str = ""
    messages: list[dict[str, Any]] | None = None
    generation: InferGenerationConfig | None = None


@dataclass
class InferResponse:
    text: str
    prompt: str
    output_ids: list[int]


class InferEngine:
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        tokenizer: Any,
        processor: Any,
        device: str | None = None,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        default_generation: InferGenerationConfig | None = None,
    ) -> None:
        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(resolved_device)
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer
        self.processor = processor
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.default_generation = default_generation or InferGenerationConfig()

    @classmethod
    def from_runtime_config(
        cls,
        config: RuntimeConfig,
        *,
        device: str | None = None,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        default_generation: InferGenerationConfig | None = None,
    ) -> "InferEngine":
        artifacts = build_model_tokenizer_processor(config)
        return cls(
            model=artifacts.model,
            tokenizer=artifacts.tokenizer,
            processor=artifacts.processor,
            device=device,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            default_generation=default_generation,
        )

    @classmethod
    def from_model_config(cls, config: InferModelConfig) -> "InferEngine":
        runtime_config = RuntimeConfig()
        runtime_config.model.model_type = config.model_type
        runtime_config.model.model_name_or_path = config.model_name_or_path
        runtime_config.model.trust_remote_code = bool(config.trust_remote_code)
        runtime_config.model.attn_implementation = config.attn_implementation
        runtime_config.model.torch_dtype = config.torch_dtype
        runtime_config.model.finetune.mode = config.finetune_mode
        return cls.from_runtime_config(
            runtime_config,
            device=config.device,
            min_pixels=config.min_pixels,
            max_pixels=config.max_pixels,
            default_generation=config.generation,
        )

    def run(self, request: InferRequest) -> InferResponse:
        messages = request.messages or self._build_messages(
            user_prompt=request.user_prompt,
            system_prompt=request.system_prompt,
        )
        prompt = self._apply_chat_template(messages)
        image = Image.open(request.image_path).convert("RGB")
        batch = self._run_processor(prompt=prompt, image=image)
        generation = request.generation or self.default_generation
        generated = self._generate(batch=batch, generation=generation)
        prompt_len = int(batch["input_ids"].shape[1])
        output_ids = generated[0][prompt_len:].detach().cpu()
        text = self._decode(output_ids)
        return InferResponse(
            text=text,
            prompt=prompt,
            output_ids=[int(x) for x in output_ids.tolist()],
        )

    def _build_messages(self, *, user_prompt: str, system_prompt: str) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if system_prompt.strip():
            messages.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                }
            )
        messages.append(
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": user_prompt}],
            }
        )
        return messages

    def _apply_chat_template(self, messages: list[dict[str, Any]]) -> str:
        owner = self.processor if hasattr(self.processor, "apply_chat_template") else self.tokenizer
        return owner.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _run_processor(self, *, prompt: str, image: Any) -> dict[str, torch.Tensor]:
        kwargs: dict[str, Any] = {
            "text": [prompt],
            "images": [image],
            "padding": True,
            "return_tensors": "pt",
        }
        if self.min_pixels is not None:
            kwargs["min_pixels"] = int(self.min_pixels)
        if self.max_pixels is not None:
            kwargs["max_pixels"] = int(self.max_pixels)
        batch = self.processor(**kwargs)
        return self._move_batch_to_device(batch)

    def _move_batch_to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        moved: dict[str, Any] = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                moved[key] = value.to(self.device)
            else:
                moved[key] = value
        return moved

    @torch.no_grad()
    def _generate(self, *, batch: dict[str, Any], generation: InferGenerationConfig) -> torch.Tensor:
        kwargs = {
            "max_new_tokens": int(generation.max_new_tokens),
            "do_sample": bool(generation.do_sample),
            "top_p": float(generation.top_p),
            "repetition_penalty": float(generation.repetition_penalty),
        }
        if generation.do_sample:
            kwargs["temperature"] = float(generation.temperature)
        else:
            kwargs["temperature"] = 0.0
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            kwargs["eos_token_id"] = int(eos_token_id)
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is not None:
            kwargs["pad_token_id"] = int(pad_token_id)
        return self.model.generate(**batch, **kwargs)

    def _decode(self, token_ids: torch.Tensor) -> str:
        ids = token_ids.tolist()
        if hasattr(self.tokenizer, "decode"):
            return str(self.tokenizer.decode(ids, skip_special_tokens=True)).strip()
        if hasattr(self.tokenizer, "batch_decode"):
            decoded = self.tokenizer.batch_decode([ids], skip_special_tokens=True)
            if decoded:
                return str(decoded[0]).strip()
        return " ".join(str(x) for x in ids)

