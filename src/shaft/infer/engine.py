from __future__ import annotations

from dataclasses import dataclass
import copy
from typing import Any

import torch
from PIL import Image

from shaft.config import RuntimeConfig
from shaft.model import ModelMeta, build_model_tokenizer_processor
from shaft.template import Template

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
        model_meta: ModelMeta,
        template: Template,
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
        self.model_meta = model_meta
        self.template = template
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
            model_meta=artifacts.model_meta,
            template=artifacts.template,
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
        runtime_config.model.template = config.template
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
        return self.template.apply_chat_template(
            processor=self.processor,
            tokenizer=self.tokenizer,
            messages=messages,
        )

    def _run_processor(self, *, prompt: str, image: Any) -> dict[str, torch.Tensor]:
        batch = self.model_meta.processor_policy.build_inputs(
            processor=self.processor,
            prompt_texts=[prompt],
            images=[image],
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
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
        do_sample = bool(generation.do_sample)
        gen_config = getattr(self.model, "generation_config", None)
        if gen_config is None:
            kwargs = {
                "max_new_tokens": int(generation.max_new_tokens),
                "do_sample": do_sample,
                "repetition_penalty": float(generation.repetition_penalty),
            }
            if do_sample:
                kwargs["top_p"] = float(generation.top_p)
                kwargs["temperature"] = float(generation.temperature)
            return self.model.generate(**batch, **kwargs)

        gen_config = copy.deepcopy(gen_config)
        gen_config.max_new_tokens = int(generation.max_new_tokens)
        gen_config.do_sample = do_sample
        gen_config.repetition_penalty = float(generation.repetition_penalty)
        if do_sample:
            gen_config.top_p = float(generation.top_p)
            gen_config.temperature = float(generation.temperature)
        else:
            # Avoid invalid sampling-only warning flags in GenerationConfig validate.
            gen_config.top_p = 1.0
            gen_config.top_k = 50
            gen_config.temperature = 1.0
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            gen_config.eos_token_id = int(eos_token_id)
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is not None:
            gen_config.pad_token_id = int(pad_token_id)

        return self.model.generate(
            **batch,
            generation_config=gen_config,
        )

    def _decode(self, token_ids: torch.Tensor) -> str:
        return self.template.decode(tokenizer=self.tokenizer, token_ids=token_ids.tolist())
