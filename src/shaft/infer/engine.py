from __future__ import annotations

from abc import ABC, abstractmethod
import base64
from dataclasses import dataclass
import copy
import json
import mimetypes
import time
from typing import Any
import urllib.error
import urllib.request

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
    min_pixels: int | None = None
    max_pixels: int | None = None
    backend_options: dict[str, Any] | None = None


@dataclass
class InferResponse:
    text: str
    prompt: str
    output_ids: list[int]
    latency_ms: float | None = None
    backend: str | None = None


class InferAdapter(ABC):
    @abstractmethod
    def run(self, request: InferRequest) -> InferResponse:
        raise NotImplementedError


class HFLocalInferAdapter(InferAdapter):
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

    def run(self, request: InferRequest) -> InferResponse:
        messages = request.messages or self._build_messages(
            user_prompt=request.user_prompt,
            system_prompt=request.system_prompt,
        )
        prompt = self._apply_chat_template(messages)
        with Image.open(request.image_path) as image_obj:
            image = image_obj.convert("RGB")
        effective_min_pixels = request.min_pixels if request.min_pixels is not None else self.min_pixels
        effective_max_pixels = request.max_pixels if request.max_pixels is not None else self.max_pixels
        batch = self._run_processor(
            prompt=prompt,
            image=image,
            min_pixels=effective_min_pixels,
            max_pixels=effective_max_pixels,
        )
        generation = request.generation or self.default_generation
        generated = self._generate(batch=batch, generation=generation)
        prompt_len = int(batch["input_ids"].shape[1])
        output_ids = generated[0][prompt_len:].detach().cpu()
        text = self._decode(output_ids)
        return InferResponse(
            text=text,
            prompt=prompt,
            output_ids=[int(x) for x in output_ids.tolist()],
            backend="hf_local",
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

    def _run_processor(
        self,
        *,
        prompt: str,
        image: Any,
        min_pixels: int | None,
        max_pixels: int | None,
    ) -> dict[str, torch.Tensor]:
        batch = self.model_meta.processor_policy.build_inputs(
            processor=self.processor,
            prompt_texts=[prompt],
            images=[image],
            min_pixels=min_pixels,
            max_pixels=max_pixels,
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
                kwargs["top_k"] = int(generation.top_k)
                kwargs["temperature"] = float(generation.temperature)
            return self.model.generate(**batch, **kwargs)

        gen_config = copy.deepcopy(gen_config)
        gen_config.max_new_tokens = int(generation.max_new_tokens)
        gen_config.do_sample = do_sample
        gen_config.repetition_penalty = float(generation.repetition_penalty)
        if do_sample:
            gen_config.top_p = float(generation.top_p)
            gen_config.top_k = int(generation.top_k)
            gen_config.temperature = float(generation.temperature)
        else:
            # Avoid invalid sampling-only warning flags in GenerationConfig validate.
            gen_config.top_p = 1.0
            gen_config.top_k = int(generation.top_k)
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


class VLLMOpenAIInferAdapter(InferAdapter):
    """Call vLLM OpenAI-compatible API as remote infer backend."""

    def __init__(
        self,
        *,
        endpoint: str,
        model_name: str,
        api_key: str | None = None,
        timeout_seconds: float = 60.0,
        default_generation: InferGenerationConfig | None = None,
    ) -> None:
        endpoint_value = str(endpoint).strip()
        if not endpoint_value:
            raise ValueError("vLLM backend requires a non-empty endpoint.")
        model_name_value = str(model_name).strip()
        if not model_name_value:
            raise ValueError("vLLM backend requires a non-empty model_name.")
        self.endpoint = endpoint_value.rstrip("/")
        self.chat_completions_url = self._resolve_chat_completions_url(self.endpoint)
        self.model_name = model_name_value
        self.api_key = str(api_key).strip() if api_key is not None else None
        self.timeout_seconds = float(timeout_seconds)
        self.default_generation = default_generation or InferGenerationConfig()

    @staticmethod
    def _resolve_chat_completions_url(endpoint: str) -> str:
        normalized = endpoint.rstrip("/")
        if normalized.endswith("/v1"):
            return f"{normalized}/chat/completions"
        return f"{normalized}/v1/chat/completions"

    @staticmethod
    def _encode_image_data_url(image_path: str) -> str:
        mime_type, _ = mimetypes.guess_type(image_path)
        content_type = mime_type or "image/png"
        with open(image_path, "rb") as handle:
            raw = handle.read()
        b64 = base64.b64encode(raw).decode("ascii")
        return f"data:{content_type};base64,{b64}"

    def _build_messages(self, *, image_path: str, user_prompt: str, system_prompt: str) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": self._encode_image_data_url(image_path)},
                    },
                    {"type": "text", "text": user_prompt},
                ],
            }
        )
        return messages

    @staticmethod
    def _extract_text(payload: dict[str, Any]) -> str:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"Invalid vLLM response payload: missing choices. payload={payload!r}")
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise ValueError(f"Invalid vLLM response payload: choices[0] is not object. payload={payload!r}")
        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise ValueError(f"Invalid vLLM response payload: message is missing. payload={payload!r}")
        content = message.get("content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "".join(parts).strip()
        return str(content).strip()

    def run(self, request: InferRequest) -> InferResponse:
        t0 = time.perf_counter()
        generation = request.generation or self.default_generation
        messages = request.messages or self._build_messages(
            image_path=request.image_path,
            user_prompt=request.user_prompt,
            system_prompt=request.system_prompt,
        )
        do_sample = bool(generation.do_sample)
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": int(generation.max_new_tokens),
            "repetition_penalty": float(generation.repetition_penalty),
        }
        mm_processor_kwargs: dict[str, int] = {}
        if request.min_pixels is not None:
            mm_processor_kwargs["min_pixels"] = int(request.min_pixels)
        if request.max_pixels is not None:
            mm_processor_kwargs["max_pixels"] = int(request.max_pixels)
        if mm_processor_kwargs:
            payload["mm_processor_kwargs"] = mm_processor_kwargs

        if request.backend_options:
            for key, value in request.backend_options.items():
                if key in payload:
                    continue
                payload[str(key)] = value

        if do_sample:
            payload["temperature"] = float(generation.temperature)
            payload["top_p"] = float(generation.top_p)
            payload["top_k"] = int(generation.top_k)
        else:
            payload["temperature"] = 0.0
            payload["top_p"] = 1.0
            payload["top_k"] = int(generation.top_k)

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(
            self.chat_completions_url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"vLLM HTTP error {exc.code} at {self.chat_completions_url}: {body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"vLLM request failed for {self.chat_completions_url}: {exc.reason}"
            ) from exc

        try:
            response_payload = json.loads(raw.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("vLLM returned non-JSON response body.") from exc

        text = self._extract_text(response_payload)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return InferResponse(
            text=text,
            prompt=request.user_prompt,
            output_ids=[],
            latency_ms=latency_ms,
            backend="vllm_openai",
        )


class InferEngine:
    def __init__(self, *, adapter: InferAdapter):
        self.adapter = adapter

    @classmethod
    def from_model_config(cls, config: InferModelConfig) -> "InferEngine":
        backend_name = str(config.backend).strip().lower()
        if backend_name == "vllm_openai":
            model_name = str(config.served_model_name or config.model_name_or_path).strip()
            adapter: InferAdapter = VLLMOpenAIInferAdapter(
                endpoint=str(config.endpoint or "").strip(),
                model_name=model_name,
                api_key=config.api_key,
                timeout_seconds=float(config.request_timeout_seconds),
                default_generation=config.generation,
            )
            return cls(adapter=adapter)
        if backend_name != "hf_local":
            raise NotImplementedError(f"Infer backend {backend_name!r} is not implemented yet.")

        runtime_config = RuntimeConfig()
        runtime_config.model.model_type = config.model_type
        runtime_config.model.model_name_or_path = config.model_name_or_path
        runtime_config.model.template = config.template
        runtime_config.model.trust_remote_code = bool(config.trust_remote_code)
        runtime_config.model.attn_implementation = config.attn_implementation
        runtime_config.model.torch_dtype = config.torch_dtype
        runtime_config.model.finetune.mode = config.finetune_mode
        artifacts = build_model_tokenizer_processor(runtime_config)
        adapter = HFLocalInferAdapter(
            model=artifacts.model,
            tokenizer=artifacts.tokenizer,
            processor=artifacts.processor,
            model_meta=artifacts.model_meta,
            template=artifacts.template,
            device=config.device,
            min_pixels=config.min_pixels,
            max_pixels=config.max_pixels,
            default_generation=config.generation,
        )
        return cls(adapter=adapter)

    def run(self, request: InferRequest) -> InferResponse:
        return self.adapter.run(request)
