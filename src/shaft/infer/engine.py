from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import copy
import json
import time
from typing import Any
import urllib.error
import urllib.request

import torch
from PIL import Image

from shaft.config import RuntimeConfig
from shaft.model import ShaftModelAdapter, build_model_tokenizer_processor
from shaft.model.generation import align_model_generation_config, set_model_use_cache
from shaft.template import Template

from shaft.utils.qwen_pixel_budget import image_to_data_url_with_qwen_pixel_budget
from .schema import InferEngineConfig, InferGenerationConfig


@dataclass
class ShaftInferRequest:
    image_path: str
    user_prompt: str = ""
    system_prompt: str = ""
    messages: list[dict[str, Any]] | None = None
    generation: InferGenerationConfig | None = None
    min_pixels: int | None = None
    max_pixels: int | None = None
    backend_options: dict[str, Any] | None = None


@dataclass
class ShaftInferResponse:
    text: str
    prompt: str
    output_ids: list[int]
    latency_ms: float | None = None
    backend: str | None = None


class InferAdapter(ABC):
    @abstractmethod
    def run(self, request: ShaftInferRequest) -> ShaftInferResponse:
        raise NotImplementedError


class HFLocalInferAdapter(InferAdapter):
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        tokenizer: Any,
        processor: Any,
        model_adapter: ShaftModelAdapter,
        template: Template,
        device: str | None = None,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        default_generation: InferGenerationConfig | None = None,
    ) -> None:
        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._is_sharded = bool(getattr(model, "hf_device_map", None))
        if self._is_sharded:
            self.model = model.eval()
            self.device = _resolve_sharded_input_device(model)
        else:
            self.device = torch.device(resolved_device)
            self.model = model.to(self.device).eval()
        self._enable_generation_cache()
        self.tokenizer = tokenizer
        self.processor = processor
        self.model_adapter = model_adapter
        self.template = template
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.default_generation = default_generation or InferGenerationConfig()

    def _enable_generation_cache(self) -> None:
        _ = set_model_use_cache(self.model, enabled=True)

    def run(self, request: ShaftInferRequest) -> ShaftInferResponse:
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
        return ShaftInferResponse(
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
        batch = self.model_adapter.build_processor_inputs(
            processor=self.processor,
            tokenizer=self.tokenizer,
            prompt_texts=[prompt],
            images=[image],
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            padding_side="left",
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
        self._enable_generation_cache()
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
        gen_config.use_cache = True
        align_model_generation_config(
            gen_config,
            tokenizer=self.tokenizer,
            max_new_tokens=int(generation.max_new_tokens),
            do_sample=do_sample,
            temperature=float(generation.temperature),
            top_p=float(generation.top_p),
            top_k=int(generation.top_k),
            repetition_penalty=float(generation.repetition_penalty),
        )

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
        model_type: str = "",
        template_type: str | None = None,
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
        self.model_type = str(model_type).strip().lower()
        self.template_type = str(template_type).strip().lower() if template_type else None
        self.api_key = str(api_key).strip() if api_key is not None else None
        self.timeout_seconds = float(timeout_seconds)
        self.default_generation = default_generation or InferGenerationConfig()

    @staticmethod
    def _resolve_chat_completions_url(endpoint: str) -> str:
        normalized = endpoint.rstrip("/")
        if normalized.endswith("/chat/completions"):
            return normalized
        if normalized.endswith("/v1"):
            return f"{normalized}/chat/completions"
        return f"{normalized}/v1/chat/completions"

    @staticmethod
    def _encode_image_data_url(
        image_path: str,
        *,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
    ) -> str:
        data_url, _ = image_to_data_url_with_qwen_pixel_budget(
            image_path,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        return data_url

    def _build_messages(
        self,
        *,
        image_path: str,
        user_prompt: str,
        system_prompt: str,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": self._encode_image_data_url(
                                image_path,
                                min_pixels=min_pixels,
                                max_pixels=max_pixels,
                            )
                        },
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

    def _default_chat_template_kwargs(self) -> dict[str, Any]:
        if self.model_type not in {"qwen35vl", "qwen36vl"}:
            return {}
        template_type = self.template_type or "qwen35vl"
        if template_type == "qwen35vl_thinking":
            return {"enable_thinking": True, "preserve_thinking": True}
        return {"enable_thinking": False, "preserve_thinking": False}

    def run(self, request: ShaftInferRequest) -> ShaftInferResponse:
        t0 = time.perf_counter()
        generation = request.generation or self.default_generation
        effective_min_pixels = request.min_pixels
        effective_max_pixels = request.max_pixels
        messages = request.messages or self._build_messages(
            image_path=request.image_path,
            user_prompt=request.user_prompt,
            system_prompt=request.system_prompt,
            min_pixels=effective_min_pixels,
            max_pixels=effective_max_pixels,
        )
        do_sample = bool(generation.do_sample)
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": int(generation.max_new_tokens),
            "repetition_penalty": float(generation.repetition_penalty),
        }
        default_chat_template_kwargs = self._default_chat_template_kwargs()
        if default_chat_template_kwargs:
            payload["chat_template_kwargs"] = default_chat_template_kwargs

        if request.backend_options:
            _validate_no_pixel_budget_backend_options(request.backend_options)
            for key, value in request.backend_options.items():
                if str(key) == "chat_template_kwargs":
                    payload["chat_template_kwargs"] = value
                    continue
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
        return ShaftInferResponse(
            text=text,
            prompt=request.user_prompt,
            output_ids=[],
            latency_ms=latency_ms,
            backend="vllm_openai",
        )


def _validate_no_pixel_budget_backend_options(options: dict[str, Any]) -> None:
    blocked = {
        "min_pixels",
        "min-pixels",
        "max_pixels",
        "max-pixels",
        "mm_processor_kwargs",
        "mm-processor-kwargs",
    }
    for key, value in options.items():
        if value in (None, "", False):
            continue
        normalized = str(key).strip().lower()
        if normalized in blocked:
            raise ValueError(
                f"backend_options must not set {key!r}; pixel budget is applied by "
                "resizing the image before the request."
            )


def _resolve_sharded_input_device(model: torch.nn.Module) -> torch.device:
    model_device = getattr(model, "device", None)
    if model_device is not None:
        return torch.device(model_device)
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict):
        for value in device_map.values():
            if isinstance(value, int):
                return torch.device(f"cuda:{value}")
            normalized = str(value)
            if normalized and normalized not in {"cpu", "disk"}:
                return torch.device(normalized)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ShaftInferEngine:
    def __init__(self, *, adapter: InferAdapter):
        self.adapter = adapter

    @classmethod
    def from_engine_config(cls, config: InferEngineConfig) -> "ShaftInferEngine":
        backend_name = str(config.backend).strip().lower()
        if backend_name == "vllm_openai":
            model_name = str(config.served_model_name or config.model_name_or_path).strip()
            adapter: InferAdapter = VLLMOpenAIInferAdapter(
                endpoint=str(config.endpoint or "").strip(),
                model_name=model_name,
                model_type=config.model_type,
                template_type=config.template,
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
        runtime_config.model.device_map = config.device_map
        runtime_config.model.finetune.mode = config.load_mode
        artifacts = build_model_tokenizer_processor(runtime_config)
        adapter = HFLocalInferAdapter(
            model=artifacts.model,
            tokenizer=artifacts.tokenizer,
            processor=artifacts.processor,
            model_adapter=artifacts.model_adapter,
            template=artifacts.template,
            device=config.device,
            min_pixels=config.min_pixels,
            max_pixels=config.max_pixels,
            default_generation=config.generation,
        )
        return cls(adapter=adapter)

    def run(self, request: ShaftInferRequest) -> ShaftInferResponse:
        return self.adapter.run(request)
