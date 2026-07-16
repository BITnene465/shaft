from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import copy
import http.client
import json
import socket
import time
from typing import Any
import urllib.error
import urllib.request

import torch

from shaft.config import RuntimeConfig
from shaft.model import ShaftModelAdapter, build_model_meta, build_model_tokenizer_processor
from shaft.model.generation import align_model_generation_config, set_model_use_cache
from shaft.template import Template
from shaft.template import ShaftChatRenderer

from .execution import (
    ShaftInferAdapterCapabilities,
    ShaftInferExecutionControl,
    ShaftInferExecutionControlUnsupportedError,
)
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
    execution: ShaftInferExecutionControl | None = None


@dataclass
class ShaftInferResponse:
    text: str
    prompt: str
    output_ids: list[int]
    latency_ms: float | None = None
    backend: str | None = None


class InferAdapter(ABC):
    capabilities = ShaftInferAdapterCapabilities()

    def validate_execution_control(
        self,
        execution: ShaftInferExecutionControl | None,
    ) -> None:
        if execution is None:
            return
        execution.checkpoint(context=f"Infer adapter {type(self).__name__!r}")
        if execution.requires_deadline and not self.capabilities.supports_deadline:
            raise ShaftInferExecutionControlUnsupportedError(
                f"Infer adapter {type(self).__name__!r} cannot honor an absolute deadline. "
                "The backend is not safely preemptible, so Shaft refuses to start work instead "
                "of leaving background inference running."
            )
        if execution.requires_cancellation and not self.capabilities.supports_cancellation:
            raise ShaftInferExecutionControlUnsupportedError(
                f"Infer adapter {type(self).__name__!r} cannot honor cooperative cancellation. "
                "Shaft refuses to start work instead of leaving background inference running."
            )

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
        self.chat_renderer = ShaftChatRenderer.from_components(
            processor=processor,
            tokenizer=tokenizer,
        )
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.default_generation = default_generation or InferGenerationConfig()

    def _enable_generation_cache(self) -> None:
        _ = set_model_use_cache(self.model, enabled=True)

    def run(self, request: ShaftInferRequest) -> ShaftInferResponse:
        self.validate_execution_control(request.execution)
        effective_min_pixels = (
            request.min_pixels if request.min_pixels is not None else self.min_pixels
        )
        effective_max_pixels = (
            request.max_pixels if request.max_pixels is not None else self.max_pixels
        )
        prepared = self.model_adapter.inference_policy.prepare_local(
            image_path=request.image_path,
            user_prompt=request.user_prompt,
            system_prompt=request.system_prompt,
            messages=request.messages,
            min_pixels=effective_min_pixels,
            max_pixels=effective_max_pixels,
            backend_options=request.backend_options,
            template=self.template,
            renderer=self.chat_renderer,
        )
        batch = self._run_processor(
            prompt=prepared.prompt,
            image=prepared.image,
            min_pixels=prepared.min_pixels,
            max_pixels=prepared.max_pixels,
        )
        generation = request.generation or self.default_generation
        generated = self._generate(batch=batch, generation=generation)
        prompt_len = int(batch["input_ids"].shape[1])
        output_ids = generated[0][prompt_len:].detach().cpu()
        text = self._decode(output_ids)
        return ShaftInferResponse(
            text=text,
            prompt=prepared.prompt,
            output_ids=[int(x) for x in output_ids.tolist()],
            backend="hf_local",
        )

    def _run_processor(
        self,
        *,
        prompt: str,
        image: Any,
        min_pixels: int | None,
        max_pixels: int | None,
    ) -> dict[str, torch.Tensor]:
        processed_batch = self.model_adapter.build_processor_batch(
            processor=self.processor,
            tokenizer=self.tokenizer,
            prompt_texts=[prompt],
            images=[image],
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            input_mode="generation",
        )
        return self._move_batch_to_device(processed_batch.model_inputs)

    def _move_batch_to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        moved: dict[str, Any] = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                moved[key] = value.to(self.device)
            else:
                moved[key] = value
        return moved

    @torch.no_grad()
    def _generate(
        self, *, batch: dict[str, Any], generation: InferGenerationConfig
    ) -> torch.Tensor:
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

    capabilities = ShaftInferAdapterCapabilities(supports_deadline=True)

    def __init__(
        self,
        *,
        endpoint: str,
        model_name: str,
        model_adapter: ShaftModelAdapter,
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
        self.model_adapter = model_adapter
        self.api_key = str(api_key).strip() if api_key is not None else None
        self.timeout_seconds = float(timeout_seconds)
        if self.timeout_seconds <= 0:
            raise ValueError("vLLM timeout_seconds must be > 0.")
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
    def _extract_text(payload: dict[str, Any]) -> str:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"Invalid vLLM response payload: missing choices. payload={payload!r}")
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise ValueError(
                f"Invalid vLLM response payload: choices[0] is not object. payload={payload!r}"
            )
        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise ValueError(
                f"Invalid vLLM response payload: message is missing. payload={payload!r}"
            )
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

    def run(self, request: ShaftInferRequest) -> ShaftInferResponse:
        t0 = time.perf_counter()
        request_started = time.monotonic()
        self.validate_execution_control(request.execution)
        if request.execution is not None:
            request.execution.checkpoint(context="vLLM request preparation")
        generation = request.generation or self.default_generation
        prepared = self.model_adapter.inference_policy.prepare_openai(
            image_path=request.image_path,
            user_prompt=request.user_prompt,
            system_prompt=request.system_prompt,
            messages=request.messages,
            min_pixels=request.min_pixels,
            max_pixels=request.max_pixels,
            backend_options=request.backend_options,
            template_type=self.model_adapter.template_type,
        )
        if request.execution is not None:
            request.execution.checkpoint(context="vLLM request preparation")
        do_sample = bool(generation.do_sample)
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": prepared.messages,
            "max_tokens": int(generation.max_new_tokens),
            "repetition_penalty": float(generation.repetition_penalty),
        }
        for key, value in prepared.backend_options.items():
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
        request_deadline = request_started + self.timeout_seconds
        if request.execution is not None and request.execution.deadline_monotonic is not None:
            request_deadline = min(
                request_deadline,
                request.execution.deadline_monotonic,
            )
        try:
            connect_timeout = _remaining_deadline_timeout(
                request_deadline,
                context="vLLM request",
            )
            with urllib.request.urlopen(req, timeout=connect_timeout) as response:
                raw = _read_http_response_with_deadline(
                    response,
                    deadline_monotonic=request_deadline,
                )
        except urllib.error.HTTPError as exc:
            try:
                body_bytes = _read_http_response_with_deadline(
                    exc,
                    deadline_monotonic=request_deadline,
                )
            finally:
                exc.close()
            body = body_bytes.decode("utf-8", errors="replace")
            raise RuntimeError(
                f"vLLM HTTP error {exc.code} at {self.chat_completions_url}: {body}"
            ) from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (socket.timeout, TimeoutError)):
                raise TimeoutError(
                    f"vLLM request deadline expired for {self.chat_completions_url}."
                ) from exc
            raise RuntimeError(
                f"vLLM request failed for {self.chat_completions_url}: {exc.reason}"
            ) from exc
        except (socket.timeout, TimeoutError) as exc:
            raise TimeoutError(
                f"vLLM request deadline expired for {self.chat_completions_url}."
            ) from exc

        try:
            response_payload = json.loads(raw.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("vLLM returned non-JSON response body.") from exc

        text = self._extract_text(response_payload)
        _remaining_deadline_timeout(
            request_deadline,
            context="vLLM request",
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return ShaftInferResponse(
            text=text,
            prompt=request.user_prompt,
            output_ids=[],
            latency_ms=latency_ms,
            backend="vllm_openai",
        )


def _remaining_deadline_timeout(deadline_monotonic: float, *, context: str) -> float:
    remaining = float(deadline_monotonic) - time.monotonic()
    if remaining <= 0:
        raise TimeoutError(f"{context} deadline expired.")
    return remaining


def _read_http_response_with_deadline(
    response: Any,
    *,
    deadline_monotonic: float,
) -> bytes:
    chunks: list[bytes] = []
    while True:
        remaining = _remaining_deadline_timeout(
            deadline_monotonic,
            context="vLLM response body",
        )
        socket_bound = _set_http_response_socket_timeout(response, remaining)
        if _is_http_response(response) and not socket_bound:
            raise RuntimeError(
                "Cannot bind the vLLM HTTP response socket to the request deadline; "
                "refusing an unbounded response-body read."
            )
        chunk = response.read(64 * 1024)
        if not chunk:
            break
        chunks.append(bytes(chunk))
    _remaining_deadline_timeout(
        deadline_monotonic,
        context="vLLM response body",
    )
    return b"".join(chunks)


def _is_http_response(response: Any) -> bool:
    return isinstance(response, http.client.HTTPResponse) or isinstance(
        getattr(response, "fp", None),
        http.client.HTTPResponse,
    )


def _set_http_response_socket_timeout(response: Any, timeout_seconds: float) -> bool:
    candidates: list[Any] = [response]
    fp = getattr(response, "fp", None)
    if fp is not None:
        candidates.append(fp)
        nested_fp = getattr(fp, "fp", None)
        if nested_fp is not None:
            candidates.append(nested_fp)
    for candidate in tuple(candidates):
        raw = getattr(candidate, "raw", None)
        if raw is not None:
            candidates.append(raw)
    for candidate in candidates:
        socket_obj = getattr(candidate, "_sock", None)
        if socket_obj is None and isinstance(candidate, socket.socket):
            socket_obj = candidate
        settimeout = getattr(socket_obj, "settimeout", None)
        if callable(settimeout):
            settimeout(float(timeout_seconds))
            return True
    return False


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
            model_meta = build_model_meta(config.model_type)
            model_adapter = model_meta.resolve_adapter(
                model_name_or_path=config.model_name_or_path,
                template_type=config.template,
            )
            adapter: InferAdapter = VLLMOpenAIInferAdapter(
                endpoint=str(config.endpoint or "").strip(),
                model_name=model_name,
                model_adapter=model_adapter,
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

    def validate_execution_control(
        self,
        execution: ShaftInferExecutionControl | None,
    ) -> None:
        self.adapter.validate_execution_control(execution)
