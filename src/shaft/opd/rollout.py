from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import json
from typing import Any

import torch

from shaft.config import OPDRolloutConfig
from shaft.model.generation import restore_model_use_cache, set_model_use_cache


@dataclass(frozen=True, slots=True)
class OPDRolloutRequest:
    """Canonical rollout request shared by local HF and external vLLM backends."""

    model: torch.nn.Module
    model_inputs: dict[str, Any]
    generation_prompt_token_ids: tuple[tuple[int, ...], ...]
    prompt_token_ids: tuple[tuple[int, ...], ...]
    ordered_images: tuple[tuple[Any, ...] | None, ...]
    sample_ids: tuple[str, ...]
    request_ids: tuple[str, ...]
    model_version: int
    accelerator: Any
    processing_class: Any

    def __post_init__(self) -> None:
        input_ids = self.model_inputs.get("input_ids")
        attention_mask = self.model_inputs.get("attention_mask")
        if not torch.is_tensor(input_ids) or input_ids.ndim != 2:
            raise ValueError("OPD rollout request requires 2-D input_ids.")
        if not torch.is_tensor(attention_mask) or tuple(attention_mask.shape) != tuple(
            input_ids.shape
        ):
            raise ValueError("OPD rollout request attention_mask must match input_ids.")
        batch_size = int(input_ids.shape[0])
        for field_name, values in (
            ("generation_prompt_token_ids", self.generation_prompt_token_ids),
            ("prompt_token_ids", self.prompt_token_ids),
            ("ordered_images", self.ordered_images),
            ("sample_ids", self.sample_ids),
            ("request_ids", self.request_ids),
        ):
            if len(values) != batch_size:
                raise ValueError(f"OPD rollout {field_name} must match batch cardinality.")
        if any(not value for value in self.generation_prompt_token_ids):
            raise ValueError(
                "OPD rollout generation_prompt_token_ids rows must not be empty."
            )
        if any(not value for value in self.prompt_token_ids):
            raise ValueError("OPD rollout prompt_token_ids rows must not be empty.")
        if any(not value.strip() for value in self.request_ids):
            raise ValueError("OPD rollout request IDs must not be empty.")
        if len(set(self.request_ids)) != len(self.request_ids):
            raise ValueError("OPD rollout request IDs must be unique within a local batch.")
        if type(self.model_version) is not int or self.model_version < 0:
            raise ValueError("OPD rollout model_version must be a non-negative integer.")
        for row_index, prompt_ids in enumerate(self.prompt_token_ids):
            local_ids = input_ids[row_index][attention_mask[row_index].to(dtype=torch.bool)]
            if tuple(int(value) for value in local_ids.tolist()) != prompt_ids:
                raise ValueError(
                    "OPD rollout prompt token IDs differ from local processor output; "
                    f"row={row_index}."
                )


@dataclass(frozen=True, slots=True)
class OPDRolloutResult:
    """Canonical on-policy completion batch consumed by OPD scoring."""

    sequences: torch.Tensor
    attention_mask: torch.Tensor
    completion_mask: torch.Tensor


class OPDRolloutBackend(ABC):
    """Generate one OPD completion batch without coupling the trainer to a backend."""

    name: str
    exact_resume_supported: bool = False
    requires_raw_media: bool = False

    def __init__(self, config: OPDRolloutConfig, *, seed: int = 0) -> None:
        self.config = config
        self.seed = int(seed)
        self.telemetry = None

    def bind_telemetry(self, telemetry: Any) -> None:
        self.telemetry = telemetry

    def prepare(
        self,
        *,
        model: torch.nn.Module,
        accelerator: Any,
        processing_class: Any,
    ) -> None:
        """Bind runtime resources before Trainer applies distributed wrappers."""

        _ = model, accelerator, processing_class

    def _phase(self, name: str):
        return nullcontext() if self.telemetry is None else self.telemetry.phase(name)

    @abstractmethod
    def generate(self, request: OPDRolloutRequest) -> OPDRolloutResult:
        raise NotImplementedError

    @staticmethod
    def completion_mask(
        completion_ids: torch.Tensor,
        *,
        eos_token_id: int | None,
        pad_token_id: int | None,
    ) -> torch.Tensor:
        mask = torch.ones_like(completion_ids, dtype=torch.bool)
        if pad_token_id is not None and pad_token_id != eos_token_id:
            mask &= completion_ids.ne(int(pad_token_id))
        if eos_token_id is not None:
            eos = completion_ids.eq(int(eos_token_id))
            eos_count = eos.to(dtype=torch.int64).cumsum(dim=-1)
            before_or_at_first_eos = eos_count.eq(0) | (eos & eos_count.eq(1))
            mask &= before_or_at_first_eos
        return mask

    @staticmethod
    def _tokenizer(processing_class: Any) -> Any:
        tokenizer = processing_class
        if tokenizer is not None and hasattr(tokenizer, "tokenizer"):
            tokenizer = tokenizer.tokenizer
        return tokenizer


class HFLocalOPDRolloutBackend(OPDRolloutBackend):
    """Stateless local-HF rollout whose randomness is owned by Torch RNG state."""

    name = "hf_local"
    exact_resume_supported = True

    def _generation_kwargs(self, processing_class: Any) -> dict[str, Any]:
        tokenizer = self._tokenizer(processing_class)
        kwargs: dict[str, Any] = {
            "max_new_tokens": int(self.config.max_new_tokens),
            "do_sample": bool(self.config.do_sample),
            "repetition_penalty": float(self.config.repetition_penalty),
            "pad_token_id": getattr(tokenizer, "pad_token_id", None),
            "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        }
        if kwargs["do_sample"]:
            kwargs.update(
                {
                    "temperature": float(self.config.temperature),
                    "top_p": float(self.config.top_p),
                    "top_k": int(self.config.top_k),
                }
            )
        return {key: value for key, value in kwargs.items() if value is not None}

    def generate(self, request: OPDRolloutRequest) -> OPDRolloutResult:
        input_ids = request.model_inputs["input_ids"]
        attention_mask = request.model_inputs["attention_mask"]
        prompt_width = int(input_ids.shape[1])
        generation_model = request.accelerator.unwrap_model(request.model)
        was_training = bool(generation_model.training)
        previous_use_cache = set_model_use_cache(generation_model, enabled=True)
        generation_model.eval()
        try:
            with self._phase("rollout_generate"):
                # Generated token IDs immediately feed the trainable student score
                # pass. inference_mode would tag those tensors as inference-only,
                # which FSDP correctly rejects when autograd saves downstream values.
                with torch.no_grad():
                    generated = generation_model.generate(
                        **request.model_inputs,
                        **self._generation_kwargs(request.processing_class),
                    )
        finally:
            restore_model_use_cache(generation_model, previous_use_cache)
            if was_training:
                generation_model.train()
        sequences = getattr(generated, "sequences", generated)
        if not torch.is_tensor(sequences) or sequences.ndim != 2:
            raise TypeError("OPD student.generate must return a 2-D token tensor.")
        sequences = sequences.clone()
        if int(sequences.shape[0]) != int(input_ids.shape[0]):
            raise ValueError("OPD rollout changed batch cardinality.")
        if int(sequences.shape[1]) <= prompt_width:
            raise ValueError("OPD rollout produced no completion tokens.")
        if not torch.equal(sequences[:, :prompt_width], input_ids):
            raise ValueError("OPD rollout did not preserve the exact padded prompt tokens.")

        completion_ids = sequences[:, prompt_width:]
        generation_kwargs = self._generation_kwargs(request.processing_class)
        completion_mask = self.completion_mask(
            completion_ids,
            eos_token_id=generation_kwargs.get("eos_token_id"),
            pad_token_id=generation_kwargs.get("pad_token_id"),
        )
        self._validate_nonempty(completion_mask, request.sample_ids)
        return self._assemble_result(
            sequences=sequences,
            prompt_attention_mask=attention_mask,
            completion_mask=completion_mask,
        )

    @staticmethod
    def _validate_nonempty(
        completion_mask: torch.Tensor,
        sample_ids: tuple[str, ...],
    ) -> None:
        empty_rows = completion_mask.sum(dim=-1).eq(0)
        if bool(empty_rows.any().item()):
            empty_sample_ids = [
                sample_ids[index]
                for index in empty_rows.nonzero(as_tuple=False).reshape(-1).tolist()
            ]
            raise ValueError(
                f"OPD rollout produced an empty completion; sample_ids={empty_sample_ids}."
            )

    @staticmethod
    def _assemble_result(
        *,
        sequences: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        completion_mask: torch.Tensor,
    ) -> OPDRolloutResult:
        full_attention_mask = torch.cat(
            [prompt_attention_mask, completion_mask.to(dtype=prompt_attention_mask.dtype)],
            dim=-1,
        )
        full_completion_mask = torch.cat(
            [torch.zeros_like(prompt_attention_mask, dtype=torch.bool), completion_mask],
            dim=-1,
        )
        return OPDRolloutResult(
            sequences=sequences,
            attention_mask=full_attention_mask,
            completion_mask=full_completion_mask,
        )


def _load_vllm_generation_type():
    try:
        from trl.generation.vllm_generation import VLLMGeneration
    except Exception as exc:  # noqa: BLE001 - optional backend import
        raise ImportError(
            "OPD vLLM rollout requires compatible TRL/vLLM dependencies; install "
            '`uv pip install -e ".[rlhf,serve]"`.'
        ) from exc
    return VLLMGeneration


class _ExplicitCUDAAcceleratorView:
    """Delegate Accelerator while publishing the concrete CUDA index PyNccl requires."""

    def __init__(self, accelerator: Any) -> None:
        self._accelerator = accelerator
        device = torch.device(accelerator.device)
        self.device = (
            torch.device("cuda", torch.cuda.current_device())
            if device.type == "cuda" and device.index is None
            else device
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._accelerator, name)


class VLLMOPDRolloutBackend(OPDRolloutBackend):
    """On-policy vLLM adapter with versioned student-weight synchronization."""

    name = "vllm"
    exact_resume_supported = True
    requires_raw_media = True

    def __init__(self, config: OPDRolloutConfig, *, seed: int = 0) -> None:
        super().__init__(config, seed=seed)
        self._generation: Any | None = None
        self._prepared_model: torch.nn.Module | None = None
        self._synchronized_model_version: int | None = None

    def prepare(
        self,
        *,
        model: torch.nn.Module,
        accelerator: Any,
        processing_class: Any,
    ) -> None:
        if self._generation is not None:
            if model is not self._prepared_model:
                raise RuntimeError(
                    "OPD vLLM rollout cannot be rebound to a different student model."
                )
            return
        self._generation = self._build_generation(
            model=model,
            accelerator=accelerator,
            processing_class=processing_class,
        )
        self._prepared_model = model

    def _build_generation(
        self,
        *,
        model: torch.nn.Module,
        accelerator: Any,
        processing_class: Any,
    ) -> Any:
        vllm = self.config.vllm
        implementation = _load_vllm_generation_type()
        temperature = float(self.config.temperature) if self.config.do_sample else 0.0
        accelerator_view = _ExplicitCUDAAcceleratorView(accelerator)
        return implementation(
            model=model,
            accelerator=accelerator_view,
            processing_class=processing_class,
            mode=vllm.mode,
            structured_outputs_regex=vllm.structured_outputs_regex,
            server_base_url=vllm.server_base_url,
            server_host=vllm.server_host,
            server_port=vllm.server_port,
            server_timeout=vllm.server_timeout,
            group_port=vllm.group_port,
            tensor_parallel_size=vllm.tensor_parallel_size,
            gpu_memory_utilization=vllm.gpu_memory_utilization,
            max_model_length=vllm.max_model_length,
            max_num_seqs=vllm.max_num_seqs,
            enable_sleep_mode=vllm.enable_sleep_mode,
            model_impl=vllm.model_impl,
            trust_remote_code=vllm.trust_remote_code,
            repetition_penalty=float(self.config.repetition_penalty),
            temperature=temperature,
            top_p=float(self.config.top_p),
            top_k=int(self.config.top_k),
            min_p=float(self.config.min_p),
            max_completion_length=int(self.config.max_new_tokens),
            logprobs=None,
            generation_kwargs={},
        )

    def _request_seed(self, request: OPDRolloutRequest) -> int:
        payload = {
            "version": "shaft-opd-vllm-request-seed-v1",
            "seed": self.seed,
            "model_version": request.model_version,
            "request_ids": request.request_ids,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).digest()
        return int.from_bytes(digest[:8], "big", signed=False) % (2**63 - 1)

    def _ensure_synchronized(self, request: OPDRolloutRequest) -> Any:
        if self._generation is None:
            raise RuntimeError(
                "OPD vLLM rollout was not prepared before distributed model wrapping."
            )
        runtime_model = request.accelerator.unwrap_model(request.model)
        if runtime_model is not self._prepared_model:
            raise RuntimeError(
                "OPD vLLM rollout request does not reference the prepared student model."
            )
        if self._synchronized_model_version != request.model_version:
            with self._phase("rollout_weight_sync"):
                self._generation.sync_weights()
            self._synchronized_model_version = request.model_version
        return self._generation

    def generate(self, request: OPDRolloutRequest) -> OPDRolloutResult:
        generation = self._ensure_synchronized(request)
        generation.generation_kwargs = {"seed": self._request_seed(request)}
        with self._phase("rollout_generate"):
            prompt_ids, completion_rows, _logprobs, _logprob_token_ids = generation.generate(
                prompts=[list(row) for row in request.generation_prompt_token_ids],
                images=[None if row is None else list(row) for row in request.ordered_images],
                num_generations=1,
            )
        normalized_prompts = tuple(tuple(int(value) for value in row) for row in prompt_ids)
        if len(normalized_prompts) != len(request.prompt_token_ids):
            raise ValueError("vLLM rollout changed prompt batch cardinality.")
        if normalized_prompts != request.prompt_token_ids:
            mismatch_index = next(
                index
                for index, (returned, expected) in enumerate(
                    zip(normalized_prompts, request.prompt_token_ids, strict=True)
                )
                if returned != expected
            )
            returned = normalized_prompts[mismatch_index]
            expected = request.prompt_token_ids[mismatch_index]
            first_difference = next(
                (
                    index
                    for index, (returned_id, expected_id) in enumerate(
                        zip(returned, expected)
                    )
                    if returned_id != expected_id
                ),
                min(len(returned), len(expected)),
            )
            raise ValueError(
                "vLLM returned prompt token IDs that differ from the local processor "
                "contract: "
                f"sample_id={request.sample_ids[mismatch_index]!r}, "
                f"expected_length={len(expected)}, returned_length={len(returned)}, "
                f"first_difference={first_difference}, "
                f"expected_window={expected[max(0, first_difference - 4):first_difference + 5]}, "
                f"returned_window={returned[max(0, first_difference - 4):first_difference + 5]}."
            )
        if len(completion_rows) != len(request.prompt_token_ids):
            raise ValueError("vLLM rollout changed batch cardinality.")
        if any(not row for row in completion_rows):
            empty = [
                request.sample_ids[index]
                for index, row in enumerate(completion_rows)
                if not row
            ]
            raise ValueError(f"vLLM rollout produced empty completions; sample_ids={empty}.")

        tokenizer = self._tokenizer(request.processing_class)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = eos_token_id
        if pad_token_id is None:
            raise ValueError("OPD vLLM rollout requires tokenizer pad_token_id or eos_token_id.")
        device = request.model_inputs["input_ids"].device
        dtype = request.model_inputs["input_ids"].dtype
        width = max(len(row) for row in completion_rows)
        completion_ids = torch.full(
            (len(completion_rows), width),
            int(pad_token_id),
            dtype=dtype,
            device=device,
        )
        raw_mask = torch.zeros((len(completion_rows), width), dtype=torch.bool, device=device)
        for row_index, row in enumerate(completion_rows):
            values = torch.tensor(row, dtype=dtype, device=device)
            completion_ids[row_index, : len(row)] = values
            raw_mask[row_index, : len(row)] = True
        semantic_mask = self.completion_mask(
            completion_ids,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
        completion_mask = raw_mask & semantic_mask
        HFLocalOPDRolloutBackend._validate_nonempty(completion_mask, request.sample_ids)
        sequences = torch.cat((request.model_inputs["input_ids"], completion_ids), dim=-1)
        return HFLocalOPDRolloutBackend._assemble_result(
            sequences=sequences,
            prompt_attention_mask=request.model_inputs["attention_mask"],
            completion_mask=completion_mask,
        )
