from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Protocol

from safetensors.torch import save_file
import torch

from shaft.config import RuntimeConfig
from shaft.codec import decode_qwen_bbox_2d_list
from shaft.data import SFTCollator, SFTDataset, SFTRecord, ShaftDataCenter
from shaft.model import (
    ShaftProcessedBatch,
    ShaftProcessorTokenLayout,
    build_model_input_artifacts,
    build_model_tokenizer_processor,
    materialize_resolved_model_artifact_identity,
    resolve_model_plan,
)
from shaft.opd.input_abi import ShaftOPDInputABI, build_opd_input_abi
from shaft.training.distribution_loss import TeacherDistribution

from .artifact import (
    ARTIFACT_VERSION,
    ShaftOfflineKDInputContract,
    build_offline_kd_input_contract,
    canonical_sha256,
    file_sha256,
    media_content_fingerprint,
    offline_kd_artifact_identity,
)
from .media_plan import media_plan_from_item


DENYLIST_VERSION = "shaft-offline-kd-denylist-v1"
BUILD_STATE_VERSION = "shaft-offline-kd-build-state-v1"


def _require_digest(value: str, *, role: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise ValueError(f"{role} must be a SHA-256 digest.")
    return normalized


@dataclass(frozen=True, slots=True)
class OfflineKDDenylist:
    sample_ids: frozenset[str]
    image_paths: frozenset[str]
    fingerprint: str

    @classmethod
    def empty(cls) -> "OfflineKDDenylist":
        payload = {"version": DENYLIST_VERSION, "sample_ids": [], "image_paths": []}
        return cls(frozenset(), frozenset(), canonical_sha256(payload))

    @classmethod
    def load(cls, path: str | Path | None) -> "OfflineKDDenylist":
        if path is None:
            return cls.empty()
        denylist_path = Path(path).resolve()
        payload = json.loads(denylist_path.read_text(encoding="utf-8"))
        expected = {"version", "sample_ids", "image_paths"}
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ValueError("Offline KD denylist fields differ from the v1 contract.")
        if payload["version"] != DENYLIST_VERSION:
            raise ValueError(f"Unsupported Offline KD denylist version {payload['version']!r}.")
        for name in ("sample_ids", "image_paths"):
            values = payload[name]
            if not isinstance(values, list) or not all(
                isinstance(value, str) and value.strip() for value in values
            ):
                raise TypeError(f"Offline KD denylist {name} must be a non-empty-string list.")
            if len(values) != len(set(values)):
                raise ValueError(f"Offline KD denylist {name} contains duplicates.")
        image_paths = frozenset(
            str((denylist_path.parent / value).resolve())
            if not Path(value).is_absolute()
            else str(Path(value).resolve())
            for value in payload["image_paths"]
        )
        canonical = {
            "version": DENYLIST_VERSION,
            "sample_ids": sorted(payload["sample_ids"]),
            "image_paths": sorted(image_paths),
        }
        return cls(
            sample_ids=frozenset(payload["sample_ids"]),
            image_paths=image_paths,
            fingerprint=canonical_sha256(canonical),
        )

    def excludes(self, item: Mapping[str, Any]) -> bool:
        sample_id = str(item.get("sample_id", "")).strip()
        image_paths = {
            str(Path(str(path)).resolve()) for path in tuple(item.get("image_paths") or ())
        }
        return sample_id in self.sample_ids or bool(image_paths & self.image_paths)


@dataclass(frozen=True, slots=True)
class OfflineKDDistributionSpec:
    mode: str
    temperature: float | None = None
    top_k: int | None = None

    def __post_init__(self) -> None:
        mode = str(self.mode).strip().lower()
        temperature = None if self.temperature is None else float(self.temperature)
        top_k = None if self.top_k is None else int(self.top_k)
        if mode not in {"dense_logits", "topk_tail"}:
            raise ValueError("Offline KD distribution mode must be dense_logits or topk_tail.")
        if mode == "dense_logits" and temperature is not None:
            raise ValueError("Offline KD dense logits storage must not bind a temperature.")
        if mode == "topk_tail" and (
            temperature is None
            or not torch.isfinite(torch.tensor(temperature))
            or temperature <= 0
        ):
            raise ValueError("Offline KD top-k distribution temperature must be finite and > 0.")
        if (mode == "topk_tail") != (top_k is not None):
            raise ValueError("Offline KD distribution top_k must be set only for topk_tail.")
        if top_k is not None and top_k <= 0:
            raise ValueError("Offline KD distribution top_k must be > 0.")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "top_k", top_k)


class OfflineKDTeacherScorer(Protocol):
    def score(self, batch: "OfflineKDScoringBatch") -> tuple[TeacherDistribution, ...]:
        """Return one teacher distribution per source row."""


@dataclass(frozen=True, slots=True)
class OfflineKDScoringBatch:
    model_inputs: Mapping[str, Any]
    completion_mask: torch.Tensor
    input_token_ids: tuple[torch.Tensor, ...]
    prompt_completion_masks: tuple[torch.Tensor, ...]
    images: tuple[tuple[Any, ...], ...]
    vllm_prompt_token_ids: tuple[tuple[int, ...], ...] | None = None

    def __post_init__(self) -> None:
        row_count = len(self.input_token_ids)
        if self.completion_mask.ndim != 2 or int(self.completion_mask.shape[0]) != row_count:
            raise ValueError("Offline KD scoring completion mask must be batch-first and 2-D.")
        if len(self.prompt_completion_masks) != row_count or len(self.images) != row_count:
            raise ValueError("Offline KD scoring batch fields disagree on row count.")
        if self.vllm_prompt_token_ids is not None and (
            len(self.vllm_prompt_token_ids) != row_count
            or any(not row for row in self.vllm_prompt_token_ids)
        ):
            raise ValueError("Offline KD vLLM prompt IDs must be non-empty and batch-aligned.")
        for token_ids, mask in zip(self.input_token_ids, self.prompt_completion_masks):
            if token_ids.ndim != 1 or mask.ndim != 1 or token_ids.numel() != mask.numel():
                raise ValueError("Offline KD scoring prompt IDs and completion mask must align.")


@dataclass(frozen=True, slots=True)
class ShaftOfflineKDVLLMCollation:
    model_inputs: dict[str, Any]
    prompt_token_ids: tuple[tuple[int, ...], ...]
    images: tuple[tuple[Any, ...], ...]


@dataclass(frozen=True, slots=True)
class ShaftOfflineKDVLLMPromptCollation:
    expanded_prompt_token_ids: tuple[torch.Tensor, ...]
    prompt_token_ids: tuple[tuple[int, ...], ...]
    images: tuple[tuple[Any, ...], ...]


@dataclass(frozen=True, slots=True)
class _OfflineKDOutputHeadABIAdapter:
    out_features: int


@dataclass(frozen=True, slots=True)
class _OfflineKDForwardABIAdapter:
    forward: Any
    logits_vocab_size: int

    def get_output_embeddings(self) -> _OfflineKDOutputHeadABIAdapter:
        return _OfflineKDOutputHeadABIAdapter(out_features=self.logits_vocab_size)


@dataclass(frozen=True, slots=True)
class _OfflineKDArtifactsABIAdapter:
    tokenizer: Any
    processor: Any
    model_adapter: Any
    model: _OfflineKDForwardABIAdapter


def _build_offline_kd_input_abi(artifacts: Any) -> ShaftOPDInputABI:
    """Build the shared input ABI without extending or changing OPD runtime code."""

    if hasattr(artifacts, "model"):
        return build_opd_input_abi(artifacts)

    forward_owner = getattr(artifacts, "forward_owner", None)
    forward = getattr(forward_owner, "forward", None)
    logits_vocab_size = getattr(artifacts, "logits_vocab_size", None)
    if not callable(forward) or type(logits_vocab_size) is not int or logits_vocab_size <= 0:
        raise TypeError(
            "Offline KD input-only artifacts require callable forward_owner.forward and "
            "a positive logits_vocab_size."
        )
    forward_adapter = _OfflineKDForwardABIAdapter(
        forward=forward,
        logits_vocab_size=logits_vocab_size,
    )
    artifact_adapter = _OfflineKDArtifactsABIAdapter(
        tokenizer=artifacts.tokenizer,
        processor=artifacts.processor,
        model_adapter=artifacts.model_adapter,
        model=forward_adapter,
    )
    return build_opd_input_abi(artifact_adapter)


def prepare_offline_kd_scoring_items(
    items: list[dict[str, Any]],
    *,
    model_adapter: Any,
    min_pixels: int | None,
    max_pixels: int | None,
) -> list[dict[str, Any]]:
    """Apply the model policy's smart resize exactly once before either scorer path."""

    image_rows = SFTCollator._processor_image_rows(items)
    prepared: list[dict[str, Any]] = []
    for item, raw_row in zip(items, image_rows, strict=True):
        media_plan = media_plan_from_item(item)
        row_min_pixels = min_pixels if media_plan is None else media_plan.min_pixels
        row_max_pixels = max_pixels if media_plan is None else media_plan.max_pixels
        images = tuple(raw_row) if isinstance(raw_row, (list, tuple)) else (raw_row,)
        if media_plan is not None and len(images) != 1:
            raise ValueError("Offline KD per-row media plans currently require exactly one image.")
        if media_plan is not None:
            media_plan.validate_image_size(images[0])
        resized_images = tuple(
            model_adapter.prepare_rollout_image(
                image,
                min_pixels=row_min_pixels,
                max_pixels=row_max_pixels,
            )
            for image in images
        )
        resolved = dict(item)
        resolved["images"] = resized_images
        resolved["image"] = (
            resized_images[0] if len(resized_images) == 1 else resized_images
        )
        prepared.append(resolved)
    return prepared


class ShaftOfflineKDVLLMScoringCollator(SFTCollator):
    """Build local expanded inputs and the structured, unexpanded vLLM prompt together."""

    def __init__(self, *, image_token_id: int | None, **kwargs: Any) -> None:
        min_pixels = kwargs.pop("min_pixels", None)
        max_pixels = kwargs.pop("max_pixels", None)
        if min_pixels is not None or max_pixels is not None:
            raise ValueError(
                "Offline KD vLLM collation requires pre-resized images and no pixel budget."
            )
        super().__init__(min_pixels=None, max_pixels=None, **kwargs)
        self.image_token_id = image_token_id
        self._captured_processed_batch: ShaftProcessedBatch | None = None
        self._captured_plans: tuple[Any, ...] = ()
        self._captured_layouts: tuple[ShaftProcessorTokenLayout | None, ...] = ()
        self._captured_images: tuple[tuple[Any, ...], ...] = ()

    def _run_processor(
        self,
        prompt_texts: list[str],
        images: list[Any],
        *,
        dataset_names: list[str | None] | None = None,
    ) -> ShaftProcessedBatch:
        processed = super()._run_processor(
            prompt_texts,
            images,
            dataset_names=dataset_names,
        )
        self._captured_processed_batch = processed
        self._captured_images = tuple(
            tuple(row) if isinstance(row, (list, tuple)) else (row,)
            for row in images
        )
        return processed

    def _build_prefix_token_layouts(
        self,
        *,
        plans: list[Any],
        processed_batch: ShaftProcessedBatch,
        max_length: int | None = None,
    ) -> list[ShaftProcessorTokenLayout | None]:
        layouts = super()._build_prefix_token_layouts(
            plans=plans,
            processed_batch=processed_batch,
            max_length=max_length,
        )
        self._captured_plans = tuple(plans)
        self._captured_layouts = tuple(layouts)
        return layouts

    @staticmethod
    def _retained_rendered_prefix(
        *,
        plan: Any,
        layout: ShaftProcessorTokenLayout | None,
        processed_prefix_indices: tuple[int, ...],
    ) -> tuple[int, ...]:
        rendered_ids = tuple(int(value) for value in plan.rendered_prefix_token_ids)
        if layout is None:
            return rendered_ids
        retained = set(int(value) for value in processed_prefix_indices)
        output: list[int] = []
        for rendered_index, token_id in enumerate(rendered_ids):
            start = int(layout.processed_boundaries[rendered_index])
            end = int(layout.processed_boundaries[rendered_index + 1])
            retained_count = sum(index in retained for index in range(start, end))
            if retained_count == 0:
                continue
            if retained_count != end - start:
                raise ValueError(
                    "Offline KD prefix truncation split one processor-expanded token span."
                )
            output.append(token_id)
        return tuple(output)

    def collate_for_vllm(
        self,
        items: list[dict[str, Any]],
    ) -> ShaftOfflineKDVLLMCollation:
        self._captured_processed_batch = None
        self._captured_plans = ()
        self._captured_layouts = ()
        self._captured_images = ()
        model_inputs = super().__call__(items)
        processed_batch = self._captured_processed_batch
        if processed_batch is None or len(self._captured_plans) != len(items):
            raise RuntimeError("Offline KD vLLM collation did not capture its processor plan.")
        rows = tuple(
            self.template.build_supervised_row(
                plan=plan,
                tokenizer=self.tokenizer,
                processed_batch=processed_batch,
                row_index=row_index,
                prefix_token_layout=layout,
                add_eos_token=self.add_eos_token,
                ignore_index=self.ignore_index,
                include_targets_in_inputs=self.include_targets_in_inputs,
                max_length=self.max_length,
            )
            for row_index, (plan, layout) in enumerate(
                zip(self._captured_plans, self._captured_layouts, strict=True)
            )
        )
        prompt_rows: list[tuple[int, ...]] = []
        for row_index, (item, plan, layout, row) in enumerate(
            zip(
                items,
                self._captured_plans,
                self._captured_layouts,
                rows,
                strict=True,
            )
        ):
            attention_mask = model_inputs["attention_mask"][row_index].bool()
            collated_ids = model_inputs["input_ids"][row_index][attention_mask]
            if not torch.equal(collated_ids, row.input_ids):
                raise RuntimeError(
                    "Offline KD reconstructed supervision row differs from SFT collation."
                )
            prefix = self._retained_rendered_prefix(
                plan=plan,
                layout=layout,
                processed_prefix_indices=row.processed_prefix_indices,
            )
            if self.image_token_id is not None:
                image_count = len(self._captured_images[row_index])
                placeholder_count = prefix.count(int(self.image_token_id))
                if placeholder_count != image_count:
                    raise ValueError(
                        "Offline KD vLLM prompt must contain one unexpanded image placeholder "
                        f"per image: placeholders={placeholder_count}, images={image_count}."
                    )
            target_start = len(row.processed_prefix_indices)
            target_ids = tuple(int(value) for value in row.input_ids[target_start:].tolist())
            prompt_rows.append((*prefix, *target_ids))
        return ShaftOfflineKDVLLMCollation(
            model_inputs=model_inputs,
            prompt_token_ids=tuple(prompt_rows),
            images=self._captured_images,
        )

    def collate_prompts_for_vllm(
        self,
        items: list[dict[str, Any]],
    ) -> ShaftOfflineKDVLLMPromptCollation:
        self._captured_processed_batch = None
        self._captured_plans = ()
        self._captured_layouts = ()
        self._captured_images = ()
        plans = [
            self.template.build_prompt_plan(item=item, renderer=self.chat_renderer)
            for item in items
        ]
        images = self._processor_image_rows(items)
        processed_batch = self._run_processor(
            [plan.prompt_text for plan in plans],
            images,
            dataset_names=[item.get("dataset_name") for item in items],
        )
        layouts = self._build_prefix_token_layouts(
            plans=plans,
            processed_batch=processed_batch,
        )
        rows = tuple(
            self.template.build_prompt_row(
                plan=plan,
                tokenizer=self.tokenizer,
                processed_batch=processed_batch,
                row_index=row_index,
                prefix_token_layout=layout,
                max_length=self.max_length,
            )
            for row_index, (plan, layout) in enumerate(zip(plans, layouts, strict=True))
        )
        prompt_rows: list[tuple[int, ...]] = []
        for row_index, (plan, layout, row) in enumerate(
            zip(plans, layouts, rows, strict=True)
        ):
            prefix = self._retained_rendered_prefix(
                plan=plan,
                layout=layout,
                processed_prefix_indices=row.processed_prefix_indices,
            )
            if self.image_token_id is not None:
                image_count = len(self._captured_images[row_index])
                if prefix.count(int(self.image_token_id)) != image_count:
                    raise ValueError(
                        "Offline KD vLLM prompt must contain one unexpanded image placeholder "
                        "per image."
                    )
            prompt_rows.append(prefix)
        return ShaftOfflineKDVLLMPromptCollation(
            expanded_prompt_token_ids=tuple(row.input_ids for row in rows),
            prompt_token_ids=tuple(prompt_rows),
            images=self._captured_images,
        )


def _distribution_from_logits(
    logits: torch.Tensor,
    *,
    spec: OfflineKDDistributionSpec,
) -> TeacherDistribution:
    if spec.mode == "dense_logits":
        return TeacherDistribution.from_dense_logits(logits)
    return TeacherDistribution.from_topk_logits(
        logits,
        top_k=int(spec.top_k or 0),
        temperature=float(spec.temperature or 0.0),
    )


class HFOfflineKDTeacherScorer:
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        model_adapter: Any,
        distribution_spec: OfflineKDDistributionSpec,
    ) -> None:
        self.model = model
        self.model_adapter = model_adapter
        self.distribution_spec = distribution_spec
        self.model.eval()

    def score(self, batch: OfflineKDScoringBatch) -> tuple[TeacherDistribution, ...]:
        forwarded = dict(batch.model_inputs)
        for name in (
            "labels",
            "loss_scale",
            "_shaft_batch_stats",
            "_shaft_varlen_layout",
            "_shaft_media_manifest",
            "meta",
        ):
            forwarded.pop(name, None)
        forwarded = self.model_adapter.prepare_sft_forward_inputs(
            model=self.model,
            inputs=forwarded,
        )
        with torch.inference_mode():
            outputs = self.model(**forwarded)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
        if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
            raise TypeError("Offline KD teacher scorer must return 3-D logits.")
        distributions: list[TeacherDistribution] = []
        for row_index in range(int(logits.shape[0])):
            shifted_mask = batch.completion_mask[row_index, 1:].to(device=logits.device)
            flattened_logits = logits[row_index, :-1, :][shifted_mask]
            distributions.append(
                _distribution_from_logits(
                    flattened_logits,
                    spec=self.distribution_spec,
                )
            )
        return tuple(distributions)


class VLLMOfflineKDTeacherScorer:
    """Teacher-forced vLLM prompt-logprob scorer with exact alignment checks."""

    def __init__(
        self,
        *,
        model_name_or_path: str,
        distribution_spec: OfflineKDDistributionSpec,
        vocab_size: int,
        trust_remote_code: bool,
        revision: str | None,
        dtype: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_length: int | None = None,
        enforce_eager: bool = False,
        engine: Any | None = None,
    ) -> None:
        self.distribution_spec = distribution_spec
        self.vocab_size = int(vocab_size)
        if self.vocab_size <= 0:
            raise ValueError("vLLM Offline KD vocab_size must be > 0.")
        self.prompt_logprobs = (
            -1
            if distribution_spec.mode == "dense_logits"
            or float(distribution_spec.temperature or 1.0) != 1.0
            else int(distribution_spec.top_k or 0)
        )
        if engine is None:
            try:
                from vllm import LLM
            except ImportError as exc:
                raise ImportError(
                    "vLLM Offline KD scoring requires the serve extra."
                ) from exc
            kwargs: dict[str, Any] = {
                "model": model_name_or_path,
                "trust_remote_code": bool(trust_remote_code),
                "revision": revision,
                "dtype": dtype,
                "tensor_parallel_size": int(tensor_parallel_size),
                "gpu_memory_utilization": float(gpu_memory_utilization),
                "enforce_eager": bool(enforce_eager),
                "max_logprobs": (
                    -1 if self.prompt_logprobs == -1 else self.prompt_logprobs
                ),
                "logprobs_mode": "raw_logprobs",
            }
            if max_model_length is not None:
                kwargs["max_model_len"] = int(max_model_length)
            engine = LLM(**kwargs)
        self.engine = engine

    @staticmethod
    def _position_values(value: Any) -> dict[int, float]:
        if not isinstance(value, Mapping) or not value:
            raise ValueError("vLLM Offline KD prompt logprobs are missing one position.")
        output: dict[int, float] = {}
        for raw_token_id, raw_logprob in value.items():
            token_id = int(raw_token_id)
            logprob = getattr(raw_logprob, "logprob", raw_logprob)
            output[token_id] = float(logprob)
        return output

    def _full_log_probs(self, value: Any) -> torch.Tensor:
        observed = self._position_values(value)
        if set(observed) != set(range(self.vocab_size)):
            raise ValueError(
                "vLLM full-vocabulary prompt logprobs do not cover the declared logits ABI."
            )
        return torch.tensor(
            [observed[token_id] for token_id in range(self.vocab_size)],
            dtype=torch.float32,
        )

    def _topk_row(self, value: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        top_k = int(self.distribution_spec.top_k or 0)
        observed = self._position_values(value)
        ranked = sorted(observed.items(), key=lambda item: (-item[1], item[0]))
        if len(ranked) < top_k:
            raise ValueError("vLLM returned fewer prompt logprobs than Offline KD top_k.")
        selected = ranked[:top_k]
        token_ids = torch.tensor([item[0] for item in selected], dtype=torch.long)
        log_probs = torch.tensor([item[1] for item in selected], dtype=torch.float32)
        if top_k == self.vocab_size:
            return token_ids, log_probs, None
        selected_mass = log_probs.exp().sum()
        epsilon = torch.finfo(torch.float32).eps
        tail = torch.log1p(-selected_mass.clamp(max=1.0 - epsilon))
        return token_ids, log_probs, tail

    def score(self, batch: OfflineKDScoringBatch) -> tuple[TeacherDistribution, ...]:
        try:
            from vllm import SamplingParams
        except ImportError as exc:
            raise ImportError("vLLM Offline KD scoring requires vllm.") from exc
        if batch.vllm_prompt_token_ids is None:
            raise ValueError(
                "vLLM Offline KD scoring requires structured unexpanded prompt token IDs."
            )
        prompts: list[dict[str, Any]] = []
        for token_ids, images in zip(
            batch.vllm_prompt_token_ids,
            batch.images,
            strict=True,
        ):
            prompt: dict[str, Any] = {
                "prompt_token_ids": [int(value) for value in token_ids],
            }
            if images:
                prompt["multi_modal_data"] = {"image": list(images)}
            prompts.append(prompt)
        params = SamplingParams(
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            max_tokens=1,
            prompt_logprobs=self.prompt_logprobs,
            flat_logprobs=True,
            detokenize=False,
        )
        outputs = self.engine.generate(prompts, params, use_tqdm=False)
        if len(outputs) != len(prompts):
            raise ValueError("vLLM Offline KD output count differs from request count.")
        distributions: list[TeacherDistribution] = []
        for row_index, (request, output) in enumerate(zip(prompts, outputs, strict=True)):
            observed_prompt_ids = tuple(int(value) for value in (output.prompt_token_ids or ()))
            expected_prompt_ids = tuple(
                int(value) for value in batch.input_token_ids[row_index].tolist()
            )
            if observed_prompt_ids != expected_prompt_ids:
                first_difference = next(
                    (
                        index
                        for index, (observed_id, expected_id) in enumerate(
                            zip(observed_prompt_ids, expected_prompt_ids)
                        )
                        if observed_id != expected_id
                    ),
                    min(len(observed_prompt_ids), len(expected_prompt_ids)),
                )
                window_start = max(0, first_difference - 4)
                window_end = first_difference + 5
                raise ValueError(
                    "vLLM Offline KD prompt token IDs drifted from Shaft collation: "
                    f"row_index={row_index}, expected_length={len(expected_prompt_ids)}, "
                    f"observed_length={len(observed_prompt_ids)}, "
                    f"first_difference={first_difference}, "
                    f"expected_window={expected_prompt_ids[window_start:window_end]}, "
                    f"observed_window={observed_prompt_ids[window_start:window_end]}."
                )
            prompt_logprobs = output.prompt_logprobs
            if prompt_logprobs is None or len(prompt_logprobs) != len(expected_prompt_ids):
                raise ValueError("vLLM Offline KD prompt logprob positions do not align.")
            position_values = [
                prompt_logprobs[position]
                for position, selected in enumerate(
                    batch.prompt_completion_masks[row_index].tolist()
                )
                if selected
            ]
            if not position_values:
                raise ValueError("vLLM Offline KD row has no completion positions.")
            if self.prompt_logprobs == -1:
                dense_log_probs = torch.stack(
                    [self._full_log_probs(value) for value in position_values]
                )
                distributions.append(
                    _distribution_from_logits(
                        dense_log_probs,
                        spec=self.distribution_spec,
                    )
                )
                continue
            topk_rows = [self._topk_row(value) for value in position_values]
            tails = [row[2] for row in topk_rows]
            distributions.append(
                TeacherDistribution(
                    kind="topk_tail",
                    vocab_size=self.vocab_size,
                    topk_token_ids=torch.stack([row[0] for row in topk_rows]),
                    topk_log_probs=torch.stack([row[1] for row in topk_rows]),
                    tail_log_probs=(
                        None
                        if all(tail is None for tail in tails)
                        else torch.stack([tail for tail in tails if tail is not None])
                    ),
                    temperature=float(self.distribution_spec.temperature or 0.0),
                )
            )
        return tuple(distributions)


@dataclass(frozen=True, slots=True)
class OfflineKDPseudoLabelGeneration:
    raw_text: str
    request_prompt_token_ids: tuple[int, ...]
    expanded_prompt_token_ids: tuple[int, ...]
    completion_token_ids: torch.Tensor
    distribution: TeacherDistribution
    finish_reason: str


@dataclass(frozen=True, slots=True)
class _DetectionPseudoKDRequest:
    source_index: int
    source_item: dict[str, Any]
    prepared_item: dict[str, Any]
    prompt_token_ids: tuple[int, ...]
    expanded_prompt_token_ids: torch.Tensor
    images: tuple[Any, ...]


def validate_detection_pseudo_label(raw_text: str) -> list[dict[str, Any]]:
    """Strictly validate the v5.8 detection JSON without repairing or reserializing it."""

    result = decode_qwen_bbox_2d_list(
        raw_text,
        allowed_labels={"shape", "icon", "image", "line"},
    )
    if not result.valid:
        raise ValueError(
            "Detection pseudo label must be exact JSON and match the strict schema: "
            f"{result.error}"
        )
    assert isinstance(result.parsed, list)
    return result.parsed


class VLLMOfflineKDGreedyGenerator:
    """One-pass greedy pseudo-label generator that keeps output top-k plus tail mass."""

    def __init__(
        self,
        *,
        model_name_or_path: str,
        vocab_size: int,
        trust_remote_code: bool,
        revision: str | None,
        dtype: str,
        top_k: int = 64,
        max_tokens: int = 8000,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_length: int | None = None,
        enforce_eager: bool = False,
        engine: Any | None = None,
    ) -> None:
        self.vocab_size = int(vocab_size)
        self.top_k = int(top_k)
        self.max_tokens = int(max_tokens)
        if self.vocab_size <= 0 or not 0 < self.top_k <= self.vocab_size:
            raise ValueError("Greedy Offline KD requires 0 < top_k <= vocab_size.")
        if self.max_tokens <= 0:
            raise ValueError("Greedy Offline KD max_tokens must be > 0.")
        if engine is None:
            try:
                from vllm import LLM
            except ImportError as exc:
                raise ImportError("vLLM greedy Offline KD requires the serve extra.") from exc
            kwargs: dict[str, Any] = {
                "model": model_name_or_path,
                "trust_remote_code": bool(trust_remote_code),
                "revision": revision,
                "dtype": dtype,
                "tensor_parallel_size": int(tensor_parallel_size),
                "gpu_memory_utilization": float(gpu_memory_utilization),
                "enforce_eager": bool(enforce_eager),
                "max_logprobs": self.top_k,
                "logprobs_mode": "raw_logprobs",
            }
            if max_model_length is not None:
                kwargs["max_model_len"] = int(max_model_length)
            engine = LLM(**kwargs)
        self.engine = engine

    def _distribution(self, rows: Any) -> TeacherDistribution:
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
            raise ValueError("vLLM greedy output logprobs are missing.")
        token_rows: list[torch.Tensor] = []
        probability_rows: list[torch.Tensor] = []
        tail_rows: list[torch.Tensor] = []
        for value in rows:
            observed = VLLMOfflineKDTeacherScorer._position_values(value)
            ranked = sorted(observed.items(), key=lambda item: (-item[1], item[0]))
            if len(ranked) < self.top_k:
                raise ValueError("vLLM returned fewer generation logprobs than requested top_k.")
            selected = ranked[: self.top_k]
            token_ids = torch.tensor([item[0] for item in selected], dtype=torch.long)
            log_probs = torch.tensor([item[1] for item in selected], dtype=torch.float32)
            token_rows.append(token_ids)
            probability_rows.append(log_probs)
            if self.top_k < self.vocab_size:
                epsilon = torch.finfo(torch.float32).eps
                tail_rows.append(
                    torch.log1p(-log_probs.exp().sum().clamp(max=1.0 - epsilon))
                )
        return TeacherDistribution(
            kind="topk_tail",
            vocab_size=self.vocab_size,
            topk_token_ids=torch.stack(token_rows),
            topk_log_probs=torch.stack(probability_rows),
            tail_log_probs=None if self.top_k == self.vocab_size else torch.stack(tail_rows),
            temperature=1.0,
        )

    def generate(
        self,
        *,
        prompt_token_ids: tuple[tuple[int, ...], ...],
        expanded_prompt_token_ids: tuple[torch.Tensor, ...],
        images: tuple[tuple[Any, ...], ...],
    ) -> tuple[OfflineKDPseudoLabelGeneration, ...]:
        try:
            from vllm import SamplingParams
        except ImportError as exc:
            raise ImportError("vLLM greedy Offline KD requires vllm.") from exc
        prompts: list[dict[str, Any]] = []
        for token_ids, image_row in zip(prompt_token_ids, images, strict=True):
            prompt: dict[str, Any] = {"prompt_token_ids": list(token_ids)}
            if image_row:
                prompt["multi_modal_data"] = {"image": list(image_row)}
            prompts.append(prompt)
        params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            max_tokens=self.max_tokens,
            logprobs=self.top_k,
            flat_logprobs=True,
            detokenize=True,
        )
        outputs = self.engine.generate(prompts, params, use_tqdm=False)
        if len(outputs) != len(prompts):
            raise ValueError("vLLM greedy output count differs from request count.")
        generated: list[OfflineKDPseudoLabelGeneration] = []
        for request, expected_expanded, output in zip(
            prompts, expanded_prompt_token_ids, outputs, strict=True
        ):
            observed_prompt = tuple(int(value) for value in (output.prompt_token_ids or ()))
            expected_prompt = tuple(int(value) for value in expected_expanded.tolist())
            if observed_prompt != expected_prompt:
                raise ValueError(
                    "vLLM greedy expanded prompt token IDs drifted from Shaft collation."
                )
            if not getattr(output, "outputs", None) or len(output.outputs) != 1:
                raise ValueError("vLLM greedy generation requires exactly one output sequence.")
            candidate = output.outputs[0]
            completion_ids = torch.tensor(candidate.token_ids, dtype=torch.long)
            if completion_ids.numel() == 0 or len(candidate.logprobs or ()) != completion_ids.numel():
                raise ValueError("vLLM greedy token IDs and output logprobs do not align.")
            generated.append(
                OfflineKDPseudoLabelGeneration(
                    raw_text=str(candidate.text),
                    request_prompt_token_ids=tuple(request["prompt_token_ids"]),
                    expanded_prompt_token_ids=observed_prompt,
                    completion_token_ids=completion_ids,
                    distribution=self._distribution(candidate.logprobs),
                    finish_reason=str(candidate.finish_reason or ""),
                )
            )
        return tuple(generated)


class VLLMOfflineKDAsyncGreedyGenerator(VLLMOfflineKDGreedyGenerator):
    """Async vLLM generator that lets the caller continuously refill request slots."""

    def __init__(
        self,
        *,
        model_name_or_path: str,
        vocab_size: int,
        trust_remote_code: bool,
        revision: str | None,
        dtype: str,
        top_k: int = 64,
        max_tokens: int = 8000,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_length: int | None = None,
        enforce_eager: bool = False,
        engine: Any | None = None,
    ) -> None:
        self.vocab_size = int(vocab_size)
        self.top_k = int(top_k)
        self.max_tokens = int(max_tokens)
        if self.vocab_size <= 0 or not 0 < self.top_k <= self.vocab_size:
            raise ValueError("Greedy Offline KD requires 0 < top_k <= vocab_size.")
        if self.max_tokens <= 0:
            raise ValueError("Greedy Offline KD max_tokens must be > 0.")
        if engine is None:
            try:
                from vllm.engine.arg_utils import AsyncEngineArgs
                from vllm.v1.engine.async_llm import AsyncLLM
            except ImportError as exc:
                raise ImportError("Async vLLM greedy Offline KD requires the serve extra.") from exc
            kwargs: dict[str, Any] = {
                "model": model_name_or_path,
                "trust_remote_code": bool(trust_remote_code),
                "revision": revision,
                "dtype": dtype,
                "tensor_parallel_size": int(tensor_parallel_size),
                "gpu_memory_utilization": float(gpu_memory_utilization),
                "enforce_eager": bool(enforce_eager),
                "max_logprobs": self.top_k,
                "logprobs_mode": "raw_logprobs",
                "disable_log_stats": True,
            }
            if max_model_length is not None:
                kwargs["max_model_len"] = int(max_model_length)
            engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**kwargs))
        self.engine = engine

    async def generate_one(
        self,
        *,
        request_id: str,
        prompt_token_ids: tuple[int, ...],
        expanded_prompt_token_ids: torch.Tensor,
        images: tuple[Any, ...],
    ) -> OfflineKDPseudoLabelGeneration:
        try:
            from vllm import SamplingParams
            from vllm.sampling_params import RequestOutputKind
        except ImportError as exc:
            raise ImportError("Async vLLM greedy Offline KD requires vllm.") from exc
        prompt: dict[str, Any] = {"prompt_token_ids": list(prompt_token_ids)}
        if images:
            prompt["multi_modal_data"] = {"image": list(images)}
        params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            max_tokens=self.max_tokens,
            logprobs=self.top_k,
            flat_logprobs=True,
            detokenize=True,
            output_kind=RequestOutputKind.FINAL_ONLY,
        )
        output = None
        async for candidate_output in self.engine.generate(prompt, params, request_id):
            output = candidate_output
        if output is None:
            raise ValueError("Async vLLM greedy generation returned no final output.")
        observed_prompt = tuple(int(value) for value in (output.prompt_token_ids or ()))
        expected_prompt = tuple(int(value) for value in expanded_prompt_token_ids.tolist())
        if observed_prompt != expected_prompt:
            raise ValueError("vLLM greedy expanded prompt token IDs drifted from Shaft collation.")
        if not getattr(output, "outputs", None) or len(output.outputs) != 1:
            raise ValueError("vLLM greedy generation requires exactly one output sequence.")
        candidate = output.outputs[0]
        completion_ids = torch.tensor(candidate.token_ids, dtype=torch.long)
        if completion_ids.numel() == 0 or len(candidate.logprobs or ()) != completion_ids.numel():
            raise ValueError("vLLM greedy token IDs and output logprobs do not align.")
        return OfflineKDPseudoLabelGeneration(
            raw_text=str(candidate.text),
            request_prompt_token_ids=prompt_token_ids,
            expanded_prompt_token_ids=observed_prompt,
            completion_token_ids=completion_ids,
            distribution=self._distribution(candidate.logprobs),
            finish_reason=str(candidate.finish_reason or ""),
        )

    def shutdown(self) -> None:
        shutdown = getattr(self.engine, "shutdown", None)
        if callable(shutdown):
            shutdown()


@dataclass(frozen=True, slots=True)
class OfflineKDArtifactRow:
    source_payload: dict[str, Any]
    input_token_ids: torch.Tensor
    completion_token_ids: torch.Tensor
    media_sha256: bytes
    distribution: TeacherDistribution
    source_index: int | None = None


class OfflineKDArtifactWriter:
    """Atomic writer for the public v1 offline-KD artifact contract."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        teacher_model: str,
        teacher_checkpoint_fingerprint: str,
        input_abi: ShaftOPDInputABI,
        input_contract: ShaftOfflineKDInputContract,
        distribution_spec: OfflineKDDistributionSpec,
        source_fingerprint: str,
        denylist_fingerprint: str,
        shard_rows: int = 128,
        shard_max_bytes: int = 512 * 1024 * 1024,
        storage_dtype: torch.dtype = torch.float16,
        resume: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir).resolve()
        if self.output_dir.exists():
            raise FileExistsError(f"Offline KD output already exists: {self.output_dir}")
        if int(shard_rows) <= 0:
            raise ValueError("Offline KD shard_rows must be > 0.")
        if type(shard_max_bytes) is not int or shard_max_bytes <= 0:
            raise ValueError("Offline KD shard_max_bytes must be a positive integer.")
        if storage_dtype not in {torch.float16, torch.bfloat16, torch.float32}:
            raise ValueError("Offline KD storage dtype must be float16, bfloat16, or float32.")
        self.teacher_model = str(teacher_model).strip()
        if not self.teacher_model:
            raise ValueError("Offline KD teacher model identity must not be empty.")
        self.teacher_checkpoint_fingerprint = _require_digest(
            teacher_checkpoint_fingerprint,
            role="teacher_checkpoint_fingerprint",
        )
        self.source_fingerprint = _require_digest(
            source_fingerprint, role="source_fingerprint"
        )
        self.denylist_fingerprint = _require_digest(
            denylist_fingerprint, role="denylist_fingerprint"
        )
        self.input_abi = input_abi
        self.input_contract = input_contract
        self.distribution_spec = distribution_spec
        self.shard_rows = int(shard_rows)
        self.shard_max_bytes = shard_max_bytes
        self.storage_dtype = storage_dtype
        self.resume = bool(resume)
        if (
            distribution_spec.top_k is not None
            and distribution_spec.top_k > input_abi.logits_vocab_size
        ):
            raise ValueError("Offline KD producer top_k exceeds the logits vocabulary.")
        self.artifact_id = offline_kd_artifact_identity(
            teacher={
                "model": self.teacher_model,
                "checkpoint_fingerprint": self.teacher_checkpoint_fingerprint,
            },
            input_abi=input_abi.to_dict(),
            input_contract=input_contract.to_dict(),
            distribution=self._distribution_payload(),
            build={
                "source_fingerprint": self.source_fingerprint,
                "denylist_fingerprint": self.denylist_fingerprint,
            },
        )
        self.output_dir.parent.mkdir(parents=True, exist_ok=True)
        self._rows: list[OfflineKDArtifactRow] = []
        self._pending_bytes = 0
        self._shards: dict[str, str] = {}
        self._row_count = 0
        self.resume_source_index = 0
        self._finalized = False
        if self.resume:
            self._temporary_dir = self.output_dir.parent / f".{self.output_dir.name}.building"
            if self._temporary_dir.exists():
                self._restore_build_state()
            else:
                self._temporary_dir.mkdir()
                self._jsonl = (self._temporary_dir / "train.jsonl").open(
                    "w", encoding="utf-8"
                )
                self._write_build_state()
        else:
            self._temporary_dir = Path(
                tempfile.mkdtemp(
                    prefix=f".{self.output_dir.name}.building-",
                    dir=str(self.output_dir.parent),
                )
            )
            self._jsonl = (self._temporary_dir / "train.jsonl").open(
                "w", encoding="utf-8"
            )
        self._last_source_index = self.resume_source_index - 1

    def _distribution_payload(self) -> dict[str, Any]:
        return {
            "mode": self.distribution_spec.mode,
            "temperature": self.distribution_spec.temperature,
            "top_k": self.distribution_spec.top_k,
            "vocab_size": self.input_abi.logits_vocab_size,
        }

    @property
    def _build_state_path(self) -> Path:
        return self._temporary_dir / "build_state.json"

    def _build_state_payload(self) -> dict[str, Any]:
        self._jsonl.flush()
        jsonl_size = os.fstat(self._jsonl.fileno()).st_size
        return {
            "version": BUILD_STATE_VERSION,
            "artifact_id": self.artifact_id,
            "shards": dict(self._shards),
            "row_count": self._row_count,
            "jsonl_size": jsonl_size,
            "resume_source_index": self.resume_source_index,
            "writer": {
                "shard_rows": self.shard_rows,
                "shard_max_bytes": self.shard_max_bytes,
                "storage_dtype": str(self.storage_dtype),
            },
        }

    def _write_build_state(self) -> None:
        if not self.resume:
            return
        payload = self._build_state_payload()
        temporary_path = self._build_state_path.with_suffix(".json.tmp")
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, self._build_state_path)

    def _restore_build_state(self) -> None:
        state_path = self._temporary_dir / "build_state.json"
        jsonl_path = self._temporary_dir / "train.jsonl"
        if not state_path.is_file() or not jsonl_path.is_file():
            raise ValueError(
                "Offline KD resumable staging directory has no complete build state."
            )
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        expected_keys = {
            "version",
            "artifact_id",
            "shards",
            "row_count",
            "jsonl_size",
            "resume_source_index",
            "writer",
        }
        if not isinstance(payload, dict) or set(payload) != expected_keys:
            raise ValueError("Offline KD build state fields differ from the resume protocol.")
        if payload["version"] != BUILD_STATE_VERSION:
            raise ValueError(f"Unsupported Offline KD build state {payload['version']!r}.")
        if payload["artifact_id"] != self.artifact_id:
            raise ValueError("Offline KD build state artifact identity differs from this run.")
        expected_writer = {
            "shard_rows": self.shard_rows,
            "shard_max_bytes": self.shard_max_bytes,
            "storage_dtype": str(self.storage_dtype),
        }
        if payload["writer"] != expected_writer:
            raise ValueError("Offline KD build state writer settings differ from this run.")
        shards = payload["shards"]
        if not isinstance(shards, dict) or not all(
            isinstance(name, str) and isinstance(digest, str)
            for name, digest in shards.items()
        ):
            raise TypeError("Offline KD build state shards must be a string mapping.")
        self._shards = {
            name: _require_digest(digest, role=f"build_state.shards.{name}")
            for name, digest in shards.items()
        }
        expected_shard_names = {
            f"teacher-{index:05d}.safetensors"
            for index in range(1, len(self._shards) + 1)
        }
        if set(self._shards) != expected_shard_names:
            raise ValueError(
                "Offline KD build state shard names must be a contiguous producer sequence."
            )
        for name, digest in self._shards.items():
            path = (self._temporary_dir / name).resolve()
            if not path.is_relative_to(self._temporary_dir.resolve()):
                raise ValueError("Offline KD build state shard resolves outside staging.")
            if file_sha256(path) != digest:
                raise ValueError(f"Offline KD resumable shard checksum mismatch for {name!r}.")
        for path in self._temporary_dir.glob("teacher-*.safetensors"):
            if path.name not in self._shards:
                path.unlink()
        self._row_count = int(payload["row_count"])
        self.resume_source_index = int(payload["resume_source_index"])
        jsonl_size = int(payload["jsonl_size"])
        if min(self._row_count, self.resume_source_index, jsonl_size) < 0:
            raise ValueError("Offline KD build state counters must be non-negative.")
        with jsonl_path.open("r+b") as handle:
            if os.fstat(handle.fileno()).st_size < jsonl_size:
                raise ValueError("Offline KD resumable JSONL is shorter than committed state.")
            handle.truncate(jsonl_size)
        self._jsonl = jsonl_path.open("a", encoding="utf-8")

    def add(self, row: OfflineKDArtifactRow) -> None:
        if self._finalized:
            raise RuntimeError("Cannot add rows after Offline KD artifact finalization.")
        distribution = row.distribution
        expected_kind = self.distribution_spec.mode
        if distribution.kind != expected_kind:
            raise ValueError(
                "Offline KD row distribution kind differs from the writer distribution spec."
            )
        if distribution.vocab_size != self.input_abi.logits_vocab_size:
            raise ValueError("Offline KD row distribution vocabulary differs from input ABI.")
        if distribution.kind == "topk_tail" and (
            distribution.top_k != self.distribution_spec.top_k
            or not math.isclose(
                float(distribution.temperature or 0.0),
                float(self.distribution_spec.temperature or 0.0),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(
                "Offline KD row top-k projection differs from the writer distribution spec."
            )
        if distribution.num_positions != int(row.completion_token_ids.numel()):
            raise ValueError("Offline KD row distribution does not align with completion tokens.")
        if len(row.media_sha256) != 32:
            raise ValueError("Offline KD row media fingerprint must contain 32 bytes.")
        if self.resume:
            if type(row.source_index) is not int or row.source_index < self.resume_source_index:
                raise ValueError(
                    "Resumable Offline KD rows require source_index at or after "
                    "resume_source_index."
                )
            if row.source_index <= self._last_source_index:
                raise ValueError(
                    "Resumable Offline KD rows require strictly increasing source_index values."
                )
        prepared = self._prepare_row(row)
        row_bytes = self._row_tensor_bytes(prepared)
        if self._rows and self._pending_bytes + row_bytes > self.shard_max_bytes:
            self._flush_shard()
        self._rows.append(prepared)
        if self.resume:
            assert prepared.source_index is not None
            self._last_source_index = prepared.source_index
        self._pending_bytes += row_bytes
        if len(self._rows) >= self.shard_rows or self._pending_bytes >= self.shard_max_bytes:
            self._flush_shard()

    def _prepare_row(self, row: OfflineKDArtifactRow) -> OfflineKDArtifactRow:
        distribution = row.distribution
        if distribution.kind == "dense_logits":
            assert distribution.dense_logits is not None
            prepared_distribution = TeacherDistribution.from_dense_logits(
                distribution.dense_logits.detach().cpu().to(dtype=self.storage_dtype)
            )
        else:
            assert distribution.topk_token_ids is not None
            assert distribution.topk_log_probs is not None
            prepared_distribution = TeacherDistribution(
                kind="topk_tail",
                vocab_size=distribution.vocab_size,
                topk_token_ids=distribution.topk_token_ids.detach().cpu().long(),
                # Probability buckets are small relative to dense logits. Keep them in
                # FP32 so the default FP16 dense storage policy cannot invalidate their
                # normalization after serialization.
                topk_log_probs=distribution.topk_log_probs.detach().cpu().float(),
                tail_log_probs=(
                    None
                    if distribution.tail_log_probs is None
                    else distribution.tail_log_probs.detach().cpu().float()
                ),
                temperature=distribution.temperature,
            )
        return OfflineKDArtifactRow(
            source_payload=dict(row.source_payload),
            input_token_ids=row.input_token_ids.detach().cpu().long(),
            completion_token_ids=row.completion_token_ids.detach().cpu().long(),
            media_sha256=bytes(row.media_sha256),
            distribution=prepared_distribution,
            source_index=row.source_index,
        )

    @staticmethod
    def _row_tensor_bytes(row: OfflineKDArtifactRow) -> int:
        tensors = [row.input_token_ids, row.completion_token_ids]
        distribution = row.distribution
        if distribution.dense_logits is not None:
            tensors.append(distribution.dense_logits)
        if distribution.topk_token_ids is not None:
            tensors.append(distribution.topk_token_ids)
        if distribution.topk_log_probs is not None:
            tensors.append(distribution.topk_log_probs)
        if distribution.tail_log_probs is not None:
            tensors.append(distribution.tail_log_probs)
        return 32 + sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    def _flush_shard(self) -> None:
        if not self._rows:
            return
        shard_index = len(self._shards) + 1
        shard_name = f"teacher-{shard_index:05d}.safetensors"
        position_offsets = [0]
        input_offsets = [0]
        for row in self._rows:
            position_offsets.append(position_offsets[-1] + row.distribution.num_positions)
            input_offsets.append(input_offsets[-1] + int(row.input_token_ids.numel()))
        tensors: dict[str, torch.Tensor] = {
            "row_offsets": torch.tensor(position_offsets, dtype=torch.long),
            "completion_token_ids": torch.cat(
                [row.completion_token_ids.detach().cpu().long() for row in self._rows]
            ),
            "input_row_offsets": torch.tensor(input_offsets, dtype=torch.long),
            "input_token_ids": torch.cat(
                [row.input_token_ids.detach().cpu().long() for row in self._rows]
            ),
            "media_sha256": torch.tensor(
                [list(row.media_sha256) for row in self._rows], dtype=torch.uint8
            ),
        }
        if self.distribution_spec.mode == "dense_logits":
            tensors["dense_logits"] = torch.cat(
                [row.distribution.dense_logits for row in self._rows]
            )
        else:
            tensors["topk_token_ids"] = torch.cat(
                [row.distribution.topk_token_ids for row in self._rows]
            ).detach().cpu().long()
            tensors["topk_log_probs"] = torch.cat(
                [row.distribution.topk_log_probs for row in self._rows]
            )
            tail_rows = [row.distribution.tail_log_probs for row in self._rows]
            if not all(tail is None for tail in tail_rows):
                if any(tail is None for tail in tail_rows):
                    raise ValueError("Offline KD shard mixes distributions with different tails.")
                tensors["tail_log_probs"] = torch.cat(tail_rows)
        shard_path = self._temporary_dir / shard_name
        save_file(tensors, str(shard_path))
        with shard_path.open("rb") as handle:
            os.fsync(handle.fileno())
        self._shards[shard_name] = file_sha256(shard_path)
        for local_row, row in enumerate(self._rows):
            payload = dict(row.source_payload)
            if "distillation_ref" in payload:
                raise ValueError("Producer source payload already contains distillation_ref.")
            payload["distillation_ref"] = {
                "artifact_id": self.artifact_id,
                "shard": shard_name,
                "row": local_row,
            }
            self._jsonl.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._row_count += 1
        self._jsonl.flush()
        os.fsync(self._jsonl.fileno())
        source_indices = [
            row.source_index for row in self._rows if row.source_index is not None
        ]
        if source_indices:
            self.resume_source_index = max(source_indices) + 1
        self._write_build_state()
        self._rows.clear()
        self._pending_bytes = 0

    def finalize(self) -> Path:
        if self._finalized:
            raise RuntimeError("Offline KD artifact was already finalized.")
        self._flush_shard()
        if self._row_count <= 0:
            raise ValueError("Offline KD artifact cannot be finalized without rows.")
        self._jsonl.flush()
        os.fsync(self._jsonl.fileno())
        self._jsonl.close()
        manifest = {
            "version": ARTIFACT_VERSION,
            "artifact_id": self.artifact_id,
            "teacher": {
                "model": self.teacher_model,
                "checkpoint_fingerprint": self.teacher_checkpoint_fingerprint,
            },
            "input_abi": self.input_abi.to_dict(),
            "input_contract": self.input_contract.to_dict(),
            "distribution": self._distribution_payload(),
            "build": {
                "source_fingerprint": self.source_fingerprint,
                "denylist_fingerprint": self.denylist_fingerprint,
            },
            "shards": self._shards,
        }
        manifest_path = self._temporary_dir / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(self._temporary_dir, self.output_dir)
        self._finalized = True
        state_path = self.output_dir / "build_state.json"
        if state_path.exists():
            state_path.unlink()
        return self.output_dir

    def abort(self) -> None:
        if not self._jsonl.closed:
            self._jsonl.close()
        if self._temporary_dir.exists() and not self.resume:
            shutil.rmtree(self._temporary_dir)

    def __enter__(self) -> "OfflineKDArtifactWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        _ = exc, traceback
        if exc_type is not None or not self._finalized:
            self.abort()


def _source_payload(item: Mapping[str, Any]) -> dict[str, Any]:
    image_paths = tuple(str(path) for path in tuple(item.get("image_paths") or ()))
    payload = dict(item.get("extra", {}))
    reserved = {
        "images",
        "image_path",
        "sample_id",
        "target_text",
        "target_reasoning_content",
        "messages",
        "system_prompt",
        "user_prompt",
        "prompt_args",
    }
    collisions = reserved & set(payload)
    if collisions:
        raise ValueError(
            "Offline KD source extra fields collide with canonical SFT fields: "
            f"{sorted(collisions)}."
        )
    payload.update({
        "images": list(image_paths),
        "sample_id": str(item.get("sample_id", "")),
        "target_text": str(item["target_text"]),
        "system_prompt": str(item.get("system_prompt", "")),
        "user_prompt": str(item.get("user_prompt", "")),
        "prompt_args": dict(item.get("prompt_args", {})),
    })
    if item.get("messages") is not None:
        payload["messages"] = item["messages"]
    if item.get("target_reasoning_content") is not None:
        payload["target_reasoning_content"] = str(item["target_reasoning_content"])
    if len(image_paths) == 1:
        payload.pop("images")
        payload["image_path"] = image_paths[0]
    return payload


def _validate_producer_config(config: RuntimeConfig) -> None:
    if config.algorithm.name != "sft":
        raise ValueError("Offline KD producer requires an SFT config with fixed targets.")
    for dataset in config.data.datasets:
        if dataset.source_type != "jsonl_sft":
            raise ValueError("Offline KD producer accepts only jsonl_sft sources.")
        if dataset.online_transforms:
            raise ValueError(
                "Offline KD producer cannot materialize online transforms because their media "
                "effects are not representable by immutable source paths."
            )


def _build_prompt_only_selection_dataset(
    config: RuntimeConfig,
    *,
    max_rows: int | None,
    source_rank: int = 0,
    source_world_size: int = 1,
) -> tuple[SFTDataset, str]:
    if (
        type(source_rank) is not int
        or type(source_world_size) is not int
        or source_world_size <= 0
        or source_rank < 0
        or source_rank >= source_world_size
    ):
        raise ValueError("Prompt-only source rank must satisfy 0 <= rank < world_size.")
    datasets = [dataset for dataset in config.data.datasets if dataset.enabled]
    if len(datasets) != 1:
        raise ValueError("Prompt-only Offline KD currently requires exactly one dataset.")
    source = datasets[0]
    if source.source_type != "jsonl_sft" or len(source.train_paths) != 1:
        raise ValueError(
            "Prompt-only Offline KD requires one explicit jsonl_sft train path."
        )
    if source.offline_transforms or source.online_transforms:
        raise ValueError("Prompt-only Offline KD does not accept data transforms.")
    if config.data.schedule.shuffle:
        raise ValueError("Prompt-only Offline KD selection must set data.schedule.shuffle=false.")
    path = Path(source.train_paths[0]).resolve()
    records: list[SFTRecord] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            source_index = line_no - 1
            if max_rows is not None and source_index >= max_rows:
                break
            if source_index % source_world_size != source_rank:
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise TypeError(f"Prompt-only selection row must be an object: {path}:{line_no}")
            if raw.get("target_text") != "":
                raise ValueError(
                    "Prompt-only selection target_text must be exactly empty: "
                    f"{path}:{line_no}"
                )
            raw_images = raw.get("images")
            if raw_images is None:
                raw_images = [raw.get("image_path", raw.get("image"))]
            if isinstance(raw_images, (str, Path)):
                raw_images = [raw_images]
            if not isinstance(raw_images, list) or not raw_images or any(
                not isinstance(value, (str, Path)) or not str(value).strip()
                for value in raw_images
            ):
                raise ValueError(f"Prompt-only selection requires image paths: {path}:{line_no}")
            image_paths = tuple(
                str(
                    Path(value).resolve()
                    if Path(value).is_absolute()
                    else (path.parent / Path(value)).resolve()
                )
                for value in raw_images
            )
            reserved = {
                "images",
                "image_path",
                "image",
                "target_text",
                "messages",
                "system_prompt",
                "user_prompt",
                "prompt_args",
                "sample_id",
                "dataset_name",
            }
            records.append(
                SFTRecord(
                    image_paths=image_paths,
                    target_text="",
                    dataset_name=source.dataset_name,
                    sample_id=str(raw.get("sample_id", "")).strip() or None,
                    messages=raw.get("messages"),
                    system_prompt=str(raw.get("system_prompt", "")),
                    user_prompt=str(raw.get("user_prompt", "")),
                    prompt_args=dict(raw.get("prompt_args") or {}),
                    extra={key: value for key, value in raw.items() if key not in reserved},
                )
            )
    if not records:
        raise ValueError("Prompt-only Offline KD selection contains no rows.")
    source_fingerprint = canonical_sha256(
        {
            "kind": "prompt_only_selection",
            "path": str(path),
            "sha256": file_sha256(path),
            "selected_rows": len(records),
            "max_rows": max_rows,
            "source_rank": source_rank,
            "source_world_size": source_world_size,
            "dataset_name": source.dataset_name,
        }
    )
    return (
        SFTDataset(
            records,
            split="train",
            media_snapshot_id=config.data.media_snapshot_id,
            suppress_decompression_bomb_warning=True,
        ),
        source_fingerprint,
    )


def produce_offline_kd_artifact(
    config: RuntimeConfig,
    *,
    output_dir: str | Path,
    distribution_spec: OfflineKDDistributionSpec,
    denylist_path: str | Path,
    shard_rows: int = 128,
    shard_max_bytes: int = 512 * 1024 * 1024,
    storage_dtype: torch.dtype = torch.float16,
    scorer_backend: str = "hf",
    scoring_batch_size: int = 1,
    max_rows: int | None = None,
    vllm_engine: Any | None = None,
    vllm_options: Mapping[str, Any] | None = None,
    resume: bool = False,
    expected_teacher_checkpoint_fingerprint: str | None = None,
) -> Path:
    """Run deterministic teacher forcing and atomically publish a v1 artifact."""

    _validate_producer_config(config)
    denylist = OfflineKDDenylist.load(denylist_path)
    if max_rows is not None and (type(max_rows) is not int or max_rows <= 0):
        raise ValueError("Offline KD max_rows must be a positive integer when provided.")
    data_center = ShaftDataCenter(
        config.data,
        seed=config.experiment.seed,
        train_sample_budget=max_rows,
    )
    bundle = data_center.build_dataset_bundle(SFTDataset)
    dataset = bundle.train_dataset
    source_fingerprint = str(bundle.train_execution_fingerprint or "").strip()
    _require_digest(source_fingerprint, role="train execution fingerprint")
    model_plan = materialize_resolved_model_artifact_identity(
        resolve_model_plan(config, require_immutable_artifact=False)
    )
    if not model_plan.artifact_identity.complete:
        raise ValueError(
            "Offline KD producer requires a complete immutable teacher artifact identity; "
            f"reasons={list(model_plan.artifact_identity.incomplete_reasons)}."
        )
    teacher_checkpoint_fingerprint = model_plan.artifact_identity.fingerprint
    if expected_teacher_checkpoint_fingerprint is not None and (
        _require_digest(
            expected_teacher_checkpoint_fingerprint,
            role="expected_teacher_checkpoint_fingerprint",
        )
        != teacher_checkpoint_fingerprint
    ):
        raise ValueError(
            "Expected teacher checkpoint fingerprint differs from Shaft's resolved immutable "
            "model artifact identity."
        )
    normalized_backend = str(scorer_backend).strip().lower()
    if normalized_backend not in {"hf", "vllm"}:
        raise ValueError("Offline KD scorer_backend must be 'hf' or 'vllm'.")
    if type(scoring_batch_size) is not int or scoring_batch_size <= 0:
        raise ValueError("Offline KD scoring_batch_size must be a positive integer.")
    if normalized_backend == "hf":
        if vllm_engine is not None or vllm_options:
            raise ValueError("Offline KD HF scorer does not accept vLLM runtime options.")
        artifacts = build_model_tokenizer_processor(config, resolved_model_plan=model_plan)
        resolved_scorer: OfflineKDTeacherScorer = HFOfflineKDTeacherScorer(
            model=artifacts.model,
            model_adapter=artifacts.model_adapter,
            distribution_spec=distribution_spec,
        )
    else:
        artifacts = build_model_input_artifacts(config, resolved_model_plan=model_plan)
        options = dict(vllm_options or {})
        resolved_scorer = VLLMOfflineKDTeacherScorer(
            model_name_or_path=model_plan.effective_model_name_or_path,
            distribution_spec=distribution_spec,
            vocab_size=artifacts.logits_vocab_size,
            trust_remote_code=model_plan.trust_remote_code,
            revision=model_plan.resolved_revision,
            dtype=str(config.model.torch_dtype),
            engine=vllm_engine,
            **options,
        )
    input_abi = _build_offline_kd_input_abi(artifacts)
    input_contract = build_offline_kd_input_contract(config)
    image_token_id: int | None = None
    if normalized_backend == "vllm":
        raw_image_token_id = getattr(artifacts.processor, "image_token_id", None)
        if raw_image_token_id is None:
            image_token = getattr(artifacts.processor, "image_token", None)
            if image_token is not None:
                raw_image_token_id = artifacts.tokenizer.convert_tokens_to_ids(image_token)
        if raw_image_token_id is None and model_plan.model_meta.family == "qwen":
            raise ValueError(
                "Qwen vLLM Offline KD requires a processor image token ID to validate "
                "one structured placeholder per image."
            )
        image_token_id = (
            None if raw_image_token_id is None else int(raw_image_token_id)
        )
    collator_kwargs = {
        "model_adapter": artifacts.model_adapter,
        "template": artifacts.template,
        "processor": artifacts.processor,
        "tokenizer": artifacts.tokenizer,
        # Pixel budgets have already been applied to the shared PIL objects below.
        "min_pixels": None,
        "max_pixels": None,
        "max_length": config.data.max_length,
        "add_eos_token": config.data.add_eos_token,
        "include_targets_in_inputs": True,
        "include_metadata": False,
        "loss_scale_name": "default",
        "layout": "padded",
        "packing_mode": "none",
        "collect_stats": False,
    }
    collator: SFTCollator
    if normalized_backend == "vllm":
        collator = ShaftOfflineKDVLLMScoringCollator(
            image_token_id=image_token_id,
            **collator_kwargs,
        )
    else:
        collator = SFTCollator(**collator_kwargs)
    with OfflineKDArtifactWriter(
        output_dir,
        teacher_model=model_plan.model_meta.model_type,
        teacher_checkpoint_fingerprint=teacher_checkpoint_fingerprint,
        input_abi=input_abi,
        input_contract=input_contract,
        distribution_spec=distribution_spec,
        source_fingerprint=source_fingerprint,
        denylist_fingerprint=denylist.fingerprint,
        shard_rows=shard_rows,
        shard_max_bytes=shard_max_bytes,
        storage_dtype=storage_dtype,
        resume=resume,
    ) as writer:
        pending_items: list[tuple[int, dict[str, Any]]] = []

        def flush_items() -> None:
            if not pending_items:
                return
            indexed_items = list(pending_items)
            pending_items.clear()
            items = [item for _, item in indexed_items]
            scoring_items = prepare_offline_kd_scoring_items(
                items,
                model_adapter=artifacts.model_adapter,
                min_pixels=config.data.min_pixels,
                max_pixels=config.data.max_pixels,
            )
            vllm_prompt_token_ids = None
            if isinstance(collator, ShaftOfflineKDVLLMScoringCollator):
                vllm_collation = collator.collate_for_vllm(scoring_items)
                batch = vllm_collation.model_inputs
                image_rows = vllm_collation.images
                vllm_prompt_token_ids = vllm_collation.prompt_token_ids
            else:
                batch = collator(scoring_items)
                image_rows = tuple(
                    tuple(row) if isinstance(row, (list, tuple)) else (row,)
                    for row in SFTCollator._processor_image_rows(scoring_items)
                )
            labels = batch["labels"]
            attention_mask = batch["attention_mask"]
            completion_mask = labels.ne(-100)
            input_token_ids = tuple(
                batch["input_ids"][row_index][attention_mask[row_index].bool()]
                for row_index in range(len(items))
            )
            distributions = resolved_scorer.score(
                OfflineKDScoringBatch(
                    model_inputs=batch,
                    completion_mask=completion_mask,
                    input_token_ids=input_token_ids,
                    prompt_completion_masks=tuple(
                        completion_mask[row_index][attention_mask[row_index].bool()]
                        for row_index in range(len(items))
                    ),
                    images=image_rows,
                    vllm_prompt_token_ids=vllm_prompt_token_ids,
                )
            )
            if len(distributions) != len(items):
                raise ValueError("Offline KD scorer distribution count differs from input rows.")
            for row_index, ((source_index, item), teacher_distribution) in enumerate(
                zip(indexed_items, distributions)
            ):
                image_paths = tuple(
                    str(path) for path in tuple(item.get("image_paths") or ())
                )
                writer.add(
                    OfflineKDArtifactRow(
                        source_payload=_source_payload(item),
                        input_token_ids=input_token_ids[row_index],
                        completion_token_ids=labels[row_index][completion_mask[row_index]],
                        media_sha256=media_content_fingerprint(image_paths),
                        distribution=teacher_distribution,
                        source_index=source_index,
                    )
                )

        for index in range(writer.resume_source_index, len(dataset)):
            item = dataset[index]
            if denylist.excludes(item):
                continue
            pending_items.append((index, item))
            if len(pending_items) >= scoring_batch_size:
                flush_items()
        flush_items()
        return writer.finalize()


def produce_detection_pseudo_kd_artifact(
    config: RuntimeConfig,
    *,
    output_dir: str | Path,
    denylist_path: str | Path,
    top_k: int = 64,
    max_tokens: int = 8000,
    shard_rows: int = 128,
    shard_max_bytes: int = 512 * 1024 * 1024,
    scoring_batch_size: int = 1,
    max_rows: int | None = None,
    source_rank: int = 0,
    source_world_size: int = 1,
    vllm_engine: Any | None = None,
    vllm_options: Mapping[str, Any] | None = None,
    resume: bool = False,
    expected_teacher_checkpoint_fingerprint: str | None = None,
) -> Path:
    """Greedily generate detection pseudo labels and top-k distributions in one pass."""

    _validate_producer_config(config)
    if type(scoring_batch_size) is not int or scoring_batch_size <= 0:
        raise ValueError("Offline KD scoring_batch_size must be a positive integer.")
    if max_rows is not None and (type(max_rows) is not int or max_rows <= 0):
        raise ValueError("Offline KD max_rows must be a positive integer when provided.")
    distribution_spec = OfflineKDDistributionSpec(
        mode="topk_tail", temperature=1.0, top_k=top_k
    )
    denylist = OfflineKDDenylist.load(denylist_path)
    dataset, source_fingerprint = _build_prompt_only_selection_dataset(
        config,
        max_rows=max_rows,
        source_rank=source_rank,
        source_world_size=source_world_size,
    )
    _require_digest(source_fingerprint, role="train execution fingerprint")
    model_plan = materialize_resolved_model_artifact_identity(
        resolve_model_plan(config, require_immutable_artifact=False)
    )
    if not model_plan.artifact_identity.complete:
        raise ValueError(
            "Offline KD producer requires a complete immutable teacher artifact identity; "
            f"reasons={list(model_plan.artifact_identity.incomplete_reasons)}."
        )
    teacher_checkpoint_fingerprint = model_plan.artifact_identity.fingerprint
    if expected_teacher_checkpoint_fingerprint is not None and (
        _require_digest(
            expected_teacher_checkpoint_fingerprint,
            role="expected_teacher_checkpoint_fingerprint",
        )
        != teacher_checkpoint_fingerprint
    ):
        raise ValueError(
            "Expected teacher checkpoint fingerprint differs from Shaft's resolved immutable "
            "model artifact identity."
        )
    artifacts = build_model_input_artifacts(config, resolved_model_plan=model_plan)
    raw_image_token_id = getattr(artifacts.processor, "image_token_id", None)
    if raw_image_token_id is None:
        image_token = getattr(artifacts.processor, "image_token", None)
        if image_token is not None:
            raw_image_token_id = artifacts.tokenizer.convert_tokens_to_ids(image_token)
    if raw_image_token_id is None and model_plan.model_meta.family == "qwen":
        raise ValueError("Qwen greedy Offline KD requires a processor image token ID.")
    image_token_id = None if raw_image_token_id is None else int(raw_image_token_id)
    options = dict(vllm_options or {})
    generator = VLLMOfflineKDAsyncGreedyGenerator(
        model_name_or_path=model_plan.effective_model_name_or_path,
        vocab_size=artifacts.logits_vocab_size,
        trust_remote_code=model_plan.trust_remote_code,
        revision=model_plan.resolved_revision,
        dtype=str(config.model.torch_dtype),
        top_k=top_k,
        max_tokens=max_tokens,
        engine=vllm_engine,
        **options,
    )
    collator = ShaftOfflineKDVLLMScoringCollator(
        image_token_id=image_token_id,
        model_adapter=artifacts.model_adapter,
        template=artifacts.template,
        processor=artifacts.processor,
        tokenizer=artifacts.tokenizer,
        min_pixels=None,
        max_pixels=None,
        max_length=config.data.max_length,
        add_eos_token=config.data.add_eos_token,
        include_targets_in_inputs=True,
        include_metadata=False,
        loss_scale_name="default",
        layout="padded",
        packing_mode="none",
        collect_stats=False,
    )
    bad_case_path = Path(f"{Path(output_dir).resolve()}.bad_cases.jsonl")
    bad_case_path.parent.mkdir(parents=True, exist_ok=True)
    bad_mode = "a" if resume else "w"
    input_abi = _build_offline_kd_input_abi(artifacts)
    with bad_case_path.open(bad_mode, encoding="utf-8") as bad_cases, OfflineKDArtifactWriter(
        output_dir,
        teacher_model=model_plan.model_meta.model_type,
        teacher_checkpoint_fingerprint=teacher_checkpoint_fingerprint,
        input_abi=input_abi,
        input_contract=build_offline_kd_input_contract(config),
        distribution_spec=distribution_spec,
        source_fingerprint=source_fingerprint,
        denylist_fingerprint=denylist.fingerprint,
        shard_rows=shard_rows,
        shard_max_bytes=shard_max_bytes,
        storage_dtype=torch.float16,
        resume=resume,
    ) as writer:
        def write_bad_case(
            source_index: int,
            item: Mapping[str, Any],
            generation: OfflineKDPseudoLabelGeneration,
            error: Exception,
        ) -> None:
            bad_cases.write(
                json.dumps(
                    {
                        "source_index": source_index,
                        "sample_id": item.get("sample_id"),
                        "image_paths": list(item.get("image_paths") or ()),
                        "raw_text": generation.raw_text,
                        "completion_token_ids": generation.completion_token_ids.tolist(),
                        "finish_reason": generation.finish_reason,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            bad_cases.flush()

        def prepare_request(
            source_index: int,
            item: dict[str, Any],
        ) -> _DetectionPseudoKDRequest:
            prepared = prepare_offline_kd_scoring_items(
                [item],
                model_adapter=artifacts.model_adapter,
                min_pixels=config.data.min_pixels,
                max_pixels=config.data.max_pixels,
            )
            prompt_collation = collator.collate_prompts_for_vllm(prepared)
            return _DetectionPseudoKDRequest(
                source_index=source_index,
                source_item=item,
                prepared_item=prepared[0],
                prompt_token_ids=prompt_collation.prompt_token_ids[0],
                expanded_prompt_token_ids=prompt_collation.expanded_prompt_token_ids[0],
                images=prompt_collation.images[0],
            )

        def materialize_row(
            request: _DetectionPseudoKDRequest,
            generation: OfflineKDPseudoLabelGeneration,
        ) -> OfflineKDArtifactRow:
            supervised = dict(request.prepared_item)
            supervised["target_text"] = generation.raw_text
            full = collator.collate_for_vllm([supervised])
            labels = full.model_inputs["labels"]
            attention_mask = full.model_inputs["attention_mask"]
            completion_mask = labels.ne(-100)
            full_vllm_ids = full.prompt_token_ids[0]
            expected_vllm_ids = (
                *generation.request_prompt_token_ids,
                *generation.completion_token_ids.tolist(),
            )
            if full_vllm_ids != expected_vllm_ids:
                raise ValueError(
                    "Greedy output token IDs do not round-trip through the student SFT "
                    f"template for sample_id={request.source_item.get('sample_id')!r}."
                )
            row_attention = attention_mask[0].bool()
            input_token_ids = full.model_inputs["input_ids"][0][row_attention]
            completion_token_ids = labels[0][completion_mask[0]]
            expected_expanded_ids = torch.tensor(
                (
                    *generation.expanded_prompt_token_ids,
                    *generation.completion_token_ids.tolist(),
                ),
                dtype=torch.long,
                device=input_token_ids.device,
            )
            if not torch.equal(input_token_ids, expected_expanded_ids):
                raise ValueError(
                    "Greedy expanded prompt+completion IDs differ from local SFT collation "
                    f"for sample_id={request.source_item.get('sample_id')!r}."
                )
            if not torch.equal(completion_token_ids.cpu(), generation.completion_token_ids):
                raise ValueError(
                    "Greedy completion IDs differ from local SFT supervision IDs for "
                    f"sample_id={request.source_item.get('sample_id')!r}."
                )
            return OfflineKDArtifactRow(
                source_payload=_source_payload(supervised),
                input_token_ids=input_token_ids,
                completion_token_ids=completion_token_ids,
                media_sha256=media_content_fingerprint(
                    tuple(
                        str(path)
                        for path in request.source_item.get("image_paths") or ()
                    )
                ),
                distribution=generation.distribution,
                source_index=request.source_index,
            )

        async def run_pipeline() -> None:
            active: dict[asyncio.Task[OfflineKDPseudoLabelGeneration], _DetectionPseudoKDRequest] = {}
            submitted_order: deque[int] = deque()
            completed: dict[int, OfflineKDArtifactRow | None] = {}
            next_source_index = writer.resume_source_index
            source_exhausted = False
            reorder_limit = max(scoring_batch_size * 32, scoring_batch_size)

            async def submit_next() -> bool:
                nonlocal next_source_index, source_exhausted
                while next_source_index < len(dataset):
                    source_index = next_source_index
                    next_source_index += 1
                    item = dataset[source_index]
                    if denylist.excludes(item):
                        continue
                    request = await asyncio.to_thread(prepare_request, source_index, item)
                    task = asyncio.create_task(
                        generator.generate_one(
                            request_id=f"offline-kd-{source_rank}-{source_index}",
                            prompt_token_ids=request.prompt_token_ids,
                            expanded_prompt_token_ids=request.expanded_prompt_token_ids,
                            images=request.images,
                        )
                    )
                    active[task] = request
                    submitted_order.append(source_index)
                    return True
                source_exhausted = True
                return False

            def commit_ready() -> None:
                while submitted_order and submitted_order[0] in completed:
                    source_index = submitted_order.popleft()
                    row = completed.pop(source_index)
                    if row is not None:
                        writer.add(row)

            async def refill() -> None:
                while (
                    not source_exhausted
                    and len(active) < scoring_batch_size
                    and len(active) + len(completed) < reorder_limit
                ):
                    if not await submit_next():
                        break

            await refill()
            try:
                while active:
                    done, _ = await asyncio.wait(
                        active,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for task in done:
                        request = active.pop(task)
                        generation = task.result()
                        try:
                            if generation.finish_reason != "stop":
                                raise ValueError(
                                    "Detection pseudo label did not finish with "
                                    "finish_reason='stop'."
                                )
                            validate_detection_pseudo_label(generation.raw_text)
                        except ValueError as exc:
                            write_bad_case(
                                request.source_index,
                                request.source_item,
                                generation,
                                exc,
                            )
                            completed[request.source_index] = None
                        else:
                            completed[request.source_index] = await asyncio.to_thread(
                                materialize_row,
                                request,
                                generation,
                            )
                    commit_ready()
                    await refill()
                commit_ready()
                if submitted_order or completed:
                    raise RuntimeError("Async Offline KD pipeline did not drain in source order.")
            finally:
                for task in active:
                    task.cancel()
                if active:
                    await asyncio.gather(*active, return_exceptions=True)

        try:
            asyncio.run(run_pipeline())
            result = writer.finalize()
            bad_cases.flush()
            return result
        finally:
            generator.shutdown()
