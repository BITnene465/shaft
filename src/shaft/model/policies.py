from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from shaft.plugins import Registry
from transformers import __version__ as transformers_version

from .types import (
    DefaultPeftPolicy,
    PeftPolicy,
    ProcessorPolicy,
    ShaftProcessedBatch,
    ShaftProcessorCostEstimate,
    ShaftProcessorTokenLayout,
)


@dataclass(frozen=True)
class QwenVLProcessorPolicy(ProcessorPolicy):
    def _validate_pixel_budget_contract(
        self,
        *,
        min_pixels: int | None,
        max_pixels: int | None,
    ) -> None:
        if not self.supports_pixel_budget and (
            min_pixels is not None or max_pixels is not None
        ):
            raise ValueError(
                "Qwen VL exact cost estimation received a pixel budget, but its "
                "ProcessorPolicy does not forward pixel budgets to the processor."
            )

    def _resolve_image_cost_contract(self, processor: Any) -> tuple[Any, int, int, Any]:
        image_processor = getattr(processor, "image_processor", None)
        patch_size = int(getattr(image_processor, "patch_size", 0) or 0)
        merge_size = int(getattr(image_processor, "merge_size", 0) or 0)
        if patch_size <= 0 or merge_size <= 0:
            raise ValueError(
                "Qwen VL cost estimation requires processor.image_processor.patch_size "
                "and merge_size."
            )
        patch_estimator = getattr(
            image_processor,
            "get_number_of_image_patches",
            None,
        )
        if not callable(patch_estimator):
            raise ValueError(
                "Qwen VL exact cost estimation requires "
                "image_processor.get_number_of_image_patches()."
            )
        return image_processor, patch_size, merge_size, patch_estimator

    def cost_semantics_signature(
        self,
        *,
        processor: Any,
        min_pixels: int | None,
        max_pixels: int | None,
    ) -> tuple[object, ...]:
        self._validate_pixel_budget_contract(
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        image_processor, patch_size, merge_size, patch_estimator = (
            self._resolve_image_cost_contract(processor)
        )
        estimator = getattr(patch_estimator, "__func__", patch_estimator)
        return (
            "shaft-qwen-vl-processor-cost-semantics-v1",
            bool(self.supports_pixel_budget),
            str(transformers_version),
            f"{type(processor).__module__}.{type(processor).__qualname__}",
            f"{type(image_processor).__module__}.{type(image_processor).__qualname__}",
            f"{getattr(estimator, '__module__', '')}.{getattr(estimator, '__qualname__', '')}",
            patch_size,
            int(getattr(image_processor, "temporal_patch_size", 0) or 0),
            merge_size,
            repr(getattr(image_processor, "size", None)),
            getattr(processor, "image_token_id", None),
            repr(getattr(processor, "image_token", None)),
            None if min_pixels is None else int(min_pixels),
            None if max_pixels is None else int(max_pixels),
        )

    def estimate_image_cost(
        self,
        *,
        processor: Any,
        image_sizes: tuple[tuple[int, int], ...],
        min_pixels: int | None,
        max_pixels: int | None,
    ) -> ShaftProcessorCostEstimate:
        self._validate_pixel_budget_contract(
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        image_processor, _, merge_size, get_number_of_image_patches = (
            self._resolve_image_cost_contract(processor)
        )
        images_kwargs: dict[str, int] = {}
        if min_pixels is not None:
            images_kwargs["min_pixels"] = int(min_pixels)
        if max_pixels is not None:
            images_kwargs["max_pixels"] = int(max_pixels)
        vision_patches = 0
        processed_image_tokens = 0
        for width, height in image_sizes:
            patches = int(
                get_number_of_image_patches(
                    height=int(height),
                    width=int(width),
                    images_kwargs=images_kwargs,
                )
            )
            if patches % (merge_size * merge_size) != 0:
                raise ValueError("Qwen VL vision patches do not align with merge_size.")
            vision_patches += patches
            processed_image_tokens += patches // (merge_size * merge_size)
        return ShaftProcessorCostEstimate(
            processed_image_tokens=processed_image_tokens,
            vision_patches=vision_patches,
            exact=True,
        )

    def estimate_token_layout(
        self,
        *,
        processor: Any,
        tokenizer: Any,
        rendered_token_ids: tuple[int, ...],
        image_costs: tuple[ShaftProcessorCostEstimate, ...],
    ) -> ShaftProcessorTokenLayout:
        image_token_id = getattr(processor, "image_token_id", None)
        if image_token_id is None:
            image_token = getattr(processor, "image_token", None)
            if image_token is not None and hasattr(tokenizer, "convert_tokens_to_ids"):
                image_token_id = tokenizer.convert_tokens_to_ids(image_token)
        if image_token_id is None:
            raise ValueError(
                "Qwen VL token-layout estimation requires image_token_id or image_token."
            )

        processed_boundaries = [0]
        image_index = 0
        for token_id in rendered_token_ids:
            increment = 1
            if int(token_id) == int(image_token_id):
                if image_index >= len(image_costs):
                    raise ValueError(
                        "Qwen VL rendered prompt has more image placeholders than image costs."
                    )
                increment = int(image_costs[image_index].processed_image_tokens)
                if increment <= 0:
                    raise ValueError(
                        "A Qwen VL image placeholder must expand to at least one token."
                    )
                image_index += 1
            processed_boundaries.append(processed_boundaries[-1] + increment)
        if image_index != len(image_costs):
            raise ValueError(
                "Qwen VL image costs do not match the rendered image placeholder count: "
                f"placeholders={image_index}, image_costs={len(image_costs)}."
            )
        return ShaftProcessorTokenLayout(
            processed_boundaries=tuple(processed_boundaries)
        )

    def build_token_layout(
        self,
        *,
        rendered_token_ids: tuple[int, ...],
        processed_batch: ShaftProcessedBatch,
        row_index: int,
    ) -> ShaftProcessorTokenLayout:
        processed_token_ids, attention_mask = self._extract_token_row(
            processed_batch=processed_batch,
            row_index=row_index,
        )
        mm_token_type_ids = processed_batch.model_inputs.get("mm_token_type_ids")
        if mm_token_type_ids is None:
            return super().build_token_layout(
                rendered_token_ids=rendered_token_ids,
                processed_batch=processed_batch,
                row_index=row_index,
            )

        token_ids = [int(value) for value in processed_token_ids.tolist()]
        token_types = [
            int(value) for value in mm_token_type_ids[row_index][attention_mask].tolist()
        ]
        if len(token_types) != len(token_ids):
            raise ValueError("mm_token_type_ids must align with processed input_ids.")

        canonical_ids: list[int] = []
        processed_boundaries = [0]
        cursor = 0
        while cursor < len(token_ids):
            token_id = token_ids[cursor]
            token_type = token_types[cursor]
            end = cursor + 1
            if token_type != 0:
                while (
                    end < len(token_ids)
                    and token_types[end] == token_type
                    and token_ids[end] == token_id
                ):
                    end += 1
            canonical_ids.append(token_id)
            processed_boundaries.append(end)
            cursor = end

        return self._finalize_token_layout(
            rendered_token_ids=rendered_token_ids,
            canonical_token_ids=canonical_ids,
            processed_boundaries=tuple(processed_boundaries),
            processed_token_count=len(token_ids),
        )


@dataclass(frozen=True)
class SmokeVLMProcessorPolicy(ProcessorPolicy):
    def cost_semantics_signature(
        self,
        *,
        processor: Any,
        min_pixels: int | None,
        max_pixels: int | None,
    ) -> tuple[object, ...]:
        _ = processor, min_pixels, max_pixels
        return ("shaft-smoke-vlm-processor-cost-semantics-v1",)

    def estimate_image_cost(
        self,
        *,
        processor: Any,
        image_sizes: tuple[tuple[int, int], ...],
        min_pixels: int | None,
        max_pixels: int | None,
    ) -> ShaftProcessorCostEstimate:
        _ = processor, min_pixels, max_pixels
        return ShaftProcessorCostEstimate(
            processed_image_tokens=0,
            vision_patches=16 * len(image_sizes),
            exact=True,
        )

    def estimate_token_layout(
        self,
        *,
        processor: Any,
        tokenizer: Any,
        rendered_token_ids: tuple[int, ...],
        image_costs: tuple[ShaftProcessorCostEstimate, ...],
    ) -> ShaftProcessorTokenLayout:
        _ = processor, tokenizer, image_costs
        return ShaftProcessorTokenLayout(
            processed_boundaries=tuple(range(len(rendered_token_ids) + 1))
        )

PROCESSOR_POLICY_REGISTRY: Registry[ProcessorPolicy] = Registry("model_processor_policy")
PEFT_POLICY_REGISTRY: Registry[PeftPolicy] = Registry("model_peft_policy")


def register_processor_policy(name: str, policy: ProcessorPolicy):
    return PROCESSOR_POLICY_REGISTRY.register(name, policy)


def register_peft_policy(name: str, policy: PeftPolicy):
    return PEFT_POLICY_REGISTRY.register(name, policy)


def build_processor_policy(name: str) -> ProcessorPolicy:
    return copy.deepcopy(PROCESSOR_POLICY_REGISTRY.get(name))


def build_peft_policy(name: str) -> PeftPolicy:
    return copy.deepcopy(PEFT_POLICY_REGISTRY.get(name))


register_processor_policy(
    "qwen_vl",
    QwenVLProcessorPolicy(
        supports_pixel_budget=True,
        supports_exact_image_cost=True,
        whole_batch_model_input_names=(
            "pixel_values",
            "image_grid_thw",
            "pixel_values_videos",
            "video_grid_thw",
            "second_per_grid_ts",
        ),
    ),
)
register_processor_policy(
    "smoke_vlm",
    SmokeVLMProcessorPolicy(
        supports_pixel_budget=False,
        supports_exact_image_cost=True,
        sample_aligned_model_input_names=("pixel_values",),
    ),
)
register_processor_policy(
    "identity",
    ProcessorPolicy(
        supports_pixel_budget=False,
        sample_aligned_model_input_names=("pixel_values",),
    ),
)

register_peft_policy("all_linear", DefaultPeftPolicy(target_modules=["all-linear"]))
