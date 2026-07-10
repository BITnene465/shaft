from __future__ import annotations

import copy
from dataclasses import dataclass

from shaft.plugins import Registry

from .types import (
    DefaultPeftPolicy,
    PeftPolicy,
    ProcessorPolicy,
    ShaftProcessedBatch,
    ShaftProcessorTokenLayout,
)


@dataclass(frozen=True)
class QwenVLProcessorPolicy(ProcessorPolicy):
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
    "identity",
    ProcessorPolicy(
        supports_pixel_budget=False,
        sample_aligned_model_input_names=("pixel_values",),
    ),
)

register_peft_policy("all_linear", DefaultPeftPolicy(target_modules=["all-linear"]))
