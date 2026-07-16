from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
import inspect
from types import MethodType
from typing import Any

import torch
from transformers import __version__ as transformers_version

from .types import (
    SequenceExecutionPolicy,
    ShaftProcessorMediaManifest,
    ShaftSequenceExecutionContract,
)


_QWEN3VL_CORE_MODULE = "transformers.models.qwen3_vl.modeling_qwen3_vl"
_QWEN3VL_CORE_CLASS = "Qwen3VLModel"
_QWEN35_CORE_TYPES = (
    ("transformers.models.qwen3_5.modeling_qwen3_5", "Qwen3_5Model"),
    ("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe", "Qwen3_5MoeModel"),
)


@dataclass(frozen=True)
class Qwen3VLSequenceExecutionPolicy(SequenceExecutionPolicy):
    """Prepare Qwen3VL padding-free inputs without crossing logical samples."""

    def _capability_signature(self) -> tuple[str, ...]:
        return (
            "shaft-qwen3vl-sequence-execution-v1",
            f"transformers={transformers_version}",
            f"flash-attn={self._package_version('flash-attn')}",
        )

    def build_contract(
        self,
        *,
        layout: str,
        device_type: str,
        attention_implementation: str | None,
        torch_dtype: str,
        distributed_strategy: str,
        torch_compile: bool = False,
    ) -> ShaftSequenceExecutionContract:
        normalized_layout = str(layout).strip().lower()
        device = str(device_type).strip().lower()
        attention = str(attention_implementation or "").strip().lower()
        dtype = str(torch_dtype).strip().lower()
        strategy = str(distributed_strategy).strip().lower()
        if normalized_layout == "varlen":
            if bool(torch_compile):
                raise ValueError("Qwen3VL varlen does not yet support torch.compile.")
            if device == "cpu":
                if attention not in {"", "eager", "sdpa"}:
                    raise ValueError(
                        "CPU Qwen3VL varlen is a correctness oracle and requires "
                        "eager or SDPA."
                    )
                if dtype not in {"fp32", "float32", "bf16", "bfloat16"}:
                    raise ValueError(
                        "CPU Qwen3VL varlen correctness oracle requires float32 "
                        "or bfloat16."
                    )
            elif device == "cuda":
                if attention != "flash_attention_2":
                    raise ValueError(
                        "CUDA Qwen3VL varlen requires attention_implementation="
                        "'flash_attention_2'."
                    )
                if dtype not in {"bf16", "bfloat16", "fp16", "float16"}:
                    raise ValueError(
                        "CUDA Qwen3VL varlen requires bfloat16 or float16 dtype."
                    )
                if strategy != "ddp":
                    raise ValueError("CUDA Qwen3VL varlen currently supports DDP only.")
            else:
                raise ValueError(
                    f"Qwen3VL varlen does not support device_type={device!r}."
                )
        return ShaftSequenceExecutionContract(
            layout=normalized_layout,
            device_type=device,
            attention_implementation=attention or None,
            torch_dtype=dtype,
            distributed_strategy=strategy,
            torch_compile=torch_compile,
            capability_signature=self._capability_signature(),
        )

    def validate_runtime(
        self,
        *,
        model: Any,
        contract: ShaftSequenceExecutionContract,
    ) -> None:
        if contract.capability_signature != self._capability_signature():
            raise ValueError("Qwen3VL sequence execution contract is stale or foreign.")
        if contract.layout != "varlen":
            return
        qwen_core = self._resolve_core(model)
        if contract.device_type == "cpu":
            actual_backends = self._actual_attention_implementations(
                qwen_core
            )
            if not actual_backends or any(
                backend not in {"eager", "sdpa"} for backend in actual_backends
            ):
                raise ValueError(
                    "CPU Qwen3VL varlen could not verify an eager/SDPA concrete backend."
                )
            return
        actual_backends = self._actual_attention_implementations(
            qwen_core
        )
        if not actual_backends or any(
            backend != "flash_attention_2" for backend in actual_backends
        ):
            raise ValueError(
                "CUDA Qwen3VL varlen model did not retain the requested "
                "flash_attention_2 backend."
            )

    def prepare_training_inputs(
        self,
        *,
        model: Any,
        inputs: dict[str, Any],
    ) -> dict[str, Any]:
        prepared = dict(inputs)
        layout = prepared.pop("_shaft_varlen_layout", None)
        if layout is None:
            return prepared
        if not self._is_varlen_layout_plan(layout):
            raise ValueError("Qwen3VL varlen received an invalid host-side layout plan.")
        manifest = prepared.pop("_shaft_media_manifest", None)
        if not isinstance(manifest, ShaftProcessorMediaManifest):
            raise ValueError(
                "Qwen3VL varlen requires a validated one-image, image-only media manifest."
            )

        input_ids = self._require_flat_sequence(prepared, "input_ids")
        mm_token_type_ids = self._require_flat_sequence(
            prepared,
            "mm_token_type_ids",
        )
        if tuple(mm_token_type_ids.shape) != tuple(input_ids.shape):
            raise ValueError("Qwen3VL varlen mm_token_type_ids must align with input_ids.")
        if layout.total_tokens != int(input_ids.shape[-1]):
            raise ValueError("Qwen3VL varlen layout does not cover the flattened input_ids.")
        if prepared.get("use_cache") is True:
            raise ValueError("Qwen3VL varlen training does not accept use_cache=True.")
        for field_name in (
            "past_key_values",
            "cache_position",
            "inputs_embeds",
            "position_ids",
        ):
            if prepared.get(field_name) is not None:
                raise ValueError(
                    f"Qwen3VL varlen training does not accept precomputed {field_name}."
                )
        attention_mask = prepared.get("attention_mask")
        if attention_mask is not None:
            if (
                not torch.is_tensor(attention_mask)
                or tuple(attention_mask.shape) != tuple(input_ids.shape)
                or not bool(attention_mask.to(dtype=torch.bool).all())
            ):
                raise ValueError(
                    "Qwen3VL varlen attention_mask must be absent or an all-valid flat row."
                )

        qwen_core = self._resolve_core(model)
        self._validate_media_alignment(
            prepared=prepared,
            layout=layout,
            manifest=manifest,
            qwen_core=qwen_core,
        )

        image_grid_thw = prepared["image_grid_thw"]
        position_parts: list[torch.Tensor] = []
        expected_start = 0
        for segment in layout.segments:
            if int(segment.start) != expected_start:
                raise ValueError("Qwen3VL varlen segments must be contiguous and ordered.")
            expected_start = int(segment.stop)
            media = manifest.segments[int(segment.processor_row_index)]
            start = int(segment.start)
            stop = int(segment.stop)
            segment_ids = input_ids[:, start:stop]
            segment_types = mm_token_type_ids[:, start:stop]
            segment_grid = image_grid_thw[
                media.image_grids.start : media.image_grids.stop
            ]
            mrope_positions, _ = qwen_core.get_rope_index(
                input_ids=segment_ids,
                mm_token_type_ids=segment_types,
                image_grid_thw=segment_grid,
                video_grid_thw=None,
                attention_mask=None,
            )
            expected_shape = (3, 1, stop - start)
            if not torch.is_tensor(mrope_positions) or tuple(mrope_positions.shape) != expected_shape:
                raise ValueError(
                    "Qwen3VL get_rope_index returned an incompatible position layout."
                )
            scalar_positions = torch.arange(
                stop - start,
                dtype=input_ids.dtype,
                device=input_ids.device,
            ).view(1, 1, -1)
            position_parts.append(
                torch.cat(
                    (scalar_positions, mrope_positions.to(device=input_ids.device)),
                    dim=0,
                )
            )
        if expected_start != layout.total_tokens:
            raise ValueError("Qwen3VL varlen segments do not cover the full flattened row.")

        prepared.pop("attention_mask", None)
        prepared["position_ids"] = torch.cat(position_parts, dim=-1)
        prepared["use_cache"] = False
        return prepared

    @staticmethod
    def _require_flat_sequence(inputs: dict[str, Any], name: str) -> torch.Tensor:
        value = inputs.get(name)
        if not torch.is_tensor(value) or value.ndim != 2 or int(value.shape[0]) != 1:
            raise ValueError(f"Qwen3VL varlen {name} must have shape [1, total_tokens].")
        return value

    @staticmethod
    def _validate_media_alignment(
        *,
        prepared: dict[str, Any],
        layout: Any,
        manifest: ShaftProcessorMediaManifest,
        qwen_core: Any,
    ) -> None:
        image_grid_thw = prepared.get("image_grid_thw")
        pixel_values = prepared.get("pixel_values")
        if not torch.is_tensor(image_grid_thw) or tuple(image_grid_thw.shape) != (
            manifest.image_grid_count,
            3,
        ):
            raise ValueError("Qwen3VL varlen image_grid_thw does not match its manifest.")
        if not torch.is_tensor(pixel_values) or pixel_values.ndim < 1:
            raise ValueError("Qwen3VL varlen requires tensor pixel_values.")
        if int(pixel_values.shape[0]) != manifest.image_patch_count:
            raise ValueError("Qwen3VL varlen pixel_values does not match its media manifest.")
        if len(manifest.segments) != layout.logical_segment_count:
            raise ValueError("Qwen3VL varlen media and sequence segment counts differ.")
        if prepared.get("video_grid_thw") is not None or prepared.get("pixel_values_videos") is not None:
            raise ValueError("Qwen3VL varlen first release supports image-only samples.")

        for sequence_segment in layout.segments:
            row_index = int(sequence_segment.processor_row_index)
            if row_index < 0 or row_index >= len(manifest.segments):
                raise ValueError("Qwen3VL varlen media row is out of range.")
            media_segment = manifest.segments[row_index]
            if media_segment.processor_row_index != row_index:
                raise ValueError("Qwen3VL varlen media rows are not aligned.")
            if media_segment.image_grids.length != 1:
                raise ValueError(
                    "Qwen3VL varlen currently requires exactly one image per logical segment."
                )
            grid = image_grid_thw[media_segment.image_grids.start]
            expected_patches = int(grid.to(dtype=torch.long, device="cpu").prod().item())
            if media_segment.image_patches.length != expected_patches:
                raise ValueError("Qwen3VL varlen image patch slice does not match its grid.")

            start = int(sequence_segment.start)
            stop = int(sequence_segment.stop)
            token_types = prepared["mm_token_type_ids"][0, start:stop]
            if bool(((token_types != 0) & (token_types != 1)).any()):
                raise ValueError("Qwen3VL varlen first release accepts image/text tokens only.")
            image_mask = token_types.eq(1)
            previous_image = torch.cat(
                (
                    torch.zeros(1, dtype=torch.bool, device=image_mask.device),
                    image_mask[:-1],
                )
            )
            image_runs = int((image_mask & ~previous_image).sum().item())
            if image_runs != media_segment.image_grids.length:
                raise ValueError(
                    "Qwen3VL varlen image modality runs do not match the media manifest."
                )

            merge_size = int(qwen_core.config.vision_config.spatial_merge_size)
            if merge_size <= 0 or expected_patches % (merge_size * merge_size):
                raise ValueError("Qwen3VL varlen image grid is incompatible with merge_size.")
            expected_image_tokens = expected_patches // (merge_size * merge_size)
            image_token_count = int(image_mask.sum().item())
            input_ids = prepared["input_ids"][0, start:stop]
            placeholder_count = int(
                input_ids.eq(int(qwen_core.config.image_token_id)).sum().item()
            )
            if image_token_count != expected_image_tokens or placeholder_count != image_token_count:
                raise ValueError(
                    "Qwen3VL varlen image placeholder tokens do not match the media grid."
                )

    @staticmethod
    def _is_varlen_layout_plan(value: Any) -> bool:
        value_type = type(value)
        return bool(
            value_type.__module__ == "shaft.data.batching"
            and value_type.__name__ == "ShaftVarlenLayoutPlan"
            and hasattr(value, "segments")
            and hasattr(value, "total_tokens")
            and hasattr(value, "logical_segment_count")
        )

    @staticmethod
    def _actual_attention_implementations(qwen_core: Any) -> tuple[str, ...]:
        configs = [getattr(qwen_core, "config", None)]
        language_model = getattr(qwen_core, "language_model", None)
        if language_model is not None:
            configs.append(getattr(language_model, "config", None))
        return tuple(
            str(getattr(config, "_attn_implementation", "") or "").strip().lower()
            for config in configs
            if config is not None
        )

    @staticmethod
    def _package_version(package: str) -> str:
        try:
            return version(package)
        except PackageNotFoundError:
            return "missing"

    def _resolve_core(self, model: Any) -> Any:
        queue = [model]
        visited: set[int] = set()
        while queue:
            candidate = queue.pop(0)
            if candidate is None or id(candidate) in visited:
                continue
            visited.add(id(candidate))
            candidate_type = type(candidate)
            if (
                candidate_type.__module__ == _QWEN3VL_CORE_MODULE
                and candidate_type.__name__ == _QWEN3VL_CORE_CLASS
                and callable(getattr(candidate, "get_rope_index", None))
            ):
                return candidate
            for attribute in ("module", "base_model", "model"):
                child = getattr(candidate, attribute, None)
                if child is not None and child is not candidate:
                    queue.append(child)
        raise ValueError(
            "Qwen3VL varlen requires a trusted Transformers Qwen3VLModel core; "
            "remote-code and unknown wrappers are not enabled."
        )


@dataclass(frozen=True)
class Qwen35VLSequenceExecutionPolicy(Qwen3VLSequenceExecutionPolicy):
    """Packed execution for Qwen3.5/Qwen3.6 hybrid attention models.

    Qwen3.6 is currently implemented by Transformers through the qwen3_5
    architecture.  In addition to M-RoPE resets, the linear-attention state and
    causal convolution require explicit segment boundaries.
    """

    def _capability_signature(self) -> tuple[str, ...]:
        return (
            "shaft-qwen35vl-hybrid-sequence-execution-v2",
            "vision-kwarg-filter=v1",
            f"transformers={transformers_version}",
            f"flash-attn={self._package_version('flash-attn')}",
            f"flash-linear-attention={self._package_version('flash-linear-attention')}",
            f"causal-conv1d={self._package_version('causal-conv1d')}",
        )

    def build_contract(
        self,
        *,
        layout: str,
        device_type: str,
        attention_implementation: str | None,
        torch_dtype: str,
        distributed_strategy: str,
        torch_compile: bool = False,
    ) -> ShaftSequenceExecutionContract:
        normalized_layout = str(layout).strip().lower()
        if normalized_layout != "varlen":
            return super().build_contract(
                layout=layout,
                device_type=device_type,
                attention_implementation=attention_implementation,
                torch_dtype=torch_dtype,
                distributed_strategy=distributed_strategy,
                torch_compile=torch_compile,
            )
        device = str(device_type).strip().lower()
        attention = str(attention_implementation or "").strip().lower()
        dtype = str(torch_dtype).strip().lower()
        strategy = str(distributed_strategy).strip().lower()
        if torch_compile:
            raise ValueError("Qwen3.5/3.6 varlen does not yet support torch.compile.")
        if device != "cuda":
            raise ValueError(
                "Qwen3.5/3.6 hybrid varlen requires CUDA kernels; CPU execution "
                "would not safely isolate linear-attention state."
            )
        if attention != "flash_attention_2":
            raise ValueError(
                "CUDA Qwen3.5/3.6 varlen requires "
                "attention_implementation='flash_attention_2'."
            )
        if dtype not in {"bf16", "bfloat16", "fp16", "float16"}:
            raise ValueError("CUDA Qwen3.5/3.6 varlen requires bfloat16 or float16 dtype.")
        if strategy != "ddp":
            raise ValueError("CUDA Qwen3.5/3.6 varlen currently supports DDP only.")
        missing = [
            package
            for package in ("flash-attn", "flash-linear-attention", "causal-conv1d")
            if self._package_version(package) == "missing"
        ]
        if missing:
            raise ImportError(
                "Qwen3.5/3.6 varlen requires CUDA isolation kernels: "
                + ", ".join(missing)
                + "."
            )
        return ShaftSequenceExecutionContract(
            layout=normalized_layout,
            device_type=device,
            attention_implementation=attention,
            torch_dtype=dtype,
            distributed_strategy=strategy,
            torch_compile=False,
            capability_signature=self._capability_signature(),
        )

    def prepare_training_inputs(
        self,
        *,
        model: Any,
        inputs: dict[str, Any],
    ) -> dict[str, Any]:
        layout = inputs.get("_shaft_varlen_layout")
        prepared = super().prepare_training_inputs(model=model, inputs=inputs)
        if layout is None:
            return prepared
        boundaries = [0, *(int(segment.stop) for segment in layout.segments)]
        if boundaries[-1] != int(layout.total_tokens):
            raise ValueError("Qwen3.5/3.6 varlen boundaries do not cover all tokens.")
        lengths = [
            int(segment.stop) - int(segment.start) for segment in layout.segments
        ]
        sequence_ids = torch.cat(
            [
                torch.full(
                    (length,),
                    index,
                    dtype=torch.int32,
                    device=prepared["input_ids"].device,
                )
                for index, length in enumerate(lengths)
            ],
            dim=0,
        ).view(1, -1)
        cu_seqlens = torch.tensor(
            boundaries,
            dtype=torch.int32,
            device=prepared["input_ids"].device,
        )
        prepared["seq_idx"] = sequence_ids
        prepared["cu_seq_lens_q"] = cu_seqlens
        prepared["cu_seq_lens_k"] = cu_seqlens
        prepared["max_length_q"] = max(lengths)
        prepared["max_length_k"] = max(lengths)
        return prepared

    def validate_runtime(
        self,
        *,
        model: Any,
        contract: ShaftSequenceExecutionContract,
    ) -> None:
        super().validate_runtime(model=model, contract=contract)
        if contract.layout != "varlen":
            return
        core = self._resolve_core(model)
        if not bool(getattr(core, "_shaft_sequence_kwarg_filter_v2", False)):
            raise ValueError(
                "Qwen3.5/3.6 hybrid varlen runtime compatibility adapter is missing."
            )
        language_model = getattr(core, "language_model", None)
        layers = tuple(getattr(language_model, "layers", ()) or ())
        linear_layers = [
            layer
            for layer in layers
            if str(getattr(layer, "layer_type", "")).strip().lower()
            == "linear_attention"
        ]
        if not linear_layers:
            raise ValueError(
                "Qwen3.5/3.6 hybrid varlen could not verify any linear-attention layer."
            )
        for layer in linear_layers:
            linear_attn = getattr(layer, "linear_attn", None)
            if (
                linear_attn is None
                or getattr(linear_attn, "causal_conv1d_fn", None) is None
                or getattr(linear_attn, "chunk_gated_delta_rule", None) is None
            ):
                raise ValueError(
                    "Qwen3.5/3.6 hybrid varlen did not retain causal-conv1d and "
                    "flash-linear-attention isolation kernels."
                )

    def configure_runtime(
        self,
        *,
        model: Any,
        contract: ShaftSequenceExecutionContract,
    ) -> None:
        if contract.layout != "varlen":
            return
        core = self._resolve_core(model)
        if bool(getattr(core, "_shaft_sequence_kwarg_filter_v2", False)):
            return
        language_model = getattr(core, "language_model", None)
        self._require_variadic_keyword_contract(
            getattr(core, "forward", None),
            description="Qwen3.5/3.6 multimodal forward",
        )
        self._require_variadic_keyword_contract(
            getattr(language_model, "forward", None),
            description="Qwen3.5/3.6 language forward",
        )
        sequence_only_fields = {
            "seq_idx",
            "cu_seq_lens_q",
            "cu_seq_lens_k",
            "max_length_q",
            "max_length_k",
        }
        wrapped_methods = 0
        for method_name in ("get_image_features", "get_video_features"):
            original = getattr(core, method_name, None)
            if not callable(original):
                continue
            required_media_field = (
                "pixel_values" if method_name == "get_image_features" else "pixel_values_videos"
            )
            signature = inspect.signature(original)
            if required_media_field not in signature.parameters:
                raise ValueError(
                    f"Upstream {method_name} no longer exposes {required_media_field}; "
                    "the versioned Qwen3.5/3.6 runtime adapter is incompatible."
                )

            def _filtered_media_features(
                instance: Any,
                *args: Any,
                __original=original,
                **kwargs: Any,
            ) -> Any:
                _ = instance
                for field_name in sequence_only_fields:
                    kwargs.pop(field_name, None)
                return __original(*args, **kwargs)

            setattr(
                core,
                method_name,
                MethodType(_filtered_media_features, core),
            )
            wrapped_methods += 1
        if wrapped_methods == 0:
            raise ValueError(
                "Qwen3.5/3.6 hybrid varlen could not locate upstream media "
                "feature methods for boundary-kwarg isolation."
            )
        setattr(core, "_shaft_sequence_kwarg_filter_v2", True)

    @staticmethod
    def _require_variadic_keyword_contract(method: Any, *, description: str) -> None:
        if not callable(method):
            raise ValueError(f"{description} is unavailable in the installed Transformers.")
        try:
            signature = inspect.signature(method)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Cannot inspect {description} capability contract.") from exc
        if not any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        ):
            raise ValueError(
                f"{description} no longer accepts sequence boundary kwargs; "
                "refusing an unverified hybrid varlen runtime."
            )

    def _resolve_core(self, model: Any) -> Any:
        queue = [model]
        visited: set[int] = set()
        while queue:
            candidate = queue.pop(0)
            if candidate is None or id(candidate) in visited:
                continue
            visited.add(id(candidate))
            candidate_type = type(candidate)
            if (
                (candidate_type.__module__, candidate_type.__name__)
                in _QWEN35_CORE_TYPES
                and callable(getattr(candidate, "get_rope_index", None))
            ):
                return candidate
            for attribute in ("module", "base_model", "model"):
                child = getattr(candidate, attribute, None)
                if child is not None and child is not candidate:
                    queue.append(child)
        raise ValueError(
            "Qwen3.5/3.6 varlen requires a trusted Transformers Qwen3_5Model "
            "or Qwen3_5MoeModel core; remote-code and unknown wrappers are not enabled."
        )
