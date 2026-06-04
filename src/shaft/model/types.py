from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
import importlib.util
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from packaging.version import InvalidVersion, Version
import torch

from .sharding import ModelShardingPolicy


def _dedupe_non_empty(values: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(item).strip() for item in values if str(item).strip()))


def _matches_group_prefix(name: str, prefix: str) -> bool:
    normalized_name = str(name).strip()
    normalized_prefix = str(prefix).strip()
    return bool(normalized_prefix) and (
        normalized_name == normalized_prefix or normalized_name.startswith(f"{normalized_prefix}.")
    )


def _missing_requires(requires: tuple[str, ...]) -> list[str]:
    missing: list[str] = []
    for requirement in requires:
        if requirement.startswith("module:"):
            module_name = requirement.removeprefix("module:").strip()
            if not module_name or importlib.util.find_spec(module_name) is None:
                missing.append(requirement)
            continue
        operator = ""
        package = requirement
        expected_version = ""
        for candidate in (">=", "=="):
            if candidate in requirement:
                package, expected_version = requirement.split(candidate, 1)
                operator = candidate
                break
        package = package.strip()
        if package and importlib.util.find_spec(package) is None:
            missing.append(requirement)
            continue
        if package and operator and expected_version:
            try:
                installed = Version(version(package))
                expected = Version(expected_version.strip())
            except (InvalidVersion, PackageNotFoundError):
                missing.append(requirement)
                continue
            if operator == ">=" and installed < expected:
                missing.append(requirement)
            elif operator == "==" and installed != expected:
                missing.append(requirement)
    return missing


@contextmanager
def _temporary_processor_padding_side(
    *,
    tokenizer: Any | None,
    processor: Any,
    padding_side: str | None,
):
    normalized = str(padding_side).strip().lower() if padding_side is not None else ""
    if not normalized:
        yield
        return
    if normalized not in {"left", "right"}:
        raise ValueError("padding_side must be 'left' or 'right'.")

    previous: list[tuple[Any, str]] = []
    seen: set[int] = set()
    candidates = [tokenizer, getattr(processor, "tokenizer", None)]
    try:
        for candidate in candidates:
            if candidate is None or not hasattr(candidate, "padding_side"):
                continue
            candidate_id = id(candidate)
            if candidate_id in seen:
                continue
            seen.add(candidate_id)
            previous.append((candidate, str(getattr(candidate, "padding_side"))))
            setattr(candidate, "padding_side", normalized)
        yield
    finally:
        for candidate, value in reversed(previous):
            setattr(candidate, "padding_side", value)


@dataclass(frozen=True)
class ModelCapabilities:
    supports_pixel_budget: bool = True
    is_multimodal: bool = True


@dataclass(frozen=True)
class ModelModuleGroups:
    language_model: tuple[str, ...] = ()
    vision_tower: tuple[str, ...] = ()
    aligner: tuple[str, ...] = ()
    generator: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for key in ("language_model", "vision_tower", "aligner", "generator"):
            value = getattr(self, key)
            if isinstance(value, str):
                coerced = (value,) if value.strip() else ()
            else:
                coerced = _dedupe_non_empty(tuple(value))
            object.__setattr__(self, key, coerced)

    def prefixes_for_group(self, group_name: str) -> tuple[str, ...]:
        normalized = str(group_name).strip().lower()
        if normalized not in {"language_model", "vision_tower", "aligner", "generator"}:
            raise KeyError(f"Unknown model module group: {group_name!r}")
        return getattr(self, normalized)

    def resolve_group_for_name(self, name: str) -> str | None:
        normalized_name = str(name).strip()
        if not normalized_name:
            return None

        best_group: str | None = None
        best_prefix_len = -1
        for group_name in ("language_model", "vision_tower", "aligner", "generator"):
            for prefix in self.prefixes_for_group(group_name):
                if not _matches_group_prefix(normalized_name, prefix):
                    continue
                prefix_len = len(prefix)
                if prefix_len > best_prefix_len:
                    best_group = group_name
                    best_prefix_len = prefix_len
        return best_group


@dataclass(frozen=True)
class ProcessorPolicy:
    supports_pixel_budget: bool = True

    def build_inputs(
        self,
        *,
        processor: Any,
        tokenizer: Any | None,
        prompt_texts: list[str],
        images: list[Any],
        min_pixels: int | None,
        max_pixels: int | None,
        padding_side: str | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "text": prompt_texts,
            "images": images,
            "padding": True,
            "return_tensors": "pt",
        }
        if self.supports_pixel_budget:
            if min_pixels is not None:
                kwargs["min_pixels"] = int(min_pixels)
            if max_pixels is not None:
                kwargs["max_pixels"] = int(max_pixels)
        with _temporary_processor_padding_side(
            tokenizer=tokenizer,
            processor=processor,
            padding_side=padding_side,
        ):
            return processor(**kwargs)


class PeftPolicy(ABC):
    @abstractmethod
    def default_target_modules(self) -> list[str]:
        raise NotImplementedError

    def resolve_target_modules(self, target_modules: list[str]) -> list[str]:
        normalized = [str(item).strip() for item in target_modules if str(item).strip()]
        if not normalized or normalized == ["auto"]:
            return self.default_target_modules()
        return normalized


@dataclass(frozen=True)
class DefaultPeftPolicy(PeftPolicy):
    target_modules: list[str]

    def default_target_modules(self) -> list[str]:
        return list(self.target_modules)


class ModelLoader(ABC):
    @abstractmethod
    def build(
        self,
        config: Any,
        *,
        model_meta: "ModelMeta",
        model_adapter: "ShaftModelAdapter",
    ) -> "ModelArtifacts":
        raise NotImplementedError


@dataclass(frozen=True)
class ModelGroup:
    name: str
    model_ids: tuple[str, ...] = ()
    template: str | None = None
    capabilities: ModelCapabilities | None = None
    module_groups: ModelModuleGroups | None = None
    processor_policy: ProcessorPolicy | None = None
    peft_policy: PeftPolicy | None = None
    sharding_policy: ModelShardingPolicy | None = None
    requires: tuple[str, ...] = ()
    additional_saved_files: tuple[str, ...] = ()

    def matches(self, model_name_or_path: str) -> bool:
        normalized = str(model_name_or_path).strip().rstrip("/").lower()
        if not normalized:
            return False
        basename = normalized.rsplit("/", 1)[-1]
        return any(
            candidate == basename or candidate == normalized
            for candidate in (str(item).strip().lower() for item in self.model_ids if str(item).strip())
        )


@dataclass(frozen=True)
class ModelMeta:
    model_type: str
    family: str
    default_template: str
    hf_model_types: tuple[str, ...] = ()
    model_groups: tuple[ModelGroup, ...] = ()
    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)
    module_groups: ModelModuleGroups = field(default_factory=ModelModuleGroups)
    processor_policy: ProcessorPolicy = field(default_factory=ProcessorPolicy)
    peft_policy: PeftPolicy = field(default_factory=lambda: DefaultPeftPolicy(target_modules=["all-linear"]))
    sharding_policy: ModelShardingPolicy = field(default_factory=ModelShardingPolicy)
    requires: tuple[str, ...] = ()
    additional_saved_files: tuple[str, ...] = ()
    loader: ModelLoader | None = None

    def with_loader(self, loader: ModelLoader) -> "ModelMeta":
        return ModelMeta(
            model_type=self.model_type,
            family=self.family,
            default_template=self.default_template,
            hf_model_types=self.hf_model_types,
            model_groups=self.model_groups,
            capabilities=self.capabilities,
            module_groups=self.module_groups,
            processor_policy=self.processor_policy,
            peft_policy=self.peft_policy,
            sharding_policy=self.sharding_policy,
            requires=self.requires,
            additional_saved_files=self.additional_saved_files,
            loader=loader,
        )

    def resolve_adapter(
        self,
        *,
        model_name_or_path: str,
        template_type: str | None = None,
    ) -> "ShaftModelAdapter":
        matched = self.get_matched_model_group(model_name_or_path)
        resolved_template = str(template_type).strip().lower() if template_type else None
        if not resolved_template:
            resolved_template = (
                matched.template if matched is not None and matched.template else self.default_template
            )
        capabilities = (
            matched.capabilities if matched is not None and matched.capabilities is not None else self.capabilities
        )
        module_groups = (
            matched.module_groups if matched is not None and matched.module_groups is not None else self.module_groups
        )
        processor_policy = (
            matched.processor_policy
            if matched is not None and matched.processor_policy is not None
            else self.processor_policy
        )
        peft_policy = (
            matched.peft_policy if matched is not None and matched.peft_policy is not None else self.peft_policy
        )
        sharding_policy = (
            matched.sharding_policy
            if matched is not None and matched.sharding_policy is not None
            else self.sharding_policy
        )
        requires = list(self.requires)
        if matched is not None:
            requires.extend(matched.requires)
        additional_saved_files = list(self.additional_saved_files)
        if matched is not None:
            additional_saved_files.extend(matched.additional_saved_files)
        return ShaftModelAdapter(
            model_type=self.model_type,
            family=self.family,
            model_name_or_path=str(model_name_or_path),
            template_type=str(resolved_template).strip(),
            capabilities=capabilities,
            module_groups=module_groups,
            processor_policy=processor_policy,
            peft_policy=peft_policy,
            sharding_policy=sharding_policy,
            requires=_dedupe_non_empty(tuple(requires)),
            additional_saved_files=_dedupe_non_empty(tuple(additional_saved_files)),
            group_name=matched.name if matched is not None else None,
            model_meta=self,
        )

    def default_target_modules(self) -> list[str]:
        return self.peft_policy.default_target_modules()

    def resolve_target_modules(self, target_modules: list[str]) -> list[str]:
        return self.peft_policy.resolve_target_modules(target_modules)

    @property
    def candidate_templates(self) -> tuple[str, ...]:
        candidates = [self.default_template]
        candidates.extend(group.template for group in self.model_groups if group.template)
        return _dedupe_non_empty(tuple(candidates))

    def get_matched_model_group(self, model_name_or_path: str) -> ModelGroup | None:
        for group in self.model_groups:
            if group.matches(model_name_or_path):
                return group
        return None

    def resolve_template_type(self, model_name_or_path: str | None = None) -> str:
        if model_name_or_path:
            matched = self.get_matched_model_group(model_name_or_path)
            if matched is not None and matched.template:
                return matched.template
        return self.default_template

    def all_requires(self, model_name_or_path: str | None = None) -> list[str]:
        merged = list(self.requires)
        if model_name_or_path:
            matched = self.get_matched_model_group(model_name_or_path)
            if matched is not None:
                merged.extend(matched.requires)
        else:
            for group in self.model_groups:
                merged.extend(group.requires)
        return list(_dedupe_non_empty(tuple(merged)))

    def required_saved_files(self, model_name_or_path: str | None = None) -> tuple[str, ...]:
        merged = list(self.additional_saved_files)
        if model_name_or_path:
            matched = self.get_matched_model_group(model_name_or_path)
            if matched is not None:
                merged.extend(matched.additional_saved_files)
        else:
            for group in self.model_groups:
                merged.extend(group.additional_saved_files)
        return _dedupe_non_empty(tuple(merged))

    def check_requires(self, model_name_or_path: str | None = None) -> None:
        missing = _missing_requires(tuple(self.all_requires(model_name_or_path)))
        if missing:
            raise ImportError(
                f"Missing required packages for model_type={self.model_type!r}: {missing}"
            )


@dataclass(frozen=True)
class ShaftModelAdapter:
    model_type: str
    family: str
    model_name_or_path: str
    template_type: str
    capabilities: ModelCapabilities
    module_groups: ModelModuleGroups
    processor_policy: ProcessorPolicy
    peft_policy: PeftPolicy
    sharding_policy: ModelShardingPolicy = field(default_factory=ModelShardingPolicy)
    requires: tuple[str, ...] = ()
    additional_saved_files: tuple[str, ...] = ()
    group_name: str | None = None
    model_meta: ModelMeta | None = None

    def default_target_modules(self) -> list[str]:
        return self.peft_policy.default_target_modules()

    def resolve_target_modules(self, target_modules: list[str]) -> list[str]:
        return self.peft_policy.resolve_target_modules(target_modules)

    def resolve_fsdp_transformer_layer_cls_to_wrap(self, values: list[str]) -> list[str]:
        try:
            return self.sharding_policy.resolve_fsdp_transformer_layer_cls_to_wrap(values)
        except ValueError as exc:
            raise ValueError(
                "train.distributed.fsdp.transformer_layer_cls_to_wrap=['auto'] is not available "
                f"for model.model_type={self.model_type!r}. Configure explicit transformer layer class names."
            ) from exc

    def build_processor_inputs(
        self,
        *,
        processor: Any,
        tokenizer: Any | None = None,
        prompt_texts: list[str],
        images: list[Any],
        min_pixels: int | None,
        max_pixels: int | None,
        padding_side: str | None = None,
    ) -> dict[str, Any]:
        return self.processor_policy.build_inputs(
            processor=processor,
            tokenizer=tokenizer,
            prompt_texts=prompt_texts,
            images=images,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            padding_side=padding_side,
        )

    def required_saved_files(self) -> tuple[str, ...]:
        return _dedupe_non_empty(self.additional_saved_files)

    def check_requires(self) -> None:
        missing = _missing_requires(self.requires)
        if missing:
            raise ImportError(
                f"Missing required packages for model_type={self.model_type!r}: {missing}"
            )

    def build_model_info(
        self,
        *,
        torch_dtype: torch.dtype | str,
        max_model_len: int | None = None,
        quant_method: str | None = None,
        quant_bits: int | None = None,
    ) -> "ModelInfo":
        return ModelInfo(
            model_type=self.model_type,
            model_dir=self.model_name_or_path,
            torch_dtype=torch_dtype,
            max_model_len=max_model_len,
            quant_method=quant_method,
            quant_bits=quant_bits,
            is_multimodal=self.capabilities.is_multimodal,
            family=self.family,
        )

    def build_template(self):
        from shaft.template import build_template_from_meta, resolve_template_meta

        template_meta = resolve_template_meta(template_type=self.template_type, model_adapter=self)
        return build_template_from_meta(template_meta)


@dataclass(frozen=True)
class ModelInfo:
    model_type: str
    model_dir: str
    torch_dtype: torch.dtype | str
    max_model_len: int | None = None
    quant_method: str | None = None
    quant_bits: int | None = None
    is_multimodal: bool = False
    family: str | None = None


@dataclass
class ModelArtifacts:
    model: torch.nn.Module
    tokenizer: object
    processor: object
    model_meta: ModelMeta
    model_adapter: ShaftModelAdapter
    model_info: ModelInfo
    template: object
    finetune_plan: object | None = None
