from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import importlib.util
from typing import Any

import torch


@dataclass(frozen=True)
class ModelCapabilities:
    supports_pixel_budget: bool = True
    is_multimodal: bool = True


@dataclass(frozen=True)
class ProcessorPolicy:
    supports_pixel_budget: bool = True

    def build_inputs(
        self,
        *,
        processor: Any,
        prompt_texts: list[str],
        images: list[Any],
        min_pixels: int | None,
        max_pixels: int | None,
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
    def build(self, config: Any, *, model_meta: "ModelMeta") -> "ModelArtifacts":
        raise NotImplementedError


@dataclass(frozen=True)
class ModelGroup:
    name: str
    model_ids: tuple[str, ...] = ()
    template: str | None = None
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
    model_groups: tuple[ModelGroup, ...] = ()
    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)
    processor_policy: ProcessorPolicy = field(default_factory=ProcessorPolicy)
    peft_policy: PeftPolicy = field(default_factory=lambda: DefaultPeftPolicy(target_modules=["all-linear"]))
    requires: tuple[str, ...] = ()
    additional_saved_files: tuple[str, ...] = ()
    loader: ModelLoader | None = None

    def with_loader(self, loader: ModelLoader) -> "ModelMeta":
        return ModelMeta(
            model_type=self.model_type,
            family=self.family,
            default_template=self.default_template,
            model_groups=self.model_groups,
            capabilities=self.capabilities,
            processor_policy=self.processor_policy,
            peft_policy=self.peft_policy,
            requires=self.requires,
            additional_saved_files=self.additional_saved_files,
            loader=loader,
        )

    def default_target_modules(self) -> list[str]:
        return self.peft_policy.default_target_modules()

    def resolve_target_modules(self, target_modules: list[str]) -> list[str]:
        return self.peft_policy.resolve_target_modules(target_modules)

    @property
    def candidate_templates(self) -> tuple[str, ...]:
        candidates = [self.default_template]
        candidates.extend(group.template for group in self.model_groups if group.template)
        return tuple(dict.fromkeys(str(item).strip() for item in candidates if str(item).strip()))

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
        return list(dict.fromkeys(str(item).strip() for item in merged if str(item).strip()))

    def required_saved_files(self, model_name_or_path: str | None = None) -> tuple[str, ...]:
        merged = list(self.additional_saved_files)
        if model_name_or_path:
            matched = self.get_matched_model_group(model_name_or_path)
            if matched is not None:
                merged.extend(matched.additional_saved_files)
        else:
            for group in self.model_groups:
                merged.extend(group.additional_saved_files)
        return tuple(dict.fromkeys(str(item).strip() for item in merged if str(item).strip()))

    def check_requires(self, model_name_or_path: str | None = None) -> None:
        missing: list[str] = []
        for requirement in self.all_requires(model_name_or_path):
            package = requirement.split(">=", 1)[0].split("==", 1)[0].strip()
            if package and importlib.util.find_spec(package) is None:
                missing.append(requirement)
        if missing:
            raise ImportError(
                f"Missing required packages for model_type={self.model_type!r}: {missing}"
            )


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
    model_info: ModelInfo
    template: object
