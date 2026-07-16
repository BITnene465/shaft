from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
import functools
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import DEFAULT, MagicMock, patch, sentinel

import pytest
import torch.distributed as dist
from transformers.models.auto.auto_factory import _LazyAutoMapping

import shaft.model.artifact_identity as model_artifact_identity
import shaft.utils.semantic_identity as semantic_identity
from shaft.model.artifact_identity import validate_loaded_remote_code_identity
from shaft.config import RuntimeConfig
from shaft.model import (
    ModelCapabilities,
    ModelGroup,
    ModelMeta,
    build_model_tokenizer_processor,
    build_model_meta,
    resolve_local_model_descriptor,
    resolve_model_plan,
    validate_model_artifact_checkpointability,
    validate_resolved_model_artifact,
)
from shaft.model.policies import build_peft_policy, build_processor_policy
from shaft.model.qwen3vl import _resolve_attn_implementation
from shaft.model.qwen3vl import Qwen3VLLoader
from shaft.model.generation import align_model_generation_config
from shaft.template import resolve_template_meta
from shaft.utils.semantic_identity import (
    callable_semantic_fingerprint,
    component_semantic_fingerprint,
)


_SEMANTIC_GLOBAL_SCALE = 1
_SEMANTIC_RUNTIME_REGISTRY: dict[str, int] = {}


class _SemanticPolicy:
    def __init__(self, scale: int) -> None:
        self.scale = scale

    def apply(self, value: int) -> int:
        return value * self.scale


_SEMANTIC_POLICY = _SemanticPolicy(1)


def _external_muon_type(step_value: int) -> type:
    def step(self) -> int:
        _ = self
        return step_value

    step.__module__ = "third_party.optim"
    step.__name__ = "step"
    step.__qualname__ = "Muon.step"
    return type(
        "Muon",
        (),
        {
            "__module__": "third_party.optim",
            "step": step,
        },
    )


_SEMANTIC_EXTERNAL_MUON = _external_muon_type(1)


class _LazySemanticRegistry(Mapping[str, int]):
    def __getitem__(self, key: str) -> int:
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        raise AssertionError("semantic identity must not iterate a lazy mapping")

    def __len__(self) -> int:
        return 1

    def items(self):
        raise AssertionError("semantic identity must not materialize a lazy mapping")


_SEMANTIC_LAZY_REGISTRY: Mapping[str, int] = _LazySemanticRegistry()


def _extension_policy(step_value: int):
    def policy() -> int:
        return step_value

    policy.__module__ = "fixture_semantic_extension"
    policy.__name__ = "policy"
    policy.__qualname__ = "policy"
    return policy


_SEMANTIC_EXTENSION_MODULE = ModuleType("fixture_semantic_extension")
_SEMANTIC_EXTENSION_MODULE.policy = _extension_policy(1)


def _semantic_scaled_loss(value: int) -> int:
    return value * _SEMANTIC_GLOBAL_SCALE


def _semantic_registry_value() -> int:
    return _SEMANTIC_RUNTIME_REGISTRY["item-0"]


def _semantic_lazy_registry_value() -> Mapping[str, int]:
    return _SEMANTIC_LAZY_REGISTRY


def _semantic_policy_value(value: int) -> int:
    return _SEMANTIC_POLICY.apply(value)


def _semantic_external_optimizer_type() -> type:
    return _SEMANTIC_EXTERNAL_MUON


def _semantic_extension_policy_value() -> int:
    return _SEMANTIC_EXTENSION_MODULE.policy()


def _loader_build_v1(self, config, *, model_meta, model_adapter, sequence_execution_contract=None):
    _ = self, config, model_meta, model_adapter, sequence_execution_contract
    return "v1"


def _loader_build_v2(self, config, *, model_meta, model_adapter, sequence_execution_contract=None):
    _ = self, config, model_meta, model_adapter, sequence_execution_contract
    return "v2"


def test_qwen3vl_meta_exposes_family_and_policies() -> None:
    model_meta = build_model_meta("qwen3vl")
    assert model_meta.family == "qwen"
    assert model_meta.processor_policy.supports_pixel_budget is True
    assert model_meta.module_groups.language_model == ("model",)
    assert model_meta.module_groups.vision_tower == ("model.visual",)
    assert model_meta.module_groups.aligner == (
        "model.visual.merger",
        "model.visual.deepstack_merger_list",
    )
    assert model_meta.module_groups.generator == ("lm_head",)
    assert model_meta.default_target_modules() == ["all-linear"]
    assert model_meta.default_template == "qwen3vl"
    assert model_meta.requires == ()
    assert model_meta.additional_saved_files == ()
    assert len(model_meta.model_groups) == 1
    assert model_meta.candidate_templates == ("qwen3vl",)


def test_generation_alignment_accepts_a_scalar_existing_eos_token() -> None:
    target = SimpleNamespace(
        config=SimpleNamespace(eos_token_id=None, bos_token_id=None, pad_token_id=None),
        generation_config=SimpleNamespace(
            do_sample=False,
            eos_token_id=2,
            bos_token_id=None,
            pad_token_id=None,
        ),
    )
    tokenizer = SimpleNamespace(eos_token_id=2, bos_token_id=1, pad_token_id=0)

    align_model_generation_config(target, tokenizer=tokenizer)

    assert target.config.eos_token_id == 2
    assert target.generation_config.eos_token_id == [2]


def test_qwen35vl_meta_exposes_family_and_policies() -> None:
    model_meta = build_model_meta("qwen35vl")
    assert model_meta.family == "qwen"
    assert model_meta.hf_model_types == ("qwen3_5", "qwen3_5_moe")
    assert model_meta.processor_policy.supports_pixel_budget is True
    assert model_meta.module_groups.language_model == ("model.language_model",)
    assert model_meta.module_groups.vision_tower == ("model.visual",)
    assert model_meta.module_groups.aligner == (
        "model.visual.merger",
        "model.visual.deepstack_merger_list",
    )
    assert model_meta.module_groups.generator == ("lm_head",)
    assert model_meta.default_target_modules() == ["all-linear"]
    assert model_meta.default_template == "qwen35vl"
    assert model_meta.requires == (
        "transformers>=5.10.1",
        "module:transformers.models.qwen3_5",
    )
    assert model_meta.candidate_templates == ("qwen35vl",)


def test_qwen36vl_alias_uses_same_template() -> None:
    model_meta = build_model_meta("qwen36vl")
    assert model_meta.default_template == "qwen35vl"
    assert model_meta.hf_model_types == ("qwen3_5", "qwen3_5_moe")
    assert model_meta.resolve_template_type("models/Qwen3.6-27B") == "qwen35vl"


def test_qwen35vl_dense_fsdp_auto_layers() -> None:
    adapter = build_model_meta("qwen35vl").resolve_adapter(model_name_or_path="models/Qwen3.6-27B")
    assert adapter.resolve_fsdp_transformer_layer_cls_to_wrap(["auto"]) == [
        "Qwen3_5DecoderLayer",
        "Qwen3_5VisionBlock",
    ]


def test_qwen35vl_moe_fsdp_auto_layers() -> None:
    adapter = build_model_meta("qwen35vl").resolve_adapter(
        model_name_or_path="models/Qwen3.6-35B-A3B"
    )
    assert adapter.resolve_fsdp_transformer_layer_cls_to_wrap(["auto"]) == [
        "Qwen3_5MoeDecoderLayer",
        "Qwen3_5MoeVisionBlock",
    ]
    assert adapter.requires == (
        "transformers>=5.10.1",
        "module:transformers.models.qwen3_5",
        "module:transformers.models.qwen3_5_moe",
    )


def test_qwen35vl_unknown_local_name_selects_variant_from_hf_config(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "arbitrary-release-name"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5_moe",
                "architectures": ["Qwen3_5MoeForConditionalGeneration"],
                "text_config": {"layer_types": ["linear_attention", "full_attention"]},
            }
        ),
        encoding="utf-8",
    )
    descriptor = resolve_local_model_descriptor(model_dir)
    assert descriptor is not None
    assert descriptor.hf_model_type == "qwen3_5_moe"

    adapter = build_model_meta("qwen35vl").resolve_adapter(
        model_name_or_path=str(model_dir),
        descriptor=descriptor,
    )
    assert adapter.group_name == "moe"
    assert adapter.resolve_fsdp_transformer_layer_cls_to_wrap(["auto"]) == [
        "Qwen3_5MoeDecoderLayer",
        "Qwen3_5MoeVisionBlock",
    ]


def test_qwen35vl_descriptor_overrides_misleading_catalog_basename(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "qwen3.6-27b"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5_moe",
                "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            }
        ),
        encoding="utf-8",
    )
    descriptor = resolve_local_model_descriptor(model_dir)
    assert descriptor is not None

    adapter = build_model_meta("qwen36vl").resolve_adapter(
        model_name_or_path=str(model_dir),
        descriptor=descriptor,
    )

    assert adapter.group_name == "moe"


def test_qwen35vl_custom_hub_checkpoint_resolves_config_before_group_selection() -> None:
    config = RuntimeConfig()
    config.model.model_type = "qwen36vl"
    config.model.model_name_or_path = "my-org/Qwen3.6-domain-sft"
    config.model.revision = "release-v2"

    with patch(
        "shaft.model.descriptor.PretrainedConfig.get_config_dict",
        return_value=(
            {
                "model_type": "qwen3_5_moe",
                "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            },
            {},
        ),
    ) as resolver:
        plan = resolve_model_plan(config)

    assert plan.model_adapter.group_name == "moe"
    assert plan.descriptor is not None
    assert plan.descriptor.source == "hf://my-org/Qwen3.6-domain-sft@release-v2"
    assert plan.effective_model_name_or_path == "my-org/Qwen3.6-domain-sft"
    assert plan.fingerprint
    resolver.assert_called_once_with(
        "my-org/Qwen3.6-domain-sft",
        revision="release-v2",
        cache_dir=None,
        local_files_only=False,
    )


def test_qwen35_model_plan_binds_inherited_loader_implementation() -> None:
    config = RuntimeConfig()
    config.model.model_type = "qwen35vl"
    config.model.model_name_or_path = "my-org/Qwen3.5-domain-sft"
    descriptor = (
        {
            "model_type": "qwen3_5",
            "architectures": ["Qwen3_5ForConditionalGeneration"],
        },
        {"_commit_hash": "a" * 40},
    )
    with (
        patch(
            "shaft.model.descriptor.PretrainedConfig.get_config_dict",
            return_value=descriptor,
        ),
        patch.object(Qwen3VLLoader, "build", _loader_build_v1),
    ):
        original = resolve_model_plan(config)
    with (
        patch(
            "shaft.model.descriptor.PretrainedConfig.get_config_dict",
            return_value=descriptor,
        ),
        patch.object(Qwen3VLLoader, "build", _loader_build_v2),
    ):
        changed = resolve_model_plan(config)

    assert original.fingerprint != changed.fingerprint


def test_component_identity_ignores_transient_local_class_helpers() -> None:
    class _Policy:
        def execute(self):
            return "stable"

    policy = _Policy()
    original = component_semantic_fingerprint(policy, role="fixture_policy")

    def _temporary_helper(self):
        return self.execute()

    _Policy.temporary_helper = _temporary_helper
    changed = component_semantic_fingerprint(policy, role="fixture_policy")

    assert changed == original


def test_callable_identity_binds_live_module_constants(monkeypatch) -> None:
    original = callable_semantic_fingerprint(
        _semantic_scaled_loss,
        role="fixture_loss",
    )
    monkeypatch.setattr(
        "tests.test_model_meta._SEMANTIC_GLOBAL_SCALE",
        9,
    )
    changed = callable_semantic_fingerprint(
        _semantic_scaled_loss,
        role="fixture_loss",
    )

    assert changed != original


def test_callable_identity_binds_same_named_external_callable_replacement(
    monkeypatch,
) -> None:
    original = callable_semantic_fingerprint(
        _semantic_external_optimizer_type,
        role="fixture_external_optimizer",
    )
    monkeypatch.setattr(
        "tests.test_model_meta._SEMANTIC_EXTERNAL_MUON",
        _external_muon_type(2),
    )
    changed = callable_semantic_fingerprint(
        _semantic_external_optimizer_type,
        role="fixture_external_optimizer",
    )

    assert changed != original


def test_component_identity_binds_live_class_constants(monkeypatch) -> None:
    class _Policy:
        scale = 1

        def execute(self, value: int) -> int:
            return value * self.scale

    policy = _Policy()
    original = component_semantic_fingerprint(policy, role="fixture_policy")
    monkeypatch.setattr(_Policy, "scale", 9)
    changed = component_semantic_fingerprint(policy, role="fixture_policy")

    assert changed != original


def test_component_identity_binds_replacement_of_declared_method(monkeypatch) -> None:
    class _Policy:
        def execute(self) -> str:
            return "original"

    policy = _Policy()
    original = component_semantic_fingerprint(policy, role="fixture_policy")

    def _replacement(self) -> str:
        _ = self
        return "changed"

    monkeypatch.setattr(_Policy, "execute", _replacement)
    changed = component_semantic_fingerprint(policy, role="fixture_policy")

    assert changed != original


def test_callable_identity_does_not_coerce_mapping_keys() -> None:
    def _builder(mapping):
        def _policy():
            return mapping

        return _policy

    integer_key = callable_semantic_fingerprint(
        _builder({1: "value"}),
        role="fixture_policy",
    )
    string_key = callable_semantic_fingerprint(
        _builder({"1": "value"}),
        role="fixture_policy",
    )

    assert integer_key != string_key


def test_callable_identity_supports_named_stdlib_sentinels() -> None:
    def _builder(default):
        def _policy(value=default):
            return value

        return _policy

    default_identity = callable_semantic_fingerprint(
        _builder(DEFAULT),
        role="fixture_policy",
    )
    repeated_identity = callable_semantic_fingerprint(
        _builder(DEFAULT),
        role="fixture_policy",
    )
    alternate_identity = callable_semantic_fingerprint(
        _builder(sentinel.ALTERNATE),
        role="fixture_policy",
    )

    assert default_identity == repeated_identity
    assert default_identity != alternate_identity


def test_callable_identity_binds_torch_distributed_reduce_op_defaults() -> None:
    def _builder(op):
        def _policy(value=op):
            return value

        return _policy

    sum_identity = callable_semantic_fingerprint(
        _builder(dist.ReduceOp.SUM),
        role="fixture_reduce_op",
    )
    repeated_identity = callable_semantic_fingerprint(
        _builder(dist.ReduceOp.SUM),
        role="fixture_reduce_op",
    )
    max_identity = callable_semantic_fingerprint(
        _builder(dist.ReduceOp.MAX),
        role="fixture_reduce_op",
    )

    assert sum_identity == repeated_identity
    assert sum_identity != max_identity


def test_callable_identity_hashes_large_runtime_containers_without_expanding_payload(
    monkeypatch,
) -> None:
    registry = {f"item-{index}": index for index in range(50_000)}
    monkeypatch.setattr(
        "tests.test_model_meta._SEMANTIC_RUNTIME_REGISTRY",
        registry,
    )
    payload = semantic_identity._callable_payload(  # noqa: SLF001
        _semantic_registry_value,
        context=semantic_identity._SemanticContext(),  # noqa: SLF001
    )
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    original = callable_semantic_fingerprint(
        _semantic_registry_value,
        role="fixture_registry_policy",
    )
    registry["item-49"] = -1
    changed = callable_semantic_fingerprint(
        _semantic_registry_value,
        role="fixture_registry_policy",
    )

    assert len(serialized) < 20_000
    assert changed != original


def test_callable_identity_rejects_opaque_lazy_mapping() -> None:
    with pytest.raises(TypeError, match="shaft_semantic_state"):
        callable_semantic_fingerprint(
            _semantic_lazy_registry_value,
            role="fixture_lazy_registry",
        )


def test_callable_and_component_identity_reject_custom_mapping_state() -> None:
    class _ValueMapping(Mapping[str, int]):
        def __init__(self, value: int) -> None:
            self.value = value

        def __getitem__(self, key: str) -> int:
            if key == "value":
                return self.value
            raise KeyError(key)

        def __iter__(self) -> Iterator[str]:
            yield "value"

        def __len__(self) -> int:
            return 1

    def _builder(mapping: Mapping[str, int]):
        def _policy(value=mapping):
            return value["value"]

        return _policy

    class _Observer:
        def __init__(self, mapping: Mapping[str, int]) -> None:
            self.mapping = mapping

    @dataclass
    class _DataclassMapping(Mapping[str, int]):
        value: int

        def __getitem__(self, key: str) -> int:
            if key == "value":
                return self.value
            raise KeyError(key)

        def __iter__(self) -> Iterator[str]:
            yield "value"

        def __len__(self) -> int:
            return 1

    for mapping in (
        _ValueMapping(1),
        _ValueMapping(2),
        _DataclassMapping(1),
        _DataclassMapping(2),
    ):
        with pytest.raises(TypeError, match="shaft_semantic_state"):
            callable_semantic_fingerprint(
                _builder(mapping),
                role="fixture_mapping_policy",
            )
        with pytest.raises(TypeError, match="shaft_semantic_state"):
            component_semantic_fingerprint(
                _Observer(mapping),
                role="fixture_mapping_observer",
            )


def test_callable_identity_binds_nested_object_type_and_implementation() -> None:
    class _PolicyA:
        def value(self) -> int:
            return 1

    class _PolicyB:
        def value(self) -> int:
            return 2

    def _builder(policy):
        def _execute() -> int:
            return policy.value()

        return _execute

    first = callable_semantic_fingerprint(
        _builder(_PolicyA()),
        role="fixture_nested_policy",
    )
    second = callable_semantic_fingerprint(
        _builder(_PolicyB()),
        role="fixture_nested_policy",
    )

    assert first != second


def test_callable_identity_binds_live_same_type_object_implementation(monkeypatch) -> None:
    original = callable_semantic_fingerprint(
        _semantic_policy_value,
        role="fixture_global_policy",
    )

    def _replacement(self, value: int) -> int:
        return value + self.scale

    monkeypatch.setattr(_SemanticPolicy, "apply", _replacement)
    changed = callable_semantic_fingerprint(
        _semantic_policy_value,
        role="fixture_global_policy",
    )

    assert changed != original


def test_hf_lazy_auto_mapping_identity_binds_behavior_not_access_cache(monkeypatch) -> None:
    mapping = _LazyAutoMapping(
        OrderedDict([("fixture", "FixtureConfig")]),
        OrderedDict([("fixture", "FixtureModel")]),
    )

    def _builder(value):
        def _policy(default=value):
            return default

        return _policy

    original = callable_semantic_fingerprint(
        _builder(mapping),
        role="fixture_hf_lazy_mapping",
    )
    mapping._modules["fixture_cache"] = object()  # noqa: SLF001
    cache_changed = callable_semantic_fingerprint(
        _builder(mapping),
        role="fixture_hf_lazy_mapping",
    )
    mapping._reverse_config_mapping = {"OtherConfig": "fixture"}  # noqa: SLF001
    behavior_changed = callable_semantic_fingerprint(
        _builder(mapping),
        role="fixture_hf_lazy_mapping",
    )

    assert cache_changed == original
    assert behavior_changed != original

    def _replacement_get(self, key, default=None):
        _ = self, key
        return default

    monkeypatch.setattr(_LazyAutoMapping, "get", _replacement_get)
    implementation_changed = callable_semantic_fingerprint(
        _builder(mapping),
        role="fixture_hf_lazy_mapping",
    )

    assert implementation_changed != behavior_changed


def test_callable_identity_binds_live_referenced_module_attribute(monkeypatch) -> None:
    original = callable_semantic_fingerprint(
        _semantic_extension_policy_value,
        role="fixture_extension_policy",
    )
    monkeypatch.setattr(
        _SEMANTIC_EXTENSION_MODULE,
        "policy",
        _extension_policy(9),
    )
    changed = callable_semantic_fingerprint(
        _semantic_extension_policy_value,
        role="fixture_extension_policy",
    )

    assert changed != original


def test_component_identity_rejects_opaque_declared_class_attribute() -> None:
    class _OpaquePolicy(Mapping[str, int]):
        def __getitem__(self, key: str) -> int:
            raise KeyError(key)

        def __iter__(self) -> Iterator[str]:
            raise AssertionError("opaque mapping must not be materialized")

        def __len__(self) -> int:
            return 1

    class _Component:
        policy = _OpaquePolicy()

        def execute(self) -> object:
            return self.policy

    with pytest.raises(TypeError, match="shaft_semantic_state"):
        component_semantic_fingerprint(
            _Component(),
            role="fixture_component",
        )


def test_component_identity_binds_cached_property_getter_and_binding(monkeypatch) -> None:
    class _Component:
        @functools.cached_property
        def policy(self) -> int:
            return 1

    original = component_semantic_fingerprint(
        _Component(),
        role="fixture_component",
    )

    def _replacement(self) -> int:
        return 2

    monkeypatch.setattr(
        _Component,
        "policy",
        functools.cached_property(_replacement),
    )
    changed = component_semantic_fingerprint(
        _Component(),
        role="fixture_component",
    )

    assert changed != original


def test_callable_identity_binds_same_typed_global_object_state(monkeypatch) -> None:
    original = callable_semantic_fingerprint(
        _semantic_policy_value,
        role="fixture_global_policy",
    )
    monkeypatch.setattr(
        "tests.test_model_meta._SEMANTIC_POLICY",
        _SemanticPolicy(9),
    )
    changed = callable_semantic_fingerprint(
        _semantic_policy_value,
        role="fixture_global_policy",
    )

    assert changed != original


def test_runtime_constant_identity_enforces_depth_and_node_budgets(
    monkeypatch,
) -> None:
    nested: object = 0
    for _ in range(65):
        nested = [nested]
    with pytest.raises(ValueError, match="depth budget"):
        semantic_identity._runtime_constant_payload(  # noqa: SLF001
            nested,
            context=semantic_identity._SemanticContext(),  # noqa: SLF001
        )

    monkeypatch.setattr(semantic_identity, "_RUNTIME_CONSTANT_NODE_BUDGET", 3)
    with pytest.raises(ValueError, match="node budget"):
        semantic_identity._runtime_constant_payload(  # noqa: SLF001
            [1, 2, 3],
            context=semantic_identity._SemanticContext(),  # noqa: SLF001
        )


def test_dynamic_attributes_cannot_fabricate_semantic_state() -> None:
    class _DynamicProxy:
        def __getattr__(self, name: str):
            if name == "shaft_semantic_state":
                raise AssertionError("dynamic semantic-state lookup is forbidden")
            raise AttributeError(name)

    for value in (MagicMock(), _DynamicProxy()):
        assert semantic_identity._semantic_state_provider(value) is None  # noqa: SLF001
        assert (  # noqa: SLF001
            semantic_identity._runtime_constant_payload(
                value,
                context=semantic_identity._SemanticContext(),  # noqa: SLF001
            )
            is semantic_identity._UNENCODABLE_RUNTIME_VALUE  # noqa: SLF001
        )


def test_peft_policy_semantic_identity_has_bounded_process_memory() -> None:
    script = """
import json
import resource
import time

from shaft.model.policies import build_peft_policy
from shaft.utils.semantic_identity import component_semantic_fingerprint

policy = build_peft_policy("all_linear")
before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
started = time.monotonic()
component_semantic_fingerprint(policy, role="peft")
after_first = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
for _ in range(10):
    component_semantic_fingerprint(policy, role="peft")
after_repeated = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(json.dumps({
    "elapsed_seconds": time.monotonic() - started,
    "first_delta_kib": after_first - before,
    "repeated_delta_kib": after_repeated - after_first,
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        text=True,
        timeout=20,
    )
    metrics = json.loads(completed.stdout.strip().splitlines()[-1])

    assert metrics["elapsed_seconds"] < 3.0
    assert metrics["first_delta_kib"] < 100 * 1024
    assert metrics["repeated_delta_kib"] < 32 * 1024


def test_hub_model_plan_binds_resolved_immutable_commit() -> None:
    config = RuntimeConfig()
    config.model.model_type = "qwen3vl"
    config.model.model_name_or_path = "my-org/Qwen3-VL-domain"
    config.model.revision = "main"
    commit = "A" * 40

    with patch(
        "shaft.model.descriptor.PretrainedConfig.get_config_dict",
        return_value=(
            {
                "model_type": "qwen3_vl",
                "architectures": ["Qwen3VLForConditionalGeneration"],
            },
            {"_commit_hash": commit},
        ),
    ):
        plan = resolve_model_plan(config)

    assert plan.revision == "main"
    assert plan.resolved_revision == commit.lower()
    assert plan.artifact_identity.complete is True
    assert plan.artifact_identity.kind == "hf_hub"
    assert plan.artifact_identity.resolved_revision == commit.lower()


def test_unresolved_hub_revision_is_not_checkpointable() -> None:
    config = RuntimeConfig()
    config.model.model_type = "qwen3vl"
    config.model.model_name_or_path = "my-org/Qwen3-VL-domain"

    with patch(
        "shaft.model.descriptor.PretrainedConfig.get_config_dict",
        return_value=(
            {
                "model_type": "qwen3_vl",
                "architectures": ["Qwen3VLForConditionalGeneration"],
            },
            {},
        ),
    ):
        plan = resolve_model_plan(config)

    assert plan.artifact_identity.complete is False
    with pytest.raises(ValueError, match="immutable base-model artifact identity"):
        validate_model_artifact_checkpointability(
            plan,
            save_strategy="steps",
            resume_requested=False,
        )


def test_local_model_plan_fingerprint_binds_weight_bytes(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "local-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_vl",
                "architectures": ["Qwen3VLForConditionalGeneration"],
            }
        ),
        encoding="utf-8",
    )
    weights = model_dir / "model.safetensors"
    weights.write_bytes(b"old-weights")
    config = RuntimeConfig()
    config.model.model_type = "qwen3vl"
    config.model.model_name_or_path = str(model_dir)

    original = resolve_model_plan(config, require_immutable_artifact=True)
    weights.write_bytes(b"new-weights")
    changed = resolve_model_plan(config, require_immutable_artifact=True)

    assert original.artifact_identity.complete is True
    assert changed.artifact_identity.complete is True
    assert original.artifact_identity.file_manifest[0][:2] == (
        "model.safetensors",
        len(b"old-weights"),
    )
    assert original.artifact_identity.fingerprint != changed.artifact_identity.fingerprint
    assert original.fingerprint != changed.fingerprint


def test_complete_local_model_plan_fingerprint_survives_artifact_relocation(
    tmp_path: Path,
) -> None:
    config_bytes = json.dumps(
        {
            "model_type": "qwen3_vl",
            "architectures": ["Qwen3VLForConditionalGeneration"],
        },
        sort_keys=True,
    ).encode("utf-8")
    roots = (tmp_path / "first" / "model", tmp_path / "second" / "model")
    for root in roots:
        root.mkdir(parents=True)
        (root / "config.json").write_bytes(config_bytes)
        (root / "model.safetensors").write_bytes(b"same-weights")

    def resolve(root: Path, *, immutable: bool):
        config = RuntimeConfig()
        config.model.model_type = "qwen3vl"
        config.model.model_name_or_path = str(root)
        return resolve_model_plan(
            config,
            require_immutable_artifact=immutable,
        )

    first = resolve(roots[0], immutable=True)
    relocated = resolve(roots[1], immutable=True)

    assert first.configured_model_name_or_path != relocated.configured_model_name_or_path
    assert first.effective_model_name_or_path != relocated.effective_model_name_or_path
    assert first.model_adapter.model_name_or_path != relocated.model_adapter.model_name_or_path
    assert first.artifact_identity.complete is True
    assert first.artifact_identity.fingerprint == relocated.artifact_identity.fingerprint
    assert first.fingerprint == relocated.fingerprint

    # The portability exception applies only to a complete content identity.
    incomplete_first = resolve(roots[0], immutable=False)
    incomplete_relocated = resolve(roots[1], immutable=False)
    assert incomplete_first.artifact_identity.complete is False
    assert incomplete_first.fingerprint != incomplete_relocated.fingerprint

    (roots[1] / "model.safetensors").write_bytes(b"new--weights")
    content_changed = resolve(roots[1], immutable=True)
    assert content_changed.artifact_identity.fingerprint != first.artifact_identity.fingerprint
    assert content_changed.fingerprint != first.fingerprint


def test_hub_model_plan_identity_remains_repository_and_revision_bound() -> None:
    commit = "a" * 40

    def resolve(repo_id: str, resolved_commit: str):
        config = RuntimeConfig()
        config.model.model_type = "qwen3vl"
        config.model.model_name_or_path = repo_id
        config.model.revision = "main"
        with patch(
            "shaft.model.descriptor.PretrainedConfig.get_config_dict",
            return_value=(
                {
                    "model_type": "qwen3_vl",
                    "architectures": ["Qwen3VLForConditionalGeneration"],
                },
                {"_commit_hash": resolved_commit},
            ),
        ):
            return resolve_model_plan(config, require_immutable_artifact=True)

    original = resolve("my-org/model-a", commit)
    other_repo = resolve("my-org/model-b", commit)
    other_revision = resolve("my-org/model-a", "b" * 40)

    assert original.artifact_identity.complete is True
    assert original.fingerprint != other_repo.fingerprint
    assert original.fingerprint != other_revision.fingerprint


def test_local_model_builder_hashes_each_artifact_exactly_twice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir = tmp_path / "local-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen3_vl"}),
        encoding="utf-8",
    )
    (model_dir / "model.safetensors").write_bytes(b"weights")
    config = RuntimeConfig()
    config.model.model_type = "qwen3vl"
    config.model.model_name_or_path = str(model_dir)
    original_hash = model_artifact_identity._file_sha256
    hashed: list[Path] = []

    def _counted_hash(path: Path) -> str:
        hashed.append(path)
        return original_hash(path)

    monkeypatch.setattr(model_artifact_identity, "_file_sha256", _counted_hash)

    plan = resolve_model_plan(config, require_immutable_artifact=True)

    def _loader_after_single_baseline(*_args, **_kwargs):
        assert [path.name for path in hashed] == [
            "model.safetensors",
            "config.json",
        ]
        return SimpleNamespace(model=None, tokenizer=None, processor=None)

    with patch.object(
        Qwen3VLLoader,
        "build",
        side_effect=_loader_after_single_baseline,
    ):
        build_model_tokenizer_processor(config, resolved_model_plan=plan)

    assert [path.name for path in hashed] == [
        "model.safetensors",
        "config.json",
        "model.safetensors",
        "config.json",
    ]


def test_local_model_builder_detects_same_stat_content_change_before_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir = tmp_path / "local-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen3_vl"}),
        encoding="utf-8",
    )
    weight_path = model_dir / "model.safetensors"
    weight_path.write_bytes(b"old-weights")
    config = RuntimeConfig()
    config.model.model_type = "qwen3vl"
    config.model.model_name_or_path = str(model_dir)
    frozen_stat = model_artifact_identity._stat_signature(weight_path)
    plan = resolve_model_plan(config, require_immutable_artifact=True)

    weight_path.write_bytes(b"new-weights")
    original_stat = model_artifact_identity._stat_signature
    monkeypatch.setattr(
        model_artifact_identity,
        "_stat_signature",
        lambda path: frozen_stat if path == weight_path else original_stat(path),
    )

    with (
        patch.object(
            Qwen3VLLoader,
            "build",
            return_value=SimpleNamespace(model=None, tokenizer=None, processor=None),
        ),
        pytest.raises(
            ValueError,
            match="Base-model weights or local model code changed",
        ),
    ):
        build_model_tokenizer_processor(config, resolved_model_plan=plan)


def test_local_model_builder_rejects_observable_change_before_loader(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "local-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen3_vl"}),
        encoding="utf-8",
    )
    weight_path = model_dir / "model.safetensors"
    weight_path.write_bytes(b"weights")
    config = RuntimeConfig()
    config.model.model_type = "qwen3vl"
    config.model.model_name_or_path = str(model_dir)
    plan = resolve_model_plan(config, require_immutable_artifact=True)
    weight_path.write_bytes(b"larger-weights")

    with patch.object(
        Qwen3VLLoader,
        "build",
        return_value=SimpleNamespace(model=None, tokenizer=None, processor=None),
    ) as loader:
        with pytest.raises(RuntimeError, match="metadata changed"):
            build_model_tokenizer_processor(config, resolved_model_plan=plan)

    loader.assert_not_called()


def test_local_model_builder_detects_same_stat_content_change_during_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir = tmp_path / "local-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen3_vl"}),
        encoding="utf-8",
    )
    weight_path = model_dir / "model.safetensors"
    weight_path.write_bytes(b"old-weights")
    config = RuntimeConfig()
    config.model.model_type = "qwen3vl"
    config.model.model_name_or_path = str(model_dir)
    frozen_stat = model_artifact_identity._stat_signature(weight_path)
    plan = resolve_model_plan(config, require_immutable_artifact=True)
    original_stat = model_artifact_identity._stat_signature
    monkeypatch.setattr(
        model_artifact_identity,
        "_stat_signature",
        lambda path: frozen_stat if path == weight_path else original_stat(path),
    )

    def _mutating_loader(*_args, **_kwargs):
        weight_path.write_bytes(b"new-weights")
        return SimpleNamespace(model=None, tokenizer=None, processor=None)

    with (
        patch.object(Qwen3VLLoader, "build", side_effect=_mutating_loader),
        pytest.raises(
            ValueError,
            match="Base-model weights or local model code changed",
        ),
    ):
        build_model_tokenizer_processor(config, resolved_model_plan=plan)


def test_local_model_identity_never_reuses_same_stat_sampled_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir = tmp_path / "local-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen3_vl"}),
        encoding="utf-8",
    )
    weight_path = model_dir / "model.safetensors"
    weight_path.write_bytes(b"a" * (1024 * 1024 + 1))
    config = RuntimeConfig()
    config.model.model_type = "qwen3vl"
    config.model.model_name_or_path = str(model_dir)
    frozen_stat = model_artifact_identity._stat_signature(weight_path)

    original = resolve_model_plan(config, require_immutable_artifact=True)
    # This byte is in the one-byte gap left by the former 16-window probe for a
    # 1 MiB + 1 file. Some overlay/shared filesystems also preserve all fields in
    # the old stat signature for an immediate same-size rewrite.
    with weight_path.open("r+b", buffering=0) as stream:
        stream.seek(983_040)
        stream.write(b"B")
    monkeypatch.setattr(
        model_artifact_identity,
        "_stat_signature",
        lambda path: (
            frozen_stat
            if path.name == "model.safetensors"
            else {
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
                "ctime_ns": path.stat().st_ctime_ns,
                "device": path.stat().st_dev,
                "inode": path.stat().st_ino,
            }
        ),
    )

    changed = resolve_model_plan(config, require_immutable_artifact=True)

    assert original.artifact_identity.fingerprint != changed.artifact_identity.fingerprint


def test_invalid_resolved_hub_revision_is_not_checkpointable() -> None:
    config = RuntimeConfig()
    config.model.model_type = "qwen3vl"
    config.model.model_name_or_path = "my-org/Qwen3-VL-domain"
    config.model.revision = "main"

    with patch(
        "shaft.model.descriptor.PretrainedConfig.get_config_dict",
        return_value=(
            {
                "model_type": "qwen3_vl",
                "architectures": ["Qwen3VLForConditionalGeneration"],
            },
            {"_commit_hash": "main"},
        ),
    ):
        plan = resolve_model_plan(config, require_immutable_artifact=True)

    assert plan.artifact_identity.complete is False
    assert plan.artifact_identity.incomplete_reasons == ("invalid_resolved_hub_revision",)


def test_external_remote_code_repo_is_not_checkpointable_without_code_revision() -> None:
    config = RuntimeConfig()
    config.model.model_type = "qwen3vl"
    config.model.model_name_or_path = "my-org/Qwen3-VL-domain"
    config.model.revision = "main"
    config.model.trust_remote_code = True

    with patch(
        "shaft.model.descriptor.PretrainedConfig.get_config_dict",
        return_value=(
            {
                "model_type": "qwen3_vl",
                "architectures": ["Qwen3VLForConditionalGeneration"],
                "auto_map": {
                    "AutoModelForImageTextToText": (
                        "other-org/code-repo--modeling_qwen.CustomModel"
                    )
                },
            },
            {"_commit_hash": "a" * 40},
        ),
    ):
        plan = resolve_model_plan(config, require_immutable_artifact=True)

    assert plan.artifact_identity.complete is False
    assert plan.artifact_identity.incomplete_reasons == (
        "unresolved_external_remote_code_revision:other-org/code-repo",
    )


def test_local_model_external_remote_code_repo_is_not_checkpointable(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "local-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_vl",
                "architectures": ["Qwen3VLForConditionalGeneration"],
                "auto_map": {
                    "AutoModelForImageTextToText": (
                        "other-org/code-repo--modeling_qwen.CustomModel"
                    )
                },
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "model.safetensors").write_bytes(b"weights")
    config = RuntimeConfig()
    config.model.model_type = "qwen3vl"
    config.model.model_name_or_path = str(model_dir)
    config.model.trust_remote_code = True

    plan = resolve_model_plan(config, require_immutable_artifact=True)

    assert plan.artifact_identity.complete is False
    assert plan.artifact_identity.incomplete_reasons == (
        "unresolved_external_remote_code_revision:other-org/code-repo",
    )


def test_late_remote_code_validation_checks_nested_processor_commit() -> None:
    model_commit = "a" * 40
    code_commit = "b" * 40
    dynamic_tokenizer_type = type(
        "DynamicTokenizer",
        (),
        {"__module__": f"transformers_modules.other.repo.{code_commit}.tokenization"},
    )
    processor = type("Processor", (), {})()
    processor.tokenizer = dynamic_tokenizer_type()

    with pytest.raises(ValueError, match="outside the pinned model commit"):
        validate_loaded_remote_code_identity(
            model=object(),
            tokenizer=None,
            processor=processor,
            expected_model_revision=model_commit,
            strict=True,
        )


def test_late_remote_code_validation_accepts_model_commit_namespace() -> None:
    model_commit = "a" * 40
    dynamic_model_type = type(
        "DynamicModel",
        (),
        {"__module__": f"transformers_modules.my_org.repo.{model_commit}.modeling"},
    )

    validate_loaded_remote_code_identity(
        model=dynamic_model_type(),
        tokenizer=None,
        processor=None,
        expected_model_revision=model_commit,
        strict=True,
    )


def test_late_local_remote_code_validation_rejects_hub_module() -> None:
    code_commit = "b" * 40
    dynamic_processor_type = type(
        "DynamicProcessor",
        (),
        {"__module__": f"transformers_modules.other.repo.{code_commit}.processing"},
    )

    with pytest.raises(ValueError, match="not part of the local artifact manifest"):
        validate_loaded_remote_code_identity(
            model=object(),
            tokenizer=None,
            processor=dynamic_processor_type(),
            expected_model_revision=None,
            strict=True,
        )


def test_late_local_remote_code_validation_accepts_local_module_namespace() -> None:
    local_model_type = type(
        "LocalModel",
        (),
        {"__module__": "transformers_modules.local_model.modeling"},
    )

    validate_loaded_remote_code_identity(
        model=local_model_type(),
        tokenizer=None,
        processor=None,
        expected_model_revision=None,
        strict=True,
    )


@pytest.mark.parametrize(
    "weight_map",
    [
        {"": "model-00001-of-00001.safetensors"},
        {" layer.weight ": "model-00001-of-00001.safetensors"},
        {"layer.weight": None},
        {"layer.weight": 1},
        {"layer.weight": ""},
        {"layer.weight": " model-00001-of-00001.safetensors"},
    ],
)
def test_local_weight_index_rejects_invalid_keys_and_values(
    tmp_path: Path,
    weight_map: dict[str, object],
) -> None:
    model_dir = tmp_path / "indexed-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen3_vl"}),
        encoding="utf-8",
    )
    (model_dir / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}),
        encoding="utf-8",
    )
    config = RuntimeConfig()
    config.model.model_type = "qwen3vl"
    config.model.model_name_or_path = str(model_dir)

    with pytest.raises(ValueError, match="HF weight index"):
        resolve_model_plan(config, require_immutable_artifact=True)


@pytest.mark.parametrize(
    "index_json",
    [
        "[]",
        ('{"weight_map":{"layer.weight":"one.safetensors","layer.weight":"two.safetensors"}}'),
    ],
)
def test_local_weight_index_rejects_non_object_and_duplicate_keys(
    tmp_path: Path,
    index_json: str,
) -> None:
    model_dir = tmp_path / "indexed-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen3_vl"}),
        encoding="utf-8",
    )
    (model_dir / "model.safetensors.index.json").write_text(
        index_json,
        encoding="utf-8",
    )
    config = RuntimeConfig()
    config.model.model_type = "qwen3vl"
    config.model.model_name_or_path = str(model_dir)

    with pytest.raises(ValueError, match="weight index"):
        resolve_model_plan(config, require_immutable_artifact=True)


@pytest.mark.parametrize(
    "shard_name",
    [
        "../outside.safetensors",
        "nested/../../outside.safetensors",
        "/tmp/outside.safetensors",
        "..\\outside.safetensors",
        "nested//model.safetensors",
        "C:/outside.safetensors",
    ],
)
def test_local_weight_index_rejects_escaping_shard_paths(
    tmp_path: Path,
    shard_name: str,
) -> None:
    model_dir = tmp_path / "indexed-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen3_vl"}),
        encoding="utf-8",
    )
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer.weight": shard_name}}),
        encoding="utf-8",
    )
    config = RuntimeConfig()
    config.model.model_type = "qwen3vl"
    config.model.model_name_or_path = str(model_dir)

    with pytest.raises(ValueError, match="canonical relative paths"):
        resolve_model_plan(config, require_immutable_artifact=True)


def test_local_weight_index_rejects_symlink_escape(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "indexed-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen3_vl"}),
        encoding="utf-8",
    )
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"weights")
    (model_dir / "model-00001-of-00001.safetensors").symlink_to(outside)
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "layer.weight": "model-00001-of-00001.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    config = RuntimeConfig()
    config.model.model_type = "qwen3vl"
    config.model.model_name_or_path = str(model_dir)

    with pytest.raises(ValueError, match="including through symlinks"):
        resolve_model_plan(config, require_immutable_artifact=True)


def test_local_unindexed_weight_rejects_symlink_escape(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "local-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen3_vl"}),
        encoding="utf-8",
    )
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"weights")
    (model_dir / "model.safetensors").symlink_to(outside)
    config = RuntimeConfig()
    config.model.model_type = "qwen3vl"
    config.model.model_name_or_path = str(model_dir)

    with pytest.raises(ValueError, match="Local HF weight files must stay inside"):
        resolve_model_plan(config, require_immutable_artifact=True)


def test_local_remote_code_rejects_symlink_escape(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "local-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen3_vl"}),
        encoding="utf-8",
    )
    (model_dir / "model.safetensors").write_bytes(b"weights")
    outside = tmp_path / "modeling_external.py"
    outside.write_text("class ExternalModel: pass\n", encoding="utf-8")
    (model_dir / "modeling_local.py").symlink_to(outside)
    config = RuntimeConfig()
    config.model.model_type = "qwen3vl"
    config.model.model_name_or_path = str(model_dir)
    config.model.trust_remote_code = True

    with pytest.raises(ValueError, match="Local HF identity files must stay inside"):
        resolve_model_plan(config, require_immutable_artifact=True)


def test_local_remote_code_rejects_symlinked_package_directory(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "local-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen3_vl"}),
        encoding="utf-8",
    )
    (model_dir / "model.safetensors").write_bytes(b"weights")
    outside_package = tmp_path / "external_package"
    outside_package.mkdir()
    (outside_package / "modeling.py").write_text(
        "class ExternalModel: pass\n",
        encoding="utf-8",
    )
    (model_dir / "package").symlink_to(outside_package, target_is_directory=True)
    config = RuntimeConfig()
    config.model.model_type = "qwen3vl"
    config.model.model_name_or_path = str(model_dir)
    config.model.trust_remote_code = True

    with pytest.raises(ValueError, match="remote-code directory symlinks"):
        resolve_model_plan(config, require_immutable_artifact=True)


@pytest.mark.parametrize(
    ("weight_name", "payload"),
    [
        ("model.safetensors.index.json", b'{"weight_map": {}}'),
        ("model.safetensors", b""),
    ],
)
def test_empty_local_weight_artifact_is_rejected(
    tmp_path: Path,
    weight_name: str,
    payload: bytes,
) -> None:
    model_dir = tmp_path / "empty-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen3_vl"}),
        encoding="utf-8",
    )
    (model_dir / weight_name).write_bytes(payload)
    config = RuntimeConfig()
    config.model.model_type = "qwen3vl"
    config.model.model_name_or_path = str(model_dir)

    with pytest.raises(ValueError, match="empty|zero-byte"):
        resolve_model_plan(config, require_immutable_artifact=True)


def test_local_model_identity_detects_config_change_after_plan(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "local-model"
    model_dir.mkdir()
    config_path = model_dir / "config.json"
    config_path.write_text(
        json.dumps({"model_type": "qwen3_vl"}),
        encoding="utf-8",
    )
    (model_dir / "model.safetensors").write_bytes(b"weights")
    config = RuntimeConfig()
    config.model.model_type = "qwen3vl"
    config.model.model_name_or_path = str(model_dir)
    plan = resolve_model_plan(config, require_immutable_artifact=True)

    config_path.write_text(
        json.dumps({"model_type": "qwen4_vl"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Base-model weights or local model code changed"):
        validate_resolved_model_artifact(plan)


def test_non_checkpoint_local_model_plan_does_not_hash_weights(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir = tmp_path / "local-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen3_vl"}),
        encoding="utf-8",
    )
    (model_dir / "model.safetensors").write_bytes(b"weights")
    config = RuntimeConfig()
    config.model.model_type = "qwen3vl"
    config.model.model_name_or_path = str(model_dir)
    monkeypatch.setattr(
        model_artifact_identity,
        "_file_sha256",
        lambda _path: pytest.fail("non-checkpoint model resolution hashed weights"),
    )

    plan = resolve_model_plan(config)
    validate_resolved_model_artifact(plan)

    assert plan.artifact_identity.complete is False
    assert plan.artifact_identity.incomplete_reasons == ("local_hf_identity_not_materialized",)


def test_local_hf_config_without_weights_is_not_checkpointable(tmp_path: Path) -> None:
    model_dir = tmp_path / "config-only"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen3_vl"}),
        encoding="utf-8",
    )
    config = RuntimeConfig()
    config.model.model_type = "qwen3vl"
    config.model.model_name_or_path = str(model_dir)

    plan = resolve_model_plan(config, require_immutable_artifact=True)

    assert plan.artifact_identity.incomplete_reasons == ("missing_local_hf_weight_files",)
    validate_model_artifact_checkpointability(
        plan,
        save_strategy="no",
        resume_requested=False,
    )
    with pytest.raises(ValueError, match="missing_local_hf_weight_files"):
        validate_model_artifact_checkpointability(
            plan,
            save_strategy="epoch",
            resume_requested=False,
        )


def test_local_hf_weights_without_config_are_not_checkpointable(tmp_path: Path) -> None:
    model_dir = tmp_path / "weights-only"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_bytes(b"weights")
    config = RuntimeConfig()
    config.model.model_type = "qwen3vl"
    config.model.model_name_or_path = str(model_dir)

    plan = resolve_model_plan(config, require_immutable_artifact=True)

    assert plan.artifact_identity.incomplete_reasons == ("missing_local_hf_config",)
    with pytest.raises(ValueError, match="missing_local_hf_config"):
        validate_model_artifact_checkpointability(
            plan,
            save_strategy="steps",
            resume_requested=False,
        )


def test_hub_descriptor_overrides_a_misleading_catalog_basename() -> None:
    config = RuntimeConfig()
    config.model.model_type = "qwen36vl"
    config.model.model_name_or_path = "my-org/Qwen3.6-27B"

    with patch(
        "shaft.model.descriptor.PretrainedConfig.get_config_dict",
        return_value=(
            {
                "model_type": "qwen3_5_moe",
                "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            },
            {},
        ),
    ):
        plan = resolve_model_plan(config)

    assert plan.model_adapter.group_name == "moe"


@pytest.mark.parametrize(
    "repo_id",
    [
        "custom-org/not-actually-qwen",
        "models/not-actually-qwen",
        "outputs/not-actually-qwen",
        "checkpoints/not-actually-qwen",
        "artifacts/not-actually-qwen",
    ],
)
def test_single_variant_custom_hub_checkpoint_still_validates_hf_config(
    repo_id: str,
) -> None:
    config = RuntimeConfig()
    config.model.model_type = "qwen3vl"
    config.model.model_name_or_path = repo_id

    with patch(
        "shaft.model.descriptor.PretrainedConfig.get_config_dict",
        return_value=(
            {
                "model_type": "unrelated_vlm",
                "architectures": ["UnrelatedVisionLanguageModel"],
            },
            {},
        ),
    ) as resolver:
        with pytest.raises(ValueError, match="not a registered variant"):
            resolve_model_plan(config)

    resolver.assert_called_once()


def test_qwen35vl_conflicting_model_type_and_architecture_fail_closed(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "conflicting-qwen"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5",
                "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            }
        ),
        encoding="utf-8",
    )
    config = RuntimeConfig()
    config.model.model_type = "qwen35vl"
    config.model.model_name_or_path = str(model_dir)

    with pytest.raises(ValueError, match="does not match any registered model group"):
        resolve_model_plan(config)


def test_full_init_checkpoint_is_the_model_plan_truth_source(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-moe"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5_moe",
                "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            }
        ),
        encoding="utf-8",
    )
    config = RuntimeConfig()
    config.model.model_type = "qwen36vl"
    config.model.model_name_or_path = "models/Qwen3.6-27B"

    plan = resolve_model_plan(config, init_from_checkpoint=str(checkpoint))

    assert plan.init_kind == "full_checkpoint"
    assert plan.effective_model_name_or_path == str(checkpoint)
    assert plan.model_adapter.model_name_or_path == str(checkpoint)
    assert plan.model_adapter.group_name == "moe"


def test_adapter_init_keeps_base_artifact_as_model_plan_truth_source(
    tmp_path: Path,
) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "models/Qwen3.6-27B"}),
        encoding="utf-8",
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"")
    config = RuntimeConfig()
    config.model.model_type = "qwen36vl"
    config.model.model_name_or_path = "models/Qwen3.6-27B"

    plan = resolve_model_plan(config, init_from_checkpoint=str(adapter))

    assert plan.init_kind == "adapter"
    assert plan.effective_model_name_or_path == "models/Qwen3.6-27B"
    assert plan.model_adapter.group_name == "dense"
    assert plan.adapter_init is not None
    assert plan.adapter_init.base_model_name_or_path == "models/Qwen3.6-27B"


def test_adapter_init_fingerprint_binds_the_adapter_artifact(tmp_path: Path) -> None:
    config = RuntimeConfig()
    config.model.model_type = "qwen36vl"
    config.model.model_name_or_path = "models/Qwen3.6-27B"
    fingerprints = []
    for name in ("adapter-a", "adapter-b"):
        adapter = tmp_path / name
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": "models/Qwen3.6-27B"}),
            encoding="utf-8",
        )
        (adapter / "adapter_model.safetensors").write_bytes(name.encode("utf-8"))
        fingerprints.append(
            resolve_model_plan(config, init_from_checkpoint=str(adapter)).fingerprint
        )

    assert fingerprints[0] != fingerprints[1]


def test_adapter_init_rejects_a_different_declared_base_variant(
    tmp_path: Path,
) -> None:
    dense = tmp_path / "dense"
    dense.mkdir()
    (dense / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5"}),
        encoding="utf-8",
    )
    moe = tmp_path / "moe"
    moe.mkdir()
    (moe / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5_moe",
                "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            }
        ),
        encoding="utf-8",
    )
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": str(moe)}),
        encoding="utf-8",
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"placeholder")
    config = RuntimeConfig()
    config.model.model_type = "qwen36vl"
    config.model.model_name_or_path = str(dense)

    with pytest.raises(ValueError, match="base variant differs"):
        resolve_model_plan(config, init_from_checkpoint=str(adapter))


def test_qwen35vl_unknown_hf_architecture_fails_closed(tmp_path: Path) -> None:
    model_dir = tmp_path / "future-qwen"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen_future_vl",
                "architectures": ["QwenFutureForConditionalGeneration"],
            }
        ),
        encoding="utf-8",
    )
    descriptor = resolve_local_model_descriptor(model_dir)
    assert descriptor is not None

    with pytest.raises(ValueError, match="not a registered variant"):
        build_model_meta("qwen35vl").resolve_adapter(
            model_name_or_path=str(model_dir),
            descriptor=descriptor,
        )


def test_model_requires_check_validates_minimum_versions() -> None:
    model_meta = ModelMeta(
        model_type="dummy",
        family="dummy",
        default_template="smoke_vlm",
        requires=("transformers>=999.0.0",),
    )
    with pytest.raises(ImportError, match="transformers>=999.0.0"):
        model_meta.check_requires()


def test_model_requires_check_validates_required_modules() -> None:
    model_meta = ModelMeta(
        model_type="dummy",
        family="dummy",
        default_template="smoke_vlm",
        requires=("module:package_that_does_not_exist_for_shaft_tests.submodule",),
    )
    with pytest.raises(ImportError, match="package_that_does_not_exist"):
        model_meta.check_requires()


def test_model_meta_can_match_registered_model_name() -> None:
    model_meta = build_model_meta("smoke_vlm")
    assert model_meta.uses_hf_artifacts is False
    matched = model_meta.get_matched_model_group("models/Smoke-VLM")
    assert matched is not None
    assert matched.name == "default"


def test_model_meta_resolves_template_from_matched_group() -> None:
    model_meta = build_model_meta("smoke_vlm")
    assert model_meta.resolve_template_type("models/Smoke-VLM") == "smoke_vlm"


def test_model_meta_check_requires_raises_for_missing_package() -> None:
    model_meta = ModelMeta(
        model_type="dummy",
        family="dummy",
        default_template="smoke_vlm",
        model_groups=(ModelGroup(name="default"),),
        requires=("package_that_does_not_exist_for_shaft_tests>=1.0",),
    )
    with pytest.raises(ImportError):
        model_meta.check_requires()


def test_model_meta_can_resolve_unified_model_adapter() -> None:
    model_meta = build_model_meta("smoke_vlm")
    adapter = model_meta.resolve_adapter(model_name_or_path="models/Smoke-VLM")
    assert adapter.model_type == "smoke_vlm"
    assert adapter.group_name == "default"
    assert adapter.template_type == "smoke_vlm"
    assert adapter.default_target_modules() == ["all-linear"]
    assert adapter.required_saved_files() == ("smoke_tokenizer.json", "smoke_processor.json")


def test_model_group_can_override_template_and_policies() -> None:
    model_meta = ModelMeta(
        model_type="dummy",
        family="dummy",
        default_template="smoke_vlm",
        capabilities=ModelCapabilities(is_multimodal=True),
        processor_policy=build_processor_policy("qwen_vl"),
        peft_policy=build_peft_policy("all_linear"),
        model_groups=(
            ModelGroup(
                name="compact",
                model_ids=("dummy-compact",),
                template="qwen3vl",
                capabilities=ModelCapabilities(is_multimodal=False),
                processor_policy=build_processor_policy("identity"),
                requires=("pkg_a>=1.0",),
                additional_saved_files=("extra.json",),
            ),
        ),
    )
    adapter = model_meta.resolve_adapter(model_name_or_path="dummy-compact")
    assert adapter.template_type == "qwen3vl"
    assert adapter.processor_policy.supports_pixel_budget is False
    assert adapter.capabilities.is_multimodal is False
    assert adapter.requires == ("pkg_a>=1.0",)
    assert adapter.required_saved_files() == ("extra.json",)


def test_qwen3vl_flash_attention_falls_back_without_flash_attn() -> None:
    with patch("shaft.model.qwen3vl.importlib.util.find_spec", return_value=None):
        with pytest.warns(UserWarning, match="flash-attn"):
            resolved = _resolve_attn_implementation("flash_attention_2")
    assert resolved is None


def test_qwen3vl_required_flash_attention_never_silently_falls_back() -> None:
    with patch("shaft.model.qwen3vl.importlib.util.find_spec", return_value=None):
        with pytest.raises(ImportError, match="varlen.*flash-attn"):
            _resolve_attn_implementation("flash_attention_2", required=True)


def test_model_meta_can_resolve_template_meta() -> None:
    model_adapter = build_model_meta("qwen3vl").resolve_adapter(
        model_name_or_path="models/Qwen3-VL-4B-Instruct"
    )
    template_meta = resolve_template_meta(model_adapter=model_adapter)
    assert template_meta.template_type == "qwen3vl"
    assert template_meta.template_cls.__name__ == "Qwen3VLTemplate"
