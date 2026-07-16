from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Callable

import torch
from peft import (
    PeftModel,
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from safetensors.torch import load as load_safetensors

from shaft.config import RuntimeConfig

from . import qwen35vl as _qwen35vl  # noqa: F401
from . import qwen3vl as _qwen3vl  # noqa: F401
from . import smoke_vlm as _smoke_vlm  # noqa: F401
from .artifact_identity import (
    LocalModelArtifactLoadGuard,
    validate_loaded_remote_code_identity,
)
from .resolution import (
    ResolvedAdapterInit,
    ResolvedModelPlan,
    prepare_resolved_model_artifact_load,
    resolve_model_plan,
    validate_resolved_model_artifact,
)
from .types import (
    LoadedAdapterArtifacts,
    ModelArtifacts,
    ShaftModelAdapter,
    ShaftSequenceExecutionContract,
)


def _validate_hf_sharded_checkpoint_files(path: Path) -> None:
    if not path.is_dir():
        return

    index_path: Path | None = None
    for candidate in (
        path / "model.safetensors.index.json",
        path / "pytorch_model.bin.index.json",
    ):
        if candidate.exists():
            index_path = candidate
            break
    if index_path is None:
        return

    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid HF checkpoint index JSON: {index_path}") from exc

    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"HF checkpoint index has no object weight_map: {index_path}")

    expected_files = sorted(
        {
            str(value).strip()
            for value in weight_map.values()
            if isinstance(value, str) and str(value).strip()
        }
    )
    missing = [name for name in expected_files if not (path / name).is_file()]
    if missing:
        preview = ", ".join(missing[:8])
        suffix = "" if len(missing) <= 8 else f", ... (+{len(missing) - 8} more)"
        temp_dir = path / "._____temp"
        temp_hint = ""
        if temp_dir.exists():
            temp_matches = [name for name in missing if (temp_dir / name).exists()]
            if temp_matches:
                temp_hint = (
                    f" {len(temp_matches)} missing shard(s) are still in {temp_dir}; "
                    "the model download is likely incomplete."
                )
        raise FileNotFoundError(
            f"HF sharded checkpoint is incomplete under {path}: missing {len(missing)} "
            f"file(s): {preview}{suffix}.{temp_hint}"
        )


def _normalize_name_list(value) -> list[str]:
    if isinstance(value, str):
        return [str(value)]
    if isinstance(value, (list, tuple, set)):
        return [str(x) for x in value]
    return []


def _expected_adapter_names_from_artifacts(
    artifacts: ModelArtifacts,
) -> tuple[list[str] | None, list[str] | None]:
    peft_config = getattr(artifacts.model, "peft_config", None)
    if isinstance(peft_config, dict):
        peft_config = peft_config.get("default") or (
            next(iter(peft_config.values())) if peft_config else None
        )
    if peft_config is not None:
        # PEFT canonicalizes a resolved full-module plan into the suffix set it
        # persists in adapter_config.json. Compare that runtime truth here; the
        # exact adapter state-key/shape check below still proves the expansion
        # selects precisely the same concrete modules.
        return (
            _normalize_name_list(getattr(peft_config, "target_modules", None)),
            _normalize_name_list(getattr(peft_config, "modules_to_save", None)),
        )
    finetune_plan = getattr(artifacts, "finetune_plan", None)
    if finetune_plan is None or getattr(finetune_plan, "adapter_plan", None) is None:
        return None, None
    adapter_plan = finetune_plan.adapter_plan
    return (
        list(getattr(adapter_plan, "resolved_target_modules", ()) or ()),
        list(getattr(adapter_plan, "modules_to_save", ()) or ()),
    )


def _resolve_default_peft_config(model: PeftModel):
    peft_config = getattr(model, "peft_config", None)
    if isinstance(peft_config, dict):
        if "default" in peft_config:
            return peft_config["default"]
        if peft_config:
            return next(iter(peft_config.values()))
        return None
    return peft_config


def _validate_adapter_artifact(adapter_init: ResolvedAdapterInit) -> None:
    init_path = Path(adapter_init.path)
    config_path = init_path / "adapter_config.json"
    try:
        current_config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid adapter config JSON: {config_path}") from exc
    if not isinstance(current_config, dict):
        raise TypeError(f"Adapter config must be a JSON object: {config_path}")
    current_config_json = json.dumps(
        current_config,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    if current_config_json != adapter_init.config_json:
        raise ValueError(
            "PEFT adapter config changed after ResolvedModelPlan construction."
        )
    current_weight_manifest = tuple(
        (
            candidate.name,
            int(candidate.stat().st_size),
            _file_sha256(candidate),
        )
        for candidate in (
            init_path / "adapter_model.safetensors",
            init_path / "adapter_model.bin",
        )
        if candidate.is_file()
    )
    if current_weight_manifest != adapter_init.weight_manifest:
        raise ValueError(
            "PEFT adapter weights changed after ResolvedModelPlan construction."
        )


def _load_exact_adapter_state(
    model: PeftModel,
    *,
    adapter_init: ResolvedAdapterInit,
) -> None:
    # Revalidate after the potentially long base-model build, then deserialize
    # the same verified byte snapshot. A path-based loader would reopen the file
    # after validation and leave a TOCTOU window.
    _validate_adapter_artifact(adapter_init)
    peft_state = _load_verified_adapter_weights(adapter_init)
    expected_state = get_peft_model_state_dict(
        model,
        adapter_name="default",
    )
    missing_keys = sorted(set(expected_state).difference(peft_state))
    unexpected_keys = sorted(set(peft_state).difference(expected_state))
    shape_mismatches = sorted(
        key
        for key in set(expected_state).intersection(peft_state)
        if tuple(expected_state[key].shape) != tuple(peft_state[key].shape)
    )
    if missing_keys or unexpected_keys or shape_mismatches:
        raise ValueError(
            "PEFT adapter state does not exactly match the resolved finetune plan: "
            f"missing={missing_keys[:8]}, unexpected={unexpected_keys[:8]}, "
            f"shape_mismatches={shape_mismatches[:8]}."
        )
    load_result = set_peft_model_state_dict(
        model,
        peft_state,
        adapter_name="default",
    )
    unexpected_after_load = tuple(
        getattr(load_result, "unexpected_keys", ()) or ()
    )
    if unexpected_after_load:
        raise ValueError(
            "PEFT adapter loader left unexpected state keys: "
            f"{unexpected_after_load[:8]}."
        )


def _load_verified_adapter_weights(
    adapter_init: ResolvedAdapterInit,
) -> dict[str, torch.Tensor]:
    directory = Path(adapter_init.path)
    weight_path = next(
        (
            candidate
            for candidate in (
                directory / "adapter_model.safetensors",
                directory / "adapter_model.bin",
            )
            if candidate.is_file()
        ),
        None,
    )
    if weight_path is None:
        raise FileNotFoundError(f"PEFT adapter weights are missing: {directory}.")
    expected_by_name = {
        name: (size, sha256)
        for name, size, sha256 in adapter_init.weight_manifest
    }
    expected = expected_by_name.get(weight_path.name)
    if expected is None:
        raise ValueError(
            "PEFT adapter weight file differs from ResolvedModelPlan manifest."
        )
    payload = weight_path.read_bytes()
    actual = (len(payload), hashlib.sha256(payload).hexdigest())
    if actual != expected:
        raise ValueError(
            "PEFT adapter weights changed after ResolvedModelPlan construction."
        )
    if weight_path.suffix == ".safetensors":
        state = load_safetensors(payload)
    else:
        state = torch.load(
            io.BytesIO(payload),
            map_location=torch.device("cpu"),
            weights_only=True,
        )
    if not isinstance(state, dict):
        raise TypeError("PEFT adapter weights must deserialize to a state dictionary.")
    return state


def _validate_adapter_compatibility(
    config: RuntimeConfig,
    adapter_config: dict[str, object],
    path: Path,
    *,
    expected_target_modules: list[str] | None = None,
    expected_modules_to_save: list[str] | None = None,
) -> None:
    mode = str(config.model.finetune.mode).strip().lower()
    if mode == "full":
        raise ValueError(f"init_from={path} is a PEFT adapter checkpoint, but finetune.mode is 'full'.")
    if mode not in {"lora", "dora", "qlora"}:
        raise ValueError(f"Unsupported finetune mode for adapter init: {mode!r}")

    use_dora = bool(adapter_config.get("use_dora", False))
    if mode == "dora" and not use_dora:
        raise ValueError(f"Adapter at {path} is LoRA, but finetune.mode='dora'.")
    if mode != "dora" and use_dora:
        raise ValueError(f"Adapter at {path} is DoRA, but finetune.mode={mode!r}.")

    expected_r = int(config.model.finetune.lora_r)
    adapter_r = int(adapter_config.get("r", expected_r))
    if adapter_r != expected_r:
        raise ValueError(f"LoRA rank mismatch: adapter={adapter_r}, config={expected_r}.")

    expected_alpha = int(config.model.finetune.lora_alpha)
    adapter_alpha = int(adapter_config.get("lora_alpha", expected_alpha))
    if adapter_alpha != expected_alpha:
        raise ValueError(f"LoRA alpha mismatch: adapter={adapter_alpha}, config={expected_alpha}.")

    expected_bias = str(config.model.finetune.lora_bias).strip().lower()
    adapter_bias = str(adapter_config.get("bias", expected_bias)).strip().lower()
    if adapter_bias != expected_bias:
        raise ValueError(f"LoRA bias mismatch: adapter={adapter_bias!r}, config={expected_bias!r}.")

    adapter_target_modules = _normalize_name_list(adapter_config.get("target_modules"))
    if expected_target_modules is not None and adapter_target_modules:
        if sorted(expected_target_modules) != sorted(adapter_target_modules):
            raise ValueError("LoRA target_modules mismatch between adapter and current config.")

    if expected_modules_to_save is not None:
        adapter_modules_to_save = _normalize_name_list(adapter_config.get("modules_to_save"))
        if sorted(expected_modules_to_save) != sorted(adapter_modules_to_save):
            raise ValueError("LoRA modules_to_save mismatch between adapter and current config.")

    expected_rslora = bool(config.model.finetune.use_rslora)
    adapter_rslora = bool(adapter_config.get("use_rslora", expected_rslora))
    if adapter_rslora != expected_rslora:
        raise ValueError(
            f"LoRA use_rslora mismatch: adapter={adapter_rslora}, config={expected_rslora}."
        )


@dataclass(frozen=True)
class ShaftPreparedModelBuild:
    """Pure-local preparation for one model loader invocation.

    The preparation deliberately contains every rank-local check which can run
    before a loader that may own distributed collectives.  The corresponding
    finalize function closes the immutable-artifact and remote-code identity
    after the loader returns.
    """

    config: RuntimeConfig
    model_plan: ResolvedModelPlan
    sequence_execution_contract: ShaftSequenceExecutionContract | None
    artifact_load_guard: LocalModelArtifactLoadGuard | None
    adapter_config: dict[str, object] | None = None


ModelBuildLocalPhaseRunner = Callable[[str, Callable[[], Any]], Any]


def prepare_model_build(
    config: RuntimeConfig,
    *,
    init_from_checkpoint: str | None = None,
    sequence_execution_contract: ShaftSequenceExecutionContract | None = None,
    resolved_model_plan: ResolvedModelPlan | None = None,
) -> ShaftPreparedModelBuild:
    """Resolve and validate all local state required before the raw loader."""

    model_plan = resolved_model_plan or resolve_model_plan(
        config,
        init_from_checkpoint=init_from_checkpoint,
    )
    if init_from_checkpoint != model_plan.init_from_checkpoint:
        raise ValueError(
            "init_from_checkpoint differs from the supplied ResolvedModelPlan."
        )
    return _prepare_resolved_model_build(
        config,
        model_plan=model_plan,
        sequence_execution_contract=sequence_execution_contract,
        apply_adapter_init=True,
    )


def _prepare_resolved_model_build(
    config: RuntimeConfig,
    *,
    model_plan: ResolvedModelPlan,
    sequence_execution_contract: ShaftSequenceExecutionContract | None,
    apply_adapter_init: bool,
) -> ShaftPreparedModelBuild:
    runtime_config = copy.deepcopy(config)
    runtime_config.model.model_name_or_path = model_plan.effective_model_name_or_path
    runtime_config.model.revision = model_plan.resolved_revision
    model_path = Path(runtime_config.model.model_name_or_path)
    _validate_hf_sharded_checkpoint_files(model_path)
    artifact_load_guard = prepare_resolved_model_artifact_load(model_plan)
    model_meta = model_plan.model_meta
    model_adapter = model_plan.model_adapter
    model_adapter.check_requires()
    assert model_meta.loader is not None

    adapter_config: dict[str, object] | None = None
    if apply_adapter_init and model_plan.init_kind == "adapter":
        assert model_plan.adapter_init is not None
        init_path = Path(model_plan.adapter_init.path)
        adapter_config = model_plan.adapter_init.config_dict()
        _validate_adapter_artifact(model_plan.adapter_init)
        _validate_adapter_compatibility(
            config,
            adapter_config,
            init_path,
        )

    return ShaftPreparedModelBuild(
        config=runtime_config,
        model_plan=model_plan,
        sequence_execution_contract=sequence_execution_contract,
        artifact_load_guard=artifact_load_guard,
        adapter_config=adapter_config,
    )


def invoke_model_loader(
    prepared: ShaftPreparedModelBuild,
) -> ModelArtifacts:
    """Invoke only the loader API which is allowed to own collectives."""

    model_meta = prepared.model_plan.model_meta
    model_adapter = prepared.model_plan.model_adapter
    assert model_meta.loader is not None
    return model_meta.loader.build(
        prepared.config,
        model_meta=model_meta,
        model_adapter=model_adapter,
        sequence_execution_contract=prepared.sequence_execution_contract,
    )


def finalize_model_build(
    prepared: ShaftPreparedModelBuild,
    artifacts: ModelArtifacts,
) -> ModelArtifacts:
    """Run pure-local post-loader identity and adapter validation."""

    model_plan = prepared.model_plan
    validate_resolved_model_artifact(
        model_plan,
        load_guard=prepared.artifact_load_guard,
    )
    if (
        model_plan.require_immutable_artifact
        and model_plan.trust_remote_code
        and model_plan.artifact_identity.kind in {"hf_hub", "local_hf"}
    ):
        validate_loaded_remote_code_identity(
            model=artifacts.model,
            tokenizer=artifacts.tokenizer,
            processor=artifacts.processor,
            expected_model_revision=(
                model_plan.resolved_revision
                if model_plan.artifact_identity.kind == "hf_hub"
                else None
            ),
            strict=True,
        )

    if prepared.adapter_config is not None:
        assert model_plan.adapter_init is not None
        assert prepared.adapter_config is not None
        init_path = Path(model_plan.adapter_init.path)
        if not isinstance(artifacts.model, PeftModel):
            raise TypeError(
                "Adapter init requires a PEFT model, but current mode did not create one."
            )
        expected_target_modules, expected_modules_to_save = (
            _expected_adapter_names_from_artifacts(artifacts)
        )
        if expected_target_modules is None:
            peft_config = _resolve_default_peft_config(artifacts.model)
            expected_target_modules = _normalize_name_list(
                getattr(
                    peft_config,
                    "target_modules",
                    prepared.config.model.finetune.target_modules,
                )
            )
            expected_modules_to_save = _normalize_name_list(
                getattr(peft_config, "modules_to_save", None)
            )
        _validate_adapter_compatibility(
            prepared.config,
            prepared.adapter_config,
            init_path,
            expected_target_modules=expected_target_modules,
            expected_modules_to_save=expected_modules_to_save,
        )
        _load_exact_adapter_state(
            artifacts.model,
            adapter_init=model_plan.adapter_init,
        )
    return artifacts


def resolve_model_adapter_from_config(
    config: RuntimeConfig,
    *,
    model_meta=None,
) -> ShaftModelAdapter:
    plan = resolve_model_plan(config)
    if model_meta is not None and plan.model_meta is not model_meta:
        raise ValueError("Explicit model_meta differs from the resolved model plan.")
    return plan.model_adapter


def build_model_tokenizer_processor(
    config: RuntimeConfig,
    *,
    init_from_checkpoint: str | None = None,
    sequence_execution_contract: ShaftSequenceExecutionContract | None = None,
    resolved_model_plan: ResolvedModelPlan | None = None,
    local_phase_runner: ModelBuildLocalPhaseRunner | None = None,
) -> ModelArtifacts:
    """Build artifacts through prepare -> raw loader -> finalize.

    Ordinary callers retain the original one-call API.  Distributed training
    pipelines supply ``local_phase_runner`` so only the local prepare/finalize
    phases are enclosed by rank-status convergence; the raw loader invocation
    is intentionally never nested in that envelope.
    """

    def prepare() -> ShaftPreparedModelBuild:
        return prepare_model_build(
            config,
            init_from_checkpoint=init_from_checkpoint,
            sequence_execution_contract=sequence_execution_contract,
            resolved_model_plan=resolved_model_plan,
        )

    prepared = (
        prepare()
        if local_phase_runner is None
        else local_phase_runner("prepare", prepare)
    )
    artifacts = invoke_model_loader(prepared)

    def finalize() -> ModelArtifacts:
        return finalize_model_build(prepared, artifacts)

    finalized = (
        finalize()
        if local_phase_runner is None
        else local_phase_runner("finalize", finalize)
    )
    return finalized


def load_adapter_artifacts(
    config: RuntimeConfig,
    *,
    adapter_path: str,
    resolved_model_plan: ResolvedModelPlan | None = None,
) -> LoadedAdapterArtifacts:
    """Load an adapter exactly for merge/export without inventing training config.

    The persisted PEFT config is authoritative for the adapter topology, while the
    resolved model plan remains authoritative for base artifact and model variant.
    The state is loaded only after exact key/shape validation.
    """

    model_plan = resolved_model_plan or resolve_model_plan(
        config,
        init_from_checkpoint=adapter_path,
    )
    if model_plan.init_from_checkpoint != adapter_path:
        raise ValueError("adapter_path differs from the supplied ResolvedModelPlan.")
    if model_plan.init_kind != "adapter" or model_plan.adapter_init is None:
        raise ValueError(f"Expected a PEFT adapter checkpoint: {adapter_path}.")

    adapter_init = model_plan.adapter_init
    _validate_adapter_artifact(adapter_init)
    adapter_config = get_peft_config(adapter_init.config_dict())
    peft_type = getattr(adapter_config, "peft_type", None)
    peft_type_value = getattr(peft_type, "value", peft_type)
    if str(peft_type_value).strip().lower() != "lora":
        raise ValueError(
            "Shaft merge currently supports LoRA-family adapters only; "
            f"received peft_type={peft_type_value!r}."
        )

    base_config = copy.deepcopy(config)
    base_config.model.finetune.mode = "full"
    prepared = _prepare_resolved_model_build(
        base_config,
        model_plan=model_plan,
        sequence_execution_contract=None,
        apply_adapter_init=False,
    )
    artifacts = finalize_model_build(
        prepared,
        invoke_model_loader(prepared),
    )
    adapter_model = get_peft_model(artifacts.model, adapter_config)
    if not isinstance(adapter_model, PeftModel):
        raise TypeError("Resolved adapter config did not produce a PEFT model.")
    _load_exact_adapter_state(adapter_model, adapter_init=adapter_init)
    return LoadedAdapterArtifacts(
        model=adapter_model,
        tokenizer=artifacts.tokenizer,
        processor=artifacts.processor,
        model_adapter=artifacts.model_adapter,
    )


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
