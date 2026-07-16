from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import shutil

from peft import PeftConfig

from shaft.config import RuntimeConfig
from shaft.model import (
    build_model_meta,
    load_adapter_artifacts,
    resolve_model_plan,
)
from shaft.training.batch_planning import (
    load_batching_run_metadata,
    load_checkpoint_batching_metadata,
)
from shaft.training.checkpointing import (
    CheckpointLayout,
    ensure_hf_export_layout,
    inspect_checkpoint_layout,
)


@dataclass(frozen=True)
class ExportMergeResult:
    output_dir: Path
    base_model_path: str
    adapter_path: Path
    layout: CheckpointLayout


def inspect_hf_artifact(path: str | Path) -> CheckpointLayout:
    return inspect_checkpoint_layout(path)


def infer_base_model_from_adapter(adapter_path: str | Path) -> str | None:
    config = PeftConfig.from_pretrained(str(adapter_path))
    value = getattr(config, "base_model_name_or_path", None)
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _build_export_runtime_config(
    *,
    model_type: str,
    model_name_or_path: str,
    template: str | None,
    trust_remote_code: bool,
    torch_dtype: str,
    revision: str | None = None,
    cache_dir: str | None = None,
    local_files_only: bool = False,
) -> RuntimeConfig:
    config = RuntimeConfig()
    config.model.model_type = str(model_type).strip().lower()
    config.model.model_name_or_path = str(model_name_or_path)
    config.model.template = template
    config.model.trust_remote_code = bool(trust_remote_code)
    config.model.torch_dtype = str(torch_dtype)
    config.model.revision = revision
    config.model.cache_dir = cache_dir
    config.model.local_files_only = bool(local_files_only)
    config.model.finetune.mode = "full"
    return config


def _resolve_model_export_meta(
    *,
    model_type: str | None,
    model_name_or_path: str | None,
    template: str | None,
    revision: str | None = None,
    cache_dir: str | None = None,
    local_files_only: bool = False,
):
    if model_type is None:
        return None
    model_meta = build_model_meta(str(model_type).strip().lower())
    if model_name_or_path is None:
        return model_meta
    config = _build_export_runtime_config(
        model_type=str(model_type),
        model_name_or_path=str(model_name_or_path),
        template=template,
        trust_remote_code=True,
        torch_dtype="bfloat16",
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    return resolve_model_plan(config).model_adapter


def validate_hf_artifact(
    path: str | Path,
    *,
    finetune_mode: str,
    model_type: str | None = None,
    model_name_or_path: str | None = None,
    template: str | None = None,
    revision: str | None = None,
    cache_dir: str | None = None,
    local_files_only: bool = False,
) -> CheckpointLayout:
    model_meta = _resolve_model_export_meta(
        model_type=model_type,
        model_name_or_path=model_name_or_path,
        template=template,
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    ensure_hf_export_layout(path, finetune_mode=finetune_mode, model_meta=model_meta)
    return inspect_hf_artifact(path)


def _save_processing_assets(*, output_dir: Path, processor, tokenizer) -> None:
    if processor is not None and hasattr(processor, "save_pretrained"):
        processor.save_pretrained(output_dir)
        return
    if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(output_dir)


_HF_PROCESSING_ASSET_NAMES = {
    "added_tokens.json",
    "chat_template.jinja",
    "chat_template.json",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "smoke_tokenizer.json",
    "smoke_processor.json",
}


def _overlay_adapter_processing_assets(
    *,
    adapter_dir: Path,
    output_dir: Path,
    additional_saved_files: tuple[str, ...] = (),
) -> None:
    candidate_names = set(_HF_PROCESSING_ASSET_NAMES)
    candidate_names.update(str(name).strip() for name in additional_saved_files if str(name).strip())
    for name in sorted(candidate_names):
        source = adapter_dir / name
        if not source.is_file():
            continue
        target = output_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _validate_adapter_base_provenance(
    *,
    adapter_dir: Path,
    base_model_plan,
    allow_unverified_base_model: bool,
) -> None:
    if bool(allow_unverified_base_model):
        return
    errors: list[Exception] = []
    loaded_metadata = False
    loaders = [
        (load_checkpoint_batching_metadata, adapter_dir),
        (load_batching_run_metadata, adapter_dir),
    ]
    if adapter_dir.name == "best" or adapter_dir.name.startswith("checkpoint-"):
        loaders.append((load_batching_run_metadata, adapter_dir.parent))
    for loader, metadata_root in loaders:
        try:
            metadata = loader(metadata_root)
        except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
            errors.append(exc)
            continue
        loaded_metadata = True
        train_input_contract = getattr(metadata, "train_input_contract", None)
        if train_input_contract is None:
            errors.append(
                ValueError(
                    f"Metadata at {metadata_root} has no training input contract."
                )
            )
            continue
        actual = str(train_input_contract.model_plan_fingerprint)
        expected = str(base_model_plan.fingerprint)
        if actual != expected:
            raise ValueError(
                "Adapter checkpoint base-model identity differs from the requested merge "
                f"base: checkpoint={actual!r}, requested={expected!r}. Pass "
                "--allow-unverified-base-model true only after independently verifying "
                "that this mismatch is intentional."
            )
        return
    if not loaded_metadata:
        raise ValueError(
            "Adapter merge cannot prove which base-model bytes produced this adapter: "
            "valid Shaft checkpoint metadata is missing. Pass "
            "--allow-unverified-base-model true only after independently verifying "
            "the base model."
        ) from (errors[-1] if errors else None)
    raise ValueError(
        "Adapter merge found checkpoint metadata, but none of the available scopes "
        "contains a training input contract. Pass --allow-unverified-base-model true "
        "only after independently verifying the base model."
    ) from (errors[-1] if errors else None)


def merge_peft_adapter(
    *,
    model_type: str,
    adapter_path: str | Path,
    output_dir: str | Path,
    base_model_path: str | None = None,
    template: str | None = None,
    trust_remote_code: bool = True,
    torch_dtype: str = "bfloat16",
    safe_serialization: bool = True,
    max_shard_size: str = "5GB",
    revision: str | None = None,
    cache_dir: str | None = None,
    local_files_only: bool = False,
    allow_unverified_base_model: bool = False,
) -> ExportMergeResult:
    adapter_dir = Path(adapter_path)
    ensure_hf_export_layout(adapter_dir, finetune_mode="lora")
    resolved_base_model = base_model_path or infer_base_model_from_adapter(adapter_dir)
    if resolved_base_model is None:
        raise ValueError(
            "Unable to infer base model from adapter_config.json. Please provide --base-model."
        )

    target_dir = Path(output_dir)
    if target_dir.exists():
        if not target_dir.is_dir():
            raise ValueError(f"Output path exists and is not a directory: {target_dir}")
        if any(target_dir.iterdir()):
            raise ValueError(f"Output directory must be empty: {target_dir}")

    config = _build_export_runtime_config(
        model_type=model_type,
        model_name_or_path=resolved_base_model,
        template=template,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    model_plan = resolve_model_plan(
        config,
        init_from_checkpoint=str(adapter_dir),
        require_immutable_artifact=True,
    )
    if not model_plan.artifact_identity.complete:
        raise ValueError(
            "Adapter merge requires an immutable base-model artifact identity: "
            f"{list(model_plan.artifact_identity.incomplete_reasons)}."
        )
    base_model_plan = replace(
        model_plan,
        init_from_checkpoint=None,
        init_kind="base",
        adapter_init=None,
    )
    _validate_adapter_base_provenance(
        adapter_dir=adapter_dir,
        base_model_plan=base_model_plan,
        allow_unverified_base_model=allow_unverified_base_model,
    )
    artifacts = load_adapter_artifacts(
        config,
        adapter_path=str(adapter_dir),
        resolved_model_plan=model_plan,
    )
    merged_model = artifacts.model.merge_and_unload()
    target_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(
        target_dir,
        safe_serialization=bool(safe_serialization),
        max_shard_size=max_shard_size,
    )
    _save_processing_assets(
        output_dir=target_dir,
        processor=artifacts.processor,
        tokenizer=artifacts.tokenizer,
    )
    _overlay_adapter_processing_assets(
        adapter_dir=adapter_dir,
        output_dir=target_dir,
        additional_saved_files=artifacts.model_adapter.required_saved_files(),
    )
    ensure_hf_export_layout(
        target_dir,
        finetune_mode="full",
        model_meta=artifacts.model_adapter,
    )
    return ExportMergeResult(
        output_dir=target_dir,
        base_model_path=str(resolved_base_model),
        adapter_path=adapter_dir,
        layout=inspect_hf_artifact(target_dir),
    )
