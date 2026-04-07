from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from vlm_structgen.core.utils.io import ensure_dir, write_json


@dataclass(frozen=True)
class AdapterBundleSpec:
    route: str
    checkpoint_dir: Path
    bundle_dir_name: str | None = None


@dataclass
class DeploymentBundleResult:
    output_dir: Path
    base_model_dir: Path
    adapters_manifest_path: Path
    adapter_dirs: dict[str, Path]


_TOKENIZER_AND_PROCESSOR_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.json",
    "preprocessor_config.json",
    "processor_config.json",
    "image_processor_config.json",
    "generation_config.json",
)


def _sanitize_bundle_name(name: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z._-]+", "_", name.strip())
    return sanitized.strip("_") or "adapter"


def _copy_file_if_exists(source: Path, target: Path) -> bool:
    if not source.exists():
        return False
    ensure_dir(target.parent)
    shutil.copy2(source, target)
    return True


def _copy_dir_contents(source_dir: Path, target_dir: Path) -> None:
    ensure_dir(target_dir)
    for entry in source_dir.iterdir():
        destination = target_dir / entry.name
        if entry.is_dir():
            shutil.copytree(entry, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(entry, destination)


def _resolve_base_model_source(source_dir: Path) -> tuple[Path, Path]:
    if (source_dir / "config.json").exists():
        return source_dir, source_dir
    raise FileNotFoundError(f"Missing base model config: {source_dir / 'config.json'}")


def _resolve_adapter_source(checkpoint_dir: Path) -> Path:
    root_adapter_config = checkpoint_dir / "adapter_config.json"
    root_adapter_weights = checkpoint_dir / "adapter_model.safetensors"
    root_adapter_weights_bin = checkpoint_dir / "adapter_model.bin"
    if root_adapter_config.exists() and (root_adapter_weights.exists() or root_adapter_weights_bin.exists()):
        return checkpoint_dir
    raise FileNotFoundError(
        f"Missing PEFT adapter files in checkpoint: {checkpoint_dir}"
    )


def _copy_base_model_bundle(source_dir: Path, target_dir: Path) -> None:
    base_model_source, aux_source = _resolve_base_model_source(source_dir)
    ensure_dir(target_dir)

    _copy_dir_contents(base_model_source, target_dir)
    if aux_source is source_dir and base_model_source is source_dir:
        return

    for filename in _TOKENIZER_AND_PROCESSOR_FILES:
        _copy_file_if_exists(aux_source / filename, target_dir / filename)


def _copy_adapter_bundle(source_dir: Path, target_dir: Path) -> None:
    adapter_source = _resolve_adapter_source(source_dir)
    ensure_dir(target_dir)
    for filename in ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin", "README.md"):
        _copy_file_if_exists(adapter_source / filename, target_dir / filename)


def export_deployment_bundle(
    *,
    base_source_dir: str | Path,
    adapter_specs: list[AdapterBundleSpec],
    output_dir: str | Path,
    overwrite: bool = False,
) -> DeploymentBundleResult:
    output_dir = Path(output_dir)
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. Pass overwrite=True to replace it."
            )
        shutil.rmtree(output_dir)
    ensure_dir(output_dir)

    base_model_dir = ensure_dir(output_dir / "base_model")
    adapters_root = ensure_dir(output_dir / "adapters")
    manifests_root = ensure_dir(output_dir / "manifests")

    _copy_base_model_bundle(Path(base_source_dir), base_model_dir)

    adapter_dirs: dict[str, Path] = {}
    manifest_adapters: dict[str, dict[str, str]] = {}
    for spec in adapter_specs:
        bundle_dir_name = spec.bundle_dir_name or _sanitize_bundle_name(spec.route)
        adapter_output_dir = ensure_dir(adapters_root / bundle_dir_name)
        _copy_adapter_bundle(Path(spec.checkpoint_dir), adapter_output_dir)
        adapter_dirs[spec.route] = adapter_output_dir
        manifest_adapters[spec.route] = {
            "path": f"adapters/{bundle_dir_name}",
            "source_checkpoint": str(Path(spec.checkpoint_dir)),
        }

    manifest_path = manifests_root / "adapters.json"
    write_json(
        manifest_path,
        {
            "format": "peft_base_model_plus_adapters",
            "base_model": {
                "path": "base_model",
                "source": str(Path(base_source_dir)),
            },
            "adapters": manifest_adapters,
        },
    )

    return DeploymentBundleResult(
        output_dir=output_dir,
        base_model_dir=base_model_dir,
        adapters_manifest_path=manifest_path,
        adapter_dirs=adapter_dirs,
    )
