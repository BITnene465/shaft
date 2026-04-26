from __future__ import annotations

from dataclasses import dataclass
import shutil
from pathlib import Path

from transformers.trainer_utils import get_last_checkpoint

from shaft.config import RuntimeConfig
from shaft.model import ModelMeta, ShaftModelAdapter


@dataclass(frozen=True)
class CheckpointLayout:
    path: Path
    kind: str
    has_trainer_state: bool


def inspect_checkpoint_layout(path: str | Path) -> CheckpointLayout:
    target = Path(path)
    has_trainer_state = (target / "trainer_state.json").exists()
    has_adapter = (target / "adapter_config.json").exists() and (
        (target / "adapter_model.safetensors").exists() or (target / "adapter_model.bin").exists()
    )
    has_full_model = (target / "config.json").exists() and (
        (target / "model.safetensors").exists()
        or (target / "model.safetensors.index.json").exists()
        or (target / "pytorch_model.bin").exists()
        or (target / "pytorch_model.bin.index.json").exists()
    )
    if has_adapter:
        kind = "adapter"
    elif has_full_model:
        kind = "full"
    elif has_trainer_state:
        kind = "trainer_state_only"
    else:
        kind = "unknown"
    return CheckpointLayout(path=target, kind=kind, has_trainer_state=has_trainer_state)


def ensure_hf_export_layout(
    path: str | Path,
    *,
    finetune_mode: str,
    model_meta: ModelMeta | ShaftModelAdapter | None = None,
) -> None:
    layout = inspect_checkpoint_layout(path)
    mode = str(finetune_mode).strip().lower()
    if mode == "full":
        if layout.kind != "full":
            raise ValueError(f"Expected a full HF export at {path}, found {layout.kind!r}.")
        if model_meta is not None:
            missing = [name for name in model_meta.required_saved_files() if not (Path(path) / name).exists()]
            if missing:
                raise ValueError(f"Missing additional saved files in export {path}: {missing}")
        return
    if mode in {"lora", "dora", "qlora"}:
        if layout.kind != "adapter":
            raise ValueError(f"Expected a PEFT adapter export at {path}, found {layout.kind!r}.")
        return
    raise ValueError(f"Unsupported finetune mode: {finetune_mode!r}.")


def resolve_best_export_dir(output_dir: str | Path) -> Path:
    return Path(output_dir) / "best"


def resolve_resume_checkpoint(path: str | Path | None) -> str | None:
    if path is None:
        return None
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"resume_from checkpoint path not found: {target}")
    if (target / "trainer_state.json").exists():
        return str(target)
    last_checkpoint = get_last_checkpoint(str(target))
    if last_checkpoint is not None:
        return str(last_checkpoint)
    raise ValueError(f"No trainer checkpoint found under: {target}")


def prune_root_output_layout(output_dir: str | Path) -> None:
    root = Path(output_dir)
    if not root.is_dir():
        return

    has_checkpoint_dir = any(
        item.is_dir() and item.name.startswith("checkpoint-") for item in root.iterdir()
    )
    if not has_checkpoint_dir and not (root / "best").exists():
        return

    layout = inspect_checkpoint_layout(root)
    if layout.kind == "unknown":
        return

    if not any(
        item.is_dir() and (item.name == "best" or item.name.startswith("checkpoint-"))
        for item in root.iterdir()
    ):
        return

    for item in root.iterdir():
        if item.name.startswith("."):
            continue
        if item.is_dir() and (item.name == "best" or item.name.startswith("checkpoint-")):
            continue
        if item.is_dir():
            shutil.rmtree(item)
        elif item.is_file():
            item.unlink(missing_ok=True)


def validate_resume_checkpoint(path: str | Path, *, finetune_mode: str) -> None:
    layout = inspect_checkpoint_layout(path)
    mode = str(finetune_mode).strip().lower()
    if not layout.has_trainer_state:
        raise ValueError(f"resume_from requires trainer_state.json in checkpoint: {path}")
    if mode == "full":
        if layout.kind != "full":
            raise ValueError(f"Expected full-model checkpoint for resume under mode='full', found {layout.kind!r}.")
        return
    if mode in {"lora", "dora", "qlora"}:
        if layout.kind != "adapter":
            raise ValueError(f"Expected adapter checkpoint for resume under mode={mode!r}, found {layout.kind!r}.")
        return
    raise ValueError(f"Unsupported finetune mode: {finetune_mode!r}")


def validate_training_state_policy(config: RuntimeConfig) -> None:
    train_cfg = config.train
    eval_cfg = config.eval
    if not train_cfg.load_best_model_at_end:
        return
    if not eval_cfg.enabled:
        raise ValueError("train.load_best_model_at_end=true requires eval.enabled=true.")
    if train_cfg.save_strategy == "no":
        raise ValueError("load_best_model_at_end requires train.save_strategy != 'no'.")
    if eval_cfg.eval_strategy == "no":
        raise ValueError("load_best_model_at_end requires eval.eval_strategy != 'no'.")
    if train_cfg.save_strategy != eval_cfg.eval_strategy:
        raise ValueError("save_strategy and eval_strategy must match when load_best_model_at_end=true.")
    if train_cfg.save_strategy == "steps" and int(train_cfg.save_steps) % int(eval_cfg.eval_steps) != 0:
        raise ValueError("When using step-based best model loading, save_steps must be a multiple of eval_steps.")
