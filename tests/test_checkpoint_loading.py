from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from vlm_structgen.core.utils.checkpoint import (
    _resolve_dense_target_model,
    load_initial_model_checkpoint,
    load_training_checkpoint,
    save_training_checkpoint,
)


class _DummyScheduler:
    def state_dict(self):
        return getattr(self, "state", {})

    def load_state_dict(self, state):
        self.state = state


class _DummyTokenizer:
    def save_pretrained(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        (output_dir / "tokenizer.json").write_text("{}", encoding="utf-8")


class _DummyProcessor:
    def save_pretrained(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "preprocessor_config.json").write_text("{}", encoding="utf-8")


class _DummyModelConfig:
    def to_json_file(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text(
            json.dumps({"model_type": "dummy"}, ensure_ascii=False),
            encoding="utf-8",
        )


class _DummyBaseModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = _DummyModelConfig()
        self.linear = torch.nn.Linear(2, 2)

    @classmethod
    def from_pretrained(cls, save_directory: str | Path, **kwargs) -> "_DummyBaseModel":
        del kwargs
        save_directory = Path(save_directory)
        model = cls()
        weights_path = save_directory / "weights.json"
        if weights_path.exists():
            payload = json.loads(weights_path.read_text(encoding="utf-8"))
            model.linear.weight.data.copy_(
                torch.tensor(payload["linear.weight"], dtype=model.linear.weight.dtype)
            )
            model.linear.bias.data.copy_(
                torch.tensor(payload["linear.bias"], dtype=model.linear.bias.dtype)
            )
        return model

    def save_pretrained(
        self,
        save_directory: str | Path,
        safe_serialization: bool = True,
        **kwargs,
    ) -> None:
        del safe_serialization, kwargs
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        self.config.to_json_file(save_directory)
        payload = {
            "linear.weight": self.linear.weight.detach().tolist(),
            "linear.bias": self.linear.bias.detach().tolist(),
        }
        (save_directory / "weights.json").write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )
        torch.save(payload, save_directory / "model.safetensors")


class _DummyPeftModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = _DummyModelConfig()
        self.base_model = _DummyBaseModel()
        self.loaded_adapters: list[tuple[str, str, bool]] = []

    def save_pretrained(
        self,
        save_directory: str | Path,
        safe_serialization: bool = True,
        save_embedding_layers: str | bool = "auto",
        **kwargs,
    ) -> None:
        del safe_serialization, save_embedding_layers, kwargs
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        (save_directory / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": "dummy-base"}, ensure_ascii=False),
            encoding="utf-8",
        )
        torch.save({"adapter.weight": torch.ones(1)}, save_directory / "adapter_model.bin")
        (save_directory / "README.md").write_text("# dummy\n", encoding="utf-8")

    def load_adapter(
        self,
        model_id: str | Path,
        adapter_name: str,
        is_trainable: bool = False,
        **kwargs,
    ):
        del kwargs
        self.loaded_adapters.append((str(model_id), adapter_name, is_trainable))
        return None


class _DummyModelWithBaseAttr(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base_model = _DummyBaseModel()
        self.linear = torch.nn.Linear(2, 2)


class CheckpointLoadingTests(unittest.TestCase):
    def test_resume_loads_adapter_optimizer_scheduler_and_rng(self) -> None:
        source_model = _DummyPeftModel()
        optimizer = torch.optim.SGD(source_model.parameters(), lr=0.1)
        scheduler = _DummyScheduler()

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir)
            save_training_checkpoint(
                checkpoint_dir=checkpoint_dir,
                model=source_model,
                tokenizer=_DummyTokenizer(),
                processor=_DummyProcessor(),
                optimizer=optimizer,
                scheduler=scheduler,
                trainer_state={"global_step": 12},
                config_dict={"experiment": {"name": "unit-test"}, "finetune": {"mode": "lora"}},
            )

            target_model = _DummyPeftModel()
            target_model.base_model.linear.weight.data.fill_(0.0)
            target_model.base_model.linear.bias.data.fill_(0.0)
            target_optimizer = torch.optim.SGD(target_model.parameters(), lr=0.1)
            target_scheduler = _DummyScheduler()

            load_calls: list[tuple[str, bool]] = []

            real_torch_load = torch.load

            def recording_load(path, *args, **kwargs):
                load_calls.append((Path(path).name, bool(kwargs.get("weights_only"))))
                return real_torch_load(path, *args, **kwargs)

            with patch("vlm_structgen.core.utils.checkpoint.torch.load", side_effect=recording_load):
                with patch("vlm_structgen.core.utils.checkpoint.set_rng_state") as set_rng_state:
                    trainer_state = load_training_checkpoint(
                        checkpoint_dir=checkpoint_dir,
                        model=target_model,
                        optimizer=target_optimizer,
                        scheduler=target_scheduler,
                        strict=True,
                        resume_training_state=True,
                    )

            self.assertEqual(trainer_state["global_step"], 12)
            self.assertEqual(
                load_calls,
                [
                    ("optimizer.pt", False),
                    ("scheduler.pt", False),
                    ("rng_state.pt", False),
                ],
            )
            set_rng_state.assert_called_once()
            self.assertEqual(len(target_model.loaded_adapters), 1)
            loaded_model_id, loaded_adapter_name, is_trainable = target_model.loaded_adapters[0]
            self.assertEqual(Path(loaded_model_id), checkpoint_dir)
            self.assertEqual(loaded_adapter_name, "default")
            self.assertTrue(is_trainable)
            self.assertFalse((checkpoint_dir / "base_model").exists())

    def test_legacy_layout_is_rejected(self) -> None:
        model = _DummyPeftModel()

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir)
            legacy_model_dir = checkpoint_dir / "model"
            legacy_model_dir.mkdir(parents=True)
            torch.save({"dummy": 1}, legacy_model_dir / "state_dict.pt")

            with self.assertRaises(FileNotFoundError):
                load_training_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    model=model,
                    strict=True,
                    resume_training_state=False,
                )

    def test_full_checkpoint_init_loads_dense_weights(self) -> None:
        source_model = _DummyBaseModel()
        source_model.linear.weight.data.fill_(3.0)
        source_model.linear.bias.data.fill_(2.0)

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir)
            save_training_checkpoint(
                checkpoint_dir=checkpoint_dir,
                model=source_model,
                tokenizer=_DummyTokenizer(),
                processor=_DummyProcessor(),
                optimizer=None,
                scheduler=None,
                trainer_state={"global_step": 7},
                config_dict={"experiment": {"name": "unit-test"}, "finetune": {"mode": "full"}},
            )

            target_model = _DummyBaseModel()
            target_model.linear.weight.data.zero_()
            target_model.linear.bias.data.zero_()

            meta = load_initial_model_checkpoint(
                checkpoint_dir=checkpoint_dir,
                model=target_model,
                strict=True,
            )

            self.assertEqual(meta["checkpoint_layout"], "full_model")
            self.assertTrue(meta["has_base_model"])
            self.assertTrue(torch.allclose(target_model.linear.weight, source_model.linear.weight))
            self.assertTrue(torch.allclose(target_model.linear.bias, source_model.linear.bias))

    def test_dense_target_resolution_prefers_top_level_model_for_non_peft(self) -> None:
        model = _DummyModelWithBaseAttr()
        resolved = _resolve_dense_target_model(model)
        self.assertIs(resolved, model)


if __name__ == "__main__":
    unittest.main()
