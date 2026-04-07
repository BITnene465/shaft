from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from vlm_structgen.core.utils.checkpoint import (
    load_initial_model_checkpoint,
    load_training_checkpoint,
    save_training_checkpoint,
)


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
        torch.save({"linear.weight": self.linear.weight.detach()}, save_directory / "model.safetensors")


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

    def get_base_model(self) -> "_DummyPeftModel":
        return self.base_model


class PeftCheckpointLayoutTests(unittest.TestCase):
    def test_save_training_checkpoint_writes_adapter_layout_for_lora(self) -> None:
        model = _DummyPeftModel()

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir)
            save_training_checkpoint(
                checkpoint_dir=checkpoint_dir,
                model=model,
                tokenizer=_DummyTokenizer(),
                processor=_DummyProcessor(),
                optimizer=None,
                scheduler=None,
                trainer_state={"global_step": 3},
                config_dict={"experiment": {"name": "unit-test"}, "finetune": {"mode": "lora"}},
            )

            self.assertTrue((checkpoint_dir / "adapter_config.json").exists())
            self.assertTrue((checkpoint_dir / "adapter_model.bin").exists())
            self.assertTrue((checkpoint_dir / "base_model" / "config.json").exists())
            self.assertTrue((checkpoint_dir / "base_model" / "model.safetensors").exists())
            self.assertFalse((checkpoint_dir / "state_dict.pt").exists())
            self.assertTrue((checkpoint_dir / "tokenizer_config.json").exists())
            self.assertTrue((checkpoint_dir / "tokenizer.json").exists())
            self.assertTrue((checkpoint_dir / "preprocessor_config.json").exists())
            self.assertTrue((checkpoint_dir / "trainer_state.json").exists())
            self.assertTrue((checkpoint_dir / "meta.json").exists())

    def test_load_training_checkpoint_loads_adapter_layout_for_lora(self) -> None:
        source_model = _DummyPeftModel()

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir)
            save_training_checkpoint(
                checkpoint_dir=checkpoint_dir,
                model=source_model,
                tokenizer=_DummyTokenizer(),
                processor=_DummyProcessor(),
                optimizer=None,
                scheduler=None,
                trainer_state={"global_step": 5},
                config_dict={"experiment": {"name": "unit-test"}, "finetune": {"mode": "lora"}},
            )

            target_model = _DummyPeftModel()
            trainer_state = load_training_checkpoint(
                checkpoint_dir=checkpoint_dir,
                model=target_model,
                strict=True,
                resume_training_state=True,
            )

            self.assertEqual(trainer_state["global_step"], 5)
            self.assertEqual(len(target_model.loaded_adapters), 1)
            loaded_model_id, loaded_adapter_name, is_trainable = target_model.loaded_adapters[0]
            self.assertEqual(Path(loaded_model_id), checkpoint_dir)
            self.assertEqual(loaded_adapter_name, "default")
            self.assertTrue(is_trainable)

    def test_load_initial_model_checkpoint_keeps_adapter_trainable(self) -> None:
        source_model = _DummyPeftModel()

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir)
            save_training_checkpoint(
                checkpoint_dir=checkpoint_dir,
                model=source_model,
                tokenizer=_DummyTokenizer(),
                processor=_DummyProcessor(),
                optimizer=None,
                scheduler=None,
                trainer_state={"global_step": 5},
                config_dict={"experiment": {"name": "unit-test"}, "finetune": {"mode": "lora"}},
            )

            target_model = _DummyPeftModel()
            load_initial_model_checkpoint(
                checkpoint_dir=checkpoint_dir,
                model=target_model,
                strict=True,
            )

            self.assertEqual(len(target_model.loaded_adapters), 1)
            _, _, is_trainable = target_model.loaded_adapters[0]
            self.assertTrue(is_trainable)


if __name__ == "__main__":
    unittest.main()
