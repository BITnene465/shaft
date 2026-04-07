from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from vlm_structgen.core.utils.checkpoint import load_training_checkpoint, save_training_checkpoint


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


class CheckpointLayoutMigrationTests(unittest.TestCase):
    def test_save_training_checkpoint_writes_flat_layout(self) -> None:
        model = torch.nn.Linear(2, 2)

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
                config_dict={"experiment": {"name": "unit-test"}},
            )

            self.assertTrue((checkpoint_dir / "state_dict.pt").exists())
            self.assertTrue((checkpoint_dir / "tokenizer_config.json").exists())
            self.assertTrue((checkpoint_dir / "tokenizer.json").exists())
            self.assertTrue((checkpoint_dir / "preprocessor_config.json").exists())
            self.assertTrue((checkpoint_dir / "trainer_state.json").exists())
            self.assertTrue((checkpoint_dir / "meta.json").exists())

    def test_load_training_checkpoint_accepts_flat_layout(self) -> None:
        source_model = torch.nn.Linear(2, 2)

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
                config_dict={"experiment": {"name": "unit-test"}},
            )

            target_model = torch.nn.Linear(2, 2)
            trainer_state = load_training_checkpoint(
                checkpoint_dir=checkpoint_dir,
                model=target_model,
                strict=True,
                resume_training_state=False,
            )

            self.assertEqual(trainer_state["global_step"], 5)
            for key, value in source_model.state_dict().items():
                self.assertTrue(torch.allclose(value, target_model.state_dict()[key]))

    def test_load_training_checkpoint_fallbacks_to_legacy_layout(self) -> None:
        model = torch.nn.Linear(2, 2)

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir)
            legacy_model_dir = checkpoint_dir / "model"
            legacy_model_dir.mkdir(parents=True)
            torch.save(model.state_dict(), legacy_model_dir / "state_dict.pt")

            target_model = torch.nn.Linear(2, 2)
            load_training_checkpoint(
                checkpoint_dir=checkpoint_dir,
                model=target_model,
                strict=True,
                resume_training_state=False,
            )

            for key, value in model.state_dict().items():
                self.assertTrue(torch.allclose(value, target_model.state_dict()[key]))


if __name__ == "__main__":
    unittest.main()
