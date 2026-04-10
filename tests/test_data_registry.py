from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from vlm_structgen.core.config import ExperimentRuntimeConfig, load_config
from vlm_structgen.core.data.registry_loader import resolve_training_data_sources


class DataRegistryTests(unittest.TestCase):
    def test_mixed_train_config_resolves_from_registry(self) -> None:
        config_path = Path("configs/train/train_mixed_full_ft_4b.yaml")
        config = load_config(config_path)
        resolved = resolve_training_data_sources(config, config_path=config_path)

        self.assertEqual(resolved.source_mode, "registry_only")
        self.assertEqual(
            resolved.train_paths,
            [
                "data/two_stage/stage1/train_mixed.jsonl",
                "data/two_stage/stage2/train.jsonl",
            ],
        )
        self.assertEqual(
            resolved.train_routes,
            [
                "grounding/arrow",
                "keypoint_sequence/arrow",
            ],
        )
        self.assertEqual(
            sorted(resolved.route_option_defaults.keys()),
            ["grounding/arrow", "keypoint_sequence/arrow"],
        )
        self.assertEqual(resolved.route_option_defaults["grounding/arrow"]["mix_weight"], 1.0)
        self.assertEqual(
            resolved.route_prompt_defaults["grounding/arrow"]["profile"],
            "arrow.grounding.stage1.v2",
        )
        self.assertIn(
            "Locate every instance",
            str(resolved.route_prompt_defaults["grounding/arrow"]["user_prompt"]),
        )

    def test_registry_mode_rejects_unknown_dataset_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            registry_path = temp_root / "registry.yaml"
            registry_path.write_text(
                "datasets:\n"
                "  s1:\n"
                "    task_type: grounding\n"
                "    domain_type: arrow\n"
                "    train_path: data/train.jsonl\n"
                "    val_path: data/val.jsonl\n",
                encoding="utf-8",
            )
            config = ExperimentRuntimeConfig()
            config.data.registry_path = str(registry_path)
            config.data.train_datasets = ["missing"]
            config.data.val_datasets = ["s1"]

            with self.assertRaisesRegex(ValueError, "Unknown dataset id"):
                resolve_training_data_sources(config, config_path=temp_root / "train.yaml")

    def test_registry_mode_requires_dataset_ids(self) -> None:
        config = ExperimentRuntimeConfig()
        config.data.registry_path = "configs/data_registry/arrow.yaml"
        config.data.train_datasets = []
        config.data.val_datasets = []
        with self.assertRaisesRegex(ValueError, "non-empty data.train_datasets"):
            resolve_training_data_sources(config, config_path="configs/train/train_stage1_lora_4b.yaml")


if __name__ == "__main__":
    unittest.main()
