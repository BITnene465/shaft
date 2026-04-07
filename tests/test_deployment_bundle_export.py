from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from vlm_structgen.core.modeling import AdapterBundleSpec, export_deployment_bundle


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


class DeploymentBundleExportTests(unittest.TestCase):
    def test_export_deployment_bundle_copies_base_and_adapters(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            base_source = temp_dir / "base_source"
            _write_json(base_source / "config.json", {"model_type": "dummy"})
            (base_source / "model.safetensors").write_text("base-model", encoding="utf-8")
            _write_json(base_source / "tokenizer_config.json", {"tokenizer_class": "Dummy"})
            _write_json(base_source / "tokenizer.json", {"version": 1})
            _write_json(base_source / "preprocessor_config.json", {"processor_class": "Dummy"})

            adapter1 = temp_dir / "adapter1"
            _write_json(adapter1 / "adapter_config.json", {"base_model_name_or_path": "dummy-base"})
            (adapter1 / "adapter_model.safetensors").write_text("adapter-one", encoding="utf-8")
            (adapter1 / "README.md").write_text("# adapter 1\n", encoding="utf-8")

            adapter2 = temp_dir / "adapter2"
            _write_json(adapter2 / "adapter_config.json", {"base_model_name_or_path": "dummy-base"})
            (adapter2 / "adapter_model.bin").write_text("adapter-two", encoding="utf-8")

            output_dir = temp_dir / "bundle"
            result = export_deployment_bundle(
                base_source_dir=base_source,
                adapter_specs=[
                    AdapterBundleSpec(route="grounding/arrow", checkpoint_dir=adapter1),
                    AdapterBundleSpec(route="keypoint_sequence/arrow", checkpoint_dir=adapter2),
                ],
                output_dir=output_dir,
                overwrite=False,
            )

            self.assertEqual(result.output_dir, output_dir)
            self.assertTrue((output_dir / "base_model" / "config.json").exists())
            self.assertTrue((output_dir / "base_model" / "model.safetensors").exists())
            self.assertTrue((output_dir / "base_model" / "tokenizer_config.json").exists())
            self.assertTrue((output_dir / "base_model" / "tokenizer.json").exists())
            self.assertTrue((output_dir / "base_model" / "preprocessor_config.json").exists())
            self.assertTrue((output_dir / "adapters" / "grounding_arrow" / "adapter_config.json").exists())
            self.assertTrue((output_dir / "adapters" / "grounding_arrow" / "adapter_model.safetensors").exists())
            self.assertTrue((output_dir / "adapters" / "keypoint_sequence_arrow" / "adapter_config.json").exists())
            self.assertTrue((output_dir / "adapters" / "keypoint_sequence_arrow" / "adapter_model.bin").exists())

            manifest = json.loads((output_dir / "manifests" / "adapters.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["format"], "peft_base_model_plus_adapters")
            self.assertEqual(manifest["base_model"]["path"], "base_model")
            self.assertEqual(manifest["adapters"]["grounding/arrow"]["path"], "adapters/grounding_arrow")
            self.assertEqual(
                manifest["adapters"]["keypoint_sequence/arrow"]["path"],
                "adapters/keypoint_sequence_arrow",
            )


if __name__ == "__main__":
    unittest.main()
