from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from vlm_structgen.domains.arrow.data.two_stage import prepare_stage2_data


class Stage2MultiviewPrepareTests(unittest.TestCase):
    def test_prepare_stage2_data_generates_multiple_train_views_and_single_val_view(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            input_dir = temp_dir / "input"
            output_dir = temp_dir / "output"
            image_dir = input_dir
            image_dir.mkdir(parents=True, exist_ok=True)

            image_path = image_dir / "sample.png"
            Image.new("RGB", (64, 64), color="black").save(image_path)
            record = {
                "task_type": "grounding",
                "domain_type": "arrow",
                "sample_id": "sample_0001",
                "image_path": str(image_path),
                "image_width": 64,
                "image_height": 64,
                "instances": [
                    {
                        "label": "single_arrow",
                        "bbox": [16.0, 16.0, 32.0, 32.0],
                        "keypoints": [[16.0, 16.0], [32.0, 32.0]],
                    }
                ],
            }
            for split in ("train", "val"):
                with (input_dir / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")

            report = prepare_stage2_data(
                input_dir=input_dir,
                output_dir=output_dir,
                train_padding_ratios=[0.2, 0.3, 0.45],
                val_padding_ratio=0.3,
                num_bins=1000,
                num_workers=1,
            )

            train_lines = (output_dir / "stage2" / "train.jsonl").read_text(encoding="utf-8").strip().splitlines()
            val_lines = (output_dir / "stage2" / "val.jsonl").read_text(encoding="utf-8").strip().splitlines()
            train_samples = [json.loads(line) for line in train_lines]
            val_samples = [json.loads(line) for line in val_lines]

            self.assertEqual(len(train_samples), 3)
            self.assertEqual(len(val_samples), 1)
            self.assertEqual(report["train_padding_ratios"], [0.2, 0.3, 0.45])
            self.assertEqual(report["val_padding_ratio"], 0.3)
            self.assertTrue(any("__pad200" in sample["sample_id"] for sample in train_samples))
            self.assertTrue(any("__pad300" in sample["sample_id"] for sample in train_samples))
            self.assertTrue(any("__pad450" in sample["sample_id"] for sample in train_samples))
            self.assertIn("__pad300", val_samples[0]["sample_id"])
            self.assertEqual(val_samples[0]["padding_ratio"], 0.3)


if __name__ == "__main__":
    unittest.main()
