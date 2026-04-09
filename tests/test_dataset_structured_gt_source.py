from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from vlm_structgen.core.data.dataset import SFTDataset
from vlm_structgen.tasks.keypoint_sequence.adapter import build_keypoint_sequence_adapter


class DummyTokenizer:
    def __call__(
        self,
        text,
        *,
        add_special_tokens: bool = False,
        return_attention_mask: bool = False,
        return_token_type_ids: bool = False,
    ):
        del add_special_tokens
        del return_attention_mask
        del return_token_type_ids
        return {"input_ids": list(range(len(text)))}


class StructuredGTSourceTests(unittest.TestCase):
    def test_dataset_rebuilds_stage2_target_from_structured_gt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            image_path = temp_path / "sample.png"
            jsonl_path = temp_path / "sample.jsonl"
            Image.new("RGB", (100, 100), color="black").save(image_path)

            record = {
                "task_type": "keypoint_sequence",
                "domain_type": "arrow",
                "sample_id": "sample-1",
                "image_path": str(image_path),
                "image_width": 100,
                "image_height": 100,
                "system_prompt": "",
                "user_prompt": "",
                "target_text": "{\"keypoints_2d\":[[999,999],[999,999]]}",
                "loss_meta": {"field_char_spans": {"coordinates": [[0, 3]]}},
                "gt_struct": {
                    "task_type": "keypoint_sequence",
                    "domain_type": "arrow",
                    "label": "single_arrow",
                    "keypoints": [[10.0, 20.0], [30.0, 40.0]],
                    "keypoints_2d": [[100, 200], [300, 400]],
                },
                "instances": [
                    {
                        "label": "single_arrow",
                        "bbox": [5.0, 10.0, 35.0, 45.0],
                        "keypoints": [[10.0, 20.0], [30.0, 40.0]],
                    }
                ],
            }
            with jsonl_path.open("w", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

            dataset = SFTDataset(
                jsonl_path=jsonl_path,
                num_bins=1000,
                system_prompt="",
                user_prompt="predict",
            )
            sample = dataset[0]

            adapter = build_keypoint_sequence_adapter(domain_type="arrow", num_bins=1000, task_options={})
            expected_training_target = adapter.build_training_target(
                record["gt_struct"],
                image_width=record["image_width"],
                image_height=record["image_height"],
            )

            self.assertEqual(sample["target_text"], expected_training_target["target_text"])
            self.assertEqual(sample["loss_meta"], expected_training_target["loss_meta"])
            self.assertNotEqual(sample["target_text"], record["target_text"])

            lengths = dataset.get_target_token_lengths(DummyTokenizer())
            self.assertEqual(lengths, [len(expected_training_target["target_text"])])

    def test_dataset_supports_route_level_prompt_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            image_path = temp_path / "sample.png"
            jsonl_path = temp_path / "sample.jsonl"
            Image.new("RGB", (100, 100), color="black").save(image_path)

            record = {
                "task_type": "grounding",
                "domain_type": "arrow",
                "sample_id": "sample-2",
                "image_path": str(image_path),
                "image_width": 100,
                "image_height": 100,
                "instances": [
                    {
                        "label": "single_arrow",
                        "bbox": [10.0, 10.0, 30.0, 30.0],
                    }
                ],
            }
            with jsonl_path.open("w", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

            dataset = SFTDataset(
                jsonl_path=jsonl_path,
                num_bins=1000,
                system_prompt="global-system",
                user_prompt="global-user",
                route_prompts={
                    "grounding/arrow": {
                        "system_prompt": "route-system",
                        "user_prompt": "route-user",
                    }
                },
            )
            sample = dataset[0]
            self.assertEqual(sample["system_prompt"], "route-system")
            self.assertEqual(sample["user_prompt"], "route-user")


if __name__ == "__main__":
    unittest.main()
