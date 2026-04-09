from __future__ import annotations

import json
import importlib.util
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import types
from unittest.mock import patch

from PIL import Image

from vlm_structgen.mixing.route_loader import (
    RouteAwareTrainLoader,
    RouteEpochController,
    build_route_aware_train_loader,
    collect_route_groups,
    extract_route_weights,
)


class RouteMixingTests(unittest.TestCase):
    def test_dataset_accepts_multiple_jsonl_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            image_a = temp_path / "a.png"
            image_b = temp_path / "b.png"
            jsonl_a = temp_path / "a.jsonl"
            jsonl_b = temp_path / "b.jsonl"
            Image.new("RGB", (8, 8), color="black").save(image_a)
            Image.new("RGB", (8, 8), color="white").save(image_b)
            with jsonl_a.open("w", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "sample_id": "a",
                            "image_path": str(image_a),
                            "image_width": 8,
                            "image_height": 8,
                            "task_type": "grounding",
                            "domain_type": "arrow",
                            "instances": [],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            with jsonl_b.open("w", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "sample_id": "b",
                            "image_path": str(image_b),
                            "image_width": 8,
                            "image_height": 8,
                            "task_type": "keypoint_sequence",
                            "domain_type": "arrow",
                            "instances": [
                                {
                                    "label": "single_arrow",
                                    "bbox": [0, 0, 1, 1],
                                    "keypoints": [[0, 0], [1, 1]],
                                }
                            ],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            dataset_cls = self._load_dataset_class()
            dataset = dataset_cls(
                jsonl_path=[jsonl_a, jsonl_b],
                num_bins=1000,
                system_prompt="",
                user_prompt="",
            )

            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset.records[0]["sample_id"], "a")
            self.assertEqual(dataset.records[1]["sample_id"], "b")

    def test_collect_route_groups_and_route_weights(self) -> None:
        records = [
            {"task_type": "grounding", "domain_type": "arrow"},
            {"task_type": "grounding", "domain_type": "arrow"},
            {"task_type": "keypoint_sequence", "domain_type": "arrow"},
        ]
        self.assertEqual(
            collect_route_groups(records),
            {
                "grounding/arrow": [0, 1],
                "keypoint_sequence/arrow": [2],
            },
        )
        self.assertEqual(
            extract_route_weights(
                {
                    "grounding/arrow": {"mix_weight": 2.5},
                    "keypoint_sequence/arrow": {"sampling_weight": 0.5},
                },
                ["grounding/arrow", "keypoint_sequence/arrow"],
            ),
            {
                "grounding/arrow": 2.5,
                "keypoint_sequence/arrow": 0.5,
            },
        )

    def test_route_controller_propagates_epoch_and_builds_weighted_schedule(self) -> None:
        sampler_a = SimpleNamespace(epoch=None, set_epoch=lambda epoch: setattr(sampler_a, "epoch", epoch))
        sampler_b = SimpleNamespace(epoch=None, set_epoch=lambda epoch: setattr(sampler_b, "epoch", epoch))
        route_loaders = {
            "grounding/arrow": SimpleNamespace(sampler=sampler_a),
            "keypoint_sequence/arrow": SimpleNamespace(sampler=sampler_b),
        }
        controller = RouteEpochController(
            route_loaders=route_loaders,
            route_weights={"grounding/arrow": 0.5, "keypoint_sequence/arrow": 1.5},
            weight_resolution=10,
            seed=13,
        )
        controller.set_epoch(3)
        self.assertEqual(sampler_a.epoch, 3)
        self.assertEqual(sampler_b.epoch, 3)

        schedule = controller.build_route_schedule(
            ["grounding/arrow", "keypoint_sequence/arrow"],
            cycle_index=0,
        )
        counts = Counter(schedule)
        self.assertEqual(counts["grounding/arrow"], 5)
        self.assertEqual(counts["keypoint_sequence/arrow"], 15)

    def test_route_aware_loader_wrapper_exposes_sampler(self) -> None:
        loader = RouteAwareTrainLoader(
            {"grounding/arrow": SimpleNamespace()},
            controller=RouteEpochController(route_loaders={}, route_weights={}),
        )
        self.assertTrue(hasattr(loader, "sampler"))

    def test_build_route_aware_train_loader_uses_explicit_dist_split_and_keeps_single_route_batches(self) -> None:
        class FakeDataset:
            def __init__(self, items: list[dict[str, str]]) -> None:
                self.items = items
                self.records = items

            def __len__(self) -> int:
                return len(self.items)

            def __getitem__(self, index: int) -> dict[str, str]:
                return self.items[index]

        class FakeSubset:
            def __init__(self, dataset: FakeDataset, indices: list[int]) -> None:
                self.dataset = dataset
                self.indices = list(indices)

            def __len__(self) -> int:
                return len(self.indices)

            def __getitem__(self, index: int) -> dict[str, str]:
                return self.dataset[self.indices[index]]

        class FakeDistributedSampler:
            def __init__(
                self,
                dataset: FakeSubset,
                *,
                num_replicas: int,
                rank: int,
                shuffle: bool,
                seed: int,
            ) -> None:
                self.dataset = dataset
                self.num_replicas = num_replicas
                self.rank = rank
                self.shuffle = shuffle
                self.seed = seed
                self.epoch = None

            def set_epoch(self, epoch: int) -> None:
                self.epoch = epoch

        class FakeDataLoader:
            def __init__(
                self,
                dataset: FakeSubset,
                *,
                batch_size: int,
                shuffle: bool,
                sampler: FakeDistributedSampler | None,
                num_workers: int,
                pin_memory: bool,
                persistent_workers: bool,
                collate_fn,
            ) -> None:
                self.dataset = dataset
                self.batch_size = batch_size
                self.shuffle = shuffle
                self.sampler = sampler
                self.num_workers = num_workers
                self.pin_memory = pin_memory
                self.persistent_workers = persistent_workers
                self.collate_fn = collate_fn

            def __len__(self) -> int:
                size = len(self.dataset)
                return (size + self.batch_size - 1) // self.batch_size

            def __iter__(self):
                order = list(range(len(self.dataset)))
                if self.sampler is not None and getattr(self.sampler, "shuffle", False):
                    order = list(reversed(order))
                for start in range(0, len(order), self.batch_size):
                    batch = [self.dataset[idx] for idx in order[start : start + self.batch_size]]
                    yield self.collate_fn(batch)

        fake_torch = types.ModuleType("torch")
        fake_utils = types.ModuleType("torch.utils")
        fake_data = types.ModuleType("torch.utils.data")
        fake_data.DataLoader = FakeDataLoader
        fake_data.DistributedSampler = FakeDistributedSampler
        fake_data.Subset = FakeSubset
        fake_torch.utils = fake_utils
        fake_utils.data = fake_data

        dataset = FakeDataset(
            [
                {"task_type": "grounding", "domain_type": "arrow", "sample_id": "g1"},
                {"task_type": "grounding", "domain_type": "arrow", "sample_id": "g2"},
                {"task_type": "keypoint_sequence", "domain_type": "arrow", "sample_id": "k1"},
            ]
        )

        with patch.dict(
            "sys.modules",
            {
                "torch": fake_torch,
                "torch.utils": fake_utils,
                "torch.utils.data": fake_data,
            },
        ):
            loader = build_route_aware_train_loader(
                dataset=dataset,
                collator=lambda batch: batch,
                batch_size=1,
                num_workers=0,
                pin_memory=False,
                persistent_workers=False,
                distributed=True,
                world_size=4,
                rank=2,
                shuffle=True,
                route_options={
                    "grounding/arrow": {"mix_weight": 1.0},
                    "keypoint_sequence/arrow": {"mix_weight": 1.0},
                },
                seed=11,
            )

        self.assertTrue(hasattr(loader, "sampler"))
        self.assertEqual(loader.sampler.epoch, 0)
        self.assertEqual(loader.route_loaders["grounding/arrow"].sampler.num_replicas, 4)
        self.assertEqual(loader.route_loaders["grounding/arrow"].sampler.rank, 2)
        self.assertEqual(loader.route_loaders["grounding/arrow"].sampler.seed, 11)

        loader.sampler.set_epoch(7)
        self.assertEqual(loader.route_loaders["grounding/arrow"].sampler.epoch, 7)

        batches = list(loader)
        observed_routes = {
            (batch[0]["task_type"], batch[0]["domain_type"])
            for batch in batches
        }
        self.assertEqual(
            observed_routes,
            {
                ("grounding", "arrow"),
                ("keypoint_sequence", "arrow"),
            },
        )
        self.assertTrue(all(len(batch) == 1 for batch in batches))

    def _load_dataset_class(self):
        root_dir = Path(__file__).resolve().parents[1]
        dataset_path = root_dir / "src" / "vlm_structgen" / "core" / "data" / "dataset.py"

        def load_jsonl(path):
            with Path(path).open("r", encoding="utf-8") as handle:
                return [json.loads(line) for line in handle if line.strip()]

        def get_adapter(*_args, **_kwargs):
            raise AssertionError("get_adapter should not be called in the dataset constructor test.")

        def render_prompt_template(template, condition):
            return template.format(**condition)

        fake_root = types.ModuleType("vlm_structgen")
        fake_root.__path__ = []
        fake_core = types.ModuleType("vlm_structgen.core")
        fake_core.__path__ = []
        fake_utils = types.ModuleType("vlm_structgen.core.utils")
        fake_utils.__path__ = []
        fake_registry = types.ModuleType("vlm_structgen.core.registry")
        fake_registry.get_adapter = get_adapter
        fake_prompting = types.ModuleType("vlm_structgen.core.prompting")
        fake_prompting.render_prompt_template = render_prompt_template
        fake_io = types.ModuleType("vlm_structgen.core.utils.io")
        fake_io.load_jsonl = load_jsonl

        with patch.dict(
            sys.modules,
            {
                "vlm_structgen": fake_root,
                "vlm_structgen.core": fake_core,
                "vlm_structgen.core.utils": fake_utils,
                "vlm_structgen.core.registry": fake_registry,
                "vlm_structgen.core.prompting": fake_prompting,
                "vlm_structgen.core.utils.io": fake_io,
            },
        ):
            spec = importlib.util.spec_from_file_location("vlm_structgen_core_data_dataset_test", dataset_path)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        return module.SFTDataset


if __name__ == "__main__":
    unittest.main()
