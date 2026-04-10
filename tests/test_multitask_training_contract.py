from __future__ import annotations

import sys
import tempfile
import types
from importlib.machinery import ModuleSpec
from pathlib import Path
import unittest


def _install_torch_stub() -> None:
    if "torch" in sys.modules:
        return

    fake_torch = types.ModuleType("torch")
    fake_torch.__spec__ = ModuleSpec("torch", loader=None)

    class _FakeDType:
        def __init__(self, name: str) -> None:
            self.name = name

        def __repr__(self) -> str:
            return f"torch.{self.name}"

        def __hash__(self) -> int:
            return hash(self.name)

        def __eq__(self, other: object) -> bool:
            return isinstance(other, _FakeDType) and other.name == self.name

    class _FakeTensor(list):
        def __init__(self, values, dtype=None):
            super().__init__(values)
            self.dtype = dtype or fake_torch.float32
            self.device = fake_torch.device("cpu")

        def item(self):  # noqa: D401
            return float(self[0]) if self else 0.0

        def clone(self):
            return _FakeTensor(self[:], dtype=self.dtype)

        def to(self, *_args, **_kwargs):
            return self

        def contiguous(self):
            return self

        def nelement(self):
            return len(self)

        def numel(self):
            return len(self)

        def data_ptr(self):
            return id(self)

        def untyped_storage(self):
            return types.SimpleNamespace(data_ptr=lambda: id(self), nbytes=lambda: len(self))

        def storage(self):
            return types.SimpleNamespace(data_ptr=lambda: id(self), size=lambda: len(self))

        def view(self, *_args, **_kwargs):
            return self

        def __getitem__(self, item):
            value = super().__getitem__(item)
            if isinstance(item, slice):
                return _FakeTensor(value, dtype=self.dtype)
            return _FakeTensor([value], dtype=self.dtype)

    class _FakeReduceOp:
        SUM = object()

    class _FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def manual_seed_all(_seed: int) -> None:
            return None

    class _FakeDistributed:
        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def is_initialized() -> bool:
            return False

        @staticmethod
        def get_rank() -> int:
            return 0

        @staticmethod
        def get_world_size() -> int:
            return 1

        @staticmethod
        def init_process_group(*_args, **_kwargs) -> None:
            return None

        @staticmethod
        def destroy_process_group() -> None:
            return None

        @staticmethod
        def barrier() -> None:
            return None

        @staticmethod
        def all_reduce(*_args, **_kwargs) -> None:
            return None

    class _FakeDDP:
        def __init__(self, module, *args, **kwargs) -> None:
            del args, kwargs
            self.module = module

    fake_torch_nn_functional = types.ModuleType("torch.nn.functional")
    fake_torch_nn_functional.__spec__ = ModuleSpec("torch.nn.functional", loader=None)
    fake_torch_nn_functional.embedding = lambda *args, **kwargs: _FakeTensor([0])  # type: ignore[assignment]
    fake_torch_nn_functional.pad = lambda tensor, *args, **kwargs: tensor  # type: ignore[assignment]

    fake_torch_nn = types.ModuleType("torch.nn")
    fake_torch_nn.__spec__ = ModuleSpec("torch.nn", loader=None)
    fake_torch_nn_parallel = types.ModuleType("torch.nn.parallel")
    fake_torch_nn_parallel.__spec__ = ModuleSpec("torch.nn.parallel", loader=None)
    fake_torch_nn_utils = types.ModuleType("torch.nn.utils")
    fake_torch_nn_utils.__spec__ = ModuleSpec("torch.nn.utils", loader=None)

    class _FakeModule:
        pass

    fake_torch_nn.Module = _FakeModule  # type: ignore[assignment]
    fake_torch_nn.parallel = fake_torch_nn_parallel  # type: ignore[assignment]
    fake_torch_nn.utils = fake_torch_nn_utils  # type: ignore[assignment]
    fake_torch_nn.functional = fake_torch_nn_functional  # type: ignore[assignment]
    fake_torch_nn_parallel.DistributedDataParallel = _FakeDDP  # type: ignore[assignment]
    fake_torch_nn_utils.clip_grad_norm_ = lambda *_args, **_kwargs: types.SimpleNamespace(item=lambda: 0.0)  # type: ignore[assignment]

    fake_torch.dtype = _FakeDType
    fake_torch.bool = _FakeDType("bool")
    fake_torch.uint8 = _FakeDType("uint8")
    fake_torch.int8 = _FakeDType("int8")
    fake_torch.int16 = _FakeDType("int16")
    fake_torch.int32 = _FakeDType("int32")
    fake_torch.int64 = _FakeDType("int64")
    fake_torch.long = fake_torch.int64
    fake_torch.float16 = _FakeDType("float16")
    fake_torch.float32 = _FakeDType("float32")
    fake_torch.float64 = _FakeDType("float64")
    fake_torch.half = fake_torch.float16
    fake_torch.float = fake_torch.float32
    fake_torch.double = fake_torch.float64
    fake_torch.bfloat16 = _FakeDType("bfloat16")
    fake_torch.complex64 = _FakeDType("complex64")
    fake_torch.complex128 = _FakeDType("complex128")
    fake_torch.Tensor = _FakeTensor
    fake_torch.device = lambda *args, **kwargs: types.SimpleNamespace(type=args[0] if args else "cpu", index=None, args=args, kwargs=kwargs)  # type: ignore[assignment]
    fake_torch.manual_seed = lambda _seed: None  # type: ignore[assignment]
    fake_torch.no_grad = lambda: types.SimpleNamespace(__enter__=lambda self: None, __exit__=lambda self, exc_type, exc, tb: None)
    fake_torch.autocast = lambda *args, **kwargs: types.SimpleNamespace(__enter__=lambda self: None, __exit__=lambda self, exc_type, exc, tb: None)
    fake_torch.tensor = lambda values, *args, **kwargs: _FakeTensor(list(values), dtype=kwargs.get("dtype"))  # type: ignore[assignment]
    fake_torch.zeros = lambda shape, *args, **kwargs: _FakeTensor([0] * int(shape[0]), dtype=kwargs.get("dtype"))  # type: ignore[assignment]
    fake_torch.ones = lambda shape, *args, **kwargs: _FakeTensor([1] * int(shape[0]), dtype=kwargs.get("dtype"))  # type: ignore[assignment]
    fake_torch.full = lambda shape, fill_value, *args, **kwargs: _FakeTensor([fill_value] * int(shape[0]), dtype=kwargs.get("dtype"))  # type: ignore[assignment]
    fake_torch.ones_like = lambda tensor, *args, **kwargs: _FakeTensor([1] * len(tensor), dtype=kwargs.get("dtype", getattr(tensor, "dtype", None)))  # type: ignore[assignment]
    fake_torch.zeros_like = lambda tensor, *args, **kwargs: _FakeTensor([0] * len(tensor), dtype=kwargs.get("dtype", getattr(tensor, "dtype", None)))  # type: ignore[assignment]
    fake_torch.cat = lambda tensors, *args, **kwargs: _FakeTensor([item for tensor in tensors for item in tensor], dtype=getattr(tensors[0], "dtype", None))  # type: ignore[assignment]
    fake_torch.stack = lambda tensors, *args, **kwargs: _FakeTensor([tensor[:] for tensor in tensors], dtype=getattr(tensors[0], "dtype", None))  # type: ignore[assignment]
    fake_torch.nn = fake_torch_nn  # type: ignore[assignment]
    fake_torch.cuda = _FakeCuda()
    fake_torch.random = types.SimpleNamespace(get_rng_state=lambda: b"", set_rng_state=lambda _state: None)
    fake_torch.distributed = _FakeDistributed()
    fake_torch.ReduceOp = _FakeReduceOp
    fake_torch.__version__ = "2.1.0"
    fake_torch.__file__ = "/tmp/fake_torch.py"
    fake_torch.__path__ = []  # type: ignore[attr-defined]
    fake_torch.__getattr__ = lambda name: _FakeDType(name) if name.endswith(("16", "32", "64")) else None  # type: ignore[attr-defined]

    fake_safetensors = types.ModuleType("safetensors")
    fake_safetensors.__spec__ = ModuleSpec("safetensors", loader=None)
    fake_safetensors.safe_open = lambda *args, **kwargs: None  # type: ignore[assignment]
    fake_safetensors.deserialize = lambda *args, **kwargs: None  # type: ignore[assignment]
    fake_safetensors.serialize = lambda *args, **kwargs: b""  # type: ignore[assignment]
    fake_safetensors.serialize_file = lambda *args, **kwargs: None  # type: ignore[assignment]
    fake_safetensors.torch = types.ModuleType("safetensors.torch")
    fake_safetensors.torch.__spec__ = ModuleSpec("safetensors.torch", loader=None)
    fake_safetensors.torch.save_file = lambda *args, **kwargs: None  # type: ignore[assignment]
    fake_safetensors.torch.load_file = lambda *args, **kwargs: {}  # type: ignore[assignment]
    fake_safetensors.torch.storage_ptr = lambda *_args, **_kwargs: 0  # type: ignore[assignment]
    fake_safetensors.torch.storage_size = lambda *_args, **_kwargs: 0  # type: ignore[assignment]
    fake_torch_utils = types.ModuleType("torch.utils")
    fake_torch_utils.__spec__ = ModuleSpec("torch.utils", loader=None)
    fake_torch_utils_pytree = types.ModuleType("torch.utils._pytree")
    fake_torch_utils_pytree.__spec__ = ModuleSpec("torch.utils._pytree", loader=None)
    fake_torch_utils_pytree.Context = list  # type: ignore[assignment]
    fake_torch_utils_pytree.register_pytree_node = lambda *args, **kwargs: None  # type: ignore[assignment]
    fake_torch_utils._pytree = fake_torch_utils_pytree  # type: ignore[attr-defined]
    fake_torch.utils = fake_torch_utils  # type: ignore[assignment]

    sys.modules["torch"] = fake_torch
    sys.modules["torch.nn"] = fake_torch_nn
    sys.modules["torch.nn.functional"] = fake_torch_nn_functional
    sys.modules["torch.nn.parallel"] = fake_torch_nn_parallel
    sys.modules["torch.nn.utils"] = fake_torch_nn_utils
    sys.modules["torch.distributed"] = fake_torch.distributed
    sys.modules["torch.utils"] = fake_torch_utils
    sys.modules["torch.utils._pytree"] = fake_torch_utils_pytree
    sys.modules["safetensors"] = fake_safetensors
    sys.modules["safetensors.torch"] = fake_safetensors.torch
    fake_generation = types.ModuleType("vlm_structgen.core.utils.generation")
    fake_generation.__spec__ = ModuleSpec("vlm_structgen.core.utils.generation", loader=None)
    fake_generation.build_generate_kwargs = lambda *args, **kwargs: {}  # type: ignore[assignment]
    fake_generation.trim_generated_ids_at_eos = lambda generated_ids, _eos_token_id: generated_ids  # type: ignore[assignment]
    sys.modules["vlm_structgen.core.utils.generation"] = fake_generation
    fake_torch.nn.__spec__ = ModuleSpec("torch.nn", loader=None)
    fake_torch.nn.parallel.__spec__ = ModuleSpec("torch.nn.parallel", loader=None)
    fake_torch.distributed.__spec__ = ModuleSpec("torch.distributed", loader=None)


_install_torch_stub()

from vlm_structgen.core.config import ExperimentRuntimeConfig, load_config
from vlm_structgen.core.eval.evaluator import Evaluator
from vlm_structgen.core.train.trainer import Trainer


def _make_trainer_stub(config: ExperimentRuntimeConfig) -> Trainer:
    trainer = Trainer.__new__(Trainer)
    trainer.config = config
    trainer.best_metric = float("-inf") if config.eval.monitor_mode == "max" else float("inf")
    return trainer


def _make_evaluator_stub(
    *,
    task_route_options: dict[str, dict[str, object]] | None = None,
) -> Evaluator:
    return Evaluator(
        num_bins=1000,
        tokenizer=types.SimpleNamespace(),
        max_new_tokens=128,
        task_route_options=task_route_options,
    )


def _compose_multi_task_score(task_scores: dict[str, float], route_weights: dict[str, float]) -> float:
    missing_routes = sorted(set(route_weights) - set(task_scores))
    if missing_routes:
        raise KeyError(f"Missing task scores for routes: {missing_routes}")
    total_weight = sum(float(weight) for weight in route_weights.values())
    if total_weight <= 0:
        raise ValueError("route_weights must sum to a positive value.")
    weighted_sum = 0.0
    for route, weight in route_weights.items():
        score = float(task_scores[route])
        if score < 0.0 or score > 1.0:
            raise ValueError(f"task score for {route!r} must be normalized to [0, 1], got {score!r}.")
        weighted_sum += float(weight) * score
    return weighted_sum / total_weight


class MultiTaskRouteContractTests(unittest.TestCase):
    def test_deprecated_monitor_metric_field_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "deprecated_eval_field.yaml"
            config_path.write_text(
                "eval:\n"
                "  monitor_metric: val/end_to_end_score\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "monitor_metric"):
                load_config(config_path)

    def test_multi_task_score_controls_best_checkpoint_selection(self) -> None:
        config = ExperimentRuntimeConfig()
        config.eval.best_metric = "val/multi_task_score"
        config.eval.monitor_mode = "max"
        trainer = _make_trainer_stub(config)

        task_scores = {
            "grounding/arrow": 0.80,
            "keypoint_sequence/arrow": 0.50,
        }

        score_equal_weight = _compose_multi_task_score(
            task_scores,
            {
                "grounding/arrow": 0.5,
                "keypoint_sequence/arrow": 0.5,
            },
        )
        score_keypoint_heavy = _compose_multi_task_score(
            task_scores,
            {
                "grounding/arrow": 0.2,
                "keypoint_sequence/arrow": 0.8,
            },
        )

        self.assertAlmostEqual(score_equal_weight, 0.65)
        self.assertAlmostEqual(score_keypoint_heavy, 0.56)
        self.assertLess(score_keypoint_heavy, score_equal_weight)

        trainer._maybe_update_best(
            {
                "val/multi_task_score": score_equal_weight,
                "eval_loss": 0.123,
            }
        )
        self.assertAlmostEqual(trainer.best_metric, score_equal_weight)
        self.assertTrue(
            trainer._is_best(
                {
                    "val/multi_task_score": score_equal_weight,
                    "eval_loss": 0.001,
                }
            )
        )
        trainer._maybe_update_best(
            {
                "val/multi_task_score": score_keypoint_heavy,
                "eval_loss": 0.0001,
            }
        )
        self.assertAlmostEqual(trainer.best_metric, score_equal_weight)
        self.assertFalse(
            trainer._is_best(
                {
                    "val/multi_task_score": score_keypoint_heavy,
                    "eval_loss": 0.0001,
                }
            )
        )

    def test_best_metric_controls_checkpoint_selection(self) -> None:
        config = ExperimentRuntimeConfig()
        config.eval.best_metric = "val/multi_task_score"
        config.eval.monitor_mode = "max"
        trainer = _make_trainer_stub(config)

        trainer._maybe_update_best(
            {
                "val/multi_task_score": 0.61,
                "val/end_to_end_score": 0.95,
            }
        )
        self.assertAlmostEqual(trainer.best_metric, 0.61)
        self.assertTrue(
            trainer._is_best(
                {
                    "val/multi_task_score": 0.61,
                    "val/end_to_end_score": 0.10,
                }
            )
        )
        self.assertFalse(
            trainer._is_best(
                {
                    "val/multi_task_score": 0.60,
                    "val/end_to_end_score": 0.99,
                }
            )
        )

    def test_eval_loss_is_auxiliary_when_multi_task_score_is_primary(self) -> None:
        config = ExperimentRuntimeConfig()
        config.eval.best_metric = "val/multi_task_score"
        config.eval.monitor_mode = "max"
        trainer = _make_trainer_stub(config)

        trainer._maybe_update_best(
            {
                "val/multi_task_score": 0.72,
                "eval_loss": 1.23,
            }
        )
        self.assertAlmostEqual(trainer.best_metric, 0.72)

        trainer._maybe_update_best(
            {
                "val/multi_task_score": 0.70,
                "eval_loss": 0.01,
            }
        )
        self.assertAlmostEqual(trainer.best_metric, 0.72)

    def test_evaluator_aggregates_routes_and_multitask_score(self) -> None:
        evaluator = _make_evaluator_stub(
            task_route_options={
                "grounding/arrow": {
                    "eval_primary_metric": "bbox_precision_at_iou50",
                    "eval_metric_weight": 0.5,
                    "eval_metric_normalizer": "identity",
                },
                "keypoint_sequence/arrow": {
                    "eval_primary_metric": "end_to_end_score",
                    "eval_metric_weight": 0.5,
                    "eval_metric_normalizer": "identity",
                },
            }
        )

        summary = evaluator.summarize(
            {
                "samples": 6.0,
                "parse_success_lenient": 6.0,
                "parse_success_strict": 5.0,
                "structured_samples": 0.0,
                "grounding_samples": 4.0,
                "stage2_samples": 2.0,
                "gt_instances": 2.0,
                "pred_instances": 0.0,
                "bbox_tp": 3.0,
                "bbox_fp": 1.0,
                "bbox_fn": 0.0,
                "bbox_iou_sum": 2.4,
                "point_distance_sum": 8.0,
                "point_count": 4.0,
                "keypoint_count_exact": 1.0,
                "end_to_end_correct": 1.0,
                "__route__::grounding::arrow::samples": 4.0,
                "__route__::grounding::arrow::parse_success_lenient": 4.0,
                "__route__::grounding::arrow::parse_success_strict": 3.0,
                "__route__::grounding::arrow::bbox_tp": 3.0,
                "__route__::grounding::arrow::bbox_fp": 1.0,
                "__route__::grounding::arrow::bbox_fn": 0.0,
                "__route__::grounding::arrow::bbox_iou_sum": 2.4,
                "__route__::keypoint_sequence::arrow::samples": 2.0,
                "__route__::keypoint_sequence::arrow::parse_success_lenient": 2.0,
                "__route__::keypoint_sequence::arrow::parse_success_strict": 2.0,
                "__route__::keypoint_sequence::arrow::gt_instances": 2.0,
                "__route__::keypoint_sequence::arrow::point_distance_sum": 8.0,
                "__route__::keypoint_sequence::arrow::point_count": 4.0,
                "__route__::keypoint_sequence::arrow::keypoint_count_exact": 1.0,
                "__route__::keypoint_sequence::arrow::end_to_end_correct": 1.0,
            }
        )

        self.assertAlmostEqual(summary["val/routes/grounding__arrow/samples"], 4.0)
        self.assertAlmostEqual(summary["val/routes/keypoint_sequence__arrow/samples"], 2.0)
        self.assertAlmostEqual(summary["val/routes/grounding__arrow/bbox_precision_at_iou50"], 0.75)
        self.assertAlmostEqual(summary["val/routes/grounding__arrow/normalized_primary_metric"], 0.75)
        self.assertAlmostEqual(summary["val/routes/keypoint_sequence__arrow/end_to_end_score"], 0.5)
        self.assertAlmostEqual(summary["val/routes/keypoint_sequence__arrow/normalized_primary_metric"], 0.5)
        self.assertAlmostEqual(summary["val/multi_task_score"], 0.625)

    def test_existing_single_task_configs_keep_their_legacy_metrics(self) -> None:
        stage1_config = load_config("configs/train/train_stage1_lora_4b.yaml")
        self.assertEqual(stage1_config.task.route, "grounding/arrow")
        self.assertEqual(stage1_config.eval.best_metric, "val/bbox_f1_at_iou50")

        stage2_config = load_config("configs/train/train_stage2_lora_4b.yaml")
        self.assertEqual(stage2_config.task.route, "keypoint_sequence/arrow")
        self.assertEqual(stage2_config.eval.best_metric, "val/end_to_end_score")

    def test_new_4b_mixed_full_ft_config_exposes_two_routes_and_multitask_metric(self) -> None:
        mixed_full_ft = load_config("configs/train/train_mixed_full_ft_4b.yaml")
        self.assertEqual(mixed_full_ft.eval.best_metric, "val/multi_task_score")
        self.assertEqual(mixed_full_ft.finetune.mode, "full")
        self.assertFalse(mixed_full_ft.model.freeze_vision_tower)
        self.assertEqual(mixed_full_ft.task.route, None)
        self.assertEqual(
            mixed_full_ft.data.train_route_map,
            {
                "data/two_stage/stage1/train_mixed.jsonl": "grounding/arrow",
                "data/two_stage/stage2/train.jsonl": "keypoint_sequence/arrow",
            },
        )
        self.assertEqual(
            mixed_full_ft.data.val_route_map,
            {
                "data/two_stage/stage1/val_full.jsonl": "grounding/arrow",
                "data/two_stage/stage2/val.jsonl": "keypoint_sequence/arrow",
            },
        )
        self.assertEqual(
            sorted(mixed_full_ft.task.route_options.keys()),
            ["grounding/arrow", "keypoint_sequence/arrow"],
        )
        self.assertEqual(
            sorted(mixed_full_ft.prompt.route_prompts.keys()),
            ["grounding/arrow", "keypoint_sequence/arrow"],
        )


if __name__ == "__main__":
    unittest.main()
