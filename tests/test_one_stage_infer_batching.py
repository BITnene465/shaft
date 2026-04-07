from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image


def _load_infer_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "arrow" / "infer.py"
    spec = importlib.util.spec_from_file_location("arrow_infer_script", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load script module: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeRunner:
    def __init__(self, *, batch_size: int) -> None:
        self.settings = SimpleNamespace(batch_size=batch_size)
        self.batch_sizes: list[int] = []

    def predict_batch(self, images, *, max_new_tokens=None):
        self.batch_sizes.append(len(images))
        return [
            (
                f"raw-{index}",
                {
                    "generation": {
                        "requested_max_new_tokens": int(max_new_tokens or 0),
                        "generated_tokens": 1,
                        "returned_tokens": 1,
                        "stop_reason": "eos_or_unknown",
                        "closed_json_payload": True,
                        "hit_max_new_tokens": False,
                    },
                    "lenient": {
                        "ok": True,
                        "prediction": {"instances": [{"label": "single_arrow"}]},
                        "error": None,
                        "recovered_prefix": False,
                    },
                    "strict": {
                        "ok": True,
                        "prediction": {"instances": [{"label": "single_arrow"}]},
                        "error": None,
                        "recovered_prefix": False,
                    },
                },
            )
            for index, _ in enumerate(images, start=1)
        ]


class OneStageInferBatchingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = _load_infer_script_module()

    def test_resolve_batch_size_prefers_cli_override(self) -> None:
        runner = _FakeRunner(batch_size=4)
        args = SimpleNamespace(batch_size=2)
        self.assertEqual(self.module._resolve_batch_size(args, runner), 2)

    def test_directory_inference_uses_predict_batch(self) -> None:
        runner = _FakeRunner(batch_size=2)
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            image_paths: list[Path] = []
            for index in range(5):
                image_path = tmp_path / f"sample_{index}.png"
                Image.new("RGB", (8, 8), color=(index, index, index)).save(image_path)
                image_paths.append(image_path)

            manifest = self.module._run_directory_inference(
                runner=runner,
                image_paths=image_paths,
                output_dir=tmp_path / "outputs",
                batch_size=2,
                max_new_tokens=64,
                save_preview=False,
            )

            self.assertEqual(runner.batch_sizes, [2, 2, 1])
            self.assertEqual(len(manifest), 5)
            self.assertTrue(all(entry["strict_ok"] for entry in manifest))
            self.assertTrue(all(entry["preview_path"] is None for entry in manifest))


if __name__ == "__main__":
    unittest.main()
