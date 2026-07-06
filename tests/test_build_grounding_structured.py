from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    script_path = Path("scripts/tasks/build_grounding_structured.py").resolve()
    spec = importlib.util.spec_from_file_location("build_grounding_structured", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_read_split_accepts_vlm_json_manifest(tmp_path: Path) -> None:
    module = _load_module()
    split_path = tmp_path / "vlm.test.json"
    split_path.write_text(
        json.dumps(
            {
                "schema": "vlm_data.test_split.v2",
                "name": "vlm.test",
                "task": "vlm",
                "split": "test",
                "items": [
                    {"id": "sample_a", "image_path": "images/sample_a.png"},
                    {"image_path": "images/sample_b.jpg"},
                    {"json_path": "part1/json/custom.json", "image_path": "images/ignored.png"},
                ],
            }
        ),
        encoding="utf-8",
    )

    assert module._read_split(split_path) == [
        "json/sample_a.json",
        "json/sample_b.json",
        "part1/json/custom.json",
    ]
