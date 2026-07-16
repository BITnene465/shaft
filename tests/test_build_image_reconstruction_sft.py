from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from PIL import Image


def _write_raw_sample(
    raw_root: Path,
    *,
    stem: str,
    image_types: list[str],
) -> None:
    image_path = raw_root / "images" / f"{stem}.png"
    Image.new("RGB", (320, 240), "white").save(image_path)
    instances = []
    for index, image_type in enumerate(image_types):
        left = 10 + index * 60
        instances.append(
            {
                "label": "image",
                "bbox": [left, 20, left + 50, 100],
                "extra": {"parameters": {"image_type": image_type}},
            }
        )
    (raw_root / "json" / f"{stem}.json").write_text(
        json.dumps(
            {
                "image_width": 320,
                "image_height": 240,
                "instances": instances,
            }
        ),
        encoding="utf-8",
    )


def test_build_image_reconstruction_uses_stratified_band_and_excludes_test(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    (raw_root / "images").mkdir(parents=True)
    (raw_root / "json").mkdir(parents=True)
    (raw_root / "splits").mkdir(parents=True)
    image_types = [
        "chart",
        "diagram",
        "document",
        "illustration",
        "infographic",
        "map",
        "medical",
        "microscopy",
        "other",
        "photo",
        "rendering",
        "screenshot",
        "table",
    ]
    for image_type in image_types:
        _write_raw_sample(raw_root, stem=f"train_{image_type}", image_types=[image_type])
    _write_raw_sample(raw_root, stem="train_chart_extra", image_types=["chart", "chart"])
    _write_raw_sample(raw_root, stem="test_sample", image_types=["screenshot"])
    manifest = raw_root / "splits/main.test.json"
    manifest.write_text(
        json.dumps({"items": [{"id": "test_sample", "image_path": "images/test_sample.png"}]}),
        encoding="utf-8",
    )
    output_root = tmp_path / "output"
    subprocess.run(
        [
            sys.executable,
            "scripts/tasks/build_image_reconstruction_sft.py",
            "--raw-root",
            str(raw_root),
            "--output-root",
            str(output_root),
            "--exclude-manifests",
            str(manifest),
            "--minimum-per-class",
            "2",
            "--maximum-per-class",
            "2",
            "--workers",
            "1",
            "--clean",
        ],
        cwd=Path.cwd(),
        check=True,
        capture_output=True,
        text=True,
    )

    task_root = output_root / "image_reconstruction"
    structured = [
        json.loads(line) for line in (task_root / "structured/train.jsonl").read_text().splitlines()
    ]
    sft = [json.loads(line) for line in (task_root / "sft/train.jsonl").read_text().splitlines()]
    assert len(structured) == len(sft) == 26
    distribution = Counter(
        json.loads(row["target_text"])["parameters"]["image_type"] for row in sft
    )
    assert set(distribution) == set(image_types)
    assert set(distribution.values()) == {2}
    assert "test_sample" not in {row["extra"]["source_sample_id"] for row in sft}
    for structured_row, sft_row in zip(structured, sft, strict=True):
        assert structured_row["sample_id"] == sft_row["sample_id"]
        target = json.loads(sft_row["target_text"])
        assert set(target) == {"type", "parameters"}
        assert target["type"] == "image"
        assert set(target["parameters"]) == {"image_type"}
        image_path = (task_root / "structured" / structured_row["image_path"]).resolve()
        assert image_path.is_file()


def test_stratified_selector_caps_heads_and_adds_views_for_rare_classes() -> None:
    import importlib.util

    script_path = Path("scripts/tasks/build_image_reconstruction_sft.py").resolve()
    spec = importlib.util.spec_from_file_location("build_image_reconstruction_sft", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    candidates = []
    for image_type in sorted(module.ALLOWED_IMAGE_TYPES):
        count = 3 if image_type == "chart" else 1
        for index in range(count):
            candidates.append(
                module.Candidate(
                    stem=f"{image_type}_{index}",
                    instance_index=0,
                    image_path=Path(f"{image_type}_{index}.png"),
                    image_width=100,
                    image_height=100,
                    bbox=(10, 10, 90, 90),
                    image_type=image_type,
                )
            )
    selected, counts = module._select_candidates(
        candidates,
        minimum_per_class=2,
        maximum_per_class=2,
        seed=42,
    )

    distribution = Counter(item.candidate.image_type for item in selected)
    assert set(distribution) == module.ALLOWED_IMAGE_TYPES
    assert set(distribution.values()) == {2}
    assert counts["selected_total"] == 26
    assert any(item.view_index > 0 for item in selected if item.candidate.image_type == "document")
