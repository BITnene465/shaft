from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image


def test_build_background_sft_excludes_test_images_and_keeps_boolean_target(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    images_dir = raw_root / "images"
    splits_dir = raw_root / "splits"
    images_dir.mkdir(parents=True)
    splits_dir.mkdir(parents=True)
    records = []
    for index, background in enumerate((False, True, False, True)):
        sample_id = f"sample_{index}"
        image_name = f"{sample_id}.png"
        Image.new("RGB", (64 + index, 48 + index), "white").save(images_dir / image_name)
        records.append(
            {
                "id": sample_id,
                "image_path": image_name,
                "background": background,
                "background_level": 4 if background else 0,
                "reason": "audit only",
            }
        )
    annotations = raw_root / "background.jsonl"
    annotations.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    test_manifest = splits_dir / "main.test.json"
    test_manifest.write_text(
        json.dumps(
            {
                "items": [
                    {"id": "sample_1", "image_path": "images/sample_1.png"},
                ]
            }
        ),
        encoding="utf-8",
    )
    output_root = tmp_path / "output"

    subprocess.run(
        [
            sys.executable,
            "scripts/tasks/build_background_sft.py",
            "--raw-root",
            str(raw_root),
            "--annotations",
            str(annotations),
            "--output-root",
            str(output_root),
            "--exclude-manifests",
            str(test_manifest),
            "--workers",
            "1",
            "--max-image-edge",
            "60",
            "--clean",
        ],
        cwd=Path.cwd(),
        check=True,
        capture_output=True,
        text=True,
    )

    task_root = output_root / "background"
    structured = [
        json.loads(line) for line in (task_root / "structured/train.jsonl").read_text().splitlines()
    ]
    sft = [json.loads(line) for line in (task_root / "sft/train.jsonl").read_text().splitlines()]
    assert len(structured) == len(sft) == 3
    assert "sample_1" not in {row["sample_id"] for row in structured}
    assert not (task_root / "structured/val.jsonl").read_text()
    assert not (task_root / "sft/val.jsonl").read_text()

    for structured_row, sft_row in zip(structured, sft, strict=True):
        assert structured_row["sample_id"] == sft_row["sample_id"]
        assert structured_row["image_path"] == sft_row["image_path"]
        target = json.loads(sft_row["target_text"])
        assert target == {"background": structured_row["background"]}
        assert "background_level" not in sft_row["target_text"]
        assert "reason" not in sft_row["target_text"]
        image_path = (task_root / "structured" / structured_row["image_path"]).resolve()
        assert image_path.is_file()
        with Image.open(image_path) as image:
            assert image.size == (
                structured_row["image_width"],
                structured_row["image_height"],
            )
            assert max(image.size) <= 60
        source_path = images_dir / f"{structured_row['sample_id']}.png"
        with Image.open(source_path) as source_image:
            assert max(source_image.size) > 60
