from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from shaft.data.sources import load_jsonl_records


def test_new_message_format_extracts_target_and_drops_tail_assistant(tmp_path: Path) -> None:
    image = tmp_path / "img.png"
    Image.new("RGB", (8, 8), color=(0, 0, 0)).save(image)
    jsonl = tmp_path / "samples.jsonl"
    sample = {
        "image": "img.png",
        "dataset_id": "demo",
        "sample_id": "s1",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "detect"}]},
            {"role": "assistant", "content": "{\"ok\":1}"},
        ],
    }
    jsonl.write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")
    records = load_jsonl_records(jsonl, dataset_id="fallback")
    assert len(records) == 1
    record = records[0]
    assert record.dataset_id == "demo"
    assert record.sample_id == "s1"
    assert record.target_text == "{\"ok\":1}"
    assert Path(record.image_path).is_absolute()
    assert len(record.messages or []) == 2
    assert record.messages[-1]["role"] == "user"


def test_missing_target_raises(tmp_path: Path) -> None:
    image = tmp_path / "img.png"
    Image.new("RGB", (8, 8), color=(0, 0, 0)).save(image)
    jsonl = tmp_path / "bad.jsonl"
    sample = {
        "image_path": str(image),
        "messages": [{"role": "user", "content": "only user"}],
    }
    jsonl.write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_jsonl_records(jsonl, dataset_id="x")
