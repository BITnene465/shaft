from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from shaft.config import DataSourceConfig
from shaft.data.sources import (
    DATA_SOURCE_REGISTRY,
    build_data_source,
    load_jsonl_dpo_records,
    load_jsonl_ppo_records,
    load_jsonl_records,
)


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


def test_jsonl_loader_reports_aggregated_errors(tmp_path: Path) -> None:
    image = tmp_path / "img.png"
    Image.new("RGB", (8, 8), color=(0, 0, 0)).save(image)
    jsonl = tmp_path / "bad_many.jsonl"
    rows = [
        {"image_path": str(image), "target_text": "{\"ok\":1}"},
        {"image_path": str(image)},  # missing target
        "not-a-json-object",  # invalid row type
        {"image_path": str(image)},  # missing target
    ]
    with jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    with pytest.raises(ValueError) as excinfo:
        load_jsonl_records(jsonl, dataset_id="x", max_errors_to_report=2)
    message = str(excinfo.value)
    assert "Failed to parse 3 row(s)" in message
    assert "Examples:" in message
    assert "L2:" in message
    assert "L3:" in message


def test_jsonl_source_supports_multi_paths(tmp_path: Path) -> None:
    image = tmp_path / "img.png"
    Image.new("RGB", (8, 8), color=(0, 0, 0)).save(image)
    train_a = tmp_path / "train_a.jsonl"
    train_b = tmp_path / "train_b.jsonl"
    val_a = tmp_path / "val_a.jsonl"
    for path, sample_id in ((train_a, "a"), (train_b, "b"), (val_a, "v")):
        row = {
            "image_path": str(image),
            "sample_id": sample_id,
            "target_text": "{\"ok\":1}",
            "user_prompt": "return json",
        }
        path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    source = build_data_source(
        DataSourceConfig(
            name="demo",
            train_paths=[str(train_a), str(train_b)],
            val_paths=[str(val_a)],
        )
    )
    train_records = source.load_split("train")
    val_records = source.load_split("val")
    assert [record.sample_id for record in train_records] == ["a", "b"]
    assert [record.sample_id for record in val_records] == ["v"]


def test_jsonl_source_registered() -> None:
    assert DATA_SOURCE_REGISTRY.has("jsonl_sft")
    assert DATA_SOURCE_REGISTRY.has("jsonl_dpo")
    assert DATA_SOURCE_REGISTRY.has("jsonl_ppo")


def test_dpo_jsonl_source_parses_pairwise_fields(tmp_path: Path) -> None:
    image = tmp_path / "img.png"
    Image.new("RGB", (8, 8), color=(0, 0, 0)).save(image)
    jsonl = tmp_path / "dpo.jsonl"
    row = {
        "image_path": str(image),
        "chosen": "{\"ok\":1}",
        "rejected": "{\"ok\":0}",
    }
    jsonl.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    records = load_jsonl_dpo_records(jsonl, dataset_id="dpo_ds")
    assert len(records) == 1
    assert records[0].chosen_text == "{\"ok\":1}"
    assert records[0].rejected_text == "{\"ok\":0}"


def test_ppo_jsonl_source_parses_prompt_fields(tmp_path: Path) -> None:
    image = tmp_path / "img.png"
    Image.new("RGB", (8, 8), color=(0, 0, 0)).save(image)
    jsonl = tmp_path / "ppo.jsonl"
    row = {
        "image_path": str(image),
        "prompt": "detect objects",
    }
    jsonl.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    records = load_jsonl_ppo_records(jsonl, dataset_id="ppo_ds")
    assert len(records) == 1
    assert records[0].user_prompt == "detect objects"
