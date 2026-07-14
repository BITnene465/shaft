from __future__ import annotations

import json
from pathlib import Path
import pickle

import pytest
from PIL import Image

from shaft.config import DatasetSourceConfig
from shaft.data import PPODataset, PPORecord, ShaftArrowRecordStore, ShaftDatasetMeta
from shaft.data.sources import (
    DATA_SOURCE_REGISTRY,
    build_data_source,
    load_jsonl_dpo_records,
    load_jsonl_ppo_records,
    load_jsonl_sft_records,
)


def test_new_message_format_extracts_target_and_drops_tail_assistant(tmp_path: Path) -> None:
    image = tmp_path / "img.png"
    Image.new("RGB", (8, 8), color=(0, 0, 0)).save(image)
    jsonl = tmp_path / "samples.jsonl"
    sample = {
        "image": "img.png",
        "dataset_name": "demo",
        "sample_id": "s1",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "detect"}]},
            {"role": "assistant", "content": "{\"ok\":1}"},
        ],
    }
    jsonl.write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")
    records = load_jsonl_sft_records(jsonl, dataset_name="fallback")
    assert len(records) == 1
    record = records[0]
    assert record.dataset_name == "fallback"
    assert record.extra["source_dataset_name"] == "demo"
    assert "sample_id" not in record.extra
    assert "dataset_name" not in record.extra
    assert record.sample_id == "s1"
    assert record.target_text == "{\"ok\":1}"
    assert Path(record.image_path).is_absolute()
    assert len(record.messages or []) == 2
    assert record.messages[-1]["role"] == "user"


def test_jsonl_loader_builds_and_reuses_memory_mapped_arrow_store(tmp_path: Path) -> None:
    image = tmp_path / "img.png"
    Image.new("RGB", (8, 8), color=(0, 0, 0)).save(image)
    jsonl = tmp_path / "samples.jsonl"
    jsonl.write_text(
        json.dumps({"image_path": str(image), "target_text": "{}"}) + "\n",
        encoding="utf-8",
    )
    cache_dir = tmp_path / "record-cache"

    first = load_jsonl_sft_records(jsonl, dataset_name="demo", cache_dir=cache_dir)
    second = load_jsonl_sft_records(jsonl, dataset_name="demo", cache_dir=cache_dir)

    assert isinstance(first, ShaftArrowRecordStore)
    assert first.cache_path == second.cache_path
    assert Path(first.cache_path).suffix == ".arrow"
    assert first[0].target_text == "{}"

    restored = pickle.loads(pickle.dumps(first))
    assert restored[0] == first[0]

    jsonl.write_text(
        "\n".join(
            [
                json.dumps({"image_path": str(image), "target_text": "{}"}),
                json.dumps({"image_path": str(image), "target_text": '{"new":1}'}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    refreshed = load_jsonl_sft_records(jsonl, dataset_name="demo", cache_dir=cache_dir)
    assert refreshed.cache_path != first.cache_path
    assert len(refreshed) == 2


def test_arrow_record_validator_requires_a_validation_fingerprint(
    tmp_path: Path,
) -> None:
    image = tmp_path / "img.png"
    Image.new("RGB", (8, 8), color=(0, 0, 0)).save(image)
    jsonl = tmp_path / "samples.jsonl"
    jsonl.write_text(
        json.dumps({"image_path": str(image), "target_text": "{}"}) + "\n",
        encoding="utf-8",
    )
    cache_dir = tmp_path / "record-cache"
    load_jsonl_sft_records(jsonl, dataset_name="demo", cache_dir=cache_dir)
    calls: list[str] = []

    with pytest.raises(ValueError, match="record_validator.*validation_fingerprint"):
        load_jsonl_sft_records(
            jsonl,
            dataset_name="demo",
            cache_dir=cache_dir,
            record_validator=lambda record: calls.append(record.target_text),
        )
    assert calls == []

    with pytest.raises(ValueError, match="record_validator.*validation_fingerprint"):
        load_jsonl_sft_records(
            jsonl,
            dataset_name="demo",
            cache_dir=cache_dir,
            validation_fingerprint="prompt-schema-v1",
        )

    validated = load_jsonl_sft_records(
        jsonl,
        dataset_name="demo",
        cache_dir=cache_dir,
        record_validator=lambda record: calls.append(record.target_text),
        validation_fingerprint="prompt-schema-v1",
    )
    assert calls == ["{}"]
    calls.clear()

    cached = load_jsonl_sft_records(
        jsonl,
        dataset_name="demo",
        cache_dir=cache_dir,
        record_validator=lambda record: calls.append(record.target_text),
        validation_fingerprint="prompt-schema-v1",
    )
    assert cached.cache_path == validated.cache_path
    assert calls == []

    revalidated = load_jsonl_sft_records(
        jsonl,
        dataset_name="demo",
        cache_dir=cache_dir,
        record_validator=lambda record: calls.append(record.target_text),
        validation_fingerprint="prompt-schema-v2",
    )
    assert revalidated.cache_path != validated.cache_path
    assert calls == ["{}"]


def test_sft_prompt_args_are_a_typed_record_field_and_survive_arrow_cache(
    tmp_path: Path,
) -> None:
    image = tmp_path / "img.png"
    Image.new("RGB", (8, 8), color=(0, 0, 0)).save(image)
    jsonl = tmp_path / "samples.jsonl"
    jsonl.write_text(
        json.dumps(
            {
                "image_path": str(image),
                "target_text": "{}",
                "prompt_args": {
                    "label": "shape",
                    "bbox": [1, 2, 30, 40],
                    "flag": True,
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    record = load_jsonl_sft_records(
        jsonl,
        dataset_name="demo",
        cache_dir=tmp_path / "cache",
    )[0]

    assert record.prompt_args == {
        "label": "shape",
        "bbox": [1, 2, 30, 40],
        "flag": True,
    }
    assert "prompt_args" not in record.extra


def test_sft_source_rejects_non_object_prompt_args(tmp_path: Path) -> None:
    image = tmp_path / "img.png"
    Image.new("RGB", (8, 8), color=(0, 0, 0)).save(image)
    jsonl = tmp_path / "bad-prompt-args.jsonl"
    jsonl.write_text(
        json.dumps(
            {
                "image_path": str(image),
                "target_text": "{}",
                "prompt_args": '{"label":"shape"}',
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="prompt_args.*JSON object"):
        load_jsonl_sft_records(jsonl, dataset_name="demo", cache_dir=tmp_path / "cache")


def test_empty_prompt_messages_can_use_prompt_args(tmp_path: Path) -> None:
    image = tmp_path / "img.png"
    Image.new("RGB", (8, 8), color=(0, 0, 0)).save(image)
    jsonl = tmp_path / "assistant-only.jsonl"
    jsonl.write_text(
        json.dumps(
            {
                "image_path": str(image),
                "messages": [{"role": "assistant", "content": "{}"}],
                "prompt_args": {"value": "x"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    record = load_jsonl_sft_records(jsonl, dataset_name="demo", cache_dir=tmp_path / "cache")[0]

    assert record.messages == []
    assert record.prompt_args == {"value": "x"}


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
        load_jsonl_sft_records(jsonl, dataset_name="x")


def test_jsonl_loader_reports_aggregated_errors(tmp_path: Path) -> None:
    image = tmp_path / "img.png"
    Image.new("RGB", (8, 8), color=(0, 0, 0)).save(image)
    jsonl = tmp_path / "bad_many.jsonl"
    rows = [
        {"image_path": str(image), "target_text": "{\"ok\":1}"},
        {"image_path": str(image)},
        "not-a-json-object",
        {"image_path": str(image)},
    ]
    with jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    with pytest.raises(ValueError) as excinfo:
        load_jsonl_sft_records(jsonl, dataset_name="x", max_errors_to_report=2)
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

    dataset_meta = ShaftDatasetMeta.from_config(
        DatasetSourceConfig(
            dataset_name="demo",
            train_paths=[str(train_a), str(train_b)],
            val_paths=[str(val_a)],
        )
    )
    source = build_data_source(dataset_meta)
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
    records = load_jsonl_dpo_records(jsonl, dataset_name="dpo_ds")
    assert len(records) == 1
    assert records[0].chosen_text == "{\"ok\":1}"
    assert records[0].rejected_text == "{\"ok\":0}"


def test_ppo_jsonl_source_parses_prompt_fields(tmp_path: Path) -> None:
    jsonl = tmp_path / "ppo.jsonl"
    row = {
        "prompt": "detect objects",
    }
    jsonl.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    records = load_jsonl_ppo_records(jsonl, dataset_name="ppo_ds")
    assert len(records) == 1
    assert records[0].user_prompt == "detect objects"
    assert records[0].image_path is None


def test_ppo_dataset_does_not_decode_unused_images(tmp_path: Path) -> None:
    missing_image = tmp_path / "does-not-exist.png"
    dataset = PPODataset(
        [PPORecord(image_path=str(missing_image), sample_id="p1", user_prompt="text only")]
    )

    sample = dataset[0]

    assert "image" not in sample
    assert sample["image_path"] == str(missing_image)
