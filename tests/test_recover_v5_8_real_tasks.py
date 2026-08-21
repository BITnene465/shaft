from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

from PIL import Image
import pytest

from shaft.config import (
    DatasetSourceConfig,
    PromptSourceConfig,
    PromptSourceFormulationSourceConfig,
    RuntimeConfig,
)
from shaft.data import SFTDataset, ShaftDataCenter


def _load_module():
    script = Path("scripts/tasks/recover_v5_8_real_tasks.py").resolve()
    spec = importlib.util.spec_from_file_location("recover_v5_8_real_tasks", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _write_image(path: Path, *, size: tuple[int, int] = (12, 8)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(20, 30, 40)).save(path)
    return path


def _write_manifest(path: Path, items: list[dict[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"items": items}) + "\n", encoding="utf-8")
    return path


def _fixture(tmp_path: Path) -> argparse.Namespace:
    source = tmp_path / "source"
    output = tmp_path / "output"
    background_rows = []
    background_sft = []
    annotations = []
    for sample_id, background in (("a", True), ("b", False)):
        relative = f"../images/train/00/{sample_id}.png"
        _write_image(source / "background/images/train/00" / f"{sample_id}.png")
        structured = {
            "sample_id": sample_id,
            "image_path": relative,
            "image_width": 12,
            "image_height": 8,
            "background": background,
            "extra": {"task": "background"},
        }
        background_rows.append(structured)
        background_sft.append(
            {
                "image_path": relative,
                "sample_id": sample_id,
                "dataset_name": "background",
                "system_prompt": "legacy",
                "user_prompt": "legacy",
                "target_text": json.dumps(
                    {"background": background}, separators=(",", ":")
                ),
                "extra": {"prompt_id": "legacy"},
            }
        )
        annotations.append(
            {
                "id": sample_id,
                "image_path": f"{sample_id}.png",
                "background": background,
                "background_level": 4 if background else 0,
            }
        )
    annotations.append(
        {
            "id": "test",
            "image_path": "test.png",
            "background": False,
            "background_level": 0,
        }
    )
    _write_jsonl(source / "background/structured/train.jsonl", background_rows)
    _write_jsonl(source / "background/sft/train.jsonl", background_sft)

    selection_rows = []
    image_rows = []
    image_sft = []
    for index, image_type in enumerate(("photo", "chart")):
        base_id = f"img__image_{index:04d}"
        sample_id = f"{base_id}__context_00"
        relative = f"../images/train/00/{sample_id}.png"
        _write_image(
            source / "image_context_reconstruction/images/train/00" / f"{sample_id}.png"
        )
        selection_rows.append(
            {
                "sample_id": base_id,
                "instances": [{"label": "image", "bbox": [1, 1, 10, 7]}],
                "extra": {
                    "source_json": f"json/source_{index}.json",
                    "source_image": f"images/source_{index}.png",
                    "source_instance_index": index,
                    "source_bbox": [1, 1, 10, 7],
                },
            }
        )
        structured = {
            "sample_id": sample_id,
            "image_path": relative,
            "image_width": 12,
            "image_height": 8,
            "instances": [
                {
                    "label": "image",
                    "bbox": [1, 1, 10, 7],
                    "parameters": {"image_type": image_type},
                }
            ],
            "extra": {
                "source_json": f"json/source_{index}.json",
                "source_instance_index": index,
                "proposal_bbox_2d": [10, 20, 800, 900],
            },
        }
        image_rows.append(structured)
        image_sft.append(
            {
                "image_path": relative,
                "sample_id": sample_id,
                "dataset_name": "image_context_reconstruction",
                "system_prompt": "",
                "user_prompt": "",
                "prompt_args": {"proposal_bbox_2d": [10, 20, 800, 900]},
                "target_text": json.dumps(
                    {"type": "image", "parameters": {"image_type": image_type}},
                    separators=(",", ":"),
                ),
                "extra": {"prompt_pool_id": "legacy"},
            }
        )
    _write_jsonl(
        source / "image_context_reconstruction/selection/train.jsonl", selection_rows
    )
    _write_jsonl(
        source / "image_context_reconstruction/structured/train.jsonl", image_rows
    )
    _write_jsonl(source / "image_context_reconstruction/sft/train.jsonl", image_sft)

    annotations_path = _write_jsonl(tmp_path / "annotations.jsonl", annotations)
    main_test = _write_manifest(
        tmp_path / "splits/main.test.json", [{"id": "test", "image_path": "test.png"}]
    )
    inpainting_test = _write_manifest(tmp_path / "splits/inpainting.test.json", [])
    vlm_test = _write_manifest(
        tmp_path / "splits/vlm.test.json",
        [
            {"id": f"canonical_{index:03d}", "image_path": f"canonical_{index:03d}.png"}
            for index in range(175)
        ],
    )
    return argparse.Namespace(
        source_bundle_root=str(source),
        background_annotations=str(annotations_path),
        test_manifests=[str(main_test), str(inpainting_test), str(vlm_test)],
        canonical_test_manifest=str(vlm_test),
        canonical_test_sha256=hashlib.sha256(vlm_test.read_bytes()).hexdigest(),
        current_test_manifests=[],
        output_root=str(output),
        background_prompt_pool="configs/prompts/pools/background.v5.8.yaml",
        image_prompt_pool="configs/prompts/pools/image_context_reconstruction.v5.8.yaml",
        workers=2,
        clean=False,
    )


def test_recover_v5_8_real_tasks_builds_promptsource_ready_outputs(tmp_path: Path) -> None:
    module = _load_module()
    args = _fixture(tmp_path)

    summary = module.build(args)

    output = Path(args.output_root)
    background = [
        json.loads(line)
        for line in (output / "background/sft/train.jsonl").read_text().splitlines()
    ]
    image = [
        json.loads(line)
        for line in (
            output
            / "image_context_reconstruction/sft/formulations/image_type/train.jsonl"
        )
        .read_text()
        .splitlines()
    ]
    assert summary["status"] == "passed"
    assert summary["background"]["train_rows"] == 2
    assert summary["image_context_reconstruction"]["train_rows"] == 2
    assert summary["canonical_test_gate"]["rows"] == 175
    assert summary["canonical_test_gate"]["overlap"] == 0
    assert set(summary["source_artifact_sha256"]) == {
        "background/sft/train.jsonl",
        "background/structured/train.jsonl",
        "image_context_reconstruction/selection/train.jsonl",
        "image_context_reconstruction/sft/train.jsonl",
        "image_context_reconstruction/structured/train.jsonl",
    }
    assert len(summary["background"]["prompt_pool_sha256"]) == 64
    assert len(summary["image_context_reconstruction"]["prompt_pool_sha256"]) == 64
    assert all(row["system_prompt"] == row["user_prompt"] == "" for row in background)
    assert all(row["prompt_args"] == {} for row in background)
    assert {json.loads(row["target_text"])["background"] for row in background} == {
        True,
        False,
    }
    assert all(row["system_prompt"] == row["user_prompt"] == "" for row in image)
    assert all(row["image_path"].startswith("../../../images/train/") for row in image)
    assert {
        json.loads(row["target_text"])["parameters"]["image_type"] for row in image
    } == {"photo", "chart"}
    assert (
        output
        / "raw/imports/banana_v5_3_replay_20260722/annotations.jsonl"
    ).is_file()


def test_recovery_failure_keeps_existing_output_when_cleaning(tmp_path: Path) -> None:
    module = _load_module()
    args = _fixture(tmp_path)
    output = Path(args.output_root)
    sentinel = output / "background/sentinel.txt"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text("keep\n", encoding="utf-8")
    args.clean = True
    source_sft = Path(args.source_bundle_root) / "background/sft/train.jsonl"
    rows = [json.loads(line) for line in source_sft.read_text().splitlines()]
    rows[0]["target_text"] = '{"background":false}'
    _write_jsonl(source_sft, rows)

    with pytest.raises(ValueError, match="Background structured/SFT mismatch"):
        module.build(args)

    assert sentinel.read_text(encoding="utf-8") == "keep\n"


def test_recovery_rejects_noncanonical_test_manifest(tmp_path: Path) -> None:
    module = _load_module()
    args = _fixture(tmp_path)
    canonical = Path(args.canonical_test_manifest)
    _write_manifest(canonical, [{"id": "only-one", "image_path": "only-one.png"}])
    args.canonical_test_sha256 = hashlib.sha256(canonical.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="must contain exactly 175 items"):
        module.build(args)


def test_recovered_outputs_run_through_promptsource_data_center(tmp_path: Path) -> None:
    module = _load_module()
    args = _fixture(tmp_path)
    module.build(args)
    output = Path(args.output_root)

    background_config = RuntimeConfig()
    background_config.data.media_snapshot_id = "recovery-test-v1"
    background_config.data.schedule.mixing = "concat"
    background_config.data.schedule.shuffle = False
    background_config.data.datasets = [
        DatasetSourceConfig(
            dataset_name="background",
            train_path=str(output / "background/sft/train.jsonl"),
            use_for_eval=False,
        )
    ]
    background_config.data.prompt_sources = {
        "background": PromptSourceConfig(
            path="configs/prompts/pools/background.v5.8.yaml"
        )
    }
    background_bundle = ShaftDataCenter(
        background_config.data, seed=42
    ).build_dataset_bundle(SFTDataset)
    assert background_bundle.train_sampler is not None
    background_ref = next(iter(background_bundle.train_sampler))
    background_sample = background_bundle.train_dataset[background_ref]
    assert background_sample["extra"]["prompt_source"]["variant_id"] == "detailed"
    assert json.loads(background_sample["target_text"]) == {"background": True}

    image_config = RuntimeConfig()
    image_config.data.media_snapshot_id = "recovery-test-v1"
    image_config.data.schedule.mixing = "concat"
    image_config.data.schedule.shuffle = False
    image_config.data.datasets = [
        DatasetSourceConfig(
            dataset_name="image_context_reconstruction",
            use_for_eval=False,
        )
    ]
    image_config.data.prompt_sources = {
        "image_context_reconstruction": PromptSourceConfig(
            path="configs/prompts/pools/image_context_reconstruction.v5.8.yaml",
            formulation_sources={
                "image_type": PromptSourceFormulationSourceConfig(
                    train_path=str(
                        output
                        / "image_context_reconstruction/sft/formulations/image_type/train.jsonl"
                    )
                )
            },
        )
    }
    image_bundle = ShaftDataCenter(image_config.data, seed=42).build_dataset_bundle(SFTDataset)
    assert image_bundle.train_sampler is not None
    image_ref = next(iter(image_bundle.train_sampler))
    planned = image_bundle.train_dataset.get_planning_item(image_ref)
    image_sample = image_bundle.train_dataset[image_ref]
    assert planned["target_text"] == image_sample["target_text"]
    assert image_sample["extra"]["prompt_source"]["formulation_id"] == "image_type"
    assert image_sample["extra"]["prompt_source"]["variant_id"] in {
        "detailed",
        "concise",
    }
    assert json.loads(image_sample["target_text"])["parameters"]["image_type"] == "photo"
