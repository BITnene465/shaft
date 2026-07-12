from __future__ import annotations

import json
from pathlib import Path
import warnings

from PIL import Image
import pytest
from torch.utils.data import DataLoader

from shaft.config import DatasetSourceConfig, PromptSamplingConfig, RuntimeConfig
from shaft.data import (
    DPODataset,
    SFTDataset,
    SFTRecord,
    ShaftDataCenter,
    ShaftSampleSampler,
)
from shaft.data.transforms import ONLINE_TRANSFORM_REGISTRY

_MARK_DATASET_TRANSFORM = "mark_dataset_for_data_center_tests"


def _first_sample(batch):
    return batch[0]


if not ONLINE_TRANSFORM_REGISTRY.has(_MARK_DATASET_TRANSFORM):

    @ONLINE_TRANSFORM_REGISTRY.register(_MARK_DATASET_TRANSFORM)
    def _mark_dataset_transform(sample: dict[str, object]) -> dict[str, object]:
        updated = dict(sample)
        extra = dict(updated.get("extra", {}))
        extra["marked_dataset"] = updated.get("dataset_name")
        updated["extra"] = extra
        return updated


def _write_image(path: Path) -> Path:
    Image.new("RGB", (8, 8), color=(0, 0, 0)).save(path)
    return path


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def _write_prompt_pool(
    path: Path,
    *,
    pool_id: str,
    version: str = "test-version",
    prompts: list[tuple[str, str, str]],
) -> Path:
    prompt_lines = []
    for variant_id, system_prompt, user_prompt in prompts:
        prompt_lines.extend(
            [
                f"  - id: {variant_id}",
                f"    system_prompt: {system_prompt!r}",
                f"    user_prompt: {user_prompt!r}",
            ]
        )
    path.write_text(
        "\n".join(
            [
                "metadata:",
                f"  id: {pool_id}",
                f"  version: {version}",
                "prompts:",
                *prompt_lines,
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_data_center_builds_sft_dataset_pair(tmp_path: Path) -> None:
    image = _write_image(tmp_path / "img.png")
    train_a = _write_jsonl(
        tmp_path / "train_a.jsonl",
        [
            {"image_path": str(image), "target_text": "{\"a\":1}", "sample_id": "a1"},
            {"image_path": str(image), "target_text": "{\"a\":2}", "sample_id": "a2"},
        ],
    )
    train_b = _write_jsonl(
        tmp_path / "train_b.jsonl",
        [{"image_path": str(image), "target_text": "{\"b\":1}", "sample_id": "b1"}],
    )
    val_a = _write_jsonl(
        tmp_path / "val_a.jsonl",
        [{"image_path": str(image), "target_text": "{\"va\":1}", "sample_id": "va1"}],
    )
    val_b = _write_jsonl(
        tmp_path / "val_b.jsonl",
        [{"image_path": str(image), "target_text": "{\"vb\":1}", "sample_id": "vb1"}],
    )

    config = RuntimeConfig()
    config.experiment.seed = 7
    config.data.mix_strategy = "concat"
    config.data.shuffle = False
    config.data.datasets = [
        DatasetSourceConfig(
            dataset_name="ds_a",
            train_path=str(train_a),
            val_path=str(val_a),
            online_transforms=[_MARK_DATASET_TRANSFORM],
        ),
        DatasetSourceConfig(
            dataset_name="ds_b",
            train_path=str(train_b),
            val_path=str(val_b),
        ),
    ]

    center = ShaftDataCenter(config.data, seed=config.experiment.seed)
    dataset_bundle = center.build_dataset_bundle(SFTDataset)
    train_dataset = dataset_bundle.train_dataset
    val_dataset = dataset_bundle.eval_dataset

    assert len(train_dataset) == 3
    assert len(val_dataset) == 2
    assert dataset_bundle.eval_datasets_by_name is not None
    assert set(dataset_bundle.eval_datasets_by_name.keys()) == {"ds_a", "ds_b"}
    assert len(dataset_bundle.eval_datasets_by_name["ds_a"]) == 1
    assert len(dataset_bundle.eval_datasets_by_name["ds_b"]) == 1
    sample_a = train_dataset[0]
    sample_b = train_dataset[2]
    assert sample_a["dataset_name"] == "ds_a"
    assert sample_a["extra"]["marked_dataset"] == "ds_a"
    assert sample_b["dataset_name"] == "ds_b"
    assert "marked_dataset" not in sample_b["extra"]
    assert isinstance(train_dataset.records, dict)
    assert isinstance(dataset_bundle.train_sampler, ShaftSampleSampler)


def test_bounded_data_center_enables_worker_warning_suppression_only_for_train(
    tmp_path: Path,
) -> None:
    image = _write_image(tmp_path / "img.png")
    train_path = _write_jsonl(
        tmp_path / "train.jsonl",
        [{"image_path": str(image), "target_text": "train"}],
    )
    val_path = _write_jsonl(
        tmp_path / "val.jsonl",
        [{"image_path": str(image), "target_text": "val"}],
    )
    config = RuntimeConfig()
    config.data.batching.strategy = "bounded_cost_aware"
    config.data.media_snapshot_id = "fixture-media-v1"
    config.data.datasets = [
        DatasetSourceConfig(
            dataset_name="fixture",
            train_path=str(train_path),
            val_path=str(val_path),
        )
    ]

    bundle = ShaftDataCenter(config.data).build_dataset_bundle(SFTDataset)

    assert bundle.train_dataset.suppress_decompression_bomb_warning is True
    assert bundle.eval_dataset.suppress_decompression_bomb_warning is False


def test_bounded_dataset_suppresses_pil_bomb_warning_but_not_hard_error(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image_path = tmp_path / "large-for-threshold.png"
    Image.new("RGB", (10, 10), color=(0, 0, 0)).save(image_path)
    record = SFTRecord(image_path=str(image_path), target_text="target")
    bounded = SFTDataset(
        [record],
        suppress_decompression_bomb_warning=True,
    )

    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 60)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bounded[0]
    assert not any(
        issubclass(item.category, Image.DecompressionBombWarning)
        for item in caught
    )

    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 40)
    with pytest.raises(Image.DecompressionBombError):
        bounded[0]


def test_data_center_prompt_sampling_applies_only_to_train(tmp_path: Path) -> None:
    image = _write_image(tmp_path / "img.png")
    train_path = _write_jsonl(
        tmp_path / "train.jsonl",
        [
            {
                "image_path": str(image),
                "target_text": "{\"a\":1}",
                "sample_id": "sample-1",
                "system_prompt": "canonical system",
                "user_prompt": "canonical user",
            }
        ],
    )
    val_path = _write_jsonl(
        tmp_path / "val.jsonl",
        [
            {
                "image_path": str(image),
                "target_text": "{\"v\":1}",
                "sample_id": "val-1",
                "system_prompt": "canonical system",
                "user_prompt": "canonical user",
            }
        ],
    )
    prompt_pool = _write_prompt_pool(
        tmp_path / "prompt_pool.yaml",
        pool_id="prompt.pool",
        version="test-version",
        prompts=[
            ("a", "system a", "user a"),
            ("b", "system b", "user b"),
        ],
    )

    config = RuntimeConfig()
    config.experiment.seed = 11
    config.data.mix_strategy = "concat"
    config.data.shuffle = False
    config.data.prompt_sampling = PromptSamplingConfig(
        enabled=True,
        train_only=True,
        seed=99,
        pools={"ds": str(prompt_pool)},
    )
    config.data.datasets = [
        DatasetSourceConfig(dataset_name="ds", train_path=str(train_path), val_path=str(val_path))
    ]

    center = ShaftDataCenter(config.data, seed=config.experiment.seed)
    dataset_bundle = center.build_dataset_bundle(SFTDataset)

    train_sample = dataset_bundle.train_dataset[0]
    assert train_sample["user_prompt"] in {"user a", "user b"}
    assert train_sample["system_prompt"] in {"system a", "system b"}
    assert train_sample["extra"]["runtime_prompt_id"] in {"prompt.pool.a", "prompt.pool.b"}
    assert train_sample["extra"]["runtime_prompt_version"] == "test-version"
    assert train_sample["extra"]["runtime_prompt_draw_id"] == 0

    assert dataset_bundle.train_sampler is not None
    dataset_bundle.train_sampler.set_epoch(3)
    refreshed_ref = next(iter(dataset_bundle.train_sampler))
    refreshed_sample = dataset_bundle.train_dataset[refreshed_ref]
    assert refreshed_sample["extra"]["runtime_prompt_draw_id"] == 3

    val_sample = dataset_bundle.eval_dataset[0]
    assert val_sample["user_prompt"] == "canonical user"
    assert val_sample["system_prompt"] == "canonical system"
    assert "runtime_prompt_id" not in val_sample["extra"]


def test_data_center_builds_dpo_dataset_pair(tmp_path: Path) -> None:
    image = _write_image(tmp_path / "img.png")
    train_path = _write_jsonl(
        tmp_path / "train_dpo.jsonl",
        [
            {
                "image_path": str(image),
                "chosen_text": "{\"ok\":1}",
                "rejected_text": "{\"ok\":0}",
                "sample_id": "d1",
            }
        ],
    )
    val_path = _write_jsonl(
        tmp_path / "val_dpo.jsonl",
        [
            {
                "image_path": str(image),
                "chosen_text": "{\"ok\":2}",
                "rejected_text": "{\"ok\":1}",
                "sample_id": "d2",
            }
        ],
    )

    config = RuntimeConfig()
    config.algorithm.name = "dpo"
    config.data.datasets = [
        DatasetSourceConfig(
            dataset_name="dpo_ds",
            source_type="jsonl_dpo",
            train_path=str(train_path),
            val_path=str(val_path),
        )
    ]

    center = ShaftDataCenter(config.data, seed=config.experiment.seed)
    dataset_bundle = center.build_dataset_bundle(DPODataset)
    train_dataset = dataset_bundle.train_dataset
    val_dataset = dataset_bundle.eval_dataset

    assert len(train_dataset) == 1
    assert len(val_dataset) == 1
    sample = train_dataset[0]
    assert sample["dataset_name"] == "dpo_ds"
    assert sample["chosen_text"] == "{\"ok\":1}"
    assert sample["rejected_text"] == "{\"ok\":0}"


def test_data_center_skips_val_for_train_only_dataset(tmp_path: Path) -> None:
    image = _write_image(tmp_path / "img.png")
    train_a = _write_jsonl(
        tmp_path / "train_a.jsonl",
        [{"image_path": str(image), "target_text": "{\"a\":1}", "sample_id": "a1"}],
    )
    train_b = _write_jsonl(
        tmp_path / "train_b.jsonl",
        [{"image_path": str(image), "target_text": "{\"b\":1}", "sample_id": "b1"}],
    )
    val_a = _write_jsonl(
        tmp_path / "val_a.jsonl",
        [{"image_path": str(image), "target_text": "{\"va\":1}", "sample_id": "va1"}],
    )

    config = RuntimeConfig()
    config.data.mix_strategy = "concat"
    config.data.shuffle = False
    config.data.datasets = [
        DatasetSourceConfig(
            dataset_name="eval_ds",
            train_path=str(train_a),
            val_path=str(val_a),
            use_for_eval=True,
        ),
        DatasetSourceConfig(
            dataset_name="train_only_ds",
            train_path=str(train_b),
            use_for_eval=False,
        ),
    ]

    center = ShaftDataCenter(config.data, seed=config.experiment.seed)
    dataset_bundle = center.build_dataset_bundle(SFTDataset)
    train_dataset = dataset_bundle.train_dataset
    val_dataset = dataset_bundle.eval_dataset

    assert len(train_dataset) == 2
    assert len(val_dataset) == 1
    assert dataset_bundle.eval_datasets_by_name is not None
    assert set(dataset_bundle.eval_datasets_by_name.keys()) == {"eval_ds"}
    assert train_dataset[0]["dataset_name"] == "eval_ds"
    assert train_dataset[1]["dataset_name"] == "train_only_ds"
    assert val_dataset[0]["dataset_name"] == "eval_ds"


def test_data_center_sampler_passes_plan_cycle_in_sample_ref(tmp_path: Path) -> None:
    image = _write_image(tmp_path / "img.png")
    train_a = _write_jsonl(
        tmp_path / "train_a.jsonl",
        [{"image_path": str(image), "target_text": "{\"a\":1}", "sample_id": f"a{i}"} for i in range(4)],
    )
    train_b = _write_jsonl(
        tmp_path / "train_b.jsonl",
        [{"image_path": str(image), "target_text": "{\"b\":1}", "sample_id": f"b{i}"} for i in range(4)],
    )
    val_a = _write_jsonl(
        tmp_path / "val_a.jsonl",
        [{"image_path": str(image), "target_text": "{\"va\":1}", "sample_id": "va1"}],
    )
    val_b = _write_jsonl(
        tmp_path / "val_b.jsonl",
        [{"image_path": str(image), "target_text": "{\"vb\":1}", "sample_id": "vb1"}],
    )

    config = RuntimeConfig()
    config.experiment.seed = 5
    config.data.mix_strategy = "concat"
    config.data.shuffle = True
    config.data.datasets = [
        DatasetSourceConfig(dataset_name="ds_a", train_path=str(train_a), val_path=str(val_a)),
        DatasetSourceConfig(dataset_name="ds_b", train_path=str(train_b), val_path=str(val_b)),
    ]

    center = ShaftDataCenter(config.data, seed=config.experiment.seed)
    dataset_bundle = center.build_dataset_bundle(SFTDataset)
    train_dataset = dataset_bundle.train_dataset

    assert isinstance(dataset_bundle.train_sampler, ShaftSampleSampler)
    first = list(dataset_bundle.train_sampler)
    first_sample = train_dataset[first[0]]
    dataset_bundle.train_sampler.set_epoch(1)
    second = list(dataset_bundle.train_sampler)
    second_sample = train_dataset[second[0]]
    assert first != second
    assert first_sample["_sample_context"]["draw_id"] == 0
    assert second_sample["_sample_context"]["draw_id"] == len(train_dataset)


def test_sample_context_crosses_persistent_worker_boundary(tmp_path: Path) -> None:
    image = _write_image(tmp_path / "img.png")
    train_path = _write_jsonl(
        tmp_path / "train.jsonl",
        [
            {"image_path": str(image), "target_text": "{}", "sample_id": f"s{i}"}
            for i in range(2)
        ],
    )
    val_path = _write_jsonl(
        tmp_path / "val.jsonl",
        [{"image_path": str(image), "target_text": "{}", "sample_id": "v"}],
    )
    config = RuntimeConfig()
    config.data.mix_strategy = "concat"
    config.data.shuffle = False
    config.data.datasets = [
        DatasetSourceConfig(dataset_name="ds", train_path=str(train_path), val_path=str(val_path))
    ]
    bundle = ShaftDataCenter(config.data, seed=13).build_dataset_bundle(SFTDataset)
    assert bundle.train_sampler is not None
    loader = DataLoader(
        bundle.train_dataset,
        batch_size=1,
        sampler=bundle.train_sampler,
        num_workers=1,
        persistent_workers=True,
        collate_fn=_first_sample,
    )

    first = next(iter(loader))
    bundle.train_sampler.set_epoch(1)
    second = next(iter(loader))

    assert first["_sample_context"]["draw_id"] == 0
    assert second["_sample_context"]["draw_id"] == len(bundle.train_dataset)


def test_data_center_builds_unsharded_train_sampler_for_hf_trainer(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("RANK", "3")
    image = _write_image(tmp_path / "img.png")
    train_a = _write_jsonl(
        tmp_path / "train_a.jsonl",
        [{"image_path": str(image), "target_text": "{\"a\":1}", "sample_id": f"a{i}"} for i in range(8)],
    )
    train_b = _write_jsonl(
        tmp_path / "train_b.jsonl",
        [{"image_path": str(image), "target_text": "{\"b\":1}", "sample_id": f"b{i}"} for i in range(8)],
    )
    val_a = _write_jsonl(
        tmp_path / "val_a.jsonl",
        [{"image_path": str(image), "target_text": "{\"va\":1}", "sample_id": "va1"}],
    )
    val_b = _write_jsonl(
        tmp_path / "val_b.jsonl",
        [{"image_path": str(image), "target_text": "{\"vb\":1}", "sample_id": "vb1"}],
    )

    config = RuntimeConfig()
    config.experiment.seed = 5
    config.data.mix_strategy = "concat"
    config.data.shuffle = False
    config.data.datasets = [
        DatasetSourceConfig(dataset_name="ds_a", train_path=str(train_a), val_path=str(val_a)),
        DatasetSourceConfig(dataset_name="ds_b", train_path=str(train_b), val_path=str(val_b)),
    ]

    center = ShaftDataCenter(config.data, seed=config.experiment.seed)
    dataset_bundle = center.build_dataset_bundle(SFTDataset)

    assert len(dataset_bundle.train_sampler) == 16
    assert len(dataset_bundle.train_dataset) == 16
    assert dataset_bundle.train_sampler.rank == 0
    assert dataset_bundle.train_sampler.world_size == 1


def test_data_center_applies_step_sample_budget_to_plan(tmp_path: Path) -> None:
    image = _write_image(tmp_path / "img.png")
    train_path = _write_jsonl(
        tmp_path / "train.jsonl",
        [{"image_path": str(image), "target_text": "{}", "sample_id": "s"}],
    )
    val_path = _write_jsonl(
        tmp_path / "val.jsonl",
        [{"image_path": str(image), "target_text": "{}", "sample_id": "v"}],
    )
    config = RuntimeConfig()
    config.data.datasets = [
        DatasetSourceConfig(dataset_name="ds", train_path=str(train_path), val_path=str(val_path))
    ]

    bundle = ShaftDataCenter(
        config.data,
        seed=3,
        train_sample_budget=17,
    ).build_dataset_bundle(SFTDataset)

    assert len(bundle.train_dataset) == 17
    assert bundle.train_sampler is not None
    assert len(bundle.train_sampler) == 17


def test_bounded_data_center_exposes_schedule_without_duration_sized_plan(
    tmp_path: Path,
) -> None:
    image = _write_image(tmp_path / "img.png")
    train_path = _write_jsonl(
        tmp_path / "train.jsonl",
        [{"image_path": str(image), "target_text": "{}", "sample_id": "s"}],
    )
    val_path = _write_jsonl(
        tmp_path / "val.jsonl",
        [{"image_path": str(image), "target_text": "{}", "sample_id": "v"}],
    )
    config = RuntimeConfig()
    config.data.batching.strategy = "bounded_cost_aware"
    config.data.media_snapshot_id = "data-center-fixture-v1"
    config.data.mix_strategy = "concat"
    config.data.shuffle = False
    config.data.datasets = [
        DatasetSourceConfig(
            dataset_name="ds",
            train_path=str(train_path),
            val_path=str(val_path),
        )
    ]

    bundle = ShaftDataCenter(
        config.data,
        seed=3,
        train_sample_budget=1_000_000,
    ).build_dataset_bundle(SFTDataset)

    assert bundle.train_sampler is None
    assert bundle.train_schedule is not None
    assert bundle.train_dataset.sample_plan is None
    assert bundle.train_dataset.sample_schedule is bundle.train_schedule
    assert bundle.train_dataset.media_snapshot_id == "data-center-fixture-v1"
    ref = bundle.train_schedule.ref_at(999_999)
    assert bundle.train_dataset.get_planning_item(ref)["sample_id"] == "s"
