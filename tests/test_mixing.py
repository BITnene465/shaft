from __future__ import annotations

from shaft.data import MixedDatasetBuilder, SFTRecord, ShaftMixedIndexSampler


def _records(dataset_name: str, n: int):
    return [
        SFTRecord(image_path=f"/tmp/{dataset_name}_{i}.png", target_text="{}", dataset_name=dataset_name)
        for i in range(n)
    ]


def test_concat_strategy() -> None:
    builder = MixedDatasetBuilder(seed=1)
    indices = builder.build_indices(
        {"a": _records("a", 3), "b": _records("b", 2)},
        {"a": 1.0, "b": 1.0},
        strategy="concat",
        shuffle=False,
    )
    assert len(indices) == 5
    assert indices[0][0] == "a"
    assert indices[-1][0] == "b"


def test_interleave_under_respects_shorter_dataset() -> None:
    builder = MixedDatasetBuilder(seed=1)
    indices = builder.build_indices(
        {"a": _records("a", 10), "b": _records("b", 2)},
        {"a": 1.0, "b": 1.0},
        strategy="interleave_under",
        shuffle=False,
    )
    by_dataset = {"a": 0, "b": 0}
    for dataset_name, _ in indices:
        by_dataset[dataset_name] += 1
    assert by_dataset["b"] == 2
    assert by_dataset["a"] >= 2


def test_weight_zero_disables_dataset() -> None:
    builder = MixedDatasetBuilder(seed=1)
    indices = builder.build_indices(
        {"a": _records("a", 3), "b": _records("b", 3)},
        {"a": 1.0, "b": 0.0},
        strategy="interleave_over",
        shuffle=False,
    )
    assert all(dataset_name == "a" for dataset_name, _ in indices)


def test_mixed_index_sampler_static_does_not_refresh_twice() -> None:
    sampler = ShaftMixedIndexSampler(
        {"a": _records("a", 3), "b": _records("b", 2)},
        {"a": 1.0, "b": 1.0},
        strategy="concat",
        refresh_mode="static",
        shuffle=False,
        seed=7,
    )

    first_ids = [sample_id for sample_id in (f"{dataset_name}_{row_index}" for dataset_name, row_index in sampler.current_indices)]
    assert len(sampler) == 5
    assert sampler.epoch == 0
    assert sampler.refresh_count == 1
    sampler.set_epoch(1)
    second_ids = [sample_id for sample_id in (f"{dataset_name}_{row_index}" for dataset_name, row_index in sampler.current_indices)]
    assert sampler.epoch == 1
    assert sampler.refresh_count == 1
    assert first_ids == second_ids


def test_mixed_index_sampler_epoch_refresh_rebuilds_each_epoch() -> None:
    sampler = ShaftMixedIndexSampler(
        {"a": _records("a", 4), "b": _records("b", 4)},
        {"a": 1.0, "b": 1.0},
        strategy="concat",
        refresh_mode="epoch_refresh",
        shuffle=True,
        seed=7,
    )

    first_indices = list(sampler.current_indices)
    assert sampler.epoch == 0
    assert sampler.refresh_count == 1

    sampler.set_epoch(1)
    second_indices = list(sampler.current_indices)
    assert sampler.epoch == 1
    assert sampler.refresh_count == 2
    assert first_indices != second_indices


def test_mixed_index_sampler_shards_for_distributed() -> None:
    sampler_rank0 = ShaftMixedIndexSampler(
        {"a": _records("a", 3), "b": _records("b", 1)},
        {"a": 1.0, "b": 1.0},
        strategy="concat",
        refresh_mode="static",
        shuffle=False,
        seed=3,
        rank=0,
        world_size=2,
    )
    sampler_rank1 = ShaftMixedIndexSampler(
        {"a": _records("a", 3), "b": _records("b", 1)},
        {"a": 1.0, "b": 1.0},
        strategy="concat",
        refresh_mode="static",
        shuffle=False,
        seed=3,
        rank=1,
        world_size=2,
    )

    assert len(sampler_rank0) == len(sampler_rank1) == 2
    assert sampler_rank0.current_indices != sampler_rank1.current_indices
