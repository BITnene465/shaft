from __future__ import annotations

from shaft.data import MixedDatasetBuilder, SFTRecord


def _records(dataset_id: str, n: int):
    return [
        SFTRecord(image_path=f"/tmp/{dataset_id}_{i}.png", target_text="{}", dataset_id=dataset_id)
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
    for dataset_id, _ in indices:
        by_dataset[dataset_id] += 1
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
    assert all(dataset_id == "a" for dataset_id, _ in indices)
