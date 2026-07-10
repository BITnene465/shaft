from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from shaft.config import PromptSamplingConfig
from shaft.data import SFTRecord, build_offline_pipeline, build_online_pipeline
from shaft.data.transforms import build_prompt_sampling_transform


def test_offline_dedup() -> None:
    records = [
        SFTRecord(image_path="/tmp/a.png", target_text="{}", dataset_name="d"),
        SFTRecord(image_path="/tmp/a.png", target_text="{}", dataset_name="d"),
        SFTRecord(image_path="/tmp/a.png", target_text='{"x":1}', dataset_name="d"),
    ]
    pipeline = build_offline_pipeline(["dedup_image_target"])
    out = pipeline(records)
    assert len(out) == 2


def test_online_identity() -> None:
    pipeline = build_online_pipeline(["identity"])
    sample = {"x": 1}
    out = pipeline(sample)
    assert out["x"] == 1


def test_prompt_sampling_transform_records_version(tmp_path: Path) -> None:
    version = "test-version"
    prompt_pool = tmp_path / "pool.yaml"
    prompt_pool.write_text(
        "\n".join(
            [
                "metadata:",
                "  id: pool.test",
                f"  version: {version}",
                "prompts:",
                "  - id: main",
                "    system_prompt: sys",
                "    user_prompt: user",
                "",
            ]
        ),
        encoding="utf-8",
    )
    transform = build_prompt_sampling_transform(
        PromptSamplingConfig(enabled=True, seed=1, pools={"ds": str(prompt_pool)})
    )

    sample = transform({"dataset_name": "ds", "sample_id": "s1", "extra": {}})

    assert sample["system_prompt"] == "sys"
    assert sample["user_prompt"] == "user"
    assert sample["extra"]["runtime_prompt_id"] == "pool.test.main"
    assert sample["extra"]["runtime_prompt_version"] == version


def test_prompt_sampling_transform_requires_pool_version(tmp_path: Path) -> None:
    prompt_pool = tmp_path / "pool.yaml"
    prompt_pool.write_text(
        "\n".join(
            [
                "metadata:",
                "  id: pool.test",
                "prompts:",
                "  - id: main",
                "    user_prompt: user",
                "",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Missing prompt pool version"):
        build_prompt_sampling_transform(
            PromptSamplingConfig(enabled=True, pools={"ds": str(prompt_pool)})
        )


def test_prompt_sampling_uses_draw_id_and_configured_weights(tmp_path: Path) -> None:
    prompt_pool = tmp_path / "weighted_pool.yaml"
    prompt_pool.write_text(
        """
metadata:
  id: pool.weighted
  version: test-version
prompts:
  - id: disabled
    sampling_weight: 0
    user_prompt: never
  - id: active
    sampling_weight: 1
    user_prompt: always
""".strip()
        + "\n",
        encoding="utf-8",
    )
    transform = build_prompt_sampling_transform(
        PromptSamplingConfig(enabled=True, seed=9, pools={"ds": str(prompt_pool)})
    )

    samples = [
        transform(
            {
                "dataset_name": "ds",
                "sample_id": "same-row",
                "_sample_context": {"draw_id": draw_id},
                "extra": {},
            }
        )
        for draw_id in range(4)
    ]

    assert {sample["user_prompt"] for sample in samples} == {"always"}
    assert [sample["extra"]["runtime_prompt_draw_id"] for sample in samples] == list(range(4))


def test_prompt_sampling_defaults_to_equal_probability(tmp_path: Path) -> None:
    prompt_pool = tmp_path / "equal_pool.yaml"
    prompt_pool.write_text(
        """
metadata:
  id: pool.equal
  version: test-version
prompts:
  - id: a
    user_prompt: a
  - id: b
    user_prompt: b
""".strip()
        + "\n",
        encoding="utf-8",
    )
    transform = build_prompt_sampling_transform(
        PromptSamplingConfig(enabled=True, seed=9, pools={"ds": str(prompt_pool)})
    )

    counts = Counter(
        transform(
            {
                "dataset_name": "ds",
                "sample_id": "same-row",
                "_sample_context": {"draw_id": draw_id},
                "extra": {},
            }
        )["user_prompt"]
        for draw_id in range(2000)
    )

    assert counts["a"] / 2000 == pytest.approx(0.5, abs=0.04)
    assert counts["b"] / 2000 == pytest.approx(0.5, abs=0.04)


def test_prompt_sampling_normalizes_large_finite_weights(tmp_path: Path) -> None:
    prompt_pool = tmp_path / "large_weight_pool.yaml"
    prompt_pool.write_text(
        """
metadata:
  id: pool.large
  version: test-version
prompts:
  - id: a
    sampling_weight: 1.0e308
    user_prompt: a
  - id: b
    sampling_weight: 1.0e308
    user_prompt: b
""".strip()
        + "\n",
        encoding="utf-8",
    )
    transform = build_prompt_sampling_transform(
        PromptSamplingConfig(enabled=True, seed=21, pools={"ds": str(prompt_pool)})
    )

    counts = Counter(
        transform(
            {
                "dataset_name": "ds",
                "sample_id": "same-row",
                "_sample_context": {"draw_id": draw_id},
                "extra": {},
            }
        )["user_prompt"]
        for draw_id in range(2000)
    )

    assert counts["a"] / 2000 == pytest.approx(0.5, abs=0.05)
    assert counts["b"] / 2000 == pytest.approx(0.5, abs=0.05)
