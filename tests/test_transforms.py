from __future__ import annotations

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
