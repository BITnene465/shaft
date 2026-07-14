from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from shaft.config import PromptSamplingConfig
from shaft.data import SFTRecord, build_offline_pipeline, build_online_pipeline
from shaft.data.transforms import (
    build_prompt_sampling_transform,
    planning_online_transform_fingerprint,
)


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


def test_prompt_sampling_renders_sample_arguments_and_records_audit_hashes(
    tmp_path: Path,
) -> None:
    prompt_pool = tmp_path / "dynamic_pool.yaml"
    prompt_pool.write_text(
        """
metadata:
  id: pool.dynamic
  version: v2
arguments:
  kind:
    type: enum
    values: [shape, line]
  bbox:
    type: bbox_2d_0_999
prompts:
  - id: main
    system_prompt: Return JSON only.
    user_prompt_template: "Reconstruct {{ kind }} at {{ bbox | json }}."
""".strip()
        + "\n",
        encoding="utf-8",
    )
    transform = build_prompt_sampling_transform(
        PromptSamplingConfig(enabled=True, seed=3, pools={"ds": str(prompt_pool)})
    )

    sample = transform(
        {
            "dataset_name": "ds",
            "sample_id": "s1",
            "prompt_args": {"kind": "shape", "bbox": [1, 2, 30, 40]},
            "extra": {},
        }
    )

    assert sample["user_prompt"] == "Reconstruct shape at [1,2,30,40]."
    assert sample["system_prompt"] == "Return JSON only."
    assert sample["extra"]["runtime_prompt_renderer_version"]
    assert len(sample["extra"]["runtime_prompt_template_sha256"]) == 64
    assert len(sample["extra"]["runtime_prompt_arguments_sha256"]) == 64
    assert len(sample["extra"]["runtime_prompt_rendered_sha256"]) == 64

    changed = transform(
        {
            "dataset_name": "ds",
            "sample_id": "s1",
            "prompt_args": {"kind": "line", "bbox": [1, 2, 30, 40]},
            "extra": {},
        }
    )
    assert changed["extra"]["runtime_prompt_arguments_sha256"] != (
        sample["extra"]["runtime_prompt_arguments_sha256"]
    )
    assert changed["extra"]["runtime_prompt_rendered_sha256"] != (
        sample["extra"]["runtime_prompt_rendered_sha256"]
    )


def test_prompt_schema_change_changes_transform_fingerprint(tmp_path: Path) -> None:
    paths = []
    for name, argument_type in (("string", "string"), ("enum", "enum")):
        path = tmp_path / f"{name}.yaml"
        values = "\n    values: [x]" if argument_type == "enum" else ""
        path.write_text(
            f"""
metadata:
  id: pool.fingerprint
  version: v1
arguments:
  value:
    type: {argument_type}{values}
prompts:
  - id: main
    user_prompt_template: "{{{{ value }}}}"
""".strip()
            + "\n",
            encoding="utf-8",
        )
        paths.append(path)

    fingerprints = [
        planning_online_transform_fingerprint(
            build_prompt_sampling_transform(
                PromptSamplingConfig(enabled=True, pools={"ds": str(path)})
            )
        )
        for path in paths
    ]

    assert fingerprints[0] != fingerprints[1]


def test_prompt_sampling_rejects_messages_with_prompt_args(tmp_path: Path) -> None:
    prompt_pool = tmp_path / "pool.yaml"
    prompt_pool.write_text(
        """
metadata:
  id: pool.dynamic
  version: v1
arguments:
  value:
    type: string
prompts:
  - id: main
    user_prompt_template: "Value: {{ value }}"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    transform = build_prompt_sampling_transform(
        PromptSamplingConfig(enabled=True, pools={"ds": str(prompt_pool)})
    )

    with pytest.raises(ValueError, match="messages.*prompt_args"):
        transform(
            {
                "dataset_name": "ds",
                "sample_id": "s1",
                "messages": [{"role": "user", "content": []}],
                "prompt_args": {"value": "x"},
                "extra": {},
            }
        )


def test_disabled_prompt_sampling_rejects_nonempty_prompt_args() -> None:
    transform = build_prompt_sampling_transform(PromptSamplingConfig(enabled=False))

    with pytest.raises(ValueError, match="prompt sampling is disabled"):
        transform(
            {
                "dataset_name": "ds",
                "sample_id": "s1",
                "prompt_args": {"value": "x"},
            }
        )
