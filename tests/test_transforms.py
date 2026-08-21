from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from shaft.config import PromptSourceConfig, PromptSourceFormulationSourceConfig
from shaft.data import SFTRecord, build_offline_pipeline, build_online_pipeline
from shaft.data.prompt_source import (
    ShaftFormulationRecordStore,
    build_prompt_source_resolver,
)
from shaft.data.transforms import planning_online_transform_fingerprint


def _write_formulation_pool(path: Path) -> None:
    path.write_text(
        """
metadata:
  id: pool.reconstruction
  version: v1
formulations:
  - id: a
    sampling_weight: 1
    prompts:
      - id: direct
        system_prompt: JSON only.
        user_prompt: Reconstruct A.
      - id: alternate
        user_prompt: Output attribute A.
  - id: b
    sampling_weight: 1
    prompts:
      - id: direct
        user_prompt: Reconstruct B.
  - id: ab
    sampling_weight: 4
    prompts:
      - id: direct
        user_prompt: Reconstruct A and B.
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _formulation_sources(*formulation_ids: str) -> dict[
    str, PromptSourceFormulationSourceConfig
]:
    return {
        formulation_id: PromptSourceFormulationSourceConfig(train_path="unused.jsonl")
        for formulation_id in formulation_ids
    }


def _static_config(
    path: Path,
    *,
    seed: int = 7,
    formulation_ids: tuple[str, ...] = ("a", "b", "ab"),
) -> PromptSourceConfig:
    return PromptSourceConfig(
        path=str(path),
        seed=seed,
        formulation_sources=_formulation_sources(*formulation_ids),
    )


def _sample(
    draw_id: int,
    *,
    targets: dict[str, str] | None = None,
    reasoning_targets: dict[str, str | None] | None = None,
) -> dict[str, object]:
    resolved_targets = targets or {
        "a": '{"value":1}',
        "b": "[2,3]",
        "ab": '{"A":{"value":1},"B":[2,3]}',
    }
    store = ShaftFormulationRecordStore(
        {
            formulation_id: [
                SFTRecord(
                    image_path="/tmp/same-row.png",
                    dataset_name="ds",
                    sample_id="same-row",
                    target_text=target_text,
                    target_reasoning_content=(reasoning_targets or {}).get(
                        formulation_id
                    ),
                )
            ]
            for formulation_id, target_text in resolved_targets.items()
        }
    )
    record = store[0]
    sample = {
        "dataset_name": "ds",
        "sample_id": "same-row",
        "target_text": "",
        "target_reasoning_content": None,
        "prompt_args": {},
        "system_prompt": "",
        "user_prompt": "",
        "messages": None,
        "_split": "train",
        "_sample_context": {
            "draw_id": draw_id,
            "transform_seed": 99,
        },
        "extra": {},
    }
    sample.update(record.runtime_sample_fields())
    return sample


def test_offline_dedup() -> None:
    records = [
        SFTRecord(image_path="/tmp/a.png", target_text="{}", dataset_name="d"),
        SFTRecord(image_path="/tmp/a.png", target_text="{}", dataset_name="d"),
        SFTRecord(image_path="/tmp/a.png", target_text='{"x":1}', dataset_name="d"),
    ]
    pipeline = build_offline_pipeline(["dedup_image_target"])
    out = pipeline(records)
    assert len(out) == 2


def test_offline_dedup_preserves_distinct_reasoning_targets() -> None:
    records = [
        SFTRecord(
            image_path="/tmp/a.png",
            target_text="answer",
            target_reasoning_content="reasoning one",
        ),
        SFTRecord(
            image_path="/tmp/a.png",
            target_text="answer",
            target_reasoning_content="reasoning two",
        ),
    ]

    assert len(build_offline_pipeline(["dedup_image_target"])(records)) == 2


def test_online_identity() -> None:
    pipeline = build_online_pipeline(["identity"])
    sample = {"x": 1}
    out = pipeline(sample)
    assert out["x"] == 1


def test_inline_formulation_store_fingerprint_binds_materialized_targets() -> None:
    common = {
        "image_path": "/tmp/a.png",
        "dataset_name": "ds",
        "sample_id": "same",
    }
    first = ShaftFormulationRecordStore(
        {
            "a": [SFTRecord(target_text="A", **common)],
            "full": [SFTRecord(target_text="FULL-1", **common)],
        }
    )
    second = ShaftFormulationRecordStore(
        {
            "a": [SFTRecord(target_text="A", **common)],
            "full": [SFTRecord(target_text="FULL-2", **common)],
        }
    )

    assert first.fingerprint != second.fingerprint


def test_inline_formulation_store_fingerprint_binds_reasoning_targets() -> None:
    common = {
        "image_path": "/tmp/a.png",
        "dataset_name": "ds",
        "sample_id": "same",
        "target_text": "answer",
    }
    first = ShaftFormulationRecordStore(
        {"full": [SFTRecord(target_reasoning_content="reasoning-1", **common)]}
    )
    second = ShaftFormulationRecordStore(
        {"full": [SFTRecord(target_reasoning_content="reasoning-2", **common)]}
    )

    assert first.fingerprint != second.fingerprint


def test_prompt_source_selects_materialized_a_b_and_ab(tmp_path: Path) -> None:
    pool = tmp_path / "pool.yaml"
    _write_formulation_pool(pool)
    resolver = build_prompt_source_resolver({"ds": _static_config(pool)}, default_seed=3)
    selected = {
        resolved["extra"]["prompt_source"]["formulation_id"]: resolved
        for draw_id in range(200)
        for resolved in (resolver(_sample(draw_id)),)
    }
    assert set(selected) == {"a", "b", "ab"}
    a, b, ab = (selected[formulation_id] for formulation_id in ("a", "b", "ab"))

    assert a["user_prompt"] in {"Reconstruct A.", "Output attribute A."}
    assert a["target_text"] == '{"value":1}'
    assert b["user_prompt"] == "Reconstruct B."
    assert b["target_text"] == "[2,3]"
    assert ab["user_prompt"] == "Reconstruct A and B."
    assert ab["target_text"] == '{"A":{"value":1},"B":[2,3]}'
    assert [
        item["extra"]["prompt_source"]["formulation_id"]
        for item in (a, b, ab)
    ] == ["a", "b", "ab"]

    invalid_context = _sample(0)
    invalid_context["_sample_context"]["draw_id"] = 1.5  # type: ignore[index]
    with pytest.raises(ValueError, match="draw_id must be an integer"):
        resolver(invalid_context)


def test_prompt_source_selects_reasoning_from_the_same_formulation(tmp_path: Path) -> None:
    pool = tmp_path / "pool.yaml"
    _write_formulation_pool(pool)
    resolver = build_prompt_source_resolver({"ds": _static_config(pool)}, default_seed=3)
    reasoning_targets = {
        "a": "reason-a",
        "b": None,
        "ab": "reason-ab",
    }

    selected = {
        resolved["extra"]["prompt_source"]["formulation_id"]: resolved
        for draw_id in range(200)
        for resolved in (
            resolver(_sample(draw_id, reasoning_targets=reasoning_targets)),
        )
    }

    assert selected["a"]["target_reasoning_content"] == "reason-a"
    assert selected["b"]["target_reasoning_content"] is None
    assert selected["ab"]["target_reasoning_content"] == "reason-ab"


def test_prompt_source_selects_only_the_configured_shared_pool_subset(
    tmp_path: Path,
) -> None:
    pool = tmp_path / "pool.yaml"
    _write_formulation_pool(pool)
    subset_config = PromptSourceConfig(
        path=str(pool),
        seed=7,
        formulation_sources={
            "b": PromptSourceFormulationSourceConfig(train_path="unused.jsonl")
        },
    )
    subset_resolver = build_prompt_source_resolver(
        {"ds": subset_config},
        default_seed=3,
    )
    full_resolver = build_prompt_source_resolver(
        {"ds": _static_config(pool)},
        default_seed=3,
    )

    resolved = [
        subset_resolver(_sample(draw_id, targets={"b": "[2,3]"}))
        for draw_id in range(100)
    ]

    assert {
        item["extra"]["prompt_source"]["formulation_id"] for item in resolved
    } == {"b"}
    assert {item["user_prompt"] for item in resolved} == {"Reconstruct B."}
    assert {item["target_text"] for item in resolved} == {"[2,3]"}
    assert subset_resolver.fingerprint != full_resolver.fingerprint


def test_prompt_source_rejects_formulation_source_outside_shared_pool(
    tmp_path: Path,
) -> None:
    pool = tmp_path / "pool.yaml"
    _write_formulation_pool(pool)
    config = PromptSourceConfig(
        path=str(pool),
        formulation_sources={
            "unknown": PromptSourceFormulationSourceConfig(train_path="unused.jsonl")
        },
    )

    with pytest.raises(ValueError, match="must be a subset.*unknown=\\['unknown'\\]"):
        build_prompt_source_resolver({"ds": config}, default_seed=3)


def test_prompt_source_records_one_structured_audit_object(tmp_path: Path) -> None:
    pool = tmp_path / "pool.yaml"
    _write_formulation_pool(pool)
    resolver = build_prompt_source_resolver({"ds": _static_config(pool)}, default_seed=3)

    resolved = resolver(_sample(20))
    audit = resolved["extra"]["prompt_source"]

    assert audit["pool_id"] == "pool.reconstruction"
    assert audit["pool_version"] == "v1"
    assert audit["formulation_id"] in {"a", "b", "ab"}
    assert audit["draw_id"] == 20
    assert "source_draw_id" not in audit
    assert len(audit["prompt_program_sha256"]) == 64
    assert len(audit["arguments_sha256"]) == 64
    assert len(audit["user_prompt_sha256"]) == 64
    assert len(audit["target_text_sha256"]) == 64
    assert len(audit["target_reasoning_content_sha256"]) == 64
    assert not any(str(key).startswith("runtime_prompt_") for key in resolved["extra"])


def test_prompt_variant_sampling_preserves_static_formulation_distribution(
    tmp_path: Path,
) -> None:
    pool = tmp_path / "pool.yaml"
    _write_formulation_pool(pool)
    config = _static_config(pool, seed=11)
    resolver = build_prompt_source_resolver({"ds": config}, default_seed=3)

    counts = Counter(
        resolver(_sample(draw_id))["extra"]["prompt_source"]["formulation_id"]
        for draw_id in range(3000)
    )

    assert counts["a"] / 3000 == pytest.approx(1 / 6, abs=0.03)
    assert counts["b"] / 3000 == pytest.approx(1 / 6, abs=0.03)
    assert counts["ab"] / 3000 == pytest.approx(4 / 6, abs=0.03)


def test_prompt_source_randomly_samples_arbitrary_configured_formulations(
    tmp_path: Path,
) -> None:
    pool = tmp_path / "arbitrary-subsets.yaml"
    pool.write_text(
        """
metadata: {id: pool.arbitrary-subsets, version: v1}
formulations:
  - id: geometry
    sampling_weight: 1
    prompts: [{id: main, user_prompt: geometry}]
  - id: style
    sampling_weight: 2
    prompts: [{id: main, user_prompt: style}]
  - id: geometry_style
    sampling_weight: 3
    prompts: [{id: main, user_prompt: geometry-style}]
  - id: geometry_text_links
    sampling_weight: 4
    prompts: [{id: main, user_prompt: geometry-text-links}]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    resolver = build_prompt_source_resolver(
        {
            "ds": _static_config(
                pool,
                seed=29,
                formulation_ids=(
                    "geometry",
                    "style",
                    "geometry_style",
                    "geometry_text_links",
                ),
            )
        },
        default_seed=3,
    )
    targets = {
        "geometry": '{"subset":"geometry"}',
        "style": '{"subset":"style"}',
        "geometry_style": '{"subset":"geometry_style"}',
        "geometry_text_links": '{"subset":"geometry_text_links"}',
    }

    def resolve(draw_id: int) -> dict[str, object]:
        return resolver(_sample(draw_id, targets=targets))

    resolved = [resolve(draw_id) for draw_id in range(4000)]
    formulation_ids = [
        item["extra"]["prompt_source"]["formulation_id"] for item in resolved
    ]
    counts = Counter(formulation_ids)

    assert counts["geometry"] / 4000 == pytest.approx(0.1, abs=0.03)
    assert counts["style"] / 4000 == pytest.approx(0.2, abs=0.03)
    assert counts["geometry_style"] / 4000 == pytest.approx(0.3, abs=0.03)
    assert counts["geometry_text_links"] / 4000 == pytest.approx(0.4, abs=0.03)
    assert formulation_ids == [
        resolve(draw_id)["extra"]["prompt_source"]["formulation_id"]
        for draw_id in range(4000)
    ]
    for item, formulation_id in zip(resolved, formulation_ids, strict=True):
        assert item["target_text"] == targets[formulation_id]


def test_top_level_prompts_are_one_materialized_default_formulation(
    tmp_path: Path,
) -> None:
    pool = tmp_path / "shorthand.yaml"
    pool.write_text(
        """
metadata: {id: pool.shorthand, version: v1}
prompts:
  - id: main
    user_prompt: first
  - id: alternate
    user_prompt: second
""".strip()
        + "\n",
        encoding="utf-8",
    )
    resolver = build_prompt_source_resolver(
        {"ds": PromptSourceConfig(path=str(pool), seed=9)},
        default_seed=3,
    )
    sample = {
        "dataset_name": "ds",
        "sample_id": "same-row",
        "target_text": "materialized",
        "prompt_args": {},
        "system_prompt": "",
        "user_prompt": "",
        "messages": None,
        "_split": "train",
        "_sample_context": {"draw_id": 0},
        "extra": {},
    }

    resolved = resolver(sample)

    assert resolved["user_prompt"] in {"first", "second"}
    assert resolved["target_text"] == "materialized"
    assert resolved["extra"]["prompt_source"]["formulation_id"] == "default"


def test_explicit_formulation_pool_requires_nonempty_eligibility(
    tmp_path: Path,
) -> None:
    pool = tmp_path / "pool.yaml"
    _write_formulation_pool(pool)

    with pytest.raises(ValueError, match="require a non-empty formulation_sources subset"):
        build_prompt_source_resolver(
            {"ds": PromptSourceConfig(path=str(pool))},
            default_seed=3,
        )


def test_prompt_source_rejects_all_zero_eligible_subset(tmp_path: Path) -> None:
    pool = tmp_path / "pool.yaml"
    _write_formulation_pool(pool)
    config = _static_config(pool, formulation_ids=("a",))
    payload = pool.read_text(encoding="utf-8").replace(
        "  - id: a\n    sampling_weight: 1",
        "  - id: a\n    sampling_weight: 0",
    )
    pool.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match="eligibility subset needs at least one positive"):
        build_prompt_source_resolver({"ds": config}, default_seed=3)


def test_materialized_mode_is_identity_and_rejects_ignored_args() -> None:
    resolver = build_prompt_source_resolver({}, default_seed=3)
    materialized = {
        "dataset_name": "ds",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        "target_text": "answer",
        "prompt_args": {},
        "extra": {},
    }

    assert resolver(materialized) is materialized

    with pytest.raises(ValueError, match="no configured PromptSource"):
        resolver({**materialized, "prompt_args": {"value": "ignored"}})


def test_pool_mode_rejects_materialized_messages(tmp_path: Path) -> None:
    pool = tmp_path / "pool.yaml"
    _write_formulation_pool(pool)
    resolver = build_prompt_source_resolver({"ds": _static_config(pool)}, default_seed=3)
    sample = _sample(0)
    sample["messages"] = [{"role": "user", "content": []}]

    with pytest.raises(ValueError, match="pool mode.*messages"):
        resolver(sample)


def test_pool_mode_rejects_materialized_system_prompt(tmp_path: Path) -> None:
    pool = tmp_path / "pool.yaml"
    _write_formulation_pool(pool)
    resolver = build_prompt_source_resolver({"ds": _static_config(pool)}, default_seed=3)
    sample = _sample(0)
    sample["system_prompt"] = "ignored"

    with pytest.raises(ValueError, match="pool mode.*system_prompt"):
        resolver(sample)


def test_prompt_source_rejects_unknown_prompt_variant_keys(tmp_path: Path) -> None:
    pool = tmp_path / "pool.yaml"
    pool.write_text(
        """
metadata: {id: pool.invalid, version: v1}
prompts:
  - id: main
    user_prompt: answer
    typo_weight: 1
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unknown prompt variant keys.*typo_weight"):
        build_prompt_source_resolver(
            {"ds": PromptSourceConfig(path=str(pool))},
            default_seed=3,
        )


@pytest.mark.parametrize("legacy_key", ["target", "target_template"])
def test_prompt_source_rejects_online_target_assembly(
    tmp_path: Path,
    legacy_key: str,
) -> None:
    pool = tmp_path / "pool.yaml"
    value = "materialized" if legacy_key == "target" else "'{{ value }}'"
    pool.write_text(
        f"""
metadata: {{id: pool.invalid-target, version: v1}}
formulations:
  - id: a
    {legacy_key}: {value}
    prompts: [{{id: main, user_prompt: answer}}]
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=f"Unknown PromptSource formulation keys.*{legacy_key}"):
        build_prompt_source_resolver(
            {"ds": PromptSourceConfig(path=str(pool))},
            default_seed=3,
        )


def test_prompt_source_fingerprint_binds_prompt_schema(tmp_path: Path) -> None:
    paths = []
    for name, argument_type in (("string", "string"), ("enum", "enum")):
        path = tmp_path / f"{name}.yaml"
        values = ", values: [x]" if argument_type == "enum" else ""
        path.write_text(
            f"""
metadata: {{id: pool.fingerprint, version: v1}}
arguments:
  value: {{type: {argument_type}{values}}}
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
            build_prompt_source_resolver(
                {"ds": PromptSourceConfig(path=str(path))},
                default_seed=3,
            )
        )
        for path in paths
    ]

    assert fingerprints[0] != fingerprints[1]
