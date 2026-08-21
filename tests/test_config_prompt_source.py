from __future__ import annotations

from pathlib import Path

import pytest

from shaft.config import load_config
from tests.support.configs import load_config_from_yaml, write_config_yaml


pytestmark = pytest.mark.component


def test_prompt_sources_config_normalizes_and_resolves_paths(tmp_path: Path) -> None:
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    (prompt_dir / "pool.yaml").write_text(
        "metadata:\n  id: pool.test\n  version: test-version\n"
        "prompts:\n  - id: main\n    user_prompt: a\n",
        encoding="utf-8",
    )
    payload = """
eval:
  enabled: false
data:
  prompt_sources:
    ds1:
      path: prompts/pool.yaml
      apply_to: train
      seed: 123
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
"""
    cfg = load_config_from_yaml(tmp_path, payload)

    source = cfg.data.prompt_sources["ds1"]
    assert source.path == str((prompt_dir / "pool.yaml").resolve())
    assert source.apply_to == "train"
    assert source.seed == 123


def test_materialized_formulation_source_paths_are_resolved(tmp_path: Path) -> None:
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    (prompt_dir / "pool.yaml").write_text(
        """
metadata: {id: pool.formulations, version: v1}
formulations:
  - id: points
    prompts: [{id: main, user_prompt: points}]
  - id: full
    prompts: [{id: main, user_prompt: full}]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    payload = """
eval:
  enabled: false
data:
  prompt_sources:
    ds1:
      path: prompts/pool.yaml
      formulation_sources:
        points: {train_path: data/points.jsonl}
        full: {train_paths: [data/full-a.jsonl, data/full-b.jsonl]}
  datasets:
    - dataset_name: ds1
      use_for_eval: false
"""

    cfg = load_config_from_yaml(tmp_path, payload)

    source = cfg.data.prompt_sources["ds1"]
    assert cfg.data.datasets[0].train_paths == []
    assert source.formulation_sources["points"].train_paths == [
        str((tmp_path / "data/points.jsonl").resolve())
    ]
    assert source.formulation_sources["full"].train_paths == [
        str((tmp_path / "data/full-a.jsonl").resolve()),
        str((tmp_path / "data/full-b.jsonl").resolve()),
    ]


def test_formulation_sources_cannot_mix_with_top_level_train_paths(tmp_path: Path) -> None:
    payload = """
eval:
  enabled: false
data:
  prompt_sources:
    ds1:
      path: prompt.yaml
      formulation_sources:
        a: {train_path: data/a.jsonl}
  datasets:
    - dataset_name: ds1
      train_path: data/default.jsonl
      use_for_eval: false
"""

    with pytest.raises(ValueError, match="cannot combine top-level train_paths"):
        load_config_from_yaml(tmp_path, payload)


def test_formulation_val_sources_require_apply_to_all(tmp_path: Path) -> None:
    payload = """
eval:
  enabled: false
data:
  prompt_sources:
    ds1:
      path: prompt.yaml
      apply_to: train
      formulation_sources:
        a: {train_path: data/a-train.jsonl, val_path: data/a-val.jsonl}
  datasets:
    - dataset_name: ds1
      use_for_eval: false
"""

    with pytest.raises(ValueError, match="val_paths require apply_to='all'"):
        load_config_from_yaml(tmp_path, payload)


def test_apply_to_all_uses_formulation_val_sources(tmp_path: Path) -> None:
    payload = """
eval:
  enabled: true
data:
  prompt_sources:
    ds1:
      path: prompt.yaml
      apply_to: all
      formulation_sources:
        a: {train_path: data/a-train.jsonl, val_path: data/a-val.jsonl}
        full: {train_path: data/full-train.jsonl, val_path: data/full-val.jsonl}
  datasets:
    - dataset_name: ds1
      use_for_eval: true
"""

    cfg = load_config_from_yaml(tmp_path, payload)

    assert cfg.data.datasets[0].val_paths == []
    assert cfg.data.prompt_sources["ds1"].formulation_sources["full"].val_paths == [
        str((tmp_path / "data/full-val.jsonl").resolve())
    ]


@pytest.mark.parametrize("apply_to", ["eval", "training", ""])
def test_prompt_sources_reject_invalid_apply_to(
    tmp_path: Path,
    apply_to: str,
) -> None:
    payload = f"""
data:
  prompt_sources:
    ds1:
      path: prompt.yaml
      apply_to: {apply_to!r}
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
"""

    with pytest.raises(ValueError, match="apply_to"):
        load_config(write_config_yaml(tmp_path, payload))


def test_prompt_sources_reject_unknown_dataset(tmp_path: Path) -> None:
    payload = """
data:
  prompt_sources:
    missing:
      path: prompt.yaml
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
"""

    with pytest.raises(ValueError, match="unknown datasets.*missing"):
        load_config(write_config_yaml(tmp_path, payload))


def test_removed_prompt_source_schedule_is_rejected(tmp_path: Path) -> None:
    payload = """
data:
  prompt_sources:
    ds1:
      path: prompt.yaml
      schedule:
        interpolation: step
        points: []
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
"""

    with pytest.raises(ValueError, match="Unknown config keys.*schedule"):
        load_config(write_config_yaml(tmp_path, payload))


def test_legacy_prompt_sampling_config_is_rejected(tmp_path: Path) -> None:
    payload = """
data:
  transforms:
    prompt_sampling:
      enabled: true
      pools: {ds1: prompt.yaml}
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
"""

    with pytest.raises(ValueError, match="Unknown config keys at data:.*transforms"):
        load_config(write_config_yaml(tmp_path, payload))
