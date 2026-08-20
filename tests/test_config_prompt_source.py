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
data:
  prompt_sources:
    ds1:
      path: prompts/pool.yaml
      apply_to: train
      seed: 123
      schedule:
        interpolation: linear
        points:
          - source_draw: 0
            weights: {default: 1.0}
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
    assert source.schedule.interpolation == "linear"
    assert source.schedule.points[0].source_draw == 0
    assert source.schedule.points[0].weights == {"default": 1.0}


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


@pytest.mark.parametrize(
    ("points", "match"),
    [
        (
            "- source_draw: 1\n            weights: {a: 1}",
            "first source_draw must be 0",
        ),
        (
            "- source_draw: 0\n            weights: {a: 1}\n"
            "          - source_draw: 0\n            weights: {a: 1}",
            "strictly increasing",
        ),
        (
            "- source_draw: 0\n            weights: {a: -1}",
            "finite and >= 0",
        ),
        (
            "- source_draw: 0\n            weights: {a: 1}\n"
            "          - source_draw: 1.5\n            weights: {a: 1}",
            "source_draw must be an integer",
        ),
    ],
)
def test_prompt_source_schedule_rejects_invalid_points(
    tmp_path: Path,
    points: str,
    match: str,
) -> None:
    payload = f"""
data:
  prompt_sources:
    ds1:
      path: prompt.yaml
      schedule:
        interpolation: step
        points:
          {points}
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
"""

    with pytest.raises((TypeError, ValueError), match=match):
        load_config(write_config_yaml(tmp_path, payload))


def test_prompt_source_schedule_rejects_duplicate_normalized_formulation_ids(
    tmp_path: Path,
) -> None:
    payload = """
data:
  prompt_sources:
    ds1:
      path: prompt.yaml
      schedule:
        points:
          - source_draw: 0
            weights:
              a: 1
              " a": 1
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
"""

    with pytest.raises(ValueError, match="duplicate normalized formulation id 'a'"):
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
