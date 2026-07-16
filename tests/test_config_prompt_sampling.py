from __future__ import annotations

from pathlib import Path

import pytest

from shaft.config import load_config
from tests.support.configs import load_config_from_yaml, write_config_yaml


pytestmark = pytest.mark.component


def test_prompt_sampling_config_normalizes_and_resolves_paths(tmp_path: Path) -> None:
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    (prompt_dir / "pool.yaml").write_text(
        "metadata:\n  id: pool.test\n  version: test-version\nprompts:\n  - id: main\n    user_prompt: a\n",
        encoding="utf-8",
    )
    payload = """
data:
  transforms:
    prompt_sampling:
      enabled: true
      train_only: true
      seed: 123
      pools:
        ds1: prompts/pool.yaml
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
"""
    cfg = load_config_from_yaml(tmp_path, payload)

    assert cfg.data.transforms.prompt_sampling.enabled is True
    assert cfg.data.transforms.prompt_sampling.train_only is True
    assert cfg.data.transforms.prompt_sampling.seed == 123
    assert cfg.data.transforms.prompt_sampling.pools == {"ds1": str((prompt_dir / "pool.yaml").resolve())}


def test_prompt_sampling_requires_pool_for_every_active_dataset(tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompt.yaml"
    prompt_path.write_text(
        "metadata:\n  id: p\n  version: test-version\nprompts:\n  - id: main\n    user_prompt: p\n",
        encoding="utf-8",
    )
    payload = f"""
data:
  transforms:
    prompt_sampling:
      enabled: true
      pools:
        ds1: {prompt_path}
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
    - dataset_name: ds2
      train_path: train.jsonl
      val_path: val.jsonl
"""
    config_path = write_config_yaml(tmp_path, payload)

    with pytest.raises(ValueError, match="requires prompt pools for all active train/eval datasets"):
        load_config(config_path)


def test_prompt_sampling_does_not_require_pool_for_zero_weight_train_source(
    tmp_path: Path,
) -> None:
    prompt_path = tmp_path / "prompt.yaml"
    prompt_path.write_text(
        "metadata:\n  id: p\n  version: test-version\nprompts:\n  - id: main\n    user_prompt: p\n",
        encoding="utf-8",
    )
    payload = f"""
data:
  transforms:
    prompt_sampling:
      enabled: true
      pools:
        ds1: {prompt_path}
  datasets:
    - dataset_name: ds1
      train_path: train.jsonl
      val_path: val.jsonl
    - dataset_name: disabled_train
      train_path: unused.jsonl
      weight: 0
      use_for_eval: false
"""

    config = load_config(write_config_yaml(tmp_path, payload))

    assert config.data.datasets[1].weight == 0
