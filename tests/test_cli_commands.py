from __future__ import annotations

from unittest.mock import patch

import pytest

from shaft.cli.train import build_parser, main
from shaft.config import DatasetSourceConfig, RuntimeConfig
from tests.support.cli import capture_algorithm_runner


pytestmark = pytest.mark.contract


def _valid_runtime_config(*, source_type: str = "jsonl_sft") -> RuntimeConfig:
    config = RuntimeConfig()
    config.data.datasets = [
        DatasetSourceConfig(
            dataset_name="fixture",
            source_type=source_type,
            train_paths=["train.jsonl"],
            val_paths=["val.jsonl"],
        )
    ]
    return config


def test_parser_dispatches_to_sft() -> None:
    parser = build_parser()
    args = parser.parse_args(["sft", "--config", "dummy.yaml"])
    command_cls = getattr(args, "_command_cls")
    assert command_cls.__name__ == "SFTCommand"


def test_main_runs_sft_command() -> None:
    cfg = _valid_runtime_config()
    captured: dict[str, str] = {}

    with patch("shaft.cli.common.load_config", return_value=cfg):
        with patch("shaft.cli.common.run_sft", side_effect=capture_algorithm_runner(captured)):
            main(["sft", "--config", "dummy.yaml"])
    assert captured["algorithm"] == "sft"


def test_main_runs_rlhf_command() -> None:
    cfg = _valid_runtime_config(source_type="jsonl_dpo")
    captured: dict[str, str] = {}

    with patch("shaft.cli.common.load_config", return_value=cfg):
        with patch("shaft.cli.common.run_rlhf", side_effect=capture_algorithm_runner(captured)):
            main(["rlhf", "--config", "dummy.yaml", "--algorithm", "dpo"])
    assert captured["algorithm"] == "dpo"


def test_main_runs_grpo_command() -> None:
    cfg = _valid_runtime_config()
    cfg.eval.enabled = False
    captured: dict[str, str] = {}

    with patch("shaft.cli.common.load_config", return_value=cfg):
        with patch("shaft.cli.common.run_rlhf", side_effect=capture_algorithm_runner(captured)):
            main(["rlhf", "--config", "dummy.yaml", "--algorithm", "grpo"])
    assert captured["algorithm"] == "grpo"


def test_main_defaults_to_sft_when_command_omitted() -> None:
    cfg = _valid_runtime_config()
    captured: dict[str, str] = {}

    with patch("shaft.cli.common.load_config", return_value=cfg):
        with patch("shaft.cli.common.run_sft", side_effect=capture_algorithm_runner(captured)):
            main(["--config", "dummy.yaml"])
    assert captured["algorithm"] == "sft"
