from __future__ import annotations

from unittest.mock import patch

from shaft.cli.registry import COMMAND_REGISTRY
from shaft.cli.train import build_parser, main
from shaft.config import RuntimeConfig


def test_command_registry_has_expected_commands() -> None:
    assert "sft" in COMMAND_REGISTRY.keys()
    assert "rlhf" in COMMAND_REGISTRY.keys()


def test_parser_dispatches_to_sft() -> None:
    parser = build_parser()
    args = parser.parse_args(["sft", "--config", "dummy.yaml"])
    command_cls = getattr(args, "_command_cls")
    assert command_cls.__name__ == "SFTCommand"


def test_main_runs_sft_command() -> None:
    cfg = RuntimeConfig()
    captured: dict[str, str] = {}

    def _fake_run(config):
        captured["algorithm"] = config.algorithm.name
        return {"ok": 1}

    with patch("shaft.cli.common.load_config", return_value=cfg):
        with patch("shaft.cli.common.run_sft", side_effect=_fake_run):
            main(["sft", "--config", "dummy.yaml"])
    assert captured["algorithm"] == "sft"


def test_main_runs_rlhf_command() -> None:
    cfg = RuntimeConfig()
    captured: dict[str, str] = {}

    def _fake_run(config):
        captured["algorithm"] = config.algorithm.name
        return {"ok": 1}

    with patch("shaft.cli.common.load_config", return_value=cfg):
        with patch("shaft.cli.common.run_rlhf", side_effect=_fake_run):
            main(["rlhf", "--config", "dummy.yaml", "--algorithm", "dpo"])
    assert captured["algorithm"] == "dpo"


def test_main_defaults_to_sft_when_command_omitted() -> None:
    cfg = RuntimeConfig()
    captured: dict[str, str] = {}

    def _fake_run(config):
        captured["algorithm"] = config.algorithm.name
        return {"ok": 1}

    with patch("shaft.cli.common.load_config", return_value=cfg):
        with patch("shaft.cli.common.run_sft", side_effect=_fake_run):
            main(["--config", "dummy.yaml"])
    assert captured["algorithm"] == "sft"
