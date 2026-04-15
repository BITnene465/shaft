from __future__ import annotations

from unittest.mock import patch

from shaft.export.cli import build_parser, main


def test_export_parser_dispatches_merge_peft() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "merge-peft",
            "--model-type",
            "qwen3vl",
            "--adapter-path",
            "adapter",
            "--output-dir",
            "merged",
        ]
    )
    assert args.command == "merge-peft"
    assert args.model_type == "qwen3vl"


def test_export_main_runs_inspect() -> None:
    with patch("shaft.export.cli.inspect_hf_artifact") as mocked:
        mocked.return_value.path = "ckpt"
        mocked.return_value.kind = "full"
        mocked.return_value.has_trainer_state = True
        main(["inspect", "--path", "ckpt"])
    mocked.assert_called_once_with("ckpt")
