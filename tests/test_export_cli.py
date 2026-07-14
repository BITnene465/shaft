from __future__ import annotations

from unittest.mock import patch

from shaft.cli.export import build_parser, main


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
    with patch("shaft.cli.export.inspect_hf_artifact") as mocked:
        mocked.return_value.path = "ckpt"
        mocked.return_value.kind = "full"
        mocked.return_value.has_trainer_state = True
        main(["inspect", "--path", "ckpt"])
    mocked.assert_called_once_with("ckpt")


def test_export_main_forwards_hf_locator_options_to_validate() -> None:
    with patch("shaft.cli.export.validate_hf_artifact") as mocked:
        mocked.return_value.path = "adapter"
        mocked.return_value.kind = "adapter"
        main(
            [
                "validate",
                "--path",
                "adapter",
                "--finetune-mode",
                "lora",
                "--model-type",
                "qwen3vl",
                "--model-name-or-path",
                "org/model",
                "--revision",
                "release-v2",
                "--cache-dir",
                "/tmp/hf-cache",
                "--local-files-only",
                "true",
            ]
        )

    mocked.assert_called_once_with(
        "adapter",
        finetune_mode="lora",
        model_type="qwen3vl",
        model_name_or_path="org/model",
        template=None,
        revision="release-v2",
        cache_dir="/tmp/hf-cache",
        local_files_only=True,
    )


def test_export_main_forwards_hf_locator_options_to_merge() -> None:
    with patch("shaft.cli.export.merge_peft_adapter") as mocked:
        mocked.return_value.output_dir = "merged"
        mocked.return_value.base_model_path = "org/model"
        mocked.return_value.adapter_path = "adapter"
        mocked.return_value.layout.kind = "full"
        main(
            [
                "merge-peft",
                "--model-type",
                "qwen3vl",
                "--adapter-path",
                "adapter",
                "--output-dir",
                "merged",
                "--base-model",
                "org/model",
                "--revision",
                "release-v2",
                "--cache-dir",
                "/tmp/hf-cache",
                "--local-files-only",
                "true",
            ]
        )

    assert mocked.call_args.kwargs["revision"] == "release-v2"
    assert mocked.call_args.kwargs["cache_dir"] == "/tmp/hf-cache"
    assert mocked.call_args.kwargs["local_files_only"] is True
