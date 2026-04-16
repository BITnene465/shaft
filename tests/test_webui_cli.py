from __future__ import annotations

from unittest.mock import patch

from shaft.cli.web import main


def test_webui_cli_forwards_arguments() -> None:
    with patch("shaft.cli.web.launch_webui") as mocked_launch:
        main(
            [
                "--host",
                "0.0.0.0",
                "--port",
                "9999",
                "--base-config",
                "configs/train/train_sft_4b.yaml",
                "--share",
            ]
        )
    mocked_launch.assert_called_once_with(
        host="0.0.0.0",
        port=9999,
        base_config_path="configs/train/train_sft_4b.yaml",
        share=True,
    )


def test_webui_cli_uses_auto_port_by_default() -> None:
    with patch("shaft.cli.web.launch_webui") as mocked_launch:
        main([])
    mocked_launch.assert_called_once_with(
        host="127.0.0.1",
        port=None,
        base_config_path="configs/train/train_sft_4b.yaml",
        share=False,
    )


def test_webui_cli_treats_keyboard_interrupt_as_clean_exit() -> None:
    with patch("shaft.cli.web.launch_webui", side_effect=KeyboardInterrupt):
        main([])
