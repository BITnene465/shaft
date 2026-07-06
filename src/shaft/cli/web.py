from __future__ import annotations

import argparse

from shaft.webui import main as launch_webui
from shaft.webui.app import DEFAULT_SFT_CONFIG


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the Shaft Web UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Optional fixed port. Omit to let the server choose an available port.",
    )
    parser.add_argument("--base-config", default=DEFAULT_SFT_CONFIG)
    parser.add_argument(
        "--share",
        action="store_true",
        help="Deprecated compatibility flag. Ignored by the FastAPI web UI.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        launch_webui(
            host=args.host,
            port=args.port,
            base_config_path=args.base_config,
            share=args.share,
        )
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
