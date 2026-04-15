from __future__ import annotations

from unittest.mock import patch

from shaft.cli.infer import build_parser, main


def test_infer_parser_accepts_required_args() -> None:
    parser = build_parser()
    args = parser.parse_args(["--config", "infer.yaml", "--image", "image.png"])
    assert args.config == "infer.yaml"
    assert args.image == "image.png"
    assert args.inputs == "{}"


def test_infer_main_runs_pipeline() -> None:
    fake_config = object()

    class _FakePipeline:
        def run(self, *, image_path: str, inputs: dict):
            assert image_path == "image.png"
            assert inputs == {"task": "arrow"}
            return {"ok": True}

    with patch("shaft.cli.infer.load_infer_config", return_value=fake_config) as load_cfg:
        with patch("shaft.cli.infer.InferPipeline.from_config", return_value=_FakePipeline()) as build_pipeline:
            main(["--config", "infer.yaml", "--image", "image.png", "--inputs", '{"task":"arrow"}'])
    load_cfg.assert_called_once_with("infer.yaml")
    build_pipeline.assert_called_once_with(fake_config)
