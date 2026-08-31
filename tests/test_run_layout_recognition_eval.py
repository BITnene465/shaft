from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

from PIL import Image
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/tasks/run_layout_recognition_eval.py"
SPEC = importlib.util.spec_from_file_location("run_layout_recognition_eval", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_cli_defaults_match_layout_inference_contract() -> None:
    parser = MODULE.build_parser()

    detection = parser.parse_args(
        [
            "detect",
            "--work-dir",
            "/tmp/work",
            "--image-dir",
            "/tmp/images",
            "--detection-prompt",
            "/tmp/detection.yaml",
        ]
    )
    prepare = parser.parse_args(
        [
            "prepare-reconstruction",
            "--work-dir",
            "/tmp/work",
            "--image-dir",
            "/tmp/images",
        ]
    )
    reconstruction = parser.parse_args(
        [
            "reconstruct",
            "--work-dir",
            "/tmp/work",
            "--shape-prompt",
            "/tmp/shape.yaml",
            "--line-prompt",
            "/tmp/line.yaml",
        ]
    )
    merge = parser.parse_args(
        ["merge", "--work-dir", "/tmp/work", "--image-dir", "/tmp/images"]
    )
    evaluation = parser.parse_args(
        [
            "evaluate",
            "--work-dir",
            "/tmp/work",
            "--image-dir",
            "/tmp/images",
            "--evaluator",
            "/tmp/eval.py",
            "--run-name",
            "run",
            "--checkpoint",
            "/tmp/checkpoint",
            "--gt-revision",
            "revision",
        ]
    )

    assert (detection.min_pixels, detection.max_pixels) == (500_000, 4_000_000)
    assert detection.endpoint_max_inflight == 4
    assert detection.allow_invalid_output is False
    assert prepare.padding_ratio == 0.65
    assert prepare.allow_missing_detection is False
    assert (reconstruction.min_pixels, reconstruction.max_pixels) == (500_000, 4_000_000)
    assert reconstruction.allow_invalid_output is False
    assert merge.allow_missing_detection is False
    assert merge.allow_missing_reconstruction is False
    assert (evaluation.detection_min_pixels, evaluation.detection_max_pixels) == (
        500_000,
        4_000_000,
    )


def test_checkpoint_weight_artifact_supports_sharded_and_single_file_hf_weights(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    single_file = checkpoint / "model.safetensors"
    single_file.write_bytes(b"single")

    assert MODULE.checkpoint_weight_artifact(checkpoint) == single_file

    index = checkpoint / "model.safetensors.index.json"
    index.write_text('{"weight_map": {}}', encoding="utf-8")

    assert MODULE.checkpoint_weight_artifact(checkpoint) == index


def test_checkpoint_weight_artifact_rejects_missing_hf_weights(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()

    with pytest.raises(FileNotFoundError, match="neither sharded index nor single-file"):
        MODULE.checkpoint_weight_artifact(checkpoint)


def test_prepare_reconstruction_can_explicitly_skip_missing_detection(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    pred_dir = tmp_path / "work/detection/pred"
    image_dir.mkdir()
    pred_dir.mkdir(parents=True)
    for stem in ("complete", "missing"):
        Image.new("RGB", (320, 240), "white").save(image_dir / f"{stem}.png")
    (pred_dir / "complete.json").write_text(
        MODULE.json.dumps(
            {
                "size": [320, 240],
                "layout": [
                    {"type": "shape", "bbox": [10, 20, 110, 120], "parameters": {}}
                ],
            }
        ),
        encoding="utf-8",
    )

    args = MODULE.argparse.Namespace(
        image_dir=image_dir,
        work_dir=tmp_path / "work",
        padding_ratio=0.65,
        minimum_crop_size=256,
        force=False,
        allow_missing_detection=True,
    )
    MODULE.prepare_reconstruction(args)

    summary = MODULE.json.loads(
        (tmp_path / "work/reconstruction/prepare_summary.json").read_text(encoding="utf-8")
    )
    assert summary["requests"] == 1
    assert summary["missing_detection_count"] == 1
    assert summary["missing_detection_stems"] == ["missing"]
    assert summary["allow_missing_detection"] is True


def test_main_prompt_supports_v58_default_and_reconstruction_formulations() -> None:
    prompt_root = SCRIPT.parents[2] / "configs/prompts/pools"

    detection = MODULE.main_prompt(prompt_root / "grounding_layout.v5.8.yaml")
    shape = MODULE.main_prompt(
        prompt_root / "shape_context_reconstruction.v5.8.yaml"
    )
    line = MODULE.main_prompt(
        prompt_root / "line_context_reconstruction.v5.8.yaml"
    )

    assert detection.variant_id == "detailed"
    assert shape.variant_id == "detailed"
    assert line.variant_id == "detailed"
    assert ".reconstruction." in shape.prompt_id
    assert ".reconstruction." in line.prompt_id


def test_parse_detection_dequantizes_and_normalizes() -> None:
    result = MODULE.parse_detection(
        '[{"bbox_2d":[100,200,900,800],"label":"shape"}]',
        width=1000,
        height=500,
    )

    assert result == [
        {"type": "shape", "bbox": [100, 100, 900, 400], "parameters": {}}
    ]


def test_context_crop_contains_detection_and_respects_bounds() -> None:
    crop = MODULE.context_crop_box(
        [0, 10, 20, 30],
        image_width=640,
        image_height=480,
        minimum_size=256,
    )

    assert crop[0] == 0
    assert crop[1] <= 10
    assert crop[2] >= 20
    assert crop[3] >= 30
    assert crop[2] - crop[0] >= 256
    assert crop[3] - crop[1] >= 256


def test_quantize_bbox_in_crop_matches_training_coordinate_contract() -> None:
    bbox = [179, 42, 301, 103]
    crop_box = (99, 0, 381, 256)

    assert MODULE.quantize_bbox_in_crop(bbox, crop_box) == [284, 165, 718, 404]


def test_convert_geometry_maps_nested_line_points_to_full_image() -> None:
    converted = MODULE.convert_geometry(
        {"points": [[[0, 0], [999, 999]]]},
        key="parameters",
        crop_box=(100, 50, 300, 150),
        image_width=1000,
        image_height=500,
    )

    assert converted == {"points": [[[100, 50], [299, 149]]]}


def test_convert_geometry_maps_nested_shape_point_collections_to_full_image() -> None:
    converted = MODULE.convert_geometry(
        {
            "corners": [
                {"type": "sharp", "point": [0, 0]},
                {
                    "type": "round",
                    "top_left": [100, 200],
                    "bottom_right": [900, 800],
                },
            ],
            "body_corners": [[0, 999], [999, 0]],
            "split_corners": [[250, 500], [750, 500]],
            "fill": {"stops": [[0, 0], [999, 999]]},
        },
        key="parameters",
        crop_box=(100, 50, 300, 150),
        image_width=1000,
        image_height=500,
    )

    assert converted == {
        "corners": [
            {"type": "sharp", "point": [100, 50]},
            {
                "type": "round",
                "top_left": [120, 70],
                "bottom_right": [279, 129],
            },
        ],
        "body_corners": [[100, 149], [299, 50]],
        "split_corners": [[150, 100], [249, 100]],
        "fill": {"stops": [[100, 50], [299, 149]]},
    }


def test_parse_reconstruction_accepts_complete_card_contract() -> None:
    parameters = MODULE.parse_reconstruction(
        MODULE.json.dumps(
            {
                "type": "shape",
                "parameters": {
                    "shape_type": "card",
                    "border": {"type": "uniform", "style": "solid", "color": "#112233"},
                    "fill": [
                        {"type": "uniform", "color": "#FFFFFF"},
                        {"type": "complex"},
                    ],
                    "effect": {"type": "none"},
                    "corners": [
                        {"type": "sharp", "point": [100, 100]},
                        {"type": "sharp", "point": [900, 100]},
                        {"type": "sharp", "point": [900, 900]},
                        {"type": "sharp", "point": [100, 900]},
                    ],
                    "splits": [
                        {
                            "type": "uniform",
                            "style": "dash",
                            "color": "#445566",
                            "split_corners": [
                                {"type": "sharp", "point": [100, 300]},
                                {"type": "sharp", "point": [900, 300]},
                            ],
                        }
                    ],
                },
            }
        ),
        expected_label="shape",
    )

    assert parameters["shape_type"] == "card"
    assert len(parameters["fill"]) == 2


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({"shape_type": "pill"}, "unsupported value"),
        (
            {
                "shape_type": "card",
                "border": {"type": "none"},
                "fill": [{"type": "uniform", "color": "#FFFFFF"}],
                "effect": {"type": "none"},
                "corners": [
                    {"type": "sharp", "point": [0, 0]},
                    {"type": "sharp", "point": [999, 0]},
                    {"type": "sharp", "point": [999, 999]},
                    {"type": "sharp", "point": [0, 999]},
                ],
                "splits": [
                    {
                        "type": "none",
                        "split_corners": [
                            {"type": "sharp", "point": [0, 500]},
                            {"type": "sharp", "point": [999, 500]},
                        ],
                    }
                ],
            },
            "fill count must equal split count plus one",
        ),
    ],
)
def test_parse_reconstruction_rejects_invalid_shape_contract(
    parameters: dict[str, object], message: str
) -> None:
    content = MODULE.json.dumps({"type": "shape", "parameters": parameters})

    with pytest.raises(ValueError, match=message):
        MODULE.parse_reconstruction(content, expected_label="shape")


def test_parse_reconstruction_validates_line_segments_and_markers() -> None:
    valid = {
        "line_type": "curved",
        "line_style": "path",
        "is_single": True,
        "points": [[[10, 20], [30, 40], [50, 60], [70, 80]]],
        "dash_style": "solid",
        "begin_arrow": "none",
        "end_arrow": "triangle",
        "fill": {"type": "uniform", "color": "#112233"},
        "border": {"type": "none"},
    }
    assert MODULE.parse_reconstruction(
        MODULE.json.dumps({"type": "line", "parameters": valid}),
        expected_label="line",
    ) == valid

    invalid = {**valid, "points": [[[10, 20], [70, 80]]], "end_arrow": "pointy"}
    with pytest.raises(ValueError, match="exactly four points"):
        MODULE.parse_reconstruction(
            MODULE.json.dumps({"type": "line", "parameters": invalid}),
            expected_label="line",
        )


def test_flatten_line_style_matches_real_v1_contract() -> None:
    converted = MODULE.flatten_line_style(
        {
            "line_type": "straight",
            "fill": {"type": "uniform", "color": "#112233"},
            "border": {"type": "uniform", "style": "solid", "color": "#445566"},
        }
    )

    assert converted["fill_color"] == "#112233"
    assert converted["has_border"] is True
    assert converted["border_style"] == "solid"
    assert converted["border_color"] == "#445566"
    assert "fill" not in converted
    assert "border" not in converted


def test_select_images_filters_requested_stems_and_rejects_missing(tmp_path: Path) -> None:
    for stem in ("00001", "00002"):
        Image.new("RGB", (8, 8)).save(tmp_path / f"{stem}.png")

    selected = MODULE.select_images(tmp_path, "00002")

    assert [path.stem for path in selected] == ["00002"]
    with pytest.raises(ValueError, match="00003"):
        MODULE.select_images(tmp_path, "00003")


def test_endpoint_slot_pool_reuses_the_first_released_replica() -> None:
    pool = MODULE.EndpointSlotPool(
        ["http://one", "http://two"], max_inflight_per_endpoint=1
    )
    first_lease = pool.acquire()
    second_lease = pool.acquire()
    assert first_lease.__enter__() == "http://one"
    assert second_lease.__enter__() == "http://two"

    second_lease.__exit__(None, None, None)
    with pool.acquire() as endpoint:
        assert endpoint == "http://two"
    first_lease.__exit__(None, None, None)


def test_endpoint_slot_pool_initial_burst_reaches_every_replica() -> None:
    pool = MODULE.EndpointSlotPool(
        ["http://one", "http://two"], max_inflight_per_endpoint=2
    )

    with pool.acquire() as first, pool.acquire() as second:
        assert [first, second] == ["http://one", "http://two"]


def test_call_vllm_disables_thinking(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return b'{"choices":[{"message":{"content":"[]"},"finish_reason":"stop"}]}'

    def fake_urlopen(http_request: object, *, timeout: float) -> FakeResponse:
        captured["body"] = MODULE.json.loads(http_request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(MODULE.request, "urlopen", fake_urlopen)
    MODULE.call_vllm(
        endpoint="http://localhost:8000",
        served_model="model",
        image_url="data:image/png;base64,AA==",
        system_prompt="system",
        user_prompt="user",
        max_tokens=100,
        timeout_seconds=12.0,
    )

    assert captured["body"]["chat_template_kwargs"] == {
        "enable_thinking": False,
        "preserve_thinking": False,
    }


def test_call_vllm_rejects_length_finish(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return b'{"choices":[{"message":{"content":"[]"},"finish_reason":"length"}]}'

    monkeypatch.setattr(MODULE.request, "urlopen", lambda *_args, **_kwargs: FakeResponse())
    with pytest.raises(ValueError, match="did not stop cleanly"):
        MODULE.call_vllm(
            endpoint="http://localhost:8000",
            served_model="model",
            image_url="data:image/png;base64,AA==",
            system_prompt="system",
            user_prompt="user",
            max_tokens=100,
            timeout_seconds=12.0,
        )


def test_artifact_state_is_scoped_to_expected_ids(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    error_dir = tmp_path / "errors"
    artifact_dir.mkdir()
    error_dir.mkdir()
    (artifact_dir / "expected.json").write_text("{}", encoding="utf-8")
    (artifact_dir / "stale.json").write_text("{}", encoding="utf-8")
    (error_dir / "missing.json").write_text("{}", encoding="utf-8")

    complete, unexpected, errors = MODULE.artifact_state(
        {"expected", "missing"},
        artifact_dir=artifact_dir,
        error_dir=error_dir,
    )

    assert complete == {"expected"}
    assert unexpected == {"stale"}
    assert errors == {"missing"}


def test_detect_can_reuse_a_recorded_invalid_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (8, 8)).save(image_dir / "sample.png")
    work_dir = tmp_path / "work"
    error_dir = work_dir / "detection" / "errors"
    error_dir.mkdir(parents=True)
    (error_dir / "sample.json").write_text(
        MODULE.json.dumps({"sample_id": "sample", "error": "length"}),
        encoding="utf-8",
    )
    prompt = type(
        "Prompt",
        (),
        {
            "prompt_id": "test.prompt",
            "system_prompt": "system",
            "user_prompt": "user",
        },
    )()
    monkeypatch.setattr(MODULE, "main_prompt", lambda _: prompt)
    monkeypatch.setattr(
        MODULE,
        "call_vllm",
        lambda **_: pytest.fail("a recorded invalid output must not be generated again"),
    )
    args = MODULE.argparse.Namespace(
        work_dir=work_dir,
        image_dir=image_dir,
        detection_prompt=tmp_path / "prompt.yaml",
        endpoints=["http://unused"],
        endpoint_max_inflight=1,
        served_model="unused",
        workers=1,
        timeout_seconds=1.0,
        retries=0,
        min_pixels=500_000,
        max_pixels=2_000_000,
        max_tokens=8_000,
        include_stems=None,
        allow_invalid_output=True,
        force=False,
    )

    MODULE.detect(args)

    summary = MODULE.json.loads(
        (work_dir / "detection/summary.json").read_text(encoding="utf-8")
    )
    assert summary["complete"] == 0
    assert summary["errors"] == 1
    assert summary["endpoint_scheduling"] == "dynamic_available_slot"


def test_reconstruct_can_reuse_a_recorded_invalid_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work_dir = tmp_path / "work"
    reconstruction = work_dir / "reconstruction"
    reconstruction.mkdir(parents=True)
    row = {
        "request_id": "sample__det_0000_shape",
        "sample_id": "sample",
        "detection_index": 0,
        "label": "shape",
        "crop_path": str(tmp_path / "crop.png"),
        "image_size": [8, 8],
        "crop_box": [0, 0, 8, 8],
        "proposal_bbox_2d": [0, 0, 999, 999],
        "proposal_bbox_full": [0, 0, 7, 7],
        "proposal_source": "detection",
        "gt_read": False,
    }
    (reconstruction / "manifest.jsonl").write_text(
        MODULE.json.dumps(row) + "\n", encoding="utf-8"
    )
    errors = reconstruction / "errors"
    errors.mkdir()
    (errors / "sample__det_0000_shape.json").write_text(
        MODULE.json.dumps({**row, "error": "length"}), encoding="utf-8"
    )
    monkeypatch.setattr(MODULE, "main_prompt", lambda _: object())
    monkeypatch.setattr(
        MODULE,
        "call_vllm",
        lambda **_: pytest.fail("a recorded invalid output must not be generated again"),
    )
    args = MODULE.argparse.Namespace(
        work_dir=work_dir,
        shape_prompt=tmp_path / "shape.yaml",
        line_prompt=tmp_path / "line.yaml",
        endpoints=["http://unused"],
        endpoint_max_inflight=1,
        served_model="unused",
        workers=1,
        timeout_seconds=1.0,
        retries=0,
        min_pixels=500_000,
        max_pixels=4_000_000,
        max_tokens=8_000,
        allow_invalid_output=True,
        force=False,
    )

    MODULE.reconstruct(args)

    summary = MODULE.json.loads(
        (reconstruction / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["complete"] == 0
    assert summary["errors"] == 1


def test_merge_can_drop_failed_reconstruction_when_explicitly_allowed(
    tmp_path: Path,
) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (8, 8)).save(image_dir / "sample.png")
    work_dir = tmp_path / "work"
    detection_dir = work_dir / "detection" / "pred"
    detection_dir.mkdir(parents=True)
    (detection_dir / "sample.json").write_text(
        MODULE.json.dumps(
            {
                "layout": [
                    {"type": "shape", "bbox": [0, 0, 7, 7], "parameters": {}},
                    {"type": "text", "bbox": [1, 1, 6, 6], "parameters": {}},
                ]
            }
        ),
        encoding="utf-8",
    )
    args = MODULE.argparse.Namespace(
        work_dir=work_dir,
        image_dir=image_dir,
        dataset_name="real_v1",
        allow_missing_detection=False,
        allow_missing_reconstruction=False,
    )

    with pytest.raises(FileNotFoundError):
        MODULE.merge(args)
    args.allow_missing_reconstruction = True

    MODULE.merge(args)

    result = MODULE.json.loads(
        (work_dir / "final/real_v1/pred/sample.json").read_text(encoding="utf-8")
    )
    summary = MODULE.json.loads(
        (work_dir / "final/summary.json").read_text(encoding="utf-8")
    )
    assert "parameters" not in result["layout"][0]
    assert summary["installed_reconstructions"] == 0
    assert summary["dropped_reconstructions"] == 1


def test_merge_can_explicitly_omit_image_with_missing_detection(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (8, 8)).save(image_dir / "missing.png")
    work_dir = tmp_path / "work"
    args = MODULE.argparse.Namespace(
        work_dir=work_dir,
        image_dir=image_dir,
        dataset_name="real_v1",
        allow_missing_detection=True,
        allow_missing_reconstruction=True,
    )

    MODULE.merge(args)

    summary = MODULE.json.loads(
        (work_dir / "final/summary.json").read_text(encoding="utf-8")
    )
    assert summary["expected_images"] == 1
    assert summary["images"] == 0
    assert summary["missing_detection_count"] == 1
    assert summary["missing_detection_stems"] == ["missing"]


def test_package_uses_explicit_dataset_and_preserves_existing_method(tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    result_root = tmp_path / "result"
    source = work_dir / "final" / "real_v2" / "pred"
    source.mkdir(parents=True)
    (source / "sample.json").write_text("{}", encoding="utf-8")
    existing_method = result_root / "method"
    (existing_method / "real_v1" / "pred").mkdir(parents=True)
    (existing_method / "real_v1" / "pred" / "old.json").write_text(
        "{}", encoding="utf-8"
    )
    args = MODULE.argparse.Namespace(
        work_dir=work_dir,
        result_root=result_root,
        run_name="method",
        dataset_name="real_v2",
    )

    MODULE.package(args)

    assert (existing_method / "real_v2" / "pred" / "sample.json").is_file()
    assert (existing_method / "real_v1" / "pred" / "old.json").is_file()
