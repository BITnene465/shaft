from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval_bench.artifacts import (
    DEFAULT_STORE_ROOT,
    BenchmarkArtifacts,
    RunArtifacts,
    StoreLayout,
    load_prediction,
)
from eval_bench.benchmark import create_benchmark_from_raw_data
from eval_bench.schema import (
    BenchmarkRef,
    EvalRunManifest,
    EvalSpec,
    InferenceParams,
    ModelRef,
    PredictionDocument,
    PredictionInstance,
    PromptRef,
)


def test_prediction_document_allows_detection_without_keypoints(tmp_path: Path) -> None:
    doc = PredictionDocument(
        image="part1/images/a.png",
        instances=[PredictionInstance(label="icon", bbox=[1, 2, 10, 20])],
        metadata={"producer": "eval_bench"},
    )

    artifacts = RunArtifacts(tmp_path, "run1")
    path = artifacts.write_prediction(doc, task="detection")

    loaded = load_prediction(path, task="detection")
    assert loaded.image == "part1/images/a.png"
    assert loaded.instances[0].label == "icon"
    assert "keypoints" not in json.loads(path.read_text())["instances"][0]


def test_keypoint_document_allows_missing_keypoints_for_metric_failure() -> None:
    doc = PredictionDocument(
        image="part1/images/a.png",
        instances=[PredictionInstance(label="arrow", bbox=[1, 2, 10, 20])],
        metadata={"producer": "eval_bench"},
    )

    payload = doc.to_dict(task="keypoint")

    assert payload["instances"][0]["label"] == "arrow"
    assert "keypoints" not in payload["instances"][0]


def test_prediction_document_rejects_malformed_instance_payload(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            {
                "image": "part1/images/a.png",
                "status": "predicted",
                "instances": ["not-an-object"],
                "metadata": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"instances\[0\] must be an object"):
        load_prediction(path, task="detection")


def test_run_manifest_validates_benchmark_and_spec_task(tmp_path: Path) -> None:
    manifest = EvalRunManifest(
        run_id="run1",
        model=ModelRef(model_id="model", path="outputs/model/best"),
        benchmark=BenchmarkRef(
            benchmark_id="raw.val.detection",
            root="/eval_bench_store/benchmarks/raw.val.detection/data",
            split="val",
            tasks=["detection", "keypoint"],
            manifest_path="/eval_bench_store/benchmarks/raw.val.detection/splits/val.txt",
        ),
        spec=EvalSpec(
            spec_id="det.latest",
            task="detection",
            prompt=PromptRef(prompt_id="grounding_layout.sft.v3"),
            inference=InferenceParams(max_tokens=4096, max_pixels=1048576),
        ),
    )

    artifacts = RunArtifacts(tmp_path, manifest.run_id)
    path = artifacts.write_manifest(manifest)

    payload = json.loads(path.read_text())
    assert payload["run_id"] == "run1"
    assert payload["benchmark"]["tasks"] == ["detection", "keypoint"]
    assert payload["spec"]["inference"]["max_tokens"] == 4096


def test_prediction_path_keeps_raw_data_part_layout(tmp_path: Path) -> None:
    artifacts = RunArtifacts(tmp_path, "run1")
    path = artifacts.prediction_path("part2/images/pic001.png")
    assert path == tmp_path / "runs" / "run1" / "predictions" / "part2" / "json" / "pic001.json"


def test_store_layout_keeps_db_and_runs_outside_outputs() -> None:
    layout = StoreLayout()
    assert DEFAULT_STORE_ROOT == Path("eval_bench_store")
    assert layout.db_path == Path("eval_bench_store/db/eval_bench.sqlite")
    assert layout.runs_dir == Path("eval_bench_store/runs")
    assert layout.benchmarks_dir == Path("eval_bench_store/benchmarks")


def test_create_benchmark_copies_raw_data_validation_subset(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw_data"
    (raw_root / "part1" / "images").mkdir(parents=True)
    (raw_root / "part1" / "json").mkdir(parents=True)
    image_path = raw_root / "part1" / "images" / "a.png"
    image_path.write_bytes(b"fake-image")
    raw_json = raw_root / "part1" / "json" / "a.json"
    raw_json.write_text(
        json.dumps(
            {
                "schema": "shaft.raw_data.v1",
                "image_path": "part1/images/a.png",
                "image_width": 10,
                "image_height": 10,
                "instances": [{"label": "icon", "bbox": [1, 2, 3, 4]}],
            }
        ),
        encoding="utf-8",
    )
    split_path = raw_root / "splits" / "layout_val.txt"
    split_path.parent.mkdir()
    split_path.write_text("part1/json/a.json\n", encoding="utf-8")

    manifest = create_benchmark_from_raw_data(
        store_root=tmp_path / "store",
        benchmark_id="layout_val_v1",
        tasks=["detection", "keypoint"],
        source_root=raw_root,
        source_manifest=split_path,
        split="val",
        layers=["layout"],
    )

    artifacts = BenchmarkArtifacts(tmp_path / "store", "layout_val_v1")
    assert manifest.sample_count == 1
    assert Path(manifest.root) == artifacts.data_dir
    assert (artifacts.data_dir / "part1" / "json" / "a.json").exists()
    assert (artifacts.data_dir / "part1" / "images" / "a.png").exists()
    assert artifacts.split_path("val").read_text(encoding="utf-8") == "part1/json/a.json\n"
    benchmark_payload = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    assert benchmark_payload["tasks"] == ["detection", "keypoint"]
    assert benchmark_payload["layers"] == ["layout"]
    assert benchmark_payload["labels"] == ["icon"]


def test_inference_params_validate_service_launcher_fields() -> None:
    params = InferenceParams(
        backend="vllm_openai",
        service_id="local-vllm-0",
        cuda_visible_devices="0,1",
        tensor_parallel_size=2,
        port=8000,
        max_model_len=65536,
        gpu_memory_utilization=0.9,
        max_num_seqs=16,
    )

    params.validate()

    with pytest.raises(ValueError, match="gpu_memory_utilization"):
        InferenceParams(gpu_memory_utilization=1.2).validate()
