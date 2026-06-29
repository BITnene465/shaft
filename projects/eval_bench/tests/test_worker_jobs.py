from __future__ import annotations

import json
from pathlib import Path

from eval_bench.database import EvalBenchDatabase
from eval_bench.sample_paths import sample_image_path
from eval_bench.worker import (
    EvalBenchWorker,
    _resolve_prompt,
)
from support.files import write_json as _write_json
from support.jobs import ephemeral_eval_job_payload as _ephemeral_eval_job_payload
from support.jobs import eval_job_payload as _eval_job_payload


def test_image_path_from_gt_uses_existing_non_png_suffix(tmp_path: Path) -> None:
    image_path = tmp_path / "part2" / "images" / "prod_000876.jpg"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"image")

    resolved = sample_image_path(
        Path("part2/json/prod_000876.json"),
        {},
        root=tmp_path,
    )

    assert resolved == Path("part2/images/prod_000876.jpg")


def test_worker_preserves_cancelled_job_status(tmp_path: Path) -> None:
    database = EvalBenchDatabase(tmp_path)
    job = database.create_job(
        kind="eval",
        payload=_ephemeral_eval_job_payload(),
        status="running",
    )
    database.cancel_job(job.job_id)

    processed = EvalBenchWorker(tmp_path).process_job(job.job_id)

    assert processed.status == "cancelled"
    assert processed.metadata["progress_phase"] == "cancelled"
    assert processed.metadata["worker_action"] == "cancelled"


def test_worker_prepares_run_manifest_from_queued_job(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "benchmarks" / "bench1" / "benchmark.json",
        {
            "benchmark_id": "bench1",
            "tasks": ["detection", "keypoint"],
            "layers": ["layout", "arrow"],
            "split": "val",
            "sample_count": 1,
            "root": str(tmp_path / "benchmarks" / "bench1" / "data"),
            "manifest_path": str(tmp_path / "benchmarks" / "bench1" / "splits" / "val.txt"),
        },
    )
    database = EvalBenchDatabase(tmp_path)
    job = database.create_job(
        kind="eval",
        payload=_eval_job_payload(
            model_id="model-a",
            model_path="outputs/model-a/best",
            benchmark_id="bench1",
            task="keypoint",
            prompt_id="point_arrow.test.main",
            system_prompt="system snapshot",
            prompt_text="predict arrow endpoints",
            service_id="local-vllm-0",
            cuda_visible_devices="0,1,2",
            tensor_parallel_size=3,
            port=8001,
            max_model_len=65536,
            gpu_memory_utilization=0.82,
            max_num_seqs=16,
            trust_remote_code=True,
            generation_config="vllm",
            dtype="bfloat16",
            kv_cache_dtype="auto",
            load_format="auto",
            max_num_batched_tokens=8192,
            max_tokens=4096,
            temperature=0.1,
            top_p=0.9,
            max_pixels=1048576,
            batch_size=2,
        ),
    )

    processed = EvalBenchWorker(tmp_path).process_next()

    assert processed is not None
    assert processed.job_id == job.job_id
    assert processed.status == "succeeded"
    assert "resolved_manifest" not in processed.metadata
    run_path = tmp_path / "runs" / job.job_id / "run.json"
    run_payload = json.loads(run_path.read_text(encoding="utf-8"))
    assert run_payload["status"] == "queued"
    assert run_payload["spec"]["task"] == "keypoint"
    assert run_payload["spec"]["target_labels"] == ["arrow"]
    assert run_payload["spec"]["metadata"]["target_labels_source"] == "suite_default"
    assert run_payload["spec"]["inference"]["batch_size"] == 2
    assert run_payload["spec"]["inference"]["service_id"] == "local-vllm-0"
    assert run_payload["spec"]["inference"]["cuda_visible_devices"] == "0,1,2"
    assert run_payload["spec"]["inference"]["tensor_parallel_size"] == 3
    assert run_payload["spec"]["inference"]["port"] == 8001
    assert run_payload["spec"]["inference"]["max_model_len"] == 65536
    assert run_payload["spec"]["inference"]["gpu_memory_utilization"] == 0.82
    assert run_payload["spec"]["inference"]["max_num_seqs"] == 16
    assert run_payload["spec"]["inference"]["trust_remote_code"] is True
    assert run_payload["spec"]["inference"]["generation_config"] == "vllm"
    assert run_payload["spec"]["inference"]["dtype"] == "bfloat16"
    assert run_payload["spec"]["inference"]["kv_cache_dtype"] == "auto"
    assert run_payload["spec"]["inference"]["load_format"] == "auto"
    assert run_payload["spec"]["inference"]["max_num_batched_tokens"] == 8192
    assert run_payload["spec"]["inference"]["temperature"] == 0.1
    assert run_payload["spec"]["inference"]["top_p"] == 0.9
    assert run_payload["spec"]["inference"]["max_pixels"] == 1048576
    assert run_payload["spec"]["prompt"]["prompt_id"] == "point_arrow.test.main"
    assert run_payload["spec"]["prompt"]["text_hash"]
    assert run_payload["spec"]["prompt"]["metadata"]["source"] == "inline"
    assert run_payload["spec"]["prompt"]["metadata"]["system_prompt"] == "system snapshot"
    assert run_payload["spec"]["prompt"]["metadata"]["user_prompt"] == "predict arrow endpoints"
    assert run_payload["metadata"]["source_job_id"] == job.job_id
    assert "job_manifest" not in run_payload["metadata"]


def test_worker_marks_invalid_job_failed(tmp_path: Path) -> None:
    database = EvalBenchDatabase(tmp_path)
    job = database.create_job(
        kind="eval",
        payload=_eval_job_payload(
            model_id="model-a",
            model_path="outputs/model-a/best",
            benchmark_id="missing",
            task="detection",
            prompt_id="grounding_layout.test.main",
        ),
    )

    processed = EvalBenchWorker(tmp_path).process_next()

    assert processed is not None
    assert processed.job_id == job.job_id
    assert processed.status == "failed"
    assert "benchmark manifest does not exist" in str(processed.error)


def test_worker_default_prompt_resolution_is_repo_rooted(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    system_prompt, user_prompt, prompt_id = _resolve_prompt(
        {"prompt_id": "grounding_layout.test.main"},
        task="detection",
    )

    assert "Return only valid compact JSON" in system_prompt
    assert "Detect all visible top-level layout elements" in user_prompt
    assert prompt_id.startswith("shaft.grounding_layout.prompt_pool.")
    assert prompt_id.endswith(".main")


def test_worker_dry_run_writes_predictions_and_report(tmp_path: Path) -> None:
    split_path = tmp_path / "benchmarks" / "bench1" / "splits" / "val.txt"
    split_path.parent.mkdir(parents=True)
    split_path.write_text("part1/json/a.json\n", encoding="utf-8")
    _write_json(
        tmp_path / "benchmarks" / "bench1" / "benchmark.json",
        {
            "benchmark_id": "bench1",
            "tasks": ["detection", "keypoint"],
            "layers": ["layout", "arrow"],
            "split": "val",
            "sample_count": 1,
            "root": str(tmp_path / "benchmarks" / "bench1" / "data"),
            "manifest_path": str(split_path),
        },
    )
    _write_json(
        tmp_path / "benchmarks" / "bench1" / "data" / "part1" / "json" / "a.json",
        {
            "image_path": "part1/images/a.png",
            "instances": [{"label": "icon", "bbox": [0, 0, 10, 10]}],
        },
    )
    database = EvalBenchDatabase(tmp_path)
    job = database.create_job(
        kind="eval",
        payload=_eval_job_payload(
            model_id="dry-model",
            model_path="outputs/dry/best",
            benchmark_id="bench1",
            task="detection",
            prompt_id="grounding_layout.test.main",
            backend="dry_run",
            runtime_mode="external",
            metadata={"notes": "checkpoint=5000; full benchmark sweep"},
        ),
    )

    processed = EvalBenchWorker(tmp_path).process_next()

    assert processed is not None
    assert processed.status == "succeeded"
    assert processed.metadata["worker_action"] == "dry_run"
    assert processed.metadata["progress_phase"] == "succeeded"
    assert processed.metadata["progress_done"] == 1
    assert processed.metadata["progress_total"] == 1
    assert processed.metadata["progress_current_sample"] == "part1/json/a.json"
    assert processed.metadata["run_note_path"].endswith("runs/" + job.job_id + "/note.json")
    note = json.loads((tmp_path / "runs" / job.job_id / "note.json").read_text(encoding="utf-8"))
    assert note["note"] == "checkpoint=5000; full benchmark sweep"
    assert (tmp_path / "runs" / job.job_id / "predictions" / "part1" / "json" / "a.json").exists()
    report_path = tmp_path / "runs" / job.job_id / "reports" / "metrics.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["prediction_file_count"] == 1
    assert report["target_labels"] == ["icon", "image", "shape"]
    assert report["target_labels_source"] == "suite_default"
    assert report["recall_iou50"] == 0.0
    run_payload = json.loads(
        (tmp_path / "runs" / job.job_id / "run.json").read_text(encoding="utf-8")
    )
    assert run_payload["status"] == "succeeded"
