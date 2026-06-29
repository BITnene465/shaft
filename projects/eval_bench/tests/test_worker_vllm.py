from __future__ import annotations

import json
from pathlib import Path
import threading
import time

from eval_bench.adapters.vllm_openai import GeneratedText
from eval_bench.database import EvalBenchDatabase
from eval_bench.worker import (
    EvalBenchWorker,
)
from support.files import write_json as _write_json
from support.jobs import eval_job_payload as _eval_job_payload


def test_worker_vllm_openai_writes_predictions_raw_outputs_and_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class FakeAdapter:
        def __init__(self, **kwargs):
            assert kwargs["endpoint"] == "http://127.0.0.1:8000"
            assert kwargs["served_model_name"] == "served-model"

        def generate(self, **kwargs):
            assert kwargs["user_prompt"] == "detect icons"
            assert kwargs["max_tokens"] == 4096
            assert kwargs["max_pixels"] == 1048576
            assert kwargs["extra_body"] == {
                "chat_template_kwargs": {
                    "enable_thinking": False,
                    "preserve_thinking": False,
                }
            }
            return GeneratedText(
                text='[{"label":"icon","bbox_2d":[0,0,1000,1000]}]',
                latency_ms=12.5,
                raw_response={
                    "choices": [{"finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 9, "completion_tokens": 7, "total_tokens": 16},
                },
                image_request={"source_pixels": 5000, "target_pixels": 5000, "resized": False},
                finish_reason="stop",
                usage={"prompt_tokens": 9, "completion_tokens": 7, "total_tokens": 16},
            )

    monkeypatch.setattr("eval_bench.worker.OpenAICompatibleVLLMAdapter", FakeAdapter)
    split_path = tmp_path / "benchmarks" / "bench1" / "splits" / "val.txt"
    split_path.parent.mkdir(parents=True)
    split_path.write_text("part1/json/a.json\n", encoding="utf-8")
    data_root = tmp_path / "benchmarks" / "bench1" / "data"
    (data_root / "part1" / "images").mkdir(parents=True)
    (data_root / "part1" / "images" / "a.png").write_bytes(b"image")
    _write_json(
        tmp_path / "benchmarks" / "bench1" / "benchmark.json",
        {
            "benchmark_id": "bench1",
            "tasks": ["detection"],
            "layers": ["layout"],
            "split": "val",
            "sample_count": 1,
            "root": str(data_root),
            "manifest_path": str(split_path),
        },
    )
    _write_json(
        data_root / "part1" / "json" / "a.json",
        {
            "image_path": "part1/images/a.png",
            "image_width": 100,
            "image_height": 50,
            "instances": [{"label": "icon", "bbox": [0, 0, 100, 50]}],
        },
    )
    database = EvalBenchDatabase(tmp_path)
    payload = _eval_job_payload(
        model_id="model-a",
        model_path="outputs/model-a/best",
        served_model_name="served-model",
        benchmark_id="bench1",
        task="detection",
        prompt_id="grounding_layout.test.main",
        prompt_text="detect icons",
        backend="vllm_openai",
        endpoint="http://127.0.0.1:8000",
        max_tokens=4096,
        max_pixels=1048576,
    )
    payload["manifest"]["eval"]["inference_extra"] = {
        "extra_body": {
            "chat_template_kwargs": {
                "enable_thinking": False,
                "preserve_thinking": False,
            }
        }
    }
    job = database.create_job(
        kind="eval",
        payload=payload,
    )

    processed = EvalBenchWorker(tmp_path).process_next()

    assert processed is not None
    assert processed.status == "succeeded"
    assert processed.metadata["worker_action"] == "vllm_openai"
    assert processed.metadata["progress_phase"] == "succeeded"
    assert processed.metadata["progress_done"] == 1
    assert processed.metadata["progress_total"] == 1
    assert processed.metadata["progress_current_sample"] == "part1/json/a.json"
    prediction_path = tmp_path / "runs" / job.job_id / "predictions" / "part1" / "json" / "a.json"
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    assert prediction["instances"][0]["bbox"] == [0.0, 0.0, 100.0, 50.0]
    assert prediction["metadata"]["latency_ms"] == 12.5
    assert prediction["metadata"]["finish_reason"] == "stop"
    assert prediction["metadata"]["completion_tokens"] == 7
    assert prediction["metadata"]["prompt_tokens"] == 9
    assert prediction["metadata"]["total_tokens"] == 16
    assert prediction["metadata"]["truncated_by_max_tokens"] is False
    assert prediction["metadata"]["parser"]["decode_valid"] is True
    assert prediction["metadata"]["parser"]["decode_partial"] is False
    assert prediction["metadata"]["inference_params"]["max_pixels"] == 1048576
    assert prediction["metadata"]["inference_params"]["extra"]["extra_body"] == {
        "chat_template_kwargs": {
            "enable_thinking": False,
            "preserve_thinking": False,
        }
    }
    assert prediction["metadata"]["inference_params"]["image_request"]["resized"] is False
    assert (tmp_path / "runs" / job.job_id / "raw_outputs" / "part1" / "txt" / "a.txt").exists()
    report = json.loads(
        (tmp_path / "runs" / job.job_id / "reports" / "metrics.json").read_text(encoding="utf-8")
    )
    assert report["precision_iou50"] == 1.0
    assert report["recall_iou50"] == 1.0
    assert report["finish_reason_counts"] == {"stop": 1}
    assert report["truncated_prediction_count"] == 0
    assert report["partial_decode_count"] == 0
    assert report["decode_empty_repair_count"] == 0
    assert report["decode_error_count"] == 0


def test_worker_vllm_openai_runs_requests_concurrently(
    tmp_path: Path,
    monkeypatch,
) -> None:
    lock = threading.Lock()
    active = 0
    max_active = 0
    calls: list[str] = []

    class FakeAdapter:
        def __init__(self, **kwargs):
            pass

        def generate(self, **kwargs):
            nonlocal active, max_active
            image_name = Path(kwargs["image_path"]).name
            with lock:
                active += 1
                max_active = max(max_active, active)
                calls.append(image_name)
            time.sleep(0.05)
            with lock:
                active -= 1
            return GeneratedText(
                text='[{"label":"icon","bbox_2d":[0,0,1000,1000]}]',
                latency_ms=12.5,
                raw_response={"choices": []},
                image_request={"source_pixels": 5000, "target_pixels": 5000, "resized": False},
            )

    monkeypatch.setattr("eval_bench.worker.OpenAICompatibleVLLMAdapter", FakeAdapter)
    split_path = tmp_path / "benchmarks" / "bench1" / "splits" / "val.txt"
    split_path.parent.mkdir(parents=True)
    split_path.write_text(
        "\n".join(f"part1/json/{name}.json" for name in ["a", "b", "c", "d"]) + "\n",
        encoding="utf-8",
    )
    data_root = tmp_path / "benchmarks" / "bench1" / "data"
    (data_root / "part1" / "images").mkdir(parents=True)
    _write_json(
        tmp_path / "benchmarks" / "bench1" / "benchmark.json",
        {
            "benchmark_id": "bench1",
            "tasks": ["detection"],
            "layers": ["layout"],
            "split": "val",
            "sample_count": 4,
            "root": str(data_root),
            "manifest_path": str(split_path),
        },
    )
    for name in ["a", "b", "c", "d"]:
        (data_root / "part1" / "images" / f"{name}.png").write_bytes(b"image")
        _write_json(
            data_root / "part1" / "json" / f"{name}.json",
            {
                "image_path": f"part1/images/{name}.png",
                "image_width": 100,
                "image_height": 50,
                "instances": [{"label": "icon", "bbox": [0, 0, 100, 50]}],
            },
        )
    database = EvalBenchDatabase(tmp_path)
    job = database.create_job(
        kind="eval",
        payload=_eval_job_payload(
            model_id="model-a",
            model_path="outputs/model-a/best",
            served_model_name="served-model",
            benchmark_id="bench1",
            task="detection",
            prompt_id="grounding_layout.test.main",
            prompt_text="detect icons",
            backend="vllm_openai",
            endpoint="http://127.0.0.1:8000",
            max_tokens=4096,
            max_num_seqs=3,
            batch_size=1,
        ),
    )

    processed = EvalBenchWorker(tmp_path).process_next()

    assert processed is not None
    assert processed.status == "succeeded"
    assert max_active > 1
    assert sorted(calls) == ["a.png", "b.png", "c.png", "d.png"]
    prediction_path = tmp_path / "runs" / job.job_id / "predictions" / "part1" / "json" / "a.json"
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    assert prediction["metadata"]["inference_params"]["request_concurrency"] == 3
    assert len(list((tmp_path / "runs" / job.job_id / "predictions").rglob("*.json"))) == 4
