from __future__ import annotations

from pathlib import Path

from eval_bench.job_spec import job_templates, preflight_job_payload
from projects.eval_bench.tests.support.files import write_json


def test_preflight_checks_benchmark_model_and_command(tmp_path: Path) -> None:
    model_path = tmp_path / "outputs" / "model" / "best"
    model_path.mkdir(parents=True)
    benchmark_root = tmp_path / "benchmarks" / "bench1"
    (benchmark_root / "splits").mkdir(parents=True)
    (benchmark_root / "splits" / "val.txt").write_text("part1/json/a.json\n", encoding="utf-8")
    write_json(
        benchmark_root / "benchmark.json",
        {
            "benchmark_id": "bench1",
            "tasks": ["detection"],
            "layers": ["layout"],
            "split": "val",
            "sample_count": 1,
            "root": str(benchmark_root / "data"),
            "manifest_path": str(benchmark_root / "splits" / "val.txt"),
        },
    )
    manifest = job_templates()["eval_job"]["manifest"]
    manifest["runtime"]["args"]["model"] = str(model_path)
    manifest["runtime"]["args"]["port"] = 65530
    manifest["eval"]["benchmark_id"] = "bench1"

    result = preflight_job_payload({"manifest": manifest}, store_root=tmp_path)

    assert result["ok"] is True
    assert result["errors"] == []
    assert result["resolved_payload"]["benchmark_id"] == "bench1"
    assert "--model" in result["runtime_command"]


def test_preflight_rejects_missing_benchmark_split_manifest(tmp_path: Path) -> None:
    model_path = tmp_path / "outputs" / "model" / "best"
    model_path.mkdir(parents=True)
    benchmark_root = tmp_path / "benchmarks" / "bench1"
    write_json(
        benchmark_root / "benchmark.json",
        {
            "benchmark_id": "bench1",
            "tasks": ["detection"],
            "layers": ["layout"],
            "split": "val",
            "sample_count": 1,
            "root": str(benchmark_root / "data"),
            "manifest_path": str(benchmark_root / "splits" / "val.txt"),
        },
    )
    manifest = job_templates()["eval_job"]["manifest"]
    manifest["runtime"]["args"]["model"] = str(model_path)
    manifest["eval"]["benchmark_id"] = "bench1"

    result = preflight_job_payload({"manifest": manifest}, store_root=tmp_path)

    assert result["ok"] is False
    assert any("benchmark split manifest does not exist" in error for error in result["errors"])


def test_preflight_reports_missing_required_eval_inputs(tmp_path: Path) -> None:
    result = preflight_job_payload(
        {"manifest": {"kind": "eval_job", "runtime": {}, "eval": {}}},
        store_root=tmp_path,
    )

    assert result["ok"] is False
    assert any("model_id" in error for error in result["errors"])
    assert any("benchmark_id" in error for error in result["errors"])


def test_preflight_rejects_tensor_parallel_size_not_dividing_attention_heads(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "outputs" / "model" / "best"
    model_path.mkdir(parents=True)
    write_json(model_path / "config.json", {"text_config": {"num_attention_heads": 32}})
    benchmark_root = tmp_path / "benchmarks" / "bench1"
    write_json(
        benchmark_root / "benchmark.json",
        {
            "benchmark_id": "bench1",
            "tasks": ["detection"],
            "layers": ["arrow"],
            "split": "val",
            "sample_count": 1,
            "root": str(benchmark_root / "data"),
            "manifest_path": str(benchmark_root / "splits" / "val.txt"),
        },
    )
    manifest = job_templates()["eval_job"]["manifest"]
    manifest["runtime"]["args"]["model"] = str(model_path)
    manifest["runtime"]["args"]["tensor-parallel-size"] = 3
    manifest["eval"]["benchmark_id"] = "bench1"

    result = preflight_job_payload({"manifest": manifest}, store_root=tmp_path)

    assert result["ok"] is False
    assert any("attention heads" in error and "tp=3" in error for error in result["errors"])


def test_preflight_rejects_unwired_preannotate_job(tmp_path: Path) -> None:
    source_root = tmp_path / "raw_data"
    source_manifest = source_root / "splits" / "val.txt"
    source_manifest.parent.mkdir(parents=True)
    source_manifest.write_text("", encoding="utf-8")

    result = preflight_job_payload(
        {
            "manifest": {
                "kind": "preannotate_job",
                "runtime": {"mode": "existing_service", "engine": "vllm_openai"},
                "preannotate": {
                    "source_root": str(source_root),
                    "source_manifest": str(source_manifest),
                    "output_root": str(tmp_path / "preannotations"),
                    "task": "detection",
                    "prompt_id": "grounding_layout.test.main",
                    "inference_extra": {
                        "extra_body": {
                            "chat_template_kwargs": {
                                "enable_thinking": False,
                                "preserve_thinking": False,
                            }
                        }
                    },
                },
            }
        },
        store_root=tmp_path,
    )

    assert result["ok"] is False
    assert result["resolved_payload"]["inference_extra"] == {
        "extra_body": {
            "chat_template_kwargs": {
                "enable_thinking": False,
                "preserve_thinking": False,
            }
        }
    }
    assert any("preannotate execution is not wired" in error for error in result["errors"])
