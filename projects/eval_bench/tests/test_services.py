from __future__ import annotations

from pathlib import Path

import pytest

from eval_bench import services as services_module
from eval_bench.services import EvalBenchServiceManager, build_vllm_command


def test_service_manager_registers_local_vllm_and_builds_command(tmp_path: Path) -> None:
    manager = EvalBenchServiceManager(tmp_path)
    record = manager.register_service(
        {
            "kind": "local_vllm",
            "service_id": "local-vllm-0",
            "model_path": "outputs/model/best",
            "served_model_name": "qwen3vl-best",
            "host": "127.0.0.1",
            "port": 8000,
            "cuda_visible_devices": "0,1",
            "tensor_parallel_size": 2,
            "max_model_len": 65536,
            "gpu_memory_utilization": 0.9,
            "max_num_seqs": 16,
            "extra_args": ["--trust-remote-code"],
        }
    )

    command = build_vllm_command(record)

    assert record.service_id == "local-vllm-0"
    assert record.config["cuda_visible_devices"] == "0,1"
    assert command[:3] == [command[0], "-m", "vllm.entrypoints.openai.api_server"]
    assert "--model" in command
    assert "outputs/model/best" in command
    assert "--tensor-parallel-size" in command
    assert "2" in command
    assert "--max-model-len" in command
    assert "65536" in command
    assert "--trust-remote-code" in command


def test_service_manager_rejects_incomplete_service_configs(tmp_path: Path) -> None:
    manager = EvalBenchServiceManager(tmp_path)

    with pytest.raises(ValueError, match="model_path"):
        manager.register_service({"kind": "local_vllm", "port": 8000})

    with pytest.raises(ValueError, match="endpoint"):
        manager.register_service({"kind": "external_vllm", "service_id": "external"})


def test_service_manager_exposes_launch_command_without_starting_process(tmp_path: Path) -> None:
    manager = EvalBenchServiceManager(tmp_path)
    manager.register_service(
        {
            "kind": "local_vllm",
            "service_id": "local-vllm-0",
            "model_path": "outputs/model/best",
            "port": 8000,
        }
    )

    command = manager.launch_command("local-vllm-0")

    assert command[1:4] == ["-m", "vllm.entrypoints.openai.api_server", "--model"]


def test_service_manager_health_probe_updates_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = EvalBenchServiceManager(tmp_path)
    manager.register_service(
        {
            "kind": "external_vllm",
            "service_id": "external",
            "endpoint": "http://127.0.0.1:8000/v1",
        }
    )

    def fake_probe(endpoint: str, *, timeout_s: float) -> dict:
        return {
            "ok": True,
            "status": "ready",
            "status_code": 200,
            "url": f"{endpoint}/models",
            "message": "HTTP 200",
            "checked_at": "2026-05-09T00:00:00Z",
        }

    monkeypatch.setattr(services_module, "_probe_openai_endpoint", fake_probe)

    record = manager.check_service_health("external")

    assert record.status == "running"
    assert record.runtime["health"]["ok"] is True
    assert record.runtime["health"]["status_code"] == 200


def test_service_manager_detects_dead_local_process(tmp_path: Path) -> None:
    manager = EvalBenchServiceManager(tmp_path)
    manager.register_service(
        {
            "kind": "local_vllm",
            "service_id": "local-vllm-0",
            "model_path": "outputs/model/best",
            "port": 8000,
        }
    )
    manager.database.update_service_runtime(
        "local-vllm-0",
        status="running",
        runtime={"pid": 999_999_999, "endpoint": "http://127.0.0.1:8000"},
    )

    record = manager.check_service_health("local-vllm-0")

    assert record.status == "stopped"
    assert record.runtime["health"]["status"] == "stopped"


def test_service_manager_reads_log_tail(tmp_path: Path) -> None:
    manager = EvalBenchServiceManager(tmp_path)
    manager.register_service(
        {
            "kind": "local_vllm",
            "service_id": "local-vllm-0",
            "model_path": "outputs/model/best",
            "port": 8000,
        }
    )
    log_path = tmp_path / "services" / "local-vllm-0" / "service.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text("line1\nline2\nline3\n", encoding="utf-8")
    manager.database.update_service_runtime(
        "local-vllm-0",
        status="running",
        runtime={"log_path": str(log_path), "pid": 999_999_999},
    )

    payload = manager.service_log("local-vllm-0", max_lines=2)

    assert payload["log_path"] == str(log_path)
    assert payload["lines"] == ["line2\n", "line3\n"]
    assert payload["text"] == "line2\nline3\n"
