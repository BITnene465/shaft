from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from eval_bench import services as services_module
from eval_bench.dashboard import create_app


pytestmark = pytest.mark.contract


def test_dashboard_exposes_service_registry(tmp_path: Path) -> None:
    app = create_app(store_root=tmp_path, frontend_dist=tmp_path / "dist")
    client = TestClient(app)

    created = client.post(
        "/api/services",
        json={
            "kind": "local_vllm",
            "service_id": "local-vllm-0",
            "model_path": "outputs/model/best",
            "served_model_name": "qwen3vl-best",
            "port": 8000,
            "cuda_visible_devices": "0,1",
            "tensor_parallel_size": 2,
        },
    )

    assert created.status_code == 201
    payload = created.json()
    assert payload["service_id"] == "local-vllm-0"
    assert payload["config"]["model_path"] == "outputs/model/best"

    services_payload = client.get("/api/services").json()
    services = services_payload["services"]
    assert services_payload["total"] == 1
    assert services[0]["service_id"] == "local-vllm-0"
    filtered = client.get(
        "/api/services",
        params={"kind": "local_vllm", "status": "registered", "query": "qwen3vl"},
    ).json()
    assert filtered["filters"] == {
        "kind": "local_vllm",
        "status": "registered",
        "query": "qwen3vl",
    }
    assert filtered["total"] == 1
    assert filtered["services"][0]["service_id"] == "local-vllm-0"

    detail = client.get("/api/services/local-vllm-0")
    assert detail.status_code == 200
    assert detail.json()["service"]["service_id"] == "local-vllm-0"
    assert client.get("/api/services/not_found").status_code == 404

    command = client.get("/api/services/local-vllm-0/command").json()["command"]
    assert command[1:4] == ["-m", "vllm.entrypoints.openai.api_server", "--model"]
    assert "outputs/model/best" in command

    invalid = client.post("/api/services", json={"kind": "external_vllm"})
    assert invalid.status_code == 400


def test_dashboard_exposes_service_health_and_logs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    app = create_app(store_root=tmp_path, frontend_dist=tmp_path / "dist")
    client = TestClient(app)
    client.post(
        "/api/services",
        json={
            "kind": "external_vllm",
            "service_id": "external",
            "endpoint": "http://127.0.0.1:8000/v1",
        },
    )

    health = client.post("/api/services/external/health").json()
    assert health["status"] == "running"
    assert health["runtime"]["health"]["ok"] is True

    logs = client.get("/api/services/external/logs").json()
    assert logs["service_id"] == "external"
    assert logs["lines"] == []
    assert logs["text"] == ""
