from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from typing import Any

from .artifacts import DEFAULT_STORE_ROOT, StoreLayout
from .database import EvalBenchDatabase, ServiceListPage, ServiceRecord
from .schema import utc_now_iso


REPO_ROOT = Path(__file__).resolve().parents[3]


class EvalBenchServiceManager:
    def __init__(self, root: str | Path = DEFAULT_STORE_ROOT) -> None:
        self.layout = StoreLayout(root)
        self.database = EvalBenchDatabase(root)

    def service_page(
        self,
        *,
        offset: int = 0,
        limit: int = 100,
        kind: str | None = None,
        status: str | None = None,
        query: str | None = None,
    ) -> ServiceListPage:
        for record in self.database.list_services(limit=1000):
            self._refresh_if_needed(record)
        return self.database.service_page(
            offset=offset,
            limit=limit,
            kind=kind,
            status=status,
            query=query,
        )

    def list_services(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        kind: str | None = None,
        status: str | None = None,
        query: str | None = None,
    ) -> list[ServiceRecord]:
        return self.service_page(
            offset=offset,
            limit=limit,
            kind=kind,
            status=status,
            query=query,
        ).services

    def service(self, service_id: str) -> ServiceRecord:
        return self._refresh_if_needed(self._require_service(service_id))

    def register_service(self, payload: dict[str, Any]) -> ServiceRecord:
        kind = str(payload.get("kind") or "local_vllm")
        service_id = _optional_string(payload, "service_id")
        config = {
            "model_path": _optional_string(payload, "model_path"),
            "served_model_name": _optional_string(payload, "served_model_name"),
            "endpoint": _optional_string(payload, "endpoint"),
            "host": str(payload.get("host") or "127.0.0.1"),
            "port": _optional_int(payload, "port"),
            "cuda_visible_devices": _optional_string(payload, "cuda_visible_devices"),
            "tensor_parallel_size": _optional_int(payload, "tensor_parallel_size"),
            "max_model_len": _optional_int(payload, "max_model_len"),
            "gpu_memory_utilization": _optional_float(payload, "gpu_memory_utilization"),
            "max_num_seqs": _optional_int(payload, "max_num_seqs"),
            "extra_args": _string_list(payload.get("extra_args")),
        }
        if kind == "local_vllm":
            if not config["model_path"]:
                raise ValueError("local_vllm service requires model_path.")
            if config["port"] is None:
                raise ValueError("local_vllm service requires port.")
        if kind == "external_vllm" and not config["endpoint"]:
            raise ValueError("external_vllm service requires endpoint.")
        config = {key: value for key, value in config.items() if value not in (None, [], "")}
        return self.database.upsert_service(kind=kind, config=config, service_id=service_id)

    def start_service(self, service_id: str) -> ServiceRecord:
        record = self._require_service(service_id)
        record = self._refresh_if_needed(record)
        if record.kind != "local_vllm":
            raise ValueError("Only local_vllm services can be started by Eval Bench.")
        if record.status in {"starting", "running"}:
            return record
        command = build_vllm_command(record)
        service_dir = self.layout.services_dir / record.service_id
        service_dir.mkdir(parents=True, exist_ok=True)
        log_path = service_dir / "service.log"
        env = os.environ.copy()
        cuda_devices = record.config.get("cuda_visible_devices")
        if cuda_devices:
            env["CUDA_VISIBLE_DEVICES"] = str(cuda_devices)
        log_file = log_path.open("ab")
        try:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            log_file.close()
        endpoint = str(record.config.get("endpoint") or _endpoint_from_config(record.config))
        return self.database.update_service_runtime(
            record.service_id,
            status="starting",
            runtime={
                "pid": process.pid,
                "command": command,
                "endpoint": endpoint,
                "log_path": str(log_path),
                "started_at": utc_now_iso(),
                "health": {
                    "ok": False,
                    "status": "starting",
                    "message": "process launched; endpoint not checked yet",
                    "checked_at": utc_now_iso(),
                },
            },
            error=None,
        )

    def stop_service(self, service_id: str) -> ServiceRecord:
        record = self._require_service(service_id)
        pid = _optional_int(record.runtime, "pid")
        if pid is not None and _pid_is_alive(pid):
            try:
                os.killpg(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except PermissionError as exc:
                return self.database.update_service_runtime(
                    record.service_id,
                    status="failed",
                    error=str(exc),
                )
        return self.database.update_service_runtime(
            record.service_id,
            status="stopped",
            runtime={
                "stopped_at": utc_now_iso(),
                "health": {
                    "ok": False,
                    "status": "stopped",
                    "message": "service stopped by Eval Bench",
                    "checked_at": utc_now_iso(),
                },
            },
            error=None,
        )

    def delete_service(self, service_id: str) -> dict[str, Any]:
        record = self._require_service(service_id)
        if record.status in {"starting", "running"}:
            self.stop_service(service_id)
        deleted = self.database.delete_service(service_id)
        trash_path = self.layout.move_to_trash(
            self.layout.services_dir / deleted.service_id,
            category="services",
        )
        return {
            "service": deleted.to_dict(),
            "trash_path": str(trash_path) if trash_path is not None else None,
        }

    def check_service_health(self, service_id: str, *, timeout_s: float = 2.0) -> ServiceRecord:
        record = self._require_service(service_id)
        if record.kind == "local_vllm":
            pid = _optional_int(record.runtime, "pid")
            if pid is not None and not _pid_is_alive(pid):
                return self.database.update_service_runtime(
                    record.service_id,
                    status="stopped",
                    runtime={
                        "health": {
                            "ok": False,
                            "status": "stopped",
                            "message": f"process {pid} is not alive",
                            "checked_at": utc_now_iso(),
                        },
                        "stopped_at": utc_now_iso(),
                    },
                    error=None,
                )
        endpoint = service_endpoint(record)
        health = _probe_openai_endpoint(endpoint, timeout_s=timeout_s)
        status = "running" if health["ok"] else _unhealthy_status(record)
        return self.database.update_service_runtime(
            record.service_id,
            status=status,
            runtime={
                "endpoint": endpoint,
                "health": health,
            },
            error=None if health["ok"] else str(health["message"]),
        )

    def service_log(self, service_id: str, *, max_lines: int = 200) -> dict[str, Any]:
        record = self._require_service(service_id)
        log_path_value = record.runtime.get("log_path")
        if not log_path_value:
            return {
                "service_id": record.service_id,
                "log_path": None,
                "lines": [],
                "text": "",
            }
        log_path = Path(str(log_path_value))
        if not log_path.exists():
            return {
                "service_id": record.service_id,
                "log_path": str(log_path),
                "lines": [],
                "text": "",
            }
        lines = _tail_lines(log_path, max_lines=max_lines)
        return {
            "service_id": record.service_id,
            "log_path": str(log_path),
            "lines": lines,
            "text": "".join(lines),
        }

    def launch_command(self, service_id: str) -> list[str]:
        record = self._require_service(service_id)
        if record.kind != "local_vllm":
            raise ValueError("Only local_vllm services have a local launch command.")
        return build_vllm_command(record)

    def _require_service(self, service_id: str) -> ServiceRecord:
        record = self.database.get_service(service_id)
        if record is None:
            raise KeyError(f"unknown service_id: {service_id}")
        return record

    def _refresh_if_needed(self, record: ServiceRecord) -> ServiceRecord:
        if record.status not in {"starting", "running"}:
            return record
        pid = _optional_int(record.runtime, "pid")
        if pid is not None and _pid_is_alive(pid):
            return record
        return self.database.update_service_runtime(
            record.service_id,
            status="stopped",
            runtime={
                "stopped_at": utc_now_iso(),
                "health": {
                    "ok": False,
                    "status": "stopped",
                    "message": f"process {pid} is not alive",
                    "checked_at": utc_now_iso(),
                },
            },
            error=None,
        )


def build_vllm_command(record: ServiceRecord) -> list[str]:
    return build_vllm_command_from_config(record.config, service_id=record.service_id)


def build_vllm_command_from_config(config: dict[str, Any], *, service_id: str) -> list[str]:
    python = REPO_ROOT / ".venv" / "bin" / "python"
    python_bin = str(python if python.exists() else Path(sys.executable))
    command = [
        python_bin,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        _required_config(config, "model_path"),
        "--served-model-name",
        str(config.get("served_model_name") or service_id),
        "--host",
        str(config.get("host") or "127.0.0.1"),
        "--port",
        str(_required_config(config, "port")),
    ]
    optional_flags = {
        "tensor_parallel_size": "--tensor-parallel-size",
        "max_model_len": "--max-model-len",
        "gpu_memory_utilization": "--gpu-memory-utilization",
        "max_num_seqs": "--max-num-seqs",
    }
    for key, flag in optional_flags.items():
        value = config.get(key)
        if value not in (None, ""):
            command.extend([flag, str(value)])
    for item in _string_list(config.get("extra_args")):
        command.append(item)
    return command


def service_endpoint(record: ServiceRecord) -> str:
    return str(
        record.runtime.get("endpoint")
        or record.config.get("endpoint")
        or _endpoint_from_config(record.config)
    )


def _endpoint_from_config(config: dict[str, Any]) -> str:
    host = str(config.get("host") or "127.0.0.1")
    port = str(config.get("port") or "8000")
    return f"http://{host}:{port}"


def _models_url_from_endpoint(endpoint: str) -> str:
    normalized = endpoint.strip().rstrip("/")
    if normalized.endswith("/v1/chat/completions"):
        normalized = normalized.removesuffix("/chat/completions")
    elif not normalized.endswith("/v1"):
        normalized = f"{normalized}/v1"
    return f"{normalized}/models"


def _probe_openai_endpoint(endpoint: str, *, timeout_s: float) -> dict[str, Any]:
    checked_at = utc_now_iso()
    url = _models_url_from_endpoint(endpoint)
    request = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=timeout_s) as response:
            status_code = int(response.status)
            return {
                "ok": 200 <= status_code < 300,
                "status": "ready" if 200 <= status_code < 300 else "unavailable",
                "status_code": status_code,
                "url": url,
                "message": f"HTTP {status_code}",
                "checked_at": checked_at,
            }
    except HTTPError as exc:
        status_code = int(exc.code)
        return {
            "ok": 200 <= status_code < 300,
            "status": "ready" if 200 <= status_code < 300 else "unavailable",
            "status_code": status_code,
            "url": url,
            "message": f"HTTP {status_code}",
            "checked_at": checked_at,
        }
    except (OSError, TimeoutError, URLError) as exc:
        return {
            "ok": False,
            "status": "unavailable",
            "status_code": None,
            "url": url,
            "message": str(exc),
            "checked_at": checked_at,
        }


def _unhealthy_status(record: ServiceRecord) -> str:
    if record.kind == "local_vllm":
        pid = _optional_int(record.runtime, "pid")
        if pid is not None and _pid_is_alive(pid):
            return "starting"
    return "failed"


def _tail_lines(path: Path, *, max_lines: int) -> list[str]:
    if max_lines <= 0:
        raise ValueError("max_lines must be > 0.")
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return handle.readlines()[-max_lines:]


def _required_config(config: dict[str, Any], key: str) -> str:
    value = config.get(key)
    if value in (None, ""):
        raise ValueError(f"service config requires {key}.")
    return str(value)


def _optional_string(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string when set.")
    return value.strip() or None


def _optional_int(payload: dict[str, Any], key: str) -> int | None:
    value = payload.get(key)
    if value in (None, ""):
        return None
    return int(value)


def _optional_float(payload: dict[str, Any], key: str) -> float | None:
    value = payload.get(key)
    if value in (None, ""):
        return None
    return float(value)


def _string_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [item for item in value.split() if item]
    if isinstance(value, list):
        return [str(item) for item in value]
    raise ValueError("extra_args must be a list or a shell-like string.")


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
