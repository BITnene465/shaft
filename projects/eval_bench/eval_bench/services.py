from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import sys
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from typing import Any, Mapping

from .artifacts import DEFAULT_STORE_ROOT, StoreLayout
from .database import EvalBenchDatabase, ServiceListPage, ServiceRecord
from . import runtime_resources
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
        validate_no_service_pixel_budget_args(payload, context="service registration")
        _reject_unknown_service_payload_keys(payload)
        if kind == "local_vllm":
            placement = runtime_resources.resolve_vllm_runtime_placement(
                model_path=payload.get("model_path"),
                cuda_visible_devices=payload.get("cuda_visible_devices"),
                tensor_parallel_size=payload.get("tensor_parallel_size"),
            )
            cuda_visible_devices = placement.cuda_visible_devices
            tensor_parallel_size = placement.tensor_parallel_size
        else:
            cuda_visible_devices = _optional_string(payload, "cuda_visible_devices")
            tensor_parallel_size = _optional_int(payload, "tensor_parallel_size")
        config = {
            "model_path": _optional_string(payload, "model_path"),
            "served_model_name": _optional_string(payload, "served_model_name"),
            "endpoint": _optional_string(payload, "endpoint"),
            "host": str(payload.get("host") or "127.0.0.1"),
            "port": _optional_int(payload, "port"),
            "cuda_visible_devices": cuda_visible_devices,
            "tensor_parallel_size": tensor_parallel_size,
            "max_model_len": _optional_int(payload, "max_model_len"),
            "gpu_memory_utilization": _optional_float(payload, "gpu_memory_utilization"),
            "max_num_seqs": _optional_int(payload, "max_num_seqs"),
            "trust_remote_code": _optional_bool(payload, "trust_remote_code"),
            "generation_config": _optional_string(payload, "generation_config"),
            "dtype": _optional_string(payload, "dtype"),
            "kv_cache_dtype": _optional_string(payload, "kv_cache_dtype"),
            "quantization": _optional_string(payload, "quantization"),
            "load_format": _optional_string(payload, "load_format"),
            "enforce_eager": _optional_bool(payload, "enforce_eager"),
            "disable_custom_all_reduce": _optional_bool(payload, "disable_custom_all_reduce"),
            "max_num_batched_tokens": _optional_int(payload, "max_num_batched_tokens"),
            "limit_mm_per_prompt": payload.get("limit_mm_per_prompt"),
        }
        if kind == "local_vllm":
            if not config["model_path"]:
                raise ValueError("local_vllm service requires model_path.")
            if config["port"] is None:
                raise ValueError("local_vllm service requires port.")
            validate_no_service_pixel_budget_args(config)
        if kind == "external_vllm" and not config["endpoint"]:
            raise ValueError("external_vllm service requires endpoint.")
        config = {key: value for key, value in config.items() if value not in (None, [], "", False)}
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
    validate_no_service_pixel_budget_args(config)
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
        "generation_config": "--generation-config",
        "dtype": "--dtype",
        "kv_cache_dtype": "--kv-cache-dtype",
        "quantization": "--quantization",
        "load_format": "--load-format",
        "max_num_batched_tokens": "--max-num-batched-tokens",
        "limit_mm_per_prompt": "--limit-mm-per-prompt",
    }
    for key, flag in optional_flags.items():
        value = config.get(key)
        if value not in (None, ""):
            command.extend([flag, _stringify_cli_value(value)])
    boolean_flags = {
        "trust_remote_code": "--trust-remote-code",
        "enforce_eager": "--enforce-eager",
        "disable_custom_all_reduce": "--disable-custom-all-reduce",
    }
    for key, flag in boolean_flags.items():
        if config.get(key) is True:
            command.append(flag)
    return command


def validate_no_service_pixel_budget_args(
    config: Mapping[str, Any],
    *,
    context: str = "vLLM service config",
) -> None:
    blocked_keys = {
        "min_pixels",
        "min-pixels",
        "max_pixels",
        "max-pixels",
        "mm_processor_kwargs",
        "mm-processor-kwargs",
    }
    for key, value in config.items():
        if value is None or value == "" or value is False:
            continue
        normalized = str(key).strip().lower()
        if normalized in blocked_keys:
            raise ValueError(
                f"{context} must not set {key!r}; pixel budget belongs to eval.data "
                "and is applied before each request."
            )


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


def _optional_bool(payload: dict[str, Any], key: str) -> bool | None:
    value = payload.get(key)
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    raise ValueError(f"{key} must be a boolean when set.")


def _reject_unknown_service_payload_keys(payload: Mapping[str, Any]) -> None:
    allowed = {
        "kind",
        "service_id",
        "model_path",
        "served_model_name",
        "endpoint",
        "host",
        "port",
        "cuda_visible_devices",
        "tensor_parallel_size",
        "max_model_len",
        "gpu_memory_utilization",
        "max_num_seqs",
        "trust_remote_code",
        "generation_config",
        "dtype",
        "kv_cache_dtype",
        "quantization",
        "load_format",
        "enforce_eager",
        "disable_custom_all_reduce",
        "max_num_batched_tokens",
        "limit_mm_per_prompt",
    }
    unknown = sorted(str(key) for key in payload if str(key) not in allowed)
    if unknown:
        raise ValueError(f"unsupported service config key(s): {', '.join(unknown)}")


def _stringify_cli_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
