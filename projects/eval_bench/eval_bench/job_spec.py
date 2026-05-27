from __future__ import annotations

from dataclasses import dataclass
import copy
import json
from pathlib import Path
import socket
from typing import Any, Mapping

from .artifacts import BenchmarkArtifacts, DEFAULT_STORE_ROOT
from .benchmark import resolve_benchmark_split_name, resolve_benchmark_split_path
from .label_policy import (
    TARGET_LABEL_SOURCES,
    TargetLabelPolicy,
    normalize_target_labels,
    resolve_target_label_policy,
    target_label_benchmark_messages,
    target_label_task_errors,
)
from .services import build_vllm_command_from_config


REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class ResolvedJobSpec:
    kind: str
    manifest: dict[str, Any]
    payload: dict[str, Any]


def job_templates() -> dict[str, Any]:
    return {
        "eval_job": {
            "label": "Arrow Detection Eval Job",
            "description": "启动临时模型 runtime，在 benchmark 上执行箭头检测并生成 run/report。",
            "manifest": _eval_job_manifest(
                task="detection",
                prompt_id="grounding_arrow.latest",
                parser="raw_data_detection_v1",
                metric_profile="detection_iou_v1",
                target_labels=["arrow"],
            ),
        },
        "layout_eval_job": {
            "label": "Layout Detection Eval Job",
            "description": "启动临时模型 runtime，在 benchmark 上执行 layout 检测并生成 run/report。",
            "manifest": _eval_job_manifest(
                task="detection",
                prompt_id="grounding_layout.latest",
                parser="raw_data_detection_v1",
                metric_profile="detection_iou_v1",
                target_labels=["icon", "image", "shape"],
            ),
        },
        "keypoint_eval_job": {
            "label": "Arrow Point Eval Job",
            "description": "启动临时模型 runtime，在 benchmark 上执行箭头关键点评估并生成 run/report。",
            "manifest": _eval_job_manifest(
                task="keypoint",
                prompt_id="point_arrow.latest",
                parser="raw_data_keypoint_v1",
                metric_profile="keypoint_endpoint_v1",
                target_labels=["arrow"],
            ),
        },
    }


def _eval_job_manifest(
    *,
    task: str,
    prompt_id: str,
    parser: str,
    metric_profile: str,
    target_labels: list[str],
) -> dict[str, Any]:
    return {
        "kind": "eval_job",
        "runtime": {
            "mode": "ephemeral",
            "engine": "vllm_openai",
            "env": {"CUDA_VISIBLE_DEVICES": "", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"},
            "args": {
                "model": "",
                "served-model-name": "",
                "host": "127.0.0.1",
                "port": None,
                "tensor-parallel-size": None,
                "max-model-len": 32768,
                "gpu-memory-utilization": 0.9,
                "max-num-seqs": 8,
                "trust-remote-code": True,
            },
        },
        "eval": {
            "model_id": "",
            "benchmark_id": "",
            "benchmark_split": "",
            "task": task,
            "prompt_id": prompt_id,
            "parser": parser,
            "metric_profile": metric_profile,
            "target_labels": target_labels,
            "generation": {
                "max_tokens": 4096,
                "temperature": 0,
                "top_p": 1,
            },
            "data": {
                "max_pixels": 2_000_000,
                "batch_size": 1,
            },
        },
    }


def resolve_job_payload(
    payload: dict[str, Any],
    *,
    prompt_templates: Mapping[str, Mapping[str, Any]] | None = None,
) -> ResolvedJobSpec:
    manifest = _apply_prompt_template_to_manifest(
        _manifest_from_payload(payload),
        payload=payload,
        prompt_templates=prompt_templates,
    )
    kind = str(manifest.get("kind") or payload.get("kind") or "eval_job")
    if kind == "eval":
        kind = "eval_job"
    if kind == "preannotate":
        kind = "preannotate_job"
    if kind == "eval_job":
        resolved_payload = _resolve_eval_payload(payload, manifest)
    elif kind == "preannotate_job":
        resolved_payload = _resolve_preannotate_payload(payload, manifest)
    else:
        raise ValueError(f"unsupported job kind: {kind}")
    return ResolvedJobSpec(kind=kind, manifest=manifest, payload=resolved_payload)


def preflight_job_payload(
    payload: dict[str, Any],
    *,
    store_root: str | Path = DEFAULT_STORE_ROOT,
    prompt_templates: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    command: list[str] | None = None
    resolved: ResolvedJobSpec | None = None
    try:
        resolved = resolve_job_payload(payload, prompt_templates=prompt_templates)
    except ValueError as exc:
        return {
            "ok": False,
            "errors": [str(exc)],
            "warnings": [],
            "kind": "",
            "resolved_manifest": None,
            "resolved_payload": None,
            "runtime_command": None,
        }

    if resolved.kind == "eval_job":
        _check_eval_payload(
            resolved.payload,
            store_root=Path(store_root),
            errors=errors,
            warnings=warnings,
            prompt_templates=prompt_templates,
        )
    elif resolved.kind == "preannotate_job":
        _check_preannotate_payload(resolved.payload, errors=errors, warnings=warnings)

    runtime = dict(resolved.manifest.get("runtime") or {})
    if str(runtime.get("mode") or "ephemeral") == "ephemeral":
        try:
            command = build_vllm_command_from_config(
                _runtime_config_from_payload(resolved.payload),
                service_id=str(resolved.payload.get("service_id") or resolved.payload.get("model_id") or "job"),
            )
        except ValueError as exc:
            errors.append(str(exc))
        port = _optional_int(resolved.payload, "port")
        if port is not None and not _port_available(str(resolved.payload.get("host") or "127.0.0.1"), port):
            warnings.append(f"port {port} is already in use; job may fail unless it is intentional.")

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "kind": resolved.kind,
        "resolved_manifest": resolved.manifest,
        "resolved_payload": resolved.payload,
        "runtime_command": command,
    }


def preflight_job_metadata(preflight: Mapping[str, Any]) -> dict[str, Any]:
    warnings = preflight.get("warnings")
    if not isinstance(warnings, list) or not warnings:
        return {}
    return {"preflight_warnings": [str(item) for item in warnings]}


def _manifest_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    manifest = payload.get("manifest")
    if isinstance(manifest, dict):
        return dict(manifest)
    if "runtime" in payload or "eval" in payload or "preannotate" in payload:
        return dict(payload)
    return _legacy_payload_to_manifest(payload)


def _legacy_payload_to_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    runtime_args = {
        "model": payload.get("model_path"),
        "served-model-name": payload.get("served_model_name") or payload.get("model_id"),
        "host": payload.get("host") or "127.0.0.1",
        "port": payload.get("port"),
        "tensor-parallel-size": payload.get("tensor_parallel_size"),
        "max-model-len": payload.get("max_model_len"),
        "gpu-memory-utilization": payload.get("gpu_memory_utilization"),
        "max-num-seqs": payload.get("max_num_seqs"),
    }
    runtime = {
        "mode": "existing_service" if payload.get("endpoint") else "external" if payload.get("backend") == "dry_run" else "existing_service",
        "engine": payload.get("backend") or "vllm_openai",
        "endpoint": payload.get("endpoint"),
        "service_id": payload.get("service_id"),
        "env": {"CUDA_VISIBLE_DEVICES": payload.get("cuda_visible_devices")},
        "args": {key: value for key, value in runtime_args.items() if value not in (None, "")},
    }
    return {
        "kind": "eval_job",
        "runtime": runtime,
        "eval": {
            "run_id": payload.get("run_id"),
            "model_id": payload.get("model_id"),
            "model_path": payload.get("model_path"),
            "benchmark_id": payload.get("benchmark_id"),
            "benchmark_split": payload.get("benchmark_split") or payload.get("split"),
            "task": payload.get("task"),
            "prompt_id": payload.get("prompt_id"),
            "prompt_path": payload.get("prompt_path"),
            "system_prompt": payload.get("system_prompt"),
            "prompt_text": payload.get("prompt_text") or payload.get("user_prompt"),
            "parser": payload.get("parser"),
            "metric_profile": payload.get("metric_profile"),
            "visualization_profile": payload.get("visualization_profile"),
            "target_labels": payload.get("target_labels"),
            "generation": {
                "max_tokens": payload.get("max_tokens"),
                "temperature": payload.get("temperature"),
                "top_p": payload.get("top_p"),
            },
            "data": {
                "min_pixels": payload.get("min_pixels"),
                "max_pixels": payload.get("max_pixels"),
                "batch_size": payload.get("batch_size"),
            },
        },
    }


def _apply_prompt_template_to_manifest(
    manifest: dict[str, Any],
    *,
    payload: dict[str, Any],
    prompt_templates: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, Any]:
    cloned = copy.deepcopy(manifest)
    kind = str(cloned.get("kind") or payload.get("kind") or "eval_job")
    if kind == "eval":
        kind = "eval_job"
    section_key = "preannotate" if kind == "preannotate_job" else "eval"
    section = cloned.get(section_key)
    if not isinstance(section, dict):
        return cloned
    prompt_id = _first_string(section.get("prompt_id"), payload.get("prompt_id"))
    template = _prompt_template_for(prompt_id, prompt_templates)
    if not template:
        return cloned

    section["prompt_id"] = _first_string(section.get("prompt_id"), template.get("prompt_id")) or prompt_id
    _set_default(section, "task", template.get("task"))
    _set_default(section, "system_prompt", template.get("system_prompt"))
    _set_default(section, "prompt_text", template.get("prompt_text"), template.get("user_prompt"))
    _set_default(section, "parser", template.get("parser"))
    _set_default(section, "metric_profile", template.get("metric_profile"))
    _set_default(section, "visualization_profile", template.get("visualization_profile"))
    _set_target_labels_default(section, _target_labels_from_template(template))
    section["generation"] = _merge_defaults(
        _normalized_mapping(template.get("generation")),
        _normalized_mapping(section.get("generation")),
    )
    section["data"] = _merge_defaults(
        _normalized_mapping(template.get("data")),
        _normalized_mapping(section.get("data")),
    )
    section["prompt_template"] = {
        "prompt_id": str(template.get("prompt_id") or prompt_id),
        "label": str(template.get("label") or template.get("prompt_id") or prompt_id),
        "task": str(template.get("task") or section.get("task") or ""),
    }
    return cloned


def _resolve_eval_payload(original: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    runtime = dict(manifest.get("runtime") or {})
    eval_config = dict(manifest.get("eval") or {})
    generation = dict(eval_config.get("generation") or {})
    data = dict(eval_config.get("data") or {})
    runtime_args = _normalized_mapping(runtime.get("args"))
    env = _normalized_mapping(runtime.get("env"))
    model_path = _first_string(
        eval_config.get("model_path"),
        runtime_args.get("model"),
        runtime_args.get("model_path"),
        original.get("model_path"),
    )
    default_model_id = _default_model_id_from_path(model_path)
    served_model_name = _first_string(
        runtime_args.get("served_model_name"),
        runtime_args.get("served-model-name"),
        runtime.get("served_model_name"),
        runtime.get("served-model-name"),
        eval_config.get("model_id"),
        original.get("served_model_name"),
        original.get("model_id"),
        default_model_id,
    )
    model_id = _first_string(
        eval_config.get("model_id"),
        original.get("model_id"),
        served_model_name,
        default_model_id,
    )
    runtime_mode = str(runtime.get("mode") or "ephemeral")
    backend = str(runtime.get("engine") or eval_config.get("backend") or original.get("backend") or "vllm_openai")
    host = _first_string(runtime_args.get("host"), runtime.get("host"), original.get("host"), "127.0.0.1")
    cuda_visible_devices = _first_string(
        env.get("CUDA_VISIBLE_DEVICES"),
        env.get("cuda_visible_devices"),
        runtime_args.get("cuda_visible_devices"),
        original.get("cuda_visible_devices"),
    )
    tensor_parallel_size = _first_value(
        runtime_args.get("tensor_parallel_size"),
        runtime_args.get("tensor-parallel-size"),
        original.get("tensor_parallel_size"),
    )
    if tensor_parallel_size in (None, ""):
        tensor_parallel_size = len(_cuda_visible_devices(cuda_visible_devices)) or 1
    port = _first_value(runtime_args.get("port"), runtime.get("port"), original.get("port"))
    if port in (None, "") and runtime_mode == "ephemeral" and backend == "vllm_openai":
        port = _first_available_port(host or "127.0.0.1")
    payload = {
        **{key: value for key, value in original.items() if key not in {"manifest", "runtime", "eval", "kind"}},
        "job_manifest": manifest,
        "job_kind": "eval_job",
        "runtime_mode": runtime_mode,
        "backend": backend,
        "run_id": _first_string(eval_config.get("run_id"), manifest.get("run_id"), original.get("run_id")),
        "model_id": model_id,
        "model_path": model_path,
        "benchmark_id": _first_string(eval_config.get("benchmark_id"), original.get("benchmark_id")),
        "benchmark_split": _first_string(
            eval_config.get("benchmark_split"),
            eval_config.get("split"),
            original.get("benchmark_split"),
            original.get("split"),
        ),
        "task": _first_string(eval_config.get("task"), original.get("task")),
        "prompt_id": _first_string(eval_config.get("prompt_id"), original.get("prompt_id")),
        "prompt_path": _first_string(eval_config.get("prompt_path"), original.get("prompt_path")),
        "system_prompt": _first_string(eval_config.get("system_prompt"), original.get("system_prompt")),
        "prompt_text": _first_string(eval_config.get("prompt_text"), eval_config.get("user_prompt"), original.get("prompt_text")),
        "parser": _first_string(eval_config.get("parser"), original.get("parser")),
        "metric_profile": _first_string(eval_config.get("metric_profile"), original.get("metric_profile")),
        "visualization_profile": _first_string(eval_config.get("visualization_profile"), original.get("visualization_profile")),
        "target_labels": _first_label_list(eval_config.get("target_labels"), original.get("target_labels")),
        "target_labels_source": _first_string(eval_config.get("target_labels_source"), original.get("target_labels_source")),
        "endpoint": _first_string(runtime.get("endpoint"), original.get("endpoint")),
        "service_id": _first_string(runtime.get("service_id"), original.get("service_id")),
        "served_model_name": served_model_name,
        "host": host,
        "port": port,
        "cuda_visible_devices": cuda_visible_devices,
        "tensor_parallel_size": tensor_parallel_size,
        "max_model_len": _first_value(runtime_args.get("max_model_len"), runtime_args.get("max-model-len"), original.get("max_model_len")),
        "gpu_memory_utilization": _first_value(runtime_args.get("gpu_memory_utilization"), runtime_args.get("gpu-memory-utilization"), original.get("gpu_memory_utilization")),
        "max_num_seqs": _first_value(runtime_args.get("max_num_seqs"), runtime_args.get("max-num-seqs"), original.get("max_num_seqs")),
        "extra_args": _extra_args_from_runtime_args(runtime_args, runtime.get("extra_args")),
        "max_tokens": _first_value(generation.get("max_tokens"), generation.get("max-tokens"), original.get("max_tokens"), 4096),
        "temperature": _first_value(generation.get("temperature"), original.get("temperature"), 0),
        "top_p": _first_value(generation.get("top_p"), generation.get("top-p"), original.get("top_p"), 1),
        "min_pixels": _first_value(data.get("min_pixels"), data.get("min-pixels"), original.get("min_pixels")),
        "max_pixels": _first_value(data.get("max_pixels"), data.get("max-pixels"), original.get("max_pixels")),
        "batch_size": _first_value(data.get("batch_size"), data.get("batch-size"), original.get("batch_size"), 1),
    }
    return _apply_target_label_policy(
        {key: value for key, value in payload.items() if value not in (None, "")}
    )


def _resolve_preannotate_payload(original: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    runtime = dict(manifest.get("runtime") or {})
    config = dict(manifest.get("preannotate") or {})
    eval_like = _resolve_eval_payload(
        original,
        {
            "kind": "eval_job",
            "runtime": runtime,
            "eval": {
                "model_id": config.get("model_id"),
                "model_path": config.get("model_path"),
                "benchmark_id": config.get("benchmark_id") or "preannotation_source",
                "task": config.get("task"),
                "prompt_id": config.get("prompt_id"),
                "prompt_path": config.get("prompt_path"),
                "system_prompt": config.get("system_prompt"),
                "prompt_text": config.get("prompt_text"),
                "generation": config.get("generation"),
                "data": config.get("data"),
            },
        },
    )
    return {
        **eval_like,
        "job_kind": "preannotate_job",
        "preannotate": config,
        "source_root": config.get("source_root"),
        "source_manifest": config.get("source_manifest"),
        "output_root": config.get("output_root"),
    }


def _check_eval_payload(
    payload: dict[str, Any],
    *,
    store_root: Path,
    errors: list[str],
    warnings: list[str],
    prompt_templates: Mapping[str, Mapping[str, Any]] | None = None,
) -> None:
    for key in ("model_id", "model_path", "benchmark_id", "task", "prompt_id"):
        if not _first_string(payload.get(key)):
            errors.append(f"eval job requires {key}.")
    model_path = _first_string(payload.get("model_path"))
    if model_path:
        resolved_model_path = _resolve_repo_path(model_path)
        if not resolved_model_path.exists() and str(payload.get("backend") or "") != "dry_run":
            errors.append(f"model path does not exist: {model_path}")
        else:
            _check_tensor_parallel_compatibility(
                payload,
                model_path=resolved_model_path,
                errors=errors,
                warnings=warnings,
            )
    benchmark_id = _first_string(payload.get("benchmark_id"))
    benchmark_payload: dict[str, Any] | None = None
    if benchmark_id:
        benchmark_path = BenchmarkArtifacts(store_root, benchmark_id).manifest_path
        if not benchmark_path.exists():
            errors.append(f"benchmark does not exist: {benchmark_id}")
        else:
            benchmark_payload = _load_benchmark_manifest_for_preflight(
                benchmark_path,
                errors=errors,
            )
    task = _first_string(payload.get("task"))
    if task and task not in {"detection", "keypoint"}:
        errors.append(f"unsupported task: {task}")
    if benchmark_payload is not None:
        _check_eval_split_against_benchmark(
            payload,
            benchmark_id=benchmark_id or "",
            benchmark_payload=benchmark_payload,
            errors=errors,
            warnings=warnings,
        )
        _check_eval_task_against_benchmark(
            task=task,
            benchmark_id=benchmark_id or "",
            benchmark_payload=benchmark_payload,
            errors=errors,
        )
        _check_target_labels_against_benchmark(
            payload,
            benchmark_id=benchmark_id or "",
            benchmark_payload=benchmark_payload,
            errors=errors,
            warnings=warnings,
        )
    prompt_id = _first_string(payload.get("prompt_id"))
    if prompt_id and prompt_templates is not None and prompt_id not in prompt_templates:
        if not _first_string(payload.get("prompt_path"), payload.get("prompt_text"), payload.get("user_prompt")):
            errors.append(f"unknown prompt_id and no inline prompt/path was provided: {prompt_id}")
    if str(payload.get("runtime_mode") or "") == "existing_service" and not _first_string(payload.get("endpoint")):
        warnings.append("existing_service mode has no endpoint; select a service or provide endpoint.")


def _load_benchmark_manifest_for_preflight(
    path: Path,
    *,
    errors: list[str],
) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"could not read benchmark manifest {path}: {exc}")
        return None
    if not isinstance(payload, dict):
        errors.append(f"benchmark manifest must be a JSON object: {path}")
        return None
    return payload


def _check_eval_task_against_benchmark(
    *,
    task: str | None,
    benchmark_id: str,
    benchmark_payload: dict[str, Any],
    errors: list[str],
) -> None:
    if not task:
        return
    benchmark_tasks = [str(item) for item in benchmark_payload.get("tasks") or []]
    if benchmark_tasks and task not in benchmark_tasks:
        errors.append(
            f"job task={task!r} is not available in benchmark {benchmark_id}: {benchmark_tasks}"
        )


def _check_eval_split_against_benchmark(
    payload: dict[str, Any],
    *,
    benchmark_id: str,
    benchmark_payload: dict[str, Any],
    errors: list[str],
    warnings: list[str],
) -> None:
    try:
        split = resolve_benchmark_split_name(
            benchmark_payload,
            split=_first_string(payload.get("benchmark_split"), payload.get("split")),
            task=str(payload.get("task") or ""),
            prompt_id=str(payload.get("prompt_id") or ""),
            target_labels=_label_list(payload.get("target_labels")),
        )
        split_path = resolve_benchmark_split_path(benchmark_payload, split=split)
    except (FileNotFoundError, ValueError) as exc:
        errors.append(f"benchmark split is invalid for {benchmark_id}: {exc}")
        return
    if not split_path.exists():
        errors.append(f"benchmark split manifest does not exist for {benchmark_id}: {split_path}")


def _check_target_labels_against_benchmark(
    payload: dict[str, Any],
    *,
    benchmark_id: str,
    benchmark_payload: dict[str, Any],
    errors: list[str],
    warnings: list[str],
) -> None:
    target_labels = _label_list(payload.get("target_labels"))
    if not target_labels:
        return
    errors.extend(
        target_label_task_errors(task=str(payload.get("task") or ""), labels=target_labels)
    )
    benchmark_labels = _label_list(benchmark_payload.get("labels"))
    benchmark_errors, benchmark_warnings = target_label_benchmark_messages(
        labels=target_labels,
        benchmark_labels=benchmark_labels,
        benchmark_id=benchmark_id,
    )
    errors.extend(benchmark_errors)
    warnings.extend(benchmark_warnings)


def _check_tensor_parallel_compatibility(
    payload: dict[str, Any],
    *,
    model_path: Path,
    errors: list[str],
    warnings: list[str],
) -> None:
    tensor_parallel_size = _optional_int(payload, "tensor_parallel_size")
    if tensor_parallel_size is None:
        return
    if tensor_parallel_size <= 0:
        errors.append("tensor_parallel_size must be > 0.")
        return
    visible_devices = _cuda_visible_devices(payload.get("cuda_visible_devices"))
    if visible_devices and len(visible_devices) < tensor_parallel_size:
        errors.append(
            "tensor_parallel_size exceeds CUDA_VISIBLE_DEVICES count: "
            f"tp={tensor_parallel_size}, devices={','.join(visible_devices)}"
        )
    attention_heads = _model_attention_heads(model_path)
    if attention_heads is None:
        warnings.append(f"could not read model attention heads from {model_path / 'config.json'}.")
        return
    if attention_heads % tensor_parallel_size != 0:
        errors.append(
            "tensor_parallel_size is incompatible with model attention heads: "
            f"heads={attention_heads}, tp={tensor_parallel_size}. "
            f"Use a divisor of {attention_heads}, for example 1, 2, 4, 8, 16, or 32."
        )


def _model_attention_heads(model_path: Path) -> int | None:
    config_path = model_path / "config.json"
    if not config_path.exists():
        return None
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    text_config = payload.get("text_config")
    values = []
    if isinstance(text_config, dict):
        values.append(text_config.get("num_attention_heads"))
    values.append(payload.get("num_attention_heads"))
    for value in values:
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return None


def _cuda_visible_devices(value: Any) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _check_preannotate_payload(payload: dict[str, Any], *, errors: list[str], warnings: list[str]) -> None:
    for key in ("source_root", "source_manifest", "output_root", "task", "prompt_id"):
        if not _first_string(payload.get(key)):
            errors.append(f"preannotate job requires {key}.")
    source_root = _first_string(payload.get("source_root"))
    if source_root and not _resolve_repo_path(source_root).exists():
        errors.append(f"source_root does not exist: {source_root}")
    source_manifest = _first_string(payload.get("source_manifest"))
    if source_manifest and not _resolve_repo_path(source_manifest).exists():
        errors.append(f"source_manifest does not exist: {source_manifest}")
    errors.append("preannotate execution is not wired yet; this job kind cannot be queued.")


def _runtime_config_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_path": payload.get("model_path"),
        "served_model_name": payload.get("served_model_name") or payload.get("model_id"),
        "host": payload.get("host") or "127.0.0.1",
        "port": payload.get("port"),
        "cuda_visible_devices": payload.get("cuda_visible_devices"),
        "tensor_parallel_size": payload.get("tensor_parallel_size"),
        "max_model_len": payload.get("max_model_len"),
        "gpu_memory_utilization": payload.get("gpu_memory_utilization"),
        "max_num_seqs": payload.get("max_num_seqs"),
        "extra_args": payload.get("extra_args"),
    }


def _extra_args_from_runtime_args(runtime_args: dict[str, Any], explicit: Any) -> list[str]:
    known = {
        "model",
        "model_path",
        "served_model_name",
        "served-model-name",
        "host",
        "port",
        "cuda_visible_devices",
        "tensor_parallel_size",
        "tensor-parallel-size",
        "max_model_len",
        "max-model-len",
        "gpu_memory_utilization",
        "gpu-memory-utilization",
        "max_num_seqs",
        "max-num-seqs",
    }
    extra_args = _string_list(explicit)
    for key, value in runtime_args.items():
        if key in known or value in (None, "", False):
            continue
        flag = f"--{key.replace('_', '-')}"
        if value is True:
            extra_args.append(flag)
        else:
            extra_args.extend([flag, _stringify_cli_value(value)])
    return extra_args


def _normalized_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _prompt_template_for(
    prompt_id: str | None,
    prompt_templates: Mapping[str, Mapping[str, Any]] | None,
) -> Mapping[str, Any] | None:
    if not prompt_id or prompt_templates is None:
        return None
    return prompt_templates.get(prompt_id)


def _set_default(target: dict[str, Any], key: str, *values: Any) -> None:
    if target.get(key) not in (None, ""):
        return
    value = _first_value(*values)
    if value not in (None, ""):
        target[key] = value


def _set_target_labels_default(target: dict[str, Any], labels: list[str]) -> None:
    if not labels or _label_list(target.get("target_labels")):
        return
    target["target_labels"] = labels
    target["target_labels_source"] = "prompt_metadata"


def _apply_target_label_policy(payload: dict[str, Any]) -> dict[str, Any]:
    policy = resolve_target_label_policy(
        explicit=payload.get("target_labels"),
        prompt_id=str(payload.get("prompt_id") or ""),
        task=str(payload.get("task") or ""),
    )
    source = str(payload.get("target_labels_source") or "").strip()
    if policy.source == "explicit" and source in TARGET_LABEL_SOURCES:
        policy = TargetLabelPolicy(labels=policy.labels, source=source)
    if policy.labels:
        payload["target_labels"] = policy.labels
    payload["target_labels_source"] = policy.source
    return payload


def _merge_defaults(defaults: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = {key: value for key, value in defaults.items() if value not in (None, "")}
    merged.update({key: value for key, value in overrides.items() if value not in (None, "")})
    return merged


def _target_labels_from_template(template: Mapping[str, Any]) -> list[str]:
    metadata = template.get("metadata")
    if isinstance(metadata, Mapping):
        return _label_list(metadata.get("target_labels"))
    return []


def _first_label_list(*values: Any) -> list[str]:
    for value in values:
        labels = _label_list(value)
        if labels:
            return labels
    return []


def _label_list(value: Any) -> list[str]:
    return normalize_target_labels(value)


def _string_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [item for item in value.split() if item]
    if isinstance(value, list):
        return [str(item) for item in value]
    raise ValueError("extra_args must be a list or a shell-like string.")


def _stringify_cli_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        import json

        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    return REPO_ROOT / path


def _default_model_id_from_path(value: str | None) -> str | None:
    if not value:
        return None
    path = Path(str(value).rstrip("/"))
    name = path.name
    if not name:
        return None
    if name in {"best", "latest"} or name.startswith("checkpoint-"):
        parent = path.parent.name
        if parent:
            return f"{parent}-{name}"
    return name


def _first_available_port(host: str, *, start: int = 8000, limit: int = 100) -> int:
    for port in range(start, start + limit):
        if _port_available(host, port):
            return port
    raise ValueError(f"no available runtime port found in range {start}-{start + limit - 1}.")


def _port_available(host: str, port: int) -> bool:
    check_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((check_host, int(port))) != 0


def _first_string(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
        if value not in (None, "") and not isinstance(value, (dict, list)):
            return str(value)
    return None


def _first_value(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def _optional_int(payload: dict[str, Any], key: str) -> int | None:
    value = payload.get(key)
    if value in (None, ""):
        return None
    return int(value)
