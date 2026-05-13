from __future__ import annotations

import argparse
import json
from pathlib import Path

from .artifacts import DEFAULT_STORE_ROOT, RunArtifacts, atomic_write_json, load_prediction
from .benchmark import create_benchmark_from_raw_data
from .comparison import compare_runs
from .database import EvalBenchDatabase
from .dashboard import main as serve_dashboard
from .evaluator import evaluate_run
from .perf import run_perf_smoke
from .prediction_import import import_predictions_for_benchmark
from .schema import (
    BenchmarkRef,
    EvalRunManifest,
    EvalSpec,
    InferenceParams,
    ModelRef,
    PredictionDocument,
    PredictionInstance,
    PromptRef,
    TaskKind,
    utc_now_iso,
)
from .services import EvalBenchServiceManager
from .worker import EvalBenchWorker


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Shaft Eval Bench utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    benchmark = subparsers.add_parser(
        "create-benchmark", help="Copy a raw_data split into the Eval Bench store."
    )
    benchmark.add_argument("--benchmark-id", required=True)
    benchmark.add_argument("--output-root", default=str(DEFAULT_STORE_ROOT))
    benchmark.add_argument("--task", choices=("detection", "keypoint"), action="append", required=True)
    benchmark.add_argument("--source-root", required=True)
    benchmark.add_argument("--source-manifest", required=True)
    benchmark.add_argument("--split", default="val")
    benchmark.add_argument("--layer", action="append", default=[])
    benchmark.add_argument("--overwrite", action="store_true")

    init_run = subparsers.add_parser("init-run", help="Create an immutable run manifest.")
    init_run.add_argument("--run-id", required=True)
    init_run.add_argument("--output-root", default=str(DEFAULT_STORE_ROOT))
    init_run.add_argument("--task", choices=("detection", "keypoint"), required=True)
    init_run.add_argument("--model-id", required=True)
    init_run.add_argument("--model-path", required=True)
    init_run.add_argument("--benchmark-id", required=True)
    init_run.add_argument("--benchmark-root", required=True)
    init_run.add_argument("--benchmark-manifest", default=None)
    init_run.add_argument("--benchmark-task", choices=("detection", "keypoint"), action="append")
    init_run.add_argument("--split", required=True)
    init_run.add_argument("--spec-id", required=True)
    init_run.add_argument("--prompt-id", required=True)
    init_run.add_argument("--prompt-path", default=None)
    init_run.add_argument("--backend", default="vllm_openai")
    init_run.add_argument("--endpoint", default=None)
    init_run.add_argument("--served-model-name", default=None)
    init_run.add_argument("--service-id", default=None)
    init_run.add_argument("--cuda-visible-devices", default=None)
    init_run.add_argument("--tensor-parallel-size", type=int, default=None)
    init_run.add_argument("--port", type=int, default=None)
    init_run.add_argument("--max-model-len", type=int, default=None)
    init_run.add_argument("--gpu-memory-utilization", type=float, default=None)
    init_run.add_argument("--max-num-seqs", type=int, default=None)
    init_run.add_argument("--max-tokens", type=int, default=4096)
    init_run.add_argument("--temperature", type=float, default=0.0)
    init_run.add_argument("--top-p", type=float, default=1.0)
    init_run.add_argument("--min-pixels", type=int, default=None)
    init_run.add_argument("--max-pixels", type=int, default=None)
    init_run.add_argument("--batch-size", type=int, default=1)
    init_run.add_argument("--submitter", default="local")

    validate = subparsers.add_parser("validate-prediction", help="Validate one prediction JSON.")
    validate.add_argument("path")
    validate.add_argument("--task", choices=("detection", "keypoint"), default=None)

    demo = subparsers.add_parser("write-demo-prediction", help="Write a small example prediction.")
    demo.add_argument("--output", required=True)
    demo.add_argument("--task", choices=("detection", "keypoint"), default="keypoint")

    dashboard = subparsers.add_parser(
        "serve-dashboard", help="Serve the Eval Bench dashboard and API."
    )
    dashboard.add_argument("--host", default="127.0.0.1")
    dashboard.add_argument("--port", type=int, default=None)
    dashboard.add_argument("--store-root", default=str(DEFAULT_STORE_ROOT))
    dashboard.add_argument("--frontend-dist", default=None)

    create_job = subparsers.add_parser("create-job", help="Create a persistent Eval Bench job.")
    create_job.add_argument("--output-root", default=str(DEFAULT_STORE_ROOT))
    create_job.add_argument("--kind", default="eval")
    create_job.add_argument("--payload-json", required=True)

    list_jobs = subparsers.add_parser("list-jobs", help="List persistent Eval Bench jobs.")
    list_jobs.add_argument("--output-root", default=str(DEFAULT_STORE_ROOT))
    list_jobs.add_argument("--limit", type=int, default=100)

    register_service = subparsers.add_parser("register-service", help="Register a model service.")
    register_service.add_argument("--output-root", default=str(DEFAULT_STORE_ROOT))
    register_service.add_argument(
        "--kind", choices=("local_vllm", "external_vllm"), default="local_vllm"
    )
    register_service.add_argument("--service-id", default=None)
    register_service.add_argument("--model-path", default=None)
    register_service.add_argument("--served-model-name", default=None)
    register_service.add_argument("--endpoint", default=None)
    register_service.add_argument("--host", default="127.0.0.1")
    register_service.add_argument("--port", type=int, default=None)
    register_service.add_argument("--cuda-visible-devices", default=None)
    register_service.add_argument("--tensor-parallel-size", type=int, default=None)
    register_service.add_argument("--max-model-len", type=int, default=None)
    register_service.add_argument("--gpu-memory-utilization", type=float, default=None)
    register_service.add_argument("--max-num-seqs", type=int, default=None)
    register_service.add_argument("--extra-arg", action="append", default=[])

    list_services = subparsers.add_parser("list-services", help="List model services.")
    list_services.add_argument("--output-root", default=str(DEFAULT_STORE_ROOT))

    service_command = subparsers.add_parser("service-command", help="Print vLLM launch command.")
    service_command.add_argument("--output-root", default=str(DEFAULT_STORE_ROOT))
    service_command.add_argument("--service-id", required=True)

    start_service = subparsers.add_parser("start-service", help="Start a local vLLM service.")
    start_service.add_argument("--output-root", default=str(DEFAULT_STORE_ROOT))
    start_service.add_argument("--service-id", required=True)

    service_health = subparsers.add_parser(
        "service-health",
        help="Probe a registered service endpoint and update runtime health.",
    )
    service_health.add_argument("--output-root", default=str(DEFAULT_STORE_ROOT))
    service_health.add_argument("--service-id", required=True)
    service_health.add_argument("--timeout-s", type=float, default=2.0)

    service_logs = subparsers.add_parser("service-logs", help="Print a registered service log tail.")
    service_logs.add_argument("--output-root", default=str(DEFAULT_STORE_ROOT))
    service_logs.add_argument("--service-id", required=True)
    service_logs.add_argument("--max-lines", type=int, default=200)

    stop_service = subparsers.add_parser("stop-service", help="Stop a local vLLM service.")
    stop_service.add_argument("--output-root", default=str(DEFAULT_STORE_ROOT))
    stop_service.add_argument("--service-id", required=True)

    process_next = subparsers.add_parser(
        "process-next-job", help="Process the next queued Eval Bench job."
    )
    process_next.add_argument("--output-root", default=str(DEFAULT_STORE_ROOT))
    process_next.add_argument("--kind", default="eval")

    evaluate = subparsers.add_parser("evaluate-run", help="Evaluate prediction snapshots for a run.")
    evaluate.add_argument("--output-root", default=str(DEFAULT_STORE_ROOT))
    evaluate.add_argument("--run-id", required=True)
    evaluate.add_argument("--iou-threshold", type=float, default=0.5)

    import_predictions = subparsers.add_parser(
        "import-predictions",
        help="Import external prediction JSON files as a run and optionally evaluate them.",
    )
    import_predictions.add_argument("--output-root", default=str(DEFAULT_STORE_ROOT))
    import_predictions.add_argument("--run-id", required=True)
    import_predictions.add_argument("--benchmark-id", required=True)
    import_predictions.add_argument("--prediction-root", required=True)
    import_predictions.add_argument("--task", choices=("detection", "keypoint"), required=True)
    import_predictions.add_argument("--model-id", required=True)
    import_predictions.add_argument("--model-path", default="imported")
    import_predictions.add_argument("--prompt-id", default="imported")
    import_predictions.add_argument("--spec-id", default=None)
    import_predictions.add_argument(
        "--target-label",
        dest="target_labels",
        action="append",
        default=None,
        help="Limit imported-run evaluation to this label; repeat for multiple labels.",
    )
    import_predictions.add_argument("--strict", action="store_true")
    import_predictions.add_argument("--overwrite", action="store_true")
    import_predictions.add_argument("--skip-evaluate", action="store_true")

    compare = subparsers.add_parser("compare-runs", help="Compare two evaluated runs.")
    compare.add_argument("--output-root", default=str(DEFAULT_STORE_ROOT))
    compare.add_argument("--baseline-run-id", required=True)
    compare.add_argument("--candidate-run-id", required=True)

    perf = subparsers.add_parser("perf-smoke", help="Measure common Eval Bench store paths.")
    perf.add_argument("--output-root", default=str(DEFAULT_STORE_ROOT))
    perf.add_argument("--iterations", type=int, default=5)
    perf.add_argument("--sample-limit", type=int, default=500)

    return parser


def _cmd_create_benchmark(args: argparse.Namespace) -> None:
    manifest = create_benchmark_from_raw_data(
        store_root=args.output_root,
        benchmark_id=str(args.benchmark_id),
        tasks=args.task,
        source_root=args.source_root,
        source_manifest=args.source_manifest,
        split=str(args.split),
        layers=[str(item) for item in args.layer],
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(manifest.to_dict(), ensure_ascii=False))


def _cmd_init_run(args: argparse.Namespace) -> None:
    task: TaskKind = args.task
    manifest = EvalRunManifest(
        run_id=str(args.run_id),
        submitter=str(args.submitter),
        model=ModelRef(model_id=str(args.model_id), path=str(args.model_path)),
        benchmark=BenchmarkRef(
            benchmark_id=str(args.benchmark_id),
            root=str(args.benchmark_root),
            split=str(args.split),
            tasks=args.benchmark_task or [task],
            manifest_path=args.benchmark_manifest,
        ),
        spec=EvalSpec(
            spec_id=str(args.spec_id),
            task=task,
            prompt=PromptRef(prompt_id=str(args.prompt_id), path=args.prompt_path),
            inference=InferenceParams(
                backend=str(args.backend),
                endpoint=args.endpoint,
                served_model_name=args.served_model_name,
                service_id=args.service_id,
                cuda_visible_devices=args.cuda_visible_devices,
                tensor_parallel_size=args.tensor_parallel_size,
                port=args.port,
                max_model_len=args.max_model_len,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_num_seqs=args.max_num_seqs,
                max_tokens=int(args.max_tokens),
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                min_pixels=args.min_pixels,
                max_pixels=args.max_pixels,
                batch_size=int(args.batch_size),
            ),
        ),
        artifact_root=str(Path(args.output_root) / "runs" / str(args.run_id)),
    )
    artifacts = RunArtifacts(args.output_root, manifest.run_id)
    path = artifacts.write_manifest(manifest)
    print(path)


def _cmd_validate_prediction(args: argparse.Namespace) -> None:
    doc = load_prediction(args.path, task=args.task)
    print(json.dumps({"ok": True, "image": doc.image, "instances": len(doc.instances)}))


def _cmd_write_demo_prediction(args: argparse.Namespace) -> None:
    task: TaskKind = args.task
    instances = [
        PredictionInstance(
            label="arrow" if task == "keypoint" else "icon",
            bbox=[100, 120, 420, 180],
            keypoints=[[110, 150], [420, 150]] if task == "keypoint" else None,
        )
    ]
    document = PredictionDocument(
        image="part1/images/example.png",
        instances=instances,
        metadata={
            "producer": "eval_bench",
            "run_id": "demo",
            "model_id": "demo-model",
            "task": task,
            "created_at": utc_now_iso(),
            "latency_ms": 12.3,
            "inference_params": {"max_tokens": 4096},
            "parser": {"codec": "json_any", "valid": True},
        },
    )
    document.validate(task=task)
    path = Path(args.output)
    atomic_write_json(path, document.to_dict(task=task))
    print(path)


def _cmd_serve_dashboard(args: argparse.Namespace) -> None:
    serve_dashboard(
        host=str(args.host),
        port=args.port,
        store_root=args.store_root,
        frontend_dist=args.frontend_dist,
    )


def _cmd_create_job(args: argparse.Namespace) -> None:
    payload = json.loads(str(args.payload_json))
    if not isinstance(payload, dict):
        raise ValueError("--payload-json must be a JSON object.")
    database = EvalBenchDatabase(args.output_root)
    job = database.create_job(kind=str(args.kind), payload=payload)
    print(json.dumps(job.to_dict(), ensure_ascii=False))


def _cmd_list_jobs(args: argparse.Namespace) -> None:
    database = EvalBenchDatabase(args.output_root)
    jobs = [job.to_dict() for job in database.list_jobs(limit=int(args.limit))]
    print(json.dumps({"jobs": jobs}, ensure_ascii=False))


def _cmd_register_service(args: argparse.Namespace) -> None:
    manager = EvalBenchServiceManager(args.output_root)
    record = manager.register_service(
        {
            "kind": args.kind,
            "service_id": args.service_id,
            "model_path": args.model_path,
            "served_model_name": args.served_model_name,
            "endpoint": args.endpoint,
            "host": args.host,
            "port": args.port,
            "cuda_visible_devices": args.cuda_visible_devices,
            "tensor_parallel_size": args.tensor_parallel_size,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_num_seqs": args.max_num_seqs,
            "extra_args": args.extra_arg,
        }
    )
    print(json.dumps(record.to_dict(), ensure_ascii=False))


def _cmd_list_services(args: argparse.Namespace) -> None:
    manager = EvalBenchServiceManager(args.output_root)
    records = [record.to_dict() for record in manager.list_services()]
    print(json.dumps({"services": records}, ensure_ascii=False))


def _cmd_service_command(args: argparse.Namespace) -> None:
    manager = EvalBenchServiceManager(args.output_root)
    print(json.dumps({"command": manager.launch_command(str(args.service_id))}, ensure_ascii=False))


def _cmd_start_service(args: argparse.Namespace) -> None:
    manager = EvalBenchServiceManager(args.output_root)
    print(json.dumps(manager.start_service(str(args.service_id)).to_dict(), ensure_ascii=False))


def _cmd_service_health(args: argparse.Namespace) -> None:
    manager = EvalBenchServiceManager(args.output_root)
    record = manager.check_service_health(str(args.service_id), timeout_s=float(args.timeout_s))
    print(json.dumps(record.to_dict(), ensure_ascii=False))


def _cmd_service_logs(args: argparse.Namespace) -> None:
    manager = EvalBenchServiceManager(args.output_root)
    payload = manager.service_log(str(args.service_id), max_lines=int(args.max_lines))
    print(json.dumps(payload, ensure_ascii=False))


def _cmd_stop_service(args: argparse.Namespace) -> None:
    manager = EvalBenchServiceManager(args.output_root)
    print(json.dumps(manager.stop_service(str(args.service_id)).to_dict(), ensure_ascii=False))


def _cmd_process_next_job(args: argparse.Namespace) -> None:
    worker = EvalBenchWorker(args.output_root)
    job = worker.process_next(kind=str(args.kind))
    print(json.dumps({"job": job.to_dict() if job else None}, ensure_ascii=False))


def _cmd_evaluate_run(args: argparse.Namespace) -> None:
    path = evaluate_run(
        store_root=args.output_root,
        run_id=str(args.run_id),
        iou_threshold=float(args.iou_threshold),
    )
    print(path)


def _cmd_import_predictions(args: argparse.Namespace) -> None:
    result = import_predictions_for_benchmark(
        store_root=args.output_root,
        run_id=str(args.run_id),
        benchmark_id=str(args.benchmark_id),
        prediction_root=args.prediction_root,
        task=args.task,
        model_id=str(args.model_id),
        model_path=str(args.model_path),
        prompt_id=str(args.prompt_id),
        spec_id=args.spec_id,
        target_labels=args.target_labels,
        strict=bool(args.strict),
        overwrite=bool(args.overwrite),
        evaluate=not bool(args.skip_evaluate),
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False))


def _cmd_compare_runs(args: argparse.Namespace) -> None:
    path = compare_runs(
        store_root=args.output_root,
        baseline_run_id=str(args.baseline_run_id),
        candidate_run_id=str(args.candidate_run_id),
    )
    print(path)


def _cmd_perf_smoke(args: argparse.Namespace) -> None:
    report = run_perf_smoke(
        store_root=args.output_root,
        iterations=int(args.iterations),
        sample_limit=int(args.sample_limit),
    )
    print(json.dumps(report, ensure_ascii=False))


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "create-benchmark":
        _cmd_create_benchmark(args)
    elif args.command == "init-run":
        _cmd_init_run(args)
    elif args.command == "validate-prediction":
        _cmd_validate_prediction(args)
    elif args.command == "write-demo-prediction":
        _cmd_write_demo_prediction(args)
    elif args.command == "serve-dashboard":
        _cmd_serve_dashboard(args)
    elif args.command == "create-job":
        _cmd_create_job(args)
    elif args.command == "list-jobs":
        _cmd_list_jobs(args)
    elif args.command == "register-service":
        _cmd_register_service(args)
    elif args.command == "list-services":
        _cmd_list_services(args)
    elif args.command == "service-command":
        _cmd_service_command(args)
    elif args.command == "start-service":
        _cmd_start_service(args)
    elif args.command == "service-health":
        _cmd_service_health(args)
    elif args.command == "service-logs":
        _cmd_service_logs(args)
    elif args.command == "stop-service":
        _cmd_stop_service(args)
    elif args.command == "process-next-job":
        _cmd_process_next_job(args)
    elif args.command == "evaluate-run":
        _cmd_evaluate_run(args)
    elif args.command == "import-predictions":
        _cmd_import_predictions(args)
    elif args.command == "compare-runs":
        _cmd_compare_runs(args)
    elif args.command == "perf-smoke":
        _cmd_perf_smoke(args)
    else:  # pragma: no cover
        raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    main()
