from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import asdict
import json
from pathlib import Path

from .artifacts import DEFAULT_STORE_ROOT


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
    init_run.add_argument(
        "--target-label",
        dest="target_labels",
        action="append",
        default=None,
        help="Limit detection/keypoint evaluation to this label; repeat for multiple labels.",
    )
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

    dashboard_state = subparsers.add_parser(
        "dashboard-state",
        help="Print the same coarse dashboard state used by /api/state.",
    )
    dashboard_state.add_argument("--output-root", default=str(DEFAULT_STORE_ROOT))

    scheduler_status = subparsers.add_parser(
        "scheduler-status",
        help="Print an agent-safe scheduler/resource snapshot from the job registry.",
    )
    scheduler_status.add_argument("--output-root", default=str(DEFAULT_STORE_ROOT))

    backend_logs = subparsers.add_parser("backend-logs", help="Print backend.log tail.")
    backend_logs.add_argument("--output-root", default=str(DEFAULT_STORE_ROOT))
    backend_logs.add_argument("--max-lines", type=int, default=200)

    preflight_job = subparsers.add_parser(
        "preflight-job",
        help="Resolve and validate a manifest-first Eval Bench job without enqueueing it.",
    )
    preflight_job.add_argument("--output-root", default=str(DEFAULT_STORE_ROOT))
    preflight_job.add_argument("--kind", default=None)
    preflight_source = preflight_job.add_mutually_exclusive_group(required=True)
    preflight_source.add_argument("--payload-json", default=None)
    preflight_source.add_argument("--payload-file", default=None)

    list_job_templates = subparsers.add_parser(
        "list-job-templates",
        help="List manifest-first job templates for humans and agents.",
    )
    list_job_templates.add_argument("--query", default=None)

    show_job_template = subparsers.add_parser(
        "show-job-template",
        help="Print one manifest-first job template for humans and agents.",
    )
    show_job_template.add_argument("--template-id", required=True)

    list_prompt_templates = subparsers.add_parser(
        "list-prompt-templates",
        help="List prompt templates from the registry for humans and agents.",
    )
    list_prompt_templates.add_argument("--output-root", default=str(DEFAULT_STORE_ROOT))
    list_prompt_templates.add_argument("--task", choices=("detection", "keypoint"), default=None)
    list_prompt_templates.add_argument("--query", default=None)
    list_prompt_templates.add_argument("--offset", type=int, default=0)
    list_prompt_templates.add_argument("--limit", type=int, default=100)

    show_prompt_template = subparsers.add_parser(
        "show-prompt-template",
        help="Print one prompt template from the registry for humans and agents.",
    )
    show_prompt_template.add_argument("--output-root", default=str(DEFAULT_STORE_ROOT))
    show_prompt_template.add_argument("--prompt-id", required=True)

    resolve_target_labels = subparsers.add_parser(
        "resolve-target-labels",
        help="Resolve target-label scope and candidate labels for a benchmark/prompt/task.",
    )
    resolve_target_labels.add_argument("--output-root", default=str(DEFAULT_STORE_ROOT))
    resolve_target_labels.add_argument("--benchmark-id", default=None)
    resolve_target_labels.add_argument("--task", choices=("detection", "keypoint"), default=None)
    resolve_target_labels.add_argument("--prompt-id", default=None)
    resolve_target_labels.add_argument(
        "--target-label",
        dest="target_labels",
        action="append",
        default=None,
        help="Explicit target label; repeat for multiple labels.",
    )

    upsert_prompt_template = subparsers.add_parser(
        "upsert-prompt-template",
        help="Create or update a prompt template from JSON.",
    )
    upsert_prompt_template.add_argument("--output-root", default=str(DEFAULT_STORE_ROOT))
    prompt_template_source = upsert_prompt_template.add_mutually_exclusive_group(required=True)
    prompt_template_source.add_argument("--payload-json", default=None)
    prompt_template_source.add_argument("--payload-file", default=None)

    delete_prompt_template = subparsers.add_parser(
        "delete-prompt-template",
        help="Delete a prompt template from the registry.",
    )
    delete_prompt_template.add_argument("--output-root", default=str(DEFAULT_STORE_ROOT))
    delete_prompt_template.add_argument("--prompt-id", required=True)

    create_job = subparsers.add_parser(
        "create-job",
        help="Preflight and enqueue a persistent Eval Bench job.",
    )
    create_job.add_argument("--output-root", default=str(DEFAULT_STORE_ROOT))
    create_job.add_argument("--kind", default=None)
    create_source = create_job.add_mutually_exclusive_group(required=True)
    create_source.add_argument("--payload-json", default=None)
    create_source.add_argument("--payload-file", default=None)

    list_jobs = subparsers.add_parser("list-jobs", help="List persistent Eval Bench jobs.")
    list_jobs.add_argument("--output-root", default=str(DEFAULT_STORE_ROOT))
    list_jobs.add_argument("--offset", type=int, default=0)
    list_jobs.add_argument("--limit", type=int, default=100)
    list_jobs.add_argument("--kind", default=None)
    list_jobs.add_argument("--status", default=None)
    list_jobs.add_argument("--query", default=None)

    show_job = subparsers.add_parser("show-job", help="Print one persistent Eval Bench job.")
    show_job.add_argument("--output-root", default=str(DEFAULT_STORE_ROOT))
    show_job.add_argument("--job-id", required=True)

    cancel_job = subparsers.add_parser(
        "cancel-job",
        help="Request cancellation for a queued/running job.",
    )
    cancel_job.add_argument("--output-root", default=str(DEFAULT_STORE_ROOT))
    cancel_job.add_argument("--job-id", required=True)

    delete_job = subparsers.add_parser("delete-job", help="Delete a terminal/demo job record.")
    delete_job.add_argument("--output-root", default=str(DEFAULT_STORE_ROOT))
    delete_job.add_argument("--job-id", required=True)

    job_logs = subparsers.add_parser("job-logs", help="Print a job runtime log tail.")
    job_logs.add_argument("--output-root", default=str(DEFAULT_STORE_ROOT))
    job_logs.add_argument("--job-id", required=True)
    job_logs.add_argument("--max-lines", type=int, default=200)

    list_benchmarks = subparsers.add_parser(
        "list-benchmarks", help="List benchmark manifests for humans and agents."
    )
    list_benchmarks.add_argument("--output-root", default=str(DEFAULT_STORE_ROOT))
    list_benchmarks.add_argument("--offset", type=int, default=0)
    list_benchmarks.add_argument("--limit", type=int, default=100)
    list_benchmarks.add_argument("--task", choices=("detection", "keypoint"), default=None)
    list_benchmarks.add_argument("--layer", default=None)
    list_benchmarks.add_argument("--split", default=None)
    list_benchmarks.add_argument("--query", default=None)

    show_benchmark = subparsers.add_parser(
        "show-benchmark", help="Print one benchmark summary for agents."
    )
    show_benchmark.add_argument("--output-root", default=str(DEFAULT_STORE_ROOT))
    show_benchmark.add_argument("--benchmark-id", required=True)

    list_runs = subparsers.add_parser(
        "list-runs", help="List run manifests with agent-safe filters."
    )
    list_runs.add_argument("--output-root", default=str(DEFAULT_STORE_ROOT))
    list_runs.add_argument("--offset", type=int, default=0)
    list_runs.add_argument("--limit", type=int, default=100)
    list_runs.add_argument("--task", choices=("detection", "keypoint"), default=None)
    list_runs.add_argument("--benchmark-id", default=None)
    list_runs.add_argument("--status", default=None)
    list_runs.add_argument("--label", default=None)
    list_runs.add_argument("--model-id", default=None)
    list_runs.add_argument("--prompt-id", default=None)
    list_runs.add_argument("--metric-profile", default=None)
    list_runs.add_argument("--query", default=None)

    show_run = subparsers.add_parser("show-run", help="Print one run summary for agents.")
    show_run.add_argument("--output-root", default=str(DEFAULT_STORE_ROOT))
    show_run.add_argument("--run-id", required=True)

    show_run_report = subparsers.add_parser(
        "show-run-report",
        help="Print a run metric report without reading store files directly.",
    )
    show_run_report.add_argument("--output-root", default=str(DEFAULT_STORE_ROOT))
    show_run_report.add_argument("--run-id", required=True)
    show_run_report.add_argument(
        "--summary",
        action="store_true",
        help="Print reports/summary.json instead of reports/metrics.json.",
    )

    list_run_samples = subparsers.add_parser(
        "list-run-samples",
        help="List run samples with target-label scoping and diagnostics.",
    )
    list_run_samples.add_argument("--output-root", default=str(DEFAULT_STORE_ROOT))
    list_run_samples.add_argument("--run-id", required=True)
    list_run_samples.add_argument("--offset", type=int, default=0)
    list_run_samples.add_argument("--limit", type=int, default=80)
    list_run_samples.add_argument("--label", default=None)
    list_run_samples.add_argument(
        "--error-filter",
        choices=("all", "fn", "fp", "missing", "clean"),
        default="all",
    )

    show_run_sample = subparsers.add_parser(
        "show-run-sample",
        help="Print one run sample detail with scoped GT, predictions, and diagnostics.",
    )
    show_run_sample.add_argument("--output-root", default=str(DEFAULT_STORE_ROOT))
    show_run_sample.add_argument("--run-id", required=True)
    show_run_sample.add_argument("--sample-index", type=int, required=True)

    list_benchmark_samples = subparsers.add_parser(
        "list-benchmark-samples",
        help="List benchmark samples through the store API.",
    )
    list_benchmark_samples.add_argument("--output-root", default=str(DEFAULT_STORE_ROOT))
    list_benchmark_samples.add_argument("--benchmark-id", required=True)
    list_benchmark_samples.add_argument("--offset", type=int, default=0)
    list_benchmark_samples.add_argument("--limit", type=int, default=80)
    list_benchmark_samples.add_argument("--label", default=None)

    show_benchmark_sample = subparsers.add_parser(
        "show-benchmark-sample",
        help="Print one benchmark sample detail through the store API.",
    )
    show_benchmark_sample.add_argument("--output-root", default=str(DEFAULT_STORE_ROOT))
    show_benchmark_sample.add_argument("--benchmark-id", required=True)
    show_benchmark_sample.add_argument("--sample-index", type=int, required=True)

    rank_board = subparsers.add_parser("rank-board", help="Print the run ranking board.")
    rank_board.add_argument("--output-root", default=str(DEFAULT_STORE_ROOT))
    rank_board.add_argument("--offset", type=int, default=0)
    rank_board.add_argument("--limit", type=int, default=100)
    rank_board.add_argument("--task", choices=("detection", "keypoint"), default=None)
    rank_board.add_argument("--benchmark-id", default=None)
    rank_board.add_argument("--status", default=None)
    rank_board.add_argument("--label", default=None)
    rank_board.add_argument("--model-id", default=None)
    rank_board.add_argument("--prompt-id", default=None)
    rank_board.add_argument("--metric-profile", default=None)
    rank_board.add_argument("--min-score", type=float, default=None)
    rank_board.add_argument(
        "--sort-by",
        choices=(
            "f1_iou50",
            "precision_iou50",
            "recall_iou50",
            "mean_iou",
            "prediction_count",
            "created_at",
            "run_id",
            "weighted_score",
        ),
        default="f1_iou50",
    )
    rank_board.add_argument("--sort-order", choices=("asc", "desc"), default="desc")
    rank_board.add_argument("--query", default=None)
    rank_scheme_source = rank_board.add_mutually_exclusive_group()
    rank_scheme_source.add_argument(
        "--rank-scheme-json",
        default=None,
        help="Explicit weighted ranking scheme JSON object.",
    )
    rank_scheme_source.add_argument(
        "--rank-scheme-file",
        default=None,
        help="Path to an explicit weighted ranking scheme JSON object.",
    )

    get_run_note = subparsers.add_parser("get-run-note", help="Print the editable note for a run.")
    get_run_note.add_argument("--output-root", default=str(DEFAULT_STORE_ROOT))
    get_run_note.add_argument("--run-id", required=True)

    set_run_note = subparsers.add_parser("set-run-note", help="Update the editable note for a run.")
    set_run_note.add_argument("--output-root", default=str(DEFAULT_STORE_ROOT))
    set_run_note.add_argument("--run-id", required=True)
    note_source = set_run_note.add_mutually_exclusive_group(required=True)
    note_source.add_argument("--note", default=None)
    note_source.add_argument("--note-file", default=None)

    archive_run = subparsers.add_parser("archive-run", help="Mark a run as archived.")
    archive_run.add_argument("--output-root", default=str(DEFAULT_STORE_ROOT))
    archive_run.add_argument("--run-id", required=True)

    delete_run = subparsers.add_parser("delete-run", help="Move a run artifact directory to trash.")
    delete_run.add_argument("--output-root", default=str(DEFAULT_STORE_ROOT))
    delete_run.add_argument("--run-id", required=True)

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
    list_services.add_argument("--offset", type=int, default=0)
    list_services.add_argument("--limit", type=int, default=100)
    list_services.add_argument("--kind", choices=("local_vllm", "external_vllm"), default=None)
    list_services.add_argument("--status", default=None)
    list_services.add_argument("--query", default=None)

    show_service = subparsers.add_parser("show-service", help="Print one model service.")
    show_service.add_argument("--output-root", default=str(DEFAULT_STORE_ROOT))
    show_service.add_argument("--service-id", required=True)

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

    delete_service = subparsers.add_parser(
        "delete-service",
        help="Stop and delete a registered service.",
    )
    delete_service.add_argument("--output-root", default=str(DEFAULT_STORE_ROOT))
    delete_service.add_argument("--service-id", required=True)

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

    show_comparison = subparsers.add_parser(
        "show-comparison",
        help="Print one saved comparison report without reading artifacts directly.",
    )
    show_comparison.add_argument("--output-root", default=str(DEFAULT_STORE_ROOT))
    show_comparison.add_argument("--comparison-id", default=None)
    show_comparison.add_argument("--baseline-run-id", default=None)
    show_comparison.add_argument("--candidate-run-id", default=None)

    list_comparisons = subparsers.add_parser(
        "list-comparisons", help="List saved comparison reports with agent-safe filters."
    )
    list_comparisons.add_argument("--output-root", default=str(DEFAULT_STORE_ROOT))
    list_comparisons.add_argument("--offset", type=int, default=0)
    list_comparisons.add_argument("--limit", type=int, default=100)
    list_comparisons.add_argument("--task", choices=("detection", "keypoint"), default=None)
    list_comparisons.add_argument("--baseline-run-id", default=None)
    list_comparisons.add_argument("--candidate-run-id", default=None)
    list_comparisons.add_argument("--label", default=None)
    list_comparisons.add_argument("--query", default=None)

    show_comparison_sample = subparsers.add_parser(
        "show-comparison-sample",
        help="Print one paired comparison sample through the store API.",
    )
    show_comparison_sample.add_argument("--output-root", default=str(DEFAULT_STORE_ROOT))
    show_comparison_sample.add_argument("--baseline-run-id", required=True)
    show_comparison_sample.add_argument("--candidate-run-id", required=True)
    show_comparison_sample.add_argument("--sample-index", type=int, required=True)

    perf = subparsers.add_parser("perf-smoke", help="Measure common Eval Bench store paths.")
    perf.add_argument("--output-root", default=str(DEFAULT_STORE_ROOT))
    perf.add_argument("--iterations", type=int, default=5)
    perf.add_argument("--sample-limit", type=int, default=500)

    return parser


def _cmd_create_benchmark(args: argparse.Namespace) -> None:
    from .benchmark import create_benchmark_from_raw_data

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
    from .artifacts import RunArtifacts
    from .label_policy import resolve_target_label_policy
    from .schema import (
        BenchmarkRef,
        EvalRunManifest,
        EvalSpec,
        InferenceParams,
        ModelRef,
        PromptRef,
    )

    task = args.task
    target_policy = resolve_target_label_policy(
        explicit=args.target_labels,
        prompt_id=str(args.prompt_id),
        task=task,
    )
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
            target_labels=target_policy.labels,
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
            metadata={"target_labels_source": target_policy.source},
        ),
        artifact_root=str(Path(args.output_root) / "runs" / str(args.run_id)),
    )
    artifacts = RunArtifacts(args.output_root, manifest.run_id)
    path = artifacts.write_manifest(manifest)
    print(path)


def _cmd_validate_prediction(args: argparse.Namespace) -> None:
    from .artifacts import load_prediction

    doc = load_prediction(args.path, task=args.task)
    print(json.dumps({"ok": True, "image": doc.image, "instances": len(doc.instances)}))


def _cmd_write_demo_prediction(args: argparse.Namespace) -> None:
    from .artifacts import atomic_write_json
    from .schema import PredictionDocument, PredictionInstance, utc_now_iso

    task = args.task
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
    from .dashboard import main as serve_dashboard

    serve_dashboard(
        host=str(args.host),
        port=args.port,
        store_root=args.store_root,
        frontend_dist=args.frontend_dist,
    )


def _cmd_dashboard_state(args: argparse.Namespace) -> None:
    from .store import EvalBenchStore

    print(json.dumps(EvalBenchStore(args.output_root).state().to_dict(), ensure_ascii=False))


def _cmd_scheduler_status(args: argparse.Namespace) -> None:
    from .orchestrator import EvalBenchOrchestrator

    status = EvalBenchOrchestrator.from_env(args.output_root).status()
    status["source"] = "cli_snapshot"
    print(json.dumps(status, ensure_ascii=False))


def _cmd_backend_logs(args: argparse.Namespace) -> None:
    from .log_utils import tail_text_lines
    from .store import EvalBenchStore

    log_path = EvalBenchStore(args.output_root).layout.logs_dir / "backend.log"
    line_limit = _log_line_limit(args.max_lines)
    lines = tail_text_lines(log_path, max_lines=line_limit)
    print(
        json.dumps(
            {
                "log_path": str(log_path) if log_path.exists() else None,
                "lines": lines,
                "text": "".join(lines),
            },
            ensure_ascii=False,
        )
    )


def _cmd_create_job(args: argparse.Namespace) -> None:
    from .database import EvalBenchDatabase
    from .job_spec import preflight_job_metadata, preflight_job_payload

    database = EvalBenchDatabase(args.output_root)
    preflight = preflight_job_payload(
        _job_payload_from_args(args),
        store_root=args.output_root,
        prompt_templates=_prompt_template_map(database),
    )
    if not preflight.get("ok"):
        raise ValueError(json.dumps(preflight, ensure_ascii=False))
    job = database.create_job(
        kind=_database_job_kind(str(preflight.get("kind") or "eval_job")),
        payload={
            **dict(preflight.get("resolved_payload") or {}),
            "manifest": preflight.get("resolved_manifest"),
        },
        metadata=preflight_job_metadata(preflight),
    )
    print(json.dumps(job.to_dict(), ensure_ascii=False))


def _cmd_preflight_job(args: argparse.Namespace) -> None:
    from .database import EvalBenchDatabase
    from .job_spec import preflight_job_payload

    database = EvalBenchDatabase(args.output_root)
    result = preflight_job_payload(
        _job_payload_from_args(args),
        store_root=args.output_root,
        prompt_templates=_prompt_template_map(database),
    )
    print(json.dumps(result, ensure_ascii=False))


def _cmd_list_job_templates(args: argparse.Namespace) -> None:
    from .job_spec import job_templates

    query = _normalize_cli_filter(args.query).lower()
    templates = job_templates()
    if query:
        templates = {
            template_id: template
            for template_id, template in templates.items()
            if _template_query_matches(template_id, template, query)
        }
    print(
        json.dumps(
            {
                "templates": templates,
                "total": len(templates),
                "filters": {"query": query},
            },
            ensure_ascii=False,
        )
    )


def _cmd_show_job_template(args: argparse.Namespace) -> None:
    from .job_spec import job_templates

    template_id = str(args.template_id)
    templates = job_templates()
    template = templates.get(template_id)
    if template is None:
        raise FileNotFoundError(f"job template does not exist: {template_id}")
    print(json.dumps({"template_id": template_id, "template": template}, ensure_ascii=False))


def _cmd_list_prompt_templates(args: argparse.Namespace) -> None:
    from .database import EvalBenchDatabase

    filters = {
        "task": _normalize_cli_filter(args.task),
        "query": _normalize_cli_filter(args.query),
    }
    records = [
        record.to_dict()
        for record in EvalBenchDatabase(args.output_root).list_prompt_templates(
            task=filters["task"] or None,
            limit=1000,
        )
    ]
    query = filters["query"].lower()
    if query:
        records = [
            record for record in records if _template_query_matches(str(record["prompt_id"]), record, query)
        ]
    payload = _paged_payload(
        "templates",
        records,
        offset=args.offset,
        limit=args.limit,
        filters=filters,
    )
    payload["by_id"] = {record["prompt_id"]: record for record in payload["templates"]}  # type: ignore[index]
    print(json.dumps(payload, ensure_ascii=False))


def _cmd_show_prompt_template(args: argparse.Namespace) -> None:
    from .database import EvalBenchDatabase

    record = EvalBenchDatabase(args.output_root).get_prompt_template(str(args.prompt_id))
    if record is None:
        raise FileNotFoundError(f"prompt template does not exist: {args.prompt_id}")
    print(json.dumps({"template": record.to_dict()}, ensure_ascii=False))


def _cmd_resolve_target_labels(args: argparse.Namespace) -> None:
    from .database import EvalBenchDatabase
    from .label_policy import target_label_resolution_payload
    from .store import EvalBenchStore

    database = EvalBenchDatabase(args.output_root)
    prompt_metadata: dict[str, object] = {}
    warnings: list[str] = []
    task = args.task
    if args.prompt_id:
        record = database.get_prompt_template(str(args.prompt_id))
        if record is None:
            warnings.append(f"prompt template does not exist: {args.prompt_id}")
        else:
            prompt_metadata = dict(record.metadata)
            task = task or record.task
    benchmark_labels: list[str] = []
    if args.benchmark_id:
        benchmark = next(
            (
                item
                for item in EvalBenchStore(args.output_root).benchmarks()
                if item.benchmark_id == str(args.benchmark_id)
            ),
            None,
        )
        if benchmark is None:
            raise FileNotFoundError(f"benchmark does not exist: {args.benchmark_id}")
        benchmark_labels = benchmark.labels
        if task and benchmark.tasks and task not in benchmark.tasks:
            warnings.append(
                f"task={task} is not advertised by benchmark {args.benchmark_id}: {benchmark.tasks}"
            )
    payload = target_label_resolution_payload(
        task=task,
        prompt_id=args.prompt_id,
        explicit=args.target_labels,
        prompt_metadata=prompt_metadata,
        benchmark_id=args.benchmark_id,
        benchmark_labels=benchmark_labels,
    )
    payload["warnings"] = [*payload["warnings"], *warnings]
    print(json.dumps(payload, ensure_ascii=False))


def _cmd_upsert_prompt_template(args: argparse.Namespace) -> None:
    from .database import EvalBenchDatabase

    record = EvalBenchDatabase(args.output_root).upsert_prompt_template(_json_payload_from_args(args))
    print(json.dumps(record.to_dict(), ensure_ascii=False))


def _cmd_delete_prompt_template(args: argparse.Namespace) -> None:
    from .database import EvalBenchDatabase

    record = EvalBenchDatabase(args.output_root).delete_prompt_template(str(args.prompt_id))
    print(json.dumps({"prompt_id": record.prompt_id, "deleted": True}, ensure_ascii=False))


def _cmd_list_jobs(args: argparse.Namespace) -> None:
    from .database import EvalBenchDatabase

    database = EvalBenchDatabase(args.output_root)
    page = database.job_page(
        offset=args.offset,
        limit=args.limit,
        kind=args.kind,
        status=args.status,
        query=args.query,
    )
    print(json.dumps(page.to_dict(), ensure_ascii=False))


def _cmd_show_job(args: argparse.Namespace) -> None:
    from .database import EvalBenchDatabase

    record = EvalBenchDatabase(args.output_root).get_job(str(args.job_id))
    if record is None:
        raise FileNotFoundError(f"job does not exist: {args.job_id}")
    print(json.dumps({"job": record.to_dict()}, ensure_ascii=False))


def _cmd_cancel_job(args: argparse.Namespace) -> None:
    from .database import EvalBenchDatabase

    record = EvalBenchDatabase(args.output_root).cancel_job(str(args.job_id))
    print(json.dumps(record.to_dict(), ensure_ascii=False))


def _cmd_delete_job(args: argparse.Namespace) -> None:
    from .database import EvalBenchDatabase

    record = EvalBenchDatabase(args.output_root).delete_job(str(args.job_id))
    print(json.dumps({"job": record.to_dict(), "deleted": True}, ensure_ascii=False))


def _cmd_job_logs(args: argparse.Namespace) -> None:
    from .database import EvalBenchDatabase
    from .log_utils import job_runtime_log_path, tail_text_lines

    database = EvalBenchDatabase(args.output_root)
    record = database.get_job(str(args.job_id))
    if record is None:
        raise FileNotFoundError(f"job does not exist: {args.job_id}")
    log_path = job_runtime_log_path(args.output_root, record)
    line_limit = _log_line_limit(args.max_lines)
    lines = tail_text_lines(log_path, max_lines=line_limit)
    print(
        json.dumps(
            {
                "job_id": str(args.job_id),
                "log_path": str(log_path) if log_path.exists() else None,
                "lines": lines,
                "text": "".join(lines),
            },
            ensure_ascii=False,
        )
    )


def _cmd_list_benchmarks(args: argparse.Namespace) -> None:
    from .store import EvalBenchStore

    page = EvalBenchStore(args.output_root).benchmark_page(
        offset=args.offset,
        limit=args.limit,
        task=args.task,
        layer=args.layer,
        split=args.split,
        query=args.query,
    )
    print(json.dumps(page.to_dict(), ensure_ascii=False))


def _cmd_show_benchmark(args: argparse.Namespace) -> None:
    from .store import EvalBenchStore

    benchmark = EvalBenchStore(args.output_root).benchmark(str(args.benchmark_id))
    print(json.dumps({"benchmark": asdict(benchmark)}, ensure_ascii=False))


def _cmd_list_runs(args: argparse.Namespace) -> None:
    from .store import EvalBenchStore

    page = EvalBenchStore(args.output_root).run_page(
        offset=args.offset,
        limit=args.limit,
        task=args.task,
        benchmark_id=args.benchmark_id,
        status=args.status,
        label=args.label,
        model_id=args.model_id,
        prompt_id=args.prompt_id,
        metric_profile=args.metric_profile,
        query=args.query,
    )
    print(json.dumps(page.to_dict(), ensure_ascii=False))


def _cmd_show_run(args: argparse.Namespace) -> None:
    from .store import EvalBenchStore

    run = next(
        (item for item in EvalBenchStore(args.output_root).runs() if item.run_id == str(args.run_id)),
        None,
    )
    if run is None:
        raise FileNotFoundError(f"run does not exist: {args.run_id}")
    print(json.dumps({"run": asdict(run)}, ensure_ascii=False))


def _cmd_show_run_report(args: argparse.Namespace) -> None:
    from .artifacts import RunArtifacts

    report_name = "summary.json" if bool(args.summary) else "metrics.json"
    report_path = RunArtifacts(args.output_root, str(args.run_id)).reports_dir / report_name
    if not report_path.exists():
        raise FileNotFoundError(f"run report does not exist: {report_path}")
    print(report_path.read_text(encoding="utf-8"))


def _cmd_list_run_samples(args: argparse.Namespace) -> None:
    from .store import EvalBenchStore

    page = EvalBenchStore(args.output_root).run_sample_page(
        str(args.run_id),
        offset=max(0, int(args.offset)),
        limit=max(1, int(args.limit)),
        label=args.label,
        error_filter=str(args.error_filter),
    )
    print(
        json.dumps(
            {
                "run_id": str(args.run_id),
                "offset": page.offset,
                "limit": page.limit,
                "total": page.total,
                "labels": page.labels,
                "samples": [asdict(sample) for sample in page.samples],
            },
            ensure_ascii=False,
        )
    )


def _cmd_show_run_sample(args: argparse.Namespace) -> None:
    from .store import EvalBenchStore

    detail = EvalBenchStore(args.output_root).run_sample_detail(
        str(args.run_id),
        sample_index=int(args.sample_index),
    )
    print(
        json.dumps(
            {
                "run_id": str(args.run_id),
                "sample": asdict(detail.sample),
                "gt_instances": detail.gt_instances,
                "pred_instances": detail.pred_instances,
                "raw_payload": detail.raw_payload,
                "prediction_payload": detail.prediction_payload,
                "diagnostics": detail.diagnostics,
            },
            ensure_ascii=False,
        )
    )


def _cmd_list_benchmark_samples(args: argparse.Namespace) -> None:
    from .store import EvalBenchStore

    page = EvalBenchStore(args.output_root).benchmark_sample_page(
        str(args.benchmark_id),
        offset=max(0, int(args.offset)),
        limit=max(1, int(args.limit)),
        label=args.label,
    )
    print(
        json.dumps(
            {
                "benchmark_id": str(args.benchmark_id),
                "offset": page.offset,
                "limit": page.limit,
                "total": page.total,
                "labels": page.labels,
                "samples": [asdict(sample) for sample in page.samples],
            },
            ensure_ascii=False,
        )
    )


def _cmd_show_benchmark_sample(args: argparse.Namespace) -> None:
    from .store import EvalBenchStore

    detail = EvalBenchStore(args.output_root).benchmark_sample_detail(
        str(args.benchmark_id),
        sample_index=int(args.sample_index),
    )
    print(
        json.dumps(
            {
                "benchmark_id": str(args.benchmark_id),
                "sample": asdict(detail.sample),
                "gt_instances": detail.gt_instances,
                "raw_payload": detail.raw_payload,
            },
            ensure_ascii=False,
        )
    )


def _cmd_rank_board(args: argparse.Namespace) -> None:
    from .store import EvalBenchStore

    board = EvalBenchStore(args.output_root).rank_board(
        offset=max(0, int(args.offset)),
        limit=max(1, int(args.limit)),
        task=args.task,
        benchmark_id=args.benchmark_id,
        status=args.status,
        label=args.label,
        model_id=args.model_id,
        prompt_id=args.prompt_id,
        metric_profile=args.metric_profile,
        min_score=args.min_score,
        sort_by=args.sort_by,
        sort_order=args.sort_order,
        query=args.query,
        rank_scheme=_rank_scheme_from_args(args),
    )
    print(json.dumps(board.to_dict(), ensure_ascii=False))


def _cmd_get_run_note(args: argparse.Namespace) -> None:
    from .store import EvalBenchStore

    note = EvalBenchStore(args.output_root).run_note(str(args.run_id))
    print(json.dumps(note.to_dict(), ensure_ascii=False))


def _cmd_set_run_note(args: argparse.Namespace) -> None:
    from .store import EvalBenchStore

    note_text = (
        Path(str(args.note_file)).read_text(encoding="utf-8")
        if args.note_file is not None
        else str(args.note)
    )
    note = EvalBenchStore(args.output_root).update_run_note(str(args.run_id), note_text)
    print(json.dumps(note.to_dict(), ensure_ascii=False))


def _cmd_archive_run(args: argparse.Namespace) -> None:
    from .store import EvalBenchStore

    payload = EvalBenchStore(args.output_root).archive_run(str(args.run_id))
    print(json.dumps(payload, ensure_ascii=False))


def _cmd_delete_run(args: argparse.Namespace) -> None:
    from .store import EvalBenchStore

    payload = EvalBenchStore(args.output_root).delete_run(str(args.run_id))
    print(json.dumps(payload, ensure_ascii=False))


def _cmd_register_service(args: argparse.Namespace) -> None:
    from .services import EvalBenchServiceManager

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
    from .services import EvalBenchServiceManager

    manager = EvalBenchServiceManager(args.output_root)
    page = manager.service_page(
        offset=args.offset,
        limit=args.limit,
        kind=args.kind,
        status=args.status,
        query=args.query,
    )
    print(json.dumps(page.to_dict(), ensure_ascii=False))


def _cmd_show_service(args: argparse.Namespace) -> None:
    from .services import EvalBenchServiceManager

    record = EvalBenchServiceManager(args.output_root).service(str(args.service_id))
    print(json.dumps({"service": record.to_dict()}, ensure_ascii=False))


def _cmd_service_command(args: argparse.Namespace) -> None:
    from .services import EvalBenchServiceManager

    manager = EvalBenchServiceManager(args.output_root)
    print(json.dumps({"command": manager.launch_command(str(args.service_id))}, ensure_ascii=False))


def _cmd_start_service(args: argparse.Namespace) -> None:
    from .services import EvalBenchServiceManager

    manager = EvalBenchServiceManager(args.output_root)
    print(json.dumps(manager.start_service(str(args.service_id)).to_dict(), ensure_ascii=False))


def _cmd_service_health(args: argparse.Namespace) -> None:
    from .services import EvalBenchServiceManager

    manager = EvalBenchServiceManager(args.output_root)
    record = manager.check_service_health(str(args.service_id), timeout_s=float(args.timeout_s))
    print(json.dumps(record.to_dict(), ensure_ascii=False))


def _cmd_service_logs(args: argparse.Namespace) -> None:
    from .services import EvalBenchServiceManager

    manager = EvalBenchServiceManager(args.output_root)
    payload = manager.service_log(str(args.service_id), max_lines=int(args.max_lines))
    print(json.dumps(payload, ensure_ascii=False))


def _cmd_stop_service(args: argparse.Namespace) -> None:
    from .services import EvalBenchServiceManager

    manager = EvalBenchServiceManager(args.output_root)
    print(json.dumps(manager.stop_service(str(args.service_id)).to_dict(), ensure_ascii=False))


def _cmd_delete_service(args: argparse.Namespace) -> None:
    from .services import EvalBenchServiceManager

    manager = EvalBenchServiceManager(args.output_root)
    print(json.dumps(manager.delete_service(str(args.service_id)), ensure_ascii=False))


def _cmd_process_next_job(args: argparse.Namespace) -> None:
    from .worker import EvalBenchWorker

    worker = EvalBenchWorker(args.output_root)
    job = worker.process_next(kind=str(args.kind))
    print(json.dumps({"job": job.to_dict() if job else None}, ensure_ascii=False))


def _cmd_evaluate_run(args: argparse.Namespace) -> None:
    from .evaluator import evaluate_run

    path = evaluate_run(
        store_root=args.output_root,
        run_id=str(args.run_id),
        iou_threshold=float(args.iou_threshold),
    )
    print(path)


def _cmd_import_predictions(args: argparse.Namespace) -> None:
    from .prediction_import import import_predictions_for_benchmark

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
    from .comparison import compare_runs

    path = compare_runs(
        store_root=args.output_root,
        baseline_run_id=str(args.baseline_run_id),
        candidate_run_id=str(args.candidate_run_id),
    )
    print(path)


def _cmd_show_comparison(args: argparse.Namespace) -> None:
    from .comparison import load_comparison_report

    payload = load_comparison_report(
        store_root=args.output_root,
        comparison_id=args.comparison_id,
        baseline_run_id=args.baseline_run_id,
        candidate_run_id=args.candidate_run_id,
    )
    print(json.dumps(payload, ensure_ascii=False))


def _cmd_list_comparisons(args: argparse.Namespace) -> None:
    from .comparison import filter_comparison_reports, list_comparison_reports

    filters = {
        "task": _normalize_cli_filter(args.task),
        "baseline_run_id": _normalize_cli_filter(args.baseline_run_id),
        "candidate_run_id": _normalize_cli_filter(args.candidate_run_id),
        "label": _normalize_cli_filter(args.label),
        "query": (args.query or "").strip(),
    }
    items = filter_comparison_reports(
        list_comparison_reports(store_root=args.output_root),
        task=filters["task"],
        baseline_run_id=filters["baseline_run_id"],
        candidate_run_id=filters["candidate_run_id"],
        label=filters["label"],
        query=filters["query"],
    )
    print(
        json.dumps(
            _paged_payload(
                "comparisons",
                items,
                offset=args.offset,
                limit=args.limit,
                filters=filters,
            ),
            ensure_ascii=False,
        )
    )


def _cmd_show_comparison_sample(args: argparse.Namespace) -> None:
    from .comparison import comparison_sample_detail_payload

    payload = comparison_sample_detail_payload(
        store_root=args.output_root,
        baseline_run_id=str(args.baseline_run_id),
        candidate_run_id=str(args.candidate_run_id),
        sample_index=int(args.sample_index),
    )
    print(json.dumps(payload, ensure_ascii=False))


def _cmd_perf_smoke(args: argparse.Namespace) -> None:
    from .perf import run_perf_smoke

    report = run_perf_smoke(
        store_root=args.output_root,
        iterations=int(args.iterations),
        sample_limit=int(args.sample_limit),
    )
    print(json.dumps(report, ensure_ascii=False))


def _normalize_cli_filter(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _log_line_limit(value: object) -> int:
    parsed = int(value)
    return 0 if parsed <= 0 else min(parsed, 2000)


def _paged_payload(
    key: str,
    items: list[dict],
    *,
    offset: int,
    limit: int,
    filters: dict[str, str],
) -> dict[str, object]:
    start = max(0, int(offset))
    page_limit = max(1, int(limit))
    return {
        key: items[start : start + page_limit],
        "total": len(items),
        "offset": start,
        "limit": page_limit,
        "filters": filters,
    }


def _json_payload_from_args(args: argparse.Namespace) -> dict[str, object]:
    source_text = (
        Path(str(args.payload_file)).read_text(encoding="utf-8")
        if getattr(args, "payload_file", None)
        else str(args.payload_json)
    )
    payload = json.loads(source_text)
    if not isinstance(payload, dict):
        raise ValueError("payload must be a JSON object.")
    return payload


def _rank_scheme_from_args(args: argparse.Namespace) -> dict[str, object] | None:
    source_text = None
    if getattr(args, "rank_scheme_file", None):
        source_text = Path(str(args.rank_scheme_file)).read_text(encoding="utf-8")
    elif getattr(args, "rank_scheme_json", None):
        source_text = str(args.rank_scheme_json)
    if source_text is None:
        return None
    payload = json.loads(source_text)
    if not isinstance(payload, dict):
        raise ValueError("rank scheme must be a JSON object.")
    return payload


def _job_payload_from_args(args: argparse.Namespace) -> dict[str, object]:
    payload = _json_payload_from_args(args)
    kind = getattr(args, "kind", None)
    if kind:
        payload.setdefault("kind", str(kind))
    return payload


def _template_query_matches(template_id: str, payload: dict[str, object], query: str) -> bool:
    haystack = [
        template_id,
        str(payload.get("label") or ""),
        str(payload.get("description") or ""),
        str(payload.get("task") or ""),
        str(payload.get("parser") or ""),
        str(payload.get("metric_profile") or ""),
        str(payload.get("visualization_profile") or ""),
        str(payload.get("metadata") or ""),
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
    ]
    return query in " ".join(haystack).lower()


def _prompt_template_map(database) -> dict[str, dict[str, object]]:
    return {
        record.prompt_id: record.to_dict()
        for record in database.list_prompt_templates(limit=1000)
    }


def _database_job_kind(resolved_kind: str) -> str:
    if resolved_kind == "eval_job":
        return "eval"
    if resolved_kind == "preannotate_job":
        return "preannotate"
    raise ValueError(f"unsupported job kind: {resolved_kind}")


def _command_handlers() -> dict[str, Callable[[argparse.Namespace], None]]:
    return {
        "create-benchmark": _cmd_create_benchmark,
        "init-run": _cmd_init_run,
        "validate-prediction": _cmd_validate_prediction,
        "write-demo-prediction": _cmd_write_demo_prediction,
        "serve-dashboard": _cmd_serve_dashboard,
        "dashboard-state": _cmd_dashboard_state,
        "scheduler-status": _cmd_scheduler_status,
        "backend-logs": _cmd_backend_logs,
        "preflight-job": _cmd_preflight_job,
        "list-job-templates": _cmd_list_job_templates,
        "show-job-template": _cmd_show_job_template,
        "list-prompt-templates": _cmd_list_prompt_templates,
        "show-prompt-template": _cmd_show_prompt_template,
        "resolve-target-labels": _cmd_resolve_target_labels,
        "upsert-prompt-template": _cmd_upsert_prompt_template,
        "delete-prompt-template": _cmd_delete_prompt_template,
        "create-job": _cmd_create_job,
        "list-jobs": _cmd_list_jobs,
        "show-job": _cmd_show_job,
        "cancel-job": _cmd_cancel_job,
        "delete-job": _cmd_delete_job,
        "job-logs": _cmd_job_logs,
        "list-benchmarks": _cmd_list_benchmarks,
        "show-benchmark": _cmd_show_benchmark,
        "list-runs": _cmd_list_runs,
        "show-run": _cmd_show_run,
        "show-run-report": _cmd_show_run_report,
        "list-run-samples": _cmd_list_run_samples,
        "show-run-sample": _cmd_show_run_sample,
        "list-benchmark-samples": _cmd_list_benchmark_samples,
        "show-benchmark-sample": _cmd_show_benchmark_sample,
        "rank-board": _cmd_rank_board,
        "get-run-note": _cmd_get_run_note,
        "set-run-note": _cmd_set_run_note,
        "archive-run": _cmd_archive_run,
        "delete-run": _cmd_delete_run,
        "register-service": _cmd_register_service,
        "list-services": _cmd_list_services,
        "show-service": _cmd_show_service,
        "service-command": _cmd_service_command,
        "start-service": _cmd_start_service,
        "service-health": _cmd_service_health,
        "service-logs": _cmd_service_logs,
        "stop-service": _cmd_stop_service,
        "delete-service": _cmd_delete_service,
        "process-next-job": _cmd_process_next_job,
        "evaluate-run": _cmd_evaluate_run,
        "import-predictions": _cmd_import_predictions,
        "compare-runs": _cmd_compare_runs,
        "show-comparison": _cmd_show_comparison,
        "list-comparisons": _cmd_list_comparisons,
        "show-comparison-sample": _cmd_show_comparison_sample,
        "perf-smoke": _cmd_perf_smoke,
    }


def main() -> None:
    args = _build_parser().parse_args()
    handler = _command_handlers().get(args.command)
    if handler is None:  # pragma: no cover
        raise AssertionError(f"unhandled command: {args.command}")
    handler(args)


if __name__ == "__main__":
    main()
