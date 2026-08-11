from __future__ import annotations

import argparse

from accelerate import Accelerator

from shaft.config import load_config
from shaft.model import build_model_tokenizer_processor
from shaft.opd.execution import resolve_opd_execution_plan
from shaft.opd.remote_teacher import OPDTeacherIdentity, PROTOCOL_VERSION
from shaft.opd.teacher_assembly import (
    LocalHFOPDTeacherArtifactPlan,
    model_artifact_input_identity,
)
from shaft.opd.teacher_service import OPDTeacherService, create_opd_teacher_app
from shaft.pipeline.training_args import resolve_training_compute_dtype


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve one immutable Shaft OPD teacher.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key-env")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    execution_plan = resolve_opd_execution_plan(config.opd)
    artifact_plan = execution_plan.teacher_provider_type.resolve_artifact_plan(
        config,
        checkpointing_requested=True,
    )
    if not isinstance(artifact_plan, LocalHFOPDTeacherArtifactPlan):
        raise ValueError("The teacher service config must select provider='hf_local'.")
    artifact_plan.validate(save_strategy="steps", resume_requested=False)
    accelerator = Accelerator(cpu=bool(config.train.use_cpu))
    dtype = resolve_training_compute_dtype(
        type("Args", (), {"bf16": config.train.bf16, "fp16": config.train.fp16})(),
        model_torch_dtype=config.opd.teacher.torch_dtype,
    )
    artifact_plan.build_sequence_contract(
        training_args=type("Args", (), {"torch_compile": False})(),
        device_type="cpu" if config.train.use_cpu else "cuda",
        distributed_strategy="ddp",
        torch_dtype=dtype,
    )

    def builder(runtime_config, plan, contract, **kwargs):
        _ = kwargs
        return build_model_tokenizer_processor(
            runtime_config,
            sequence_execution_contract=contract,
            resolved_model_plan=plan,
        )

    artifacts = artifact_plan.build_artifacts(builder)
    assert artifacts is not None
    provider = execution_plan.teacher_provider_type(
        config.opd.teacher,
        model=artifacts.model,
    )
    provider.prepare(accelerator)
    model_type, tokenizer_fingerprint, processor_fingerprint = model_artifact_input_identity(
        artifacts
    )
    vocab_size = getattr(getattr(artifacts.model, "config", None), "vocab_size", None)
    if vocab_size is None:
        vocab_size = getattr(
            getattr(getattr(artifacts.model, "config", None), "text_config", None),
            "vocab_size",
            None,
        )
    if vocab_size is None:
        raise ValueError("OPD teacher model config does not publish vocab_size.")
    identity = OPDTeacherIdentity(
        protocol_version=PROTOCOL_VERSION,
        artifact_fingerprint=artifact_plan.fingerprint,
        model_type=model_type,
        tokenizer_fingerprint=tokenizer_fingerprint,
        processor_fingerprint=processor_fingerprint,
        vocab_size=int(vocab_size),
    )
    service = OPDTeacherService(
        identity=identity,
        provider=provider,
        max_request_bytes=config.opd.teacher.remote.max_request_bytes,
    )
    app = create_opd_teacher_app(service, api_key_env=args.api_key_env)
    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError(
            'Serving an OPD teacher requires `uv pip install -e ".[serve]"`.'
        ) from exc
    uvicorn.run(app, host=args.host, port=int(args.port), workers=1)
