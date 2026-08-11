from __future__ import annotations

import asyncio
import hmac
import os
import threading
from typing import Any

import torch

from .remote_teacher import (
    CONTENT_TYPE,
    OPDTeacherIdentity,
    decode_teacher_score_request,
    encode_teacher_distribution,
    teacher_request_idempotency_key,
)


class OPDTeacherService:
    """Bounded, deterministic service core independent of the HTTP framework."""

    def __init__(
        self,
        *,
        identity: OPDTeacherIdentity,
        provider: Any,
        max_request_bytes: int,
    ) -> None:
        self.identity = identity
        self.provider = provider
        self.max_request_bytes = int(max_request_bytes)
        if self.max_request_bytes <= 0:
            raise ValueError("OPD teacher service max_request_bytes must be > 0.")
        self._score_lock = threading.Lock()

    def score(self, payload: bytes, *, idempotency_key: str) -> bytes:
        expected_key = teacher_request_idempotency_key(payload)
        if not hmac.compare_digest(str(idempotency_key), expected_key):
            raise ValueError("OPD teacher request idempotency key does not match its body.")
        request = decode_teacher_score_request(payload, max_bytes=self.max_request_bytes)
        model = getattr(self.provider, "model", None)
        if model is not None:
            parameter = next(model.parameters(), None)
            if parameter is not None:
                device = parameter.device
                request = type(request)(
                    model_inputs={
                        name: value.to(device=device) if torch.is_tensor(value) else value
                        for name, value in request.model_inputs.items()
                    },
                    causal_position_mask=request.causal_position_mask.to(device=device),
                    request_ids=request.request_ids,
                    objective_plan=request.objective_plan,
                )
        with self._score_lock:
            distribution = self.provider.score(request)
        return encode_teacher_distribution(distribution)


async def read_bounded_request_body(request: Any, *, max_bytes: int) -> bytes:
    """Read an ASGI request incrementally without materializing an oversize body."""
    limit = int(max_bytes)
    if limit <= 0:
        raise ValueError("OPD teacher request-body limit must be > 0.")
    payload = bytearray()
    async for chunk in request.stream():
        payload.extend(chunk)
        if len(payload) > limit:
            raise ValueError("OPD teacher request body is too large.")
    return bytes(payload)


def create_opd_teacher_app(
    service: OPDTeacherService,
    *,
    api_key_env: str | None = None,
):
    """Create the optional FastAPI transport around the tested service core."""

    try:
        from fastapi import FastAPI, Header, HTTPException, Request, Response
    except ImportError as exc:
        raise ImportError(
            'Serving an OPD teacher requires `uv pip install -e ".[serve]"`.'
        ) from exc

    expected_api_key = None
    if api_key_env:
        expected_api_key = os.environ.get(str(api_key_env))
        if not expected_api_key:
            raise ValueError(
                f"OPD teacher service API key environment variable {api_key_env!r} is empty."
            )

    app = FastAPI(title="Shaft OPD Teacher", docs_url=None, redoc_url=None)

    def authorize(authorization: str | None) -> None:
        if expected_api_key is None:
            return
        expected = f"Bearer {expected_api_key}"
        if authorization is None or not hmac.compare_digest(authorization, expected):
            raise HTTPException(status_code=401, detail="invalid bearer token")

    @app.get("/v1/identity")
    async def identity(authorization: str | None = Header(default=None)):
        authorize(authorization)
        return service.identity.to_dict()

    async def score(
        request,
        idempotency_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ):
        authorize(authorization)
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                declared_length = int(content_length)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail="invalid Content-Length") from exc
            if declared_length < 0:
                raise HTTPException(status_code=400, detail="invalid Content-Length")
            if declared_length > service.max_request_bytes:
                raise HTTPException(status_code=413, detail="request body too large")
        try:
            payload = await read_bounded_request_body(
                request,
                max_bytes=service.max_request_bytes,
            )
        except ValueError as exc:
            raise HTTPException(status_code=413, detail="request body too large") from exc
        if not idempotency_key:
            raise HTTPException(status_code=400, detail="missing Idempotency-Key")
        try:
            response = await asyncio.to_thread(
                service.score,
                payload,
                idempotency_key=idempotency_key,
            )
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return Response(content=response, media_type=CONTENT_TYPE)

    # FastAPI resolves endpoint annotations from module globals. Request is an
    # optional, function-local import so importing shaft.opd does not require
    # the serving extra. Bind the concrete type before registering the route;
    # leaving the parameter untyped makes FastAPI treat it as a query field.
    score.__annotations__["request"] = Request
    app.post("/v1/score")(score)

    return app
