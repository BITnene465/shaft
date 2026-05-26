from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from sqlalchemy import Engine, create_engine, event, text

from .artifacts import DEFAULT_STORE_ROOT, StoreLayout
from .prompt_templates import default_prompt_templates
from .schema import utc_now_iso


def _job_id(kind: str) -> str:
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    return f"{kind}_{timestamp}_{uuid4().hex[:8]}"


def _service_id(kind: str) -> str:
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    return f"{kind}_{timestamp}_{uuid4().hex[:8]}"


@dataclass(frozen=True)
class JobRecord:
    job_id: str
    kind: str
    status: str
    payload: dict[str, Any]
    created_at: str
    updated_at: str
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class JobListPage:
    offset: int
    limit: int
    total: int
    filters: dict[str, str]
    facets: dict[str, list[dict[str, Any]]]
    jobs: list[JobRecord]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ServiceRecord:
    service_id: str
    kind: str
    status: str
    config: dict[str, Any]
    created_at: str
    updated_at: str
    error: str | None = None
    runtime: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ServiceListPage:
    offset: int
    limit: int
    total: int
    filters: dict[str, str]
    facets: dict[str, list[dict[str, Any]]]
    services: list[ServiceRecord]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PromptTemplateRecord:
    prompt_id: str
    label: str
    task: str
    system_prompt: str
    user_prompt: str
    parser: str | None
    metric_profile: str | None
    visualization_profile: str | None
    generation: dict[str, Any]
    data: dict[str, Any]
    metadata: dict[str, Any]
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EvalBenchDatabase:
    def __init__(self, root: str | Path = DEFAULT_STORE_ROOT) -> None:
        self.layout = StoreLayout(root)
        self.layout.ensure()
        self.engine: Engine = create_engine(
            f"sqlite:///{self.layout.db_path}",
            future=True,
            connect_args={"timeout": 30},
        )
        event.listen(self.engine, "connect", _configure_sqlite_connection)
        self.initialize()

    def initialize(self) -> None:
        with self.engine.begin() as connection:
            connection.execute(text("PRAGMA journal_mode=WAL"))
            connection.execute(text("PRAGMA busy_timeout=5000"))
            connection.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS jobs (
                        job_id TEXT PRIMARY KEY,
                        kind TEXT NOT NULL,
                        status TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        error TEXT,
                        metadata_json TEXT NOT NULL
                    )
                    """
                )
            )
            connection.execute(
                text("CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at DESC)")
            )
            connection.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS model_services (
                        service_id TEXT PRIMARY KEY,
                        kind TEXT NOT NULL,
                        status TEXT NOT NULL,
                        config_json TEXT NOT NULL,
                        runtime_json TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        error TEXT,
                        metadata_json TEXT NOT NULL
                    )
                    """
                )
            )
            connection.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS idx_model_services_updated_at
                    ON model_services(updated_at DESC)
                    """
                )
            )
            connection.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS prompt_templates (
                        prompt_id TEXT PRIMARY KEY,
                        label TEXT NOT NULL,
                        task TEXT NOT NULL,
                        system_prompt TEXT NOT NULL,
                        user_prompt TEXT NOT NULL,
                        parser TEXT,
                        metric_profile TEXT,
                        visualization_profile TEXT,
                        generation_json TEXT NOT NULL,
                        data_json TEXT NOT NULL,
                        metadata_json TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
            )
            connection.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS idx_prompt_templates_task
                    ON prompt_templates(task, prompt_id)
                    """
                )
            )
        self.seed_prompt_templates(default_prompt_templates())

    def create_job(
        self,
        *,
        kind: str,
        payload: dict[str, Any],
        metadata: dict[str, Any] | None = None,
        status: str = "queued",
        job_id: str | None = None,
    ) -> JobRecord:
        if not kind.strip():
            raise ValueError("job kind must be non-empty.")
        if not isinstance(payload, dict):
            raise ValueError("job payload must be a dict.")
        if status not in {"queued", "running", "succeeded", "failed", "cancelled"}:
            raise ValueError(f"unsupported job status: {status}")
        now = utc_now_iso()
        record = JobRecord(
            job_id=job_id or _job_id(kind.strip()),
            kind=kind.strip(),
            status=status,
            payload=payload,
            created_at=now,
            updated_at=now,
            metadata=dict(metadata or {}),
        )
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO jobs (
                        job_id, kind, status, payload_json, created_at, updated_at, error,
                        metadata_json
                    ) VALUES (
                        :job_id, :kind, :status, :payload_json, :created_at, :updated_at,
                        :error, :metadata_json
                    )
                    """
                ),
                {
                    "job_id": record.job_id,
                    "kind": record.kind,
                    "status": record.status,
                    "payload_json": json.dumps(record.payload, ensure_ascii=False),
                    "created_at": record.created_at,
                    "updated_at": record.updated_at,
                    "error": record.error,
                    "metadata_json": json.dumps(record.metadata, ensure_ascii=False),
                },
            )
        return record

    def _row_to_job(self, row: Any) -> JobRecord:
        return JobRecord(
            job_id=str(row["job_id"]),
            kind=str(row["kind"]),
            status=str(row["status"]),
            payload=json.loads(str(row["payload_json"])),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            error=row["error"],
            metadata=json.loads(str(row["metadata_json"])),
        )

    def get_job(self, job_id: str) -> JobRecord | None:
        with self.engine.begin() as connection:
            row = connection.execute(
                text(
                    """
                    SELECT job_id, kind, status, payload_json, created_at, updated_at, error,
                           metadata_json
                    FROM jobs
                    WHERE job_id = :job_id
                    """
                ),
                {"job_id": str(job_id)},
            ).mappings().first()
            return self._row_to_job(row) if row is not None else None

    def claim_next_job(self, *, kind: str | None = None) -> JobRecord | None:
        with self.engine.begin() as connection:
            if kind is None:
                row = connection.execute(
                    text(
                        """
                        SELECT job_id, kind, status, payload_json, created_at, updated_at, error,
                               metadata_json
                        FROM jobs
                        WHERE status = 'queued'
                        ORDER BY created_at ASC, job_id ASC
                        LIMIT 1
                        """
                    )
                ).mappings().first()
            else:
                row = connection.execute(
                    text(
                        """
                        SELECT job_id, kind, status, payload_json, created_at, updated_at, error,
                               metadata_json
                        FROM jobs
                        WHERE status = 'queued' AND kind = :kind
                        ORDER BY created_at ASC, job_id ASC
                        LIMIT 1
                        """
                    ),
                    {"kind": str(kind)},
                ).mappings().first()
            if row is None:
                return None
            job_id = str(row["job_id"])
            now = utc_now_iso()
            result = connection.execute(
                text(
                    """
                    UPDATE jobs
                    SET status = 'running', updated_at = :updated_at, error = NULL
                    WHERE job_id = :job_id AND status = 'queued'
                    """
                ),
                {"job_id": job_id, "updated_at": now},
            )
            if result.rowcount != 1:
                return None
            updated = dict(row)
            updated["status"] = "running"
            updated["updated_at"] = now
            updated["error"] = None
            return self._row_to_job(updated)

    def claim_job(self, job_id: str) -> JobRecord | None:
        with self.engine.begin() as connection:
            row = connection.execute(
                text(
                    """
                    SELECT job_id, kind, status, payload_json, created_at, updated_at, error,
                           metadata_json
                    FROM jobs
                    WHERE job_id = :job_id AND status = 'queued'
                    """
                ),
                {"job_id": str(job_id)},
            ).mappings().first()
            if row is None:
                return None
            now = utc_now_iso()
            result = connection.execute(
                text(
                    """
                    UPDATE jobs
                    SET status = 'running', updated_at = :updated_at, error = NULL
                    WHERE job_id = :job_id AND status = 'queued'
                    """
                ),
                {"job_id": str(job_id), "updated_at": now},
            )
            if result.rowcount != 1:
                return None
            updated = dict(row)
            updated["status"] = "running"
            updated["updated_at"] = now
            updated["error"] = None
            return self._row_to_job(updated)

    def update_job(
        self,
        job_id: str,
        *,
        status: str,
        error: str | None = None,
        metadata_update: dict[str, Any] | None = None,
    ) -> JobRecord:
        if status not in {"queued", "running", "succeeded", "failed", "cancelled"}:
            raise ValueError(f"unsupported job status: {status}")
        current = self.get_job(job_id)
        if current is None:
            raise KeyError(f"unknown job_id: {job_id}")
        metadata = {**current.metadata, **dict(metadata_update or {})}
        now = utc_now_iso()
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    """
                    UPDATE jobs
                    SET status = :status,
                        updated_at = :updated_at,
                        error = :error,
                        metadata_json = :metadata_json
                    WHERE job_id = :job_id
                    """
                ),
                {
                    "job_id": str(job_id),
                    "status": status,
                    "updated_at": now,
                    "error": error,
                    "metadata_json": json.dumps(metadata, ensure_ascii=False),
                },
            )
        updated = self.get_job(job_id)
        if updated is None:  # pragma: no cover
            raise KeyError(f"unknown job_id after update: {job_id}")
        return updated

    def job_page(
        self,
        *,
        offset: int = 0,
        limit: int = 100,
        kind: str | None = None,
        status: str | None = None,
        query: str | None = None,
    ) -> JobListPage:
        start, page_limit = _page_bounds(offset=offset, limit=limit)
        filters = {
            "kind": _filter_value(kind),
            "status": _filter_value(status),
            "query": (query or "").strip(),
        }
        all_jobs = self.matching_jobs()
        jobs = [
            job
            for job in all_jobs
            if _job_matches_filters(
                job,
                kind=filters["kind"],
                status=filters["status"],
                query=filters["query"].lower(),
            )
        ]
        return JobListPage(
            offset=start,
            limit=page_limit,
            total=len(jobs),
            filters=filters,
            facets=_job_facets(all_jobs),
            jobs=jobs[start : start + page_limit],
        )

    def matching_jobs(
        self,
        *,
        kind: str | None = None,
        status: str | None = None,
        query: str | None = None,
    ) -> list[JobRecord]:
        filters = {
            "kind": _filter_value(kind),
            "status": _filter_value(status),
            "query": (query or "").strip(),
        }
        with self.engine.begin() as connection:
            rows = connection.execute(
                text(
                    """
                    SELECT job_id, kind, status, payload_json, created_at, updated_at, error,
                           metadata_json
                    FROM jobs
                    ORDER BY created_at DESC, job_id DESC
                    """
                )
            ).mappings()
            jobs = [
                job
                for job in (self._row_to_job(row) for row in rows)
                if _job_matches_filters(
                    job,
                    kind=filters["kind"],
                    status=filters["status"],
                    query=filters["query"].lower(),
                )
            ]
        return jobs

    def list_jobs(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        kind: str | None = None,
        status: str | None = None,
        query: str | None = None,
    ) -> list[JobRecord]:
        return self.job_page(
            offset=offset,
            limit=limit,
            kind=kind,
            status=status,
            query=query,
        ).jobs

    def cancel_job(self, job_id: str) -> JobRecord:
        current = self.get_job(job_id)
        if current is None:
            raise KeyError(f"unknown job_id: {job_id}")
        if current.status not in {"queued", "running"}:
            raise ValueError(
                "only queued or running jobs can be cancelled, "
                f"current status is {current.status!r}"
            )
        return self.update_job(
            job_id,
            status="cancelled",
            metadata_update={
                "cancel_requested": True,
                "cancel_requested_at": utc_now_iso(),
                "progress_phase": "cancelled",
                "progress_message": "Cancellation requested by user.",
                "progress_updated_at": utc_now_iso(),
            },
        )

    def delete_job(self, job_id: str) -> JobRecord:
        current = self.get_job(job_id)
        if current is None:
            raise KeyError(f"unknown job_id: {job_id}")
        with self.engine.begin() as connection:
            connection.execute(text("DELETE FROM jobs WHERE job_id = :job_id"), {"job_id": job_id})
        return current

    def upsert_service(
        self,
        *,
        kind: str,
        config: dict[str, Any],
        service_id: str | None = None,
        status: str = "registered",
        metadata: dict[str, Any] | None = None,
    ) -> ServiceRecord:
        kind = kind.strip()
        if not kind:
            raise ValueError("service kind must be non-empty.")
        if kind not in {"local_vllm", "external_vllm"}:
            raise ValueError(f"unsupported service kind: {kind}")
        if not isinstance(config, dict):
            raise ValueError("service config must be a dict.")
        if status not in {"registered", "starting", "running", "stopped", "failed"}:
            raise ValueError(f"unsupported service status: {status}")
        resolved_service_id = (service_id or config.get("service_id") or _service_id(kind)).strip()
        if not resolved_service_id:
            raise ValueError("service_id must be non-empty.")
        now = utc_now_iso()
        current = self.get_service(resolved_service_id)
        created_at = current.created_at if current is not None else now
        runtime = current.runtime if current is not None else {}
        record = ServiceRecord(
            service_id=resolved_service_id,
            kind=kind,
            status=status,
            config={key: value for key, value in config.items() if key != "service_id"},
            runtime=runtime,
            created_at=created_at,
            updated_at=now,
            metadata=dict(metadata or (current.metadata if current is not None else {})),
        )
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO model_services (
                        service_id, kind, status, config_json, runtime_json, created_at,
                        updated_at, error, metadata_json
                    ) VALUES (
                        :service_id, :kind, :status, :config_json, :runtime_json, :created_at,
                        :updated_at, :error, :metadata_json
                    )
                    ON CONFLICT(service_id) DO UPDATE SET
                        kind = excluded.kind,
                        status = excluded.status,
                        config_json = excluded.config_json,
                        updated_at = excluded.updated_at,
                        error = excluded.error,
                        metadata_json = excluded.metadata_json
                    """
                ),
                {
                    "service_id": record.service_id,
                    "kind": record.kind,
                    "status": record.status,
                    "config_json": json.dumps(record.config, ensure_ascii=False),
                    "runtime_json": json.dumps(record.runtime, ensure_ascii=False),
                    "created_at": record.created_at,
                    "updated_at": record.updated_at,
                    "error": record.error,
                    "metadata_json": json.dumps(record.metadata, ensure_ascii=False),
                },
            )
        return record

    def _row_to_service(self, row: Any) -> ServiceRecord:
        return ServiceRecord(
            service_id=str(row["service_id"]),
            kind=str(row["kind"]),
            status=str(row["status"]),
            config=json.loads(str(row["config_json"])),
            runtime=json.loads(str(row["runtime_json"])),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            error=row["error"],
            metadata=json.loads(str(row["metadata_json"])),
        )

    def get_service(self, service_id: str) -> ServiceRecord | None:
        with self.engine.begin() as connection:
            row = connection.execute(
                text(
                    """
                    SELECT service_id, kind, status, config_json, runtime_json, created_at,
                           updated_at, error, metadata_json
                    FROM model_services
                    WHERE service_id = :service_id
                    """
                ),
                {"service_id": str(service_id)},
            ).mappings().first()
            return self._row_to_service(row) if row is not None else None

    def service_page(
        self,
        *,
        offset: int = 0,
        limit: int = 100,
        kind: str | None = None,
        status: str | None = None,
        query: str | None = None,
    ) -> ServiceListPage:
        start, page_limit = _page_bounds(offset=offset, limit=limit)
        filters = {
            "kind": _filter_value(kind),
            "status": _filter_value(status),
            "query": (query or "").strip(),
        }
        with self.engine.begin() as connection:
            rows = connection.execute(
                text(
                    """
                    SELECT service_id, kind, status, config_json, runtime_json, created_at,
                           updated_at, error, metadata_json
                    FROM model_services
                    ORDER BY updated_at DESC, service_id DESC
                    """
                )
            ).mappings()
            all_services = [self._row_to_service(row) for row in rows]
            services = [
                service
                for service in all_services
                if _service_matches_filters(
                    service,
                    kind=filters["kind"],
                    status=filters["status"],
                    query=filters["query"].lower(),
                )
            ]
        return ServiceListPage(
            offset=start,
            limit=page_limit,
            total=len(services),
            filters=filters,
            facets=_service_facets(all_services),
            services=services[start : start + page_limit],
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

    def update_service_runtime(
        self,
        service_id: str,
        *,
        status: str,
        runtime: dict[str, Any] | None = None,
        error: str | None = None,
        metadata_update: dict[str, Any] | None = None,
    ) -> ServiceRecord:
        if status not in {"registered", "starting", "running", "stopped", "failed"}:
            raise ValueError(f"unsupported service status: {status}")
        current = self.get_service(service_id)
        if current is None:
            raise KeyError(f"unknown service_id: {service_id}")
        merged_runtime = dict(current.runtime)
        if runtime is not None:
            merged_runtime.update(runtime)
        metadata = {**current.metadata, **dict(metadata_update or {})}
        now = utc_now_iso()
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    """
                    UPDATE model_services
                    SET status = :status,
                        runtime_json = :runtime_json,
                        updated_at = :updated_at,
                        error = :error,
                        metadata_json = :metadata_json
                    WHERE service_id = :service_id
                    """
                ),
                {
                    "service_id": str(service_id),
                    "status": status,
                    "runtime_json": json.dumps(merged_runtime, ensure_ascii=False),
                    "updated_at": now,
                    "error": error,
                    "metadata_json": json.dumps(metadata, ensure_ascii=False),
                },
            )
        updated = self.get_service(service_id)
        if updated is None:  # pragma: no cover
            raise KeyError(f"unknown service_id after update: {service_id}")
        return updated

    def delete_service(self, service_id: str) -> ServiceRecord:
        current = self.get_service(service_id)
        if current is None:
            raise KeyError(f"unknown service_id: {service_id}")
        with self.engine.begin() as connection:
            connection.execute(
                text("DELETE FROM model_services WHERE service_id = :service_id"),
                {"service_id": service_id},
            )
        return current

    def seed_prompt_templates(self, templates: list[dict[str, Any]]) -> None:
        now = utc_now_iso()
        with self.engine.begin() as connection:
            for template in templates:
                record = self._prompt_template_from_payload(template, created_at=now, updated_at=now)
                existing = connection.execute(
                    text(
                        """
                        SELECT metadata_json
                        FROM prompt_templates
                        WHERE prompt_id = :prompt_id
                        """
                    ),
                    {"prompt_id": record.prompt_id},
                ).mappings().first()
                if existing is not None and not _is_repo_prompt_metadata(existing["metadata_json"]):
                    continue
                params = _prompt_template_params(record)
                connection.execute(
                    text(
                        """
                        INSERT INTO prompt_templates (
                            prompt_id, label, task, system_prompt, user_prompt, parser,
                            metric_profile, visualization_profile, generation_json, data_json,
                            metadata_json, created_at, updated_at
                        ) VALUES (
                            :prompt_id, :label, :task, :system_prompt, :user_prompt, :parser,
                            :metric_profile, :visualization_profile, :generation_json, :data_json,
                            :metadata_json, :created_at, :updated_at
                        )
                        ON CONFLICT(prompt_id) DO UPDATE SET
                            label = excluded.label,
                            task = excluded.task,
                            system_prompt = excluded.system_prompt,
                            user_prompt = excluded.user_prompt,
                            parser = excluded.parser,
                            metric_profile = excluded.metric_profile,
                            visualization_profile = excluded.visualization_profile,
                            generation_json = excluded.generation_json,
                            data_json = excluded.data_json,
                            metadata_json = excluded.metadata_json,
                            updated_at = excluded.updated_at
                        """
                    ),
                    params,
                )

    def upsert_prompt_template(self, payload: dict[str, Any]) -> PromptTemplateRecord:
        if not isinstance(payload, dict):
            raise ValueError("prompt template payload must be a dict.")
        current = self.get_prompt_template(str(payload.get("prompt_id") or ""))
        created_at = current.created_at if current is not None else utc_now_iso()
        record = self._prompt_template_from_payload(
            payload,
            created_at=created_at,
            updated_at=utc_now_iso(),
        )
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO prompt_templates (
                        prompt_id, label, task, system_prompt, user_prompt, parser,
                        metric_profile, visualization_profile, generation_json, data_json,
                        metadata_json, created_at, updated_at
                    ) VALUES (
                        :prompt_id, :label, :task, :system_prompt, :user_prompt, :parser,
                        :metric_profile, :visualization_profile, :generation_json, :data_json,
                        :metadata_json, :created_at, :updated_at
                    )
                    ON CONFLICT(prompt_id) DO UPDATE SET
                        label = excluded.label,
                        task = excluded.task,
                        system_prompt = excluded.system_prompt,
                        user_prompt = excluded.user_prompt,
                        parser = excluded.parser,
                        metric_profile = excluded.metric_profile,
                        visualization_profile = excluded.visualization_profile,
                        generation_json = excluded.generation_json,
                        data_json = excluded.data_json,
                        metadata_json = excluded.metadata_json,
                        updated_at = excluded.updated_at
                    """
                ),
                _prompt_template_params(record),
            )
        return record

    def get_prompt_template(self, prompt_id: str) -> PromptTemplateRecord | None:
        prompt_id = str(prompt_id).strip()
        if not prompt_id:
            return None
        with self.engine.begin() as connection:
            row = connection.execute(
                text(
                    """
                    SELECT prompt_id, label, task, system_prompt, user_prompt, parser,
                           metric_profile, visualization_profile, generation_json, data_json,
                           metadata_json, created_at, updated_at
                    FROM prompt_templates
                    WHERE prompt_id = :prompt_id
                    """
                ),
                {"prompt_id": prompt_id},
            ).mappings().first()
            return self._row_to_prompt_template(row) if row is not None else None

    def list_prompt_templates(
        self,
        *,
        task: str | None = None,
        limit: int = 500,
    ) -> list[PromptTemplateRecord]:
        if limit <= 0:
            raise ValueError("limit must be > 0.")
        with self.engine.begin() as connection:
            if task:
                rows = connection.execute(
                    text(
                        """
                        SELECT prompt_id, label, task, system_prompt, user_prompt, parser,
                               metric_profile, visualization_profile, generation_json, data_json,
                               metadata_json, created_at, updated_at
                        FROM prompt_templates
                        WHERE task = :task
                        ORDER BY prompt_id ASC
                        LIMIT :limit
                        """
                    ),
                    {"task": str(task), "limit": int(limit)},
                ).mappings()
            else:
                rows = connection.execute(
                    text(
                        """
                        SELECT prompt_id, label, task, system_prompt, user_prompt, parser,
                               metric_profile, visualization_profile, generation_json, data_json,
                               metadata_json, created_at, updated_at
                        FROM prompt_templates
                        ORDER BY task ASC, prompt_id ASC
                        LIMIT :limit
                        """
                    ),
                    {"limit": int(limit)},
                ).mappings()
            return [self._row_to_prompt_template(row) for row in rows]

    def delete_prompt_template(self, prompt_id: str) -> PromptTemplateRecord:
        current = self.get_prompt_template(prompt_id)
        if current is None:
            raise KeyError(f"unknown prompt_id: {prompt_id}")
        with self.engine.begin() as connection:
            connection.execute(
                text("DELETE FROM prompt_templates WHERE prompt_id = :prompt_id"),
                {"prompt_id": current.prompt_id},
            )
        return current

    def _prompt_template_from_payload(
        self,
        payload: dict[str, Any],
        *,
        created_at: str,
        updated_at: str,
    ) -> PromptTemplateRecord:
        prompt_id = str(payload.get("prompt_id") or "").strip()
        if not prompt_id:
            raise ValueError("prompt_id must be non-empty.")
        label = str(payload.get("label") or prompt_id).strip()
        task = str(payload.get("task") or "").strip()
        if task not in {"detection", "keypoint"}:
            raise ValueError(f"unsupported prompt template task: {task}")
        system_prompt = str(payload.get("system_prompt") or "").strip()
        user_prompt = str(payload.get("user_prompt") or payload.get("prompt_text") or "").strip()
        if not user_prompt:
            raise ValueError("user_prompt must be non-empty.")
        return PromptTemplateRecord(
            prompt_id=prompt_id,
            label=label,
            task=task,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            parser=_optional_string(payload.get("parser")),
            metric_profile=_optional_string(payload.get("metric_profile")),
            visualization_profile=_optional_string(payload.get("visualization_profile")),
            generation=_dict_or_empty(payload.get("generation")),
            data=_dict_or_empty(payload.get("data")),
            metadata=_dict_or_empty(payload.get("metadata")),
            created_at=created_at,
            updated_at=updated_at,
        )

    def _row_to_prompt_template(self, row: Any) -> PromptTemplateRecord:
        return PromptTemplateRecord(
            prompt_id=str(row["prompt_id"]),
            label=str(row["label"]),
            task=str(row["task"]),
            system_prompt=str(row["system_prompt"]),
            user_prompt=str(row["user_prompt"]),
            parser=row["parser"],
            metric_profile=row["metric_profile"],
            visualization_profile=row["visualization_profile"],
            generation=json.loads(str(row["generation_json"])),
            data=json.loads(str(row["data_json"])),
            metadata=json.loads(str(row["metadata_json"])),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )


def _prompt_template_params(record: PromptTemplateRecord) -> dict[str, Any]:
    return {
        "prompt_id": record.prompt_id,
        "label": record.label,
        "task": record.task,
        "system_prompt": record.system_prompt,
        "user_prompt": record.user_prompt,
        "parser": record.parser,
        "metric_profile": record.metric_profile,
        "visualization_profile": record.visualization_profile,
        "generation_json": json.dumps(record.generation, ensure_ascii=False),
        "data_json": json.dumps(record.data, ensure_ascii=False),
        "metadata_json": json.dumps(record.metadata, ensure_ascii=False),
        "created_at": record.created_at,
        "updated_at": record.updated_at,
    }


def _optional_string(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if value not in (None, "") and not isinstance(value, (dict, list)):
        return str(value)
    return None


def _is_repo_prompt_metadata(value: Any) -> bool:
    try:
        metadata = json.loads(str(value or "{}"))
    except json.JSONDecodeError:
        return False
    return isinstance(metadata, dict) and (
        metadata.get("source") == "repo_config" or not metadata
    )


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _page_bounds(*, offset: int, limit: int) -> tuple[int, int]:
    return max(0, int(offset)), max(1, int(limit))


def _filter_value(value: str | None) -> str:
    if value is None:
        return ""
    normalized = str(value).strip()
    return "" if normalized == "all" else normalized


def _job_matches_filters(
    job: JobRecord,
    *,
    kind: str,
    status: str,
    query: str,
) -> bool:
    if kind and job.kind != kind:
        return False
    if status and job.status != status:
        return False
    if query and not _job_query_matches(job, query):
        return False
    return True


def _job_query_matches(job: JobRecord, query: str) -> bool:
    fields = [
        job.job_id,
        job.kind,
        job.status,
        job.error,
        job.created_at,
        job.updated_at,
        json.dumps(job.payload, ensure_ascii=False, sort_keys=True),
        json.dumps(job.metadata, ensure_ascii=False, sort_keys=True),
    ]
    return any(query in str(field or "").lower() for field in fields)


def _job_facets(jobs: list[JobRecord]) -> dict[str, list[dict[str, Any]]]:
    return {
        "kinds": _facet_counts(jobs, lambda job: [job.kind or "unknown"]),
        "statuses": _facet_counts(jobs, lambda job: [job.status or "unknown"]),
    }


def _service_matches_filters(
    service: ServiceRecord,
    *,
    kind: str,
    status: str,
    query: str,
) -> bool:
    if kind and service.kind != kind:
        return False
    if status and service.status != status:
        return False
    if query and not _service_query_matches(service, query):
        return False
    return True


def _service_facets(services: list[ServiceRecord]) -> dict[str, list[dict[str, Any]]]:
    return {
        "kinds": _facet_counts(services, lambda service: [service.kind or "unknown"]),
        "statuses": _facet_counts(services, lambda service: [service.status or "unknown"]),
    }


def _facet_counts(
    items: list[Any],
    values: Callable[[Any], list[str]],
) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for item in items:
        for value in values(item):
            key = str(value or "unknown")
            counts[key] = counts.get(key, 0) + 1
    return [
        {"value": value, "count": count}
        for value, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def _service_query_matches(service: ServiceRecord, query: str) -> bool:
    fields = [
        service.service_id,
        service.kind,
        service.status,
        service.error,
        service.created_at,
        service.updated_at,
        json.dumps(service.config, ensure_ascii=False, sort_keys=True),
        json.dumps(service.runtime, ensure_ascii=False, sort_keys=True),
        json.dumps(service.metadata, ensure_ascii=False, sort_keys=True),
    ]
    return any(query in str(field or "").lower() for field in fields)


def _configure_sqlite_connection(dbapi_connection: Any, _connection_record: Any) -> None:
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.execute("PRAGMA foreign_keys=ON")
    finally:
        cursor.close()
