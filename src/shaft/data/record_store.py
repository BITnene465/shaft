from __future__ import annotations

from bisect import bisect_right
from array import array
from collections.abc import Iterator, Sequence
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Generic, TypeVar
import uuid

from .dataset import DPORecord, PPORecord, SFTRecord

RecordT = TypeVar("RecordT", SFTRecord, DPORecord, PPORecord)

# Bump whenever normalized record schema or row-building semantics change.
_CACHE_FORMAT_VERSION = "shaft-arrow-record-store-v3"
_JSON_FIELDS = {"messages", "prompt_args", "extra"}
_RECORD_TYPES = {
    record_type.__name__: record_type
    for record_type in (SFTRecord, DPORecord, PPORecord)
}


def _import_pyarrow():
    try:
        import pyarrow as pa
    except ImportError as exc:  # pragma: no cover - depends on optional train extra
        raise RuntimeError(
            "Arrow-backed Shaft data loading requires the `train` extra: "
            "install with `uv pip install -e '.[train]'`."
        ) from exc
    return pa


def _default_cache_dir() -> Path:
    configured = str(os.environ.get("SHAFT_RECORD_CACHE_DIR", "")).strip()
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache" / "shaft" / "records"


def _source_fingerprint(
    path: Path,
    *,
    dataset_name: str,
    record_type: type[RecordT],
    validation_fingerprint: str = "",
) -> str:
    stat = path.stat()
    digest = hashlib.sha256()
    digest.update(_CACHE_FORMAT_VERSION.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(path.resolve()).encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(dataset_name).encode("utf-8"))
    digest.update(b"\0")
    digest.update(record_type.__name__.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(validation_fingerprint).encode("utf-8"))
    digest.update(b"\0")
    digest.update(
        f"{stat.st_size}:{stat.st_mtime_ns}:{stat.st_ctime_ns}".encode("utf-8")
    )
    return digest.hexdigest()


def _record_to_arrow_row(record: RecordT) -> dict[str, str | None]:
    row: dict[str, str | None] = {}
    for field_info in fields(record):
        value = getattr(record, field_info.name)
        if field_info.name in _JSON_FIELDS:
            row[field_info.name] = (
                None
                if value is None
                else json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            )
        elif value is None:
            row[field_info.name] = None
        else:
            row[field_info.name] = str(value)
    return row


class ShaftArrowRecordStore(Sequence[RecordT], Generic[RecordT]):
    """Immutable, memory-mapped Arrow storage keyed by a source snapshot fingerprint."""

    def __init__(
        self,
        cache_path: str | Path,
        *,
        record_type: type[RecordT],
        fingerprint: str,
    ) -> None:
        self.cache_path = str(cache_path)
        self.record_type_name = record_type.__name__
        self.fingerprint = str(fingerprint)
        self._open()

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        *,
        dataset_name: str,
        record_type: type[RecordT],
        row_builder: Any,
        record_validator: Any | None = None,
        validation_fingerprint: str = "",
        max_errors_to_report: int = 20,
        cache_dir: str | Path | None = None,
        batch_size: int = 4096,
    ) -> ShaftArrowRecordStore[RecordT]:
        resolved_validation_fingerprint = str(validation_fingerprint).strip()
        if (record_validator is None) != (not resolved_validation_fingerprint):
            raise ValueError(
                "record_validator and validation_fingerprint must be provided together."
            )
        jsonl_path = Path(path).resolve()
        fingerprint = _source_fingerprint(
            jsonl_path,
            dataset_name=dataset_name,
            record_type=record_type,
            validation_fingerprint=resolved_validation_fingerprint,
        )
        resolved_cache_dir = (
            Path(cache_dir).expanduser() if cache_dir is not None else _default_cache_dir()
        )
        resolved_cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = resolved_cache_dir / f"{fingerprint}.arrow"
        lock_path = resolved_cache_dir / f"{fingerprint}.lock"

        with lock_path.open("a+b") as lock_handle:
            try:
                import fcntl

                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            except ImportError:  # pragma: no cover - Shaft training targets Linux
                pass
            if cache_path.exists():
                try:
                    return cls(
                        cache_path,
                        record_type=record_type,
                        fingerprint=fingerprint,
                    )
                except Exception:  # noqa: BLE001 - corrupt cache is rebuilt atomically
                    cache_path.unlink(missing_ok=True)
            cls._build_cache(
                jsonl_path,
                cache_path=cache_path,
                dataset_name=dataset_name,
                record_type=record_type,
                row_builder=row_builder,
                record_validator=record_validator,
                max_errors_to_report=max_errors_to_report,
                batch_size=batch_size,
            )
        return cls(cache_path, record_type=record_type, fingerprint=fingerprint)

    @staticmethod
    def _build_cache(
        jsonl_path: Path,
        *,
        cache_path: Path,
        dataset_name: str,
        record_type: type[RecordT],
        row_builder: Any,
        record_validator: Any | None,
        max_errors_to_report: int,
        batch_size: int,
    ) -> None:
        pa = _import_pyarrow()
        schema = pa.schema(
            [pa.field(field_info.name, pa.large_string()) for field_info in fields(record_type)],
            metadata={
                b"shaft_format": _CACHE_FORMAT_VERSION.encode("utf-8"),
                b"record_type": record_type.__name__.encode("utf-8"),
            },
        )
        temp_path = cache_path.with_name(f".{cache_path.name}.{uuid.uuid4().hex}.tmp")
        parse_errors: list[tuple[int, str]] = []
        total_parse_errors = 0
        rows: list[dict[str, str | None]] = []
        sink = pa.OSFile(str(temp_path), "wb")
        writer = pa.ipc.new_file(sink, schema)
        try:
            try:
                with jsonl_path.open("r", encoding="utf-8") as handle:
                    for line_no, line in enumerate(handle, start=1):
                        text = line.strip()
                        if not text:
                            continue
                        try:
                            raw = json.loads(text)
                            if not isinstance(raw, dict):
                                raise TypeError("Each JSONL row must be a JSON object.")
                            record = row_builder(
                                raw,
                                jsonl_path=jsonl_path,
                                line_no=line_no,
                                dataset_name=dataset_name,
                            )
                            if record_validator is not None:
                                record_validator(record)
                            row = _record_to_arrow_row(record)
                        except Exception as exc:  # noqa: BLE001 - aggregate row diagnostics
                            total_parse_errors += 1
                            if len(parse_errors) < max(1, int(max_errors_to_report)):
                                parse_errors.append((line_no, str(exc)))
                            continue
                        rows.append(row)
                        if len(rows) >= max(int(batch_size), 1):
                            writer.write_batch(pa.RecordBatch.from_pylist(rows, schema=schema))
                            rows.clear()
                if rows:
                    writer.write_batch(pa.RecordBatch.from_pylist(rows, schema=schema))
            finally:
                try:
                    writer.close()
                finally:
                    sink.close()

            if total_parse_errors:
                snippets = [f"L{line_no}: {message}" for line_no, message in parse_errors]
                raise ValueError(
                    f"Failed to parse {total_parse_errors} row(s) in {jsonl_path}. "
                    f"Examples: {'; '.join(snippets)}"
                )
            os.replace(temp_path, cache_path)
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise

    @property
    def record_type(self) -> type[RecordT]:
        return _RECORD_TYPES[self.record_type_name]

    def _open(self) -> None:
        pa = _import_pyarrow()
        self._memory_map = pa.memory_map(self.cache_path, "r")
        self._reader = pa.ipc.open_file(self._memory_map)
        metadata = self._reader.schema.metadata or {}
        if metadata.get(b"shaft_format", b"").decode("utf-8") != _CACHE_FORMAT_VERSION:
            raise ValueError(f"Unsupported Shaft Arrow cache format: {self.cache_path}")
        if metadata.get(b"record_type", b"").decode("utf-8") != self.record_type_name:
            raise ValueError(f"Arrow cache record type mismatch: {self.cache_path}")
        expected_columns = [field_info.name for field_info in fields(self.record_type)]
        if self._reader.schema.names != expected_columns:
            raise ValueError(f"Arrow cache schema mismatch: {self.cache_path}")
        self._table = self._reader.read_all()
        self._columns = {
            name: self._table.column(name)
            for name in self._table.column_names
        }

    def __len__(self) -> int:
        return int(self._table.num_rows)

    def __getitem__(self, index: int | slice) -> RecordT | list[RecordT]:
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]
        position = int(index)
        if position < 0:
            position += len(self)
        if position < 0 or position >= len(self):
            raise IndexError(position)
        payload: dict[str, Any] = {}
        for field_info in fields(self.record_type):
            value = self._columns[field_info.name][position].as_py()
            if field_info.name in _JSON_FIELDS and value is not None:
                value = json.loads(value)
            payload[field_info.name] = value
        return self.record_type(**payload)

    def __getstate__(self) -> dict[str, str]:
        return {
            "cache_path": self.cache_path,
            "record_type_name": self.record_type_name,
            "fingerprint": self.fingerprint,
        }

    def __setstate__(self, state: dict[str, str]) -> None:
        self.cache_path = state["cache_path"]
        self.record_type_name = state["record_type_name"]
        self.fingerprint = state["fingerprint"]
        self._open()


class ShaftConcatRecordStore(Sequence[RecordT], Generic[RecordT]):
    def __init__(self, stores: Sequence[Sequence[RecordT]]) -> None:
        self.stores = tuple(store for store in stores if len(store) > 0)
        self._ends: list[int] = []
        total = 0
        for store in self.stores:
            total += len(store)
            self._ends.append(total)
        fingerprints = [str(getattr(store, "fingerprint", len(store))) for store in self.stores]
        self.fingerprint = hashlib.sha256("\0".join(fingerprints).encode("utf-8")).hexdigest()

    def __len__(self) -> int:
        return self._ends[-1] if self._ends else 0

    def __getitem__(self, index: int | slice) -> RecordT | list[RecordT]:
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]
        position = int(index)
        if position < 0:
            position += len(self)
        if position < 0 or position >= len(self):
            raise IndexError(position)
        store_index = bisect_right(self._ends, position)
        start = 0 if store_index == 0 else self._ends[store_index - 1]
        return self.stores[store_index][position - start]

    def __iter__(self) -> Iterator[RecordT]:
        for store in self.stores:
            yield from store


class ShaftRecordSubset(Sequence[RecordT], Generic[RecordT]):
    def __init__(self, records: Sequence[RecordT], indices: Sequence[int]) -> None:
        self.records = records
        self.indices = array("Q", (int(index) for index in indices))
        digest = hashlib.sha256(str(getattr(records, "fingerprint", "")).encode("utf-8"))
        digest.update(self.indices.tobytes())
        self.fingerprint = digest.hexdigest()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int | slice) -> RecordT | list[RecordT]:
        if isinstance(index, slice):
            return [self.records[position] for position in self.indices[index]]
        return self.records[self.indices[int(index)]]
