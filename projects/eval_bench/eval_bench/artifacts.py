from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
import shutil
from pathlib import Path
from typing import Any

from .sample_paths import prediction_json_relative_path
from .schema import BenchmarkManifest, EvalRunManifest, PredictionDocument, TaskKind

DEFAULT_STORE_ROOT = Path("eval_bench_store")


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = asdict(payload) if is_dataclass(payload) else payload
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


class StoreLayout:
    def __init__(self, root: str | Path = DEFAULT_STORE_ROOT) -> None:
        self.root = Path(root)

    @property
    def db_dir(self) -> Path:
        return self.root / "db"

    @property
    def db_path(self) -> Path:
        return self.db_dir / "eval_bench.sqlite"

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    @property
    def benchmarks_dir(self) -> Path:
        return self.root / "benchmarks"

    @property
    def tmp_dir(self) -> Path:
        return self.root / "tmp"

    @property
    def exports_dir(self) -> Path:
        return self.root / "exports"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def services_dir(self) -> Path:
        return self.root / "services"

    @property
    def trash_dir(self) -> Path:
        return self.root / "trash"

    def ensure(self) -> None:
        for path in (
            self.root,
            self.db_dir,
            self.runs_dir,
            self.benchmarks_dir,
            self.tmp_dir,
            self.exports_dir,
            self.logs_dir,
            self.services_dir,
            self.trash_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def move_to_trash(self, path: Path, *, category: str) -> Path | None:
        if not path.exists():
            return None
        self.ensure()
        trash_category = self.trash_dir / category
        trash_category.mkdir(parents=True, exist_ok=True)
        target = trash_category / path.name
        if target.exists():
            suffix = 1
            while (trash_category / f"{path.name}.{suffix}").exists():
                suffix += 1
            target = trash_category / f"{path.name}.{suffix}"
        shutil.move(str(path), str(target))
        return target


class BenchmarkArtifacts:
    def __init__(self, root: str | Path = DEFAULT_STORE_ROOT, benchmark_id: str = "") -> None:
        self.store = StoreLayout(root)
        self.root = self.store.root
        self.benchmark_id = str(benchmark_id)
        if not self.benchmark_id:
            raise ValueError("benchmark_id must be a non-empty string.")
        self.benchmark_dir = self.store.benchmarks_dir / self.benchmark_id

    @property
    def manifest_path(self) -> Path:
        return self.benchmark_dir / "benchmark.json"

    @property
    def data_dir(self) -> Path:
        return self.benchmark_dir / "data"

    @property
    def splits_dir(self) -> Path:
        return self.benchmark_dir / "splits"

    @property
    def previews_dir(self) -> Path:
        return self.benchmark_dir / "previews"

    def split_path(self, split: str) -> Path:
        return self.splits_dir / f"{split}.txt"

    def ensure(self) -> None:
        self.store.ensure()
        for path in (self.benchmark_dir, self.data_dir, self.splits_dir, self.previews_dir):
            path.mkdir(parents=True, exist_ok=True)

    def write_manifest(self, manifest: BenchmarkManifest) -> Path:
        manifest.validate()
        self.ensure()
        atomic_write_json(self.manifest_path, manifest.to_dict())
        return self.manifest_path


class RunArtifacts:
    def __init__(self, root: str | Path = DEFAULT_STORE_ROOT, run_id: str = "") -> None:
        self.store = StoreLayout(root)
        self.root = self.store.root
        self.run_id = str(run_id)
        if not self.run_id:
            raise ValueError("run_id must be a non-empty string.")
        self.run_dir = self.store.runs_dir / self.run_id

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "run.json"

    @property
    def predictions_dir(self) -> Path:
        return self.run_dir / "predictions"

    @property
    def raw_outputs_dir(self) -> Path:
        return self.run_dir / "raw_outputs"

    @property
    def previews_dir(self) -> Path:
        return self.run_dir / "previews"

    @property
    def reports_dir(self) -> Path:
        return self.run_dir / "reports"

    @property
    def logs_dir(self) -> Path:
        return self.run_dir / "logs"

    @property
    def note_path(self) -> Path:
        return self.run_dir / "note.json"

    def ensure(self) -> None:
        self.store.ensure()
        for path in (
            self.run_dir,
            self.predictions_dir,
            self.raw_outputs_dir,
            self.previews_dir,
            self.reports_dir,
            self.logs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def write_manifest(self, manifest: EvalRunManifest) -> Path:
        manifest.validate()
        self.ensure()
        atomic_write_json(self.manifest_path, manifest.to_dict())
        return self.manifest_path

    def prediction_path(self, image: str) -> Path:
        return self.predictions_dir / prediction_json_relative_path(image)

    def write_prediction(self, prediction: PredictionDocument, *, task: TaskKind) -> Path:
        prediction.validate(task=task)
        path = self.prediction_path(prediction.image)
        atomic_write_json(path, prediction.to_dict(task=task))
        return path


def load_prediction(path: str | Path, *, task: TaskKind | None = None) -> PredictionDocument:
    document = PredictionDocument.from_dict(read_json(Path(path)))
    document.validate(task=task)
    return document
