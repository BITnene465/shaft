from __future__ import annotations

from pathlib import Path
import subprocess
import sys


SCRIPT = Path("scripts/tasks/build_drawio_shape_from_weak_labels.py").resolve()


def test_weak_label_builder_requires_explicit_local_job_path(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "the following arguments are required: --weak-job-dir" in completed.stderr
    assert not (tmp_path / "data").exists()


def test_weak_label_builder_validates_input_before_cleaning_output(tmp_path: Path) -> None:
    weak_job_dir = tmp_path / "missing-job"
    weak_job_dir.mkdir()
    output_root = tmp_path / "output"
    output_root.mkdir()
    sentinel = output_root / "keep.txt"
    sentinel.write_text("keep\n", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--weak-job-dir",
            str(weak_job_dir),
            "--output-root",
            str(output_root),
            "--clean",
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "weak_labels.json" in completed.stderr
    assert sentinel.read_text(encoding="utf-8") == "keep\n"
