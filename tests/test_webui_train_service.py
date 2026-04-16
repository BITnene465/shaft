from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import yaml

from shaft.config import RuntimeConfig
from shaft.webui.services import ShaftRunStore, ShaftSFTTrainService
from shaft.webui.types import ShaftRunRecord


class _FakeProcess:
    def __init__(self, command, **kwargs):
        self.command = command
        self.kwargs = kwargs
        self.pid = 4321
        self.return_code = None

    def poll(self):
        return self.return_code

    def wait(self, timeout=None):
        _ = timeout
        return self.return_code if self.return_code is not None else 0

    def terminate(self):
        self.return_code = -15


def test_train_service_start_run_persists_record_and_config(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "scripts").mkdir()
    (repo_root / "scripts" / "train.py").write_text("print('stub')\n", encoding="utf-8")
    source_config = repo_root / "base.yaml"
    source_config.write_text("experiment:\n  name: base\n", encoding="utf-8")
    run_store = ShaftRunStore(root_dir=tmp_path / "runs")
    service = ShaftSFTTrainService(run_store=run_store, repo_root=repo_root)
    config = RuntimeConfig()
    config.experiment.output_dir = "outputs/demo"
    resolved_yaml = "experiment:\n  name: demo\n"
    spawned: dict[str, _FakeProcess] = {}

    def _fake_popen(command, **kwargs):
        process = _FakeProcess(command, **kwargs)
        spawned["proc"] = process
        return process

    with patch("shaft.webui.services.train_service.subprocess.Popen", side_effect=_fake_popen):
        record = service.start_run(
            config_source_path=source_config,
            resolved_yaml_text=resolved_yaml,
            config=config,
        )

    assert record.status == "running"
    assert record.pid == 4321
    assert record.command[:3] == [record.command[0], "scripts/train.py", "sft"]
    saved_config = yaml.safe_load(Path(record.resolved_config_path).read_text(encoding="utf-8"))
    assert saved_config["experiment"]["run_id"] == record.run_id
    assert source_config.read_text(encoding="utf-8") == "experiment:\n  name: base\n"
    assert run_store.load_record(record.run_id) is not None
    assert spawned["proc"].kwargs["cwd"] == repo_root


def test_train_service_refresh_and_stop_update_status(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "scripts").mkdir()
    (repo_root / "scripts" / "train.py").write_text("print('stub')\n", encoding="utf-8")
    run_store = ShaftRunStore(root_dir=tmp_path / "runs")
    service = ShaftSFTTrainService(run_store=run_store, repo_root=repo_root)
    config = RuntimeConfig()
    config.experiment.output_dir = "outputs/demo"

    process = _FakeProcess([])
    with patch("shaft.webui.services.train_service.subprocess.Popen", return_value=process):
        record = service.start_run(
            config_source_path=repo_root / "base.yaml",
            resolved_yaml_text="experiment:\n  name: demo\n",
            config=config,
        )

    process.return_code = 0
    succeeded = service.refresh_run(record.run_id)
    assert succeeded is not None
    assert succeeded.status == "succeeded"
    assert succeeded.return_code == 0

    config_2 = RuntimeConfig()
    config_2.experiment.run_id = "manual-stop"
    config_2.experiment.output_dir = "outputs/demo-stop"
    process_2 = _FakeProcess([])
    with patch("shaft.webui.services.train_service.subprocess.Popen", return_value=process_2):
        record_2 = service.start_run(
            config_source_path=repo_root / "base2.yaml",
            resolved_yaml_text="experiment:\n  name: demo\n",
            config=config_2,
        )

    with patch("shaft.webui.services.train_service.os.killpg") as mocked_killpg:
        stopped = service.stop_run(record_2.run_id)
    assert stopped is not None
    assert stopped.status == "stopped"
    assert stopped.return_code == -15
    mocked_killpg.assert_called_once()


def test_train_service_marks_missing_pid_run_as_failed(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    run_store = ShaftRunStore(root_dir=tmp_path / "runs")
    service = ShaftSFTTrainService(run_store=run_store, repo_root=repo_root)
    record = run_store.save_record(
        ShaftRunRecord(
            run_id="stale-run",
            algorithm="sft",
            status="running",
            command=["python", "scripts/train.py"],
            config_source_path="base.yaml",
            resolved_config_path="resolved.yaml",
            log_path="train.log",
            output_dir="outputs/stale",
            pid=98765,
        )
    )
    with patch.object(service, "_is_pid_running", return_value=False):
        refreshed = service.refresh_run(record.run_id)
    assert refreshed is not None
    assert refreshed.status == "failed"
    assert refreshed.return_code == -1
