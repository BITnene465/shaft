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


def test_train_service_load_run_snapshot_reads_resolved_config_and_log(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    run_store = ShaftRunStore(root_dir=tmp_path / "runs")
    service = ShaftSFTTrainService(run_store=run_store, repo_root=repo_root)
    record = run_store.save_record(
        ShaftRunRecord(
            run_id="snapshot-run",
            algorithm="sft",
            status="succeeded",
            command=["python", "scripts/train.py"],
            config_source_path="base.yaml",
            resolved_config_path=str(run_store.get_resolved_config_path("snapshot-run")),
            log_path=str(run_store.get_log_path("snapshot-run")),
            output_dir="outputs/snapshot",
            return_code=0,
        )
    )
    run_store.get_run_dir(record.run_id).mkdir(parents=True, exist_ok=True)
    run_store.get_resolved_config_path(record.run_id).write_text("experiment:\n  run_id: snapshot-run\n", encoding="utf-8")
    run_store.get_log_path(record.run_id).write_text("hello webui\n", encoding="utf-8")
    finetune_summary_path = repo_root / "outputs" / "snapshot" / "shaft_finetune_summary.json"
    finetune_summary_path.parent.mkdir(parents=True, exist_ok=True)
    finetune_summary_path.write_text(
        '{"mode":"lora","trainable_params":42,"resolved_target_modules":["proj"],"modules_to_save":["lm_head"]}',
        encoding="utf-8",
    )
    optimizer_summary_path = repo_root / "outputs" / "snapshot" / "shaft_optimizer_summary.json"
    optimizer_summary_path.write_text(
        '{"total_trainable_params":42,"group_count":1,"groups":[{"logical_group":"lora_params","decay":true,"lr":0.0005,"weight_decay":0.01,"num_parameters":42,"num_tensors":2,"sample_parameter_names":["base_model.model.fc.lora_A.default.weight"]}]}',
        encoding="utf-8",
    )

    snapshot = service.load_run_snapshot(record.run_id)

    assert snapshot is not None
    assert snapshot["record"].run_id == "snapshot-run"
    assert "run_id: snapshot-run" in snapshot["resolved_config"]
    assert "hello webui" in snapshot["log"]
    assert snapshot["finetune_summary"]["resolved_target_modules"] == ["proj"]
    assert snapshot["optimizer_summary"]["groups"][0]["logical_group"] == "lora_params"


def test_train_service_delete_run_removes_local_run_store_only(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    run_store = ShaftRunStore(root_dir=tmp_path / "runs")
    service = ShaftSFTTrainService(run_store=run_store, repo_root=repo_root)
    output_dir = repo_root / "outputs" / "kept"
    output_dir.mkdir(parents=True)
    record = run_store.save_record(
        ShaftRunRecord(
            run_id="delete-me",
            algorithm="sft",
            status="succeeded",
            command=["python", "scripts/train.py"],
            config_source_path="base.yaml",
            resolved_config_path=str(run_store.get_resolved_config_path("delete-me")),
            log_path=str(run_store.get_log_path("delete-me")),
            output_dir=str(output_dir),
            return_code=0,
        )
    )

    deleted = service.delete_run(record.run_id)

    assert deleted is True
    assert run_store.get_run_dir(record.run_id).exists() is False
    assert output_dir.exists() is True


def test_train_service_delete_run_rejects_active_run(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    run_store = ShaftRunStore(root_dir=tmp_path / "runs")
    service = ShaftSFTTrainService(run_store=run_store, repo_root=repo_root)
    run_store.save_record(
        ShaftRunRecord(
            run_id="active-run",
            algorithm="sft",
            status="running",
            command=["python", "scripts/train.py"],
            config_source_path="base.yaml",
            resolved_config_path=str(run_store.get_resolved_config_path("active-run")),
            log_path=str(run_store.get_log_path("active-run")),
            output_dir="outputs/active",
            pid=12345,
        )
    )
    with patch.object(service, "_is_pid_running", return_value=True):
        try:
            service.delete_run("active-run")
        except ValueError as exc:
            assert "Stop it before deleting" in str(exc)
        else:
            raise AssertionError("Expected deleting an active run to fail")
