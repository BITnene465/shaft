from __future__ import annotations

from pathlib import Path

from shaft.webui.controller import ShaftSFTWebUIController
from shaft.webui.services import ShaftRunStore, ShaftSFTTrainService, ShaftWebUIConfigService
from shaft.webui.types import ShaftRunRecord


def test_webui_controller_initial_view_does_not_select_current_run(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    run_store = ShaftRunStore(root_dir=tmp_path / "runs")
    run_store.save_record(
        ShaftRunRecord(
            run_id="existing-run",
            algorithm="sft",
            status="succeeded",
            command=["python", "scripts/train.py"],
            config_source_path="base.yaml",
            resolved_config_path="resolved.yaml",
            log_path="train.log",
            output_dir="outputs/existing",
            return_code=0,
        )
    )
    controller = ShaftSFTWebUIController(
        config_service=ShaftWebUIConfigService(),
        train_service=ShaftSFTTrainService(run_store=run_store, repo_root=repo_root),
    )

    state = controller.build_initial_view("configs/train/train_sft_4b.yaml", "experiment:\n  name: demo\n", "<div>ok</div>")

    assert state["selected_run"] == "existing-run"
    assert state["current_run_id"] == ""
    assert state["runs"][0]["run_id"] == "existing-run"


def test_webui_controller_load_run_returns_snapshot(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    run_store = ShaftRunStore(root_dir=tmp_path / "runs")
    train_service = ShaftSFTTrainService(run_store=run_store, repo_root=repo_root)
    controller = ShaftSFTWebUIController(
        config_service=ShaftWebUIConfigService(),
        train_service=train_service,
    )
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

    payload = controller.load_run("snapshot-run")

    assert payload["ok"] is True
    assert payload["current_run_id"] == "snapshot-run"
    assert "snapshot-run" in payload["status_html"]
    assert "run_id: snapshot-run" in payload["resolved_yaml"]
    assert "hello webui" in payload["log_text"]
    assert payload["runs"][0]["run_id"] == "snapshot-run"
    assert payload["selected_run"] == "snapshot-run"


def test_webui_controller_delete_run_clears_current_snapshot(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    run_store = ShaftRunStore(root_dir=tmp_path / "runs")
    train_service = ShaftSFTTrainService(run_store=run_store, repo_root=repo_root)
    controller = ShaftSFTWebUIController(
        config_service=ShaftWebUIConfigService(),
        train_service=train_service,
    )
    record = run_store.save_record(
        ShaftRunRecord(
            run_id="delete-run",
            algorithm="sft",
            status="succeeded",
            command=["python", "scripts/train.py"],
            config_source_path="base.yaml",
            resolved_config_path=str(run_store.get_resolved_config_path("delete-run")),
            log_path=str(run_store.get_log_path("delete-run")),
            output_dir="outputs/delete-run",
            return_code=0,
        )
    )
    run_store.get_run_dir(record.run_id).mkdir(parents=True, exist_ok=True)
    run_store.get_resolved_config_path(record.run_id).write_text("experiment:\n  run_id: delete-run\n", encoding="utf-8")
    run_store.get_log_path(record.run_id).write_text("hello delete\n", encoding="utf-8")

    payload = controller.delete_run("delete-run", "delete-run")

    assert payload["ok"] is True
    assert payload["current_run_id"] == ""
    assert payload["resolved_yaml"] == ""
    assert payload["log_text"] == ""
    assert payload["runs"] == []
