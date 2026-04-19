from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from shaft.webui import create_app


def test_create_webui_app_smoke(tmp_path: Path) -> None:
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        """
algorithm:
  name: sft
data:
  datasets:
    - dataset_name: ds
      train_path: train.jsonl
      val_path: val.jsonl
train:
  report_to: ["none"]
eval:
  enabled: true
""",
        encoding="utf-8",
    )
    app = create_app(default_config_path=str(config_path))
    client = TestClient(app)

    response = client.get("/sft")

    assert response.status_code == 200
    assert "SFT Research Console" in response.text
    assert "Resolved Runtime Config" in response.text
    assert "Freeze Configuration" in response.text
    assert 'id="freeze-summary-html"' in response.text
    assert 'id="optimizer-summary-html"' in response.text
    assert "/static/webui.css" in response.text
    assert "/static/webui.js" in response.text
    assert "/static/logo.png" in response.text
    assert 'href="/rlhf/dpo"' in response.text
    assert 'href="/rlhf/ppo"' in response.text
    assert 'href="/rlhf/grpo"' in response.text


def test_webui_root_redirects_to_sft(tmp_path: Path) -> None:
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        """
algorithm:
  name: sft
data:
  datasets:
    - dataset_name: ds
      train_path: train.jsonl
      val_path: val.jsonl
train:
  report_to: ["none"]
eval:
  enabled: true
""",
        encoding="utf-8",
    )
    app = create_app(default_config_path=str(config_path))
    client = TestClient(app)

    response = client.get("/", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == "/sft"


def test_webui_rlhf_placeholder_route_smoke(tmp_path: Path) -> None:
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        """
algorithm:
  name: sft
data:
  datasets:
    - dataset_name: ds
      train_path: train.jsonl
      val_path: val.jsonl
train:
  report_to: ["none"]
eval:
  enabled: true
""",
        encoding="utf-8",
    )
    app = create_app(default_config_path=str(config_path))
    client = TestClient(app)

    response = client.get("/rlhf/dpo")

    assert response.status_code == 200
    assert "DPO Console" in response.text
    assert "Navigation shell is ready" in response.text
    assert "/static/logo.png" in response.text


def test_webui_favicon_route_redirects_to_local_svg(tmp_path: Path) -> None:
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        """
algorithm:
  name: sft
data:
  datasets:
    - dataset_name: ds
      train_path: train.jsonl
      val_path: val.jsonl
train:
  report_to: ["none"]
eval:
  enabled: true
""",
        encoding="utf-8",
    )
    app = create_app(default_config_path=str(config_path))
    client = TestClient(app)

    response = client.get("/favicon.ico", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == "/static/favicon.svg"
