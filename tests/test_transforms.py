from __future__ import annotations

from shaft.data import SFTRecord, build_offline_pipeline, build_online_pipeline


def test_offline_dedup() -> None:
    records = [
        SFTRecord(image_path="/tmp/a.png", target_text="{}", dataset_id="d"),
        SFTRecord(image_path="/tmp/a.png", target_text="{}", dataset_id="d"),
        SFTRecord(image_path="/tmp/a.png", target_text='{"x":1}', dataset_id="d"),
    ]
    pipeline = build_offline_pipeline(["dedup_image_target"])
    out = pipeline(records)
    assert len(out) == 2


def test_online_identity() -> None:
    pipeline = build_online_pipeline(["identity"])
    sample = {"x": 1}
    out = pipeline(sample)
    assert out["x"] == 1
