import importlib.util
import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image


spec = importlib.util.spec_from_file_location(
    "clean_real_raw", Path("scripts/tasks/clean_real_raw_annotations.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_exact_duplicate_not_different_path():
    a = {"type": "line", "bbox": [0, 0, 10, 10], "parameters": {"points": [[[0, 0], [10, 10]]]}}
    b = {"type": "line", "bbox": [0, 0, 10, 10], "parameters": {"points": [[[10, 0], [0, 10]]]}}
    cleaned, issues = module.inspect_annotation({"size": [10, 10], "layout": [a, a, b]})
    assert cleaned["layout"] == [a, b]
    assert sum(i["code"] == "exact_duplicate_removed" for i in issues) == 1
    assert not any(i["level"] == "reject" for i in issues)


@pytest.mark.parametrize("overflow,reject", [(0, False), (3, False), (4, True)])
def test_boundary_clipping(overflow, reject):
    d = {"size": [10, 10], "layout": [{"type": "shape", "bbox": [-overflow, 0, 10, 10]}]}
    cleaned, issues = module.inspect_annotation(d)
    assert any(i["level"] == "reject" for i in issues) == reject
    if not reject:
        assert cleaned["layout"][0]["bbox"] == [0, 0, 10, 10]
        assert not module.inspect_annotation(cleaned)[1]


def test_partial_attributes_and_source_points_preserved():
    d = {
        "size": [20, 20],
        "layout": [
            {"type": "line", "bbox": [0, 0, 20, 20], "parameters": {"points": []}},
            {
                "type": "line",
                "bbox": [0, 0, 20, 20],
                "parameters": {"points": [[[1, 1], [10, 10], [10, 10]]]},
            },
            {
                "type": "shape",
                "bbox": [0, 0, 20, 20],
                "parameters": {"shape_type": "card", "split": [], "subbbox": [[0, 0, 20, 20]]},
            },
        ],
    }
    cleaned, issues = module.inspect_annotation(d)
    assert cleaned == d
    assert not any(i["level"] != "warning" for i in issues)


def test_invalid_path_and_attribute_geometry():
    d = {
        "size": [20, 20],
        "layout": [
            {"type": "line", "bbox": [0, 0, 20, 20], "parameters": {"points": [[[1, 1], [1, 1]]]}},
            {
                "type": "shape",
                "bbox": [0, 0, 20, 20],
                "parameters": {"corners": [{"type": "sharp", "point": [30, 30]}]},
            },
        ],
    }
    _, issues = module.inspect_annotation(d)
    codes = {i["code"] for i in issues if i["level"] == "reject"}
    assert codes == {"degenerate_line_path", "attribute_point_outside_image"}


def test_unused_subbbox_does_not_reject_or_modify_source():
    d = {
        "size": [20, 20],
        "layout": [
            {
                "type": "shape",
                "bbox": [0, 0, 20, 20],
                "parameters": {"shape_type": "card", "subbbox": {"point": [999, -1]}},
            }
        ],
    }
    cleaned, issues = module.inspect_annotation(d)
    assert cleaned == d
    assert not issues


def test_endpoint_conflict_but_not_external_curve_control_point():
    d = {
        "size": [100, 100],
        "layout": [
            {
                "type": "line",
                "bbox": [40, 40, 60, 60],
                "parameters": {"points": [[[40, 40], [0, 0], [100, 100], [60, 60]]]},
            }
        ],
    }
    assert not any(i["level"] == "reject" for i in module.inspect_annotation(d)[1])
    d["layout"][0]["parameters"]["points"][0][0] = [0, 0]
    assert any(i["code"] == "line_endpoint_bbox_conflict" for i in module.inspect_annotation(d)[1])


def test_full_decode_and_duplicate_keys(tmp_path):
    path = tmp_path / "a.json"
    image = tmp_path / "a.png"
    Image.new("RGB", (10, 20)).save(image)
    path.write_text(json.dumps({"size": [10, 20], "layout": []}))
    assert module.audit_file((path, [image]))["issues"] == []
    path.write_text('{"size": [10,20], "size": [10,20], "layout": []}')
    assert module.audit_file((path, [image]))["issues"][0]["code"] == "invalid_json"
    path.write_text(json.dumps({"size": [10, 20], "layout": []}))
    image.write_bytes(b"not an image")
    assert module.audit_file((path, [image]))["issues"][0]["code"] == "image_decode_error"


def test_exif_size_is_not_mislabeled(tmp_path):
    path = tmp_path / "a.json"
    image = tmp_path / "a.jpg"
    im = Image.new("RGB", (10, 20))
    exif = im.getexif()
    exif[274] = 6
    im.save(image, exif=exif)
    path.write_text(json.dumps({"size": [20, 10], "layout": []}))
    issues = module.audit_file((path, [image]))["issues"]
    assert issues == [{"level": "warning", "code": "requires_exif_transpose"}]


def test_apply_backup_quarantine_and_repair(tmp_path, monkeypatch):
    source = tmp_path / "json"
    source.mkdir()
    obj = {"type": "shape", "bbox": [0, 0, 10, 10]}
    good = {"size": [10, 10], "layout": [obj, obj]}
    bad = {"size": [10, 10], "layout": [{"type": "shape", "bbox": [-9, 0, 10, 10]}]}
    records = []
    for name, data in [("good.json", good), ("bad.json", bad)]:
        p = source / name
        p.write_text(json.dumps(data))
        records.append(
            {
                "file": name,
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                "issues": module.inspect_annotation(data)[1],
            }
        )
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "script_sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
                "records": records,
            }
        )
    )
    monkeypatch.setattr(
        "sys.argv", ["clean", "--raw-root", str(tmp_path), "--report", str(report), "--apply"]
    )
    module.main()
    assert [p.name for p in source.iterdir()] == ["good.json"]
    assert json.loads((source / "good.json").read_text())["layout"] == [obj]
    assert (
        json.loads((tmp_path / "json.before_quality_cleaning_v5_10/good.json").read_text()) == good
    )
    assert json.loads((tmp_path / "json.quarantine_quality_cleaning_v5_10/bad.json").read_text()) == bad


def test_bare_shape_removed_but_conflicting_attributes_quarantined():
    bare = {"type": "shape", "bbox": [0, 0, 10, 10]}
    rich = {**bare, "parameters": {"shape_type": "card"}}
    d = {"size": [10, 10], "layout": [bare, rich]}
    cleaned, issues = module.inspect_annotation(d)
    assert cleaned["layout"] == [rich]
    assert any(i["code"] == "bare_shape_duplicate_removed" for i in issues)
    assert not any(i["level"] == "reject" for i in issues)
    d["layout"].append({**bare, "parameters": {"shape_type": "rectangle"}})
    assert any(
        i["code"] == "conflicting_same_bbox_shape_attributes"
        for i in module.inspect_annotation(d)[1]
    )
