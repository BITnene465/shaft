from copy import deepcopy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/tasks"))
import real_shape_contract as C


def rectangle():
    return {
        "shape_type": "rectangle",
        "border": {"type": "none", "color": "", "style": ""},
        "fill": {"type": "uniform", "color": "#FFFFFF"},
        "effect": {"type": "none"},
        "corners": [
            {"type": "sharp", "point": p} for p in ([10, 10], [10, 90], [90, 90], [90, 10])
        ],
        "body_type": "",
        "body_bbox": [],
        "body_corners": [],
        "tail": {"points": []},
    }


def test_complete_projection_is_idempotent_and_does_not_modify_raw():
    source = rectangle()
    before = deepcopy(source)
    result = C.normalize(source, 100, 100)
    assert source == before
    assert result == C.normalize(result, 100, 100)
    assert result["border"] == {"type": "none"}
    assert "body_bbox" not in result
    assert [c["point"] for c in result["corners"]] == [[10, 10], [90, 10], [90, 90], [10, 90]]


def test_missing_geometry_and_nonstandard_split_are_rejected():
    source = rectangle()
    source["corners"] = []
    with pytest.raises(ValueError):
        C.normalize(source, 100, 100)
    source = rectangle()
    source.update(
        shape_type="card",
        fill=[source["fill"], source["fill"]],
        splits=[{"type": "none", "split_corners": [source["corners"][0]]}],
    )
    with pytest.raises(ValueError, match="split_corners"):
        C.normalize(source, 100, 100)


def test_round_reversal_preserves_points_and_swaps_start_end():
    corners = [
        {"type": "round", "start": a, "mid": b, "end": c}
        for a, b, c in (
            ([20, 10], [12, 12], [10, 20]),
            ([10, 80], [12, 88], [20, 90]),
            ([80, 90], [88, 88], [90, 80]),
            ([90, 20], [88, 12], [80, 10]),
        )
    ]
    result = C.canonical_corners(corners)
    assert result[0] == {"type": "round", "start": [10, 20], "mid": [12, 12], "end": [20, 10]}
    assert C.canonical_corners(result) == result
    assert sorted(C.corner_points(corners)) == sorted(C.corner_points(result))


def test_invalid_geometry_is_not_repaired():
    for points in ([[10, 10], [90, 90], [10, 90], [90, 10]], [[10, 10], [20, 10], [30, 10]]):
        with pytest.raises(ValueError):
            C.canonical_corners([{"type": "sharp", "point": p} for p in points])
    source = rectangle()
    source["corners"][0]["point"] = [-1, 10]
    with pytest.raises(ValueError):
        C.normalize(source, 100, 100)


def test_other_and_oval_do_not_require_corners():
    assert C.normalize({"shape_type": "other"}, 100, 100) == {"shape_type": "other"}
    source = rectangle()
    source.update(shape_type="oval", corners=[])
    assert "corners" not in C.normalize(source, 100, 100)


def test_real_source_builds_three_aligned_formulations_and_checks_hashes(tmp_path):
    import json
    from PIL import Image
    import prepare_banana_v5_10_real_shape as R

    (tmp_path / "json").mkdir()
    (tmp_path / "images").mkdir()
    doc = {
        "size": [100, 100],
        "layout": [{"type": "shape", "bbox": [10, 10, 90, 90], "parameters": rectangle()}],
    }
    path = tmp_path / "json/a.json"
    path.write_text(json.dumps(doc))
    Image.new("RGB", (100, 100), "white").save(tmp_path / "images/a.png")
    source = {
        "id": "a",
        "json": "a.json",
        "image": "a.png",
        "json_sha256": R.S.sha(path),
        "sha256": R.S.sha(tmp_path / "images/a.png"),
    }
    selected, rejected, _ = R.select_source((tmp_path, source))
    assert len(selected) == 1 and not rejected
    stage = tmp_path / "output"
    stage.mkdir()
    prompt = R.S.REPO / "configs/prompts/pools/shape_context_reconstruction.v5.8.yaml"
    pool_id, schema, formulations = R.B._prompt_contract(prompt)
    spec = R.B.TaskSpec(
        "shape_context_reconstruction_real",
        "shape",
        tmp_path / "selection.jsonl",
        tmp_path,
        prompt,
        "real",
        source_dataset_id="fixture",
    )
    config = R.B.WorkerConfig(spec, stage, pool_id, schema, 465, 32, 60, 1, formulations)
    built, rejected = R.build_source((tmp_path, source, selected, config))
    assert len(built) == 1 and not rejected
    row = json.loads(built[0][0])
    assert row["extra"]["pixel_augmentation"]["profile"] == "none"
    assert [f for f, _ in built[0][1]] == ["appearance", "geometry", "reconstruction"]
    for f, text in built[0][1]:
        sft = json.loads(text)
        assert json.loads(sft["target_text"])["parameters"] == R.B._formulation_parameters(
            "shape", f, row["instances"][0]["parameters"]
        )
        assert sft["system_prompt"] == sft["user_prompt"] == ""
    (stage / "structured").mkdir()
    recipe = json.loads(
        (R.S.REPO / "configs/data/preparation/banana_v5_10_real_shape.json").read_text()
    )
    clean_path = stage / "structured" / row["image_path"]
    clean_hash = R.S.sha(clean_path)
    twin = R.jpeg_twin((stage, *built[0], recipe))
    assert twin == R.jpeg_twin((stage, *built[0], recipe))
    twin_row = json.loads(twin[0])
    assert twin_row["instances"] == row["instances"]
    assert R.S.sha(clean_path) == clean_hash
    for (_, original), (_, augmented) in zip(built[0][1], twin[1]):
        assert json.loads(original)["target_text"] == json.loads(augmented)["target_text"]
    path.write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        R.select_source((tmp_path, source))
