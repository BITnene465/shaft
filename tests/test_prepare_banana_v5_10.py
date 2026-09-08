from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _module():
    path = Path("scripts/tasks/prepare_banana_v5_10.py")
    sys.path.insert(0, str(path.parent.resolve()))
    spec = importlib.util.spec_from_file_location("prepare_banana_v5_10", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_line_inventory_preserves_valid_axis_aligned_path(tmp_path):
    import json
    from PIL import Image

    _module()
    import audit_banana_v5_10_lines as audit

    p = {
        "line_type": "straight",
        "line_style": "path",
        "is_single": True,
        "points": [[[10, 30], [80, 30]]],
        "dash_style": "solid",
        "begin_arrow": "none",
        "end_arrow": "triangle",
        "fill": {"type": "uniform", "color": "#000000"},
        "border": {"type": "none"},
    }
    assert audit.classify(p) == "single_simple"
    (tmp_path / "gt_standard").mkdir()
    (tmp_path / "img").mkdir()
    (tmp_path / "gt_standard/a.json").write_text(
        json.dumps(
            {
                "size": [100, 100],
                "layout": [{"type": "line", "bbox": [10, 30, 80, 30], "parameters": p}],
            }
        )
    )
    with Image.new("RGB", (100, 100)) as image:
        image.save(tmp_path / "img/a.png")
    result = audit.scan((tmp_path, ["a"]))
    assert result["counts"]["valid_groups"]["single_simple"] == 1
    assert result["counts"]["flags"]["zero_extent_bbox_requires_crop_adapter"] == 1
    p["points"].append([[80, 30], [80, 60]])
    assert audit.classify(p) == "multi_path"


def test_line_preview_is_deterministic_and_size_preserving():
    import json
    from PIL import Image

    _module()
    import preview_banana_v5_10_lines as preview

    recipe = json.loads(
        (preview.REPO / "configs/data/preparation/banana_v5_10_line_preview.json").read_text()
    )
    plans = preview.plans((40, 30), "sample", recipe)
    assert plans == preview.plans((40, 30), "sample", recipe)
    assert len(plans) == 6
    with Image.new("RGB", (40, 30), "white") as image:
        for plan in plans.values():
            a = preview.apply_synthetic_realism_augmentation(image, plan)
            b = preview.apply_synthetic_realism_augmentation(image, plan)
            assert a.size == image.size
            assert a.tobytes() == b.tobytes()
            a.close()
            b.close()


def test_line_single_pixel_policy_and_strict_quantization():
    import json
    import pytest

    m = _module()
    policy = json.loads((m.REPO / "configs/data/preparation/banana_v5_10_line.json").read_text())[
        "pixel_policy"
    ]
    qualities = set()
    for index in range(2000):
        plan = m.builder._sample_single_pixel_plan(
            policy=policy,
            sample_id=str(index),
            seed=465,
            image_size=(200, 200),
            target_size=(100, 100),
        )
        assert len(plan["operations"]) == 1
        op = plan["operations"][0]
        if op["name"] == "jpeg_compression":
            qualities.add(op["quality"])
            assert op["subsampling"] == 0
        else:
            assert op["name"] == "gaussian_noise" and 0.5 <= op["sigma_255"] <= 1.5
    assert qualities == set(range(40, 91))
    assert not m.builder._sample_single_pixel_plan(
        policy=policy, sample_id="tiny", seed=465, image_size=(200, 200), target_size=(1, 100)
    )["operations"]
    with pytest.raises(ValueError, match="collision"):
        m.builder._line_parameters(
            {"points": [[[0, 0], [0.1, 0.1], [500, 500], [999, 999]]]},
            left=0,
            top=0,
            crop_width=1000,
            crop_height=1000,
            strict=True,
        )


def test_line_production_small_build(tmp_path):
    import json
    import subprocess
    from PIL import Image

    m = _module()
    root = tmp_path / "source"
    (root / "gt_standard").mkdir(parents=True)
    (root / "img").mkdir()
    (root / "train.txt").write_text("a\n")
    (root / "val.txt").write_text("b\n")
    excluded = tmp_path / "excluded.txt"
    excluded.write_text("test\n")
    p = {
        "line_type": "straight",
        "line_style": "path",
        "is_single": True,
        "points": [[[10, 10], [70, 70]]],
        "dash_style": "solid",
        "begin_arrow": "none",
        "end_arrow": "triangle",
        "fill": {"type": "uniform", "color": "#111111"},
        "border": {"type": "none"},
    }
    (root / "gt_standard/a.json").write_text(
        json.dumps(
            {
                "size": [100, 100],
                "layout": [{"type": "line", "bbox": [10, 10, 70, 70], "parameters": p}],
            }
        )
    )
    with Image.new("RGB", (100, 100), "white") as image:
        image.save(root / "img/a.png")
    output = tmp_path / "output"
    subprocess.run(
        [
            sys.executable,
            str(m.REPO / "scripts/tasks/prepare_banana_v5_10_lines.py"),
            "--synthetic-root",
            str(root),
            "--exclude-manifests",
            str(excluded),
            "--output-root",
            str(output),
            "--work-root",
            str(tmp_path / "work"),
            "--workers",
            "2",
        ],
        check=True,
        timeout=180,
    )
    task = output / "line_context_reconstruction"
    assert json.loads((task / "reports/reproduction_result.json").read_text())["rows"] == 1
    assert json.loads((task / "reports/rejected_views.json").read_text()) == []
    for formulation in ("appearance", "points", "reconstruction"):
        row = json.loads((task / f"sft/formulations/{formulation}/train.jsonl").read_text())
        assert json.loads(row["target_text"])["type"] == "line"


def test_quotas_are_order_independent_and_capacity_bounded():
    m = _module()
    capacities = {"a": 100, "b": 2, "c": 37}
    result = m.quotas(capacities, 40)
    assert result == m.quotas(dict(reversed(list(capacities.items()))), 40)
    assert sum(result.values()) == 40
    assert all(0 <= result[k] <= v for k, v in capacities.items())


def test_card_side_layout_and_multiregion_protected():
    m = _module()
    corners = [{"type": "sharp", "point": p} for p in ([50, 0], [50, 100])]
    p = {
        "fill": [{"color": "#FFFFFF"}, {"color": "#EEEEEE"}],
        "splits": [{"split_corners": corners}],
        "corners": [],
    }
    protected, stratum = m.card_stratum(p, (0, 0, 100, 100))
    assert protected
    assert "side_by_side" in stratum
    p["fill"].append({"color": "#CCCCCC"})
    assert m.card_stratum(p, (0, 0, 100, 100))[0]


def test_source_dataset_identity_does_not_depend_on_directory():
    m = _module()
    from PIL import Image

    b = m.builder
    selection = b.Selection(
        "a__shape_0000", "a", 0, (10, 10, 40, 40), "img/a.png", "gt_standard/a.json"
    )
    outputs = []
    for root in (Path("/machine_a/v10"), Path("/machine_b/renamed")):
        spec = b.TaskSpec(
            "shape_context_reconstruction",
            "shape",
            Path("selection"),
            root,
            Path("pool"),
            "synthetic",
            source_dataset_id="frozen-v10",
        )
        config = b.WorkerConfig(
            spec,
            Path("unused"),
            "pool",
            None,
            465,
            4,
            60,
            1,
            ("appearance", "geometry", "reconstruction"),
            False,
        )
        with Image.new("RGB", (100, 100), "white") as image:
            outputs.append(
                b._build_row(
                    config=config,
                    selection=selection,
                    source_image=image,
                    source_instance_index=0,
                    source_bbox=selection.source_bbox,
                    source_parameters={"shape_type": "other"},
                    source_layout=None,
                    image_width=100,
                    image_height=100,
                )
            )
    assert outputs[0] == outputs[1]
    import json

    targets = [json.loads(row)["target_text"] for _, row in outputs[0][1]]
    assert len(set(targets)) == 1


def test_end_to_end_content_is_worker_and_path_independent(tmp_path):
    import json
    import os
    import subprocess
    from PIL import Image

    m = _module()
    recipe = json.loads((m.REPO / "configs/data/preparation/banana_v5_10.json").read_text())
    recipe["shape"]["rectangle_target"] = 2
    recipe["shape"]["common_card_target"] = 2
    config = tmp_path / "recipe.json"
    config.write_text(json.dumps(recipe))
    excluded = tmp_path / "excluded.txt"
    excluded.write_text("heldout\n")
    checksums = []
    for workers in (1, 2):
        root = tmp_path / f"machine-{workers}" / "renamed-source"
        (root / "gt_standard").mkdir(parents=True)
        (root / "img").mkdir()
        (root / "train.txt").write_text("b\na\nheldout\n")
        (root / "val.txt").write_text("c\n")
        for stem in ("a", "b"):
            (root / "gt_standard" / f"{stem}.json").write_text(
                json.dumps(
                    {
                        "size": [100, 100],
                        "layout": [
                            {
                                "type": "shape",
                                "bbox": [10, 10, 70, 70],
                                "parameters": {"shape_type": "other"},
                            }
                        ],
                    }
                )
            )
            with Image.new("RGB", (100, 100), "white") as image:
                image.save(root / "img" / f"{stem}.png")
        output = tmp_path / f"output-{workers}"
        subprocess.run(
            [
                sys.executable,
                str(m.REPO / "scripts/tasks/prepare_banana_v5_10.py"),
                "--recipe",
                str(config),
                "--synthetic-root",
                str(root),
                "--exclude-manifests",
                str(excluded),
                "--output-root",
                str(output),
                "--work-root",
                str(tmp_path / f"work-{workers}"),
                "--workers",
                str(workers),
            ],
            check=True,
            env={**os.environ, "PYTHONHASHSEED": str(workers)},
            timeout=180,
        )
        report = output / m.TASK / "reports/content_checksums.json"
        checksums.append(json.loads(report.read_text()))
    assert checksums[0] == checksums[1]
