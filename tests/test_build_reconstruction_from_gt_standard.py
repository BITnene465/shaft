from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    script_path = Path("scripts/tasks/build_reconstruction_from_gt_standard.py").resolve()
    spec = importlib.util.spec_from_file_location(
        "build_reconstruction_from_gt_standard",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_balanced_selection_retains_rare_shapes_and_curved_shape_lines() -> None:
    module = _load_module()
    candidates = []
    instance_index = 0
    rare_ids = set()
    for shape_type in sorted(module.SHAPE_RARE_TYPES):
        candidate = module.Candidate(
            stem="shape",
            instance_index=instance_index,
            task="shape_reconstruction",
            macro=shape_type,
            stratum=f"{shape_type}|uniform|solid|none|missing",
        )
        candidates.append(candidate)
        rare_ids.add((candidate.stem, candidate.instance_index))
        instance_index += 1
    for shape_type in module.SHAPE_HEAD_WEIGHTS_V1:
        for offset in range(20):
            candidates.append(
                module.Candidate(
                    stem=f"shape_{shape_type}",
                    instance_index=offset,
                    task="shape_reconstruction",
                    macro=shape_type,
                    stratum=f"{shape_type}|uniform|solid|none|missing",
                )
            )

    curved_shape_ids = set()
    for macro, count in {
        "curved_shape": 5,
        "curved_path": 30,
        "straight_shape": 30,
        "straight_path_multi": 30,
        "straight_path_single": 60,
    }.items():
        for offset in range(count):
            candidate = module.Candidate(
                stem=f"line_{macro}",
                instance_index=offset,
                task="line_reconstruction",
                macro=macro,
                stratum="none|line|solid|1|solid|missing|missing",
            )
            candidates.append(candidate)
            if macro == "curved_shape":
                curved_shape_ids.add((candidate.stem, candidate.instance_index))

    selected, counts = module._select_balanced_candidates(
        candidates,
        shape_target=30,
        line_target=50,
        seed=42,
    )

    selected_ids = {(candidate.stem, candidate.instance_index) for candidate in selected}
    assert rare_ids <= selected_ids
    assert curved_shape_ids <= selected_ids
    assert counts["selected_shape_reconstruction"] == 30
    assert counts["selected_line_reconstruction"] == 50
    assert counts["available_shape_type_rectangle"] == 20
    assert counts["available_line_macro_curved_shape"] == 5


def test_multiscale_padding_policy_is_deterministic_and_matches_buckets() -> None:
    module = _load_module()
    config = module.BuildConfig(
        dataset_root=Path("dataset"),
        output_root=Path("output"),
        split="train",
        selected_tasks=frozenset({"shape_reconstruction"}),
        padding_min=0.1,
        padding_max=0.2,
        seed=42,
        min_crop_size=4,
        max_aspect_ratio=60.0,
        skip_oob_bbox=False,
        multi_scale=True,
        shape_low_resolution_ratio=0.2,
        line_low_resolution_ratio=0.15,
        include_visual_other_negatives=False,
        prompt_info={},
    )

    first = module._padding_policy(
        config,
        task="shape_reconstruction",
        stem="sample",
        instance_index=3,
    )
    second = module._padding_policy(
        config,
        task="shape_reconstruction",
        stem="sample",
        instance_index=3,
    )
    assert first == second

    bucket, padding = first
    bounds = {
        "tight": (0.08, 0.15),
        "medium": (0.15, 0.25),
        "context": (0.25, 0.40),
    }
    minimum, maximum = bounds[bucket]
    assert minimum <= padding <= maximum


def test_low_resolution_resize_only_downsamples() -> None:
    module = _load_module()
    config = module.BuildConfig(
        dataset_root=Path("dataset"),
        output_root=Path("output"),
        split="train",
        selected_tasks=frozenset({"shape_reconstruction"}),
        padding_min=0.1,
        padding_max=0.2,
        seed=42,
        min_crop_size=4,
        max_aspect_ratio=60.0,
        skip_oob_bbox=False,
        multi_scale=True,
        shape_low_resolution_ratio=1.0,
        line_low_resolution_ratio=1.0,
        include_visual_other_negatives=False,
        prompt_info={},
    )

    resized = module._low_resolution_target(
        config,
        task="shape_reconstruction",
        stem="sample",
        instance_index=0,
        width=400,
        height=200,
    )
    assert resized is not None
    assert resized[0] < 400
    assert resized[1] < 200
    assert resized[0] >= 4
    assert resized[1] >= 4


def test_visual_instances_can_be_mapped_to_shape_other_candidates() -> None:
    module = _load_module()
    config = module.BuildConfig(
        dataset_root=Path("dataset"),
        output_root=Path("output"),
        split="train",
        selected_tasks=frozenset({"shape_reconstruction"}),
        padding_min=0.1,
        padding_max=0.2,
        seed=42,
        min_crop_size=4,
        max_aspect_ratio=60.0,
        skip_oob_bbox=False,
        multi_scale=True,
        shape_low_resolution_ratio=0.2,
        line_low_resolution_ratio=0.15,
        include_visual_other_negatives=True,
        prompt_info={},
    )
    candidate = module._candidate_for_instance(
        stem="sample",
        instance_index=2,
        instance={
            "type": "icon",
            "bbox": [10, 10, 90, 90],
            "parameters": None,
        },
        image_width=100,
        image_height=100,
        config=config,
    )

    assert candidate is not None
    assert candidate.task == "shape_reconstruction"
    assert candidate.macro == "icon_as_other"
    assert candidate.source_label == "icon"
