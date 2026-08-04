from __future__ import annotations

from scripts.tasks.prelabel_line_reconstruction import (
    align_points_to_reconstruction,
    clean_parameters_geometry,
    clean_medium_context_view,
    fuse_heads,
    geometry_metrics,
    gt_standard_bbox,
    globalize_parameters,
    integerize_line_parameters,
    normalize_attribute_parameters,
    proposal_axis_geometry,
    validate_arrow_heads,
    validate_forced_end_arrow,
    validate_points,
    validate_reconstruction,
)


def reconstruction_payload(*, begin: str = "none", end: str = "triangle") -> dict:
    return {
        "type": "line",
        "parameters": {
            "line_type": "straight",
            "line_style": "path",
            "is_single": True,
            "points": [[[100, 100], [900, 900]]],
            "begin_arrow": begin,
            "end_arrow": end,
            "dash_style": "solid",
            "fill_color": "#112233",
        },
    }


def head_result(payload: dict, *, geometry_valid: bool = True) -> dict:
    return {
        "selected": {
            "prediction": payload,
            "contract_valid": True,
            "geometry": {"valid": geometry_valid, "score": 1.0},
        }
    }


def test_clean_context_keeps_exact_bbox_and_valid_qwen_proposal() -> None:
    crop, proposal, _ = clean_medium_context_view(
        (100.0, 120.0, 500.0, 160.0),
        image_width=1000,
        image_height=800,
        sample_id="x",
        seed=42,
    )
    assert crop[0] <= 100 and crop[1] <= 120 and crop[2] >= 500 and crop[3] >= 160
    assert all(0 <= value <= 999 for value in proposal)
    assert proposal[2] > proposal[0] and proposal[3] > proposal[1]


def test_reconstruction_enforces_source_arrow_head_prior() -> None:
    assert not validate_reconstruction(reconstruction_payload(), source_label="arrow")
    errors = validate_reconstruction(
        reconstruction_payload(begin="none", end="none"), source_label="arrow"
    )
    assert "source_prior:arrow_requires_at_least_one_head" in errors


def test_arrow_head_recovery_contract_enforces_hard_prior() -> None:
    assert not validate_arrow_heads({"begin_arrow": "none", "end_arrow": "triangle"})
    assert "source_prior:arrow_requires_at_least_one_head" in validate_arrow_heads(
        {"begin_arrow": "none", "end_arrow": "none"}
    )


def test_forced_end_arrow_contract_excludes_none() -> None:
    assert not validate_forced_end_arrow({"end_arrow": "triangle"})
    assert validate_forced_end_arrow({"end_arrow": "none"})


def test_points_contract_enforces_is_single_relation() -> None:
    payload = {
        "type": "line",
        "parameters": {"is_single": False, "points": [[[0, 0], [999, 999]]]},
    }
    assert "is_single:false_requires_multiple_segments" in validate_points(payload)


def test_attribute_normalization_ignores_bad_geometry_and_normalizes_line_prior() -> None:
    payload = reconstruction_payload(begin="triangle", end="pointy")
    payload["parameters"]["points"] = [[[500, 500], [500, 500]]]
    parameters, errors = normalize_attribute_parameters(payload, source_label="line")
    assert not errors
    assert parameters is not None
    assert "points" not in parameters
    assert parameters["begin_arrow"] == parameters["end_arrow"] == "none"


def test_proposal_axis_geometry_uses_the_dominant_bbox_axis() -> None:
    assert proposal_axis_geometry([100, 300, 900, 500]) == {
        "is_single": True,
        "points": [[[100, 400], [900, 400]]],
    }


def test_geometry_cleanup_deduplicates_and_simplifies_straight_trace() -> None:
    parameters = reconstruction_payload()["parameters"]
    parameters["points"] = [
        [[100, 100], [100, 100], [200, 200], [300, 300], [900, 900]]
    ]
    cleaned, audit = clean_parameters_geometry(
        parameters,
        {"proposal_bbox_2d": [100, 100, 900, 900]},
    )
    assert cleaned["points"] == [[[100, 100], [900, 900]]]
    assert audit["consecutive_duplicates_removed"] == 1
    assert audit["changed"] is True


def test_geometry_cleanup_samples_curved_trace_to_four_points() -> None:
    parameters = reconstruction_payload()["parameters"]
    parameters["line_type"] = "curved"
    parameters["points"] = [[[0, 0], [100, 50], [200, 150], [400, 300], [999, 999]]]
    cleaned, audit = clean_parameters_geometry(
        parameters,
        {"proposal_bbox_2d": [0, 0, 999, 999]},
    )
    assert len(cleaned["points"][0]) == 4
    assert cleaned["points"][0][0] == [0, 0]
    assert cleaned["points"][0][-1] == [999, 999]
    assert audit["final_total_points"] == 4


def test_geometry_cleanup_removes_exact_duplicate_segments() -> None:
    parameters = reconstruction_payload()["parameters"]
    parameters["is_single"] = False
    parameters["points"] = [
        [[100, 100], [900, 900]],
        [[100, 100], [900, 900]],
        [[100, 100], [900, 100]],
    ]
    cleaned, audit = clean_parameters_geometry(
        parameters,
        {"proposal_bbox_2d": [100, 100, 900, 900]},
    )
    assert cleaned["points"] == [
        [[100, 100], [900, 900]],
        [[100, 100], [900, 100]],
    ]
    assert audit["duplicate_segments_removed"] == 1


def test_globalize_removes_duplicates_created_by_coordinate_rounding() -> None:
    parameters = reconstruction_payload()["parameters"]
    parameters["points"] = [[[100, 100], [101, 100], [900, 100]]]
    globalized, audit = globalize_parameters(
        parameters,
        {
            "crop_box": [10, 20, 12, 22],
            "crop_width": 2,
            "crop_height": 2,
            "image_width": 100,
            "image_height": 100,
            "source_bbox": [10.1, 20.1, 11.9, 20.4],
        },
    )
    assert len(globalized["points"][0]) == 2
    assert audit["consecutive_duplicates_removed"] == 1


def test_gt_standard_bbox_uses_covering_integer_coordinates() -> None:
    assert gt_standard_bbox(
        [10.8, 20.2, 30.1, 40.9], image_width=100, image_height=80
    ) == [10, 20, 31, 41]


def test_integerize_line_parameters_removes_new_pixel_duplicates() -> None:
    parameters = reconstruction_payload()["parameters"]
    parameters["points"] = [[[10.1, 20.1], [10.4, 20.4], [30.2, 20.2]]]
    integerized, audit = integerize_line_parameters(
        parameters,
        {
            "image_width": 100,
            "image_height": 80,
            "source_bbox": [10.1, 20.1, 30.2, 20.4],
        },
    )
    assert integerized["points"] == [[[10, 20], [30, 20]]]
    assert audit["consecutive_duplicates_removed"] == 1
    assert list(integerized) == [
        "line_type",
        "line_style",
        "is_single",
        "points",
        "dash_style",
        "begin_arrow",
        "end_arrow",
        "fill_color",
    ]


def test_arrow_points_are_reversed_to_match_reconstruction_endpoints() -> None:
    points, reversed_order = align_points_to_reconstruction(
        [[[900, 900], [100, 100]]],
        [[[100, 100], [900, 900]]],
        source_label="arrow",
    )
    assert reversed_order is True
    assert points == [[[100, 100], [900, 900]]]


def test_fusion_uses_point_head_and_hard_normalizes_source_line_heads() -> None:
    reconstruction = reconstruction_payload(begin="triangle", end="pointy")
    point_payload = {
        "type": "line",
        "parameters": {"is_single": True, "points": [[[110, 120], [880, 890]]]},
    }
    fused = fuse_heads(
        {"source_label": "line"},
        head_result(reconstruction),
        head_result(point_payload),
    )
    assert fused["status"] == "success"
    assert fused["geometry_source"] == "line_context_points"
    assert fused["parameters_crop_qwen"]["points"] == point_payload["parameters"]["points"]
    assert fused["parameters_crop_qwen"]["begin_arrow"] == "none"
    assert fused["parameters_crop_qwen"]["end_arrow"] == "none"


def test_fusion_uses_focused_recovery_when_arrow_attributes_miss_both_heads() -> None:
    reconstruction = reconstruction_payload(begin="none", end="none")
    point_payload = {
        "type": "line",
        "parameters": {"is_single": True, "points": [[[110, 120], [880, 890]]]},
    }
    recovery = {
        "selected": {
            "prediction": {"begin_arrow": "none", "end_arrow": "triangle"},
            "contract_valid": True,
        }
    }
    fused = fuse_heads(
        {"source_label": "arrow"},
        head_result(reconstruction),
        head_result(point_payload),
        recovery,
    )
    assert fused["status"] == "success"
    assert fused["arrow_head_source"] == "focused_arrow_head_recovery"
    assert fused["parameters_crop_qwen"]["begin_arrow"] == "none"
    assert fused["parameters_crop_qwen"]["end_arrow"] == "triangle"


def test_fusion_uses_full_image_recovery_after_crop_recovery_rejects_arrow() -> None:
    reconstruction = reconstruction_payload(begin="none", end="none")
    point_payload = {
        "type": "line",
        "parameters": {"is_single": True, "points": [[[110, 120], [880, 890]]]},
    }
    crop_recovery = {
        "selected": {
            "prediction": {"begin_arrow": "none", "end_arrow": "none"},
            "contract_valid": False,
        }
    }
    full_recovery = {
        "selected": {
            "prediction": {"begin_arrow": "none", "end_arrow": "triangle"},
            "contract_valid": True,
        }
    }
    fused = fuse_heads(
        {"source_label": "arrow"},
        head_result(reconstruction),
        head_result(point_payload),
        crop_recovery,
        full_recovery,
    )
    assert fused["status"] == "success"
    assert fused["arrow_head_source"] == "full_image_arrow_head_recovery"
    assert fused["parameters_crop_qwen"]["end_arrow"] == "triangle"


def test_fusion_uses_forced_path_end_type_as_last_arrow_prior_fallback() -> None:
    reconstruction = reconstruction_payload(begin="none", end="none")
    point_payload = {
        "type": "line",
        "parameters": {"is_single": True, "points": [[[110, 120], [880, 890]]]},
    }
    rejected = {
        "selected": {
            "prediction": {"begin_arrow": "none", "end_arrow": "none"},
            "contract_valid": False,
        }
    }
    forced = {
        "selected": {
            "prediction": {"end_arrow": "line"},
            "contract_valid": True,
        }
    }
    fused = fuse_heads(
        {"source_label": "arrow"},
        head_result(reconstruction),
        head_result(point_payload),
        rejected,
        rejected,
        forced,
    )
    assert fused["status"] == "success"
    assert fused["arrow_head_source"] == "trusted_arrow_prior_forced_path_end_type"
    assert fused["parameters_crop_qwen"]["end_arrow"] == "line"


def test_fusion_salvages_attributes_when_only_reconstruction_geometry_is_invalid() -> None:
    reconstruction_payload_value = reconstruction_payload()
    reconstruction_payload_value["parameters"]["points"] = [[[500, 500], [500, 500]]]
    reconstruction = {
        "selected": {
            "prediction": reconstruction_payload_value,
            "contract_valid": False,
            "geometry": {"valid": True, "score": 1.0},
        },
        "attempts": [
            {
                "variant": "main",
                "prediction": reconstruction_payload_value,
                "contract_valid": False,
            }
        ],
    }
    point_payload = {
        "type": "line",
        "parameters": {"is_single": True, "points": [[[110, 120], [880, 890]]]},
    }
    fused = fuse_heads(
        {"source_label": "line", "proposal_bbox_2d": [100, 100, 900, 900]},
        reconstruction,
        head_result(point_payload),
    )
    assert fused["status"] == "success"
    assert fused["attribute_source"] == "line_context_reconstruction"
    assert fused["parameters_crop_qwen"]["points"] == point_payload["parameters"]["points"]


def test_fusion_marks_bbox_axis_geometry_as_high_risk_last_resort() -> None:
    invalid_points = {
        "selected": {
            "prediction": {
                "type": "line",
                "parameters": {"is_single": True, "points": [[[500, 500], [500, 500]]]},
            },
            "contract_valid": False,
            "geometry": {"valid": True, "score": 1.0},
        }
    }
    fused = fuse_heads(
        {"source_label": "line", "proposal_bbox_2d": [100, 300, 900, 500]},
        head_result(reconstruction_payload(), geometry_valid=False),
        invalid_points,
    )
    assert fused["status"] == "success"
    assert fused["geometry_high_risk_fallback"] is True
    assert fused["parameters_crop_qwen"]["points"] == [[[100, 400], [900, 400]]]


def test_geometry_metrics_rejects_a_distant_distractor() -> None:
    metrics = geometry_metrics([[[800, 800], [900, 900]]], [100, 100, 200, 200])
    assert metrics["valid"] is False
