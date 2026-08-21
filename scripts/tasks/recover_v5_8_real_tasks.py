#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable
import uuid

from PIL import Image

from shaft.data import load_prompt_source_pool


IMAGE_TYPES = frozenset(
    {
        "chart",
        "diagram",
        "document",
        "illustration",
        "infographic",
        "map",
        "medical",
        "microscopy",
        "other",
        "photo",
        "rendering",
        "screenshot",
        "table",
    }
)
RECOVERY_SNAPSHOT_ID = "banana-v5.8-reviewed-real-recovery-v1"
RAW_IMPORT_NAME = "banana_v5_3_replay_20260722"
CANONICAL_VLM_TEST_ROWS = 175
CANONICAL_VLM_TEST_SHA256 = (
    "4bf353ef8034de8e616b7c613162dc8010e163bbda582f9be5688123b8fb50d8"
)


@dataclass(frozen=True)
class MediaJob:
    source: Path
    target: Path
    width: int
    height: int


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _read_test_ids(paths: Iterable[Path]) -> set[str]:
    ids: set[str] = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        items = payload.get("items")
        if not isinstance(items, list):
            raise ValueError(f"Invalid test manifest: {path}")
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                raise ValueError(f"Invalid test manifest item: {path}:{index}")
            sample_id = str(
                item.get("id") or Path(str(item.get("image_path") or "")).stem
            ).strip()
            if not sample_id:
                raise ValueError(f"Missing test identity: {path}:{index}")
            ids.add(sample_id)
    return ids


def _validate_canonical_test_manifest(
    path: Path,
    *,
    expected_sha256: str,
    historical_manifests: list[Path],
) -> tuple[set[str], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("items")
    if not isinstance(items, list) or len(items) != CANONICAL_VLM_TEST_ROWS:
        raise ValueError(
            "Canonical vlm test manifest must contain exactly "
            f"{CANONICAL_VLM_TEST_ROWS} items: {path}"
        )
    ids = _read_test_ids([path])
    if len(ids) != CANONICAL_VLM_TEST_ROWS:
        raise ValueError(f"Canonical vlm test manifest ids must be unique: {path}")
    digest = _sha256(path)
    if digest != expected_sha256:
        raise ValueError(
            f"Canonical vlm test manifest hash mismatch: {digest} != {expected_sha256}"
        )
    if not any(_sha256(manifest) == digest for manifest in historical_manifests):
        raise ValueError(
            "Historical test manifests do not contain the exact canonical vlm test snapshot."
        )
    return ids, digest


def _load_background_annotations(path: Path) -> dict[str, dict[str, Any]]:
    annotations: dict[str, dict[str, Any]] = {}
    image_names: set[str] = set()
    for line_no, line in enumerate(path.open(encoding="utf-8"), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        sample_id = str(row.get("id") or "").strip()
        image_name = Path(str(row.get("image_path") or "")).name
        background = row.get("background")
        level = row.get("background_level")
        if not sample_id or not image_name:
            raise ValueError(f"Missing background id/image_path: {path}:{line_no}")
        if not isinstance(background, bool):
            raise ValueError(f"Invalid background boolean: {path}:{line_no}")
        if level not in {0, 1, 4} or background != (level == 4):
            raise ValueError(f"Inconsistent reviewed background level: {path}:{line_no}")
        if sample_id in annotations or image_name in image_names:
            raise ValueError(f"Duplicate background identity: {path}:{line_no}")
        annotations[sample_id] = row
        image_names.add(image_name)
    if not annotations:
        raise ValueError(f"Background annotation source is empty: {path}")
    return annotations


def _task_media_path(task_root: Path, image_path: str) -> Path:
    parts = Path(image_path).parts
    if len(parts) < 3 or parts[0] != ".." or parts[1] != "images":
        raise ValueError(f"Expected task-local ../images path, got {image_path!r}")
    return task_root.joinpath(*parts[1:])


def _formulation_media_path(image_path: str) -> str:
    parts = Path(image_path).parts
    if len(parts) < 3 or parts[0] != ".." or parts[1] != "images":
        raise ValueError(f"Expected task-local ../images path, got {image_path!r}")
    return Path("..", "..", "..", *parts[1:]).as_posix()


def _link_or_copy(job: MediaJob) -> str:
    if not job.source.is_file():
        raise FileNotFoundError(job.source)
    job.target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(job.source, job.target)
        return "hardlink"
    except OSError:
        shutil.copy2(job.source, job.target)
        return "copy"


def _verify_media(job: MediaJob) -> str | None:
    try:
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(job.target) as image:
            image.load()
            size = image.size
        if size != (job.width, job.height):
            return f"size_mismatch:{job.target}:{size}!={(job.width, job.height)}"
    except Exception as exc:  # noqa: BLE001 - report every corrupt media input together
        return f"decode_error:{job.target}:{type(exc).__name__}:{exc}"
    return None


def _materialize_and_verify_media(
    jobs: list[MediaJob],
    *,
    workers: int,
) -> tuple[Counter[str], list[str]]:
    if len({job.target for job in jobs}) != len(jobs):
        raise ValueError("Recovery media targets must be unique.")
    counts: Counter[str] = Counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        counts.update(executor.map(_link_or_copy, jobs, chunksize=64))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        errors = [
            error
            for error in executor.map(_verify_media, jobs, chunksize=32)
            if error is not None
        ]
    counts["verified"] = len(jobs) - len(errors)
    counts["errors"] = len(errors)
    return counts, errors


def _validate_prompt_contracts(background_pool_path: Path, image_pool_path: Path) -> None:
    background = load_prompt_source_pool(background_pool_path)
    if background.version != "v5.8" or background.explicit_formulations:
        raise ValueError("Background v5.8 must be a top-level prompt pool.")
    if len(background.formulations) != 1:
        raise ValueError("Background v5.8 needs one default formulation.")
    background_variants = background.formulations[0].prompt_variants
    if tuple(item.variant_id for item in background_variants) != ("detailed",):
        raise ValueError("Background v5.8 must contain only the detailed prompt variant.")
    background_variants[0].render({}, context="background recovery preflight")

    image = load_prompt_source_pool(image_pool_path)
    if image.version != "v5.8" or not image.explicit_formulations:
        raise ValueError("Image v5.8 must use explicit formulations.")
    if tuple(item.formulation_id for item in image.formulations) != ("image_type",):
        raise ValueError("Image v5.8 needs exactly the image_type formulation.")
    if tuple(
        prompt.variant_id for prompt in image.formulations[0].prompt_variants
    ) != ("detailed", "concise"):
        raise ValueError("Image v5.8 needs detailed and concise prompt variants.")
    for prompt in image.formulations[0].prompt_variants:
        prompt.render(
            {"proposal_bbox_2d": [0, 0, 999, 999]},
            context="image recovery preflight",
        )


def _prepare_raw_import(
    staging_root: Path,
    *,
    annotations: Path,
    manifests: list[Path],
    canonical_test_manifest: Path,
    canonical_test_sha256: str,
    source_bundle_root: Path,
    source_artifacts: dict[str, str],
) -> tuple[Path, dict[str, Any]]:
    target = staging_root / "raw_import"
    _copy_file(annotations, target / annotations.name)
    for manifest in manifests:
        _copy_file(manifest, target / "splits" / manifest.name)
    summary = {
        "snapshot_id": RECOVERY_SNAPSHOT_ID,
        "role": "legacy_reviewed_source_sidecar",
        "active_compact_raw_replaced": False,
        "background_annotations": {
            "file": annotations.name,
            "sha256": _sha256(annotations),
        },
        "split_manifests": {
            manifest.name: _sha256(manifest) for manifest in manifests
        },
        "canonical_vlm_test": {
            "file": canonical_test_manifest.name,
            "rows": CANONICAL_VLM_TEST_ROWS,
            "sha256": canonical_test_sha256,
        },
        "derived_source_bundle": {
            "declared_root": str(source_bundle_root),
            "artifact_sha256": dict(sorted(source_artifacts.items())),
        },
    }
    _write_json(target / "source_manifest.json", summary)
    (target / "README.md").write_text(
        "# Banana v5.3 reviewed recovery source\n\n"
        "This sidecar preserves the reviewed full-image background annotation and the exact "
        "historical test manifests used by the recovered v5.8 tasks. It does not replace or "
        "rewrite the active compact `data/raw/json` contract. The historical enriched image-type "
        "raw source is unavailable; image-type rows are recovered from the separately verified "
        "v5.3 task bundle and are explicitly documented as recovered derived truth.\n",
        encoding="utf-8",
    )
    return target, summary


def _recover_background(
    staging_root: Path,
    *,
    source_task: Path,
    annotations_path: Path,
    annotations: dict[str, dict[str, Any]],
    excluded_ids: set[str],
    canonical_test_ids: set[str],
    source_artifacts: dict[str, str],
    prompt_pool_sha256: str,
    workers: int,
) -> tuple[Path, dict[str, Any]]:
    target = staging_root / "background"
    for relative in ("structured", "sft", "reports"):
        (target / relative).mkdir(parents=True, exist_ok=True)
    source_structured = source_task / "structured/train.jsonl"
    source_sft = source_task / "sft/train.jsonl"
    ids: set[str] = set()
    jobs: list[MediaJob] = []
    distribution: Counter[str] = Counter()
    expected_train_ids = set(annotations) - excluded_ids
    with (
        source_structured.open(encoding="utf-8") as structured_input,
        source_sft.open(encoding="utf-8") as sft_input,
        (target / "structured/train.jsonl").open("w", encoding="utf-8") as structured_output,
        (target / "sft/train.jsonl").open("w", encoding="utf-8") as sft_output,
    ):
        for line_no, (structured_line, sft_line) in enumerate(
            zip(structured_input, sft_input, strict=True),
            1,
        ):
            structured = json.loads(structured_line)
            source_row = json.loads(sft_line)
            sample_id = str(structured.get("sample_id") or "")
            if not sample_id or sample_id in ids:
                raise ValueError(f"Invalid/duplicate background sample id at row {line_no}")
            ids.add(sample_id)
            annotation = annotations.get(sample_id)
            if annotation is None or sample_id in excluded_ids:
                raise ValueError(f"Background source/test mismatch for sample {sample_id}")
            background = structured.get("background")
            if not isinstance(background, bool) or annotation["background"] != background:
                raise ValueError(f"Background target mismatch for sample {sample_id}")
            expected_target = _json_dumps({"background": background})
            if (
                source_row.get("sample_id") != sample_id
                or source_row.get("image_path") != structured.get("image_path")
                or source_row.get("target_text") != expected_target
            ):
                raise ValueError(f"Background structured/SFT mismatch at row {line_no}")

            structured_extra = dict(structured.get("extra") or {})
            structured_extra["source_annotation"] = (
                f"raw/imports/{RAW_IMPORT_NAME}/{annotations_path.name}"
            )
            structured_extra["recovery_snapshot_id"] = RECOVERY_SNAPSHOT_ID
            structured["extra"] = structured_extra
            structured_output.write(_json_dumps(structured) + "\n")

            extra = dict(source_row.get("extra") or {})
            extra.pop("prompt_id", None)
            extra.update(
                {
                    "schema_version": "banana.sft.v5.8",
                    "prompt_pool_id": "shaft.background.prompt_pool.v5.8",
                    "structured_snapshot_id": RECOVERY_SNAPSHOT_ID,
                    "recovery_source": "verified_v5_3_replay_bundle",
                    "structured_extra": structured_extra,
                }
            )
            output_row = {
                "image_path": structured["image_path"],
                "sample_id": sample_id,
                "dataset_name": "background",
                "system_prompt": "",
                "user_prompt": "",
                "prompt_args": {},
                "target_text": expected_target,
                "extra": extra,
            }
            sft_output.write(_json_dumps(output_row) + "\n")
            source_media = _task_media_path(source_task, structured["image_path"])
            target_media = _task_media_path(target, structured["image_path"])
            jobs.append(
                MediaJob(
                    source=source_media,
                    target=target_media,
                    width=int(structured["image_width"]),
                    height=int(structured["image_height"]),
                )
            )
            distribution[str(background).lower()] += 1
    if ids != expected_train_ids:
        raise ValueError(
            "Recovered background ids do not equal annotation ids minus test ids: "
            f"missing={len(expected_train_ids - ids)}, extra={len(ids - expected_train_ids)}"
        )
    canonical_overlap = ids & canonical_test_ids
    if canonical_overlap:
        raise ValueError(f"Background canonical test overlap: {sorted(canonical_overlap)[:20]}")
    (target / "structured/val.jsonl").write_text("", encoding="utf-8")
    (target / "sft/val.jsonl").write_text("", encoding="utf-8")
    media_counts, media_errors = _materialize_and_verify_media(jobs, workers=workers)
    if media_errors:
        raise ValueError(f"Background media validation failed: {media_errors[:20]}")
    summary = {
        "status": "passed",
        "snapshot_id": RECOVERY_SNAPSHOT_ID,
        "source_truth": "reviewed_background_annotation",
        "source_annotations": len(annotations),
        "excluded_test_ids": len(excluded_ids),
        "canonical_test_rows": len(canonical_test_ids),
        "canonical_test_overlap": 0,
        "train_rows": len(ids),
        "val_rows": 0,
        "target_schema": {"background": "boolean"},
        "distribution": dict(sorted(distribution.items())),
        "unique_sample_ids": len(ids),
        "unique_media": len(jobs),
        "media": dict(sorted(media_counts.items())),
        "row_prompts_empty": True,
        "prompt_pool_id": "shaft.background.prompt_pool.v5.8",
        "prompt_pool_sha256": prompt_pool_sha256,
        "source_artifact_sha256": dict(sorted(source_artifacts.items())),
    }
    _write_json(target / "reports/build_summary.json", summary)
    _write_json(target / "reports/schema_validation.json", summary)
    (target / "README.md").write_text(
        "# background v5.8\n\n"
        "Recovered from the reviewed v5.3 full-image background annotation and its verified "
        "task-local media bundle. The active compact raw does not contain background labels and "
        "is not treated as this task's truth.\n\n"
        f"- Snapshot: `{RECOVERY_SNAPSHOT_ID}`\n"
        f"- Reviewed source rows: {len(annotations)}\n"
        f"- Excluded historical test ids: {len(excluded_ids)}\n"
        f"- Train rows: {len(ids)}\n"
        f"- Background true: {distribution['true']}\n"
        f"- Background false: {distribution['false']}\n"
        "- Validation rows: 0 (train-only)\n"
        "- Target: compact JSON `{\"background\":true|false}`\n"
        "- Runtime prompt: `configs/prompts/pools/background.v5.8.yaml`, detailed only\n"
        "- SFT row prompts: empty; PromptSource injects the runtime prompt\n"
        "- Media: historical task-local pixels are preserved without additional resizing\n",
        encoding="utf-8",
    )
    return target, summary


def _recover_image_context(
    staging_root: Path,
    *,
    source_task: Path,
    historical_test_ids: set[str],
    current_test_ids: set[str],
    canonical_test_ids: set[str],
    source_artifacts: dict[str, str],
    prompt_pool_sha256: str,
    workers: int,
) -> tuple[Path, dict[str, Any]]:
    target = staging_root / "image_context_reconstruction"
    for relative in (
        "selection",
        "structured",
        "sft/formulations/image_type",
        "reports",
    ):
        (target / relative).mkdir(parents=True, exist_ok=True)
    source_selection = source_task / "selection/train.jsonl"
    source_structured = source_task / "structured/train.jsonl"
    source_sft = source_task / "sft/train.jsonl"
    ids: set[str] = set()
    media_paths: set[str] = set()
    source_instances: dict[tuple[str, int], str] = {}
    source_stems: set[str] = set()
    distribution: Counter[str] = Counter()
    jobs: list[MediaJob] = []
    output_sft_path = target / "sft/formulations/image_type/train.jsonl"
    with (
        source_selection.open(encoding="utf-8") as selection_input,
        source_structured.open(encoding="utf-8") as structured_input,
        source_sft.open(encoding="utf-8") as sft_input,
        (target / "selection/train.jsonl").open("w", encoding="utf-8") as selection_output,
        (target / "structured/train.jsonl").open("w", encoding="utf-8") as structured_output,
        output_sft_path.open("w", encoding="utf-8") as sft_output,
    ):
        for line_no, (selection_line, structured_line, sft_line) in enumerate(
            zip(selection_input, structured_input, sft_input, strict=True),
            1,
        ):
            selection = json.loads(selection_line)
            structured = json.loads(structured_line)
            source_row = json.loads(sft_line)
            sample_id = str(structured.get("sample_id") or "")
            if not sample_id or sample_id in ids:
                raise ValueError(f"Invalid/duplicate image sample id at row {line_no}")
            ids.add(sample_id)
            selection_id = str(selection.get("sample_id") or "")
            if sample_id != f"{selection_id}__context_00":
                raise ValueError(f"Image selection/structured identity mismatch at row {line_no}")
            instances = structured.get("instances")
            if not isinstance(instances, list) or len(instances) != 1:
                raise ValueError(f"Image structured row needs one instance at row {line_no}")
            instance = instances[0]
            parameters = instance.get("parameters") if isinstance(instance, dict) else None
            image_type = parameters.get("image_type") if isinstance(parameters, dict) else None
            if image_type not in IMAGE_TYPES or set(parameters) != {"image_type"}:
                raise ValueError(f"Invalid image_type target at row {line_no}: {parameters}")
            expected_target = _json_dumps(
                {"type": "image", "parameters": {"image_type": image_type}}
            )
            structured_extra = dict(structured.get("extra") or {})
            proposal_bbox = structured_extra.get("proposal_bbox_2d")
            if (
                source_row.get("sample_id") != sample_id
                or source_row.get("image_path") != structured.get("image_path")
                or source_row.get("target_text") != expected_target
                or source_row.get("prompt_args") != {"proposal_bbox_2d": proposal_bbox}
                or str(source_row.get("system_prompt") or "")
                or str(source_row.get("user_prompt") or "")
            ):
                raise ValueError(f"Image structured/SFT mismatch at row {line_no}")
            selection_extra = dict(selection.get("extra") or {})
            source_json = str(selection_extra.get("source_json") or "")
            source_index = selection_extra.get("source_instance_index")
            if not source_json or not isinstance(source_index, int):
                raise ValueError(f"Image selection lacks source identity at row {line_no}")
            if (
                structured_extra.get("source_json") != source_json
                or structured_extra.get("source_instance_index") != source_index
            ):
                raise ValueError(
                    f"Image selection/structured source identity mismatch at row {line_no}"
                )
            source_stem = Path(source_json).stem
            if source_stem in historical_test_ids or source_stem in current_test_ids:
                raise ValueError(f"Image test overlap for source {source_stem}")
            source_stems.add(source_stem)
            identity = (source_json, source_index)
            previous_type = source_instances.setdefault(identity, image_type)
            if previous_type != image_type:
                raise ValueError(f"Conflicting recovered image_type for {identity}")

            selection_output.write(_json_dumps(selection) + "\n")
            structured_extra["recovery_snapshot_id"] = RECOVERY_SNAPSHOT_ID
            structured["extra"] = structured_extra
            structured_output.write(_json_dumps(structured) + "\n")
            output_image_path = _formulation_media_path(structured["image_path"])
            extra = dict(source_row.get("extra") or {})
            extra.update(
                {
                    "schema_version": "banana.sft.v5.8",
                    "prompt_pool_id": (
                        "shaft.image_context_reconstruction.formulation_pool.v5.8"
                    ),
                    "structured_snapshot_id": RECOVERY_SNAPSHOT_ID,
                    "recovery_source": "verified_v5_3_replay_bundle",
                    "structured_extra": structured_extra,
                }
            )
            output_row = {
                "image_path": output_image_path,
                "sample_id": sample_id,
                "dataset_name": "image_context_reconstruction",
                "system_prompt": "",
                "user_prompt": "",
                "prompt_args": {"proposal_bbox_2d": proposal_bbox},
                "target_text": expected_target,
                "extra": extra,
            }
            sft_output.write(_json_dumps(output_row) + "\n")
            source_media = _task_media_path(source_task, structured["image_path"])
            target_media = _task_media_path(target, structured["image_path"])
            media_key = str(target_media)
            if media_key in media_paths:
                raise ValueError(f"Duplicate image media path at row {line_no}: {target_media}")
            media_paths.add(media_key)
            jobs.append(
                MediaJob(
                    source=source_media,
                    target=target_media,
                    width=int(structured["image_width"]),
                    height=int(structured["image_height"]),
                )
            )
            distribution[image_type] += 1
    (target / "selection/val.jsonl").write_text("", encoding="utf-8")
    (target / "structured/val.jsonl").write_text("", encoding="utf-8")
    (target / "sft/formulations/image_type/val.jsonl").write_text("", encoding="utf-8")
    media_counts, media_errors = _materialize_and_verify_media(jobs, workers=workers)
    if media_errors:
        raise ValueError(f"Image context media validation failed: {media_errors[:20]}")
    summary = {
        "status": "passed",
        "snapshot_id": RECOVERY_SNAPSHOT_ID,
        "source_truth_status": "verified_recovered_v5_3_derived_bundle",
        "enriched_raw_available": False,
        "train_rows": len(ids),
        "val_rows": 0,
        "formulations": ["image_type"],
        "target_schema": {"type": "image", "parameters": ["image_type"]},
        "image_types": sorted(IMAGE_TYPES),
        "distribution": dict(sorted(distribution.items())),
        "unique_sample_ids": len(ids),
        "unique_source_instances": len(source_instances),
        "unique_source_images": len(source_stems),
        "historical_test_overlap": 0,
        "current_test_overlap": 0,
        "canonical_test_rows": len(canonical_test_ids),
        "canonical_test_overlap": 0,
        "unique_media": len(jobs),
        "media": dict(sorted(media_counts.items())),
        "row_prompts_empty": True,
        "prompt_pool_id": (
            "shaft.image_context_reconstruction.formulation_pool.v5.8"
        ),
        "prompt_pool_sha256": prompt_pool_sha256,
        "source_artifact_sha256": dict(sorted(source_artifacts.items())),
    }
    _write_json(target / "reports/build_summary.json", summary)
    _write_json(
        target / "reports/formulation_alignment.json",
        {
            "status": "passed",
            "formulations": ["image_type"],
            "rows": len(ids),
            "identity_aligned": True,
        },
    )
    _write_json(target / "reports/schema_validation.json", summary)
    (target / "README.md").write_text(
        "# image_context_reconstruction v5.8\n\n"
        "Recovered from the complete, verified v5.3 image-context task bundle. The historical "
        "reviewed enriched raw source is no longer available, so this task is explicitly recorded "
        "as recovered derived truth rather than being promoted to active raw truth. V9 "
        "`image_type=N/A` and current compact image instances are not used as labels.\n\n"
        f"- Snapshot: `{RECOVERY_SNAPSHOT_ID}`\n"
        f"- Train rows: {len(ids)}\n"
        f"- Unique source instances: {len(source_instances)}\n"
        f"- Unique source images: {len(source_stems)}\n"
        "- Validation rows: 0 (train-only)\n"
        "- Formulation: `image_type`\n"
        "- Target: reviewed 13-class `image_type` only\n"
        "- Runtime prompt: `configs/prompts/pools/image_context_reconstruction.v5.8.yaml`\n"
        "- SFT row prompts: empty; `proposal_bbox_2d` remains prompt-only input\n"
        "- Media: original verified contextual crops, with no new pixel augmentation\n",
        encoding="utf-8",
    )
    return target, summary


def _publish_many(pairs: list[tuple[Path, Path]], *, clean: bool) -> None:
    existing = [target for _, target in pairs if target.exists()]
    if existing and not clean:
        raise FileExistsError(
            "Recovery targets already exist; rerun with --clean after review: "
            + ", ".join(str(path) for path in existing)
        )
    backups: list[tuple[Path, Path]] = []
    published: list[Path] = []
    try:
        for staging, target in pairs:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                backup = target.with_name(f".{target.name}.backup.{uuid.uuid4().hex}")
                os.replace(target, backup)
                backups.append((backup, target))
            os.replace(staging, target)
            published.append(target)
    except Exception:
        for target in reversed(published):
            if target.exists():
                shutil.rmtree(target)
        for backup, target in reversed(backups):
            if backup.exists():
                os.replace(backup, target)
        raise
    if backups:
        with ThreadPoolExecutor(max_workers=len(backups)) as executor:
            list(executor.map(shutil.rmtree, [backup for backup, _ in backups]))


def build(args: argparse.Namespace) -> dict[str, Any]:
    source_bundle_root = Path(args.source_bundle_root).resolve()
    annotations_path = Path(args.background_annotations).resolve()
    historical_manifests = [Path(path).resolve() for path in args.test_manifests]
    current_manifests = [Path(path).resolve() for path in args.current_test_manifests]
    canonical_test_manifest = Path(args.canonical_test_manifest).resolve()
    canonical_test_sha256 = str(args.canonical_test_sha256).strip().lower()
    output_root = Path(args.output_root).resolve()
    background_pool_path = Path(args.background_prompt_pool).resolve()
    image_pool_path = Path(args.image_prompt_pool).resolve()
    workers = int(args.workers)
    if workers <= 0:
        raise ValueError("workers must be positive")
    source_paths = {
        "background/structured/train.jsonl": (
            source_bundle_root / "background/structured/train.jsonl"
        ),
        "background/sft/train.jsonl": source_bundle_root / "background/sft/train.jsonl",
        "image_context_reconstruction/selection/train.jsonl": (
            source_bundle_root / "image_context_reconstruction/selection/train.jsonl"
        ),
        "image_context_reconstruction/structured/train.jsonl": (
            source_bundle_root / "image_context_reconstruction/structured/train.jsonl"
        ),
        "image_context_reconstruction/sft/train.jsonl": (
            source_bundle_root / "image_context_reconstruction/sft/train.jsonl"
        ),
    }
    required = [
        annotations_path,
        *historical_manifests,
        *current_manifests,
        canonical_test_manifest,
        *source_paths.values(),
        background_pool_path,
        image_pool_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing recovery inputs: {missing}")
    source_artifacts = {
        relative: _sha256(path) for relative, path in source_paths.items()
    }
    background_source_artifacts = {
        relative: digest
        for relative, digest in source_artifacts.items()
        if relative.startswith("background/")
    }
    image_source_artifacts = {
        relative: digest
        for relative, digest in source_artifacts.items()
        if relative.startswith("image_context_reconstruction/")
    }
    _validate_prompt_contracts(background_pool_path, image_pool_path)
    annotations = _load_background_annotations(annotations_path)
    historical_test_ids = _read_test_ids(historical_manifests)
    current_test_ids = _read_test_ids(current_manifests)
    canonical_test_ids, canonical_test_digest = _validate_canonical_test_manifest(
        canonical_test_manifest,
        expected_sha256=canonical_test_sha256,
        historical_manifests=historical_manifests,
    )
    historical_test_ids.update(canonical_test_ids)
    current_test_ids.update(canonical_test_ids)

    output_root.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=".v5_8_real_recovery.", dir=output_root))
    try:
        raw_import, raw_summary = _prepare_raw_import(
            staging_root,
            annotations=annotations_path,
            manifests=historical_manifests,
            canonical_test_manifest=canonical_test_manifest,
            canonical_test_sha256=canonical_test_digest,
            source_bundle_root=source_bundle_root,
            source_artifacts=source_artifacts,
        )
        background, background_summary = _recover_background(
            staging_root,
            source_task=source_bundle_root / "background",
            annotations_path=annotations_path,
            annotations=annotations,
            excluded_ids=historical_test_ids,
            canonical_test_ids=canonical_test_ids,
            source_artifacts=background_source_artifacts,
            prompt_pool_sha256=_sha256(background_pool_path),
            workers=workers,
        )
        image, image_summary = _recover_image_context(
            staging_root,
            source_task=source_bundle_root / "image_context_reconstruction",
            historical_test_ids=historical_test_ids,
            current_test_ids=current_test_ids,
            canonical_test_ids=canonical_test_ids,
            source_artifacts=image_source_artifacts,
            prompt_pool_sha256=_sha256(image_pool_path),
            workers=workers,
        )
        pairs = [
            (background, output_root / "background"),
            (image, output_root / "image_context_reconstruction"),
            (raw_import, output_root / "raw/imports" / RAW_IMPORT_NAME),
        ]
        _publish_many(pairs, clean=bool(args.clean))
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
    return {
        "status": "passed",
        "snapshot_id": RECOVERY_SNAPSHOT_ID,
        "workers": workers,
        "canonical_test_gate": {
            "rows": len(canonical_test_ids),
            "sha256": canonical_test_digest,
            "overlap": 0,
        },
        "source_artifact_sha256": dict(sorted(source_artifacts.items())),
        "raw_import": raw_summary,
        "background": background_summary,
        "image_context_reconstruction": image_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recover reviewed background and image-context data for Banana v5.8."
    )
    parser.add_argument("--source-bundle-root", required=True)
    parser.add_argument("--background-annotations", required=True)
    parser.add_argument("--test-manifests", nargs="+", required=True)
    parser.add_argument(
        "--canonical-test-manifest",
        default="data/raw/splits/vlm.test.json",
    )
    parser.add_argument(
        "--canonical-test-sha256",
        default=CANONICAL_VLM_TEST_SHA256,
    )
    parser.add_argument(
        "--current-test-manifests",
        nargs="*",
        default=[
            "data/raw/splits/main.test.json",
            "data/raw/splits/inpainting.test.json",
            "data/raw/splits/vlm.test.json",
        ],
    )
    parser.add_argument("--output-root", default="data")
    parser.add_argument(
        "--background-prompt-pool",
        default="configs/prompts/pools/background.v5.8.yaml",
    )
    parser.add_argument(
        "--image-prompt-pool",
        default="configs/prompts/pools/image_context_reconstruction.v5.8.yaml",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    try:
        summary = build(args)
    except (FileExistsError, FileNotFoundError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
