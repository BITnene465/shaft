from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest
from safetensors import safe_open
from safetensors.torch import save_file
import torch

from shaft.config import load_config, load_config_from_text
from shaft.config.algorithm import resolve_algorithm_profile
from shaft.cli.offline_kd_artifact import build_parser as build_artifact_parser
from shaft.loss_scale import ShaftLossScaleSpec
from shaft.model import build_model_meta
from shaft.offline_kd.artifact import (
    ARTIFACT_VERSION,
    OfflineKDArtifactReference,
    OfflineKDArtifactStore,
    ShaftOfflineKDInputContract,
    build_offline_kd_input_contract,
    clear_media_fingerprint_cache,
    media_content_fingerprint,
    merge_offline_kd_artifacts,
    offline_kd_artifact_identity,
)
from shaft.offline_kd.producer import (
    OfflineKDArtifactRow,
    OfflineKDArtifactWriter,
    OfflineKDDenylist,
    OfflineKDDistributionSpec,
    OfflineKDScoringBatch,
    ShaftOfflineKDVLLMScoringCollator,
    VLLMOfflineKDAsyncGreedyGenerator,
    VLLMOfflineKDGreedyGenerator,
    VLLMOfflineKDTeacherScorer,
    prepare_offline_kd_scoring_items,
    validate_detection_pseudo_label,
)
from shaft.offline_kd.data import load_jsonl_offline_kd_records
from shaft.offline_kd.media_plan import deterministic_detection_media_plan
from shaft.offline_kd.trainer import ShaftOfflineKDTrainer
from shaft.opd.input_abi import ShaftOPDInputABI
from shaft.template import (
    ShaftTemplatePromptPlan,
    ShaftTemplatePromptRow,
    ShaftTemplateSupervisedRow,
    ShaftTemplateSupervisionPlan,
)
from shaft.training.distribution_loss import DistributionLossComponents, TeacherDistribution
from shaft.training.sft_trainer import ShaftSFTTrainer


def _abi(*, vocab_size: int = 5, token_fingerprint: str = "1" * 64) -> ShaftOPDInputABI:
    return ShaftOPDInputABI(
        token_to_id_fingerprint=token_fingerprint,
        token_count=vocab_size,
        special_token_ids=(("eos_token", (1,)),),
        logits_vocab_size=vocab_size,
        processor_abi_fingerprint="2" * 64,
        required_model_input_names=("input_ids",),
        optional_model_input_names=("attention_mask",),
        forward_accepted_input_names=("attention_mask", "input_ids"),
        forward_required_input_names=("input_ids",),
        forward_accepts_kwargs=False,
    )


def _input_contract(*, max_length: int = 64) -> ShaftOfflineKDInputContract:
    return ShaftOfflineKDInputContract(
        max_length=max_length,
        add_eos_token=True,
        min_pixels=None,
        max_pixels=None,
        media_snapshot_id="offline-kd-test-v1",
    )


def test_qwen38_checkpoint4000_to_qwen35_4b_configs_share_input_contract() -> None:
    teacher = load_config(
        "configs/train/banana_offline_kd_teacher_qwen38_ckpt4000_v5_8.yaml"
    )
    student = load_config(
        "configs/train/banana_offline_kd_qwen35_4b_from_qwen38_ckpt4000_v5_8.yaml"
    )

    assert teacher.algorithm.name == "sft"
    assert teacher.model.model_type == "qwen38vl"
    assert teacher.model.model_name_or_path.endswith("checkpoint-4000")
    assert student.algorithm.name == "offline_kd"
    assert student.model.model_type == "qwen35vl"
    assert build_offline_kd_input_contract(teacher) == build_offline_kd_input_contract(
        student
    )


def test_detection_pseudo_kd_teacher_config_freezes_prompt_only_contract() -> None:
    config = load_config(
        "configs/train/banana_detection_pseudo_kd_teacher_qwen38_ckpt4000_v5_8.yaml"
    )

    assert config.algorithm.name == "sft"
    assert config.model.model_type == "qwen38vl"
    assert config.model.model_name_or_path.endswith("checkpoint-4000")
    assert config.data.schedule.shuffle is False
    assert config.data.max_length == 16_384
    assert config.data.min_pixels == 500_000
    assert config.data.max_pixels == 4_000_000
    assert len(config.data.datasets) == 1


def _write_artifact(
    tmp_path: Path, *, mode: str = "dense_logits"
) -> tuple[Path, OfflineKDArtifactReference]:
    abi = _abi()
    shard_path = tmp_path / "teacher-00001.safetensors"
    tensors = {
        "row_offsets": torch.tensor([0, 2], dtype=torch.long),
        "completion_token_ids": torch.tensor([2, 3], dtype=torch.long),
        "input_row_offsets": torch.tensor([0, 4], dtype=torch.long),
        "input_token_ids": torch.tensor([0, 1, 2, 3], dtype=torch.long),
        "media_sha256": torch.tensor([[7] * 32], dtype=torch.uint8),
    }
    top_k = None
    if mode == "dense_logits":
        tensors["dense_logits"] = torch.tensor(
            [[0.0, 1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0, 0.0]]
        )
    else:
        top_k = 2
        top_probs = torch.tensor([[0.5, 0.3], [0.6, 0.2]])
        tensors.update(
            {
                "topk_token_ids": torch.tensor([[4, 3], [0, 1]], dtype=torch.long),
                "topk_log_probs": top_probs.log(),
                "tail_log_probs": torch.tensor([0.2, 0.2]).log(),
            }
        )
    save_file(tensors, str(shard_path))
    shard_digest = hashlib.sha256(shard_path.read_bytes()).hexdigest()
    manifest = {
        "version": ARTIFACT_VERSION,
        "artifact_id": "",
        "teacher": {"model": "teacher", "checkpoint_fingerprint": "b" * 64},
        "input_abi": abi.to_dict(),
        "input_contract": _input_contract().to_dict(),
        "distribution": {
            "mode": mode,
            "temperature": None if mode == "dense_logits" else 1.0,
            "top_k": top_k,
            "vocab_size": 5,
        },
        "build": {
            "source_fingerprint": "c" * 64,
            "denylist_fingerprint": "d" * 64,
        },
        "shards": {shard_path.name: shard_digest},
    }
    artifact_id = offline_kd_artifact_identity(
        teacher=manifest["teacher"],
        input_abi=manifest["input_abi"],
        input_contract=manifest["input_contract"],
        distribution=manifest["distribution"],
        build=manifest["build"],
    )
    manifest["artifact_id"] = artifact_id
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, OfflineKDArtifactReference(artifact_id, shard_path.name, 0)


@pytest.mark.parametrize("mode", ["dense_logits", "topk_tail"])
def test_artifact_store_validates_and_reads_distribution(tmp_path: Path, mode: str) -> None:
    manifest_path, reference = _write_artifact(tmp_path, mode=mode)
    store = OfflineKDArtifactStore(
        manifest_path,
        student_input_abi=_abi(),
        student_input_contract=_input_contract(),
    )

    distribution = store.get(
        reference,
        completion_token_ids=torch.tensor([2, 3]),
        input_token_ids=torch.tensor([0, 1, 2, 3]),
        media_sha256=bytes([7] * 32),
    )

    assert isinstance(distribution, TeacherDistribution)
    assert distribution.kind == mode
    assert distribution.num_positions == 2
    assert distribution.vocab_size == 5


def test_artifact_store_fails_closed_on_completion_drift(tmp_path: Path) -> None:
    manifest_path, reference = _write_artifact(tmp_path)
    store = OfflineKDArtifactStore(
        manifest_path, student_input_abi=_abi(), student_input_contract=_input_contract()
    )

    with pytest.raises(ValueError, match="completion token alignment changed"):
        store.get(
            reference,
            completion_token_ids=torch.tensor([2, 4]),
            input_token_ids=torch.tensor([0, 1, 2, 3]),
            media_sha256=bytes([7] * 32),
        )


def test_artifact_store_fails_closed_on_tokenizer_abi_drift(tmp_path: Path) -> None:
    manifest_path, _ = _write_artifact(tmp_path)
    with pytest.raises(ValueError, match="token-to-ID mappings differ"):
        OfflineKDArtifactStore(
            manifest_path,
            student_input_abi=_abi(token_fingerprint="3" * 64),
            student_input_contract=_input_contract(),
        )


def test_merge_offline_kd_artifacts_rewrites_references_and_keeps_rows(tmp_path: Path) -> None:
    input_dirs = [tmp_path / "rank0", tmp_path / "rank1"]
    references = []
    for index, input_dir in enumerate(input_dirs):
        input_dir.mkdir()
        _manifest, reference = _write_artifact(input_dir, mode="topk_tail")
        references.append(reference)
        (input_dir / "train.jsonl").write_text(
            json.dumps(
                {
                    "image_path": f"/images/{index}.png",
                    "sample_id": f"sample-{index}",
                    "target_text": "[]",
                    "distillation_ref": {
                        "artifact_id": reference.artifact_id,
                        "shard": reference.shard,
                        "row": reference.row,
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )

    output = merge_offline_kd_artifacts(input_dirs, output_dir=tmp_path / "merged")

    rows = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    manifest = json.loads((output / "manifest.json").read_text())
    assert len(rows) == 2
    assert all(row["distillation_ref"]["artifact_id"] == manifest["artifact_id"] for row in rows)
    assert len(manifest["shards"]) == 2
    assert all((output / name).is_file() for name in manifest["shards"])


def test_merge_offline_kd_artifacts_rejects_spoofed_manifest_identity(tmp_path: Path) -> None:
    input_dir = tmp_path / "rank0"
    input_dir.mkdir()
    _write_artifact(input_dir, mode="topk_tail")
    manifest_path = input_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_id"] = "b" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="canonical semantic identity"):
        merge_offline_kd_artifacts([input_dir], output_dir=tmp_path / "merged")


def test_artifact_store_fails_before_collation_on_input_contract_drift(
    tmp_path: Path,
) -> None:
    manifest_path, _ = _write_artifact(tmp_path)
    with pytest.raises(ValueError, match="input contract differs"):
        OfflineKDArtifactStore(
            manifest_path,
            student_input_abi=_abi(),
            student_input_contract=_input_contract(max_length=65),
        )


def test_artifact_store_fails_closed_on_prompt_drift(tmp_path: Path) -> None:
    manifest_path, reference = _write_artifact(tmp_path)
    store = OfflineKDArtifactStore(
        manifest_path, student_input_abi=_abi(), student_input_contract=_input_contract()
    )

    with pytest.raises(ValueError, match="full input token alignment changed"):
        store.get(
            reference,
            completion_token_ids=torch.tensor([2, 3]),
            input_token_ids=torch.tensor([4, 1, 2, 3]),
            media_sha256=bytes([7] * 32),
        )


def test_artifact_store_fails_closed_on_media_drift(tmp_path: Path) -> None:
    manifest_path, reference = _write_artifact(tmp_path)
    store = OfflineKDArtifactStore(
        manifest_path, student_input_abi=_abi(), student_input_contract=_input_contract()
    )

    with pytest.raises(ValueError, match="media content changed"):
        store.get(
            reference,
            completion_token_ids=torch.tensor([2, 3]),
            input_token_ids=torch.tensor([0, 1, 2, 3]),
            media_sha256=bytes([8] * 32),
        )


def test_artifact_store_fails_closed_on_shard_mutation(tmp_path: Path) -> None:
    manifest_path, reference = _write_artifact(tmp_path)
    shard_path = tmp_path / reference.shard
    shard_path.write_bytes(shard_path.read_bytes() + b"changed")

    store = OfflineKDArtifactStore(
        manifest_path, student_input_abi=_abi(), student_input_contract=_input_contract()
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        store.get(
            reference,
            completion_token_ids=torch.tensor([2, 3]),
            input_token_ids=torch.tensor([0, 1, 2, 3]),
            media_sha256=bytes([7] * 32),
        )


def test_artifact_store_rejects_semantic_metadata_mutation(tmp_path: Path) -> None:
    manifest_path, _ = _write_artifact(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["teacher"]["model"] = "mutated-teacher"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="artifact_id does not match"):
        OfflineKDArtifactStore(
            manifest_path,
            student_input_abi=_abi(),
            student_input_contract=_input_contract(),
        )


def test_dense_artifact_identity_does_not_bind_runtime_divergence_or_temperature(
    tmp_path: Path,
) -> None:
    manifest_path, _ = _write_artifact(tmp_path, mode="dense_logits")
    store = OfflineKDArtifactStore(
        manifest_path, student_input_abi=_abi(), student_input_contract=_input_contract()
    )

    store.validate_objective(
        SimpleNamespace(
            mode="dense_logits",
            divergence="reverse_kl",
            temperature=3.0,
            top_k=None,
        )
    )


def test_topk_artifact_binds_projection_temperature_but_not_divergence(
    tmp_path: Path,
) -> None:
    manifest_path, _ = _write_artifact(tmp_path, mode="topk_tail")
    store = OfflineKDArtifactStore(
        manifest_path, student_input_abi=_abi(), student_input_contract=_input_contract()
    )
    store.validate_objective(
        SimpleNamespace(
            mode="topk_tail",
            divergence="jsd",
            temperature=1.0,
            top_k=2,
        )
    )
    with pytest.raises(ValueError, match="projection differs"):
        store.validate_objective(
            SimpleNamespace(
                mode="topk_tail",
                divergence="forward_kl",
                temperature=2.0,
                top_k=2,
            )
        )


def test_media_fingerprint_cache_invalidates_on_file_stat_identity_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_path = tmp_path / "media.bin"
    image_path.write_bytes(b"first")
    clear_media_fingerprint_cache()
    original_read_bytes = Path.read_bytes
    reads = 0

    def counting_read_bytes(path: Path) -> bytes:
        nonlocal reads
        reads += 1
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", counting_read_bytes)
    first = media_content_fingerprint((str(image_path),))
    assert media_content_fingerprint((str(image_path),)) == first
    assert reads == 1

    image_path.write_bytes(b"changed-size")
    second = media_content_fingerprint((str(image_path),))
    assert second != first
    assert reads == 2


def test_public_writer_atomically_builds_readable_shards(tmp_path: Path) -> None:
    output_dir = tmp_path / "artifact-output"
    image_path = tmp_path / "writer-image.bin"
    image_path.write_bytes(b"pixels")
    distribution = OfflineKDDistributionSpec(
        mode="dense_logits",
    )
    with OfflineKDArtifactWriter(
        output_dir,
        teacher_model="teacher",
        teacher_checkpoint_fingerprint="b" * 64,
        input_abi=_abi(),
        input_contract=_input_contract(),
        distribution_spec=distribution,
        source_fingerprint="c" * 64,
        denylist_fingerprint="d" * 64,
        shard_rows=1,
        storage_dtype=torch.float32,
    ) as writer:
        writer.add(
            OfflineKDArtifactRow(
                source_payload={
                    "image_path": str(image_path),
                    "user_prompt": "prompt",
                    "target_text": "answer",
                },
                input_token_ids=torch.tensor([0, 1, 2, 3]),
                completion_token_ids=torch.tensor([2, 3]),
                media_sha256=bytes([7] * 32),
                distribution=TeacherDistribution.from_dense_logits(
                    torch.zeros((2, 5))
                ),
            )
        )
        artifact_id = writer.artifact_id
        writer.finalize()

    assert (output_dir / "manifest.json").is_file()
    row = json.loads((output_dir / "train.jsonl").read_text(encoding="utf-8"))
    assert row["distillation_ref"]["artifact_id"] == artifact_id
    store = OfflineKDArtifactStore(
        output_dir / "manifest.json",
        student_input_abi=_abi(),
        student_input_contract=_input_contract(),
    )
    distribution = store.get(
        OfflineKDArtifactReference.from_mapping(row["distillation_ref"]),
        completion_token_ids=torch.tensor([2, 3]),
        input_token_ids=torch.tensor([0, 1, 2, 3]),
        media_sha256=bytes([7] * 32),
    )
    assert distribution.kind == "dense_logits"


def test_public_writer_default_float16_topk_round_trips(tmp_path: Path) -> None:
    output_dir = tmp_path / "topk-artifact"
    probabilities = torch.tensor(
        [
            [0.4892, 0.3101, 0.2007],
            [0.6123, 0.2187, 0.1690],
        ],
        dtype=torch.float32,
    )
    with OfflineKDArtifactWriter(
        output_dir,
        teacher_model="teacher",
        teacher_checkpoint_fingerprint="b" * 64,
        input_abi=_abi(),
        input_contract=_input_contract(),
        distribution_spec=OfflineKDDistributionSpec(
            mode="topk_tail",
            temperature=1.0,
            top_k=2,
        ),
        source_fingerprint="c" * 64,
        denylist_fingerprint="d" * 64,
        shard_rows=1,
        storage_dtype=torch.float16,
    ) as writer:
        writer.add(
            OfflineKDArtifactRow(
                source_payload={"image_path": "image", "target_text": "answer"},
                input_token_ids=torch.tensor([0, 1, 2, 3]),
                completion_token_ids=torch.tensor([2, 3]),
                media_sha256=bytes([7] * 32),
                distribution=TeacherDistribution(
                    kind="topk_tail",
                    vocab_size=5,
                    topk_token_ids=torch.tensor([[4, 3], [0, 1]]),
                    topk_log_probs=probabilities[:, :2].log(),
                    tail_log_probs=probabilities[:, 2].log(),
                    temperature=1.0,
                ),
            )
        )
        writer.finalize()

    store = OfflineKDArtifactStore(
        output_dir / "manifest.json",
        student_input_abi=_abi(),
        student_input_contract=_input_contract(),
    )
    row = json.loads((output_dir / "train.jsonl").read_text(encoding="utf-8"))
    distribution = store.get(
        OfflineKDArtifactReference.from_mapping(row["distillation_ref"]),
        completion_token_ids=torch.tensor([2, 3]),
        input_token_ids=torch.tensor([0, 1, 2, 3]),
        media_sha256=bytes([7] * 32),
    )

    assert distribution.topk_log_probs is not None
    assert distribution.topk_log_probs.dtype == torch.float32
    assert distribution.tail_log_probs is not None
    assert distribution.tail_log_probs.dtype == torch.float32


def test_writer_rejects_topk_projection_drift(tmp_path: Path) -> None:
    output_dir = tmp_path / "projection-drift"
    probabilities = torch.tensor([[0.5, 0.3, 0.2]], dtype=torch.float32)
    with OfflineKDArtifactWriter(
        output_dir,
        teacher_model="teacher",
        teacher_checkpoint_fingerprint="b" * 64,
        input_abi=_abi(),
        input_contract=_input_contract(),
        distribution_spec=OfflineKDDistributionSpec(
            mode="topk_tail",
            temperature=1.0,
            top_k=2,
        ),
        source_fingerprint="c" * 64,
        denylist_fingerprint="d" * 64,
    ) as writer:
        with pytest.raises(ValueError, match="top-k projection"):
            writer.add(
                OfflineKDArtifactRow(
                    source_payload={"image_path": "image", "target_text": "answer"},
                    input_token_ids=torch.tensor([0, 1, 2]),
                    completion_token_ids=torch.tensor([2]),
                    media_sha256=bytes([7] * 32),
                    distribution=TeacherDistribution(
                        kind="topk_tail",
                        vocab_size=5,
                        topk_token_ids=torch.tensor([[4, 3]]),
                        topk_log_probs=probabilities[:, :2].log(),
                        tail_log_probs=probabilities[:, 2].log(),
                        temperature=2.0,
                    ),
                )
            )


def test_artifact_reader_hashes_shards_without_path_read_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, reference = _write_artifact(tmp_path)
    original_read_bytes = Path.read_bytes

    def reject_shard_read_bytes(path: Path) -> bytes:
        if path.suffix == ".safetensors":
            raise AssertionError("artifact shard checksum must be streamed")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", reject_shard_read_bytes)
    store = OfflineKDArtifactStore(
        manifest_path, student_input_abi=_abi(), student_input_contract=_input_contract()
    )

    distribution = store.get(
        reference,
        completion_token_ids=torch.tensor([2, 3]),
        input_token_ids=torch.tensor([0, 1, 2, 3]),
        media_sha256=bytes([7] * 32),
    )

    assert distribution.kind == "dense_logits"


def test_writer_moves_pending_rows_to_cpu_and_flushes_by_byte_budget(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "bounded-artifact"
    with OfflineKDArtifactWriter(
        output_dir,
        teacher_model="teacher",
        teacher_checkpoint_fingerprint="b" * 64,
        input_abi=_abi(),
        input_contract=_input_contract(),
        distribution_spec=OfflineKDDistributionSpec(
            mode="dense_logits",
        ),
        source_fingerprint="c" * 64,
        denylist_fingerprint="d" * 64,
        shard_rows=128,
        shard_max_bytes=40,
        storage_dtype=torch.float16,
    ) as writer:
        for row_index in range(2):
            writer.add(
                OfflineKDArtifactRow(
                    source_payload={
                        "image_path": f"image-{row_index}",
                        "target_text": "answer",
                    },
                    input_token_ids=torch.tensor([0, 1, 2, 3]),
                    completion_token_ids=torch.tensor([2, 3]),
                    media_sha256=bytes([row_index] * 32),
                    distribution=TeacherDistribution.from_dense_logits(
                        torch.zeros((2, 5), dtype=torch.float32)
                    ),
                )
            )
        writer.finalize()

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["shards"]) == 2
    for shard_name in manifest["shards"]:
        with safe_open(output_dir / shard_name, framework="pt", device="cpu") as handle:
            assert handle.get_slice("dense_logits").get_dtype() == "F16"


def test_public_writer_cleans_failed_atomic_build(tmp_path: Path) -> None:
    output_dir = tmp_path / "failed-output"
    with pytest.raises(RuntimeError, match="stop"):
        with OfflineKDArtifactWriter(
            output_dir,
            teacher_model="teacher",
            teacher_checkpoint_fingerprint="b" * 64,
            input_abi=_abi(),
            input_contract=_input_contract(),
            distribution_spec=OfflineKDDistributionSpec(
                mode="dense_logits",
            ),
            source_fingerprint="c" * 64,
            denylist_fingerprint="d" * 64,
        ):
            raise RuntimeError("stop")
    assert not output_dir.exists()
    assert not list(tmp_path.glob(".failed-output.building-*"))


def test_resumable_writer_recovers_only_committed_shards_without_duplicate_rows(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "resumable-output"
    writer_kwargs = {
        "teacher_model": "teacher",
        "teacher_checkpoint_fingerprint": "b" * 64,
        "input_abi": _abi(),
        "input_contract": _input_contract(),
        "distribution_spec": OfflineKDDistributionSpec(mode="dense_logits"),
        "source_fingerprint": "c" * 64,
        "denylist_fingerprint": "d" * 64,
        "shard_rows": 1,
        "storage_dtype": torch.float16,
        "resume": True,
    }

    with pytest.raises(RuntimeError, match="interrupted"):
        with OfflineKDArtifactWriter(output_dir, **writer_kwargs) as writer:
            writer.add(
                OfflineKDArtifactRow(
                    source_payload={"image_path": "one", "target_text": "one"},
                    input_token_ids=torch.tensor([0, 1, 2]),
                    completion_token_ids=torch.tensor([2]),
                    media_sha256=bytes([1] * 32),
                    distribution=TeacherDistribution.from_dense_logits(
                        torch.zeros((1, 5))
                    ),
                    source_index=0,
                )
            )
            raise RuntimeError("interrupted")

    staging_dir = tmp_path / ".resumable-output.building"
    assert staging_dir.is_dir()
    with OfflineKDArtifactWriter(output_dir, **writer_kwargs) as writer:
        assert writer.resume_source_index == 1
        with pytest.raises(ValueError, match="source_index"):
            writer.add(
                OfflineKDArtifactRow(
                    source_payload={"image_path": "duplicate", "target_text": "duplicate"},
                    input_token_ids=torch.tensor([0, 1, 2]),
                    completion_token_ids=torch.tensor([2]),
                    media_sha256=bytes([1] * 32),
                    distribution=TeacherDistribution.from_dense_logits(
                        torch.zeros((1, 5))
                    ),
                    source_index=0,
                )
            )
        writer.add(
            OfflineKDArtifactRow(
                source_payload={"image_path": "two", "target_text": "two"},
                input_token_ids=torch.tensor([0, 1, 3]),
                completion_token_ids=torch.tensor([3]),
                media_sha256=bytes([2] * 32),
                distribution=TeacherDistribution.from_dense_logits(
                    torch.ones((1, 5))
                ),
                source_index=1,
            )
        )
        writer.finalize()

    rows = (output_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert len(rows) == 2
    assert len(manifest["shards"]) == 2
    assert not staging_dir.exists()
    assert not (output_dir / "build_state.json").exists()


def test_denylist_is_explicit_versioned_and_path_resolved(tmp_path: Path) -> None:
    image_path = tmp_path / "excluded.png"
    denylist_path = tmp_path / "denylist.json"
    denylist_path.write_text(
        json.dumps(
            {
                "version": "shaft-offline-kd-denylist-v1",
                "sample_ids": ["eval-1"],
                "image_paths": [image_path.name],
            }
        ),
        encoding="utf-8",
    )
    denylist = OfflineKDDenylist.load(denylist_path)

    assert denylist.excludes({"sample_id": "eval-1", "image_paths": ()})
    assert denylist.excludes({"sample_id": "train", "image_paths": (str(image_path),)})


def test_artifact_cli_requires_explicit_denylist_but_computes_teacher_identity() -> None:
    parser = build_artifact_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--config",
                "teacher.yaml",
                "--output-dir",
                "artifact",
                "--mode",
                "dense_logits",
            ]
        )
    parsed = parser.parse_args(
        [
            "--config",
            "teacher.yaml",
            "--output-dir",
            "artifact",
            "--denylist",
            "denylist.json",
            "--mode",
            "dense_logits",
            "--max-rows",
            "200",
        ]
    )
    assert not hasattr(parsed, "teacher_checkpoint_fingerprint")
    assert parsed.max_rows == 200


def test_offline_kd_jsonl_keeps_reference_outside_sft_extra(tmp_path: Path) -> None:
    _, reference = _write_artifact(tmp_path)
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"not-decoded-by-loader")
    jsonl_path = tmp_path / "train.jsonl"
    jsonl_path.write_text(
        json.dumps(
            {
                "image_path": str(image_path),
                "user_prompt": "prompt",
                "target_text": "answer",
                "distillation_ref": {
                    "artifact_id": reference.artifact_id,
                    "shard": reference.shard,
                    "row": reference.row,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    record = load_jsonl_offline_kd_records(jsonl_path, dataset_name="kd")[0]

    assert record.distillation_ref["artifact_id"] == reference.artifact_id
    assert "distillation_ref" not in record.extra


def test_offline_kd_config_is_independent_domain_and_resolves_manifest(tmp_path: Path) -> None:
    config = load_config_from_text(
        """
algorithm:
  name: offline_kd
data:
  batching:
    grouping: none
    cardinality: fixed
    packing: {mode: none}
    layout: padded
  datasets:
    - dataset_name: kd
      source_type: jsonl_offline_kd
      train_path: train.jsonl
      use_for_eval: false
train:
  loss_name: auto
  loss_scale: default
eval:
  enabled: false
offline_kd:
  artifact_manifest: artifact/manifest.json
  objective:
    mode: topk_tail
    divergence: jsd
    temperature: 2.0
    top_k: 64
  loss:
    ce_weight: 0.25
    kd_weight: 0.75
""",
        config_path=tmp_path / "train.yaml",
    )

    assert resolve_algorithm_profile("offline_kd").domain == "offline_kd"
    assert config.offline_kd.artifact_manifest == str(
        (tmp_path / "artifact/manifest.json").resolve()
    )
    assert config.offline_kd.loss.kd_weight == 0.75


def test_offline_kd_config_rejects_sft_source(tmp_path: Path) -> None:
    text = """
algorithm: {name: offline_kd}
data:
  batching:
    grouping: none
    cardinality: fixed
    packing: {mode: none}
    layout: padded
  datasets:
    - dataset_name: kd
      source_type: jsonl_sft
      train_path: train.jsonl
      use_for_eval: false
eval: {enabled: false}
offline_kd: {artifact_manifest: manifest.json}
"""
    with pytest.raises(ValueError, match="jsonl_offline_kd"):
        load_config_from_text(text, config_path=tmp_path / "train.yaml")


def test_zero_ce_weight_skips_sft_loss_but_keeps_one_student_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = object.__new__(ShaftOfflineKDTrainer)
    trainer.offline_kd_ce_weight = 0.0
    trainer.offline_kd_weight = 1.0
    trainer.model_adapter = None
    trainer.args = SimpleNamespace(average_tokens_across_devices=False, n_gpu=0)

    class Objective:
        def compute(self, student_logits, teacher_distribution):
            assert tuple(student_logits.shape) == (2, 5)
            assert teacher_distribution.num_positions == 2
            return DistributionLossComponents(
                numerator=student_logits.new_tensor(4.0),
                denominator=student_logits.new_tensor(2.0),
            )

    trainer.offline_kd_objective = Objective()

    def reject_sft_loss(*args, **kwargs):
        raise AssertionError("zero CE weight must not execute the SFT loss path")

    monkeypatch.setattr(ShaftSFTTrainer, "compute_loss", reject_sft_loss)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.forward_calls = 0

        def forward(self, input_ids, attention_mask):
            self.forward_calls += 1
            return SimpleNamespace(logits=torch.zeros((1, 3, 5), requires_grad=True))

    model = Model()
    teacher = TeacherDistribution.from_dense_logits(torch.zeros((2, 5)))
    loss = trainer.compute_loss(
        model,
        {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
            "labels": torch.tensor([[-100, 2, 3]]),
            "loss_scale": torch.ones((1, 3)),
            "_shaft_offline_kd_completion_mask": torch.tensor(
                [[False, True, True]]
            ),
            "_shaft_offline_kd_teacher_distribution": teacher,
        },
    )

    assert loss.item() == pytest.approx(2.0)
    assert model.forward_calls == 1


def test_zero_kd_weight_skips_distribution_objective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = object.__new__(ShaftOfflineKDTrainer)
    trainer.offline_kd_ce_weight = 0.25
    trainer.offline_kd_weight = 0.0

    class Objective:
        def compute(self, *args, **kwargs):
            raise AssertionError("zero KD weight must not execute distribution loss")

    trainer.offline_kd_objective = Objective()
    outputs = SimpleNamespace(logits=torch.zeros((1, 3, 5)))
    monkeypatch.setattr(
        ShaftSFTTrainer,
        "compute_loss",
        lambda *args, **kwargs: (torch.tensor(8.0), outputs),
    )
    teacher = TeacherDistribution.from_dense_logits(torch.zeros((2, 5)))

    loss = trainer.compute_loss(
        torch.nn.Identity(),
        {
            "labels": torch.tensor([[-100, 2, 3]]),
            "_shaft_offline_kd_completion_mask": torch.tensor(
                [[False, True, True]]
            ),
            "_shaft_offline_kd_teacher_distribution": teacher,
        },
    )

    assert loss.item() == pytest.approx(2.0)


class _OfflineKDResizeTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    all_special_ids = (0, 2, 99)

    def __call__(self, texts, **kwargs):
        _ = kwargs
        if isinstance(texts, str):
            texts = [texts]
        return {"input_ids": [[30] for _ in texts]}


class _OfflineKDResizeProcessor:
    image_token_id = 99

    def __init__(self) -> None:
        self.received_images = None
        self.received_kwargs = None

    @staticmethod
    def apply_chat_template(*args, **kwargs):
        _ = args, kwargs
        return "unused"

    def __call__(self, *, text, images, padding, return_tensors, **kwargs):
        _ = text, padding, return_tensors
        self.received_images = images
        self.received_kwargs = kwargs
        rows = []
        for raw_images in images:
            row_images = raw_images if isinstance(raw_images, (list, tuple)) else (raw_images,)
            row = [10]
            for image_index, _image in enumerate(row_images):
                row.extend([99, 99, 20 + image_index])
            rows.append(row)
        width = max(len(row) for row in rows)
        return {
            "input_ids": torch.tensor(
                [row + [0] * (width - len(row)) for row in rows],
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                [[1] * len(row) + [0] * (width - len(row)) for row in rows],
                dtype=torch.long,
            ),
        }


class _OfflineKDResizeTemplate:
    @staticmethod
    def _rendered_prefix(image_count: int) -> tuple[int, ...]:
        output = [10]
        for image_index in range(image_count):
            output.extend([99, 20 + image_index])
        return tuple(output)

    def build_supervision_plan(self, *, item, target_text, renderer, loss_scale_name):
        _ = renderer, loss_scale_name
        return ShaftTemplateSupervisionPlan(
            prompt_text="unused",
            target_text=target_text,
            loss_spec=ShaftLossScaleSpec(prefix_scale=0.0, target_scale=1.0),
            rendered_prefix_token_ids=self._rendered_prefix(len(item["images"])),
        )

    def build_prompt_plan(self, *, item, renderer):
        _ = renderer
        return ShaftTemplatePromptPlan(
            prompt_text="unused",
            rendered_prefix_token_ids=self._rendered_prefix(len(item["images"])),
        )

    @staticmethod
    def build_prompt_row(
        *, plan, tokenizer, processed_batch, row_index, prefix_token_layout, max_length
    ):
        _ = plan, tokenizer, prefix_token_layout, max_length
        mask = processed_batch.model_inputs["attention_mask"][row_index].bool()
        prefix = processed_batch.model_inputs["input_ids"][row_index][mask]
        return ShaftTemplatePromptRow(
            input_ids=prefix,
            attention_mask=torch.ones_like(prefix),
            processed_prefix_indices=tuple(range(prefix.numel())),
        )

    @staticmethod
    def build_supervised_row(
        *,
        plan,
        tokenizer,
        processed_batch,
        row_index,
        prefix_token_layout,
        add_eos_token,
        ignore_index,
        include_targets_in_inputs,
        max_length,
    ):
        _ = plan, tokenizer, prefix_token_layout, add_eos_token, max_length
        assert include_targets_in_inputs is True
        mask = processed_batch.model_inputs["attention_mask"][row_index].bool()
        prefix = processed_batch.model_inputs["input_ids"][row_index][mask]
        target = torch.tensor([30, 2], dtype=torch.long)
        return ShaftTemplateSupervisedRow(
            input_ids=torch.cat([prefix, target]),
            labels=torch.cat(
                [torch.full_like(prefix, ignore_index), target]
            ),
            attention_mask=torch.ones((prefix.numel() + target.numel(),), dtype=torch.long),
            processed_prefix_indices=tuple(range(prefix.numel())),
        )


def _offline_kd_resize_item(images: tuple[Image.Image, ...]) -> dict[str, object]:
    return {
        "dataset_name": "teacher",
        "sample_id": "sample",
        "image_paths": tuple(f"image-{index}.png" for index in range(len(images))),
        "image": images[0] if len(images) == 1 else images,
        "images": images,
        "target_text": "answer",
        "extra": {},
    }


def _offline_kd_vllm_collator(processor: _OfflineKDResizeProcessor):
    adapter = build_model_meta("qwen3vl").resolve_adapter(
        model_name_or_path="models/Qwen3-VL-4B-Instruct"
    )
    return adapter, ShaftOfflineKDVLLMScoringCollator(
        model_adapter=adapter,
        template=_OfflineKDResizeTemplate(),
        processor=processor,
        tokenizer=_OfflineKDResizeTokenizer(),
        image_token_id=99,
        max_length=64,
        add_eos_token=True,
        include_targets_in_inputs=True,
        include_metadata=False,
        loss_scale_name="default",
        layout="padded",
        packing_mode="none",
        collect_stats=False,
    )


def test_offline_kd_vllm_single_image_uses_same_smart_resized_pil() -> None:
    processor = _OfflineKDResizeProcessor()
    adapter, collator = _offline_kd_vllm_collator(processor)
    prepared = prepare_offline_kd_scoring_items(
        [_offline_kd_resize_item((Image.new("RGB", (100, 50)),))],
        model_adapter=adapter,
        min_pixels=None,
        max_pixels=2_048,
    )

    collation = collator.collate_for_vllm(prepared)

    resized = prepared[0]["images"][0]
    assert resized.size == (64, 32)
    assert processor.received_images[0] is resized
    assert collation.images[0][0] is resized
    assert collation.prompt_token_ids == ((10, 99, 20, 30, 2),)


def test_offline_kd_vllm_prompt_only_collation_excludes_target() -> None:
    processor = _OfflineKDResizeProcessor()
    adapter, collator = _offline_kd_vllm_collator(processor)
    prepared = prepare_offline_kd_scoring_items(
        [_offline_kd_resize_item((Image.new("RGB", (100, 50)),))],
        model_adapter=adapter,
        min_pixels=None,
        max_pixels=2_048,
    )

    collation = collator.collate_prompts_for_vllm(prepared)

    assert collation.prompt_token_ids == ((10, 99, 20),)
    assert collation.expanded_prompt_token_ids[0].tolist() == [10, 99, 99, 20]


def test_offline_kd_vllm_multi_image_keeps_one_placeholder_per_resized_pil() -> None:
    processor = _OfflineKDResizeProcessor()
    adapter, collator = _offline_kd_vllm_collator(processor)
    prepared = prepare_offline_kd_scoring_items(
        [
            _offline_kd_resize_item(
                (Image.new("RGB", (100, 50)), Image.new("RGB", (50, 100)))
            )
        ],
        model_adapter=adapter,
        min_pixels=None,
        max_pixels=2_048,
    )

    collation = collator.collate_for_vllm(prepared)

    resized_images = prepared[0]["images"]
    assert [image.size for image in resized_images] == [(64, 32), (32, 64)]
    assert all(
        processor.received_images[0][index] is image
        and collation.images[0][index] is image
        for index, image in enumerate(resized_images)
    )
    assert collation.prompt_token_ids == ((10, 99, 20, 99, 21, 30, 2),)


def test_offline_kd_local_processor_does_not_receive_pixel_budget_after_resize() -> None:
    processor = _OfflineKDResizeProcessor()
    adapter, collator = _offline_kd_vllm_collator(processor)
    prepared = prepare_offline_kd_scoring_items(
        [_offline_kd_resize_item((Image.new("RGB", (100, 50)),))],
        model_adapter=adapter,
        min_pixels=1_024,
        max_pixels=2_048,
    )

    collator.collate_for_vllm(prepared)

    assert processor.received_kwargs == {}
    assert prepared[0]["images"][0].size == (64, 32)


def test_offline_kd_scoring_uses_frozen_per_row_media_plan() -> None:
    processor = _OfflineKDResizeProcessor()
    adapter, _collator = _offline_kd_vllm_collator(processor)
    item = _offline_kd_resize_item((Image.new("RGB", (100, 50)),))
    item["extra"] = {
        "offline_kd_media_plan": deterministic_detection_media_plan(
            sample_id="raw:1", width=100, height=50, seed=465
        ).to_dict()
    }

    prepared = prepare_offline_kd_scoring_items(
        [item],
        model_adapter=adapter,
        min_pixels=1,
        max_pixels=2,
    )

    plan = item["extra"]["offline_kd_media_plan"]
    assert prepared[0]["images"][0].size == (
        plan["target_width"],
        plan["target_height"],
    )


def test_detection_pseudo_label_validator_is_strict_and_keeps_raw_order() -> None:
    raw = '[{"bbox_2d":[1,2,30,40],"label":"shape"}]'

    assert validate_detection_pseudo_label(raw) == [
        {"bbox_2d": [1, 2, 30, 40], "label": "shape"}
    ]
    with pytest.raises(ValueError, match="exactly bbox_2d and label"):
        validate_detection_pseudo_label(
            '[{"bbox_2d":[1,2,30,40],"label":"shape","score":1}]'
        )
    with pytest.raises(ValueError, match="exact JSON"):
        validate_detection_pseudo_label("```json\n[]\n```")


def test_vllm_greedy_generator_returns_same_pass_tokens_and_topk_tail() -> None:
    image = Image.new("RGB", (32, 32))

    def value(probability: float):
        return SimpleNamespace(logprob=float(torch.log(torch.tensor(probability))))

    class Engine:
        def generate(self, prompts, params, use_tqdm):
            assert use_tqdm is False
            assert prompts[0]["prompt_token_ids"] == [10, 99, 20]
            assert prompts[0]["multi_modal_data"]["image"] == [image]
            assert params.temperature == 0.0
            assert params.top_p == 1.0
            assert params.top_k == 0
            assert params.logprobs == 2
            return [
                SimpleNamespace(
                    prompt_token_ids=[10, 99, 20],
                    outputs=[
                        SimpleNamespace(
                            text="[]",
                            token_ids=[3, 4],
                            logprobs=(
                                {3: value(0.6), 1: value(0.2)},
                                {4: value(0.5), 2: value(0.25)},
                            ),
                            finish_reason="stop",
                        )
                    ],
                )
            ]

    generator = VLLMOfflineKDGreedyGenerator(
        model_name_or_path="unused",
        vocab_size=5,
        trust_remote_code=False,
        revision=None,
        dtype="float32",
        top_k=2,
        engine=Engine(),
    )

    generated = generator.generate(
        prompt_token_ids=((10, 99, 20),),
        expanded_prompt_token_ids=(torch.tensor([10, 99, 20]),),
        images=((image,),),
    )[0]

    assert generated.raw_text == "[]"
    assert generated.completion_token_ids.tolist() == [3, 4]
    assert generated.distribution.topk_token_ids.tolist() == [[3, 1], [4, 2]]
    assert torch.allclose(
        generated.distribution.tail_log_probs.exp(), torch.tensor([0.2, 0.25])
    )


def test_vllm_async_greedy_generator_returns_final_topk_tail() -> None:
    image = Image.new("RGB", (32, 32))

    def value(probability: float):
        return SimpleNamespace(logprob=float(torch.log(torch.tensor(probability))))

    class Engine:
        shutdown_called = False

        async def generate(self, prompt, params, request_id):
            assert request_id == "request-7"
            assert prompt["prompt_token_ids"] == [10, 99, 20]
            assert prompt["multi_modal_data"]["image"] == [image]
            assert params.logprobs == 2
            assert str(params.output_kind).endswith("FINAL_ONLY")
            yield SimpleNamespace(
                prompt_token_ids=[10, 99, 20],
                outputs=[
                    SimpleNamespace(
                        text="[]",
                        token_ids=[3, 4],
                        logprobs=(
                            {3: value(0.6), 1: value(0.2)},
                            {4: value(0.5), 2: value(0.25)},
                        ),
                        finish_reason="stop",
                    )
                ],
            )

        def shutdown(self):
            self.shutdown_called = True

    engine = Engine()
    generator = VLLMOfflineKDAsyncGreedyGenerator(
        model_name_or_path="unused",
        vocab_size=5,
        trust_remote_code=False,
        revision=None,
        dtype="float32",
        top_k=2,
        engine=engine,
    )
    generated = asyncio.run(
        generator.generate_one(
            request_id="request-7",
            prompt_token_ids=(10, 99, 20),
            expanded_prompt_token_ids=torch.tensor([10, 99, 20]),
            images=(image,),
        )
    )
    generator.shutdown()

    assert generated.raw_text == "[]"
    assert generated.distribution.topk_token_ids.tolist() == [[3, 1], [4, 2]]
    assert engine.shutdown_called is True


def test_vllm_scorer_uses_prompt_logprobs_and_preserves_exact_tail_mass() -> None:
    resized_image = Image.new("RGB", (64, 32))

    class Engine:
        def __init__(self):
            self.params = None

        def generate(self, prompts, params, use_tqdm):
            assert use_tqdm is False
            assert prompts[0]["prompt_token_ids"] == [10, 11, 12]
            assert prompts[0]["multi_modal_data"]["image"][0] is resized_image
            assert "mm_processor_kwargs" not in prompts[0]
            assert "min_pixels" not in prompts[0]
            assert "max_pixels" not in prompts[0]
            self.params = params

            def value(probability: float):
                return SimpleNamespace(logprob=float(torch.log(torch.tensor(probability))))

            return [
                SimpleNamespace(
                    prompt_token_ids=[10, 11, 11, 12],
                    prompt_logprobs=[
                        None,
                        None,
                        {0: value(0.5), 1: value(0.3), 4: value(0.05)},
                        {2: value(0.6), 3: value(0.2), 1: value(0.04)},
                    ],
                )
            ]

    engine = Engine()
    scorer = VLLMOfflineKDTeacherScorer(
        model_name_or_path="unused",
        distribution_spec=OfflineKDDistributionSpec(
            mode="topk_tail",
            temperature=1.0,
            top_k=2,
        ),
        vocab_size=5,
        trust_remote_code=False,
        revision=None,
        dtype="float32",
        engine=engine,
    )
    distributions = scorer.score(
        OfflineKDScoringBatch(
            model_inputs={},
            completion_mask=torch.tensor([[False, False, True, True]]),
            input_token_ids=(torch.tensor([10, 11, 11, 12]),),
            prompt_completion_masks=(torch.tensor([False, False, True, True]),),
            images=((resized_image,),),
            vllm_prompt_token_ids=((10, 11, 12),),
        )
    )

    assert engine.params.prompt_logprobs == 2
    assert len(distributions) == 1
    distribution = distributions[0]
    assert distribution.topk_token_ids.tolist() == [[0, 1], [2, 3]]
    assert distribution.tail_log_probs is not None
    assert distribution.tail_log_probs.exp().tolist() == pytest.approx([0.2, 0.2])


def test_vllm_scorer_rejects_double_image_placeholder_expansion() -> None:
    class Engine:
        @staticmethod
        def generate(prompts, params, use_tqdm):
            _ = params, use_tqdm
            assert prompts[0]["prompt_token_ids"] == [10, 99, 20, 30]
            return [
                SimpleNamespace(
                    # A second expansion of an already-expanded placeholder must never
                    # be accepted as the canonical Shaft sequence.
                    prompt_token_ids=[10, 99, 99, 99, 99, 20, 30],
                    prompt_logprobs=[None] * 7,
                )
            ]

    scorer = VLLMOfflineKDTeacherScorer(
        model_name_or_path="unused",
        distribution_spec=OfflineKDDistributionSpec(
            mode="topk_tail",
            temperature=1.0,
            top_k=2,
        ),
        vocab_size=5,
        trust_remote_code=False,
        revision=None,
        dtype="float32",
        engine=Engine(),
    )

    with pytest.raises(ValueError, match="prompt token IDs drifted"):
        scorer.score(
            OfflineKDScoringBatch(
                model_inputs={},
                completion_mask=torch.tensor([[False, False, False, False, True]]),
                input_token_ids=(torch.tensor([10, 99, 99, 20, 30]),),
                prompt_completion_masks=(
                    torch.tensor([False, False, False, False, True]),
                ),
                images=((Image.new("RGB", (64, 32)),),),
                vllm_prompt_token_ids=((10, 99, 20, 30),),
            )
        )


def test_vllm_topk_temperature_projection_requests_full_vocabulary() -> None:
    probabilities = [0.4, 0.3, 0.15, 0.1, 0.05]

    class Engine:
        def __init__(self):
            self.params = None

        def generate(self, prompts, params, use_tqdm):
            self.params = params
            mapping = {
                token_id: SimpleNamespace(logprob=float(torch.log(torch.tensor(probability))))
                for token_id, probability in enumerate(probabilities)
            }
            return [
                SimpleNamespace(
                    prompt_token_ids=prompts[0]["prompt_token_ids"],
                    prompt_logprobs=[None, mapping],
                )
            ]

    engine = Engine()
    scorer = VLLMOfflineKDTeacherScorer(
        model_name_or_path="unused",
        distribution_spec=OfflineKDDistributionSpec(
            mode="topk_tail",
            temperature=2.0,
            top_k=2,
        ),
        vocab_size=5,
        trust_remote_code=False,
        revision=None,
        dtype="float32",
        engine=engine,
    )
    distribution = scorer.score(
        OfflineKDScoringBatch(
            model_inputs={},
            completion_mask=torch.tensor([[False, True]]),
            input_token_ids=(torch.tensor([10, 11]),),
            prompt_completion_masks=(torch.tensor([False, True]),),
            images=((),),
            vllm_prompt_token_ids=((10, 11),),
        )
    )[0]

    assert engine.params.prompt_logprobs == -1
    expected = torch.tensor(probabilities).sqrt()
    expected = expected / expected.sum()
    assert distribution.topk_token_ids.tolist() == [[0, 1]]
    assert distribution.topk_log_probs is not None
    assert distribution.topk_log_probs.exp()[0].tolist() == pytest.approx(
        expected[:2].tolist()
    )
    assert distribution.tail_log_probs is not None
    assert distribution.tail_log_probs.exp().item() == pytest.approx(
        expected[2:].sum().item()
    )
