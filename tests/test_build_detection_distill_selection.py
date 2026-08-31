from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

from PIL import Image
import torch

from shaft.offline_kd import (
    OfflineKDArtifactReference,
    OfflineKDArtifactRow,
    OfflineKDArtifactStore,
    OfflineKDArtifactWriter,
    OfflineKDDistributionSpec,
    ShaftOfflineKDInputContract,
    deterministic_detection_media_plan,
    media_content_fingerprint,
)
from shaft.opd.input_abi import ShaftOPDInputABI
from shaft.training.distribution_loss import TeacherDistribution


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/tasks/build_detection_distill_selection.py"
SPEC = importlib.util.spec_from_file_location("build_detection_distill_selection", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

PACKAGE_SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts/tasks/package_detection_distillation.py"
)
PACKAGE_SPEC = importlib.util.spec_from_file_location(
    "package_detection_distillation", PACKAGE_SCRIPT
)
assert PACKAGE_SPEC is not None and PACKAGE_SPEC.loader is not None
PACKAGE_MODULE = importlib.util.module_from_spec(PACKAGE_SPEC)
sys.modules[PACKAGE_SPEC.name] = PACKAGE_MODULE
PACKAGE_SPEC.loader.exec_module(PACKAGE_MODULE)


def _image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (80, 40), color=color).save(path)


def test_selection_excludes_labeled_test_and_exact_duplicate_without_copying(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    raw = data_root / "raw_data"
    _image(raw / "images" / "labeled.png", (1, 2, 3))
    _image(raw / "images" / "test.png", (4, 5, 6))
    _image(raw / "images" / "free.png", (7, 8, 9))
    (raw / "json").mkdir(parents=True)
    (raw / "json" / "labeled.json").write_text("{}", encoding="utf-8")
    (raw / "splits").mkdir()
    (raw / "splits" / "main.test.json").write_text(
        json.dumps({"items": [{"id": "test"}]}), encoding="utf-8"
    )
    _image(data_root / "paper" / "paper.png", (10, 11, 12))
    chai = data_root / "ppt" / "chai_group"
    _image(chai / "train.png", (13, 14, 15))
    _image(chai / "eval.png", (16, 17, 18))
    (chai / "_source_manifest.tsv").write_text(
        "filename\tsplit\ntrain.png\ttrain\neval.png\teval\n", encoding="utf-8"
    )
    _image(data_root / "ppt" / "wyk" / "duplicate.png", (10, 11, 12))
    prompt = tmp_path / "prompt.yaml"
    prompt.write_text(
        """
metadata:
  id: test.grounding
  version: v5.8
prompts:
  - id: detailed
    system_prompt: Return JSON.
    user_prompt: Detect objects.
""",
        encoding="utf-8",
    )
    output = tmp_path / "selection.jsonl"

    manifest = MODULE.build_selection(
        data_root=data_root,
        output_path=output,
        prompt_path=prompt,
        prompt_variant="detailed",
        seed=465,
        workers=2,
        content_dedupe=True,
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["source_pool"] for row in rows] == [
        "raw_unlabeled",
        "paper",
        "ppt_chai_train",
    ]
    assert all(row["target_text"] == "" and row["task"] == "detection" for row in rows)
    assert all(row["prompt_variant_id"] == "detailed" for row in rows)
    assert all("offline_kd_media_plan" in row for row in rows)
    assert manifest["images_copied"] == 0
    assert manifest["exclusions"] == {"candidate_content_duplicate": 1}
    assert not any(output.parent.glob("*.png"))


def _input_abi() -> ShaftOPDInputABI:
    return ShaftOPDInputABI(
        token_to_id_fingerprint="a" * 64,
        token_count=5,
        special_token_ids=(("eos_token", (2,)),),
        logits_vocab_size=5,
        processor_abi_fingerprint="b" * 64,
        required_model_input_names=("input_ids",),
        optional_model_input_names=("attention_mask",),
        forward_accepted_input_names=("attention_mask", "input_ids"),
        forward_required_input_names=("input_ids",),
        forward_accepts_kwargs=False,
    )


def test_package_detection_distillation_reuses_public_artifact_contract(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "source" / "item.png"
    _image(image_path, (21, 22, 23))
    plan = deterministic_detection_media_plan(
        sample_id="paper:paper/item.png",
        width=80,
        height=40,
        seed=465,
    )
    target_text = '[{"bbox_2d":[1,2,30,40],"label":"shape"}]'
    source_payload = {
        "image_path": str(image_path),
        "sample_id": "paper:paper/item.png",
        "target_text": target_text,
        "source_pool": "paper",
        "source_relative_path": "paper/item.png",
        "offline_kd_media_plan": plan.to_dict(),
    }
    input_abi = _input_abi()
    input_contract = ShaftOfflineKDInputContract(
        max_length=64,
        add_eos_token=True,
        min_pixels=None,
        max_pixels=None,
        media_snapshot_id="detection-package-test-v1",
    )
    artifact_root = tmp_path / "source-artifact"
    with OfflineKDArtifactWriter(
        artifact_root,
        teacher_model="teacher",
        teacher_checkpoint_fingerprint="c" * 64,
        input_abi=input_abi,
        input_contract=input_contract,
        distribution_spec=OfflineKDDistributionSpec(
            mode="topk_tail", temperature=1.0, top_k=2
        ),
        source_fingerprint="d" * 64,
        denylist_fingerprint="e" * 64,
        shard_rows=1,
    ) as writer:
        writer.add(
            OfflineKDArtifactRow(
                source_payload=source_payload,
                input_token_ids=torch.tensor([0, 1, 3, 4]),
                completion_token_ids=torch.tensor([3, 4]),
                media_sha256=media_content_fingerprint((str(image_path),)),
                distribution=TeacherDistribution(
                    kind="topk_tail",
                    vocab_size=5,
                    topk_token_ids=torch.tensor([[3, 1], [4, 2]]),
                    topk_log_probs=torch.tensor([[0.6, 0.2], [0.5, 0.25]]).log(),
                    tail_log_probs=torch.tensor([0.2, 0.25]).log(),
                    temperature=1.0,
                ),
            )
        )
        writer.finalize()
    selection_path = tmp_path / "selection.jsonl"
    selection_path.write_text(json.dumps(source_payload) + "\n", encoding="utf-8")
    output = tmp_path / "bundle"

    PACKAGE_MODULE.build(
        argparse.Namespace(
            selection=str(selection_path),
            accepted=str(artifact_root / "train.jsonl"),
            old_artifact=str(artifact_root),
            exclusions=[],
            output=str(output),
            workers=1,
            shard_rows=1,
            shard_max_bytes=1024 * 1024,
        )
    )

    packaged_root = output / "offline_kd" / "detection"
    packaged_manifest = json.loads((packaged_root / "manifest.json").read_text())
    packaged_row = json.loads((packaged_root / "train.jsonl").read_text())
    reference = OfflineKDArtifactReference.from_mapping(packaged_row["distillation_ref"])
    store = OfflineKDArtifactStore(
        packaged_root / "manifest.json",
        student_input_abi=input_abi,
        student_input_contract=input_contract,
    )
    distribution = store.get(
        reference,
        completion_token_ids=torch.tensor([3, 4]),
        input_token_ids=torch.tensor([0, 1, 3, 4]),
        media_sha256=media_content_fingerprint((str(output / "images" / "item.png"),)),
    )

    assert packaged_manifest["artifact_id"] == reference.artifact_id
    assert distribution.topk_token_ids.tolist() == [[3, 1], [4, 2]]
    assert (output / "gt_standard" / "item.json").is_file()
