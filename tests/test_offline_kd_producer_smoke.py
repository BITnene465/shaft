from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest
import torch

from shaft.config import load_config
from shaft.offline_kd.artifact import (
    OfflineKDArtifactStore,
    build_offline_kd_input_contract,
)
from shaft.offline_kd.producer import (
    OfflineKDDistributionSpec,
    produce_offline_kd_artifact,
)
from shaft.model import build_model_tokenizer_processor
from shaft.opd.input_abi import build_opd_input_abi


def test_cpu_fake_teacher_producer_builds_training_ready_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (8, 8), color=(5, 17, 29)).save(image_path)
    train_path = tmp_path / "train.jsonl"
    train_path.write_text(
        json.dumps(
            {
                "image_path": str(image_path),
                "sample_id": "train-1",
                "user_prompt": "name a color",
                "target_text": "blue",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    denylist_path = tmp_path / "denylist.json"
    denylist_path.write_text(
        json.dumps(
            {
                "version": "shaft-offline-kd-denylist-v1",
                "sample_ids": [],
                "image_paths": [],
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "producer.yaml"
    config_path.write_text(
        f"""
experiment:
  name: producer-smoke
  output_dir: {tmp_path / 'unused-training-output'}
  seed: 7
model:
  model_type: smoke_vlm
  model_name_or_path: smoke-teacher
  torch_dtype: float32
  finetune: {{mode: full}}
algorithm: {{name: sft}}
data:
  max_length: 64
  min_pixels: 1024
  max_pixels: 2048
  schedule:
    mixing: weighted
    shuffle: true
  batching:
    grouping: none
    cardinality: fixed
    packing: {{mode: none}}
    layout: padded
  datasets:
    - dataset_name: teacher
      source_type: jsonl_sft
      train_path: {train_path}
      use_for_eval: false
  num_workers: 0
  persistent_workers: false
  pin_memory: false
  media_snapshot_id: producer-smoke-v1
train:
  duration: {{unit: steps, value: 1}}
  loss_name: auto
  loss_scale: default
  save_strategy: no
  save_final_model: false
  save_final_state: false
  load_best_model_at_end: false
  bf16: false
  fp16: false
  use_cpu: true
eval: {{enabled: false}}
""",
        encoding="utf-8",
    )
    config = load_config(config_path)
    output_dir = tmp_path / "artifact"

    result = produce_offline_kd_artifact(
        config,
        output_dir=output_dir,
        distribution_spec=OfflineKDDistributionSpec(
            mode="topk_tail",
            temperature=2.0,
            top_k=4,
        ),
        denylist_path=denylist_path,
        shard_rows=1,
        storage_dtype=torch.float32,
    )

    assert result == output_dir
    produced_rows = (output_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(produced_rows) == 1
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["teacher"]["model"] == "smoke_vlm"
    artifacts = build_model_tokenizer_processor(config)
    store = OfflineKDArtifactStore(
        output_dir / "manifest.json",
        student_input_abi=build_opd_input_abi(artifacts),
        student_input_contract=build_offline_kd_input_contract(config),
    )
    assert store.mode == "topk_tail"
    assert store.top_k == 4

    with pytest.raises(ValueError, match="Expected teacher checkpoint fingerprint differs"):
        produce_offline_kd_artifact(
            config,
            output_dir=tmp_path / "identity-mismatch",
            distribution_spec=OfflineKDDistributionSpec(
                mode="dense_logits",
            ),
            denylist_path=denylist_path,
            expected_teacher_checkpoint_fingerprint="0" * 64,
        )

    class FakeVLLMEngine:
        def generate(self, prompts, params, use_tqdm):
            assert params.prompt_logprobs == -1
            assert all("mm_processor_kwargs" not in prompt for prompt in prompts)
            assert all("min_pixels" not in prompt for prompt in prompts)
            assert all("max_pixels" not in prompt for prompt in prompts)
            uniform_logprob = -float(torch.log(torch.tensor(128.0)))
            mapping = {
                token_id: SimpleNamespace(logprob=uniform_logprob)
                for token_id in range(128)
            }
            return [
                SimpleNamespace(
                    prompt_token_ids=prompt["prompt_token_ids"],
                    prompt_logprobs=[None]
                    + [mapping for _ in range(len(prompt["prompt_token_ids"]) - 1)],
                )
                for prompt in prompts
            ]

    def reject_hf_model_load(*args, **kwargs):
        raise AssertionError("vLLM producer must not materialize the HF teacher model")

    monkeypatch.setattr(
        "shaft.offline_kd.producer.build_model_tokenizer_processor",
        reject_hf_model_load,
    )
    vllm_output_dir = tmp_path / "artifact-vllm"
    produce_offline_kd_artifact(
        config,
        output_dir=vllm_output_dir,
        distribution_spec=OfflineKDDistributionSpec(
            mode="topk_tail",
            temperature=2.0,
            top_k=4,
        ),
        denylist_path=denylist_path,
        scorer_backend="vllm",
        vllm_engine=FakeVLLMEngine(),
        storage_dtype=torch.float16,
    )
    assert (vllm_output_dir / "manifest.json").is_file()
