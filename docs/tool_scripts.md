# 工具脚本使用说明

本文件维护辅助脚本（非主训练入口）的用法：

- 推理（infer）
- Demo
- 离线评估（eval）
- LoRA merge 与部署导出

主训练入口仍是 `scripts/train.py`。

## 1. 单阶段推理

脚本：`scripts/arrow/infer.py`

单图：

```bash
python scripts/arrow/infer.py \
  --config configs/infer/infer_one_stage.yaml \
  --dense-model models/Qwen3-VL-4B-Instruct \
  --lora-adapter outputs/your_experiment/checkpoints/best \
  --image /path/to/figure.jpg
```

目录批量：

```bash
python scripts/arrow/infer.py \
  --config configs/infer/infer_one_stage.yaml \
  --dense-model models/Qwen3-VL-4B-Instruct \
  --lora-adapter outputs/your_experiment/checkpoints/best \
  --image-dir /path/to/images \
  --recursive \
  --output-dir outputs/infer_one_stage_batch
```

## 2. 两阶段推理

脚本：`scripts/arrow/infer_two_stage.py`

完整两阶段：

```bash
python scripts/arrow/infer_two_stage.py \
  --config configs/infer/infer_two_stage.yaml \
  --stage1-dense-model models/Qwen3-VL-4B-Instruct \
  --stage1-lora-adapter outputs/qwen3vl-s1-lora/4b/checkpoints/best \
  --stage2-dense-model models/Qwen3-VL-4B-Instruct \
  --stage2-lora-adapter outputs/qwen3vl-s2-lora/4b/checkpoints/best \
  --image /path/to/example.png \
  --output-dir outputs/two_stage_demo
```

如果不传 `--stage2-lora-adapter`，Stage2 使用 dense model。

## 3. Demo

- 单阶段：`app/demo.py`
- 两阶段：`app/demo_two_stage.py`

示例参数与推理脚本一致。

## 4. Stage1 离线评估

脚本：`scripts/arrow/eval_stage1_grounding.py`

```bash
python scripts/arrow/eval_stage1_grounding.py \
  --config configs/infer/infer_stage1_grounding.yaml \
  --dense-model models/Qwen3-VL-4B-Instruct \
  --lora-adapter outputs/qwen3vl-s1-lora/4b/checkpoints/best \
  --jsonl data/two_stage/stage1/val.jsonl \
  --output-dir outputs/eval/stage1_grounding/run_001
```

## 5. Stage2 离线评估

脚本：`scripts/arrow/eval_stage2_keypoints.py`

```bash
python scripts/arrow/eval_stage2_keypoints.py \
  --config configs/infer/infer_stage2_keypoint_sequence.yaml \
  --dense-model models/Qwen3-VL-4B-Instruct \
  --lora-adapter outputs/qwen3vl-s2-lora/4b/checkpoints/best \
  --jsonl data/two_stage/stage2/val.jsonl \
  --output-dir outputs/eval/stage2_keypoints/run_001
```

## 6. LoRA 合并导出

脚本：`scripts/merge_lora.py`

```bash
python scripts/merge_lora.py \
  --dense-model models/Qwen3-VL-4B-Instruct \
  --lora-adapter outputs/qwen3vl-s1-lora/4b/your-run/checkpoints/best \
  --output-dir outputs/merged/your-run
```

## 7. 部署 Bundle 导出

脚本：`scripts/export_deployment_bundle.py`

```bash
python scripts/export_deployment_bundle.py \
  --base-source-dir models/Qwen3-VL-4B-Instruct \
  --adapter grounding/arrow outputs/qwen3vl-s1-lora/4b/your-s1-run/checkpoints/best \
  --adapter keypoint_sequence/arrow outputs/qwen3vl-s2-lora/4b/your-s2-run/checkpoints/best \
  --output-dir deployment \
  --overwrite
```

输出：

- `base_model/`
- `adapters/<route>/`
- `manifests/adapters.json`
