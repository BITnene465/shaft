# Tool Scripts

本文件统一维护非主线辅助脚本的使用说明：

- 推理（infer）
- Demo
- 离线可复盘评估（eval）

主线训练入口仍然是 `scripts/train.py`，训练相关说明请看主 README。

## 1. One-Stage 推理

脚本：

- `scripts/arrow/infer.py`

单图推理示例：

```bash
python scripts/arrow/infer.py \
  --config configs/infer/infer_one_stage.yaml \
  --checkpoint outputs/your_experiment/checkpoints/best \
  --image /path/to/figure.jpg
```

目录批量推理示例：

```bash
python scripts/arrow/infer.py \
  --config configs/infer/infer_one_stage.yaml \
  --checkpoint outputs/your_experiment/checkpoints/best \
  --image-dir /path/to/images \
  --recursive \
  --output-dir outputs/infer_one_stage_batch
```

常用覆盖参数：

- `--max-new-tokens`
- `--batch-size`（目录模式；默认读 infer config 的 `batch_size`）
- `--model`
- `--device`
- `--env-file`

输出产物（`--output-dir`）：

- `reports/*.one_stage.json`
- `raw/*.raw.txt`
- `manifest.json`（目录模式）

## 2. Two-Stage 推理

脚本：

- `scripts/arrow/infer_two_stage.py`

完整两阶段推理示例：

```bash
python scripts/arrow/infer_two_stage.py \
  --config configs/infer/infer_two_stage.yaml \
  --stage1-checkpoint outputs/qwen3vl-s1-lora/4b/checkpoints/best \
  --stage2-checkpoint outputs/qwen3vl-s2-lora/4b/checkpoints/best \
  --image /path/to/example.png \
  --output-dir outputs/two_stage_demo
```

仅 Stage1 检查模式（不传 Stage2 checkpoint）：

```bash
python scripts/arrow/infer_two_stage.py \
  --config configs/infer/infer_two_stage.yaml \
  --stage1-checkpoint outputs/qwen3vl-s1-lora/4b/checkpoints/best \
  --image /path/to/example.png \
  --output-dir outputs/two_stage_stage1_only
```

常用覆盖参数：

- `--stage1-max-new-tokens`
- `--stage1-batch-size`
- `--stage2-max-new-tokens`
- `--stage2-batch-size`
- `--stage1-model`
- `--stage2-model`
- `--device`
- `--env-file`

输出产物（`--output-dir`）：

- `reports/*.two_stage.json`
- `stage1_overlay/*.png`
- `final_overlay/*.png`
- `manifest.json`（目录模式）

## 3. Demo

One-stage Demo：

- `app/demo.py`

```bash
python app/demo.py \
  --config configs/infer/infer_one_stage.yaml \
  --checkpoint outputs/your_experiment/checkpoints/best
```

Two-stage Demo：

- `app/demo_two_stage.py`

```bash
python app/demo_two_stage.py \
  --config configs/infer/infer_two_stage.yaml \
  --stage1-checkpoint outputs/qwen3vl-s1-lora/4b/checkpoints/best \
  --stage2-checkpoint outputs/qwen3vl-s2-lora/4b/checkpoints/best
```

## 4. Stage1 Grounding 离线可复盘评估

脚本：

- `scripts/arrow/eval_stage1_grounding.py`

用途：

- 在指定 JSONL 上离线评估 Stage1 grounding checkpoint
- 产出可复盘的逐样本结果和 badcase 文件

示例：

```bash
python scripts/arrow/eval_stage1_grounding.py \
  --config configs/infer/infer_one_stage.yaml \
  --checkpoint outputs/qwen3vl-s1-lora/4b/checkpoints/best \
  --jsonl data/two_stage/stage1/val.jsonl \
  --output-dir outputs/eval/stage1_grounding/run_001
```

只给 JSONL 时，图片定位规则：

- `image_path` 为绝对路径：直接使用
- `image_path` 为相对路径：先按工作目录解析，再按 JSONL 所在目录解析

关键参数：

- `--bbox-iou-threshold`
- `--max-samples`
- `--max-new-tokens`
- `--save-per-sample / --no-save-per-sample`
- `--save-badcases-topk`

输出产物：

- `summary.json`
- `per_sample.jsonl`
- `badcases_parse.jsonl`
- `badcases_metric.jsonl`

## 5. Stage2 Keypoints 离线可复盘评估

脚本：

- `scripts/arrow/eval_stage2_keypoints.py`

用途：

- 在指定 Stage2 JSONL 上离线评估 `keypoint_sequence` checkpoint
- 产出可复盘的逐样本结果和 badcase 文件

示例：

```bash
python scripts/arrow/eval_stage2_keypoints.py \
  --config configs/infer/infer_one_stage.yaml \
  --checkpoint outputs/qwen3vl-s2-lora/4b/checkpoints/best \
  --jsonl data/two_stage/stage2/val.jsonl \
  --output-dir outputs/eval/stage2_keypoints/run_001
```

只给 JSONL 时，图片定位规则：

- `image_path` 为绝对路径：直接使用
- `image_path` 为相对路径：先按工作目录解析，再按 JSONL 所在目录解析

关键参数：

- `--strict-point-distance-px`
- `--max-samples`
- `--max-new-tokens`
- `--save-per-sample / --no-save-per-sample`
- `--save-badcases-topk`

输出产物：

- `summary.json`
- `per_sample.jsonl`
- `badcases_parse.jsonl`
- `badcases_metric.jsonl`

## 6. 配置说明

推理配置与训练 YAML 分离：

- `configs/infer/infer_one_stage.yaml`
- `configs/infer/infer_two_stage.yaml`

checkpoint 路径和模型覆盖参数都通过 CLI 传入。

其中：

- one-stage 目录推理会按 `batch_size` 分批调用模型
- two-stage 目录推理支持跨图两阶段 batch 编排：
  - Stage1 整图 grounding 按 `stage1.batch_size` 或 `--stage1-batch-size`
  - Stage2 会把这一批图产生的 crop request 合并后再按 `stage2.batch_size` 或 `--stage2-batch-size` 推理

## 7. LoRA 合并导出（Merge）

脚本：

- `scripts/merge_lora.py`

用途：

- 将 LoRA 训练 checkpoint 合并为可直接加载的完整模型权重目录。
- 额外可导出 `ft_checkpoint/`，用于需要 FT 形式目录的部署或存档。
- 建议在运行时量化前先做 merge，先验证 merged FP/BF16 基线，再单独量化 stage1/stage2。

示例：

```bash
python scripts/merge_lora.py \
  --checkpoint-dir outputs/qwen3vl-s1-lora/4b/exp6-syncAug2-weighted/checkpoints/best \
  --output-dir outputs/merged/qwen3vl-s1-lora-4b-exp6-syncAug2-weighted
```

常用参数：

- `--config`：当 checkpoint 的 `meta.json` 不含完整 config 时提供。
- `--prefer-checkpoint-meta / --no-prefer-checkpoint-meta`
- `--device`
- `--safe-serialization / --no-safe-serialization`
- `--export-ft-checkpoint / --no-export-ft-checkpoint`

默认行为：

- 默认不导出 `ft_checkpoint/`
- 如需导出 FT checkpoint，显式加：`--export-ft-checkpoint`

输出产物（`--output-dir`）：

- merged 模型权重（`save_pretrained`）
- tokenizer 与 processor 文件
- `merge_meta.json`

可选产物（显式开启参数后导出）：

- `ft_checkpoint/`（full-ft checkpoint bundle）

## 8. 部署 Bundle 导出

脚本：

- `scripts/export_deployment_bundle.py`

用途：

- 将一个共享 base model 和多个 LoRA adapter 转成部署目录
- 输出 `base_model/`、`adapters/` 和 `manifests/adapters.json`

示例：

```bash
python scripts/export_deployment_bundle.py \
  --base-source-dir outputs/qwen3vl-s1-lora/4b/ufv-exp3/checkpoints/best \
  --adapter grounding/arrow outputs/qwen3vl-s1-lora/4b/ufv-exp3/checkpoints/best \
  --adapter keypoint_sequence/arrow outputs/qwen3vl-s2-lora/4b/ufv-exp3/checkpoints/best \
  --output-dir deployment \
  --overwrite
```

输出产物：

- `base_model/`
- `adapters/<route>/`
- `manifests/adapters.json`

## 9. 旧权重一次性转换

脚本：

- `scripts/convert_legacy_checkpoint.py`

用途：

- 将旧的 full-state checkpoint 一次性转换成当前 LoRA checkpoint 布局
- 仅用于历史权重迁移，不作为运行时兼容入口

示例：

```bash
python scripts/convert_legacy_checkpoint.py \
  --config configs/train/train_stage1_lora_2b.yaml \
  --source-checkpoint outputs/old_experiment/checkpoints/best \
  --output-dir outputs/converted/old_experiment-best \
  --overwrite
```
