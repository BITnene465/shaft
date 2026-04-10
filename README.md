# ArrowVLM（结构化多模态生成框架）

本仓库用于在 `Qwen3-VL` 上训练结构化视觉任务模型，当前主线是 `arrow` 域的三类任务：

- `joint_structure`（单阶段）：输出 `label + bbox_2d + keypoints_2d`
- `grounding`（两阶段 Stage1）：输出 `label + bbox_2d`
- `keypoint_sequence`（两阶段 Stage2）：输出 `keypoints_2d`

当前 Python 包名为 `vlm_structgen`（不再使用 `vlm_det`）。

## 1. 环境准备

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
```

如需显式安装 CUDA 版 PyTorch：

```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
uv pip install -e . --no-deps
```

## 2. 数据准备

### 2.1 真实标注转标准 JSONL

```bash
python scripts/arrow/prepare_data.py \
  --raw-json-dir data/raw/json \
  --image-dir data/raw/figure \
  --output-dir data/processed
```

产物：

- `data/processed/train.jsonl`
- `data/processed/val.jsonl`
- `data/processed/reports/data_cleaning_report.json`
- `data/processed/reports/split_manifest.json`

### 2.2 两阶段数据

Stage1（整图 + 滑窗 + density crop）：

```bash
python scripts/arrow/prepare_stage1_data.py \
  --input-dir data/processed \
  --output-dir data/two_stage \
  --num-workers 8 \
  --stage1-include-full-image \
  --stage1-tile-size-ratios 0.35,0.5 \
  --stage1-min-tile-size 512 \
  --stage1-max-tile-size 1280 \
  --stage1-density-min-instances 5 \
  --stage1-density-max-instances 30 \
  --stage1-dedup-iou-threshold 0.9
```

Stage2（单目标 crop，多 padding 训练视图）：

```bash
python scripts/arrow/prepare_stage2_data.py \
  --input-dir data/processed \
  --output-dir data/two_stage \
  --train-padding-ratios 0.2,0.3,0.45 \
  --val-padding-ratio 0.3 \
  --num-workers 8
```

## 3. 训练

主入口：

```bash
python scripts/train.py --config <train_config.yaml>
```

常用配置（当前仅维护 4B）：

```bash
python scripts/train.py --config configs/train/train_lora_4b.yaml
python scripts/train.py --config configs/train/train_full_ft_4b.yaml
python scripts/train.py --config configs/train/train_stage1_lora_4b.yaml
python scripts/train.py --config configs/train/train_stage2_lora_4b.yaml
python scripts/train.py --config configs/train/train_mixed_full_ft_4b.yaml
```

训练数据源默认由注册表管理：

- `configs/data_registry/arrow.yaml`

可选 CLI 覆写（无歧义参数）：

- `--run-id`
- `--stage-name`
- `--seed`
- `--epochs`
- `--lr`
- `--mix-strategy`（`concat|interleave_under|interleave_over`）
- `--init-from`
- `--resume-from`

多卡示例：

```bash
torchrun --nproc_per_node=2 scripts/train.py --config configs/train/train_mixed_full_ft_4b.yaml
```

## 4. 路由与混训约定

- 路由使用 `route` 字段（如 `grounding/arrow`）。
- `core` 通过 `route -> adapter` 的中间层注册表路由，不直接在训练链路里依赖 `task_type/domain_type`。
- 推荐使用数据集注册表模式：
  - `data.registry_path`
  - `data.train_datasets`
  - `data.val_datasets`
- JSONL 推荐显式写 `route`；`task_type/domain_type` 仅作为兼容兜底，不建议继续新增。
- 当前混训为样本级路由（同一 batch 可混合多个 route）。

## 5. 推理与评估

在线推理/服务编排属于外部系统职责。本仓库仅保留离线实验脚本与评估脚本，供训练阶段复盘使用。

辅助脚本统一文档见：

- [docs/tool_scripts.md](/home/tanjingyuan/code/arrow-vlm/docs/tool_scripts.md)

包含：

- 单阶段推理
- 两阶段推理
- Demo
- Stage1/Stage2 离线可复盘评估
- LoRA merge 与部署 bundle 导出

## 6. 协议与边界

- 训练目标是标准语言模型 `next-token prediction`。
- 结构化监督由 `codec` 提供：`target_text + loss_meta`。
- `trainer` 不解析业务字段，不做任务语义分支。
- 数据顺序（canonical order）在数据准备阶段固化。

## 7. 文档索引

- 架构： [docs/architecture.md](/home/tanjingyuan/code/arrow-vlm/docs/architecture.md)
- 标准数据格式： [docs/standard_data_format.md](/home/tanjingyuan/code/arrow-vlm/docs/standard_data_format.md)
- 训练链路： [docs/training_pipeline.md](/home/tanjingyuan/code/arrow-vlm/docs/training_pipeline.md)
- 推理链路： [docs/inference_pipeline.md](/home/tanjingyuan/code/arrow-vlm/docs/inference_pipeline.md)
- 编码与加权损失： [docs/codec_and_loss.md](/home/tanjingyuan/code/arrow-vlm/docs/codec_and_loss.md)
- 任务/域扩展： [docs/adding_task_domain.md](/home/tanjingyuan/code/arrow-vlm/docs/adding_task_domain.md)
- 多任务规范草案： [docs/dev/multitask_qwen3vl_framework_spec.md](/home/tanjingyuan/code/arrow-vlm/docs/dev/multitask_qwen3vl_framework_spec.md)
- 训练框架边界规范： [docs/dev/training_framework_boundary_spec.md](/home/tanjingyuan/code/arrow-vlm/docs/dev/training_framework_boundary_spec.md)
- 训练产物协议规范： [docs/dev/protocol_artifact_spec.md](/home/tanjingyuan/code/arrow-vlm/docs/dev/protocol_artifact_spec.md)

## 8. 许可证

MIT，见 [LICENSE](/home/tanjingyuan/code/arrow-vlm/LICENSE)。
