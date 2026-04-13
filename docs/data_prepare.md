# 数据准备说明

## 1. 真实数据清洗（标准 JSONL）

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

## 2. Stage1 数据（grounding）

Stage1 由三类样本组成：

- 原始整图
- 多尺度滑窗
- density crop

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

关键规则：

- 实例进入 crop 的条件：bbox 必须完整包含在 crop 内。
- 与任意 bbox 仅部分相交的 crop 直接丢弃。
- 近重复 crop 通过实例集合 + IoU 去重。

产物：

- `data/two_stage/stage1/train.jsonl`
- `data/two_stage/stage1/val.jsonl`
- `data/two_stage/stage1/images/train/`
- `data/two_stage/stage1/images/val/`
- `data/two_stage/reports/prepare_stage1_report.json`

## 3. Stage2 数据（keypoint_sequence）

Stage2 是单目标 crop 数据，输出目标箭头骨架：

```json
{"keypoints_2d":[[x0,y0],[x1,y1],...]}
```

```bash
python scripts/arrow/prepare_stage2_data.py \
  --input-dir data/processed \
  --output-dir data/two_stage \
  --train-padding-ratios 0.2,0.3,0.45 \
  --val-padding-ratio 0.3 \
  --num-workers 8
```

关键规则：

- 训练集可多 padding 视图；验证集固定单视图（默认 `0.3`）。
- 坐标转换到 crop-local 后再量化。
- 当前 Stage2 prompt 使用固定语义，不再注入条件字段。
- JSONL 保存结构化 GT；`target_text` 在 dataset 加载时由 codec 生成。

产物：

- `data/two_stage/stage2/train.jsonl`
- `data/two_stage/stage2/val.jsonl`
- `data/two_stage/stage2/images/train/`
- `data/two_stage/stage2/images/val/`
- `data/two_stage/reports/prepare_stage2_report.json`
