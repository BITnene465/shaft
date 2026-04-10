# 标准数据格式（JSONL）

仓库内长期维护的训练/验证数据格式为标准 JSONL。

路由（`task_type/domain_type`）可由以下任一方式提供：

- 配置绑定：`data.train_route_map` / `data.val_route_map`（推荐）
- 样本字段：JSONL 中显式提供 `task_type` 与 `domain_type`

框架不会隐式猜测 route。

## 1. 通用字段

每条记录至少包含：

| 字段 | 类型 | 必需 | 说明 |
|---|---|---|---|
| `sample_id` | `str` | 是 | 样本唯一 ID |
| `image_path` | `str` | 是 | 图片路径（相对或绝对） |
| `image_width` | `int` | 是 | 图像宽 |
| `image_height` | `int` | 是 | 图像高 |
| `task_type` | `str` | 否 | 配置未绑定 route 时需提供 |
| `domain_type` | `str` | 否 | 配置未绑定 route 时需提供 |

## 2. 业务约束

- `label` 仅允许：
  - `single_arrow`
  - `double_arrow`
- 量化坐标为 `[0, 999]` 的整数。
- `bbox_2d = [x1,y1,x2,y2]`，且 `x1 < x2`、`y1 < y2`。
- `keypoints_2d` 为 `[[x,y], ...]`。

### 2.1 点序约束

- `single_arrow`：`tail -> ... -> head`
- `double_arrow`：两端是两个 head，且左上 head 在前（`x` 小优先，`x` 相同则 `y` 小优先）

### 2.2 实例排序约束

- `joint_structure` 排序键：
  - `(y1, x1, y2, x2, tail_y, tail_x, head_y, head_x, n_points)`
- `grounding` 排序键：
  - `(y1, x1, y2, x2, label)`

排序必须在数据准备阶段固化，dataset 只读取，不做二次重排。

## 3. 单阶段：`joint_structure`

路径：

- `data/processed/train.jsonl`
- `data/processed/val.jsonl`

记录示例：

```json
{
  "sample_id": "img_001",
  "image_path": "data/processed/images/train/img_001.jpg",
  "image_width": 1920,
  "image_height": 1080,
  "task_type": "joint_structure",
  "domain_type": "arrow",
  "instances": [
    {
      "label": "single_arrow",
      "bbox_2d": [100, 200, 300, 400],
      "keypoints_2d": [[120, 250], [200, 300], [280, 350]]
    }
  ]
}
```

模型输出协议：

```json
[
  {
    "label": "single_arrow",
    "bbox_2d": [100, 200, 300, 400],
    "keypoints_2d": [[120, 250], [200, 300], [280, 350]]
  }
]
```

## 4. 两阶段 Stage1：`grounding`

路径：

- `data/two_stage/stage1/train.jsonl`
- `data/two_stage/stage1/val.jsonl`

记录示例：

```json
{
  "sample_id": "img_001_full",
  "image_path": "data/two_stage/stage1/images/train/img_001_full.jpg",
  "image_width": 1920,
  "image_height": 1080,
  "task_type": "grounding",
  "domain_type": "arrow",
  "instances": [
    {
      "label": "single_arrow",
      "bbox_2d": [100, 200, 300, 400]
    }
  ]
}
```

模型输出协议：

```json
[
  {
    "label": "single_arrow",
    "bbox_2d": [100, 200, 300, 400]
  }
]
```

补充：

- Stage1 含整图样本与 tile/crop 样本。
- 仅保留 bbox 完整落入 crop 的实例。
- 与任意 bbox 部分相交的 crop 直接丢弃。

## 5. 两阶段 Stage2：`keypoint_sequence`

路径：

- `data/two_stage/stage2/train.jsonl`
- `data/two_stage/stage2/val.jsonl`

记录示例：

```json
{
  "sample_id": "img_001_inst_0__pad300",
  "image_path": "data/two_stage/stage2/images/train/img_001_inst_0__pad300.png",
  "image_width": 512,
  "image_height": 512,
  "task_type": "keypoint_sequence",
  "domain_type": "arrow",
  "crop_box": [50, 80, 400, 420],
  "padding_ratio": 0.3,
  "gt_struct": {
    "label": "single_arrow",
    "keypoints_2d": [[80, 120], [200, 250], [380, 400]]
  }
}
```

模型输出协议：

```json
{
  "keypoints_2d": [[80, 120], [200, 250], [380, 400]]
}
```

补充：

- Stage2 每条记录对应一个目标箭头。
- 训练可多 padding 视图；验证通常固定 `padding_ratio=0.3`。
- `gt_struct` 为数据真值，`target_text` 在加载阶段由 codec 动态生成。

## 6. 数据准备命令

真实数据：

```bash
python scripts/arrow/prepare_data.py \
  --raw-json-dir data/raw/json \
  --image-dir data/raw/figure \
  --output-dir data/processed
```

Stage1：

```bash
python scripts/arrow/prepare_stage1_data.py \
  --input-dir data/processed \
  --output-dir data/two_stage
```

Stage2：

```bash
python scripts/arrow/prepare_stage2_data.py \
  --input-dir data/processed \
  --output-dir data/two_stage \
  --train-padding-ratios 0.2,0.3,0.45 \
  --val-padding-ratio 0.3
```

## 7. LabelMe 映射

| LabelMe 矩形类别 | 箭头类别 |
|---|---|
| `c0`,`c1`,`c2`,`c3` | `single_arrow` |
| `c4`,`c5`,`c6`,`c7` | `double_arrow` |
