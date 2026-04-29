---
name: shaft-grounding-layout-data-aug
description: 将 layout icon/image 标注整理为 grounding structured 数据，并按大 crop、无截断 GT、受控 hard negative 与轻成像增强规则重建数据集。
---

# Skill：grounding layout 数据增强整理

## 触发场景

- 需要把 layout 标注整理为 grounding structured 数据。
- 输入通常是 `data/raw_layout/json` 与 `data/raw_layout/images`，标签为 `icon` / `image` 的 rectangle。
- 需要重建 `data/grounding_layout/structured` 与 `data/grounding_layout/images`。
- 需要复用已经验证过的 layout 增强策略，而不是沿用 arrow 的更激进 crop 配置。

## 用户需要先明确的信息

- 输入标注目录与图片目录。
- split 来源；默认使用 `data/raw_layout/splits/train.txt` 和 `val.txt`。
- 输出目录；默认覆盖 `data/grounding_layout/structured` 与 `data/grounding_layout/images`。
- 是否只生成 structured；默认不生成 SFT。
- 是否允许覆盖旧 structured/images 产物；不要覆盖 raw 数据。

## 硬性约束

### 任务边界

- 当前方法面向 `grounding` + `layout`。
- 主字段 `instances` 只保留 `label + bbox`。
- 原始 rectangle points、crop box、原图尺寸和源实例索引放到 `extra`。
- 只保留 `icon` 与 `image` 标签。

### split

- split 先定，再增强。
- val 默认 300 张 full image；剩余为 train。
- 若 `data/raw_layout/splits` 已存在，直接使用现有 split。
- val 只保留 `full_image`，不做 crop、不做 hard negative、不做 blur/jpeg。

### crop 完整性

对每个 crop candidate，逐个 GT bbox 判断：

1. GT 完整落在 crop 内：保留该 GT，并把 bbox 平移到 crop 坐标系。
2. GT 与 crop 完全没有交集：忽略该 GT。
3. GT 与 crop 有交集但没有完整落在 crop 内：丢弃整个 crop candidate。

禁止把被截断的 GT clip 到 crop 边界后继续训练。hard negative 也必须没有任何 partial overlap。

### train 视图

- 保留所有 `full_image`。
- 生成大窗口 `density_*` crop。
- 生成限量大窗口 `sliding_*` crop。
- 生成受控 `hard_negative_*` crop。
- crop 尺寸必须足够大，默认使用 `[896, 1024, 1152, 1280]`。
- 不生成小于 896 的局部 crop，除非用户明确改策略。

### density crop

- 面向 icon/image 密集区域。
- 默认最少 2 个完整实例。
- 每张源图限量，避免密集区域近重复污染训练分布。
- 对实例集合相同且 crop 高重叠的 candidate 去重。

### sliding crop

- 只做补充，不作为主分布。
- 默认最少 4 个完整实例。
- 每张源图限量，优先保留实例更多、目标面积更大的大窗口。
- 没有完整实例且无 partial overlap 的滑窗可以进入 hard negative 候选池。

### hard negative

- 只从干净空窗口中采样：无完整 GT，且无 partial overlap。
- 空样本比例必须受控，推荐最终 train empty ratio 为 3%-5%，默认 4%。
- hard negative 不做 blur/jpeg。

### 轻成像增强

- 只对 train 的 positive crop 做。
- 每个 positive crop 最多生成一个增强版本。
- `jpeg_compression` 与 `light_blur` 二选一，不同时生成。
- full image 不做增强，hard negative 不做增强。
- 建议轻量参数：
  - JPEG quality: `72, 80, 88`
  - blur radius: `0.35, 0.5, 0.7`
- 不做 flip、rotate、perspective、强 color jitter。

## 输出 schema

Structured row 必须保持：

```json
{
  "task_type": "grounding",
  "domain_type": "layout",
  "sample_id": "web_0001__full",
  "source_sample_id": "web_0001",
  "source_type": "full_image",
  "image_path": "../images/train/web_0001__full.png",
  "image_width": 2816,
  "image_height": 1536,
  "instances": [{"label": "icon", "bbox": [x1, y1, x2, y2]}],
  "extra": {
    "view_type": "identity",
    "crop_box": [0, 0, 2816, 1536],
    "original_image_width": 2816,
    "original_image_height": 1536,
    "full_instances": [{"label": "icon", "bbox": [x1, y1, x2, y2], "points": [[x1, y1], [x2, y2]], "shape_type": "rectangle"}],
    "source_instance_indices": [0]
  }
}
```

增强样本在 `extra.augmentation` 中记录：

```json
{"name": "jpeg_compression", "quality": 80}
```

或：

```json
{"name": "light_blur", "radius": 0.5}
```

## 执行流程

1. 读取 raw JSON 与图片，过滤出 `icon/image` rectangle。
2. 读取或创建 split；val 300，train 为剩余样本。
3. 如果用户要求覆盖，清理旧的 `data/grounding_layout/structured` 与 `data/grounding_layout/images`；不要动 raw，也不要动 SFT，除非用户明确要求。
4. 多进程生成 train/val full image。
5. 对 train 生成 density/sliding 大 crop，严格执行完整 GT 过滤和去重。
6. 从干净空滑窗中全局抽样 hard negative，按最终 empty ratio 控制在 3%-5%。
7. 对每个 positive crop 生成最多一个轻成像增强。
8. 写出 `structured/train.jsonl`、`structured/val.jsonl` 与 images。
9. 写 `structured/README.md`，记录统计与参数。
10. 不要留下临时生成脚本。若需要长期复用脚本，必须作为正式 `scripts/tasks/*.py` 入口另行实现。

## 验收

必须至少校验：

- train/val JSONL 行数与对应 images 文件数一致。
- val 只有 `full_image`。
- 所有 crop 尺寸不小于策略下限。
- 所有 crop 都不存在 partial GT overlap。
- hard negative 的 `instances` 为空，且无完整 GT、无 partial overlap。
- train empty ratio 在目标范围内。
- 每个 positive crop 最多一个增强版本。
- 没有生成或修改 SFT，除非用户明确要求。
- 没有留下临时脚本文件。
