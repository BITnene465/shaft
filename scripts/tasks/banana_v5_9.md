# Banana v5.9 数据增量

v5.9 是 v5.8 的 grounding-only 增量。除 `grounding_layout` 外，catalog 中所有数据源、PromptSource
和采样权重继续复用 v5.8；新增标注中的 line points 保留在 raw JSON 中，但不激活为 reconstruction 数据。

## 来源和隔离

- 基线：`data/raw/splits/grounding_layout.train.txt` 冻结的 19,745 个 v5.8 train ID。
- 增量：`json_v5.9_new` 中 3,519 份 compact `size + layout` 标注及 `data/paper` 原图。
- 保护集：`real_v1` 与 `real_v2` 仅按 image stem 排除；不以内容哈希扩大测试排除范围。
- 质量门禁：低标注量、非法 target bbox、完全重复 instance，以及新增批次内部的精确图像冲突组。
- 原始图像不 resize、不改写；派生增强媒体只写入 `data/banana_v5_9/grounding_layout/images`。

## 构建

```bash
uv run --no-sync python scripts/tasks/prepare_banana_v5_9_grounding.py \
  --base-raw-root data/raw \
  --base-train-split data/raw/splits/grounding_layout.train.txt \
  --incoming-json-dir /path/to/json_v5.9_new \
  --incoming-image-root /path/to/data/paper \
  --real-v1-image-dir subTasks/layout_recognition/data/real_v1/img \
  --real-v2-image-dir subTasks/layout_recognition/data/real_v2/img \
  --output-root data/banana_v5_9/raw \
  --workers 40 \
  --clean

uv run --no-sync python scripts/tasks/build_grounding_structured.py \
  --raw-root data/banana_v5_9/raw \
  --output-root data/banana_v5_9 \
  --train-split data/banana_v5_9/raw/splits/grounding_layout.train.txt \
  --val-split data/banana_v5_9/raw/splits/grounding_layout.val.txt \
  --task grounding_layout \
  --workers 40 \
  --seed 42 \
  --augmentation-profile layout_multiscale_v1 \
  --clean

uv run --no-sync python scripts/tasks/build_sft_from_structured.py \
  --data-root data/banana_v5_9 \
  --task grounding_layout \
  --prompt-config grounding_layout=configs/prompts/pools/grounding_layout.v5.8.yaml \
  --prompt-variant grounding_layout=detailed \
  --workers 40 \
  --clean
```

`layout_multiscale_v1` 与 v5.8 保持一致，包含 clean resize、degraded resize、padded full、density
crop 和少量 hard-negative。数据增强 seed 保持 42，以复现 v5.8 基线样本的视图选择；训练 seed 仍独立使用
465。

训练 catalog 为 `configs/data/banana_v5_9.yaml`。它只将 `grounding_layout` 路径切换到 v5.9，其余
五个数据集继续使用 v5.8 产物和原权重。

## 已冻结结果

- source：19,745 个 v5.8 基线 + 3,306 个新增 = 23,051；新增排除 23 个质量问题和 190 个冲突重复图。
- test gate：real_v1 175 ID、real_v2 250 ID，train ID overlap 为 0。
- structured/SFT：各 67,195 行、67,195 个唯一 sample ID、67,195 个有效媒体引用。
- media：67,195 张派生媒体全部通过并行解码复验，错误为 0。
- views：full 23,051、clean resize 20,571、degraded resize 17,218、padded 2,296、density crop
  3,444、hard negative 615。
