# Banana v5.7 数据与训练合同

本文是 Banana v5.7 的当前数据、prompt 和训练配置真源。数据文件位于忽略版本控制的 `data/`，可复现的
catalog、训练 YAML、prompt pool、构建脚本和合同测试必须进入仓库。

## 1. 正式训练范围

当前 v5.7 mix 只包含以下五个 train-only 数据源：

| dataset | 行数 | 数据域 | prompt |
| --- | ---: | --- | --- |
| `grounding_layout` | 58,440 | 真实 compact raw 的多分辨率检测 | `grounding_layout.v5.7.yaml` |
| `shape_context_reconstruction` | 300,000 | v9 synthetic `gt_standard` | `shape_context_reconstruction.v5.7.yaml` |
| `line_context_reconstruction` | 300,000 | v9 synthetic `gt_standard` | `line_context_reconstruction.v5.7.yaml` |
| `line_context_points` | 137,218 | 122,218 真实 line path + 15,000 synthetic 多分支 | `line_context_points.v5.7.yaml` |
| `image_context_reconstruction` | 21,184 | 真实 reviewed 13 类 image type | `image_context_reconstruction.v5.3.yaml` |

`image_context_reconstruction` 的业务字段没有随 v5.7 改版，因此有意继续使用 v5.3 prompt。其余四个数据源
必须使用 v5.7 pool。所有 SFT 行的 `system_prompt` / `user_prompt` 为空，由运行时 prompt sampling 注入。

当前正式 mix 不包含：

- `grounding_layout_sync`：可选 synthetic detection replay，当前未物化，也未登记到 v5.7 catalog。
- `background`：不属于本轮目标。
- `shape_context_attributes`：历史 API 弱标签，不属于 v5.7。
- 任何旧的 `grounding_arrow`、`point_arrow` 或 region-reconstruction 数据。

所有五个源均为 train-only；validation JSONL 当前为空，catalog 设置 `use_for_eval: false`，训练配置设置
`eval.enabled: false`。这不是缺文件，而是当前训练合同。

## 2. 可复现配置

数据与 prompt 真源：

- `configs/data/banana_v5_7.yaml`
- `configs/prompts/pools/grounding_layout.v5.7.yaml`
- `configs/prompts/pools/shape_context_reconstruction.v5.7.yaml`
- `configs/prompts/pools/line_context_reconstruction.v5.7.yaml`
- `configs/prompts/pools/line_context_points.v5.7.yaml`
- `configs/prompts/pools/image_context_reconstruction.v5.3.yaml`

训练 YAML：

| 配置 | 用途 |
| --- | --- |
| `banana_sft_4b_v5_7.yaml` | Qwen3-VL-4B full、DDP、token-budget bounded-cost |
| `banana_sft_4b_v5_7_trial.yaml` | 4B vision-tower LR 对照实验 |
| `banana_sft_27b_qwen36_v5_7_full_zero3.yaml` | Qwen3.6-27B full、DeepSpeed ZeRO-3 |
| `banana_sft_27b_qwen36_v5_7_lora.yaml` | Qwen3.6-27B LoRA、FSDP full-shard |
| `banana_sft_27b_qwen36_v5_7_qlora.yaml` | Qwen3.6-27B QLoRA、DDP |
| `banana_sft_27b_qwen36_v5_7_re_full_zero3.yaml` | 从既有 v5.7 checkpoint 初始化的新 schedule |

`*_re_*` 不是 exact resume；它要求配置中的 `init_from_checkpoint` 已存在。不存在该 checkpoint 时，基础
v5.7 数据和配置仍完整，但这份派生运行配置不能直接启动。

## 3. 数据真源与转换链

### 3.1 真实 grounding

```text
data/raw/json + data/raw/images
  └─ data/raw/splits/grounding_layout.train.txt
      └─ build_grounding_structured.py
          └─ data/grounding_layout/structured + task-local images
              └─ build_sft_from_structured.py
                  └─ data/grounding_layout/sft
```

active raw 是 compact `size + layout[]` 合同。grounding 只读取
`shape/icon/image/line` 的 `type + bbox`，排除 `full_text`，不读取 line points 或 reconstruction
parameters。每个 source 保留一个 native clean full-image 行；train-only 增强包括连续 clean resize、少量随机
padding、单一 blur/noise degraded resize、density crop 和受控 hard negative。validation/test 不做增强。

### 3.2 Synthetic shape/line reconstruction

```text
data/regulated_layout_dataset_v9_20260802/{gt_standard,img,train.txt,val.txt}
  └─ prepare_gt_standard_v5_7.py
      └─ source-identity selection manifests
          └─ build_context_reconstruction_sft.py
              ├─ shape_context_reconstruction
              └─ line_context_reconstruction
```

`gt_standard` 是属性与几何真源；selection 只选择 source/instance，不保存平行 target 真源。每个 synthetic
crop 必须应用 `synthetic_realism_v1` 的 1–3 个尺寸不变操作；极小目标只用一个 mild 操作。proposal bbox
与 target geometry 共用整个 contextual crop 的 Qwen `0..999` 坐标系。

### 3.3 Line points

`prepare_real_line_context_points.py` 从 active compact raw 中选择所有非空、保持顺序的
`parameters.points`，不采样、不补造空 points。`build_context_reconstruction_sft.py` 将 122,218 条真实路径
与最多 15,000 条 v9 synthetic 多分支路径合并：真实 crop 保持 clean，synthetic crop 使用
`synthetic_realism_v1`。target 严格只有 `is_single + points`。

### 3.4 Image type

`image_context_reconstruction` 从 reviewed raw 重新读取 bbox 与 13 类 `image_type`，生成一张 contextual
crop。它不使用 synthetic pixel degradation，也不扩展到 v9 的 `image_type=N/A`。

## 4. 当前完整性基线

2026-08-14 全量核验结果：

- active compact raw：20,060 JSON；train 20,060，val 0，与 `vlm.test.json` 无交叉。
- raw label：`full_text=1,159,672`、`shape=415,333`、`icon=216,703`、`image=54,735`、
  `line=345,010`。
- 20,060 个 native grounding structured 行与 raw 四类 bbox 多重集逐项一致。
- 真实非空 line path：122,218；single 112,350，multi-segment 9,868。
- v9：100,000 train、500 val，无交叉；所有 train JSON/图片可读且尺寸一致。
- v9 reconstruction audit 会排除少量实例级无效项：4 个 shape bbox、992 个 line bbox、2,428 个
  curved-line point 合同错误；它们不进入 selection，但原始 source truth 保留用于审计。
- 五个数据集的 structured/SFT 行一一对应，sample id 与媒体路径唯一，所有媒体存在；SFT target 可从
  structured 精确重算，prompt args 与 JSON 合同全部通过。

raw 中有 3 组 `line` 共享同 label/bbox，但 points 路径不同，分别表示交叉或反向线路，不能按 bbox 去重。

## 5. 构建顺序

先审计并生成 v9 selection：

```bash
uv run python scripts/tasks/prepare_gt_standard_v5_7.py \
  --dataset-root data/regulated_layout_dataset_v9_20260802 \
  --output-root data/reconstruction_v5_7_selection \
  --workers 8
```

重建真实 grounding，再转换 SFT：

```bash
uv run python scripts/tasks/build_grounding_structured.py \
  --raw-root data/raw \
  --output-root data \
  --train-split data/raw/splits/grounding_layout.train.txt \
  --val-split data/raw/splits/grounding_layout.val.txt \
  --task grounding_layout --workers 8 --clean

uv run python scripts/tasks/build_sft_from_structured.py \
  --data-root data --task grounding_layout --workers 8 --clean
```

重建 synthetic shape/line 与真实 image：

```bash
uv run python scripts/tasks/build_context_reconstruction_sft.py \
  --synthetic-root data/regulated_layout_dataset_v9_20260802 \
  --shape-selection data/reconstruction_v5_7_selection/shape/train.jsonl \
  --line-selection data/reconstruction_v5_7_selection/line/train.jsonl \
  --tasks shape_context_reconstruction line_context_reconstruction \
    image_context_reconstruction \
  --workers 8 --clean
```

首次构建 line points 时先生成真实 selection，再显式合并 synthetic 多分支 selection：

```bash
uv run python scripts/tasks/prepare_real_line_context_points.py --workers 8 --clean

uv run python scripts/tasks/build_context_reconstruction_sft.py \
  --synthetic-root data/regulated_layout_dataset_v9_20260802 \
  --line-point-real-selection \
    data/reconstruction_v5_7_selection/line_points_real/train.jsonl \
  --line-point-synthetic-selection \
    data/reconstruction_v5_7_selection/line_points/train.jsonl \
  --tasks line_context_points --workers 8 --clean
```

所有 rebuild 都必须写入 staging/新目录或显式使用 `--clean`，完成后检查 train/val 交叉、JSONL/媒体
一一对应、未引用图片、task target schema、prompt version 和运行时 catalog/pool 映射。

## 6. 启动前检查

```bash
uv run python -c \
  'from shaft.config import load_config; load_config("configs/train/banana_sft_4b_v5_7.yaml")'
```

正式启动前还必须确认模型目录、外部数据快照、CUDA/FlashAttention、world size 和目标分布式后端满足对应
训练 YAML 的运行条件。配置可解析只证明 schema 和路径合同成立，不替代 GPU canary 或长程收敛验收。
