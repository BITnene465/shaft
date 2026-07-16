# Shaft 脚本使用说明

本文档说明仓库 `scripts/` 目录下的正式脚本如何使用。

范围约束：
- 只覆盖正式脚本入口
- 覆盖 `scripts/tasks/` 下的任务脚本
- **不覆盖 `scripts/tmp/`**；`tmp` 目录视为临时实验区，不属于稳定接口

## 1. 设计原则

`scripts/*.py` 的定位是**薄入口**：
- CLI 解析与命令编排放在 `src/shaft/cli`
- `scripts/*.py` 只负责调用对应 CLI 主入口

当前唯一例外是：
- `scripts/tasks/convert_grounding_structured_to_sft.py`
- `scripts/tasks/convert_grounding_structured_to_sft_row_major.py`

它属于明确的任务数据准备脚本，不是训练内核入口。

## 2. 顶层脚本

### `scripts/train.py`

用途：
- 统一训练入口
- 当前支持 `sft` 与 `rlhf` 子命令

常用形式：

```bash
python scripts/train.py sft --config configs/train/banana_sft_4b.yaml
python scripts/train.py rlhf --config configs/train/dpo_4b.yaml
```

兼容写法：

```bash
python scripts/train.py --config configs/train/banana_sft_4b.yaml
```

说明：
- 如果直接传 `--config`，当前默认走 `sft`
- 真正的命令定义在 `src/shaft/cli`

### `scripts/infer.py`

用途：
- 运行可配置的多阶段推理 pipeline

常用形式：

```bash
python scripts/infer.py \
  --config configs/infer/pipeline_smoke.yaml \
  --image path/to/image.png
```

带初始上下文：

```bash
python scripts/infer.py \
  --config configs/infer/pipeline_smoke.yaml \
  --image path/to/image.png \
  --inputs '{"document_id":"demo-001"}'
```

说明：
- `--inputs` 是 JSON 字符串
- 输出会打印为 JSON

### `scripts/export.py`

用途：
- HF 兼容导出工具
- checkpoint 布局检查
- PEFT adapter 合并

子命令：
- `inspect`
- `validate`
- `merge-peft`

示例：

```bash
python scripts/export.py inspect --path outputs/run_x/checkpoint-100
```

```bash
python scripts/export.py validate \
  --path outputs/run_x/checkpoint-100 \
  --finetune-mode lora \
  --model-type qwen3vl \
  --model-name-or-path org/model \
  --revision release-v2 \
  --cache-dir /path/to/hf-cache \
  --local-files-only true
```

```bash
python scripts/export.py merge-peft \
  --model-type qwen3vl \
  --adapter-path outputs/run_x/checkpoint-100 \
  --base-model org/model \
  --output-dir outputs/run_x/merged \
  --revision release-v2 \
  --cache-dir /path/to/hf-cache \
  --local-files-only true
```

`validate` 与 `merge-peft` 使用和训练相同的 HF locator 语义：`revision` 固定 Hub 版本，`cache-dir`
选择缓存根目录，`local-files-only=true` 禁止联网。adapter 的 base artifact 解析和模型 variant 选择均由
统一 `ResolvedModelPlan` 完成，CLI 不根据目录名猜测 dense/MoE。
`merge-peft` 默认要求 adapter 携带 Shaft 训练 metadata，并验证其中 base-model plan fingerprint。若导入的是
没有 Shaft provenance 的第三方 PEFT adapter，必须先独立核对 base，再显式增加
`--allow-unverified-base-model true`；该 escape hatch 不会关闭当前 base artifact 的完整 SHA256 校验。

### `scripts/compare_efficiency.py`

用途：比较多个已完成 run 的 committed `shaft_training_efficiency.json`，不负责启动实验。

```bash
python scripts/compare_efficiency.py outputs/fixed outputs/packed
python scripts/compare_efficiency.py --json outputs/fixed outputs/packed
```

默认要求模型 plan、数据/source、draw schedule、software/hardware、DP/GA、优化器、step span 与实际 committed
workload 一致，只允许 batch/sequence contract 作为实验轴变化。`--allow-incompatible` 仅用于明确接受非公平
条件的诊断结果，不能用于形成性能结论。


## 3. `scripts/tasks/`

`scripts/tasks/` 用于**任务级数据准备或转换脚本**。

这类脚本可以：
- 读写数据文件
- 生成训练前产物
- 服务具体业务任务

但不应：
- 承载训练内核语义
- 复制一套新的训练 CLI
- 替代 `src/shaft/cli`

### `scripts/tasks/build_grounding_structured.py`

用途：
- 从 raw bbox 标注和显式 split 生成 task-local grounding 图片与 structured JSONL
- 默认使用 `layout_multiscale_v1`，生成 native、连续多尺度、随机 padding、分级退化、density crop
  和 hard negative 视图

当前 `grounding_layout` 重建命令：

```bash
uv run python scripts/tasks/build_grounding_structured.py \
  --raw-root data/raw \
  --output-root data \
  --train-split data/raw/splits/grounding_layout.train.txt \
  --val-split data/raw/splits/grounding_layout.val.txt \
  --task grounding_layout \
  --workers 50 \
  --clean
```

默认多尺度约束：
- 目标像素在 `200704..4000000` 内按 log 空间连续采样，最终宽高按 `32` 对齐
- 离线线性放大不超过 `2x`，同源尺度像素量至少相差 `1.35x`
- clean resize、padding、degraded resize、density crop、hard negative 的目标比例分别约为
  `2.9x / 0.1x / 1.2x / 0.25x / 0.03x`
- padding 为非对称随机偏移；退化只使用单一 Gaussian blur 或 Gaussian noise
- validation/test 只保留 native clean full-image

`--augmentation-profile legacy` 只保留历史生成代码路径，不应与当前多尺度数据混合；复现旧数据时还
必须显式传入旧版 ratio 参数，不能沿用当前默认值。

### `scripts/tasks/build_grounding_layout_sync_structured.py`

用途：
- 从 `regulated_layout_dataset_v8_20260709/gt_standard` 构建独立的合成 detection 数据集
- 只读取源数据的 `train.txt`，显式排除 `val.txt`
- 每个源图只保留一条 clean full-image 视图，不做 resize、crop、blur、noise 或 padding
- 输出任务名固定为 `grounding_layout_sync`，不写入或合并到真实 `grounding_layout`

构建命令：

```bash
uv run python scripts/tasks/build_grounding_layout_sync_structured.py \
  --dataset-root data/regulated_layout_dataset_v8_20260709 \
  --output-root data/grounding_layout_sync \
  --workers 50 \
  --clean
```

关键行为：
- 读取 `gt_standard` 的 `shape/icon/image/line`，源 `arrow` 统一为 `line`
- 过滤面积达到画布 90% 的 synthetic background shape 和退化 bbox
- structured/SFT 直接引用源 PNG，不复制 117 GB 图片
- 只生成 train；正式 eval 仍使用真实 benchmark

### `scripts/tasks/build_sft_from_structured.py`

用途：
- 把当前 maintained structured 数据转成框架可训练的 `jsonl_sft`
- 同时支持 `grounding_layout`、`grounding_layout_sync` 和 `point_line`
- grounding 统一使用当前 Qwen `0..999` bbox codec 和 v5.0 canonical order

当前 detection 转换命令：

```bash
uv run python scripts/tasks/build_sft_from_structured.py \
  --data-root data \
  --task grounding_layout \
  --task grounding_layout_sync \
  --workers 50 \
  --clean
```

关键行为：
- `bbox` 会量化到 `1000` bins，输出为 `bbox_2d`
- `target_text` 是纯 JSON array
- canonical order 为
  `row_bucket(y1,20) -> x1 -> y1 -> -area -> x2 -> y2 -> label`
- `system_prompt` 和 `user_prompt` 保持为空；训练时由 prompt pool 注入
- `grounding_layout` 与 `grounding_layout_sync` 都使用
  `configs/prompts/pools/grounding_layout.v5.0.yaml`

`convert_grounding_structured_to_sft.py` 和
`convert_grounding_structured_to_sft_row_major.py` 仅用于历史数据复现，不能用于当前 v5.1 数据重建。

### `scripts/tasks/build_region_reconstruction_sft.py`

用途：
- 将既有 shape、line、image reconstruction 筛选 manifest 转换为整图区域重建任务
- 保持原筛选、采样、类别分布和 sample ID 不变，仅把输入从 crop 改为源整图
- 生成 `shape_region_reconstruction`、`line_region_reconstruction` 和
  `image_region_reconstruction` 的 structured/SFT 数据

构建命令：

```bash
uv run python scripts/tasks/build_region_reconstruction_sft.py \
  --workers 50 \
  --clean
```

关键行为：
- SFT 直接引用源整图，不生成或复制 crop 图片
- `prompt_args.bbox_2d` 按整图宽高量化为 Qwen 整数 `0..999`
- shape/line target 内的所有控制点、body bbox 和 tail 点也按同一整图宽高量化；不得改成目标框局部坐标
- `system_prompt` 和 `user_prompt` 留空，由对应 v5.2 prompt pool 在运行时注入
- 旧 reconstruction structured 数据只作为确定性选择 manifest；shape/line 参数回查
  `gt_standard`，image bbox 与已 review 的 image type 回查 raw JSON 真源
- 只生成 train，validation 保持为空；筛选与采样策略不在该转换中改变

### `scripts/tasks/build_context_reconstruction_sft.py`

用途：
- 从 v5.2 region structured manifest 选择 shape、line、image 实例，但重新从 `gt_standard` / raw
  reviewed JSON 读取属性与几何真值
- 为每个实例生成一个确定性的宽松 contextual crop，并以近似一阶段
  `prompt_args.proposal_bbox_2d` 指定目标
- 生成 `shape_context_reconstruction`、`line_context_reconstruction` 和
  `image_context_reconstruction` 的 task-local PNG、structured/SFT、README 与 build summary

正式构建命令：

```bash
uv run python scripts/tasks/build_context_reconstruction_sft.py \
  --output-root data \
  --workers 8 \
  --chunksize 8 \
  --clean
```

关键行为：
- proposal center/scale/edge noise 与四边独立 context padding 分开采样；crop 始终覆盖完整可见 bbox
  和显式 shape/line 几何
- proposal bbox 与 target 几何共享当前 crop-local Qwen 整数 `0..999` 坐标，proposal 不建立第二个
  bbox-local target frame
- 保留 line 的多 segment/forked 结构，shape 只消费 source `label=shape`，不加入 icon/image-as-other
- real image 训练排除 `data/raw/splits/vlm.test.json`；validation 明确为空
- 先写同盘 staging，task 完整成功后原子发布；发布根目录权限固定为 `0755`
- 默认最大 crop aspect ratio 为 `60`，PNG 保留原 crop 像素，训练时再由配置的 Qwen pixel budget 处理


## 4. 维护规则

新增脚本时，至少需要同步更新本文件，说明：
- 脚本用途
- 输入输出
- 关键参数
- 示例命令

如果脚本只是一次性临时实验，不应写进这里，而应留在 `scripts/tmp/`。
