# Shaft 脚本使用说明

本文档说明仓库 `scripts/` 目录下的正式脚本如何使用。

范围约束：
- 覆盖顶层正式入口和当前受维护的 v5.7 数据任务脚本
- 历史 converter 和已退出正式 mix 的任务不再作为推荐流程记录
- **不覆盖 `scripts/tmp/`**；`tmp` 目录视为临时实验区，不属于稳定接口

## 1. 设计原则

`scripts/*.py` 的定位是**薄入口**：
- CLI 解析与命令编排放在 `src/shaft/cli`
- `scripts/*.py` 只负责调用对应 CLI 主入口

`scripts/tasks/*.py` 是明确的离线数据准备入口，不是训练内核 CLI。

## 2. 顶层脚本

### `scripts/train.py`

用途：
- 统一训练入口
- 当前训练入口严格对应三个 domain：`sft`、`rl` 与 `opd`；不保留第二套 `rlhf` CLI

常用形式：

```bash
python scripts/train.py sft --config configs/train/sft_4b.yaml
python scripts/train.py rl --config configs/train/dpo_4b.yaml --algorithm dpo
python scripts/train.py opd --config /path/to/opd.yaml
```

兼容写法：

```bash
python scripts/train.py --config configs/train/sft_4b.yaml
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

### `scripts/serve_opd_teacher.py`

用途：启动一个 immutable local-HF OPD teacher 的版本化 HTTP 服务。服务端配置必须选择
`opd.teacher.provider=hf_local`；训练端使用另一份 `provider=http` 配置并固定服务发布的 artifact SHA-256。

```bash
SHAFT_TEACHER_TOKEN='...' \
python scripts/serve_opd_teacher.py \
  --config /path/to/opd_teacher_server.yaml \
  --host 0.0.0.0 \
  --port 8100 \
  --api-key-env SHAFT_TEACHER_TOKEN
```

训练端对应配置：

```yaml
opd:
  teacher:
    provider: http
    model_type: qwen3vl
    remote:
      endpoint: http://teacher-host:8100
      artifact_fingerprint: <64-char-sha256-from-server-identity>
      api_key_env: SHAFT_TEACHER_TOKEN
```

服务只暴露 `/v1/identity` 与 `/v1/score`，请求/响应使用有界 safetensors envelope。服务端和训练端从环境
变量读取 token，不应把密钥写入 YAML。

OPD 使用独立 vLLM server rollout 时，先启动与 student base artifact 相同的 TRL 服务，再在训练配置设置
`opd.rollout.backend=vllm` 和匹配的 `opd.rollout.vllm.server_port/group_port`：

```bash
CUDA_VISIBLE_DEVICES=1 trl vllm-serve \
  --model /path/to/student \
  --host 127.0.0.1 \
  --port 8000
```

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

`scripts/tasks/` 只承载可复现的离线数据准备与转换，不承载训练循环。Banana v5.7 的
数据目录、行数、prompt 映射和完整性基线见 [docs/data_v5_7.md](data_v5_7.md)。

### `scripts/tasks/prepare_gt_standard_v5_7.py`

用途：审计 v9 synthetic `gt_standard`，确认 train/val 无交叉，然后生成 shape、line 和
synthetic multi-branch line-points 的 source-identity selection。

```bash
uv run python scripts/tasks/prepare_gt_standard_v5_7.py \
  --dataset-root data/regulated_layout_dataset_v9_20260802 \
  --output-root data/reconstruction_v5_7_selection \
  --workers 8
```

只审计不写 selection：

```bash
uv run python scripts/tasks/prepare_gt_standard_v5_7.py --audit-only --workers 8
```

任何 source 级错误或 train/val 交叉都会 fail fast；少量实例级 bbox/points 错误保留在审计
摘要中，但不会进入 selection。

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
  --workers 8 \
  --clean
```

当前多尺度约束：
- 目标像素在 `200704..2000000` 内按 log 空间连续采样，最终宽高按 `32` 对齐
- 离线线性放大不超过 `2x`，同源尺度像素量至少相差 `1.35x`
- clean resize、padding、degraded resize、density crop、hard negative 的目标比例分别约为
  `0.9x / 0.1x / 0.75x / 0.15x / 0.03x`
- padding 为非对称随机偏移；退化只使用单一 Gaussian blur 或 Gaussian noise
- validation/test 只保留 native clean full-image

正式 v5.7 只构建真实 `grounding_layout`。synthetic `grounding_layout_sync` 不在当前 catalog 中，
不应在默认流程中构建或混入。

### `scripts/tasks/build_sft_from_structured.py`

用途：把当前 `grounding_layout` structured 数据转成 `jsonl_sft`。默认 prompt 是已跟踪的
`grounding_layout.v5.7.yaml`。

当前 detection 转换命令：

```bash
uv run python scripts/tasks/build_sft_from_structured.py \
  --data-root data \
  --task grounding_layout \
  --workers 8 \
  --clean
```

关键行为：
- `bbox` 会量化到 `1000` bins，输出为 `bbox_2d`
- `target_text` 是纯 JSON array
- canonical order 为
  `row_bucket(y1,20) -> x1 -> y1 -> -area -> x2 -> y2 -> label`
- `system_prompt` 和 `user_prompt` 保持为空；训练时由 prompt pool 注入
- 所有 prompt 和 structured split 会在 `--clean` 删除旧 SFT 之前完成预检。

### `scripts/tasks/build_context_reconstruction_sft.py`

用途：
- 从 v9 selection 选择 shape/line 实例，每次从 `gt_standard` 重新读取属性与几何真值
- 从 reviewed compact raw 构建 image type 和真实 line points
- 为每个实例生成一个确定性的宽松 contextual crop，并以近似一阶段
  `prompt_args.proposal_bbox_2d` 指定目标
- 生成 v5.7 正式任务的 task-local PNG、structured/SFT、README 与 build summary

正式构建命令：

```bash
uv run python scripts/tasks/build_context_reconstruction_sft.py \
  --output-root data \
  --shape-selection data/reconstruction_v5_7_selection/shape/train.jsonl \
  --line-selection data/reconstruction_v5_7_selection/line/train.jsonl \
  --workers 8 \
  --chunksize 8 \
  --clean
```

先从 active compact raw 选择全部非空真实 line points：

```bash
uv run python scripts/tasks/prepare_real_line_context_points.py \
  --raw-root data/raw \
  --train-split data/raw/splits/grounding_layout.train.txt \
  --output data/reconstruction_v5_7_selection/line_points_real/train.jsonl \
  --workers 40 \
  --clean
```

再生成真实 points，并合并维护的 15,000 条合成多分支数据：

```bash
uv run python scripts/tasks/build_context_reconstruction_sft.py \
  --raw-root data/raw \
  --synthetic-root data/regulated_layout_dataset_v9_20260802 \
  --line-point-real-selection \
    data/reconstruction_v5_7_selection/line_points_real/train.jsonl \
  --line-point-synthetic-selection \
    data/reconstruction_v5_7_selection/line_points/train.jsonl \
  --line-point-synthetic-limit 15000 \
  --tasks line_context_points \
  --workers 40 \
  --chunksize 8 \
  --clean
```

关键行为：
- proposal center/scale/edge noise 与四边独立 context padding 分开采样；crop 始终覆盖完整可见 bbox
  和显式 shape/line 几何
- proposal bbox 与 target 几何共享当前 crop-local Qwen 整数 `0..999` 坐标，proposal 不建立第二个
  bbox-local target frame
- 保留 line 的多 segment/forked 结构，shape 只消费 source `label=shape`，不加入 icon/image-as-other
- shape/line 每张合成 crop 默认使用 `synthetic_realism_v1`：确定性选择 1–3 个 resize round-trip、
  Gaussian blur/noise、JPEG 扰动并允许叠加；极小 target `<80/999` 只使用 mild 单扰动
- 所有像素扰动严格保持 crop 宽高、坐标和 target 不变；真实 image crop 不做合成扰动
- 每个 task 写入自包含的 `selection/train.jsonl`，后续重建只从中恢复 source identity，target 仍回查
  `gt_standard` / raw reviewed JSON
- `shape_context_attributes` 不属于 v5.7；不要在默认构建或 catalog 中加回该历史弱标签任务
- `line_context_points` 不复用历史 tight crop：真实数据从 active compact raw 回查 bbox 与有序
  `parameters.points`，保留全部非空 line，不做采样；为补齐分叉监督，同一任务还从 v9
  `gt_standard` 真值中加入维护的 15,000 条 `is_single=false` 合成多分支线。两类 source 都重新生成
  proposal/context crop；target 严格只有 `is_single + points`，合成单叉不会进入该任务，也不得补造
  样式、颜色或箭头端点属性
- 空 points 不进入任务；真实 source 中相邻重复点和 Qwen 量化后产生的相邻重复点只在派生 target 中
  清理，不删除对应实例。真实/合成每个入选实例都只有一张 crop，不生成多尺度副本
- `line_context_points` 中真实 crop 保持 clean；合成多叉 crop 强制使用 `synthetic_realism_v1`，不能因其
  与真实数据共处一个 task 而跳过像素域扰动
- 真实 selection 只读取 active train split，并再次排除当前 test manifest；model-facing label 始终为
  `line`
- real image 训练排除 `data/raw/splits/vlm.test.json`；validation 明确为空
- 先写同盘 staging，task 完整成功后原子发布；发布根目录权限固定为 `0755`
- 默认最大 crop aspect ratio 为 `60`；PNG 尺寸保持原 crop 尺寸，训练时再由配置的 Qwen pixel budget 处理


## 4. 维护规则

新增脚本时，至少需要同步更新本文件，说明：
- 脚本用途
- 输入输出
- 关键参数
- 示例命令

如果脚本只是一次性临时实验，不应写进这里，而应留在 `scripts/tmp/`。
