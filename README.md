# shaft（重构中）

Shaft 是一个 `HF-first` 的多模态训练与推理框架。当前主目标是把 `Qwen3VL + SFT` 主链做稳，同时保留面向 RLHF、更多模型族与推理后端的扩展骨架。

## 快速开始

```bash
uv venv --python 3.11 --prompt shaft
source .venv/bin/activate
uv pip install -e .
python scripts/train.py sft --config configs/train/banana_sft_4b.yaml
```

按用途安装扩展依赖：

```bash
# HF 训练主依赖
uv pip install -e ".[train]"

# GPU 训练增强
uv pip install -e ".[train,gpu]"

# 可选 CUDA kernel 增强
uv pip install -e ".[train,gpu,gpu-kernels]"

# RLHF
uv pip install -e ".[train,rlhf]"

# 部署 / vLLM
uv pip install -e ".[serve]"

# 离线 Eval Bench
uv pip install -e ".[eval-bench]"
```

## 统一入口

### 训练

```bash
python scripts/train.py sft --config configs/train/banana_sft_4b.yaml
python scripts/train.py rlhf --config configs/train/dpo_4b.yaml --algorithm dpo
python scripts/train.py rlhf --config configs/train/ppo_4b.yaml --algorithm ppo
python scripts/train.py rlhf --config configs/train/grpo_4b.yaml --algorithm grpo
```

### 推理

```bash
python scripts/infer.py --config configs/infer/pipeline_smoke.yaml --image /path/to/image.png
```

### 导出

```bash
python scripts/export.py inspect --path /path/to/checkpoint
python scripts/export.py validate --path /path/to/export --finetune-mode full --model-type qwen3vl
python scripts/export.py merge-peft \
  --model-type qwen3vl \
  --adapter-path /path/to/adapter \
  --base-model /path/to/base_model \
  --output-dir /path/to/merged_model
```

### Eval Bench

```bash
python scripts/eval_bench.py --help
python scripts/eval_bench.py serve-dashboard --host 127.0.0.1 --port 8765
```

Dashboard 当前提供高密度总览、benchmark、job queue、run/report 列表和交互式 run inspector；
总览页是四模块 cockpit，只保留下一步动作、F1/报告覆盖、评测闭环卡点和最近 run 产物流，
不展示细粒度评测指标或低频排障面板。
inspector 直接读取 benchmark GT 与 prediction snapshot，在原图上叠加 GT / Prediction
实例，支持 label 过滤、图层显隐、对象 hover/click 高亮、滚轮缩放和拖拽平移，主视图按图像工作台设计，低频配置和明细默认折叠，便于检查
漏检、误检和解析问题。Compare 页支持排行榜、run 成对 delta、top 改善/退化
样本列表和并排样本对比，用于比较新旧权重或 prompt/推理参数变更。每个 run 会持久化
模型路径、prompt ID/path/hash、prompt 文本快照、采样参数、pixel budget 和 vLLM 服务参数，
并在 Run Inspector 顶部按需展开，便于复盘一次评测到底用了什么配置。
Runs 表格中的备注摘要可直接跳转到对应 Run Inspector 的备注编辑面板。
Benchmarks 页可以从 raw_data split 创建 benchmark copy。
Run Inspector 和 Benchmark Inspector 都支持直接输入 sample 序号跳转。
Runs 页也可以把外部预测 JSON 目录导入为标准 run，并立即和对应 benchmark/test GT
评估对比。
Services 页可以登记外部 vLLM endpoint 或本地 vLLM OpenAI server 配置；本地服务会保存
CUDA、TP、port、max_model_len、GPU util、max_num_seqs 等启动参数，并提供 Start/Stop
入口。Job 创建采用 manifest 模板 + 自由 JSON 编辑 + preflight 校验；一次性 vLLM runtime
由 eval job 自己启动和关闭，长期 vLLM service 则在 Services 页独立管理，避免任务生命周期
和常驻服务生命周期混在一起。

说明：

- `scripts/*.py` 只做薄包装入口。
- 真实 CLI 解析与命令调度在 `src/shaft/cli`。
- 当前训练入口按 `sft / rlhf` 分流，推理与导出分别走独立 CLI。

## 配置示例

### 命名数据集 catalog

```yaml
data:
  media_snapshot_id: banana-v5.0-re2-media-v1
  batching:
    grouping: none
    cardinality: fixed
    packing:
      mode: none
    layout: padded
  catalog_path: ../data/example.yaml
  catalog_names: [arrow_multitask]
```

### 内联数据源

```yaml
data:
  batching:
    grouping: none
    cardinality: fixed
    packing:
      mode: none
    layout: padded
  datasets:
    - dataset_name: arrow_multitask
      source_type: jsonl_sft
      train_paths: [data/train.jsonl]
      val_paths: [data/val.jsonl]
      weight: 1.0
      use_for_eval: true
```

说明：

- `catalog_path` 指向命名数据集 catalog YAML。
- `catalog_names` 选择本次实验启用的命名数据集；**只有写进这里的数据集才会被加载**。
- catalog 文件里的数据集不会因为 `catalog_path` 被设置就自动全部参与训练。
- `DatasetSourceConfig.dataset_name` 是数据层统一标识字段。
- `DatasetSourceConfig` 只描述配置输入；进入数据主链后会先解析成 `ShaftDatasetMeta`。
- `use_for_eval=false` 表示该数据集只参与训练，不参与验证集构建，也不要求提供 `val_paths`。
- 仓库内置的 [`configs/data/example.yaml`](configs/data/example.yaml) 当前只是示例文件，里面的路径默认不保证存在。
- 如果你不想维护 catalog，也可以直接在训练 YAML 里写 `data.datasets`。
- 每个训练 YAML 都必须显式声明 `data.batching.grouping`、`cardinality`、`packing.mode` 与
  `layout`。缺失字段不会静默回退。

训练时长使用单一真源，step 是主路径：

```yaml
data:
  schedule:
    mixing: weighted
    shuffle: true
train:
  duration:
    unit: steps
    value: 10000
```

`weighted` 会把各数据源 `weight` 归一化为 sample draw 概率；epoch 模式仅用于有限时长兼容，写成
`duration: {unit: epochs, value: 1}`。

Qwen VL SFT 可启用有界、在线的成本感知批次：

```yaml
data:
  media_snapshot_id: banana-v5.0-re2-media-v1
  batching:
    grouping: bounded_cost
    cardinality: token_budget
    packing:
      mode: none
    layout: padded
    buffer_size: 64
    cost_cache_size: 65536
    max_tokens_per_microbatch: 10000
    resource_budgets:
      vision_patches: 16384
train:
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4
```

`bounded_cost` 不在训练前物化完整 CostPlan，只在 `buffer_size` 个轻量 `SampleRef + cost` 上工作。
`train.per_device_train_batch_size` 是每卡 physical pack count 的唯一配置真源：`fixed` 时是精确值，
`token_budget` 时是上限 `B`。在该 `packing=none` 路径中一个 pack 就是一条样本；planner 根据真实 padded
token 与 vision 总预算为每个 rank 选择 `1..B` 个 pack，
并让同一 microstep 的 rank 成本尽量接近。mixing 的 draw multiset 不丢失、不复制，HF 继续负责固定 GA、
optimizer、scheduler 和 checkpoint。re/re2 当前把 `B` 配成 2，使用 8 rank、GA4，即每个 optimizer step
处理 32–64 条 logical samples；loss 按该 optimizer frame 内跨 rank 的真实有效 token 归一化。

成本按 buffer 即时估算，图像只读 header，并使用容量受限的 LRU。多 rank 启动校验得到的首个 plan 会被
正式 sampler 直接复用；为了在任何 forward 前原子验证完整 GA frame，首个 forward 前的成本调用上界为
`buffer_size + (GA - 1) * world_size * per_device_train_batch_size`，仍与总 steps 无关。
`media_snapshot_id` 声明外部图片是不可变快照；改变媒体必须同时改变该 id。

checkpoint committed state 直接进入 HF `trainer_state.json` 的 stateful callback。所有 rank 的模型状态、
optimizer/scheduler 与 RNG 保存成功后，rank 0 原子发布 completion manifest，之后才执行 checkpoint
rotation；保存的是模型已完成 optimizer step 的 buffer/cursor，而不是 DataLoader 预取推进的 live cursor。
completion 同时绑定 versioned batch contract 与 planner spec；恢复会在加载数据/模型前先校验 batch contract
及 duration/GA/optimizer/scheduler contract。`cost_cache_size` 只影响 host LRU，不阻止 exact resume。
当前 planned batching 只开放 SFT + step duration + DDP，eval 保持普通 padded fixed batch。Qwen3VL
image SFT 已支持 `grouping=length + cardinality=fixed + packing.mode=greedy + layout=varlen`：planner 在
有界窗口内按真实 processor 后长度分组，把多个完整 logical segment 装入固定数量的 physical packs；
CUDA 执行要求 FlashAttention 2、bf16/fp16 与 DDP，未验收的模型族/backend/topology 会在加载数据和权重前
fail closed。`per_device_train_batch_size` 表示每卡 physical pack 数，不等于 pack 内 logical segment 数。
完整边界见
[`docs/training_batch_planning_design.md`](docs/training_batch_planning_design.md) 与
[`docs/config_reference.md`](docs/config_reference.md)。

Shaft 会在 dataset、base model 与 PEFT adapter 装配前初始化 `experiment.seed`。需要验证 CUDA bitwise
resume/fresh reproduction 时，再在 `train` 下设置 `full_determinism: true`；该选项还会启用确定性
CUDA/attention backward，通常有吞吐代价。默认关闭时，非确定性 kernel 产生的微小数值差异不等同于
BatchPlan 或 resume cursor 错位。SFT 训练过程中的 eval 会保存并恢复主进程训练 RNG，避免 persistent
eval workers 改变后续训练随机序列。

## 当前能力

### 训练

- `SFT`
- `DPO`
- `PPO`（受限能力，非完整生产功能）
- `GRPO`（当前复用 `jsonl_sft` 作为 prompt-target 数据；数据计划可与 TRL grouped-generation sampler
  通过无状态位置索引组合）

### 推理

- 本地 HF 推理：`hf_local`
- vLLM OpenAI 兼容后端：`vllm_openai`
- 单阶段与多阶段推理编排
- stage 级 `codec`、重试、超时、像素预算覆盖

### 导出

- HF / PEFT 目录识别
- HF 兼容导出校验
- `merge-peft` 合并 adapter 为标准 HF full export

## 架构概览

- `src/shaft/config`：配置 schema、YAML 加载、catalog 展开、归一化校验
- `src/shaft/data`：数据源、增强、mixing、dataset、collator
- `src/shaft/model`：模型族元信息、HF 加载、PEFT 包装、processor/peft policy
- `src/shaft/template`：chat template 与 decode 约定
- `src/shaft/algorithms`：SFT/DPO/PPO/GRPO trainer 装配
- `src/shaft/pipeline`：`ShaftSFTPipeline` / `ShaftRLHFPipeline`
- `src/shaft/training`：trainer、optimizer、scheduler、loss、checkpoint 规则
- `src/shaft/infer`：`ShaftInferEngine`、`ShaftInferPipeline`、codec
- `src/shaft/export`：HF 兼容导出工具链
- `src/shaft/plugins`：registry、hook、interceptor
- `src/shaft/observability`：logging、context、events、统一 progress 状态与 terminal/plain/JSON sink；TTY
  使用单行高分辨率进度、不会提前完成的百分比、自适应 `s/step`/`step/s`、ETA、loss 和多参数组 LR range
- `projects/eval_bench`：离线评测工作台子项目，管理 benchmark copy、run manifest、prediction snapshot、报告、对比与可视化

## 文档

统一文档入口见：

- [docs/README.md](docs/README.md)

重点文档：

- [docs/architecture.md](docs/architecture.md)
- [docs/module_reference.md](docs/module_reference.md)
- [docs/config_reference.md](docs/config_reference.md)
- [docs/development_workflow.md](docs/development_workflow.md)
- [docs/extension_guide.md](docs/extension_guide.md)
- [docs/testing.md](docs/testing.md)
- [docs/infer.md](docs/infer.md)
- [docs/export.md](docs/export.md)
- [projects/eval_bench/README.md](projects/eval_bench/README.md)

## 测试

快速回归：

```bash
uv run pytest -q
```

主链 smoke：

```bash
uv run pytest -q -m "smoke and not manual"
```

只跑 integration：

```bash
uv run pytest -q -m integration
```

只跑 manual：

```bash
uv run pytest -q -m manual
```

更多测试规范见 [docs/testing.md](docs/testing.md)。

## 当前说明

- 当前正式模型族实现以 `qwen3vl` 为主，`smoke_vlm` 只用于测试。
- 训练和保存遵循 HF / PEFT / TRL 标准能力。
- 旧实现已归档到 `old/`，新开发只在 `src/shaft`。
- 结构化任务离线评估子系统尚未完成。
- PPO 暂停项见 [docs/ppo_todo.md](docs/ppo_todo.md)。
