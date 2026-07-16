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

`merge-peft` 默认校验 Shaft adapter checkpoint 中记录的训练 base-model identity。第三方/旧 adapter 缺少
该 provenance 时会 fail closed；人工确认 base 后可显式使用 `--allow-unverified-base-model true`，当前 base
仍会执行完整字节 SHA256 校验。

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

checkpoint committed state 直接进入 HF `trainer_state.json` 的 stateful callback。SFT、DPO、GRPO 的 DDP/
native-HF 路径共用 `committed_manifest` 协议：所有 rank 必须具有完全相同、顺序一致的 `on_save` callback
拓扑，否则在执行 callback 前 fail closed；每个 callback 的 rank-local 结果再做 all-rank convergence。全部
callback 和模型/adapter、optimizer/scheduler、RNG 保存成功后，rank 0 原子发布
`shaft_checkpoint_commit.json`，之后才执行 checkpoint rotation。FSDP/DeepSpeed 显式走
`backend_native` 协议，由对应后端负责保存、发现、校验和 rotation，不安装该 commit wrapper，也不宣称具备
通用 manifest 的 torn/atomic 防护。planned SFT 保存的是模型已完成 optimizer step 的 buffer/cursor，而不是
DataLoader 预取推进的 live cursor；该状态作为 manifest extension 绑定 versioned batch contract、planner
spec 与 duration/GA/optimizer/scheduler contract。
`cost_cache_size` 只影响 host LRU，不阻止 exact resume。
当前 planned batching 只开放 SFT + step duration + DDP，eval 保持普通 padded fixed batch。Qwen3VL 与
HF `qwen3_5`（Qwen3.5/Qwen3.6）image SFT 已支持
`grouping=length + cardinality=fixed + packing.mode=greedy + layout=varlen`：planner 在
有界窗口内按真实 processor 后长度分组，把多个完整 logical segment 装入固定数量的 physical packs；
CUDA 执行要求 FlashAttention 2、bf16/fp16 与 DDP；Qwen3.5/3.6 hybrid attention 还要求
flash-linear-attention 与 causal-conv1d。未验收的模型族/backend/topology 会在加载数据和权重前 fail closed。
`per_device_train_batch_size` 表示每卡 physical pack 数，不等于 pack 内 logical segment 数。

训练默认生成 committed `shaft_training_efficiency.json`：统计实际 collate 后的 useful/materialized/
supervised tokens、logical-segment length 分布、vision patches、logical segments/physical packs、batch
acquire、batch prepare、host/device optimizer-frame time、critical-path p50/p95、训练窗口内 peak CUDA memory
与 DDP rank skew。它只在成功 optimizer boundary
提交，不会把 DataLoader prefetch 或
`[batch-plan-summary]` 误算成已执行吞吐。可用
`python scripts/compare_efficiency.py RUN_A RUN_B ...` 比较不同 batching/layout 的 A/B 结果；checkpoint
内的 per-rank snapshot 支持 resume 后继续完整累计。summary 内置类型化训练契约；比较器默认拒绝模型、
数据快照、logical draw stream、DP/GA、优化器或 step span 不一致的结果，只有明确诊断时才使用
`--allow-incompatible`；packing 导致 committed logical workload 不同但其它约束相同时，可用更窄的
`--allow-workload-variation` 做 capacity 对比；它仍锁定 optimizer update、microbatch 与 physical-pack 数，
不能把结果表述为等工作量 speedup。peak memory 从 HF
`on_train_begin` 建立窗口，resume 时取 checkpoint 历史与当前窗口最大值；历史缺失时明确输出 `n/a`。
snapshot set 使用 revoke、all-rank snapshot、rank-zero manifest 三阶段提交；每个
可失败的文件阶段都会先做固定 tensor 状态汇合，避免单 rank I/O 错误把其它 rank 留在 barrier。
比较器只接受采用当前 measurement contract 的 v3 summary；旧 v2 不自动迁移。
完整边界见
[`docs/training_batch_planning_design.md`](docs/training_batch_planning_design.md) 与
[`docs/config_reference.md`](docs/config_reference.md)。

Shaft 会在 dataset、base model 与 PEFT adapter 装配前初始化 `experiment.seed`。需要验证 CUDA bitwise
resume/fresh reproduction 时，在 `train` 下设置 `full_determinism: true`；三个及以上 DDP rank 还要为静态
SFT 参数图显式设置 `distributed.ddp.static_graph: true`，固定跨 checkpoint 重建时的 reducer bucket 生命周期。
两者通常都有吞吐或适用范围代价；动态图不得冒充 static graph。默认关闭时，非确定性 kernel 或 reducer
浮点归约产生的微小数值差异不等同于 BatchPlan / resume cursor 错位。SFT 训练过程中的 eval 会保存并恢复
主进程训练 RNG，避免 persistent eval workers 改变后续训练随机序列。

## 当前能力

### 训练

- `SFT`
- SFT prompt pool 支持 pool 级参数 schema 与 JSONL `prompt_args`；训练 planning 和实际读取共用受限
  `{{ name }}` / `{{ name | json }}` renderer，静态 pool 保持兼容。
- `DPO`
- `PPO`（受限能力，禁止 resume，`save_strategy` 必须为 `no`；最终导出不受影响）
- `GRPO`（当前复用 `jsonl_sft` 作为 prompt-target 数据；数据计划可与 TRL grouped-generation sampler
  通过无状态位置索引组合）
- 评估输入可用 `eval.min_pixels/max_pixels` 设置默认像素预算，并通过
  `eval.datasets.<name>.min_pixels/max_pixels` 覆盖单个数据集；SFT 的 teacher-forced loss 与在线生成共享
  同一解析结果。processor 的训练/生成 padding 由模型 policy 统一选择，不需要在 pipeline 中重复配置。
  GRPO 当前只支持 online `eval_final_score`，启用 eval 时必须设置 `loss_metrics_enabled: false`。

### 推理

- 本地 HF 推理：`hf_local`
- vLLM OpenAI 兼容后端：`vllm_openai`
- 单阶段与多阶段推理编排
- 多阶段 prompt 使用与训练相同的受限 renderer 和显式 `arguments`；旧 Python `{name}` format 语法已移除。
- stage 级 `codec`、重试、超时、像素预算覆盖
- stage timeout 使用贯穿重试、backoff 和后端 I/O 的同一个绝对 deadline；pipeline 也接受 cooperative
  cancellation。无法安全抢占的本地 HF generate 会在开始工作前明确拒绝 control，不会遗留后台推理。
  详细能力边界见 [docs/infer.md](docs/infer.md)。

### 导出

- HF / PEFT 目录识别
- HF 兼容导出校验
- `merge-peft` 合并 adapter 为标准 HF full export

## 架构概览

- `src/shaft/config`：配置 schema、YAML 加载、catalog 展开、归一化校验
- `src/shaft/data`：数据源、增强、mixing、dataset、collator
- `src/shaft/model`：模型族元信息、HF 加载、PEFT 包装、processor/inference/peft policy
- `src/shaft/template`：chat template 与 decode 约定
- `src/shaft/algorithms`：SFT/DPO/PPO/GRPO trainer 装配
- `src/shaft/pipeline`：`ShaftSFTPipeline` / `ShaftRLHFPipeline`
- `src/shaft/training`：trainer、optimizer、scheduler、loss、checkpoint 规则
- `src/shaft/infer`：`ShaftInferEngine`、`ShaftInferPipeline`、codec
- `src/shaft/export`：HF 兼容导出工具链
- `src/shaft/plugins`：registry、hook、interceptor
- `src/shaft/observability`：logging、context、events、统一 progress 状态与 terminal/plain/JSON sink；TTY
  使用高对比度自适应单行进度、标准 `s/it`/`it/s`、动态 spinner、ETA、loss、token throughput 和多参数组
  LR range；自动适配终端宽度与颜色能力，日志不会打断活动行

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

- 当前正式训练主链覆盖 `qwen3vl`、`qwen35vl` 与 `qwen36vl`；后两者共享 upstream
  `qwen3_5` / `qwen3_5_moe` architecture，但保留产品级注册项与模板入口。`smoke_vlm` 只用于测试。
- 训练和保存遵循 HF / PEFT / TRL 标准能力。
- 旧实现已归档到 `old/`，新开发只在 `src/shaft`。
- 结构化任务离线评估子系统尚未完成。
- PPO 暂停项见 [docs/ppo_todo.md](docs/ppo_todo.md)。
