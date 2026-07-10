# Shaft 成本感知训练批次顶层设计

状态：**Phase 0/1 已实现；Phase 2-4 待实现**

日期：2026-07-10

首个落地范围：`Qwen3VL + SFT`

当前运行时已落地 `ShaftSampleCost`、Qwen VL 精确 image-cost policy、有界固定样本数
`ShaftBatchPlan`、跨 DP rank 配对和 `ShaftCostAwareSampler`。动态 microbatch、全局
numerator/denominator loss、sequence packing 与 context parallel 仍仅保留本文档中的目标契约。
运行时 cost provider 只编排数据：模型 token expansion 由 `ProcessorPolicy` 估算，target/EOS/causal
supervision 由 `Template` 估算，避免模型或模板升级后出现第二套成本语义。

本文档定义 Shaft 在 sample-level mixing、成本感知 batching、sequence packing、分布式 rank
平衡和未来 context parallel 之间的统一语义。目标是让效率优化不改变数据选择和优化权重，并为后续
实现提供唯一的顶层契约。

---

## 1. 结论先行

训练批次主链固定为四个真源：

```text
SamplePlan -> CostPlan -> BatchPlan -> PackPlan
```

- `SamplePlan` 决定训练什么：数据源、样本、draw context 和 prompt variant。
- `CostPlan` 描述这些逻辑样本的可复现成本，不选择或替换样本。
- `BatchPlan` 决定如何执行：optimizer batch、全局 microstep、各 DP rank 的本地 microbatch。
- `PackPlan` 只决定本地 microbatch 的物理布局，不改变样本边界、loss 权重或 mixing 结果。

同时固定以下原则：

1. mixing 约束作用于有界的 mixing horizon / optimizer batch，不要求每个物理 microbatch 都满足比例。
2. rank 成本平衡作用于全局 microstep；同一 microstep 的各 DP rank 应尽量处理相近成本。
3. packing 只做 whole-sample、segment-isolated 的物理压缩，不允许普通 causal concat。
4. context parallel 是 job 级静态拓扑，不作为某个 batch 的动态补救手段。
5. 动态 microbatch 下，loss 必须以整个 optimizer batch 的全局 numerator/denominator 归一化，不能把
   每个本地 microbatch 的 mean 等权平均。

## 2. 目标与非目标

### 2.1 目标

- 降低文本长度和图像 patch 波动造成的 padding、rank straggler、长尾 step 和动态 OOM。
- 在不改变 sample-level mixing 语义的前提下，让短样本使用更大的本地 batch。
- 让 prompt 随机轮换、sample mixing、batch 重排和 resume 保持确定性。
- 为 Qwen3VL SFT 的 padding-free packing 提供严格正确性边界。
- 让后续模型族通过明确 policy 接入，而不是把 Qwen 特例写进通用 collator。
- 为未来 `DP x CP` 拓扑预留正确的 data-rank 和 loss 语义。

### 2.2 非目标

- 第一阶段不实现 context parallel。
- 第一阶段不为 DPO、PPO、GRPO 同时开放动态 batching 或 packing。
- 不通过复制短样本、丢弃长样本或修改 loss weight 来伪造利用率提升。
- 不在 pipeline 中实现 cost 估算、mixing 或 packing 细节。
- 不把 batch planner 扩成反馈驱动的数据课程系统。

## 3. 术语与层级

### 3.1 Logical sample

一条由 mixer 选中的训练样本。它必须包含稳定的 `ShaftSampleRef` 和 draw context；prompt variant、在线
transform 等随机结果都必须能由该 context 确定性恢复。

### 3.2 Mixing horizon

允许 batch planner 重排的一段有界 logical sample 集合。planner 可以在 horizon 内改变执行顺序，但
必须保证：

- 每个选中样本恰好消费一次；
- 不跨 horizon 无限等待更合适的长度；
- 不改变各 source 的选中 multiset；
- resume 后得到相同 horizon 和相同执行计划。

### 3.3 Optimizer batch

一次 `optimizer.step()` 共同消费的 logical sample 集合。它可以被拆成多个成本同质的全局
microstep。mixing 的短期统计和 loss denominator 至少在这个层级收口。

### 3.4 Global microstep

一次 forward/backward 同步单元。每个 data-parallel rank 获得一个 local microbatch；planner 优化的
目标是降低各 rank 之间的最大成本差。

### 3.5 Local microbatch

单个 DP rank 在一个 microstep 中处理的 logical sample 列表。样本数可以变化，但必须满足硬内存预算。

### 3.6 Packed block / segment

packed block 是物理 tensor 布局；segment 是其中一条原始 logical sample。segment 之间必须使用
`cu_seq_lens`、等价 varlen metadata 或经过证明的 block-diagonal mask 隔离。

## 4. 四个计划真源

### 4.1 `SamplePlan`：选择真源

现有 `ShaftSamplePlan` 继续拥有：

- `concat / weighted` sample-level selection；
- dataset weight；
- source row selection；
- `draw_id / plan_cycle / transform_seed`；
- 可复现的 prompt sampling 输入。

Batch planner 不得重新抽样，也不得根据长度改变 source 权重。当前 `weighted` 的权重仍表示 sample
exposure probability，不静默改成 token 或 compute 比例。

长期需要把“按固定 batch 公式预先确定 plan 长度”与 sample 映射能力拆开。`duration=steps` 下，训练时长
真源是 optimizer step 数；动态 microbatch 启用后，不能继续使用：

```text
steps * per_device_batch * gradient_accumulation * world_size
```

作为唯一 sample budget 公式。应由 `BatchPlan` 消费 draw cursor 并决定每个 optimizer step 实际需要
多少 sample refs。

### 4.2 `CostPlan`：成本真源

建议新增不可变的 `ShaftSampleCost`，至少包含：

```text
dataset_name
row_index
prompt_variant_id
llm_tokens
supervised_tokens
loss_weight_sum
image_count
vision_patches
optional estimated_compute / estimated_peak_memory
cost_fingerprint
```

第一版保留成本向量，不急于压成单一 scalar。硬预算与排序分开：

- 硬预算用于阻止 OOM：token storage、vision patches、sample count。
- `estimated_compute` 用于 rank 分配和尾延迟优化，可在 profiler 后拟合。

cost cache 必须至少绑定：

- source snapshot fingerprint；
- processor/model family 与 revision；
- template/chat rendering fingerprint；
- `min_pixels / max_pixels`；
- `max_length / add_eos_token`；
- prompt pool version 与 variant。

prompt variant 必须在 cost planning 前确定。运行时不能先按短 prompt 估算，再由 dataset 随机换成长
prompt。实现可以复用 draw context 选择 variant，并读取 per-variant cost metadata。

### 4.3 `BatchPlan`：执行真源

建议的 `ShaftBatchPlan` 层级：

```text
ShaftBatchPlan
  optimizer_steps[]
    selected_sample_refs[]
    global_microsteps[]
      rank_microbatches[dp_rank][]
      expected_rank_costs[]
    expected_loss_denominator
```

planner 的输入是 SamplePlan、CostPlan、parallel topology 和 batch budgets；输出是可恢复的 sample ref
分组。它不能读取图片、调用 processor 或计算 loss。

`BatchPlan` 是 `data` 与 `training` 之间的边界契约，不意味着 data 层拥有 optimizer 调度：training
决定 optimizer step/horizon 和目标 batch budget，data planner 只对给定 sample multiset 做分组与
rank 分配，training 再负责执行、累积和 step。

`expected_loss_denominator` 只用于提前规划和一致性校验。训练语义的最终真源仍是实际 collator 输出的
`labels/loss_scale`；两者不一致时必须显式失败或使用训练侧的安全聚合路径，不能静默采用 cost estimate。

全局 microstep 的优化目标是：

```text
minimize max(expected_rank_cost)
```

同时满足：

- 每个 rank 的硬预算；
- 每个 optimizer batch 的 sample multiset；
- 所有 DP rank 的 microstep 数一致；
- 有界 planning window；
- deterministic resume。

### 4.4 `PackPlan`：物理布局真源

`PackPlan` 在 local microbatch 已确定后执行，包含：

- segment 顺序与 offsets；
- `cu_seq_lens_q/k` 和 max segment length；
- flattened `input_ids / labels / loss_scale`；
- segment first-token label mask；
- model-specific position metadata；
- image/grid/media 到 segment 的对应关系。

packing 不得回头改变 BatchPlan 的 source composition 或 optimizer weight。

## 5. Mixing 与效率的统一语义

### 5.1 比例的四种口径

必须分别观测：

- `selected_sample_ratio`：mixer 选中多少样本；
- `supervised_token_ratio`：各 source 贡献多少有效监督 token；
- `compute_ratio`：各 source 消耗多少估算/实测计算；
- `gradient_ratio`：应用 loss scale/source weight 后的实际优化贡献。

第一版只承诺现有 sample-level mixing。若两个 source 的平均 target 长度不同，sample ratio 相同并不
意味着 token 或 gradient contribution 相同。batch planner 不负责偷偷修正这种差异。

未来若需要 token-level mixing，应新增显式的 mixing unit 或 source loss policy，不能重新解释现有
`DatasetSourceConfig.weight`。

### 5.2 允许与禁止的重排

允许：

- 在固定 planning window 内按 cost 分桶；
- 将不同 source 的样本放入同一 packed microbatch；
- 让单个 microbatch 的 source ratio 偏离目标；
- 在 optimizer batch 或更长统计 horizon 内回归目标比例。

禁止：

- 为填满 GPU 复制短样本；
- 静默丢弃长样本；
- 根据 source 长度修改 mixer 已选权重；
- 无界缓存某个样本等待“完美搭档”；
- 在 packed segments 之间开放 attention。

### 5.3 极端长度与 source 强相关

如果某个 source 几乎全是长样本、另一个 source 几乎全是短样本，严格要求每个 microbatch 精确配比会
产生不可消除的效率损失。处理顺序固定为：

1. 扩大但限制 planning/mixing horizon；
2. 允许 microbatch 比例波动，在 optimizer batch 或统计 horizon 收口；
3. 用多个成本同质 microstep 完成一次梯度累积；
4. 极端长上下文进入独立训练 stage/job。

## 6. 成本模型与 batch 约束

### 6.1 Padded microbatch

对于长度 `L_i` 的普通 padded batch：

```text
token_storage = sample_count * max(L_i)
useful_tokens = sum(L_i)
padding_ratio = 1 - useful_tokens / token_storage
```

FlashAttention 可能在 attention kernel 中 unpad，但 QKV/MLP/lm_head 和其他 sequence-aligned
activation 仍会受到 padded shape 影响。因此硬预算不能只用 `sum(L_i)`。

### 6.2 Padding-free packed microbatch

对正确的 varlen packing：

```text
token_storage = sum(L_i)
attention_work ~= sum(L_i^2)
```

不能按一个连续序列计算成 `sum(L_i)^2`；那代表发生了错误的跨样本 attention。

### 6.3 多模态成本

LLM token budget 不能替代 vision budget。至少同时约束：

- processor 后 LLM sequence tokens；
- vision encoder patch/grid 数；
- image/video 数量；
- 必要时 pixel tensor bytes。

第一版使用二维/多维 bucket；profiler 稳定后再产生单一 `estimated_compute` 排序值。

## 7. 规划算法

每个 planning window 按以下顺序处理：

1. 从 SamplePlan 取得确定性的 refs，并确定 prompt variants。
2. 从 CostPlan 取得每条 ref 的成本；缺失成本时使用显式 fallback 或拒绝 cost-aware 模式。
3. 固定 optimizer batch 的 sample multiset，不再重采样。
4. 按 token/vision cost band 划分候选。
5. 在每个 band 内构造满足硬预算的 local microbatches。
6. 将 local microbatches 分配到同一 global microstep 的各 DP rank，使最大 rank cost 最小。
7. 对每个 local microbatch 生成 PackPlan；模型不支持 packing 时退化为普通 padding。
8. 记录 plan cursor、成本统计和实际 batch metrics。

示例，假设逻辑长度为：

```text
[8, 1, 1, 1, 1, 2]
```

这些值先属于 planning window，而不是预先固定的 physical batch。普通 cost-aware batching 可形成：

```text
microbatch A: [8]
microbatch B: [2, 1, 1, 1, 1]
```

启用 whole-sample packing 后可形成：

```text
pack A: [8]
pack B: [2|1|1|1|1]
```

在 DDP 中，planner 应优先从更大的 window 找到另一个接近 `8` 的 local microbatch 与 A 同步执行，
而不是固定让一个 rank 跑 `8`、另一个 rank 跑成本远低的短样本组。

## 8. Optimizer batch 与 loss 归一化

旧实现按每个 local microbatch 独立求 mean，在以下情况不正确：

- 各 rank 有效监督 token 数不同；
- 每个 microbatch 样本数不同；
- 一个 optimizer step 包含多个大小不同的 microbatch；
- packing 后每个物理 tensor 包含不同 segment 数。

当前 `ShaftSFTTrainer` 已按实际 `labels/loss_scale` 汇总一个 optimizer batch 的全局 denominator，
并对 DDP gradient averaging 做对应 scaling。loss function 等价于：

```text
loss_numerator = sum(token_loss * token_weight)
loss_denominator = sum(token_weight)
```

trainer 必须基于实际 `labels/loss_scale` 得到 optimizer batch 的全局 denominator，并保证
DDP/gradient accumulation 后的梯度等价于：

```text
global_loss = global_numerator / global_denominator
```

实现消费实际 collator tensor，不把估算 cost 当作 loss 真源。HF 在 backward 前收集本 optimizer
batch 的 microbatches；Shaft 汇总 causal shift 后的有效 token/loss weight，在 data ranks 间求和，
再让各 local numerator 对同一个 denominator 归一化。单测锁定了不同 microbatch 切分的 loss/gradient
等价，2-rank smoke 覆盖真实 Trainer 主链。DPO/PPO/GRPO 仍沿各自 trainer 语义，不在本阶段顺带改写。

## 9. Sequence packing 边界

### 9.1 通用契约

- 只打包完整样本；不实现 wrapped sample splitting。
- 每个 segment 的 position 语义独立。
- segment 起始 label 必须阻止前一 segment 预测当前首 token。
- attention backend 必须证明无跨 segment attention。
- eval 默认不 packing，便于指标和样本输出保持可追踪。
- 不支持的模型/backend 必须在启动前 fail fast。

### 9.2 模型族 policy

通用 collator 只构造 segment/offset 结构。模型差异由 `ShaftModelAdapter` 下的 packing capability/policy
处理：

- 普通 causal LM：1D reset position IDs 或显式 varlen metadata；
- Qwen3VL：segment-local M-RoPE、`mm_token_type_ids`、image/video grid 和 visual feature 对齐；
- 其他模型族：由各自 policy 声明位置编码与多模态字段规则。

不得把 Qwen M-RoPE 算法复制到通用 data collator。优先复用上游模型公开/可验证的 position 计算能力。

### 9.3 首版支持矩阵

| 路径 | Cost bucketing | Dynamic batch | Packing |
|---|---:|---:|---:|
| Qwen3VL SFT | 首版 | 第二阶段 | 第三阶段 |
| Qwen3VL DPO | 后续 | 后续 | 暂缓 |
| GRPO | 保持 grouped sampler | 暂缓 | 暂缓 |
| PPO | 暂缓 | 暂缓 | 暂缓 |

## 10. DDP、FSDP 与 context parallel

### 10.1 Data-parallel topology

BatchPlan 必须基于 data-parallel rank/world size，而不是永久假设 global rank/world size。DDP/FSDP
当前 `dp_size == world_size`，未来 `DP x CP` 时则不同。

同一个 global microstep：

- 每个 DP rank 消费不同 local microbatch；
- 每个 CP group 内所有 rank 消费同一个 local microbatch；
- global sample count 只按 DP replicas 计数，不能乘 CP size。

### 10.2 Context parallel

CP 是静态 job topology，第一版固定 `cp_size=1`。只有当单条样本无法在单卡容纳，或长上下文已经被
profiling 证明是主瓶颈时，才建立独立 long-context 配置/stage。

未来 CP 需要同时解决：

- sample dispatch 使用 data rank；
- labels 在 sequence shard 前完成 shift；
- Qwen 多维 position IDs 与 visual masks 正确切分；
- vision feature 与本地 image-token positions 对齐；
- checkpoint、日志、sample count 和 loss denominator 使用 `DP x CP` 语义。

CP 不用于给偶发长样本动态“加两张卡”，也不用于解决短样本 padding。

## 11. Duration、resume 与确定性

### 11.1 Duration

- `duration.unit=steps` 继续表示 optimizer step 数。
- `duration.unit=epochs` 继续作为有限 source coverage/HF 兼容单位。
- Phase 1 cost-aware 仍保持固定 cardinality，sample plan budget继续由现有 global batch 公式派生。
- Phase 2 动态 microbatch 才允许 optimizer step 的 sample 数变化，并改由 BatchPlan 驱动 draw cursor。
- scheduler/warmup 仍以 optimizer step 为单位；token-bounded duration 作为独立后续能力。

### 11.2 Resume state

精确恢复至少需要持久化或可重建：

- SamplePlan fingerprint 和 global draw cursor；
- source snapshot 与 cost manifest fingerprint；
- batch planner version、seed、planning window index；
- optimizer step 和该 step 内 microstep cursor；
- pack policy/backend signature；
- DP/CP topology signature。

计划应尽量按 window 惰性、确定性重建，不把全训练期 batch tuple 全量写入 checkpoint。

Phase 1 已持久化 `shaft_batch_planning_signature.json`，绑定 source snapshot、planning-safe prompt
transform、processor/template、SamplePlan、planner version/window/seed、batch/gradient accumulation 和
DP topology。每个 `checkpoint-*` 都保存同一签名，resume 前先校验 geometry，再校验完整成本签名。
相同 horizon 的 HF mid-run skip 可以精确恢复；改变 duration、topology、data 或 processor/template
会 fail fast。把已有 run 延长为更多 steps 暂不视为 exact resume，应从权重启动新 run；支持可扩展
window cursor 属于 Phase 2。
为避免启动时全量读取图片字节，当前图像资产 manifest 只扫描 SamplePlan horizon 实际引用的唯一
canonical image header，绑定 path、stat/inode、宽高以及 Transformers/patch estimator 版本，并复用
扫描所得尺寸。它不哈希图片内容；同路径替换或改变宽高会使 resume 失败，但训练数据资产仍必须按
不可变 snapshot 管理。多 rank 共享 sidecar/mmap 仍是百万级数据的 Phase 2 工作。

## 12. 配置结构

Phase 1 已开放以下字段：

```yaml
data:
  batching:
    strategy: cost_aware          # fixed | cost_aware
    planning_window: 512
    image_size_cache_size: 8192
```

当前 `cost_aware` 只支持 `SFT + duration.unit=steps` 以及声明 exact image-cost 的模型 policy。生产主链
是 Qwen3VL；共享 `qwen_vl` policy 的 Qwen3.5/3.6 processor cost contract 也已通过 integration，但尚未
替代 Qwen3VL 主训练验收。它保持 `per_device_train_batch_size` 固定，只改变 planning window 内的执行
顺序；配置与精确语义见 `docs/config_reference.md`。

下面是 Phase 2/3 的目标结构，**尚未加入 schema，也不允许 YAML 提前使用**：

```yaml
data:
  batching:
    max_samples_per_microbatch: 8
    max_padded_tokens: 8192
    max_vision_patches: null
    rank_balance: true
    packing:
      enabled: false
      max_tokens_per_block: 8192
      whole_sample_only: true

train:
  optimizer_batch:
    target_samples: null
    target_supervised_tokens: null
    loss_normalization: supervised_tokens
```

约束建议：

- `target_samples` 与 `target_supervised_tokens` 最多显式设置一个。
- 未显式设置时，第一版可从现有 global batch 公式派生兼容目标，但 planner 仍是执行真源。
- `per_device_train_batch_size` 在 `strategy=fixed` 下保持原语义；cost-aware 模式下只可作为显式
  `max_samples_per_microbatch` 的兼容默认值，normalize 必须给出清晰日志。
- 新字段正式落地时必须同步 schema、normalize、配置文档和消费测试。

## 13. Observability

至少记录以下 per-microstep 和聚合指标：

- logical samples、supervised tokens、LLM tokens、vision patches；
- padded token slots、padding ratio、packing fill ratio；
- 每个 rank 的 estimated cost 和 `max/min` skew；
- dataloader wait、processor time、step time p50/p95/p99；
- useful tokens/s、supervised tokens/s、images/s；
- CUDA allocated/reserved/peak memory；
- 各 source 的 sample/token/compute ratios；
- fallback、oversize sample 和 cost-manifest miss 计数。

没有上述基线，不以单一 samples/s 判断 packing 或 dynamic batching 是否成功。

## 14. 分阶段实施

### Phase 0：观测与 cost contract

- **已实现**：定义 `ShaftSampleCost`、provider fingerprint 和 runtime header/cache 入口。
- **已实现**：增加首 window 与 cycle aggregate 的 padding、supervised token/loss weight、vision patch、
  inexact count、rank skew 和 cost-planning wall time。
- **已实现**：Qwen VL 通过 HF image processor 的 `get_number_of_image_patches()` 取得上游 patch 真源；
  真实 Qwen3VL/Qwen3.6 processor integration 验证 token、causal supervision 与 vision patch 精确对齐。
- 待 profiler：补充 processor time、dataloader wait、step p95 与实测 compute/memory 拟合。

### Phase 1：固定样本数的成本分桶

- **已实现**：有界 planning window、文本/vision 二维 cost bucketing 和固定 cardinality。
- **已实现**：BatchPlan 在全局 DP rank 层配对相近成本，并复用 HF BatchSampler/Accelerate 分发。
- **已实现**：确定性、mixing multiset、rank slicing、真实 2-rank Trainer/DataLoader、同 horizon
  checkpoint/resume 与 horizon-change fail-fast 测试。
- **已实现**：SFT optimizer-batch global numerator/denominator，为固定和未来动态 microbatch 共用。
- 当前限制：只支持单图 SFT record；模型必须声明精确 image-cost policy；step sample budget 必须形成
  完整 global microsteps。每个 data rank 当前仍会重建整个 global CostPlan，CPU/tokenization/image-header
  工作随 DP world size 放大；离线/mmap CostPlan 是进入百万级共享存储前的后续 P1。

### Phase 2：动态 cost-budget microbatch

- 允许 local microbatch 样本数变化。
- 引入 optimizer-batch draw plan，复用已落地的全局 numerator/denominator loss。
- 调整 step sample budget、progress、resume 和 throughput 统计语义。
- 先支持 SFT；eval 保持 fixed batch。

### Phase 3：Qwen3VL SFT padding-free packing

- 增加通用 PackPlan 和模型 packing policy。
- 只支持 whole-sample、FlashAttention varlen、SFT train path。
- 完成 packed/unpacked logits、loss、gradient 和无泄漏等价性测试。

### Phase 4：长上下文与 CP 评估

- 在 8k/16k/32k 代表性分布上确认单样本 OOM/attention 瓶颈。
- 只有收益成立时才设计 `DP x CP` 正式配置和 Qwen3VL decoder-boundary integration。
- CP 作为独立 feature，不阻塞 Phase 0-3。

## 15. 测试与验收矩阵

### 15.1 Unit / component

- batch planner 不丢、不重、不替换 SamplePlan refs；
- planning window 边界和 source multiset 保持；
- prompt variant 与 cost metadata 使用相同 draw context；
- padded/packed token 和 vision budgets 永不超限；
- oversize sample 有明确单独处理或失败策略；
- 相同 seed/fingerprint/topology 生成相同 BatchPlan；
- 不同 DP rank 取得同一全局计划的正确 local slice；
- resume cursor 在 optimizer step 内恢复到相同 microstep。

### 15.2 Loss contract

- 同一 optimizer batch 的不同 microbatch 切分得到等价 loss/gradient；
- rank 间 token 数不同仍等价于单卡 global numerator/denominator；
- `loss_scale`、ignore index、segment first token 和空监督样本正确处理；
- fixed batch 路径保持现有结果。

### 15.3 Packing contract

- unpacked 与 packed 的逐样本有效 logits/loss 在容差内一致；
- 修改 segment A 不得改变 segment B 的 logits；
- 单图、多图、多轮、不同 prompt variant 的 grid/position 对齐；
- 不支持的 attention backend/model policy 启动前失败；
- eval 默认不 packing。

### 15.4 Runtime

- CPU smoke 验证装配和最短 train/eval；
- 2-GPU DDP 验证 rank balance、loss parity、checkpoint/resume；
- 真实 Qwen3VL canary 对比 padding ratio、p95 step、tokens/s、显存峰值和 OOM；
- Phase 3 再增加 FlashAttention packing integration。

## 16. 模块落点

建议的职责落点，不要求一次性创建所有文件：

- `data`
  - sample cost metadata/cache；
  - planning window、cost bucketing、BatchPlan/PackPlan 边界数据结构；
  - batch sampler 与 collator 物理组装。
- `model`
  - packing capability；
  - position/M-RoPE 和多模态 payload policy。
- `training`
  - optimizer step/horizon 与 BatchPlan request；
  - optimizer-batch execution；
  - global loss numerator/denominator；
  - accumulation、DDP scaling、resume cursor 和运行指标。
- `pipeline`
  - 只装配 config、planner、collator、model policy 和 trainer。
- `config`
  - schema、normalize、互斥/兼容校验；不推导运行时 BatchPlan。

## 17. 完成定义

本 feature 只有在以下条件同时成立时才算完成：

1. mixing、duration、loss、resume 和 batch 语义各只有一个真源；
2. fixed batch 行为保持兼容；
3. cost-aware 模式不改变 mixer 已选 sample multiset；
4. 2-GPU DDP 的 loss/gradient 与单卡参考等价；
5. packing 通过无跨样本泄漏测试；
6. 真实 canary 证明 p95 step、有效 tokens/s 或显存至少一项有实质改善，且无稳定性回退；
7. 配置、架构、模块参考、测试文档和开发知识完成同步。
