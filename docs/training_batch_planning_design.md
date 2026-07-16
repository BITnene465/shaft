# Shaft batch planning design

状态：**bounded grouping、length grouping、whole-sample greedy packing、Qwen3VL 与
Qwen3.5/3.6 image-SFT varlen、committed efficiency telemetry 已实现；context parallel 不在本轮范围内**

## 1. 问题与设计结论

多模态 SFT 的文本长度、图像 patch 数和图片解码成本都有长尾。普通随机 batch 会产生 local padding 和
DDP rank 等待；但在训练前物化完整 CostPlan 又会扫描整个 horizon，并把 duration、mixing、batch geometry
和 resume 绑定成一个昂贵的全局规划问题。

当前实现共用一个有界 lookahead planner：bounded-cost padded 路径允许 fixed 或 token-budget cardinality；
length 路径允许 fixed physical packs，并可选 whole-sample greedy packing + varlen。token-budget 本身不是
sequence packing，它只改变一个 padded local microbatch 内有几个 physical packs。

核心原则：

- `train.per_device_train_batch_size` 是唯一 local physical-pack count 配置：fixed 时是精确值，token-budget
  时是上限。
- mixing 决定训练哪些 draw；grouping 只在有界 buffer 内重排，不丢、不复制、不改 source 权重。
- 每个 DP rank 在每个 optimizer step 都执行相同数量的 microsteps；token-budget 允许各 rank 的 local
  cardinality 不同，但每个 local batch 都非空且不超过统一上限。
- token/resource budget 是 hard cap；在 token-budget 模式下，它同时决定实际 local cardinality。
- startup 只估算一个 buffer/GA frame，与 max steps 无关。
- checkpoint 只提交模型真正完成的 optimizer boundary，不保存 DataLoader 预取后的 live cursor。
- SFT loss 使用一个 optimizer frame 内跨 GA/DP 的真实有效 token denominator，避免 variable local batch
  改变样本/token 权重。
- HF 继续拥有模型、optimizer、scheduler、GA、checkpoint 与 HF/PEFT 导出语义。

## 2. 分层配置语义

数据入口先按职责分层，batching 再拆成四个互不覆盖的轴：

```text
data
├── sources / catalog
├── schedule
│   ├── mixing
│   └── shuffle
├── transforms
│   └── prompt_sampling
└── batching
    ├── grouping
    ├── cardinality
    ├── packing
    └── layout
```

```text
grouping     none | length | bounded_cost
cardinality  fixed | token_budget
packing.mode none | greedy
layout       padded | varlen
```

当前代码支持的执行组合：

```text
grouping = none,         cardinality = fixed,              packing.mode = none, layout = padded
grouping = length,       cardinality = fixed,              packing.mode = none, layout = padded
grouping = length,       cardinality = fixed,              packing.mode = none, layout = varlen
grouping = length,       cardinality = fixed,              packing.mode = greedy, layout = varlen
grouping = bounded_cost, cardinality = fixed|token_budget, packing.mode = none, layout = padded
```

首轮兼容矩阵：

```text
grouping       cardinality   packing   layout    首轮状态
none           fixed         none      padded    已实现
length         fixed         none      padded    已实现
length         fixed         none      varlen    已实现，Qwen3VL / Qwen3.5 / Qwen3.6 image SFT
length         fixed         greedy    varlen    已实现，Qwen3VL / Qwen3.5 / Qwen3.6 image SFT
bounded_cost   fixed         none      padded    已实现
bounded_cost   token_budget  none      padded    已实现
```

首轮明确拒绝 `packing=greedy + layout=padded`、`packing=greedy + cardinality=token_budget`，以及未经
执行策略声明的模型族/attention backend。`bounded_cost + greedy` 不作为首轮发布门槛；先让通用的
length/BFD 路线稳定，再决定是否需要把多维 cost policy 与 packing 叠加，避免一次引入两个动态容量来源。

这里没有“batching 覆盖 schedule”或“layout 覆盖 packing”的优先级：schedule 选择 logical draw，transform
生成该 draw 的训练视图，grouping/cardinality 决定 physical local batch，packing 决定是否把多条逻辑序列
装入一个物理序列，layout 决定张量和 attention backend 的表示。预留值在 runtime 完整实现前必须 fail
fast，禁止静默退回 padded/fixed。旧 `data.batching.strategy`、
`max_samples_per_microbatch`、`max_padded_tokens`、`max_vision_patches`、full-horizon CostPlan 和
`train.optimizer_batch` 已删除，不做隐式迁移。

`ShaftBatchContract` 是 normalize 之后的单一运行时真源：

```text
fixed local packs = per_device_train_batch_size
token-budget local packs = [1, per_device_train_batch_size]
global pack range = local range * data_world_size
optimizer pack range = global range * gradient_accumulation_steps
```

这里的 sample-count 只在 `packing=none` 时等于 logical sequence 数。启用 packing 后统一使用下列术语：

```text
logical segment       一条原始训练样本形成的完整序列，不拆分、不环绕
physical pack         一个容量不超过 data.max_length 的物理序列，可含多个 segment
local microbatch      一个 rank 本次计划得到的 physical pack 集合
global microstep      所有 data-parallel rank 的 local microbatch
optimizer frame       gradient_accumulation_steps 个 global microstep
```

`cardinality=fixed` 时，`train.per_device_train_batch_size` 表示每个 rank 的精确 **physical pack 数**；
`cardinality=token_budget` 时表示上限。当 `packing=none` 时一个 pack 恰好只有一个 segment，因此与旧
sample batch size 完全等价；当 `packing=greedy` 时 logical sample 数动态变化，但配置语义不变。
`layout=varlen` 可以把多个 physical pack 在张量层展平为 `[1, total_tokens]`；这个执行张量的 batch 维为
1，不会反向改变计划层的 physical pack 数。

`data.max_length` 是 logical row 截断上限，也是 greedy physical pack 的容量上限，沿用
Transformers/TRL/ms-swift 的单一 sequence-length 语义，不再新增 `packing_length` 或
`max_tokens_per_pack` 同义字段。length 路径的 local token hard cap 由
`per_device_train_batch_size * data.max_length` 派生；显式
`data.batching.max_tokens_per_microbatch` 只用于 bounded-cost 路径。

pipeline、sampler spec、日志、run metadata 和 checkpoint 必须从同一个 contract 构造。任何 spec 漂移都在
startup 明确失败。

## 3. 数据真源

### 3.1 SampleSchedule

`ShaftSampleSchedule.ref_at(draw_id)` 是 horizon-independent 的 draw 映射：

- fingerprint 绑定 source names/sizes/weights、mix strategy、shuffle、seed、weighted ticket 算法版本、salt
  与 canonical base-block digest；
- 不绑定 max steps 或有限 plan 长度；
- `draw_id` 同时驱动 prompt rotation 与 deterministic online transform；
- bounded weighted mixing 要求 `shuffle=true`。

`weighted + shuffle=true` 不再为每个 draw 独立随机选择 row。source 选择使用固定配额 ticket block；常见
简单比例约为 4K。最多 64 个 source 时完整搜索 16K 上限内的 Hamilton 解；更大 catalog 使用最多 32 个
denominator-derived 候选作为 fast path，若没有满足 5% 每源相对误差合同的候选，再用 quota 可接受区间
筛出全部可能 block，并用 O(source count) Hamilton rank predicate 完整验证。因此 fast path 不会误拒
合法权重，失败路径也不会对每个 block 重复排序整个 catalog。16K 内仍无法满足时直接拒绝配置。一个
seed-specific base block 只构造一次；rotation 每 256 blocks 用 keyed SplitMix64 更换 phase，group 内使用
与 block/full-cycle 和 `1..64` rank modulus 互质的短 counter step。这样任意 block 的 source quota 一致，
有限 block 前缀还有确定性的 rank-count discrepancy 上界，并可从
`draw_id` 通过每源 ticket-position rank 查询，以 O(log quota) 求出该 source 的全局 occurrence。occurrence
按 source size 划分为独立 keyed-Feistel row permutation cycle：一轮耗尽前无重复，耗尽后才进入下一轮。
它不物化训练 horizon，也不保存 checkpoint cursor；fixed plan fingerprint 与 planned schedule fingerprint
都绑定 v3 语义，旧 v2 rotation/quota checkpoint 会 fail closed。

普通 map-style 路径仍可使用有限 `ShaftSamplePlan`。bounded SFT 直接消费 schedule，不构造 duration-sized
Python 索引或 CostPlan。

### 3.2 SampleCost

buffer entry 只保存轻量数据：

```text
SampleRef + {
  llm_tokens,
  supervised_tokens,
  vision_patches,
  loss_weight_sum,
  exact
}
```

- 图像 resize、patch 和 processed token layout 由模型 `ProcessorPolicy` 定义。
- prompt、target、EOS、截断、causal shift 和 loss scale 由 Template 定义。
- provider 只读取图片 header，不解码图片；sample cost 与 header 使用有界 LRU。
- fingerprint 绑定 record、planning-safe transform、tokenizer、processor/template semantics 和显式
  `media_snapshot_id`。

planner 不复制模型或模板语义。无法给出 exact cost 的模型不能启用 hard-budget grouping。

## 4. Planner 与 policy 边界

计划层收敛为一条管线，而不是为 length、packing 再各写一套 sampler：

```text
ShaftBatchPlanner
├── grouping policy
│   ├── none
│   ├── length
│   └── bounded_cost
├── cardinality policy
│   ├── fixed
│   └── token_budget
├── packing policy
│   ├── none
│   └── greedy
└── immutable plan
    └── global microstep
        └── rank local microbatch
            └── physical pack
                └── logical segment
```

immutable plan 是唯一结构真源。兼容属性可以从 pack/segment 层派生 flat `sample_refs`、成本与计数，
但 pipeline、sampler、collator 不得各自维护另一份映射。Dataset 只接收轻量 planned ref，并在 online
transform 完成后附加一次性 `_batch_context`：`global_microstep/plan_fingerprint/local_batch_id/pack_index/
segment_index/pack_segment_count`。原始 record、JSONL 和 transform 输出均不写回、不污染。collator 验证
缺失、重复、跨 plan 混入和非连续索引，不保留跨调用 carry buffer，因此 `num_workers`、prefetch 和 worker
调度不会改变 pack 结果。

### 4.1 Length grouping

length policy 借用 Transformers/ms-swift 的 sortish 思路，而不做全数据集排序：

1. 从 horizon-independent schedule 惰性补满 bounded lookahead。
2. 使用 exact `llm_tokens`，在窗口内按长度降序和稳定 draw id 产生候选优先级；grouping 不拥有消费
   数量，也不改变持久 buffer 顺序。
3. `packing=none` 时才消费恰好 `world_size * per_device_train_batch_size` 条，选择包含 FIFO 最老 draw 的
   相邻长度块，保证有限等待。
4. `packing=greedy` 时 packer 从整个候选窗口消费动态数量的 segment，FIFO 最老 draw 仍必须进入本轮。
5. 完成 physical packs 后再确定性分成各 rank local microbatch；GA frame balancing 只能移动完整 local
   microbatch，不能拆 pack 或破坏 hard cap。
6. 未消费 draw 恢复原始 FIFO 相对顺序，下一轮继续参与。

这保留了成熟 length grouping 的局部排序性质，又不会把 duration、million-scale 全量索引或一次性 cost
扫描带回 startup。任何 worker 数下，同一 schedule/config/state 都产生相同 logical draw multiset 和计划。

### 4.2 Greedy packing

首轮 greedy 使用 whole-sample best-fit-decreasing（BFD），行为与 TRL/ms-swift 的成熟路线对齐：

- 不拆 sample、不截掉额外 token、不跨 pack wrap，也不丢 oversize sample；单条 row 经正常截断后仍超过
  `data.max_length` 时在首次观察明确失败。
- 每个 global microstep 先创建 `world_size * per_device_train_batch_size` 个非空 physical bins；FIFO 最老
  draw 必须进入本轮，其余 seed 取窗口内最长样本。
- 剩余候选按长度降序，放入“加入后剩余容量最小”的可行 bin，稳定 draw id 负责 tie-break。
- 未装入候选保留在 lookahead；输出 pack 内 segment 按计划顺序固定。
- Qwen3VL greedy 要求显式 local `resource_budgets.vision_patches` hard guard。physical packs 分配到 rank
  时检查该 rank 全部 packs 的 aggregate patches；任何 `exact=false` cost 都拒绝。resource guard 是执行
  安全约束，不属于 grouping 优化目标。
- 例如容量为 8 时，`[8,1,1,1,1,2]` 生成 `[8]` 与 `[2,1,1,1,1]`，而不是把两条普通样本再次
  padding。

packer 算法版本、capacity、grouping/cardinality、world size 与 seed 都进入 batch contract fingerprint。

所有 lazy-planned 路径要求 step duration、horizon-independent schedule、planning-safe transforms、immutable
`media_snapshot_id` 与 exact cost provider。`length/greedy` 共同使用 `buffer_size/cost_cache_size`；greedy
额外要求正整数 `data.max_length`。首轮 greedy 的 local token hard cap 派生为
`per_device_train_batch_size * data.max_length`，不再要求用户同时配置第二个同义 token cap；显式
`max_tokens_per_microbatch` 仍只服务 bounded-cost cardinality。

### 4.3 现有 bounded-cost policy

`ShaftBatchPlanningSpec` 是 duration-independent 的不可变 sampler contract，包含：

- planner version；
- schedule/cost fingerprint；
- DP world size；
- buffer size；
- cardinality policy 与 per-device microbatch 上限；
- `max_tokens_per_microbatch`；
- 通用 `resource_budgets`（当前 Qwen 路径使用 `vision_patches`）；
- seed。

它不包含 max steps、GA、optimizer target 或完整训练 horizon。GA 只在 training adapter 的 planning frame
和 commit 映射中出现。

每个 global microstep：

1. 按 draw id 惰性补满 buffer。
2. 对新观察 sample 验证 exact cost 和单样本 hard guard。
3. 强制选择最老 draw，保证有限等待。
4. 选择成本相近的 rank anchors。fixed 模式为每个 rank 恰好填入
   `per_device_train_batch_size` 条；token-budget 模式在预算内尽量填到该上限。
5. local batch 必须满足：
   - fixed：`count == per_device_train_batch_size`
   - token-budget：`1 <= count <= per_device_train_batch_size`
   - `count * max(llm_tokens) <= max_tokens_per_microbatch`
   - 每个 resource 的 `sum(resource) <= resource_budget`
6. 快路径优先最小化 projected rank load，再以 padding waste 和稳定索引打破平局。
7. 贪心未填满时复用 deterministic bounded full-partition search；fixed 找不到完整分区会失败，
   token-budget 搜索受安全预算限制，找不到时保留已满足 hard cap 的非空可行解。
8. 删除 selected entries；未选 entry 保持 FIFO。

对 fixed `per_device_train_batch_size=1`，local padding 恒为零；grouping 的收益来自让同一 global
microstep 的各 rank 成本相近。对 token-budget `per_device_train_batch_size=2`，长样本可单独形成 B1，
短且相近的样本可形成 B2；这减少空闲算力，但仍是普通 padded batch，不是 packing。

## 5. HF / Accelerate 接口

全局 BatchSampler 顺序：

```text
[planning frame][global microstep][rank]
```

需要 planning 的 length/bounded/packing 路径统一由 `ShaftPlannedBatchSampler` 在 yield 前原子规划一个
`GA * world_size` frame，并在 frame 内按累计成本重新分配 rank batch。data sampler 不解释 optimizer
step；training adapter 把 frame size 设为 GA，callback 用 `global_step * GA` 提交 state。旧 bounded 名称
只允许作为短期 import alias，不能继续成为 pipeline/trainer 的类型分支。

Accelerate contract：

- `batch_size=None`
- `drop_last=True`
- `split_batches=False`
- `even_batches=False`
- sampler length 为 `remaining_steps * GA * world_size`

每个 rank 恰好取得 `remaining_steps * GA` 个 batch。fixed 模式每批 physical pack 数等于
`per_device_train_batch_size`；token-budget 模式每批 physical row 为 1 到该上限，且不同 rank 可以不同。
BatchSampler 对 DataLoader 输出按 `pack_index, segment_index` 展平的 planned refs；collator 再从显式上下文
恢复 pack/segment 层级，禁止根据到达顺序猜测。
`ShaftSFTTrainer` 根据 collate 后真实 `labels/loss_scale` 计算跨 GA/DP 的 global token denominator；
planner cost 只用于容量和排序。

## 6. Resume 与可观察性

DataLoader 可能预取未来 frame，因此 sampler live state 会领先模型：

1. sampler 保存 frame-boundary snapshots；
2. callback 在 `on_step_end` 提交 `global_step * GA` 对应 snapshot；
3. planner spec/committed state 作为 HF `ExportableState` 写入 checkpoint 的 `trainer_state.json`；
4. DDP/native-HF 的 training 层 checkpoint mixin 暂缓 HF rotation；converged wrapper 捕获既有 `on_save` callback
   （包括 telemetry）的 rank-local 异常并汇聚，全部 rank 成功后才由 rank 0 在独立 commit phase 原子发布
   `shaft_checkpoint_commit.json` 并执行 rotation；planned state 与 resume-contract fingerprint 进入 manifest 的 `batch_planning`
   extension，但不复制 sampler state；
5. resume 验证 manifest、全部 rank RNG、optimizer/scheduler、cardinality-bounded emitted count、planner、
   grouping 和 packer version、batch contract、source/media/cost/topology 和 training schedule；
6. sampler 从 committed state 继续，并设置 `ignore_data_skip=true`，避免 HF 二次 skip。

generic planner state 分开记录 logical draw 与 physical capacity：

```text
next_draw_id == emitted_logical_segments + len(buffer)

fixed:
emitted_physical_packs == global_microstep * world_size * per_device_train_batch_size

token_budget:
global_microstep * world_size
<= emitted_physical_packs
<= global_microstep * world_size * per_device_train_batch_size

packing=none:
emitted_logical_segments == emitted_physical_packs
```

greedy 下不得用 physical cardinality 反推 logical segment 数。累计 useful tokens、supervised tokens、vision
patches 也来自实际 emitted segments。

run root 的 `shaft_batching_run_metadata.json` 记录 grouping/cardinality/packing/layout、
`local_pack_count_range`、`global_pack_count[_range]`、`optimizer_pack_count[_range]`、DP/GA、pixel budget、
source weights、buffer/cache/budgets、完整
versioned `batch_contract`、`batch_contract_fingerprint` 和 planner 的 `planner_spec_fingerprint`。
canonical batch contract 同时进入所有 HF checkpoint；四轴、local batch、DP 或 GA 漂移都会拒绝 exact
resume。`cost_cache_size` 只影响 host LRU 命中率，保留在 audit metadata 中但不参与 exact fingerprint。
planned commit extension 还要求 metadata 的 planner fingerprint 与实际 callback spec 一致。
启动日志使用 `[batch-contract]` 展示同一 payload 的关键字段。

本轮 generic planner/state/callback 使用新版本并替换 bounded v3 双轨内核。已有
`shaft-bounded-batching-v3`、`ShaftBoundedBatchingCallback`、旧 batch-contract，以及缺少 canonical generic
callback 的 checkpoint **不做 exact resume 或隐式迁移**；此前 re/re2/v5.1 训练若要继承模型权重，使用
init-from-checkpoint 开新 schedule。只有经独立 replay 证明的迁移器未来才可开放，不能仅靠类名 alias
假装兼容。

## 7. 图像预算

`max_pixels` 是 processor 的单图 resize budget；`resource_budgets.vision_patches` 是一个 local batch 内所有
图片的 aggregate encoder-work guard，两者不是同一字段。

aggregate budget 必须能容纳 processor 允许的最大单样本。Qwen patch-size 16、
`max_pixels=4,000,000` 可能产生约 15,625 个 pre-merge patches，因此 re/re2 使用 16,384。该 guard 防止
多个大图合批，不应用 8,192 之类低于合法单样本的值误杀数据。

## 8. 测试契约

必须覆盖：

- 百万级虚拟 schedule 首个 microstep 只读取一个 buffer，不扫描完整 horizon。
- `8,1,1,1,1,2` 在 batch size 1 时选择成本接近的 global batch，local padding 为零。
- fixed 计划每个 rank cardinality 严格等于 `per_device_train_batch_size`；token-budget 计划覆盖 `1..Bcap`
  （包括 `Bcap > 2`）且不越过 token/resource cap。
- token/resource 多维 hard cap，以及贪心失败但 exact fallback 可找到解的 adversarial case。
- refill 后 emitted logical segments + buffer 精确等于 `[0, next_draw_id)`；physical packs 独立满足
  cardinality 守恒，无丢失、重复或饥饿。
- W/GA 多组合经真实 `BatchSamplerShard` 后各 rank batch 数一致；token-budget 允许 cardinality 不同。
- inexact/text/resource oversize 在首次观察停止。
- state JSON integrity、旧 bounded v3 rejection、batch contract drift、optimizer-boundary alignment。
- worker prefetch 领先时，committed resume stream 与 uninterrupted 一致。
- CPU 2-rank variable local batch 的 global-loss parity，以及 persistent workers 的
  model/optimizer/scheduler/RNG/state exact resume。
- startup 不创建 CostPlan sidecar，不调用全量 summarize/materialize。
- length grouping 的窗口排序、FIFO 等待上界和 draw multiset 守恒。
- BFD golden cases、稳定 tie-break、capacity 边界、oversize fail-fast，以及 worker 数变化不改变 pack 映射。
- `packing=none` 时新层级 plan 与旧 flat batch 完全等价。
- varlen tensor 无 padding、无普通 2D attention mask，每个 segment 首 token 的 `labels=-100`、
  `loss_scale=0`。
- packed 与逐条运行的有效 logits、weighted loss、parameter gradient parity；改变 segment A 不得影响
  segment B。
- 多个单图 segment 的 placeholder、`image_grid_thw` 和 `pixel_values` ranges 顺序保持；交换 segment 的
  metamorphic case 也必须保持对应关系。真正单 segment 多图与视频不属于首轮数据主链。
- Qwen3VL CPU eager/SDPA oracle 与 CUDA FlashAttention 2 canary；错误传入全 1 attention mask 的 negative
  test 必须能够捕获跨 segment 泄漏。
- packed 与 standalone 的 shifted loss numerator/denominator 分别守恒；不只比较最终平均 loss。
- Qwen direct invariants：scalar axis 对每段严格等于 `0..L-1`，后三轴逐元素等于 standalone upstream
  M-RoPE；交换 manifest、删除 grid/patch row 必须在 forward 前失败。
- capability negative matrix：3-axis positions、未 reset scalar positions、`use_cache=True`、非空 past、
  FA2/FLA/causal-conv 实际回退、错误 dtype/device/concrete config、SmokeVLM 和未知模型。

GPU canary 是 release gate，不能用 CPU smoke 冒充。2026-07-13 已在 CUDA 1、2 上用标准 HF tiny
`Qwen3VLForConditionalGeneration` 完成 2-rank DDP + FlashAttention 2 + bf16 的完整 vision/DeepStack
forward/backward、eval、checkpoint 与 checkpoint-1 resume。连续与恢复训练的 model bytes、optimizer、
scheduler、两 rank RNG 和 committed planning manifest 一致；`trainer_state.json` 仅因输出目录不同而保留
不同的 `best_model_checkpoint` 路径。

2026-07-14 又在 CUDA 0、1 上用真实 Qwen3.6 processor 与 tiny `Qwen3_5ForConditionalGeneration`
完成 2-rank DDP + FlashAttention 2 + FLA + causal-conv 的 greedy varlen 多模态 forward/backward、两步训练、
checkpoint、checkpoint-1 exact resume、HF save/export 与 telemetry snapshot 恢复。packed 与 standalone
hybrid-language hidden state parity、完整 vision forward 和 lm-head gradient 均通过；全程未操作
`gpu-holder`。

同日最终 release gate 在不可变源码状态下统一覆盖三组模型契约：

| gate | 执行布局 | 发布验收 |
|---|---|---|
| 真实 Qwen3VL-4B PEFT | 2-rank greedy varlen | fresh/resume、planning completion、telemetry restore、标准 PEFT + processor reload/forward |
| tiny upstream Qwen3.5/3.6 dense | 2-rank padded/varlen | fresh/resume、模型/optimizer/scheduler/RNG 等价、full HF + processor reload/forward |
| tiny upstream Qwen3.5/3.6 MoE | 2-rank padded/varlen | router/expert 结构、fresh/resume、completion、telemetry、full HF reload/forward |

统一命令选择 3 个 opt-in integration gate，结果 3/3 通过；它证明当前 Qwen variant 契约与发布链闭合，
不替代生产数据上的吞吐、峰值显存和长时间稳定性 profiler。

## 9. Varlen layout

layout 层只负责把已经确定的 logical rows 表示成模型输入，不参与 draw 选择或 pack 决策：

```text
logical supervised rows
├── padded -> [physical_rows, max_row_tokens] + 2D attention_mask
└── varlen -> [1, sum(segment_tokens)] + segment_lengths（私有元数据）
```

varlen collator 的不变量：

- 按 `physical pack -> logical segment` 的计划顺序拼接 `input_ids`、`labels`、`loss_scale` 和
  `mm_token_type_ids`。
- 不向模型传普通 2D `attention_mask`；全 1 mask 会让 Qwen3VL 的 FlashAttention 走 padding unpad 路径，
  无法表达 segment boundary。
- 每个 segment 首 token 强制 `labels=-100`、`loss_scale=0`，阻断跨样本 next-token target。
- processor 仍只对 logical batch 调用一次；Qwen `ProcessorPolicy` 同时构造 typed media manifest。首轮每个
  segment 恰好一张图，manifest 显式记录 image-grid 与 raw pixel-patch 的半开区间。所有 ranges 必须连续、
  不重叠、完整覆盖，并满足 `sum(prod(image_grid_thw)) == pixel_values.shape[0]` 以及 image placeholder token
  数等于 `prod(grid) / merge_size²`。execution policy 只消费 manifest，不重复推导 processor 语义。
- collator 输出的 `_shaft_varlen_layout` 与 `_shaft_media_manifest` 私有键必须在 trainer 调用模型前被
  execution policy 消费并移除，不能泄漏给任意 HF forward。

首轮 varlen 都经 length planned path 获得显式上下文。`packing=none` 时每个 logical row 是 singleton pack，
只是去掉 local padding；`packing=greedy` 则进一步改变 logical segments 到 physical packs 的计划映射。
普通 `grouping=none` 的 identity planned path 留待后续，collator 不根据到达顺序猜训练 pack。

## 10. Model/backend execution contract

不能用一个布尔 `supports_packing` 假装模型族通用。模型注册项提供 `SequenceExecutionPolicy`。normalize
在加载权重前拒绝不支持的 model type、device、dtype、attention backend 与 distributed strategy；模型加载
后，pipeline 立即由同一 adapter policy 校验 concrete HF class 与实际保留的 backend，再进入 planner 和
DataLoader：

```text
model family + concrete HF config
+ transformers version
+ attention implementation
+ device/backend/dtype
+ packing/layout
-> position protocol
 + segment-isolation protocol
 + media-manifest protocol
 + dependency/version requirements
```

只有上述两阶段检查都成功后才允许训练。请求 FlashAttention 2 或 varlen 时，依赖缺失、实现被 HF 静默回退、
未知 remote-code forward 或不支持的分布式策略必须 fail closed，不能回退为 padded。

当前 allowlist 是闭集：

```text
Qwen3VL image SFT + CPU + eager/SDPA + fp32/bf16   correctness oracle only
Qwen3VL image SFT + CUDA + FlashAttention 2
                    + bf16/fp16 + DDP              release path
Qwen3.5/3.6 image SFT + CUDA + FlashAttention 2
                    + FLA + causal-conv
                    + bf16/fp16 + DDP              release path
```

CUDA SDPA/eager、CUDA fp32 FA2、FSDP、DeepSpeed、torch.compile、SmokeVLM、未知或
remote-code concrete class 都拒绝。TrainingArguments 中 `per_device_train_batch_size` 仍记录 physical pack
数，DDP DataLoader 每次仍产生一个 local microbatch；模型看到的 varlen tensor batch axis 固定为 1。
首轮拒绝会从 tensor batch axis 推导 micro-batch 的 DeepSpeed/FSDP 路径。

Qwen3VL 首轮策略：

1. execution policy 的 `prepare_training_inputs` 通过经过 class/config 验证的 base-model resolver 找到外层
   `Qwen3VLModel`（不是 `Qwen3VLTextModel` 或 `Qwen3VLForConditionalGeneration`）并调用其
   `get_rope_index`。DDP/PEFT wrapper 必须由统一 resolver 解包，禁止散落 `.model.model` 猜测。
2. 位置与 media manifest 校验发生在 Trainer 把 batch 搬到 GPU 前，使用 host tensors；DataLoader worker
   不持有模型，也不在 GPU 上逐 segment 执行 upstream `.tolist()` Python loop。
3. 把每段独立计算的三轴 M-RoPE 拼接；额外在最前面加入每段从 0 重置的 scalar position row，最终
   `position_ids` 为 `[4, 1, total_tokens]`。第一轴供 Transformers 构造 block-causal boundary，后三轴供
   Qwen M-RoPE。
4. 省略 `attention_mask`，强制 `use_cache=False`。CUDA FlashAttention 2 由 reset scalar positions 推导
   varlen 边界。
5. 逐段校验 modality run、grid row 和 patch slice 数量，任何 media manifest 漂移都在 forward 前失败。

Qwen3.5/3.6 使用独立 hybrid policy：除同样的 segment-local 四轴 M-RoPE 外，还生成 `seq_idx`、
`cu_seq_lens_q/k` 与 max lengths，分别隔离 causal-conv、GatedDeltaNet/FLA 和 full attention。Transformers
5.10.1 会把语言侧 kwargs 同时传给 vision encoder，因此 policy 在模型 runtime 安装版本化 media-kwarg
filter，只从 vision feature 调用移除这些语言字段；trainer/collator 不包含模型名分支。CPU fallback 不满足
linear state isolation，明确拒绝。Qwen3.6 当前是 HF `qwen3_5` architecture 的产品 alias。

模型 variant 不再只靠目录 basename：本地目录、HF cache 或 Hub repo 的 config 都先解析为
`ResolvedModelDescriptor`，再按 `hf_model_type/architecture` 选择 dense 或 MoE sharding profile；
`revision/cache_dir/local_files_only` 与 loader 使用同一组参数。未知且无法取得 config 的多变体模型会
fail closed，避免 MoE 静默落入 dense policy。PEFT 初始化额外由 `ResolvedAdapterInit` 绑定 canonical
adapter config、base artifact 与权重 manifest；builder 不再二次推导 adapter 身份。

## 11. Loss 与模型调用

Shaft 继续用 collate 后真实 `labels/loss_scale` 计算跨 GA/DP 的 global denominator。varlen 只改变张量
表示，不改变 token 权重。只有当 resolved loss/execution contract 明确声明 Shaft 拥有 causal-LM CE、模型
返回 full logits 时，trainer 才在调用 HF model 前移除 `labels`，避免 HF 先计算一次内置 CE、Shaft 又计算
一次 weighted CE。segment boundary 的监督真源只有 `labels=-100` 与 `loss_scale=0`，不再维护第三份 mask。

必须分别记录 physical packs、logical segments、useful tokens、padding tokens、segments/pack、planner CPU
time 和 rank skew。不能再用 `batch_size` 一个数字同时表示 Tensor batch dim、pack 数和 logical sample 数。

## 12. Committed efficiency telemetry

`[batch-plan-summary]` 只描述 producer 已规划的窗口，可能包含 DataLoader prefetch 后最终未执行的 frame；它
不是吞吐真源。实际效率链分三层：

```text
SFTCollator -> _shaft_batch_stats（processor 后实际 tensor/resource）
HF Trainer  -> stage complete GA frame -> successful on_step_end commit
observability -> Trainer.log/W&B + shaft_training_efficiency.json
```

- padded 的 useful/materialized token 分别来自 `attention_mask.sum()` 与 `input_ids.numel()`；varlen 来自实际
  flattened tensor/layout。sequence length 的 sum/square 按 logical segment 统计，不能把整个 flat row 当成
  一条样本。supervised token 与训练 loss 一样使用 shifted labels，vision patches 来自 processor media
  manifest，不使用 planner estimate；optional weighted supervision 同时记录 microbatch coverage，部分覆盖
  不会冒充完整 mass。
- `get_batch_samples` 分开记录 iterator acquire（含 worker/IPC/collate 等待）与 global denominator prepare；
  `training_step` 记录 host wall time。CUDA events 从该 optimizer frame 的第一次 training step 延续到 optimizer
  完成，记录完整 device timeline；optimizer hook 另保留 host wall 诊断值。
- 只有 `on_step_end` 提交。forward/backward/OOM 异常与预取领先的数据不会进入 totals；AMP skipped update
  单独记录 `update_applied=false`。
- DDP 只在 logging/final window 做固定 numeric tensor collective：counts 求和，duration 求 rank min/mean/max，
  critical path 按 step 取最慢 rank，并累计该 critical rank 的 acquire/prepare/train/optimizer components；
  peak CUDA allocated/reserved memory 取所有 rank 的最大值；显存窗口在 HF `on_train_begin`（完成
  model/optimizer/resume state 装配后）重置，exact resume 把 checkpoint 历史峰值与当前窗口取 MAX。任一 rank
  缺少完整历史/可用窗口时结果为 `null`，不能低报或冒充 0。不在每 step 使用 object gather。monitor 启动时
  还会验证所有 rank 的 typed contract fingerprint 一致。
- checkpoint 内写每 rank 的可选 telemetry snapshot。rank 0 先 revoke 旧 snapshot set，再由所有 rank 写入
  同一 generation，最后原子发布 set manifest。三个阶段都在本地 I/O 后通过固定 tensor 汇合成功状态；
  任一 rank 失败时所有 rank 同步退出该提交并撤销 incomplete set，不在 fallible I/O 后直接进入 barrier。
  `persist=false` 与 telemetry disabled 也会撤销同名 checkpoint 中的旧 set。只有所有 rank 的 snapshot 都存在且
  generation/step/world/rank/span/typed training contract 对齐时 resume 才继续完整历史；任一 rank 缺失或不兼容时全体
  一致降级，summary 标记 `complete_history=false` 并从 checkpoint step 重新覆盖，避免 asymmetric collective、
  stale root summary 或重复累计。CUDA event coverage 同样要求每个 committed frame、每个 rank 完整一致。
- `scripts/compare_efficiency.py RUN...` 比较相同训练契约下的 committed summary，可用于 fixed padded、length
  padded、length varlen、greedy varlen 与 bounded token-budget 的 A/B。默认 identity 包含模型、数据/source、
  logical sample stream、DP/GA、optimizer/scheduler、measurement protocol、timing mode 与 step span，只允许
  batch/sequence fingerprints 变化。默认 exact-workload 还校验 logical token/segment/vision totals 与 sequence
  length sum/square-sum，防止把明显不同的 attention workload 认证为公平 padded/varlen A/B；packing 在相同 step span
  消费不同 logical workload 时可用 `--allow-workload-variation` 做 capacity comparison，仍要求
  update/microbatch/physical-pack/coverage 完全一致，并分别报告 token/segment/vision rate；它不能解释为
  等工作量 speedup。
  exact-resume 另行使用包含 finite-plan horizon 的 sample execution identity。
  `--allow-incompatible` 是显式逃生口。fixed path 的未版本化 transform/缺失 media snapshot 不改变训练
  可用性，但 source identity 会标 incomplete 并被默认比较拒绝。实验启动仍由普通训练 CLI 负责。
  comparator 只接受 v3 root summary；旧 v2 不迁移、不参与公平比较。

## 13. Context parallel 边界

context parallel 是 job-level topology，只处理单条不可分割的超长上下文；它不是普通短样本 padding、
sequence packing 或 DDP straggler 的替代方案。它需要独立的 topology、position/attention、checkpoint 和
通信契约，不与本轮三项能力捆绑实现。
