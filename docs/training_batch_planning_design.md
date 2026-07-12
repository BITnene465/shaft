# Shaft bounded cost-aware batching design

状态：**bounded-buffer runtime 已实现；sequence packing / context parallel 待实现**

## 1. 目标

多模态 SFT 的文本长度和图像 patch 数同时波动。固定 batch 会产生 padding、单卡长尾和 DDP 等待；但把
全训练期样本成本预先物化成 CostPlan，又会在第一步前扫描数十万 draw，并把 duration、mixing、resume 和
batch geometry 绑成一个难以维护的多维分箱问题。

当前方案采用与 Megatron Energon/NeMo 动态 batching 相同的核心原则：固定资源预算，让每个 optimizer
step 的样本数自然变化；只保留有界 lookahead，不解全训练 horizon。

设计目标：

- 多 rank startup 只验证一个 buffer；首个 forward 前的完整 planning-frame 成本调用仍有显式常数上界，
  与 max steps 无关。
- mixing 决定训练什么；batching 只重排 bounded buffer，不丢、不复制、不改 source 权重。
- 每个 DP rank 的 local batch 数一致，样本数可变。
- sample、padded text、vision patch 都有显式上限。
- checkpoint resume 从已完成 optimizer step 精确继续，不保存预取 live cursor。
- HF 继续拥有模型、optimizer、scheduler、GA、保存与 HF/PEFT 导出语义。

非目标：

- 本阶段不做 sequence packing。
- 本阶段不做 context parallel。
- 不把 Energon 直接作为 HF DataLoader backend；它有自己的 rank/worker sharding 与 loader state，直接接入
  会和 Accelerate 双重分片。本实现是遵循相同 contract 的薄 PyTorch BatchSampler。

## 2. 被删除的旧链

旧运行时：

```text
max_steps
  -> full-horizon SamplePlan
  -> full-horizon mmap CostPlan
  -> exact target_samples / target_supervised_tokens
  -> world_size * GA 固定槽联合分箱
  -> greedy + DFS fallback
```

它同时要求固定样本总数、固定槽数、三维硬预算、rank balance 和 exact resume，本质是多维 bin packing。
`planning_window` 只限制分箱范围，没有限制 CostPlan 物化范围；因此 640k draw 会先花十几分钟估算，再因
第一个 oversize 样本失败。

当前已删除：

- `src/shaft/data/cost_plan.py` 及 mmap/bin/manifest/reference/shared-cache probe。
- fixed `cost_aware`、`ShaftFixedBatchPlanner`、fixed guard。
- `dynamic_cost_aware` optimizer-step planner、完整 preflight、summary、DFS。
- `train.optimizer_batch.target_samples/target_supervised_tokens`。
- 全训练期 planned sample-count tuple 与基于它修指标的逻辑。

配置只保留 `fixed | bounded_cost_aware`。

## 3. 真源与数据结构

### 3.1 SampleSchedule

`ShaftSampleSchedule.ref_at(draw_id)` 是 horizon-independent 的 draw 映射：

- fingerprint 绑定 source names/sizes/weights、mix strategy、shuffle、seed；
- 不绑定 max steps 或有限 plan 长度；
- `draw_id` 同时驱动 prompt rotation 和 deterministic online transform；
- `concat` 与 shuffled weighted sampling 支持无限 draw；weighted + unshuffled 在 bounded 模式明确拒绝。

`ShaftSamplePlan` 仍提供有限 `len()`，只用于 fixed/GRPO 等有限 map-style 路径，并把可 schedule 的
`ref_at()` 委托给同一 draw mapper。bounded SFT 的 DataCenter 直接产出 schedule，Dataset 通过显式
`SampleRef` 取数，不构造 duration-sized finite plan。

### 3.2 SampleCost

每个 buffer entry 只保存：

```text
SampleRef + {
  llm_tokens,
  supervised_tokens,
  vision_patches,
  loss_weight_sum,
  exact
}
```

成本语义：

- 图像 resize、patch 与 processed token layout：模型 `ProcessorPolicy`。
- prompt、target、EOS、截断、causal shift、loss scale：Template。
- provider 只读取图像 header，不 decode；sample cost 与 header 使用 bounded LRU。
- provider fingerprint 绑定 source record、planning-safe transforms、tokenizer artifact、processor/template
  semantics 与显式 `media_snapshot_id`；schedule 只在 batching spec 中绑定，避免双真源。
- sample-cost LRU 按完整 logical draw/context 缓存，不在通用层猜测 template 可能读取哪些 item 字段；图片
  header 仍可按 canonical path 跨 draw 复用。

### 3.3 BoundedBatchingSpec

不可变 spec 包含：

- planner algorithm version；
- schedule fingerprint；
- cost fingerprint；
- DP world size；
- buffer size；
- max samples / padded tokens / vision patches；
- seed。

spec 不包含 max steps、optimizer step count、GA、full sample count 或 optimizer sample target，所以训练时长
不改变既有 draw prefix。exact Trainer resume 另有 training contract 绑定 duration、GA、optimizer 与 scheduler；
若要改变训练 schedule，应使用 `init_from_checkpoint` 而不是恢复 optimizer state。

### 3.4 BoundedBatchingState

checkpoint state 包含：

- contract fingerprint；
- completed global microstep；
- next draw id；
- FIFO buffer 的 ref + cost；
- emitted sample/LLM/supervised/vision 累计值；
- state integrity fingerprint。

buffer 中 draw id 必须严格递增、无重复、低于 next draw；每条 ref 必须能由 schedule 同 draw id 重建，并且
必须满足 `next_draw_id == emitted_samples + len(buffer)` 的 prefix 守恒。

## 4. Planner 算法

每个 global microstep：

1. 按 draw id 惰性补满 buffer。
2. 成本不精确或单样本超过 text/vision 上限时立即失败，停止继续 lookahead。
3. 取最老的 W 个 entry 作为 W 个 non-empty rank anchors。
4. 将其余 candidates 按成本和稳定 tie-break 排序。
5. 对每个 candidate，在满足以下条件的 bins 中做确定性放置：
   - `count <= max_samples_per_microbatch`
   - `count * max(llm_tokens) <= max_padded_tokens`
   - `sum(vision_patches) <= max_vision_patches`（若设置）
6. 首先最小化 feasible bin 的 projected text+vision load，再以 padding waste 和稳定索引打破平局。
7. 删除 selected entries；未选 entries 按原 FIFO 顺序进入 state_after。

oldest anchors 给出有限等待保证。算法不需要 exact number of samples，也不需要 DFS；一个 microstep 至少消费 W
个 draw，最多消费 `W * max_samples_per_microbatch`。

## 5. HF / Accelerate 接口

全局 BatchSampler 顺序：

```text
[optimizer step][GA microstep][rank]
```

`ShaftBoundedBatchSampler` 在 yield 前先原子规划一个 generic `planning_frame_size * W` frame，并在 frame
内把每个 microstep 的最大 batch 分给当前累计 load 最小的 rank。data 层不解释 optimizer step；training
adapter 把 `planning_frame_size` 设为 GA，并由 callback 做 `global_step * GA` commit 映射。任何 frame 内
后续 microstep 成本错误都会在该 frame 的第一个 forward 前暴露。

Accelerate 条件：

- `batch_size=None`
- `drop_last=True`
- `split_batches=False`
- `even_batches=False`
- global sampler length 是 `remaining_steps * GA * W`

每个 rank 恰好取得 `remaining_steps * GA` 个 non-empty batches。SFT loss 继续使用 collate 后真实
`labels/loss_scale` 计算跨 GA/DP 的 global denominator；planner cost 不能作为 loss denominator。

## 6. Resume 与预取

DataLoader 会预取未来 batch，因此 sampler live state 往往领先模型实际训练位置。保存 live cursor 会跳过尚未
forward 的样本。

正确流程：

1. sampler 只保留尚未 commit 的 planning-frame boundary snapshots；producer/consumer 距离受 DataLoader
   prefetch 有界，不删除未来仍需提交的 boundary。
2. training callback 在 `on_step_end` 用 `global_step * GA` 提交对应 snapshot。
3. callback 实现 HF `ExportableState`，committed spec/state 在 HF 写 `trainer_state.json` 时同步序列化；因此
   它发生在 checkpoint rotation 之前，没有 post-save sidecar 窗口。
4. resume 读取 trainer global step，验证 state microstep、spec、source/media/cost/topology 和 training
   schedule contract；从 run root 自动恢复时跳过缺失/损坏 bounded callback state 的较新目录。
5. sampler 从 committed state 构造，TrainingArguments 设置 `ignore_data_skip=true`，避免 HF 再 skip 一次。
6. checkpoint buffer 的 cost 在恢复时按需复核；source record/media snapshot fingerprint 变化会更早使
   contract 失败。

多 rank provider/spec/resume 构造使用统一 startup envelope：每个 rank 先捕获本地结果，再共同 gather 并
一致退出。world size > 1 时还会重放第一个 buffer 并比较 plan digest，捕获 rank-local mount/成本漂移；
sampler 运行期不插入 distributed collective。

persistent worker 重建会生成 worker base seeds。bounded DataLoader 使用独立 `torch.Generator`，不消耗恢复后的
模型/dropout RNG；planning-safe transforms 本身必须由 draw context 决定。

## 7. 图像预算

`max_pixels` 是 processor 的单图 resize budget；`max_vision_patches` 是一个 local batch 内所有图片的
aggregate encoder-work budget，两者不是同一字段。

aggregate budget 必须至少容纳 processor 允许的最大单样本。Qwen patch-size 16、`max_pixels=4,000,000`
可能产生约 15,625 pre-merge patches，因此 re/re2 使用 16,384。该预算防止多个大图合批，不用于把一个
合法的 4M-pixel 样本错误拒绝在 8,192。

## 8. Testing contract

必须覆盖：

- 百万级虚拟 schedule，单 microstep cost calls 为一个 buffer；首个 frame yield 的调用数不超过
  `buffer + (frame_size - 1) * W * max_samples`。
- adversarial `8,1,1,1,1,2` 与多维预算，不锁死多个合法解的唯一字面顺序。
- refill 前后 emitted + buffer = `[0, next_draw_id)`，无丢失、重复、饥饿。
- W/GA 多组合经真实 `BatchSamplerShard` 后每 rank batch 数一致。
- inexact、text oversize、vision oversize 在首次观察停止。
- state JSON 完整性、contract drift、optimizer boundary alignment。
- rehashed state 的 draw-prefix 守恒；duration/scheduler exact-resume drift。
- worker prefetch 已领先时，committed resume 后 stream 与 uninterrupted 完全一致。
- 2-rank + `num_workers=2 + persistent_workers=true` 下 model/optimizer/scheduler/RNG/bounded state bitwise 一致。
- startup 不创建 CostPlan sidecar/reference，不调用全量 summarize/materialize。
- rank-local provider failure、首 buffer cost drift、trainer-state 写失败均保证 all-rank 退出；保存失败且
  `save_total_limit=1` 时上一完整 checkpoint 仍可被 resolver 选择。

## 9. 后续阶段

- 用 profiler 调整 buffer/cache/caps，而不是恢复 exact sample target。
- 若需要降低每 rank 重复 cost estimation，可新增经过专项设计的 streaming/Energon backend；不能在当前
  BatchSampler 内做 distributed collective。
- sequence packing 需要独立的 segment attention/position/label/media-grid 隔离设计。
- context parallel 是 job-level topology，只服务单个超长样本，不是动态 batching 的替代品。
