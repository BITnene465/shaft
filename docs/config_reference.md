# Shaft 配置参考

本文档描述 `RuntimeConfig` 的主要配置块和推荐使用方式。配置以 YAML 为主，CLI 只允许无歧义 override。

当前 `config` 已按职责拆分为多文件实现：

- `src/shaft/config/base.py`
- `src/shaft/config/model.py`
- `src/shaft/config/data.py`
- `src/shaft/config/training.py`
- `src/shaft/config/algorithm.py`
- `src/shaft/config/runtime.py`

`src/shaft/config/schema.py` 只作为配置类型的聚合出口，不再承载全部 dataclass 实现。

## 1. 顶层结构

训练主配置由 `RuntimeConfig` 组成：

- `experiment`
- `model`
- `data`
- `algorithm`
- `train`
- `eval`
- `rlhf`
- `plugins`
- `logging`
- `progress`

## 2. `experiment`

用途：实验元信息和输出目录。

关键字段：

- `name`
- `seed`
- `output_dir`
- `run_id`

约束：

- `run_id` 用于区分同一实验模板下的不同运行实例。
- `output_dir` 应视为当前运行的唯一产物目录。

## 3. `model`

用途：模型族、模型路径和微调方式。

关键字段：

- `model_type`
- `model_name_or_path`
- `template`
- `trust_remote_code`
- `attn_implementation`
- `torch_dtype`
- `finetune`

### `model.finetune`

关键字段：

- `mode`: `full | lora | dora | qlora`
- `freeze`
- `target_modules`
- `lora_r`
- `lora_alpha`
- `lora_dropout`
- `lora_bias`
- `use_rslora`
- `qlora_load_in_4bit`
- `qlora_use_double_quant`
- `qlora_quant_type`
- `qlora_compute_dtype`

约束：

- `model_type=qwen3vl` 适用于当前 Qwen3-VL 系列，例如 `Qwen3-VL-4B-Instruct`、
  `Qwen3-VL-32B-Instruct` 和 `Qwen3-VL-30B-A3B-Instruct`。
- `model_type=qwen35vl` / `qwen36vl` 适用于 Qwen3.5 / Qwen3.6 新一代 VLM。两者共享
  同一套 loader、processor policy 和模板默认值；`qwen36vl` 是为了让训练配置保留 3.6 口径。
- Qwen3.5 / Qwen3.6 需要安装支持 `qwen3_5` / `qwen3_5_moe` 架构的 Transformers。当前
  `qwen35vl` meta 会在运行前检查 `transformers>=5.10.1` 以及
  `transformers.models.qwen3_5` 模块是否存在；MoE 模型还会检查
  `transformers.models.qwen3_5_moe`。如果当前 PyPI release 尚未包含该模型，应安装
  Transformers 主分支或项目确认过的内部 wheel。
- 仓库基础依赖允许 Transformers 4.x/5.x；当前验证过的 lock 口径固定为
  `transformers==5.10.1`，可直接支持 Qwen3.5 / Qwen3.6 的 HF 本地训练与推理。
  `qwen-next` extra 用于显式固定新一代 Qwen 口径；业务 vLLM 推理镜像使用同一份
  `uv.lock`，当前标准为 `vllm==0.19.1` + `transformers==5.10.1`。对本地 HF 训练环境，
  推荐执行：

  ```bash
  uv sync --extra dev --extra train --extra distributed --extra qwen-next --extra gpu
  ```

  业务推理环境不要自行拼装依赖版本，应使用 `docker/inference/` 中的推理镜像或用同一份
  `uv.lock` 构建。推理效果对 prompt、pixel budget、generation 参数和 JSON 解析都敏感，
  不能只对齐模型权重；镜像构建和 `shaft-contract-smoke` 验收见
  `docker/inference/README.md`。
- `qwen35vl` / `qwen36vl` 默认使用 `template=qwen35vl`，该模板会在 generation prompt 中关闭
  thinking，避免结构化 JSON 任务无意训练或生成 `<think>` 内容。确实需要 CoT 数据时，显式设置
  `model.template: qwen35vl_thinking`。
- `data.min_pixels/max_pixels` 是否以及如何传给 processor 由模型的 `ProcessorPolicy` 唯一决定；
  `qwen_vl` 使用 `images_kwargs`，通用/identity policy 默认不假设 processor 支持 pixel budget。
  新模型不得在 collator、template 或 pipeline 中再维护一份转发开关。
- 从 Qwen3-VL 切换到 Qwen3.6-VL 训练时，核心差异应只落在模型字段，例如：
  `model_type: qwen36vl`、`model_name_or_path: models/Qwen3.6-27B`、必要时把
  `train.distributed.strategy` 切到 `fsdp` 或 `deepspeed`。`data`、`algorithm`、SFT target
  格式和 Qwen3-VL 主链保持一致。
- Shaft 会对本地 `config.json` 的 HF `model_type` 做早期校验：`qwen3vl` 期望
  `qwen3_vl`，`qwen35vl` / `qwen36vl` 期望 `qwen3_5` 或 `qwen3_5_moe`。这能在模型加载前
  发现 `model.model_type` 与权重目录不匹配的问题。
- `configs/train/qwen36_sft_27b_fsdp_example.yaml` 是最小 SFT/FSDP+LoRA 训练示例；其中
  `transformer_layer_cls_to_wrap: ["auto"]` 会按 `qwen36vl` 模型族解析为 Qwen3.5/3.6 的 dense
  decoder 与 vision block 类名。当前 Qwen3.6 / Transformers 5.10 / PyTorch 2.10 组合下，
  `distributed.fsdp.activation_checkpointing` 默认关闭，保留 `train.gradient_checkpointing` 走模型侧
  checkpointing；FSDP activation wrapper 在 Qwen3.6 linear-attention 层上会触发 recompute tensor
  数量不一致。
- 8x80GB 上 Qwen3.6-27B full-parameter FSDP + AdamW 会在 optimizer step 触达显存上限；默认示例
  使用 LoRA。full fine-tune 应使用 DeepSpeed ZeRO-3、CPU offload、低精度/8-bit optimizer 或更多显存资源。
- 对本地 HF sharded checkpoint，Shaft 会在模型装配前读取 `model.safetensors.index.json` 或
  `pytorch_model.bin.index.json`，确认索引引用的 shard 文件都已存在。半下载目录会在进入
  `from_pretrained` 前直接报出缺失 shard，避免把下载不完整误判为模型架构或训练配置问题。
- `target_modules=["auto"]` 表示交给模型族 `peft policy` 自动解析。
- `freeze.groups` 当前只允许：
  - `language_model`
  - `vision_tower`
  - `aligner`
  - `generator`
- `freeze.regex` 与 `freeze.trainable_regex` 必须是合法正则。
- `init_from_checkpoint` 与 `resume_from_checkpoint` 的兼容矩阵由 `training/checkpointing.py` 统一校验。

### `model.finetune.freeze`

关键字段：

- `groups`
- `prefixes`
- `regex`
- `trainable_prefixes`
- `trainable_regex`

说明：

- `groups` 使用模型族声明的结构分组：
  - `language_model`
  - `vision_tower`
  - `aligner`
  - `generator`
- `groups` 的匹配采用“最具体前缀优先”。
  - 例如 `language_model=("model",)` 且 `vision_tower=("model.visual",)` 时，
    `model.visual.*` 会归到 `vision_tower`，不会被 `language_model` 误伤。
- `prefixes` / `regex` 用于冻结。
- `trainable_prefixes` / `trainable_regex` 用于显式解冻，优先级高于冻结规则。

执行语义：

- 训练时会先把上述配置解析为一份 `resolved finetune plan`，后续训练执行与 adapter 导入校验都消费这份计划。
- 训练启动后，CLI 会打印一份运行时 `resolved freeze summary`，并在输出目录写入：
  - `shaft_finetune_summary.json`
  - `shaft_optimizer_summary.json`
- `full`
  - 先默认全部可训练
  - 再应用冻结规则
  - 最后应用 `trainable override`
- `lora / dora / qlora`
  - 冻结规则主要作用于 `target_modules=["auto"] / ["all-linear"]` 的自动展开结果
  - 如果显式指定 `target_modules`，则保持显式配置权威
  - `trainable override` 会额外导出为 `modules_to_save`
    - 这里的前缀匹配以模块名为准，例如 `lm_head`、`model.visual.merger`
  - 这类 adapter checkpoint 仍然是 PEFT 目录；如果后续部署后端只接受 full HF model，需要先 merge

## 4. `data`

用途：数据 catalog、多数据源、mixing 与批处理行为。

关键字段：

- `catalog_path`
- `catalog_names`
- `datasets`
- `prompt_sampling`
- `batching`
- `mix_strategy`
- `num_workers`
- `prefetch_factor`
- `pin_memory`
- `persistent_workers`
- `record_cache_dir`
- `image_cache_size`
- `min_pixels`
- `max_pixels`
- `max_length`
- `add_eos_token`
- `shuffle`

### `data.batching`

所有训练 YAML 必须显式写 strategy。保留原 HF batch 顺序时：

```yaml
data:
  batching:
    strategy: fixed
```

启用有界成本感知批次时：

```yaml
data:
  media_snapshot_id: banana-v5.0-re2-media-v1
  batching:
    strategy: bounded_cost_aware
    buffer_size: 64
    cost_cache_size: 65536
    max_samples_per_microbatch: 2
    max_padded_tokens: 10000
    max_vision_patches: 16384
```

- 可选策略只有 `fixed | bounded_cost_aware`。旧 `cost_aware`、`dynamic_cost_aware`、
  `fixed_guard`、`planning_window`、`cost_plan_cache_dir`、`rank_balance` 和
  `train.optimizer_batch` 已删除；出现时按未知配置字段拒绝，不提供隐式迁移。
- `bounded_cost_aware` 当前只支持 SFT、`train.duration.unit=steps` 和 DDP（单进程也使用 DDP
  contract）。FSDP/DeepSpeed 的 variable BatchSampler 尚未专项验收，因此 normalize 阶段拒绝。
- `buffer_size` 是 planner 中最多常驻的轻量 `SampleRef + SampleCost` 数量，必须至少等于 DP world
  size。它不是全训练 horizon，也不会导致启动时扫描 `steps * samples`。
- `media_snapshot_id` 是 bounded 路径必填的不可变媒体快照 id。JSONL/Arrow record fingerprint 不会为了
  startup 安全而扫描全部外部图片；若图片集合、内容或尺寸可能改变，必须先生成新快照并更换该 id。
- `cost_cache_size` 同时限制 prompt-variant sample-cost LRU 与 canonical image-header LRU；`0` 表示
  禁用缓存。缓存不包含解码图像、processor tensor 或完整文本 token tensor。
- `max_samples_per_microbatch` 是每卡每个 local batch 的样本数上限。未显式设置时使用
  `train.per_device_train_batch_size`；该 HF 字段在 bounded 模式只提供兼容默认值和日志名义值，不代表
  每个实际 batch 的固定 cardinality。
- `max_padded_tokens` 是硬上限，计算方式为
  `local sample count * processor 后最长 LLM sequence`，不是原始字符数。
- `max_vision_patches` 是 local batch 内所有图片 pre-merge vision patch 的可选总上限，用于避免多个大图
  被合到同一 batch。它必须能容纳 processor pixel budget 允许的最大单样本；例如 Qwen patch-size 16、
  `max_pixels=4,000,000` 的配置应至少使用 16,384。单样本超限会在该 draw 首次进入 buffer 时明确失败。
- 每个 global microstep 固定输出 W 个 non-empty local batch，HF 继续按固定 GA 聚合；optimizer step 的
  实际 sample/token 数自然可变，不再维护 exact target samples/tokens。
- planner 取最老的 W 个 buffer entry 作为 anchors，再以 projected rank load 为第一目标、padding waste 为
  tie-break 合并 entry；完整 planning frame 还会按 rank 累计成本重新分配 batches。因此不会饿死长样本，
  也不会改变 mixing draw multiset。weighted mixing 在 bounded 模式要求 `data.shuffle=true`。
- 多 rank startup 对第一个 buffer 的 cost/plan digest 做一致性校验。首个 forward 前会原子规划完整 GA frame，
  cost call 上界是 `buffer + (GA - 1) * W * max_samples`，而不是完整 duration；运行中最坏 host 预取内存还
  需计入 `num_workers * prefetch_factor`。
- 所有 rank 在 immutable snapshot contract 下独立重放同一个轻量全局 BatchSampler，Accelerate 以
  `split_batches=false/even_batches=false` 选择各 rank 的 batch。sampler 内禁止 collective，避免 worker
  预取速度不同造成死锁；首 buffer 漂移和 startup 单 rank 错误会先做 all-rank 聚合再退出。
- duration-independent spec、已提交 global microstep、FIFO buffer 及实际累计成本，作为
  `ShaftBoundedBatchingCallback` stateful payload 写入 checkpoint 的 HF `trainer_state.json`。只保存已完成
  optimizer step 对应 snapshot，不保存预取推进后的 live cursor，也不维护第二个 sidecar 状态源。
- resume 会验证 source/media snapshot/mixing/prompt/tokenizer/processor/template、world size、buffer、budgets，
  以及 duration/GA/optimizer/scheduler exact-resume contract；随后从 committed state 继续并禁用 HF 二次
  data skip。persistent workers 使用 DataLoader 专用 generator，不改变模型 RNG。
- run root 的 `shaft_batching_run_metadata.json` 保存 resolved 策略、DP/GA、pixel budget、source weights、
  media snapshot id、buffer/cache/caps 和 contract fingerprint；启用 W&B 时同一 payload 写入
  `shaft_batching` run config。

### `data.datasets`

每个条目是一个 `DatasetSourceConfig`：

- `dataset_name`
- `source_type`
- `train_path`
- `val_path`
- `train_paths`
- `val_paths`
- `weight`
- `enabled`
- `use_for_eval`
- `offline_transforms`
- `online_transforms`
- `help`
- `tags`

约束：

- `catalog_path/catalog_names` 用于复用“命名数据集”。
- `catalog_path` 只表示“去哪个 catalog 文件里找”，**不会自动启用里面全部数据集**。
- 只有写进 `catalog_names` 的数据集才会被展开到最终的 `data.datasets`。
- `datasets` 用于当前 YAML 内联声明数据源。
- 实际进入 `ShaftDataCenter` 前，catalog 会先展开成标准 `datasets` 列表。
- `DatasetSourceConfig` 只描述配置输入；进入数据主链前，会先被解析成 `ShaftDatasetMeta`。
- `use_for_eval=false` 表示该数据集只参与训练 mixing，不参与验证集构建，也不要求提供 `val_path/val_paths`。
- 当 `eval.enabled=true` 时，至少要有一个 `enabled=true` 且 `use_for_eval=true` 的数据集。
- `mix_strategy` 当前支持：
  - `concat`：覆盖全部有效行；开启 `shuffle` 时每轮使用无额外索引内存的可复现置换。
  - `weighted`：以 `DatasetSourceConfig.weight` 归一化为数据源概率，并按 logical sample draw
    有放回抽样。fixed step 模式使用有限 plan；bounded step 模式直接消费 horizon-independent schedule。
- `weight=0` 会禁用该 source 的 train split 加载与抽样；若 `use_for_eval=true`，val split 仍可参与评估。
- `num_workers` 是每个 rank 的 worker 数。例如 8 rank × 4 worker 会产生 32 个读取进程。
- `prefetch_factor` 是每个 worker 预取 batch 数，仅在 `num_workers>0` 时传给 HF DataLoader。
- JSONL 首次加载时会规范化到 source snapshot 指纹化的 Arrow cache；`record_cache_dir` 可覆盖默认的
  `~/.cache/shaft/records`。后续 rank/worker 使用只读 mmap，不再各自保留完整 Python record list。
- `image_cache_size` 是每个 worker 的解码后 PIL 图像 LRU 容量，默认 `0`（关闭）。多 rank/worker
  环境应按总内存预算谨慎开启。
- `max_length` 是训练 batch 组装阶段的 token 长度上限，语义接近 Swift / LLaMA-Factory 的
  `max_length` / `cutoff_len`。当 `prefix_tokens + target_tokens + eos > max_length` 时，SFT 会按剩余
  token budget 截断 assistant target；被截断的 target 不会补 EOS，避免把半截 JSON 教成合法结束。
  训练前仍应按真实 processor 做长度审计，超限样本优先过滤、记录或拆 crop。

### `data.prompt_sampling`

用途：训练运行时对同一数据样本随机轮换等价 prompt，避免把固定 prompt 学成任务 one-hot 编码。

关键字段：

- `enabled`: 是否启用，默认 `false`。
- `train_only`: 是否只对 train dataset 生效，默认 `true`。推荐保持 `true`，让 val/eval 使用固定 canonical
  prompt。
- `seed`: prompt 采样种子；未设置时使用 `experiment.seed`。
- `pools`: 按 `dataset_name` 配置单个版本化 prompt pool YAML 文件。

示例：

```yaml
data:
  prompt_sampling:
    enabled: true
    train_only: true
    seed: 42
    pools:
      grounding_arrow: ../prompts/pools/grounding_arrow.v2.4.yaml
      point_arrow: ../prompts/pools/point_arrow.v2.4.yaml
```

Prompt pool 示例：

```yaml
metadata:
  id: shaft.example.prompt_pool.v1
  version: v1
prompts:
  - id: canonical
    sampling_weight: 3.0
    user_prompt: Return the requested JSON.
  - id: concise
    sampling_weight: 1.0
    user_prompt: JSON only.
```

约束：

- prompt pool 路径相对训练 YAML 所在目录解析；一个数据集只能指向一个 pool 文件。
- pool 只按 `dataset_name` 匹配，不能跨任务复用不同 label scope 的 prompt。
- 每个 pool YAML 必须包含 `metadata.id`、版本信息和非空 `prompts` 列表；每个 prompt variant 必须包含
  `id` 和 `user_prompt`。
- variant 可配置非负 `sampling_weight`；运行时会归一化为概率。省略时默认为 `1.0`，因此整个 pool
  省略该字段就是等概率。至少一个 variant 的 weight 必须大于 0。
- 启用后，所有正权重 train source 都必须有对应 pool；当 `train_only=false` 时，启用的 eval source
  也必须有 pool。`weight=0` 且不需要 eval prompt 的 source 不要求配置。SFT 行里的 `user_prompt` 不再
  作为 prompt 真源。
- 采样单位是 logical sample draw，采样键包含 prompt seed、sample ref 的 transform seed、
  `dataset_name + sample_id + draw_id`。同一个
  GRPO grouped-generation 位置保持同一 prompt；同一 source row 被再次抽到时可随 draw_id 轮换，且
  在 resume、多 worker 和分布式场景下仍可复现。
- 当前实现只替换 `system_prompt/user_prompt` 形式的样本；如果样本已经提供多轮 `messages`，会跳过采样并在
  `extra.prompt_sampling_skipped` 里记录原因。

补充说明：

- 仓库内置的 `configs/data/example.yaml` 当前只作为示例 catalog，不应默认视为可直接训练的数据清单。
- 如果你的实验数据较少或不需要复用 catalog，直接使用 `data.datasets` 往往更直观。

## 5. `algorithm`

用途：选择训练算法与算法级参数。

关键字段：

- `name`: `sft | dpo | ppo | grpo`
- `params`

说明：

- `params` 只保留算法的轻量补充参数。
- DPO/PPO/GRPO 的结构化核心参数在 `rlhf` 块中。

## 6. `train`

用途：训练行为、保存策略和 resume/init 规则。

关键字段：

- `duration`
- `per_device_train_batch_size`
- `gradient_accumulation_steps`
- `gradient_checkpointing`
- `learning_rate`
- `param_group_lrs`
- `no_decay_name_patterns`
- `optimizer_name`
- `scheduler_name`
- `loss_name`
- `loss_scale`
- `weight_decay`
- `warmup_ratio`
- `max_grad_norm`
- `bf16`
- `use_cpu`
- `full_determinism`
- `logging_steps`
- `save_strategy`
- `save_steps`
- `save_total_limit`
- `ddp_find_unused_parameters`
- `report_to`
- `load_best_model_at_end`
- `save_final_model`
- `save_final_state`
- `init_from_checkpoint`
- `resume_from_checkpoint`
- `distributed`

保存与恢复边界：

- `save_final_model=true` 把部署用 HF/PEFT 导出写入 `<output_dir>/best`。
- `save_final_state=true` 把最终 `trainer_state.json` 保留在 run 根目录；finetune/optimizer summary 也属于
  run metadata，root layout 清理不得删除这些文件。
- root `trainer_state.json` 只用于最终指标和 Web UI 状态展示，不等于包含 optimizer/RNG 的可恢复
  checkpoint。`resume_from_checkpoint` 指向 run 根目录时，如果存在 `checkpoint-*`，始终优先最新
  checkpoint；`best` 仍是部署导出，不作为精确训练恢复点。
- SFT/RLHF pipeline 总是在 dataset、base model 与 PEFT adapter 装配前使用 `experiment.seed` 初始化 Python/
  NumPy/PyTorch 随机状态，保证 full/PEFT fresh 初始化不依赖 Trainer 创建时机。
- `full_determinism=true` 还会在上述早期阶段调用 HF `enable_full_determinism`，并透传
  `TrainingArguments.full_determinism`，启用 PyTorch deterministic algorithms、确定性 cuBLAS/CuDNN，
  以及支持该能力的 FlashAttention deterministic backward。它用于 bitwise CUDA resume/fresh
  reproduction 验收，通常会降低吞吐；默认关闭。若默认关闭，planning/data/optimizer 状态仍可精确恢复，
  但非确定性 CUDA kernel 可能让两次运行产生正常的微小数值差异。

### `train.duration`

训练时长只有一个真源：

```yaml
train:
  duration:
    unit: steps
    value: 10000
```

- `unit=steps` 是主路径，`value` 必须为正整数。Shaft 会把它映射到 HF `max_steps`。fixed batch 的有限
  SamplePlan 使用标准 global batch 公式；bounded 模式只计算 map-style Dataset 所需的最大 draw 上界，
  runtime 从 horizon-independent schedule 惰性取数，不逐 draw 物化。
- `unit=epochs` 用于有限数据兼容，`value` 可为正浮点数。Shaft 会映射到 HF
  `num_train_epochs`；一个 epoch 的 plan 长度默认为所有有效 source 行数之和。
- YAML 不再同时维护 `epochs` 与 `max_steps`。CLI 仍提供互斥的 `--epochs` / `--max-steps` 便捷覆写。

说明：

- `train` 是 SFT 与 RLHF 共用的基础训练块。
- `optimizer_name/scheduler_name/loss_name` 走注册表。
- `distributed.strategy` 描述训练拓扑入口，当前支持：
  - `ddp`
  - `fsdp`
  - `deepspeed`
  默认是 `ddp`，表示继续使用 Hugging Face / torchrun 的常规 DDP 语义。
- `distributed.fsdp` 只维护 FSDP 配置语义，不直接启动进程；训练入口仍由 CLI / torchrun 负责。关键字段：
  - `sharding_strategy`: `full_shard | shard_grad_op | no_shard | hybrid_shard`
  - `auto_wrap_policy`: `none | transformer | size`
  - `transformer_layer_cls_to_wrap`: transformer auto-wrap 的层类名列表，默认 `["auto"]`
  - `min_num_params`: size auto-wrap 下限，必须大于等于 0
  - `activation_checkpointing`
  - `cpu_offload`
  - `use_orig_params`
  - `backward_prefetch`: `backward_pre | backward_post`，也可为空
  - `forward_prefetch`
  - `limit_all_gathers`
  - `state_dict_type`: `full_state_dict | local_state_dict | sharded_state_dict`
  - `sync_module_states`
- `distributed.fsdp.transformer_layer_cls_to_wrap=["auto"]` 会按模型族默认解析。Qwen3VL 当前解析为：
  - `Qwen3VLTextDecoderLayer`
  - `Qwen3VLVisionBlock`
- Qwen3.5 / Qwen3.6 dense 默认解析为：
  - `Qwen3_5DecoderLayer`
  - `Qwen3_5VisionBlock`
- Qwen3.5 / Qwen3.6 MoE 默认解析为：
  - `Qwen3_5MoeDecoderLayer`
  - `Qwen3_5MoeVisionBlock`
- `distributed.deepspeed` 支持 `config_path` 或 inline `config`。当 `strategy=deepspeed` 时，两者至少要提供一个；
  `config_path` 的相对路径按训练 YAML 所在目录解析。Shaft 只负责保存和校验配置真源，不在
  `config` 层展开 DeepSpeed 运行时细节。
- 当前 Shaft 仍由自定义 optimizer/scheduler 持有参数分组学习率语义；DeepSpeed 配置如果包含
  `optimizer`/`scheduler` 块会在加载阶段报错。应交给 HF Trainer 将 Shaft optimizer 接入 DeepSpeed。
- `strategy=deepspeed` 时，pipeline 会先构建 `TrainingArguments`，再执行模型 `from_pretrained`。
  这是 ZeRO-3 大模型训练的必要顺序：HF 会在 `TrainingArguments` 初始化阶段建立 DeepSpeed
  runtime config，让模型加载阶段能感知 ZeRO-3 分片语义。
- `strategy` 不是 `deepspeed` 时，Shaft 会清理 HF/Accelerate 进程级 DeepSpeed 状态，避免
  在测试、Web UI 或长驻进程里先运行 DeepSpeed 后污染后续 DDP/FSDP 训练。
- `configs/deepspeed/zero1_bf16.json`、`zero2_bf16.json`、`zero3_bf16.json` 分别是 ZeRO-1/2/3
  bf16 示例配置；ZeRO-3 示例包含保存时 gather 16-bit 权重的设置，用于保持 `trainer.save_model()`
  的 HF export 语义。
- 分片策略属于训练运行时；数据、template、task prompt 和 collator 不应该感知 FSDP/DeepSpeed。
- `gradient_checkpointing`
  - 打开后会把 `TrainingArguments.gradient_checkpointing` 设为 `true`
  - 并在模型装配阶段显式把训练态 `use_cache` 关闭
  - `qlora` 路径会同步传给 `prepare_model_for_kbit_training(..., use_gradient_checkpointing=...)`
- `param_group_lrs` 用于显式配置分组学习率。当前支持的键：
  - `language_model`
  - `vision_tower`
  - `aligner`
  - `generator`
  - `lora_params`
  - `modules_to_save`
- 没有写进 `param_group_lrs` 的组，回退到全局 `train.learning_rate`。
- `no_decay_name_patterns` 用于把额外参数名并入 `no_decay` 规则。
  - 匹配语义是“参数规范名后缀匹配”，例如：
    - `embed_tokens.weight`
    - `lm_head.weight`
  - 这条规则会叠加在默认 `no_decay` 规则之上；默认规则仍然包括：
    - `*.bias`
    - `ndim <= 1` 的参数
- 结构组与训练语义组是两层概念：
  - 结构组：
    - `language_model`
    - `vision_tower`
    - `aligner`
    - `generator`
  - 训练语义组：
    - `lora_params`
    - `modules_to_save`
- `full`
  - 主要按结构组分学习率。
- `lora / dora / qlora`
  - `lora_params` 和 `modules_to_save` 会优先于结构组命中。
  - 其余仍可训练的原始参数，再按结构组回退。
- `loss_scale` 控制哪些粗粒度区段参与 loss 计算，当前内置：
  - `default`: 监督所有 assistant 回答（包括多轮对话中的历史 assistant，以及当前 target/response）
  - `last_round`: 只监督最后一轮 assistant 回答（当前 target/response）
  - `all`: 同时监督 system/user/prefix 与 target/response
- 当前 `loss_scale` 的落点在 `template -> SFTCollator -> ShaftSFTTrainer -> training/loss.py`
  这条链上：
  - `template` 负责把多轮消息规范化为 rendered-token supervision plan，并产出单样本
    `labels / loss_scale / span`
  - `SFTCollator` 只执行一次 batch 级多模态 processor 调用、padding 与张量装配
  - `ShaftModelAdapter -> ProcessorPolicy` 负责将 canonical rendered-token span 精确投影到 processor
    展开后的 token layout；缺少模型专用映射时直接报错，不做近似对齐或 partial-image fallback
  - `training/loss.py` 负责真正的加权 next-token loss

## 7. `eval`

用途：评估开关、频率和 best model 选择。

关键字段：

- `enabled`
- `per_device_eval_batch_size`
- `eval_strategy`
- `eval_steps`
- `loss_metrics_enabled`
- `do_sample`
- `temperature`
- `max_new_tokens`
- `online_metrics_enabled`
- `datasets`
- `metric_for_best_model`
- `greater_is_better`

额外说明：

- `eval.eval_strategy: epoch` 时，可以配合 `eval.epoch_interval` 控制“每隔多少个 epoch 才进行一次 eval”。
- `train.save_strategy: epoch` 时，可以配合 `train.save_epoch_interval` 控制“每隔多少个 epoch 才保存一次 checkpoint”。
- 当 interval 不能整除总 epoch 时，训练最后一个 epoch 仍会强制执行一次对应的 eval / save，避免漏掉最终结果。

说明：

- dataset-policy eval 现在统一产出两类聚合结果：
  - `eval_final_loss`
  - `eval_final_score`
- `loss_metrics_enabled=true` 时，框架会按 dataset policy 分别计算 teacher-forced loss，并按同一套 `weight` 聚合为 `eval_final_loss`。
- `online_metrics_enabled=true` 时，框架会按同一套 dataset policy 计算生成式任务指标，并聚合为 `eval_final_score`。
- 当前在线 eval 支持 SFT 与 GRPO。GRPO 训练侧使用 `GRPODataset` 做 rollout，在线 eval 侧保留原始 SFT 样本结构并复用 `SFTCollator` 生成评估 prompt。

### 7.1 在线 eval 配置

当前版本已支持单阶段在线 eval，目标是：

- 单阶段在线 eval
- 多数据集
- 多任务
- 每个数据集只绑定一个 task
- 通过一套统一 policy 同时支持：
  - `eval_final_score`
  - `eval_final_loss`

当前 dataset 级 eval policy 包含：

- `prediction_codec`
- `target_adapter`
- `metrics`
- `primary_metric`
- `normalizer`
- `weight`

关键约束：

1. 一个 `dataset_name` 只能绑定一个 eval policy
2. 每个 dataset 只能有一个 `primary_metric`
3. 每个 dataset 的 `primary_metric` 必须归一化到 `[0, 1]`
4. `eval_final_score` 由各 dataset 的 normalized primary score 按权重加权求和得到
5. `eval_final_loss` 由各 dataset 的 teacher-forced loss 按同样的权重加权求和得到
6. dataset policy 只要求为 `use_for_eval=true` 的数据集配置；训练专用数据集不会进入这一套聚合

示意配置如下：

```yaml
eval:
  enabled: true
  eval_strategy: epoch
  loss_metrics_enabled: true
  metric_for_best_model: eval_final_score
  greater_is_better: true
  online_metrics_enabled: true
  datasets:
    det_dataset:
      prediction_codec: det_json
      target_adapter: det_annotation
      metrics:
        - name: parse_success
        - name: parse_partial_rate
        - name: det_f1
          params:
            iou_threshold: 0.5
      primary_metric: det_f1
      normalizer:
        type: identity
      weight: 0.6

    keypoint_dataset:
      prediction_codec: keypoint_json
      target_adapter: keypoint_annotation
      metrics:
        - name: parse_success
        - name: parse_partial_rate
        - name: keypoint_pck
          params:
            threshold: 0.1
      primary_metric: keypoint_pck
      normalizer:
        type: identity
      weight: 0.4
```

说明：

- 这部分当前已经可用，但实现边界仍限定在单阶段在线 eval。
- 当前内置 metric 包括 `parse_success`、`parse_partial_rate` 与 `exact_match`，结构化任务指标需要按扩展指南新增。
- 当前内置 target adapter 只有 `target_text` 与 `extra_field`。
- 当前 `normalizer.type` 只支持 `identity` 与 `range`。
- `prediction_codec`、`target_adapter`、`metric` 会在配置加载阶段校验是否已注册，避免第一次 eval 才报错。
- 启用在线 eval 时，框架会强制使用贪心评估。
- 当 `metric_for_best_model=eval_final_score` 时，要求 `online_metrics_enabled=true`，且 `greater_is_better` 会收敛为 `true`。
- 当 `metric_for_best_model=eval_final_loss` 时，要求 `loss_metrics_enabled=true`，且 `greater_is_better` 会收敛为 `false`。
- 若配置了 dataset policy 且仍保留旧式 `metric_for_best_model=eval_loss`，框架会自动收敛为：
  - 有 online eval 时使用 `eval_final_score`
  - 否则使用 `eval_final_loss`
- 启用 dataset-policy eval 时，`report_to` 会同时上报：
  - per-dataset loss
  - per-dataset metrics
  - per-dataset normalized score
  - `eval_final_loss`
  - `eval_final_score`
- 若某个 dataset 在本次 eval 中没有样本，框架会打 warning 并跳过该 dataset，不把它计入 `final_score`。
- 若希望配置语义更直观，仍建议在 YAML 中显式写出：
  - `metric_for_best_model: eval_final_score`
  - 或 `metric_for_best_model: eval_final_loss`
- codec 已经作为共享层供 `infer` 和在线 eval 共用。

详细设计见：

- [docs/online_eval_design.md](online_eval_design.md)

## 8. `rlhf`

用途：DPO/PPO/GRPO 的结构化专属参数。

### `rlhf.dpo`

- `beta`
- `label_smoothing`
- `loss_type`
- `precompute_ref_log_probs`
- `use_weighting`

### `rlhf.ppo`

- `cliprange`
- `cliprange_value`
- `kl_coef`
- `vf_coef`
- `gamma`
- `lam`
- `whiten_rewards`
- `response_length`
- `temperature`
- `num_ppo_epochs`
- `num_mini_batches`
- `local_rollout_forward_batch_size`
- `num_sample_generations`
- `stop_token`
- `value_model_mode`
- `reward_model_mode`
- `train_value_backbone`
- `allow_untrained_reward_model`
- `allow_text_only_multimodal_ppo`

说明：

- PPO 仍是受限能力，文档与实现均不应把它表述为已完成生产方案。
- 当前 PPO rollout 明确是 text-only：`jsonl_ppo` 的 `image_path` 可省略，即使提供也不会在
  `PPODataset` 中打开/解码；messages 中的 image chunk 会在 PPO collator 中移除。

### `rlhf.grpo`

- `beta`
- `rollout`
- `vllm`
- `reward_functions`

### `rlhf.grpo.rollout`

- `num_generations`
- `num_generations_eval`
- `max_completion_length`
- `temperature`
- `top_p`
- `top_k`
- `min_p`
- `repetition_penalty`
- `generation_kwargs`
- `cache_implementation`
- `use_transformers_paged`

### `rlhf.grpo.vllm`

- `enabled`
- `mode`: `server | colocate`
- `model_impl`: `vllm | transformers`
- `enable_sleep_mode`
- `structured_outputs_regex`
- `server_base_url`
- `server_host`
- `server_port`
- `server_timeout`
- `group_port`
- `gpu_memory_utilization`
- `max_model_length`
- `tensor_parallel_size`

说明：

- `rollout` 是 GRPO 采样行为的真源；旧的 flat 字段如 `max_completion_length` 和 `use_vllm` 仍兼容，但新配置应写入 `rollout / vllm`。
- `vllm.mode=colocate` 表示 vLLM 与训练进程共享同一组 GPU，适合 smoke 或单机资源有限场景；长训更推荐 `server` 模式，把 rollout 服务和训练进程拆开。
- 对 VLM GRPO，`data.max_pixels` 会在 `GRPODataset` 层先应用到 PIL 图像；否则 TRL/vLLM 会绕过 SFT collator，按原始大图展开过多 multimodal tokens。
- `vllm.max_model_length` 必须覆盖实际 prompt multimodal tokens 与 `rollout.max_completion_length` 的总长度；`max_completion_length=1024` 只限制生成长度，不限制图像 prompt 长度。
- 当前 GRPO 复用 `jsonl_sft` 数据格式：
  - prompt 来自 `messages` 或 `system_prompt + user_prompt`
  - reward target 来自 `target_text`
- 当前内置 reward 通过 `reward_functions` 配置，支持：
  - `exact_match`
  - `parse_success`
  - `grounding_det_f1`
  - `grounding_iou`
- 每个 reward function 由以下字段描述：
  - `name`
  - `codec`
  - `weight`
  - `params`
- `codec` 复用共享 `codec` 注册表，当前可以直接使用 `json_any / json_object / json_list / text`
- Shaft 会自动解析 `steps_per_generation`，保证 TRL 的 `generation_batch_size` 与 `num_generations` 整除约束成立。
- GRPO 通过 `ShaftGroupedSampleSampler` 保留 TRL grouped-generation 的 mini-repeat/repeat-count
  结构，同时按 HF epoch 设置确定性 plan cycle；因此 grouped prompt 一致且多 epoch resume 可复现。

## 9. `plugins`

- `hooks`
- `interceptors`

用途：

- 为训练主链注入横切增强点。

## 10. `logging`

- `level`
- `fmt`
- `file_path`
- `rank_zero_only`
  - 默认 `true`，表示所有 structured log 严格只由 global rank 0 输出，包括 WARNING/ERROR；分布式主链的
    rank-local 失败依靠同步 failure envelope 或 torchrun traceback，不允许普通 warning 与 rank-0 活动行
    竞争共享终端
  - 调试时若设为 `false`，text/JSON 每行都会包含 rank；多节点/多卡且配置了 `file_path` 时自动写入
    `<stem>.rank<N><suffix>`，避免各 rank 并发覆盖同一个文件。共享终端仍可能交错，因此建议同时把
    progress 设为 `plain` 或 `off`

## 11. `progress`

- `enabled`
- `display`
  - `auto`：TTY 使用单行交互显示，重定向日志/CI/WebUI 子进程使用稀疏文本状态
  - `interactive`：强制单行原地刷新
  - `plain`：强制输出无 ANSI、无 `\r` 的稀疏状态行
  - `off`：不创建终端或文本 sink；若 `persist=true`，仍保存结构化快照
- `width`
  - 单行终端显示的最大物理列宽，默认 `72`，最小 `40`；CJK/组合字符按终端 cell 宽度处理
  - 72 列优先显示高分辨率 bar、current/total、小数百分比、速度、ETA、loss 和 LR；40 列从 ETA/metric
    开始降级，但保留进度和速度。慢 step 显示 `s/step`，极快 step 显示 `step/s`；正数亚秒 ETA 显示
    `<1s`。未完成状态不会四舍五入成 `100%`。非 Unicode stream 对 bar 和整行文本一起安全降级
- `refresh_interval`
  - TTY 最小刷新间隔，默认 `0.5` 秒，必须为有限正数
- `log_interval`
  - 非 TTY 普通 update 的最小输出间隔，默认 `30.0` 秒，必须为有限正数；阶段开始/完成/失败不受节流，
    且不会被 `logging.level` 静默屏蔽
- `leave_completed`
  - 是否保留普通子阶段的完成行，默认 `false`；训练主任务仍输出一条最终摘要
- `persist`
  - 是否原子更新 `<output_dir>/shaft_progress.json`，默认 `true`

训练、loss eval、online eval 和 data/model startup 复用同一个 progress manager。bounded cost 在训练
DataLoader 内按需完成，不再创建独立的全量 startup 进度任务。终端只显示
当前前台阶段；进入 eval 时临时替换 train 行，结束后恢复 train。结构化快照保留完整任务树，供 Web UI
读取，Web UI 不解析日志来推导进度。Shaft 同时关闭 Transformers 与 Hugging Face Hub 的原生进度条；
新增长任务必须向统一 manager 发布，不能平行创建 tqdm。嵌套 eval 的 wall time 不计入恢复后的 train
step rate；失败/取消阶段会强制留下摘要。manager 对并发 advance/close 提供有序状态语义。

10,000-step 任务的 interactive 形态示例：

```text
train   ▏······· 25/10k 0.25% 6.54s/step ETA 18h07m loss 7.9 lr 2.5–5e-7
```

`lr` 为所有 optimizer param groups 的当前 min–max range；组间相同则只显示一个值。`logging_steps` 到达前
没有 loss 属于训练日志策略，但速度、ETA 和精确进度从第一步起可见。

## 12. CLI override 原则

只允许无歧义字段通过 CLI 覆盖，例如：

- `run-id`
- `seed`
- `max-steps`
- `epochs`
- `lr`
- `resume-from`
- `init-from`

禁止：

- 用 CLI 直接拼装复杂 `datasets` 列表
- 用 CLI 覆盖多层嵌套且语义不清的配置对象

`--max-steps` 与 `--epochs` 互斥；任一参数都会完整替换 `train.duration` 的 unit/value。
