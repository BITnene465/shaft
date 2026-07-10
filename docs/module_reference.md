# Shaft 模块参考

本文档是 `src/shaft` 的模块参考手册，聚焦每个模块的职责、关键类、关键函数、核心输入输出与扩展入口。

## 1. `config`

相关文件：

- `src/shaft/config/base.py`
- `src/shaft/config/model.py`
- `src/shaft/config/data.py`
- `src/shaft/config/training.py`
- `src/shaft/config/algorithm.py`
- `src/shaft/config/runtime.py`
- `src/shaft/config/schema.py`
- `src/shaft/config/loader.py`
- `src/shaft/config/normalize.py`
- `src/shaft/config/dataset_catalog.py`

### 职能

- 定义训练与推理主配置结构。
- 负责 YAML 加载、catalog 展开、路径归一化与严格校验。
- 把“外部配置形态”收敛为 `RuntimeConfig`。

### 关键类

- `RuntimeConfig`
- `ExperimentConfig`
- `ModelConfig`
- `FinetuneConfig`
- `DataConfig`
- `DatasetSourceConfig`
- `TrainConfig`
- `TrainDistributedConfig`
- `TrainFSDPConfig`
- `TrainDeepSpeedConfig`
- `EvalConfig`
- `EvalDatasetPolicyConfig`
- `EvalMetricConfig`
- `EvalNormalizerConfig`
- `RLHFConfig`

### 关键函数

- `load_config()`
- `normalize_runtime_config()`
- `resolve_dataset_catalog()`

### 核心接口

- 输入：YAML 文件路径、CLI 无歧义 override
- 输出：已校验的 `RuntimeConfig`

### 开发边界

- 允许：字段校验、默认值、配置升级、catalog 展开
- 禁止：模型构建、数据解析、训练循环

## 2. `data`

相关文件：

- `src/shaft/data/center.py`
- `src/shaft/data/meta.py`
- `src/shaft/data/sources.py`
- `src/shaft/data/dataset.py`
- `src/shaft/data/collator.py`
- `src/shaft/data/mixing.py`
- `src/shaft/data/sampler.py`
- `src/shaft/data/transforms.py`
- `src/shaft/data/registry.py`

### 职能

- 读取多数据源。
- 做离线/在线增强。
- 做样本级 mixing。
- 把 JSONL 规范化为可复用的 Arrow mmap record store。
- 按位置惰性解析无状态 sample plan。
- 产出 `Dataset`、`train sampler` 和 `Collator`。

### 关键类

- `ShaftDataCenter`
- `ShaftDatasetBundle`
- `ShaftPreparedRecords`
- `ShaftDatasetMeta`
- `BaseDataSource`
- `SFTRecord` / `DPORecord` / `PPORecord`
- `ShaftArrowRecordStore`
- `ShaftSamplePlan` / `ShaftSampleRef` / `ShaftSampleContext`
- `ShaftSampleSampler` / `ShaftGroupedSampleSampler` / `ShaftCostAwareSampler`
- `ShaftSampleCost` / `ShaftSFTSampleCostProvider` / `ShaftRowInvariantCostProvider`
- `ShaftCostPlanManifest` / `ShaftMMapCostPlanProvider`
- `ShaftBatchPlan` / `ShaftFixedBatchPlanningSpec` / `ShaftFixedBatchPlanner`
- `ShaftBatchPlanningSignature`
- `SFTDataset` / `DPODataset` / `PPODataset`
- `SFTCollator` / `DPOCollator` / `PPOCollator` / `GRPOCollator`

### 关键函数

- `build_data_source()`
- `build_dataset_metas()`
- `build_dataset_bundle()`
- `load_jsonl_sft_records()`
- `load_jsonl_dpo_records()`
- `load_jsonl_ppo_records()`
- `build_offline_pipeline()`
- `build_online_pipeline()`
- `materialize_cost_plan()`
- `load_cost_plan_reference()` / `write_cost_plan_reference()`

### 核心接口

- 输入：`DataConfig`、JSONL 文件、图像路径
- 输出：
  - `ShaftDatasetMeta`
  - 训练记录池
  - `Dataset`
  - `train sampler`
  - batch tensor 字典

### 开发边界

- 允许：样本格式、增强、mixing、collator
- 禁止：训练状态、优化器、损失函数、任务级语义路由

补充说明：

- `ShaftDatasetMeta.use_for_eval` 用于表达“该数据集是否参与验证集构建与在线 eval”。
- `ShaftDataCenter` 会始终构建训练记录池，但只为 `use_for_eval=true` 的数据集加载 `val` split。
- 训练集 mixing 的真源是 `ShaftSamplePlan`。`concat` 做覆盖式访问，`weighted` 按数据集权重做
  可复现的有放回概率抽样；sampler 只惰性发出不可变 ref，不保存全量 index。
- `ShaftSampleCost` 保存 processor 后 LLM token、causal shift 后的有效监督 token、loss weight 和
  vision patch 成本。`ShaftSFTSampleCostProvider` 使用与 dataset 相同的 sample ref/draw context 解析
  online prompt transform，只读取图片 header，不解码像素；图像 resize/token expansion 由模型
  `ProcessorPolicy` 负责，target 截断、causal shift 与 loss weight 由 `Template` 的
  `estimate_supervision_cost()` 负责。cost provider 不维护平行监督语义。
- `materialize_cost_plan()` 以 logical draw 为索引，把 runtime provider 输出流式写成固定宽度二进制
  sidecar；每条记录携带 sample-ref fingerprint，不能把重复 row 的不同 prompt draw 合并。manifest
  绑定 SamplePlan/cost fingerprint、记录数、字节数和内容 checksum，并通过文件锁与原子 rename 支持
  并发 build-on-miss。`ShaftMMapCostPlanProvider` 是 BatchPlanner 实际消费的只读成本真源。
- run root 的 `shaft_cost_plan_reference.json` 只保存共享 manifest 位置和签名；它与
  `shaft_batch_planning_signature.json` 都属于运行元数据，root export 清理不得删除。cache sidecar 不复制
  进每个 checkpoint，丢失时由 rank 0 按相同 fingerprint 重建。启动同步使用 attempt-scoped broadcast，
  durable reference 只在全量 resume 校验成功后发布，不能把旧 run root 当 rendezvous channel。
- cost-aware planning 会重复重建 online transform；只有经 `planning_safe_online_transform` 声明、可由
  logical sample context 确定性复现的 transform 才能进入该路径；声明必须携带显式版本化 fingerprint。
  未声明或依赖 module/qualname 隐式指纹的 transform 会 fail fast。
- `ShaftRowInvariantCostProvider` 只用于明确证明“同一 source row 在所有 draw 下成本不变”的测试/适配；
  prompt rotation 路径必须使用 draw-indexed provider，不能让泛化类名掩盖 row-key 语义。
- `ShaftFixedBatchPlanningSpec` 是 fixed cardinality geometry 的唯一真源；pipeline 构造一次后原样交给
  resume preflight、planner 和 sampler，signature 也从该 spec 派生。
- `ShaftFixedBatchPlanner` 在有界 window 内按文本/视觉成本分桶，保持 local sample count 不变，并把
  相近成本的 local batches 组成 global microstep。`ShaftCostAwareSampler` 将该计划展平成 HF
  `BatchSampler` 可消费的 ref 流；Accelerate 按连续 local batch 分发给各 data rank。
- `ShaftBatchPlanningSignature` 绑定 SamplePlan/source/prompt、processor/template、planner/window、
  fixed-batch/gradient-accumulation 与 DP topology；training callback 将其写入 run root 和 checkpoint，
  resume 不一致时拒绝继续。
- step duration 的 plan 长度直接等于训练所需全局样本数；epoch duration 的单轮 plan 默认等于有效
  source 行数之和。
- `data.prompt_sampling` 在运行时作为 train online transform 应用，按 `dataset_name` 从等价 prompt pool
  中按 `sampling_weight` 采样并替换 `system_prompt/user_prompt`；采样键使用 sample ref 的 `draw_id`，
  默认不作用于 val/eval。
- GRPO 当前复用 `jsonl_sft` 数据：
  - `SFTDataset` 提供 prompt-target 样本
  - `GRPODataset` 把样本适配为 TRL GRPO 所需的 `prompt / image / target_text` 字段
  - `GRPODataset` 在交给 TRL/vLLM 前按 `data.min_pixels / data.max_pixels` 调整 PIL 图像，避免 GRPO 绕过 SFT collator 后使用原始大图撑爆 multimodal token 数
- 当前 PPO 是受限的 text-only 路径：`PPODataset` 不做图像 I/O，PPO collator 只保留消息文本；
  `jsonl_ppo.image_path` 仅作为可选溯源字段。
- `SFTCollator` 与 `DPOCollator` 对一个 batch 只执行一次多模态 processor。多轮 assistant span 先由
  template 编译，再由模型 `ProcessorPolicy` 生成精确 token layout；监督行构造 API 不接收图片或
  processor，不存在逐样本重跑图片预处理的兼容分支。

## 3. `model`

相关文件：

- `src/shaft/model/types.py`
- `src/shaft/model/builder.py`
- `src/shaft/model/registry.py`
- `src/shaft/model/policies.py`
- `src/shaft/model/sharding.py`
- `src/shaft/model/finetune.py`
- `src/shaft/model/qwen3vl.py`
- `src/shaft/model/qwen35vl.py`

### 职能

- 声明模型族元信息。
- 构建 HF model/tokenizer/processor。
- 处理全量微调与 PEFT。
- 收口 processor / peft / sharding policy。

### 关键类

- `ModelMeta`
  - `hf_model_types` 是本地 HF `config.json:model_type` 兼容性校验的真源，用于在真正加载
    checkpoint 前拦截 `model.model_type` 与权重目录不匹配的问题。
- `ShaftModelAdapter`
- `ProcessorPolicy`
- `ShaftProcessedBatch`
- `ShaftProcessorTokenLayout`
- `ShaftProcessorCostEstimate`

`ShaftProcessedBatch` 保存一次 batch processor 调用的完整输出，不把允许字段限制为 Qwen 当前使用的
键。`ProcessorPolicy` 同时声明 processor 构造参数、pixel-budget forwarding、token-layout 规则和训练
输入的复制/重排，以及是否支持精确 image-cost estimation。内置 `identity` 要求 rendered tokens 与
processed tokens 完全一致且不声明成本估算能力；`qwen_vl`
显式处理 `mm_token_type_ids` 标记的多模态 token run。其他模型族必须通过 registry 注册自己的 policy，
不得让 collator 或通用 template 猜测模型字段、batch axis 或 token expansion。processor 新增
`position_ids/token_type_ids` 等 sequence-aligned 字段时，默认策略会显式拒绝，直到模型 policy 定义
如何随 target 拼接、padding 或 DPO pair 扩展。其它字段也必须进入 policy 的
`sample_aligned_model_input_names / whole_batch_model_input_names / static_model_input_names` 之一；未声明
字段会在训练装配时失败，避免升级 processor 后静默误用第 0 维。
- `ModelGroup`
- `ModelModuleGroups`
- `ShaftModelAdapter`
- `ModelLoader`
- `ModelArtifacts`
- `ModelCapabilities`
- `ProcessorPolicy`
- `PeftPolicy`
- `ModelShardingPolicy`
- `Qwen3VLLoader`
- `Qwen35VLLoader`
- `Qwen36VLLoader`

### 关键函数

- `build_model_meta()`
- `build_model_tokenizer_processor()`
- `apply_finetune_strategy()`
- `build_resolved_finetune_plan()`
- `build_freeze_plan()`
- `apply_full_freeze()`
- `resolve_adapter_target_modules()`
- `summarize_finetune()`
- `build_processor_policy()`
- `build_peft_policy()`

### 核心接口

- 输入：`RuntimeConfig.model`
- 输出：`ModelArtifacts`

### 开发边界

- 允许：模型族差异、PEFT 策略、processor 策略、依赖检查、冻结执行计划
- 禁止：数据源路径、训练调度、推理 pipeline 业务语义

补充说明：

- `ModelModuleGroups` 负责声明模型族的结构分组：
  - `language_model`
  - `vision_tower`
  - `aligner`
  - `generator`
- `src/shaft/model/freeze.py` 统一执行冻结逻辑：
  - 作为低层规则工具，负责 group/prefix/regex 匹配
  - 结构分组匹配采用最具体前缀优先，例如 `model.visual.*` 会优先归到 `vision_tower`，不会被更宽的 `language_model=model` 误伤
- `src/shaft/model/finetune_plan.py` 是冻结与 adapter 语义的单一真源：
  - 解析 full 模式下真实可训练参数集合
  - 解析 adapter 模式下真实 `target_modules / modules_to_save`
  - 产出可用于训练执行与导入校验的 `peft signature`
- `src/shaft/model/builder.py` 在 `init_from_checkpoint` 为 adapter 时，会校验：
  - `target_modules`
  - `modules_to_save`
  - `r / alpha / bias / use_rslora / use_dora`
  与当前训练配置一致，再执行权重导入

## 4. `template`

相关文件：

- `src/shaft/template/types.py`
- `src/shaft/template/base.py`
- `src/shaft/template/rendering.py`
- `src/shaft/template/delimited.py`
- `src/shaft/template/qwen.py`
- `src/shaft/template/registry.py`
- `src/shaft/template/qwen3vl.py`
- `src/shaft/template/qwen35vl.py`

### 职能

- 把消息列表转为模型可消费 prompt。
- 定义 decode 协议。
- 通过模板元信息管理模型族模板实现。
- 在训练路径中直接生成 supervision plan 与单样本 `labels / loss_scale / span`。
- supervision plan 的 span 使用 chat template 渲染后的 canonical token 坐标；模型 processor expansion
  通过 `ShaftProcessorTokenLayout` 投影，不在 template 中重复处理图片。
- `ShaftChatRenderer` 只暴露完整 chat render 和纯 tokenizer 两个操作。supervision plan 不接收
  processor、image 或 model adapter，因此模型切换后也不能在 template 内恢复图片预处理。
- `ShaftDelimitedChatTemplate` 为有稳定消息分隔符的模型提供单次完整渲染 span compiler；非分隔符模板
  必须实现自己的精确 compiler。基类没有 partial-message fallback。

### 关键类

- `TemplateMeta`
- `Template`
- `ShaftChatTemplate`
- `ShaftChatRenderer`
- `ShaftDelimitedChatTemplate`
- `ShaftTemplateSupervisionPlan`
- `ShaftTemplateSupervisedRow`
- `Qwen3VLTemplate`
- `Qwen35VLTemplate`
- `Qwen35VLThinkingTemplate`

### 关键函数

- `resolve_template_meta()`
- `build_template()`
- `build_template_from_meta()`
- `register_template()`
- `build_supervision_plan()`
- `build_supervised_row()`

### Qwen3.5 / Qwen3.6 thinking 策略

- `qwen35vl` 是默认模板，会向上游 chat template 传入 `enable_thinking=False` 和
  `preserve_thinking=False`。结构化标注、JSON grounding、point 等任务应默认使用该模板。
- `qwen35vl_thinking` 是显式 CoT 模板，会传入 `enable_thinking=True` 和
  `preserve_thinking=True`。只有当训练数据 target 本身包含可监督 reasoning 内容时才应启用。
- 模型注册项 `qwen35vl` / `qwen36vl` 默认都解析到 `qwen35vl`，防止新一代 Qwen chat template
  默认打开 `<think>` 后污染结构化输出。

### 开发边界

- 允许：messages 规范化、chat template、decode、训练 supervision span 规划
- 禁止：图像后处理、任务指标计算、训练超参数决策

## 5. `algorithms`

相关文件：

- `src/shaft/algorithms/base.py`
- `src/shaft/algorithms/sft.py`
- `src/shaft/algorithms/dpo.py`
- `src/shaft/algorithms/grpo.py`
- `src/shaft/algorithms/grpo_rewards.py`
- `src/shaft/algorithms/ppo.py`
- `src/shaft/algorithms/rlhf_utils.py`
- `src/shaft/algorithms/registry.py`

### 职能

- 基于上下文构造算法专属 trainer。
- 映射 TRL/HF 所需算法配置。

### 关键类

- `AlgorithmContext`
- `SFTAlgorithm`
- `DPOAlgorithm`
- `GRPOAlgorithm`
- `PPOAlgorithm`

### 关键函数

- `build_reference_model()`
- `build_trl_dpo_config()`
- `build_trl_grpo_config()`
- `build_trl_ppo_config()`
- `build_grpo_reward_functions()`
- `build_ppo_value_and_reward_models()`
- `validate_ppo_runtime_requirements()`

### 开发边界

- 允许：trainer 选择、算法专属辅助对象、算法配置映射
- 禁止：加载数据文件、路径解析、主流程调度

补充说明：

- `GRPOAlgorithm` 当前使用共享 `codec` 注册表与内置 reward registry 组合 reward functions。
- GRPO 配置以 `rlhf.grpo.rollout` 描述采样参数，以 `rlhf.grpo.vllm` 描述 vLLM rollout 后端；旧 flat 字段仅作为兼容入口。
- 当前内置 GRPO reward：
  - `exact_match`
  - `parse_success`
  - `grounding_det_f1`
  - `grounding_iou`
- GRPO 使用 `ShaftGroupedSampleSampler` 保留 TRL 所需的 prompt-repeat/grouped-generation 结构，
  但 sampler 按 HF `set_epoch()` 派生确定性 plan cycle，并直接输出共享 plan 的 sample ref。
  同一 generation group 得到同一 prompt，多 epoch resume 也能复现对应轮次的顺序。

## 6. `pipeline`

相关文件：

- `src/shaft/pipeline/sft.py`
- `src/shaft/pipeline/rlhf.py`
- `src/shaft/pipeline/training_args.py`
- `src/shaft/pipeline/registry.py`

### 职能

- 连接 config、data、model、algorithms、training。
- 编排训练主流程。

### 关键类

- `ShaftSFTPipeline`
- `ShaftRLHFPipeline`

### 关键函数

- `run_sft()`
- `run_rlhf()`
- `build_hf_training_args()`
- `register_pipeline()`

### 开发边界

- 允许：组件装配、resume/save 时序、回调装配
- 禁止：硬编码模型族模板、解析 JSONL、实现 loss 公式

## 7. `loss_scale`

相关文件：

- `src/shaft/loss_scale/base.py`
- `src/shaft/loss_scale/mapping.py`

### 职能

- 定义“哪些区段需要计算 loss”的策略对象。
- 为 `template` 提供多轮消息中各 role span 与 target 的监督开关或权重。
- 让 `loss_scale` 作为独立能力存在，而不是散落在 `template` 与 `loss.py` 中。

### 关键类

- `ShaftLossScale`
- `ShaftLossScaleSpec`

### 关键函数

- `build_loss_scale()`
- `register_loss_scale()`

### 开发边界

- 允许：prefix/target 粗粒度监督策略、自定义权重策略注册
- 禁止：直接计算 loss、直接操作 optimizer/scheduler、在策略中耦合模型族细节

## 8. `training`

相关文件：

- `src/shaft/training/sft_trainer.py`
- `src/shaft/training/online_eval.py`
- `src/shaft/training/optimizer_mixin.py`
- `src/shaft/training/optimizer_plan.py`
- `src/shaft/training/trl_trainers.py`
- `src/shaft/training/loss.py`
- `src/shaft/training/optimizer.py`
- `src/shaft/training/scheduler.py`
- `src/shaft/training/checkpointing.py`
- `src/shaft/training/batch_planning.py`
- `src/shaft/training/progress_callback.py`
- `src/shaft/training/distributed.py`

### 职能

- 包装 HF/TRL trainer。
- 管理 loss/optimizer/scheduler 注册表。
- 解析运行时 optimizer param-group plan。
- 统一 checkpoint 规则和分布式辅助能力。

### 关键类

- `ShaftSFTTrainer`
- `ShaftDPOTrainer`
- `ShaftPPOTrainer`
- `ShaftOptimizerMixin`
- `ShaftResolvedOptimizerPlan`
- `ShaftOnlineEvalRunner`
- `ShaftProgressCallback`
- `CheckpointLayout`
- `ShaftBatchPlanningCallback`
- `Muon`

### 关键函数

- `build_loss()`
- `build_optimizer()`
- `build_resolved_optimizer_plan()`
- `build_scheduler()`
- `register_target_adapter()`
- `inspect_checkpoint_layout()`
- `resolve_resume_checkpoint()`
- `validate_resume_checkpoint()`
- `validate_batch_planning_resume()`
- `validate_training_state_policy()`
- `prune_root_output_layout()`

### 开发边界

- 允许：HF/TRL trainer 扩展、checkpoint 规则、优化器/调度器/loss
- 禁止：数据读取、配置加载、导出发布

补充说明：

- checkpoint 与 run metadata 分层：
  - `checkpoint-*` 是包含训练状态的精确恢复点
  - `best` 是 HF/PEFT 部署导出
  - root `trainer_state.json`、`shaft_finetune_summary.json`、`shaft_optimizer_summary.json` 是持久化
    运行摘要，root export 清理必须保留
  - 从 run 根目录恢复时，最新 `checkpoint-*` 优先于 root final state
  - cost-aware SFT 额外要求 `shaft_batch_planning_signature.json` 完全一致；改变 horizon/topology/data/
    processor 必须新开 run

- `ShaftSFTTrainer` 会对一个 optimizer batch 内实际 causal labels/loss weights 求 denominator，并在 data
  ranks 间汇总；local numerator 采用同一个 global denominator，保证 gradient accumulation 与 DDP
  分割不改变 token 权重。

- `src/shaft/training/optimizer_plan.py` 是分组学习率的运行时真源：
  - 根据 `resolved finetune plan`、`model_adapter.module_groups` 和 `train.param_group_lrs`
    解析真实 optimizer param groups
  - 当前支持两层分组：
    - 结构组：
      - `language_model`
      - `vision_tower`
      - `aligner`
      - `generator`
    - 训练语义组：
      - `lora_params`
      - `modules_to_save`
- `src/shaft/training/optimizer_mixin.py` 把同一套 optimizer/scheduler 构造链复用到：
  - `ShaftSFTTrainer`
  - `ShaftDPOTrainer`
  - `ShaftPPOTrainer`
  - `ShaftGRPOTrainer`
  - 并在 optimizer 创建后写出：
    - `shaft_optimizer_summary.json`
    - 供 CLI 日志和 Web UI 回看 resolved optimizer groups
- `gradient checkpointing` 当前通过两层接通：
  - `src/shaft/config/training.py`
    - `resolve_effective_gradient_checkpointing()` 是实际运行时开关真源
    - FSDP activation checkpointing 启用时会关闭 Trainer/model 侧 gradient checkpointing，避免双重 checkpoint
  - `src/shaft/pipeline/training_args.py`
    - 负责把 effective gradient checkpointing 传给 `TrainingArguments`
  - `src/shaft/model/finetune.py`
    - 负责在训练态关闭 `use_cache`
    - `qlora` 路径会把该开关传给 `prepare_model_for_kbit_training`
- `train.distributed` 当前通过 `src/shaft/pipeline/training_args.py` 接入 HF Trainer：
  - `strategy=ddp` 保持默认 torchrun/DDP
  - `strategy=fsdp` 透传 `TrainingArguments.fsdp/fsdp_config`
  - `strategy=deepspeed` 透传 `TrainingArguments.deepspeed`
  - Qwen3VL 的 FSDP `transformer_layer_cls_to_wrap=["auto"]` 会解析为
    `Qwen3VLTextDecoderLayer` 与 `Qwen3VLVisionBlock`
  - Qwen3.5 / Qwen3.6 dense 会解析为 `Qwen3_5DecoderLayer` 与
    `Qwen3_5VisionBlock`
  - Qwen3.5 / Qwen3.6 MoE 会解析为 `Qwen3_5MoeDecoderLayer` 与
    `Qwen3_5MoeVisionBlock`
  - Qwen3.6 当前推荐关闭 FSDP activation checkpointing，使用 Trainer/model gradient checkpointing；
    full-parameter AdamW 训练还需要 ZeRO-3/offload/低内存 optimizer 或更高显存预算。
- adapter 模式下，`lora_params` 和 `modules_to_save` 会优先命中；剩余 trainable 原始参数再按结构组回退。

## 9. `codec`

相关文件：

- `src/shaft/codec/base.py`
- `src/shaft/codec/registry.py`
- `src/shaft/codec/json.py`

### 职能

- 管理共享 codec 注册表。
- 负责文本到规范结构的容错解析。
- 为 `infer` 与在线 eval 提供统一 decode 能力。

### 关键类

- `ShaftCodecResult`

### 关键函数

- `decode_with_codec()`
- `register_codec()`

### 开发边界

- 允许：JSON 修复、部分解析、共享 decode 扩展
- 禁止：指标计算、业务编排、训练循环

## 10. `metrics`

相关文件：

- `src/shaft/metrics/base.py`
- `src/shaft/metrics/registry.py`
- `src/shaft/metrics/builtin.py`
- `src/shaft/metrics/visualization.py`

### 职能

- 管理在线 eval metric 注册表。
- 提供单阶段在线 eval 的可复用指标实现。
- 提供 eval 结果保存图的共享标注渲染样式，供脚本侧离线 eval 复用。

### 关键类

- `ShaftEvalMetric`
- `ShaftVisualBox`
- `ShaftVisualLineStrip`
- `ShaftVisualPoint`

### 关键函数

- `build_eval_metric()`
- `render_labeled_visualization()`
- `render_prediction_visualization()`
- `register_eval_metric()`
- `save_labeled_visualization()`

### 开发边界

- 允许：轻量在线指标、per-dataset metric 扩展、eval 可视化标注渲染
  （含 dense zoom mosaic、动态线宽/字号、带方向箭头的 line strip）
- 禁止：文本解析、数据路由、多阶段业务编排

## 11. `infer`

相关文件：

- `src/shaft/infer/schema.py`
- `src/shaft/infer/loader.py`
- `src/shaft/infer/engine.py`
- `src/shaft/infer/pipeline.py`

### 职能

- 管理推理配置。
- 封装本地 HF 与 vLLM OpenAI 兼容后端。
- 做多阶段推理上下文传递，并复用共享 codec 解码。

### 关键类

- `InferEngineConfig`
- `InferStageConfig`
- `InferPipelineConfig`
- `ShaftInferRequest`
- `ShaftInferResponse`
- `ShaftInferEngine`
- `ShaftInferPipeline`
- `ShaftInferStageResult`

### 关键函数

- `load_infer_config()`

### 开发边界

- 允许：后端封装、stage 调度、共享 codec 调用
- 禁止：训练逻辑、离线业务脚本逻辑、任务 DSL

## 12. `export`

相关文件：

- `src/shaft/export/hf.py`

### 职能

- 检查 HF full export / PEFT adapter export。
- 合并 adapter 为标准 HF full export。

### 关键类

- `ExportMergeResult`

### 关键函数

- `inspect_hf_artifact()`
- `validate_hf_artifact()`
- `infer_base_model_from_adapter()`
- `merge_peft_adapter()`

### 开发边界

- 允许：HF/PEFT 目录校验、merge
- 禁止：引入自定义格式、发布到第三方平台

## 13. `plugins`

相关文件：

- `src/shaft/plugins/registry.py`
- `src/shaft/plugins/hooks.py`
- `src/shaft/plugins/interceptors.py`
- `src/shaft/plugins/proxy.py`

### 职能

- 提供注册表、hook、interceptor 和执行代理。

### 关键类

- `Registry`
- `HookManager`
- `TrainerHookCallback`
- `InterceptorManager`
- `ExecutionProxy`

### 关键函数

- `hook()`
- `build_hook_manager()`
- `interceptor()`
- `interceptable()`
- `build_interceptor_manager()`

### 开发边界

- 允许：横切增强
- 禁止：替代主业务流程、隐藏关键训练行为

## 14. `observability`

相关文件：

- `src/shaft/observability/logging.py`
- `src/shaft/observability/events.py`
- `src/shaft/observability/context.py`

### 职能

- 统一日志、上下文、事件。

### 关键函数

- `configure_logging()`
- `emit_event()`
- `set_log_context()`
- `bind_log_context()`
- `get_log_context()`

### 开发边界

- 允许：记录、格式化、注入上下文
- 禁止：参与训练决策、替代指标系统

## 15. `cli`

相关文件：

- `src/shaft/cli/train.py`
- `src/shaft/cli/common.py`
- `src/shaft/cli/sft.py`
- `src/shaft/cli/rlhf.py`
- `src/shaft/cli/infer.py`
- `src/shaft/cli/export.py`
- `src/shaft/cli/registry.py`

### 职能

- 定义命令。
- 做参数解析和无歧义 override。
- 路由到 pipeline / infer / export。

### 关键类

- `SFTCommand`
- `RLHFCommand`
- `CommandSpec`

### 关键函数

- `main()`
- `build_parser()`
- `add_common_train_args()`
- `apply_common_overrides()`
- `run_from_args()`
- `register_command()`

### 开发边界

- 允许：参数解析、命令调度
- 禁止：在 CLI 里堆叠训练、推理、导出业务逻辑

## 16. `webui`

相关文档：

- `docs/webui.md`

### 职能

- 面向工程师与科研人员提供 `SFT` 训练的可视化控制台。
- 复用现有 CLI、配置和日志体系。
- 做 YAML 编辑、少量高频 override、任务启动、日志与状态展示。

### 当前原则

- Web UI 是外层可视化壳，不是训练内核。
- Web UI 只应通过生成 YAML 并调用现有 CLI 进入训练主链。
- 不应在 Web UI 中复制数据、模型、算法、checkpoint 语义。

### 开发边界

- 允许：表单、预览、状态展示、日志轮询、CLI 调用封装
- 禁止：直连训练内核、发明新配置语义、引入第二套训练入口

### 当前实现结构

- `app.py`
  - `FastAPI` 路由装配与 HTML / JSON API 暴露
  - 顶部导航壳与页面路由切分（`SFT / DPO / PPO / GRPO`）
- `controller.py`
  - Web UI 事件处理与视图返回协议
- `templates/index.html`
  - 页面骨架与初始状态注入
- `static/webui.css`
  - 视觉样式与亮暗主题变量
- `static/webui.js`
  - 前端交互、状态刷新、主题切换
- `services/config_service.py`
  - YAML 读取、解析、override 应用
  - 冻结配置预览构造
- `services/train_service.py`
  - 子进程管理、run snapshot 读取
  - 运行时 freeze summary 读取
  - 运行时 optimizer group summary 读取
- `services/run_store.py`
  - 本地 run 目录、resolved config、日志与 record 管理
  - `shaft_finetune_summary.json` 读取
  - `shaft_optimizer_summary.json` 读取
  - 本地 run store 条目删除
