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
- `DataScheduleConfig`：logical draw 的 mixing / shuffle
- `DataTransformsConfig`：逐 draw 的 prompt sampling 等样本视图变换
- `PromptSamplingConfig`
- `DataBatchingConfig`：grouping / cardinality / packing / layout 的父 contract
- `DataPackingConfig`：独立的 logical-sequence packing 入口
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
- `to_resolved_payload()`：生成 canonical launch snapshot；不是 source YAML 的无损 roundtrip

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
- `src/shaft/data/cost.py`
- `src/shaft/data/batching.py`
- `src/shaft/data/dynamic_batching.py`
- `src/shaft/data/sampler.py`
- `src/shaft/data/transforms.py`
- `src/shaft/data/registry.py`
- `src/shaft/data/context_attribute_contract.py`

### 职能

- 读取并规范化多数据源，构建可复用的 Arrow mmap record store。
- 做离线/在线增强和 sample-level mixing。
- 提供 horizon-independent draw schedule 及有限长度 HF Dataset adapter。
- 按需估算 processor/template 后成本，并在有界 buffer 内形成 fixed 或 token-budget cost-grouped batches。
- 产出 Dataset、sampler 和 collator；不承载 optimizer 或训练循环。
- 为离线真实属性弱标注提供 exact-key shape attribute 合同校验；本地/API handoff gate 与仓库内 SFT
  builder 共用同一真源，防止嵌套额外字段进入监督目标；本地 API caller 不属于仓库正式入口。

### 关键类

- `ShaftDataCenter` / `ShaftDatasetBundle` / `ShaftDatasetMeta`
- `SFTRecord` / `DPORecord` / `PPORecord`
- `ShaftSampleSchedule` / `ShaftSamplePlan` / `ShaftSampleRef` /
  `ShaftSampleContext`
- `ShaftSampleSampler` / `ShaftGroupedSampleContract` / `ShaftGroupedSampleSampler` /
  `ShaftPlannedBatchSampler`
- `ShaftSampleCost` / `ShaftSFTSampleCostProvider` /
  `ShaftRowInvariantCostProvider`
- `ShaftBatchPlanningSpec` / `ShaftBatchPlanningState` /
  `ShaftBatchPlanner`
- `SFTDataset` / `DPODataset` / `PPODataset`
- `SFTCollator` / `DPOCollator` / `PPOCollator` / `GRPOCollator`

### 核心约束

- `ShaftSampleSchedule.ref_at(draw_id)` 是 bounded 训练的 sampling 真源。它绑定 source sizes/weights、
  strategy、shuffle、seed，但不绑定 max steps。有限 `ShaftSamplePlan` 只给 map-style Dataset 和普通
  sampler 提供 `len()`；bounded DataCenter 直接返回 schedule，不创建 duration-sized plan。
- `concat` 和 `weighted + shuffle=true` 都有正式 horizon-independent `ShaftSampleSchedule`；
  `weighted + shuffle=false` v3 虽已使用跨 cycle 连续的 exact-ticket stream，但当前只由 finite fixed-plan
  adapter 消费，bounded/planned API 仍 fail closed，不把尚未开放的执行路径伪装为支持。
- `weighted + shuffle=true` 使用固定配额 ticket block；最多 64 个 source 时完整搜索 16K 内的 Hamilton
  最优解，更大 catalog 先走最多 32 个 denominator-derived 候选；无合法候选时用 quota 区间和线性
  Hamilton rank predicate 完整验证剩余可行 block，不对每个 block 重复排序 catalog。每个 source 强制
  5% 相对误差上限，无法表示时 fail closed。seed-specific base block 只物化一次；rotation 每 256 blocks
  用 SplitMix64 更换 phase，group 内使用与 block 和 `1..64` rank modulus 互质的 counter step，让稀有
  source 的有限前缀具备确定性 rank discrepancy 上界。每源 ticket position list 用于
  O(log quota) occurrence rank 查询。每个 occurrence 再通过
  keyed Feistel permutation 映射到独立 row cycle，因此 canonical draw stream 在 source 耗尽前无重复，
  运行时状态与训练 horizon 无关。schedule/finite-plan fingerprint 绑定算法版本、salt 和 base-block digest，
  旧的 v2 weighted checkpoint 不能 exact resume。
- `weighted + shuffle=false` 复用相同 quota resolver，因而每个正权重 source 必须有正 ticket、resolved
  probability 相对误差不超过 5%，16K 内无法表达就 fail closed。它不随机置换 ticket，而是按每-source
  occurrence midpoint 做 exact k-way merge，得到确定性的低差异 block；finite-plan cycle 只切分这条连续
  stream，不重新舍入 quota。row occurrence 在 block/cycle 边界连续，unshuffled v2 checkpoint 不能 exact
  resume 到 v3。stream fingerprint 排除 finite horizon，plan/execution fingerprint 再绑定 `num_samples`，
  不能用跨 horizon 比较身份替代 exact-resume 身份。
- prompt sampling/online transform 使用 `draw_id/transform_seed`，并必须通过
  `planning_safe_online_transform(fingerprint=...)` 声明可重复、保持媒体 identity/geometry 和 placeholder。
- `ShaftSFTSampleCostProvider` 调用 Dataset 的 `get_planning_item()`，不解码图片，只读取按需 image
  header。sample cost 按 logical draw 缓存，header 按 canonical path 缓存；两者都受容量限制。provider
  fingerprint 绑定显式 immutable `media_snapshot_id`，但 schedule fingerprint 只由 batching spec 持有。
- `ProcessorPolicy` 是图像 patch 和 processed-token layout 真源，Template 是监督 token、EOS、截断、
  causal shift 与 loss weight 真源。cost provider 不维护模型/模板的平行实现。
- `ShaftBatchPlanner` 每次只补充至 `buffer_size`，强制选择最老 draw，并按
  `per_device_train_batch_size` 上限、cardinality policy、padded-token 和 resource budgets 做确定性分组。
  fixed 模式贪心失败时使用有界 exact fallback；未选 entry 保持 FIFO，已选、buffer、future cursor
  精确分割 draw stream。
- planner 每个 global microstep 输出 W 个非空 local batches。`ShaftPlannedBatchSampler` 只认识
  global-microstep stream 与 generic planning frame，并在 frame 内按 rank 累计 load 分配后展平给
  Accelerate；training callback 单独解释 GA/global step。sampler 内不做 collective。
- state 只保存轻量 ref/cost/cursor/实际累计指标。训练 callback 按实际完成的 optimizer boundary commit
  snapshot，不能保存 DataLoader 预取后的 live cursor。
- 普通 `ShaftSampleSampler` 交给 HF Trainer 时保持 global/unsharded，由 Accelerate 完成唯一一次 rank
  分片；预分片 sampler 会 fail fast。fixed SFT 在 rank step 数一致时可保留 token-normalized 的异长尾批；
  DPO 还要求每个同步 step 的 rank-local cardinality 相同，避免 TRL local-mean 偏权和变长 metric gather。
- `ShaftGroupedSampleSampler` 按 canonical `ShaftSamplePlan` 顺序扩展 GRPO 所需的
  mini-repeat/repeat-count，不执行第二次 position shuffle。GRPO 的 TRL `shuffle_dataset` 被显式关闭，
  source/row 顺序仍只由 `data.schedule` 决定。`ShaftGroupedSampleContract` 是 repeat 几何真源，分别绑定
  mini-repeat、group batch、iteration 与 steps-per-iteration；其 composite execution fingerprint 在模型
  加载前参与 resume 校验，不能用乘积相同但 cadence 不同的配置续训。contract 同时证明并返回真实的
  rank-local epoch microstep 数；sampler 与 checkpoint cadence 不再各自推导一份 epoch 几何。
- `data.schedule` 只决定 mixing 与 shuffle，形成确定性的 logical draw stream。
- `data.transforms.prompt_sampling` 按 dataset name 选择 prompt pool，采样键来自 draw context，默认只作用于
  train；它变换 draw view，不改变 draw 顺序或 mixing 权重。静态/参数化 variant 都由 `shaft.prompting`
  编译和渲染，SFT `prompt_args` 保持 JSON 类型进入同一 planning-safe transform。
- Arrow build-time record validator 与非空 validation fingerprint 是不可拆分的 API contract；二者必须同时
  提供或同时省略，防止 cache hit 绕过新校验。train execution fingerprint 还组合规范化 record store 与
  `media_snapshot_id`，不能只绑定 sample 数量和 transform。
- `data.batching` 依次声明 grouping、cardinality、packing 与 layout。四者没有覆盖优先级。
  `ShaftLengthBatchGrouping` 只排序有界候选，`ShaftGreedySequencePacker` 只组合完整 logical segments，
  `ShaftVarlenBatchLayout` 只组装计划后的 tensor；Qwen 的 attention/M-RoPE/media 语义归模型
  `SequenceExecutionPolicy`，不回流到 sampler 或通用 collator。
- 合成 reconstruction 数据的离线真源仍是 `gt_standard`；数据构建脚本与训练 batching 相互独立，不得把
  任务字段语义放入 planner。
- v5.3 contextual reconstruction 由 `scripts/tasks/build_context_reconstruction_sft.py` 离线生成；
  v5.2 region manifest 只选择实例，proposal/crop 审计信息留在 derived `extra`，训练主链只消费标准
  `jsonl_sft`、动态 `prompt_args` 和 task prompt pool，不解释 proposal 或几何字段语义。
- 真实 API 弱监督 shape 属性使用独立 `shape_context_attributes` dataset/task；它复用 v5.3 contextual
  crop/proposal 输入合同，但 target 有意省略 geometry，不能并入完整 `shape_context_reconstruction`。
- line 点监督使用独立 `line_context_points` dataset/task；它同样复用 v5.3 contextual crop/proposal
  输入合同，但 target 只允许完整 line DSL 的 `is_single + points` 子集。真实单叉的 source bbox 与有序
  linestrip 来自归档 point structured，clean full-image 通过归档 grounding manifest 恢复；合成补充
  数据只从 `gt_standard` 选择 `is_single=false` 且有多个 segment 的分叉线，按segment数量确定性封顶
  抽样，并使用 `synthetic_realism_v1`。该抽样不创建resize/multi-scale副本。训练主链仍只消费标准
  `jsonl_sft`，不得在 collator 中解释 source provenance、混入合成单叉或补造缺失 line 属性。
- synthetic shape/line 的 `synthetic_realism_v1` 也只属于离线 data builder：每条 crop 必须有1–3个
  尺寸不变的像素扰动，参数写入 `extra.pixel_augmentation`；训练、pipeline 和 collator 不重复推导该策略。
  task-local `selection/train.jsonl` 只保存源实例身份，属性与几何真值仍回查 source truth。

### 开发边界

- 允许：样本格式、增强、mixing、cost contract、sampler、collator。
- 禁止：optimizer step、scheduler、checkpoint 文件策略、任务级字段解析。data sampler 可提供 generic
  planning-frame 原子性，但不能解释 Trainer global step 或 GA。
- full-horizon CostPlan/mmap、fixed cost-aware planner、fixed guard、exact optimizer sample/token target
  已删除，不得以兼容名重新引入第二套运行时。

## 3. `model`

相关文件：

- `src/shaft/model/types.py`
- `src/shaft/model/builder.py`
- `src/shaft/model/registry.py`
- `src/shaft/model/descriptor.py`
- `src/shaft/model/resolution.py`
- `src/shaft/model/policies.py`
- `src/shaft/model/inference.py`
- `src/shaft/model/qwen_inference.py`
- `src/shaft/model/sequence.py`
- `src/shaft/model/sharding.py`
- `src/shaft/model/finetune.py`
- `src/shaft/model/qwen3vl.py`
- `src/shaft/model/qwen35vl.py`

### 职能

- 声明模型族元信息。
- 构建 HF model/tokenizer/processor。
- 处理全量微调与 PEFT。
- 收口 processor / inference / sequence execution / peft / sharding policy。

### 关键类

- `ModelMeta`
  - `hf_model_types` 是本地 HF `config.json:model_type` 兼容性校验的真源，用于在真正加载
    checkpoint 前拦截 `model.model_type` 与权重目录不匹配的问题。
- `ResolvedModelDescriptor`
  - 从本地或 HF cache/Hub config 解析 architecture/model type/layer facts；多 variant family 必须据此选择
    `ModelGroup`，未知或歧义 profile fail closed，catalog basename 不能覆盖 config 事实。
- `ResolvedModelPlan`
  - 一次性绑定 configured/effective artifact、init kind、descriptor、`ModelMeta`、`ShaftModelAdapter` 与 HF
    revision/cache/offline 参数。pipeline 构造一次并把同一个 plan 交给 sequence contract 与 builder，禁止
    重复解析路径、variant 或 checkpoint 类型。
  - checkpointable 本地 HF 路径在 plan 阶段建立一次完整 SHA-256 baseline；builder 用短生命周期 metadata
    load guard 包住 HF loader，并在 loader 返回后做一次完整 SHA-256 closure。常态每个文件只完整扫描两次，
    不再执行 baseline/pre-load/post-load 三次扫描；即每 rank 额外读取 `2 × artifact bytes`（HF loader 本身
    另计）。guard 不是持久 cache，不能替代 closure hash；当前也不跨 rank 共享本地 digest。
  - 本地 sharded checkpoint 的 `weight_map` 使用 strict entry/path validator；不会忽略非法 value，也不会
    hash 或加载模型目录之外（包括 symlink escape）的 shard。非 indexed weight、config 和本地 remote-code
    文件复用同一 containment validator。
- `ResolvedAdapterInit`
  - adapter init 的不可变身份，绑定 canonical `adapter_config.json`、声明的 base model、权重 manifest 与
    artifact fingerprint。builder 只消费该 plan，并以运行时 PEFT canonical target/modules-to-save 加上
    state key/shape 完整性验证，兼容 PEFT 对完整模块路径的持久化归一化。
- `LoadedAdapterArtifacts`
  - model 层给 adapter merge/export 的专用返回类型，只暴露已验证的 `PeftModel` 与 processor/tokenizer/
    model adapter；不伪造训练态 `finetune_plan`。
- `ShaftModelAdapter`
- `ProcessorPolicy`
- `ProcessorInputPolicy`
- `ShaftInferencePolicy`
- `QwenVLInferencePolicy`
- `ShaftProcessedBatch`
- `ShaftProcessorTokenLayout`
- `ShaftProcessorCostEstimate`
- `ShaftSequenceExecutionContract`
- `SequenceExecutionPolicy`

`ShaftProcessedBatch` 保存一次 batch processor 调用的完整输出，不把允许字段限制为 Qwen 当前使用的
键。`ProcessorPolicy` 同时声明 processor 构造参数、pixel-budget forwarding、token-layout 规则、训练
输入的复制/重排、精确 image-cost estimation 和版本化 `cost_semantics_signature()`。该 signature 必须
绑定 estimator 读取的全部 processor 状态；内置 `identity` 要求 rendered tokens 与
processed tokens 完全一致且不声明成本估算能力；`qwen_vl`
显式处理 `mm_token_type_ids` 标记的多模态 token run。其他模型族必须通过 registry 注册自己的 policy，
不得让 collator 或通用 template 猜测模型字段、batch axis 或 token expansion。processor 新增
`position_ids/token_type_ids` 等 sequence-aligned 字段时，默认策略会显式拒绝，直到模型 policy 定义
如何随 target 拼接、padding 或 DPO pair 扩展。其它字段也必须进入 policy 的
`sample_aligned_model_input_names / whole_batch_model_input_names / static_model_input_names` 之一；未声明
字段会在训练装配时失败，避免升级 processor 后静默误用第 0 维。

`ProcessorInputPolicy` 是 processor padding 语义真源：training mode 默认为 right padding，generation
mode 默认为 left padding。`EvalInputPolicy` 位于 config 层，解析 `data.* -> eval.* ->
eval.datasets.<name>.*` 的 pixel-budget 覆盖关系；SFT loss/online eval 与 DPO eval collator 只消费解析结果。
PPO 的 query 虽来自训练 dataloader，实际直接进入 rollout generation，因此 `PPOCollator` 在类级把默认
input mode 定义为 `generation`，异长 query 不会继承 training/right padding。GRPO 的现有评估能力只接
online `eval_final_score`；由于 `ShaftGRPOTrainer` 尚未提供可靠的 loss eval / `eval_final_loss` 聚合，config
normalize 会拒绝 `algorithm=grpo + eval.enabled=true + loss_metrics_enabled=true`。

`ShaftInferencePolicy` 是模型推理输入契约真源，经 `ModelMeta` 解析到 `ShaftModelAdapter`。它负责
media、messages、chat-template backend kwargs 与 pixel-budget 行为，并按 backend 显式 opt in；默认
policy 对本地和远端都 fail closed。`QwenVLInferencePolicy` 才拥有 Qwen smart resize 和
Qwen35/36 thinking template 参数，`src/shaft/infer` 不按模型名分支。

`ShaftSequenceExecutionContract` 是一次训练唯一的模型执行能力决议，绑定 layout、device、attention
backend、dtype、distributed strategy、compile flag 与模型 policy/依赖版本。pipeline 在数据/权重加载前
构造一次，并把同一个对象传给 loader、concrete runtime 校验和 exact-resume fingerprint；loader/trainer
不得再从 config 独立推导第二份 varlen 能力结果。模型专用 position、segment isolation 与 media manifest
逻辑由 `SequenceExecutionPolicy` 实现。
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
- `resolve_model_plan()`
- `prepare_model_build()`
- `invoke_model_loader()`
- `finalize_model_build()`
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

- 模型装配固定为三相：`prepare_model_build()` 只做 shard/require/load-guard 与 adapter 预检，
  `invoke_model_loader()` 是唯一允许拥有 HF/DeepSpeed collective 的相位，`finalize_model_build()` 做完整
  artifact SHA closure、remote-code identity 与 adapter 后验。SFT/RLHF 只给 prepare/finalize 套
  all-rank status convergence；raw loader 必须裸调用。普通 infer/export 继续调用
  `build_model_tokenizer_processor()`，由兼容 wrapper 顺序串联三相。
- `ModelModuleGroups` 负责声明模型族的结构分组：
  - `language_model`
  - `vision_tower`
  - `aligner`
  - `generator`
- `src/shaft/model/freeze.py` 统一执行冻结逻辑：
  - 作为低层规则工具，负责 group/prefix/regex 匹配
  - 结构分组匹配采用最具体前缀优先，例如 `model.visual.*` 会优先归到 `vision_tower`，不会被更宽的 `language_model=model` 误伤
- `ResolvedModelPlan` 将加载定位与轨迹语义分开：`configured_model_name_or_path`、
  `effective_model_name_or_path` 仍保留用于加载、错误报告和审计；只有已完成全文件内容摘要的本地 HF artifact
  才允许在 model-plan fingerprint 中忽略目录搬迁。Hub artifact 继续绑定 repo id + immutable commit，未完整
  物化的本地 artifact 继续绑定 locator，不会借“可迁移”语义绕过 fail-closed。
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
- `ShaftTrainerSpec`
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
- `validate_grpo_vllm_runtime_compatibility()`

### 开发边界

- 允许：trainer 选择、算法专属辅助对象、算法配置映射
- 禁止：加载数据文件、路径解析、主流程调度

补充说明：

- `GRPOAlgorithm` 当前使用共享 `codec` 注册表与内置 reward registry 组合 reward functions。
- GRPO 配置以 `rlhf.grpo.rollout` 描述采样参数，以 `rlhf.grpo.vllm` 描述 vLLM rollout 后端；旧 flat 字段仅作为兼容入口。
- GRPO `use_vllm` 在 cheap preflight 中同时经过两条独立校验：安装的 vLLM 必须满足当前 TRL vLLM extra 的
  platform requirement；若启用 checkpoint/resume，还必须证明 sampled-rollout RNG 可恢复。版本 gate 不影响
  独立 infer/vLLM 后端，RNG gate 也不能替代 package compatibility。
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
- `src/shaft/training/train_sampler_mixin.py`
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
- `ShaftGRPOTrainer`
- `ShaftOptimizerMixin`
- `ShaftTrainSamplerMixin`
- `ShaftResolvedOptimizerPlan`
- `ShaftOnlineEvalRunner`
- `ShaftProgressCallback`
- `ShaftCheckpointProtocol`
- `CheckpointLayout`
- `ShaftBatchPlanningCallback`
- `ShaftBatchingMetadataCallback`
- `ShaftBatchContract`
- `ShaftTrainingEfficiencyMonitor`
- `ShaftTrainingEfficiencyCallback`
- `Muon`

### 关键函数

- `build_loss()`
- `build_optimizer()`
- `build_resolved_optimizer_plan()`
- `build_scheduler()`
- `register_target_adapter()`
- `inspect_checkpoint_layout()`
- `resolve_resume_checkpoint()`
- `resolve_checkpoint_protocol()`
- `validate_resume_checkpoint()`
- `load_batch_planning_state()`
- `build_batch_planning_resume_contract_fingerprint()`
- `build_batch_contract()`
- `validate_batching_resume_contract()`
- `validate_training_state_policy()`
- `prune_root_output_layout()`

### 开发边界

- 允许：HF/TRL trainer 扩展、checkpoint 规则、优化器/调度器/loss
- 禁止：数据读取、配置加载、导出发布

补充说明：

- checkpoint 与 run metadata 分层：
  - `checkpoint-*` 是训练状态 generation；DDP/native-HF 只有通过 manifest validator 的目录才是 Shaft
    exact-resume 点，FSDP/DeepSpeed 的可恢复性由 backend-native contract 决定
  - `best` 是 HF/PEFT 部署导出
  - root `trainer_state.json`、`shaft_finetune_summary.json`、`shaft_optimizer_summary.json` 是持久化
    运行摘要，root export 清理必须保留
  - 从 run 根目录恢复时，最新 `checkpoint-*` 优先于 root final state
  - `ShaftCheckpointProtocol` 是 storage owner 真源：DDP/native-HF 路由到 `committed_manifest`，
    FSDP/DeepSpeed 路由到 `backend_native`
  - `ShaftCheckpointCommitMixin` 只在 `committed_manifest` 路径安装保存事务 wrapper：super save 前撤销旧
    `shaft_checkpoint_commit.json`，暂缓 HF rotation；callback-handler wrapper 先要求所有 rank 具有完全相同、
    顺序一致的普通 `on_save` callback 拓扑，再捕获每个 callback（包括 efficiency telemetry/plugin）的
    rank-local 异常并做 all-rank convergence。拓扑差异在 callback 执行前 fail closed；全部成功后才进入
    独立 commit phase、原子提交并执行 rotation。PPO 不接入该 mixin，仍禁止 resume
  - manifest 绑定非空 model/adapter、`trainer_state.json` hash、optimizer、scheduler、按保存 world size 的
    每-rank RNG，以及 checkpoint 内全部 recorded artifact 的相对路径、尺寸和 SHA-256。`committed_manifest` 的 direct-path 与
    run-root resolver 只接受通过同一 validator 的 committed checkpoint；run-root 会跳过未提交或已损坏的
    目录。`backend_native` 不安装 wrapper，沿用 HF/FSDP/DeepSpeed 的保存、发现、校验和 rotation，当前不提供
    通用 manifest 的 torn/atomic 保证
  - startup 用 `ResolvedResumeCheckpoint` 固定本次恢复的 protocol、step、content-derived generation identity、
    commit identity 与 stat guard。run-root 从新到旧验证到首个有效 generation 即停止；选中 generation 的大
    artifact 只做一次完整 digest 校验，preflight、planned state 与 train 入口复用同一 token。进入 Trainer 前
    再检查小 marker 与 stat guard，并对 generation identity 做全 rank consensus，避免重复多 GiB I/O，同时拒绝
    启动窗口内 stat-visible 的替换/改写或各 rank 同名路径对应不同内容
  - 所有训练路径都把 canonical `ShaftBatchContract` 通过 stateful callback 写入 HF
    `trainer_state.json`；改变 grouping/cardinality/packing/layout、local batch、DP world size 或 GA 时，
    exact resume 会拒绝
  - GRPO checkpoint 额外按真实 rank-local epoch microstep 几何验证 TRL generation buffer phase。完整
    grouped epoch 的末端天然是安全边界；step save、部分 epoch save 和 resume checkpoint 都使用 shortened
    accumulation 后的实际 microstep cursor 校验，不能只比较 `global_step * GA`。vLLM sampled rollout 的
    engine/server RNG 尚未持久化，因此 vLLM 训练当前禁止 checkpoint 与 resume，只允许最终模型导出
  - `ShaftTrainInputContract` 是 training 层从 logical sample 到模型输入的 exact-resume 真源。它组合
    DataCenter execution identity、train Dataset 类及其 effective runtime methods、Pillow runtime、model
    plan、model-owned processor policy、完整 tokenizer artifact 与 wrapper/backend package identity、
    processor/template、显式 collator/input-policy version、pixel/token limits；SFT 还绑定 sequence execution
    contract。该 canonical payload 与 fingerprint 随 batching metadata 一起进入 stateful callback，不把模型
    语义下沉到 data 层
  - `ShaftTrainingResumeContract` v2 是未来 optimizer update 的根契约；除 batch/model/optimizer/objective
    语义外，还直接组合 `ShaftTrainInputContract.fingerprint` 与 data execution fingerprint。batching metadata
    构造时会交叉验证三者一致，consumer 不能只保存并列字段而绕过根契约
  - tokenizer backend serialization 与完整 vocabulary 已形成内容身份后，HF 自动注入的
    `name_or_path`、cache/local flags 和 `*_file`/`*_path` 定位参数不进入语义 hash；其余 `init_kwargs`、special
    tokens、chat template、padding/truncation 和 runtime implementation 仍然绑定。model adapter 的加载路径由
    model plan 单独负责，input contract 与 SFT cost-provider fingerprint 不再重复解释同一 locator
  - 未版本化 online transform、缺少 immutable `media_snapshot_id`、Dataset/runtime identity 不完整、
    tokenizer identity 不完整、无显式 input-policy version，或 component state 含无法 canonicalize 的未知
    类型，都会把 input contract 标为 incomplete。data-side completeness 与 checkpoint 是否已有完整 payload
    在 model load 前 fail closed，processor/model 完整契约在装配后做第二阶段校验。`save_strategy=no` 的 fresh
    run 可继续并在 metadata/log 中明确记录 incomplete
  - planned SFT 额外保存 committed data state；改变 duration/optimizer/scheduler 或
    topology/data/media/processor/planner contract 时，exact resume 同样拒绝
  - planned checkpoint 会交叉校验 canonical batch-contract callback 与 planner callback；二者的 planner
    fingerprint、batch geometry、GA、committed cursor 和 resume contract 写入通用 manifest 的
    `batch_planning` extension，不再维护第二个 completion marker。旧独立 planning completion 不能作为新
    exact-resume 提交点；需要继承时使用 `init_from_checkpoint` 开新 schedule
  - `ShaftBatchContract` 是 resolved physical batch 的唯一真源；fixed 模式从
    `per_device_train_batch_size`、DP world size 与 GA 派生精确 physical-pack 计数，token-budget 模式派生
    min/max pack range，
    sampler spec 与 metadata 必须和它严格一致
  - `shaft_batching_run_metadata.json` 是 SFT/RLHF 共用的 resolved batching/audit 运行摘要，包含
    versioned canonical `batch_contract`、其 fingerprint、grouping/cardinality/packing/layout、topology、
    buffer/cache/budgets、可选的 `planner_spec_fingerprint` 与 canonical `train_input_contract`；其中 cache
    只供性能审计，不进入 exact contract。root export 清理必须保留，W&B `shaft_batching` 只镜像同一 payload
  - `shaft_training_efficiency.json` 是 collator 实际 tensor 到 optimizer commit 的执行效率真源；类型化
    contract 绑定公平 A/B identity。Trainer 在 model checkpoint 写入前发布 generation transaction，snapshot
    set 的 revoke、rank snapshot、manifest 与 transaction commit 各自通过 fixed-tensor status convergence，
    单 rank I/O 失败不会造成 collective mismatch；只在 transaction 已 committed 且
    generation/step/world/rank/contract 全部一致时续算历史

- `ShaftSFTTrainer` 会对一个 optimizer batch 内实际 causal labels/loss weights 求 denominator，并在 data
  ranks 间汇总；local numerator 采用同一个 global denominator，保证 gradient accumulation 与 DDP
  分割不改变 token 权重。
- `ShaftTrainSamplerMixin` 保留普通 `train_sampler` 路径，并在提供 `train_batch_sampler` 时通过 HF 明确
  支持的 `get_train_dataloader()` 扩展点构造 bounded grouped DataLoader。它同步复用 worker seed、
  prefetch、pin/persistent worker、MPS multiprocessing 与 Accelerate prepare 语义；throughput 使用 committed
  sample count，不改 optimizer-step 计算。
- `ShaftSFTTrainer.evaluate()` 在训练生命周期内把 eval 当作纯观察者：调用标准 HF evaluate 前后保存并
  恢复 Python、NumPy、Torch CPU 与当前 rank CUDA RNG。这样 persistent eval DataLoader 的首次 iterator
  创建不会因 uninterrupted/resume 生命周期不同而改变后续训练随机序列；独立 eval 调用保持标准行为。
- `training/reproducibility.py` 是训练启动 seed 的共享边界。SFT/RLHF pipeline 在 dataset/model/PEFT 装配
  前调用；普通模式使用 HF `set_seed`，full determinism 使用 `enable_full_determinism`，Trainer 后续仍
  保持上游标准初始化。
- `utils/semantic_identity.py` 是 model/input/resume/plugin implementation identity 的唯一编码边界。它绑定
  callable 的 code/default/closure、live class implementation、函数实际读取的 module attribute、普通嵌套
  对象的 type/implementation/state，并对无法确定性编码的值 fail closed。custom `Mapping` 必须显式实现
  `shaft_semantic_state()`，dataclass 变体也不例外；只有经过审计的 HF `_LazyAutoMapping` 使用内建有界投影，
  其中 reverse table 与 live methods 属于行为真源，`_modules` 访问缓存不属于。声明但不可编码的类属性不能
  退化为 type-only identity。PyTorch `ReduceOp.RedOpType` 以稳定 qualified type/name/value 编码，供 collective
  helper 的默认参数进入 deterministic identity。
- `utils/distributed.initialize_process_group_if_needed()` 在 fallible startup 前建立默认 group，并以 resolved
  `train.use_cpu` 而非 CUDA 可见性决定 Gloo/NCCL；model loader 与 Trainer/Accelerator constructor 作为
  collective owner 在 pure-local readiness consensus 之后直接调用，不能嵌套在 status envelope 中。
- `algorithms.ShaftTrainerSpec` 是 algorithm trainer 装配的两相边界。`prepare_trainer()` 只做 pure-local
  validation/config/model/reward 准备并返回 spec；pipeline 在全 rank `trainer-prepare` consensus 成功后，
  才于 envelope 外调用 `ShaftTrainerSpec.build()`。`build_trainer()` 仅作为兼容的串联入口，不能用于分布式
  pipeline startup。spec fingerprint 只绑定 rank-invariant 的 constructor-shaping TRL/HF args、Shaft optimizer
  选项与实际 prepared kwargs；model role 使用 bounded topology digest（module graph，以及 parameter/buffer 的
  name/shape/dtype/requires-grad，不读权重值），callable 使用共享 semantic digest，Trainer class 本身也以共享
  callable/component digest 绑定 live declared MRO/constructor，并单独绑定 constructor 读取的 live global
  helper，而非只记录 module/qualname。所有 rank 必须在 constructor 前一致，
  `local_rank/process_index/device` 等合法 rank-local 字段不能进入该指纹。
- `pipeline.execution.finalize_training_outputs()` 是 SFT/RLHF 训练结束保存的共享边界：先做全 rank readiness，
  再在 status envelope 外调用可能拥有 FSDP/DeepSpeed collective 的 `save_model()`；export layout 校验、
  `save_state()` 与 root prune 随后作为 rank-local 阶段统一收敛异常。最终 status collective 同时承担结束同步，
  不再在可能失败的 rank-local I/O 后放置裸 barrier。`best_export_dir` 在 readiness 内解析为规范化绝对路径，
  并由 readiness/finalization fingerprint 与 `save_model()` 共用；任一 rank 路径漂移必须在 model save 前拒绝。

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
    - 供 CLI 日志和外部诊断工具审计 resolved optimizer groups
  - `create_optimizer(model=None)` 与 Transformers 5.10 的 delayed optimizer consumer 对齐：FSDP1
    wrap 后传入 model 时，先从 wrapped model 重建 optimizer plan，并要求其 name/group fingerprint 与
    启动时 plan 完全一致；随后 optimizer 只持有 wrapped model 的 `Parameter` 对象。名称、trainable
    参数或分组语义发生漂移会直接失败。
  - 上述重绑定只支持 `use_orig_params=true`；`FlatParameter` 的 name/group 映射尚未实现，配置层明确拒绝
    `strategy=fsdp + use_orig_params=false`。
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
- `src/shaft/infer/execution.py`
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
- `ShaftInferExecutionControl`
- `ShaftInferAdapterCapabilities`
- `ShaftInferEngine`
- `ShaftInferPipeline`
- `ShaftInferStageResult`

### 关键函数

- `load_infer_config()`

### 开发边界

- 允许：后端封装、stage 调度、absolute deadline/cancellation 契约、共享 codec 调用
- 禁止：训练逻辑、离线业务脚本逻辑、任务 DSL

`timeout_seconds` 会解析成贯穿 stage retries/backoff/backend I/O 的同一个 absolute deadline。
vLLM adapter 将连接和分块 body read 绑定到剩余预算；本地 HF generate 当前不可安全抢占，收到
deadline/cancellation 时在工作开始前 fail closed。`ShaftInferPipeline.run` 可接收
`cancellation_event`，但 adapter 必须显式声明相应 capability。

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
  - 通过 model 层 `load_adapter_artifacts()` 复用 `ResolvedModelPlan / ResolvedAdapterInit`，先验证 base
    descriptor/variant、adapter 内容 identity 与精确 state key/shape；权重从同一份已 hash 的内存 byte
    snapshot 反序列化，再 merge 已验证的 `PeftModel`

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
- checkpointable 训练只接受显式 `trajectory_neutral=True` 的纯观测插件；marker 必须是 exact bool。实际
  manager 实例的顺序、实现与实例配置 state 进入统一 training-resume contract。纯 telemetry counter/cache
  可以在 resume 时重置，但不得影响模型输入、loss、梯度、RNG、数据游标或 TrainerControl；会影响训练轨迹的
  stateful 插件即使伪装成 neutral 也不受支持。
- `HookManager` 对 neutral observer 采用 rank-local 故障隔离：`before_step`、`after_step`、`on_save` 首次异常
  warning 一次并禁用该 observer，不给热路径增加故障共识 collective；backend-native checkpoint callback 同样
  适用。non-neutral hook 的异常保持 fail-fast。
- `ExecutionProxy` 将调用拆成 `prepare()` 与 `invoke()`：前者只执行 rank-local `before` interceptor，训练
  pipeline 先对该阶段做全 rank failure/fingerprint convergence；后者才进入可能拥有 collective 的 pipeline
  body，并仅在 target 成功后执行 `after`。collective owner 不能被包进 local status envelope。
  `prepare_pipeline_call()` 的 schedule 身份只从 `runner.interceptor_manager` 对该 point 的实际 ordered
  instances 派生，绑定 phase/name/order/position 与调用实现；配置 names 不是第二真源，同名实现漂移也会在
  body 前拒绝。

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
- 禁止：把修改 loss/梯度/RNG/TrainerControl 的插件标成 trajectory-neutral；当前没有 trajectory-affecting
  plugin state 的通用 checkpoint 恢复协议，插件私有文件不能冒充该协议
- 禁止：pipeline `before` interceptor 自行执行 distributed collective，或修改 pipeline 的零参数调用合同

## 14. `observability`

相关文件：

- `src/shaft/observability/logging.py`
- `src/shaft/observability/events.py`
- `src/shaft/observability/context.py`
- `src/shaft/observability/progress.py`
- `src/shaft/observability/efficiency.py`

### 职能

- 统一日志、上下文、事件和运行进度。
- `ShaftTrainingEfficiencySummary` 持久化 committed counts/timing/coverage/rank skew；
  `ShaftTrainingEfficiencyContract` 区分必须相同的实验 identity 与可变化的 batch/sequence 执行轴。
- `ShaftProgressManager` 是进程内进度真源；terminal/plain/JSON sink 只读消费与内部状态隔离的 snapshot；
  并发 mutation 原子提交并按 revision 顺序发布，close 不允许新 task 穿透。
- 训练回调与 online eval 只做事件适配，不持有独立进度条或刷新配置。
- `ShaftTerminalProgressPresentation` 是纯展示策略：统一高对比度 `━╸─`/ASCII bar、未知 total spinner、精确
  百分比、`s/it`/`it/s`、ETA、颜色角色与 metric 排序；未完成状态不会提前显示 `100%`。
  `ShaftTerminalProgressSink` 只维护前台 task 的短窗口 rate sample、indeterminate redraw ticker、真实
  terminal width、ANSI erase-line 与原地 I/O；ticker 只消费 immutable snapshot。嵌套阶段暂停父 task 计时，
  完成即回收 history，失败摘要不能被父任务重绘隐藏。
- terminal sink registry 按 stream 维护注册栈；`progress_safe_write()` 不得把显式目标流重定向到无关 manager，
  最新 sink 关闭后会恢复上一个仍存活的 sink。
- `logging.rank_zero_only=true` 时 structured console/file log 严格只由 rank 0 输出。非零 rank WARNING 不再
  绕过 terminal sink；需要 all-rank 调试时应显式关闭该开关，并使用 plain/off progress。all-rank 模式
  的每行包含 rank，file log 自动写入独立的 `.rank<N>` 文件。
- interactive sink 遵循单进程单共享 manager 契约；并行 pipeline 需共享 manager 或使用 plain/off。

### 关键函数

- `configure_logging()`
- `emit_event()`
- `set_log_context()`
- `bind_log_context()`
- `get_log_context()`
- `build_progress_manager()`
- `progress_safe_write()`
- `format_progress_percentage()`
- `select_progress_display_task_id()`

### 关键类

- `ShaftProgressManager`
- `ShaftProgressTask`
- `ShaftTerminalProgressPresentation`
- `ShaftTerminalProgressSink`
- `ShaftPlainProgressSink`
- `ShaftJsonProgressSink`
- `ShaftTrainingEfficiencyContract`
- `ShaftTrainingEfficiencySummary`

### 开发边界

- 允许：记录、格式化、注入上下文、维护进度状态并路由到输出 sink
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
