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
- `src/shaft/data/transforms.py`
- `src/shaft/data/registry.py`

### 职能

- 读取多数据源。
- 做离线/在线增强。
- 做样本级 mixing。
- 产出 `Dataset` 和 `Collator`。

### 关键类

- `ShaftDataCenter`
- `ShaftPreparedRecords`
- `ShaftDatasetMeta`
- `BaseDataSource`
- `SFTRecord` / `DPORecord` / `PPORecord`
- `SFTDataset` / `DPODataset` / `PPODataset`
- `SFTCollator` / `DPOCollator` / `PPOCollator`
- `MixedDatasetBuilder`

### 关键函数

- `build_data_source()`
- `build_dataset_metas()`
- `load_jsonl_sft_records()`
- `load_jsonl_dpo_records()`
- `load_jsonl_ppo_records()`
- `build_offline_pipeline()`
- `build_online_pipeline()`

### 核心接口

- 输入：`DataConfig`、JSONL 文件、图像路径
- 输出：
  - `ShaftDatasetMeta`
  - 记录列表
  - `Dataset`
  - batch tensor 字典

### 开发边界

- 允许：样本格式、增强、mixing、collator
- 禁止：训练状态、优化器、损失函数、任务级语义路由

## 3. `model`

相关文件：

- `src/shaft/model/types.py`
- `src/shaft/model/builder.py`
- `src/shaft/model/registry.py`
- `src/shaft/model/policies.py`
- `src/shaft/model/finetune.py`
- `src/shaft/model/qwen3vl.py`

### 职能

- 声明模型族元信息。
- 构建 HF model/tokenizer/processor。
- 处理全量微调与 PEFT。
- 收口 processor policy / peft policy。

### 关键类

- `ModelMeta`
- `ModelGroup`
- `ShaftModelAdapter`
- `ModelLoader`
- `ModelArtifacts`
- `ModelCapabilities`
- `ProcessorPolicy`
- `PeftPolicy`
- `Qwen3VLLoader`

### 关键函数

- `build_model_meta()`
- `build_model_tokenizer_processor()`
- `apply_finetune_strategy()`
- `summarize_finetune()`
- `build_processor_policy()`
- `build_peft_policy()`

### 核心接口

- 输入：`RuntimeConfig.model`
- 输出：`ModelArtifacts`

### 开发边界

- 允许：模型族差异、PEFT 策略、processor 策略、依赖检查
- 禁止：数据源路径、训练调度、推理 pipeline 业务语义

## 4. `template`

相关文件：

- `src/shaft/template/types.py`
- `src/shaft/template/base.py`
- `src/shaft/template/registry.py`
- `src/shaft/template/qwen3vl.py`

### 职能

- 把消息列表转为模型可消费 prompt。
- 定义 decode 协议。
- 通过模板元信息管理模型族模板实现。

### 关键类

- `TemplateMeta`
- `Template`
- `ShaftChatTemplate`
- `Qwen3VLTemplate`

### 关键函数

- `resolve_template_meta()`
- `build_template()`
- `build_template_from_meta()`
- `register_template()`

### 开发边界

- 允许：messages 规范化、chat template、decode
- 禁止：图像处理、任务后处理、训练超参数决策

## 5. `algorithms`

相关文件：

- `src/shaft/algorithms/base.py`
- `src/shaft/algorithms/sft.py`
- `src/shaft/algorithms/dpo.py`
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
- `PPOAlgorithm`

### 关键函数

- `build_reference_model()`
- `build_trl_dpo_config()`
- `build_trl_ppo_config()`
- `build_ppo_value_and_reward_models()`
- `validate_ppo_runtime_requirements()`

### 开发边界

- 允许：trainer 选择、算法专属辅助对象、算法配置映射
- 禁止：加载数据文件、路径解析、主流程调度

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

## 7. `training`

相关文件：

- `src/shaft/training/sft_trainer.py`
- `src/shaft/training/online_eval.py`
- `src/shaft/training/trl_trainers.py`
- `src/shaft/training/loss.py`
- `src/shaft/training/optimizer.py`
- `src/shaft/training/scheduler.py`
- `src/shaft/training/checkpointing.py`
- `src/shaft/training/progress_callback.py`
- `src/shaft/training/distributed.py`

### 职能

- 包装 HF/TRL trainer。
- 管理 loss/optimizer/scheduler 注册表。
- 统一 checkpoint 规则和分布式辅助能力。

### 关键类

- `ShaftSFTTrainer`
- `ShaftDPOTrainer`
- `ShaftPPOTrainer`
- `ShaftOnlineEvalRunner`
- `ShaftProgressCallback`
- `CheckpointLayout`
- `Muon`

### 关键函数

- `build_loss()`
- `build_optimizer()`
- `build_scheduler()`
- `register_target_adapter()`
- `inspect_checkpoint_layout()`
- `resolve_resume_checkpoint()`
- `validate_resume_checkpoint()`
- `validate_training_state_policy()`

### 开发边界

- 允许：HF/TRL trainer 扩展、checkpoint 规则、优化器/调度器/loss
- 禁止：数据读取、配置加载、导出发布

## 8. `codec`

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

## 9. `metrics`

相关文件：

- `src/shaft/metrics/base.py`
- `src/shaft/metrics/registry.py`
- `src/shaft/metrics/builtin.py`

### 职能

- 管理在线 eval metric 注册表。
- 提供单阶段在线 eval 的可复用指标实现。

### 关键类

- `ShaftEvalMetric`

### 关键函数

- `build_eval_metric()`
- `register_eval_metric()`

### 开发边界

- 允许：轻量在线指标、per-dataset metric 扩展
- 禁止：文本解析、数据路由、多阶段业务编排

## 10. `infer`

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

## 11. `export`

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

## 12. `plugins`

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

## 13. `observability`

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

## 12. `cli`

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
