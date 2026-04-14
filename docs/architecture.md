# Shaft 架构总览与边界

本文档面向 `src/shaft` 的当前重构版本，定义可扩展边界和核心入口，避免后续实现踩到越界。

## 1. 层级与职责

### 1.1 `config`（配置中心）
- 位置：`src/shaft/config`
- 作用：统一配置定义、校验和加载。
- 关键文件：
  - `schema.py`：`RuntimeConfig`、`ModelConfig`、`DataConfig`、`SFTTrainConfig`、`SFTEvalConfig` 等。
  - `loader.py`：YAML 配置反序列化。
  - `normalize.py`：字段归一化/默认值补齐。
- 约束：训练/算法/推理逻辑不能直接解析配置细节；都通过结构化对象读取固定字段。

### 1.2 `data`（数据系统）
- 位置：`src/shaft/data`
- 作用：把上游数据转为 `SFTRecord`，不关心模型训练循环。
- 关键文件：
  - `sources.py`：`register_data_source`、`JsonlSFTDataSource`、`load_jsonl_records`。
  - `transforms.py`：`build_offline_pipeline`、`build_online_pipeline`。
  - `mixing.py`：`MixedDatasetBuilder` 与 `interleave_under/interleave_over/concat`。
  - `dataset.py`：`SFTRecord` 数据模型。
  - `collator.py`：`SFTCollator`（仅负责编码/组 batch）。
- 约束：不出现训练循环/梯度/模型目标逻辑。

### 1.3 `model`（模型族适配）
- 位置：`src/shaft/model`
- 作用：模型加载与 PEFT 变体策略封装，暴露统一产物。
- 关键文件：
  - `registry.py`：`MODEL_REGISTRY`、`register_model`、`build_model_meta`。
  - `types.py`：`ModelMeta/ModelLoader/ModelArtifacts/ModelCapabilities/ModelGroup`。
  - `builder.py`：`build_model_tokenizer_processor`，统一入口（含 adapter/full 初始化）。
  - `qwen3vl.py`：`Qwen3VLLoader`（按模型族实现）。
- 约束：不直接拼 batch 不做数据层混合，不做分布式训练 orchestration。

### 1.4 `template`（模板元信息）
- 位置：`src/shaft/template`
- 作用：聊天模板与模型族模板映射。
- 关键文件：
  - `registry.py`：`TEMPLATE_REGISTRY`、`register_template`。
  - `types.py`：`Template`/`TemplateMeta`。
  - `base.py`：`ShaftChatTemplate`（通用模板骨架）。
  - `qwen3vl.py`：`Qwen3VLTemplate`（模型族实现）。
- 约束：模板只负责 prompt 组装与解码，不承担任务评估语义。

### 1.5 `algorithms`（算法注册层）
- 位置：`src/shaft/algorithms`
- 作用：算法抽象与实现接入。
- 关键文件：
  - `base.py`：`Algorithm` 协议。
  - `registry.py`：`ALGORITHM_REGISTRY`、`register_algorithm`。
  - `sft.py`：`SFTAlgorithm`（当前可运行路径）。
  - `dpo.py`、`ppo.py`：占位接口。
- 约束：算法层接收统一模型/数据对象，不直接改动数据层/模型层内部实现细节。

### 1.6 `pipeline`（编排层）
- 位置：`src/shaft/pipeline/train.py`
- 作用：把配置、模型、数据、算法组装为一次可执行训练流程。
- 关键文件：
  - `train.py`：`ShaftTrainPipeline`。
  - `registry.py`：`register_pipeline`。
- 流程核心：
  - `build_datasets()` -> `SFTDataset`（离线/在线 transform + mixing）
  - `build_model_tokenizer_processor()` -> `ModelArtifacts`
  - 算法入口 `ALGORITHM_REGISTRY` -> trainer
  - `trainer.train()` -> checkpoint/export
- 约束：不承载模型适配细节，不写数据字段语义解析。

### 1.7 `training`（训练内核）
- 位置：`src/shaft/training`
- 作用：`Trainer` 扩展点和通用模块。
- 关键文件：
  - `trainer.py`：`ShaftSFTTrainer`，`compute_loss/create_optimizer/create_scheduler/evaluate/save` 重写点。
  - `loss.py`：`LOSS_REGISTRY`，`auto`/`causal_lm`。
  - `optimizer.py`：`OPTIMIZER_REGISTRY`（含 `muon`）。
  - `scheduler.py`：`SCHEDULER_REGISTRY`（默认 cosine）。
  - `checkpointing.py`：`ensure_hf_export_layout`、`validate_resume_checkpoint`。
- 约束：loss/optimizer/scheduler 只负责训练内部策略，不做数据来源判断。

### 1.8 `plugins`（横切能力）
- 位置：`src/shaft/plugins`
- 作用：注册、Hook、Interceptor、Execution Proxy。
- 关键文件：
  - `registry.py`：通用 `Registry`。
  - `hooks.py`：`@hook`、`HookManager`、`TrainerHookCallback`。
  - `interceptors.py`：`@interceptor`、`interceptable`、`ExecutionProxy`（`proxy.py`）。
  - `builtin_hooks.py`、`builtin_interceptors.py`：默认埋点。
- 约束：横切逻辑只能做日志/监控/策略注入，不替代主流程职责。

### 1.9 `infer`（推理编排）
- 位置：`src/shaft/infer`
- 作用：推理引擎与多阶段编排。
- 关键文件：
  - `engine.py`：`InferEngine`（单模型/单阶段）。
  - `pipeline.py`：`InferPipeline`（多引擎、多阶段）。
  - `schema.py`：推理配置类型。

### 1.10 `cli`（命令入口）
- 位置：`src/shaft/cli`
- 作用：统一入口 `scripts/train.py` 与子命令注册。
- 关键文件：
  - `train.py`：`build_parser/main/_normalize_argv`。
  - `registry.py`：`COMMAND_REGISTRY`、`register_command`。
  - `common.py`：`add_common_train_args/apply_common_overrides/run_from_args`。
  - `sft.py`：`sft` 命令。
  - `rlhf.py`：`rlhf --algorithm` 命令骨架。

## 2. 关键边界映射（“谁负责什么”）

- 数据字段（如 `dataset_id`, `system_prompt`, `user_prompt`）的解释只在 `data` 层完成。
- 模型族差异（PEFT target module、template 归属、processor 策略、必需文件）只在 `model`/`template` 层处理。
- 评估指标和 best model 规则放在 `config + trainer + pipeline`，`trainer` 层不编码任务语义。
- 调度策略（训练命令、日志、模型导出）在 pipeline + 配置中心驱动。

## 3. 当前能力覆盖（已落地）

- 多数据源、多 jsonl、加权混合、`interleave_*`、`concat` 已在 `data` 层实现。
- 单卡/多卡基础训练、resume/init 路径校验、HF 风格 checkpoints 与 adapter/full 流水线已接通。
- 多模型推理引擎与多阶段推理管线可配置化执行。

