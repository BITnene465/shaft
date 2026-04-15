# Shaft 架构总览

本文档描述 `src/shaft` 的正式架构、模块边界和稳定接口，用于指导日常开发、架构评审、代码 review 与后续 agent 协作。

## 1. 目标与范围

### 1.1 当前目标

- 以 `Hugging Face` 生态为唯一主干。
- 围绕多模态模型训练与推理构建稳定框架。
- 优先打磨 `Qwen3VL + SFT` 主路径。
- 通过注册表和适配层支持后续模型族、算法和推理后端扩展。
- 保持训练、保存、续训、导出都兼容 HF / PEFT / TRL 标准能力。

### 1.2 当前非目标

- 不做多生态兼容层，不接入 ModelScope 等平行生态。
- 不设计自定义 checkpoint 格式。
- 不将任务级语义路由放入训练内核。
- 不把推理编排做成任务 DSL。
- 不把 PPO/RM 包装成“已完成的生产能力”。

## 2. HF-first 边界

Shaft 当前明确是 `HF-first` 框架，这个边界必须在所有设计、实现和文档中保持一致。

- 训练内核：`transformers.Trainer` 与 `trl`
- 参数高效微调：`peft`
- 权重布局：HF full export / PEFT adapter export
- 推理后端：
  - `hf_local`
  - `vllm_openai`

禁止：

1. 引入自定义模型保存格式。
2. 在训练主干中塞入非 HF 生态的基础抽象。
3. 为兼容外部平台而污染当前配置、数据、训练接口。

## 3. 架构分层

```mermaid
flowchart TD
    Scripts["scripts/*.py<br/>薄包装入口"]
    CLI["src/shaft/cli<br/>命令解析与调度"]
    Config["config<br/>schema / loader / normalize / catalog"]
    Pipeline["pipeline<br/>SFT / RLHF 编排"]
    Data["data<br/>source / transform / mixing / dataset / collator"]
    Model["model<br/>loader / adapter / policy / finetune"]
    Template["template<br/>chat template / decode protocol"]
    Algorithms["algorithms<br/>SFT / DPO / PPO trainer 装配"]
    Training["training<br/>trainer / loss / optimizer / scheduler / checkpoint"]
    Infer["infer<br/>engine / pipeline / codec"]
    Export["export<br/>inspect / validate / merge-peft"]
    Plugins["plugins<br/>registry / hook / interceptor / proxy"]
    Obs["observability<br/>logging / context / events"]

    Scripts --> CLI
    CLI --> Config
    CLI --> Pipeline
    CLI --> Infer
    CLI --> Export

    Config --> Data
    Config --> Model
    Pipeline --> Data
    Pipeline --> Model
    Pipeline --> Template
    Pipeline --> Algorithms
    Pipeline --> Training
    Pipeline --> Plugins
    Pipeline --> Obs
    Model --> Template
    Model --> Training
    Infer --> Model
    Infer --> Template
    Export --> Model
    Export --> Training
```

## 4. 模块职责矩阵

| 模块 | 职责 | 关键稳定接口 | 明确禁止 |
| --- | --- | --- | --- |
| `config` | 配置 schema、YAML 加载、catalog 展开、严格校验 | `RuntimeConfig`、`load_config()`、`normalize_runtime_config()` | 训练循环、模型构建、JSONL 解析 |
| `data` | 数据源、记录结构、增强、mixing、dataset、collator | `ShaftDataCenter`、`BaseDataSource`、`build_data_source()` | optimizer/loss、训练阶段调度、任务级语义判断 |
| `model` | 模型族元信息、HF 加载、PEFT 包装、processor/peft policy | `ModelMeta`、`ShaftModelAdapter`、`build_model_tokenizer_processor()` | 数据路径处理、训练循环、推理 stage 编排 |
| `template` | 消息规范化、chat template、decode 约定 | `TemplateMeta`、`Template`、`build_template()` | 图像处理、任务后处理、generation 参数决策 |
| `algorithms` | 构建 SFT/DPO/PPO trainer 与算法专属辅助对象 | `SFTAlgorithm`、`DPOAlgorithm`、`PPOAlgorithm` | 读取数据文件、控制 pipeline、硬编码模型族 |
| `pipeline` | 训练主链编排和阶段调度 | `ShaftSFTPipeline`、`ShaftRLHFPipeline`、`run_sft()`、`run_rlhf()` | 任务语义、数据格式解析、模型专属 patch |
| `training` | Trainer 包装、loss/optimizer/scheduler、checkpoint 规则 | `ShaftSFTTrainer`、`ShaftDPOTrainer`、`ShaftPPOTrainer`、`build_optimizer()`、`build_scheduler()` | 配置加载、数据读取、导出发布 |
| `infer` | 单阶段推理执行、多阶段上下文传递、codec 解码 | `InferEngineConfig`、`ShaftInferEngine`、`ShaftInferPipeline`、`decode_with_codec()` | 训练逻辑、离线任务 DSL |
| `export` | HF 目录检查、PEFT merge、导出校验 | `inspect_hf_artifact()`、`validate_hf_artifact()`、`merge_peft_adapter()` | 自定义产物格式、发布平台适配 |
| `plugins` | hook / interceptor / 执行代理 | `Registry`、`HookManager`、`InterceptorManager`、`ExecutionProxy` | 替代核心业务流程 |
| `observability` | 日志、上下文、事件输出 | `configure_logging()`、`emit_event()` | checkpoint 决策、训练控制 |
| `cli` | 命令解析、无歧义 override、路由到 pipeline/infer/export | `main()`、`register_command()`、`run_from_args()` | 在 CLI 中堆叠业务逻辑 |

## 5. 训练主链

```mermaid
sequenceDiagram
    participant Script as scripts/train.py
    participant CLI as shaft.cli
    participant Config as shaft.config
    participant Pipeline as shaft.pipeline
    participant Model as shaft.model
    participant Data as shaft.data
    participant Algo as shaft.algorithms
    participant Trainer as shaft.training

    Script->>CLI: sft / rlhf
    CLI->>Config: load_config()
    Config-->>CLI: RuntimeConfig
    CLI->>Pipeline: run_sft() / run_rlhf()
    Pipeline->>Model: build_model_tokenizer_processor()
    Pipeline->>Data: ShaftDataCenter.build_dataset_pair()
    Pipeline->>Algo: algorithm.build_trainer(...)
    Algo-->>Pipeline: Trainer
    Pipeline->>Trainer: train()
    Trainer-->>Pipeline: metrics
    Pipeline-->>CLI: metrics
```

### 5.1 训练阶段关键接口

- 配置：`RuntimeConfig`
- 数据：`ShaftDataCenter`
- 模型：`build_model_tokenizer_processor()`
- SFT 编排：`ShaftSFTPipeline`
- RLHF 编排：`ShaftRLHFPipeline`
- HF 参数映射：`build_hf_training_args()`
- checkpoint 规则：
  - `inspect_checkpoint_layout()`
  - `resolve_resume_checkpoint()`
  - `validate_resume_checkpoint()`
  - `validate_training_state_policy()`

### 5.2 训练主链边界

1. `pipeline` 只装配组件，不承载任务语义。
2. `algorithms` 只构建 trainer，不读取 JSONL。
3. `data` 只产出样本和 batch，不涉及 loss/optimizer。
4. `model` 只负责模型族差异，不介入数据源路径和训练调度。

## 6. 推理主链

```mermaid
sequenceDiagram
    participant Script as scripts/infer.py
    participant Loader as shaft.infer.loader
    participant Pipeline as shaft.infer.pipeline
    participant Engine as shaft.infer.engine
    participant Codec as shaft.infer.codec

    Script->>Loader: load_infer_config()
    Loader-->>Script: InferPipelineConfig
    Script->>Pipeline: ShaftInferPipeline.from_config()
    loop 每个 stage
        Pipeline->>Engine: run(ShaftInferRequest)
        Engine-->>Pipeline: ShaftInferResponse
        Pipeline->>Codec: decode_with_codec()
        Codec-->>Pipeline: parsed payload
    end
    Pipeline-->>Script: outputs + __trace__
```

### 6.1 推理主链关键接口

- schema：
  - `InferEngineConfig`
  - `InferStageConfig`
  - `InferPipelineConfig`
- engine：
  - `ShaftInferEngine`
  - `ShaftInferRequest`
  - `ShaftInferResponse`
- pipeline：
  - `ShaftInferPipeline`
  - `ShaftInferStageResult`
- codec：
  - `decode_with_codec()`
  - `register_codec()`

### 6.2 推理边界

- stage 是编排单位，不是任务定义语言。
- codec 是文本输出的结构化解码器，不负责训练时数据规约。
- `backend_options` 是后端透传区，不应该变成模型专属大杂烩配置。

## 7. 稳定接口与演进接口

### 7.1 当前建议视为稳定的接口

- `RuntimeConfig` 及其一级配置块
- `ShaftDataCenter`
- `ModelMeta` / `ShaftModelAdapter`
- `TemplateMeta` / `Template`
- `ShaftSFTPipeline` / `ShaftRLHFPipeline`
- `ShaftSFTTrainer` / `ShaftDPOTrainer` / `ShaftPPOTrainer`
- `InferEngineConfig` / `ShaftInferEngine` / `ShaftInferPipeline`
- `inspect_hf_artifact()` / `validate_hf_artifact()` / `merge_peft_adapter()`

### 7.2 当前不应在外部承诺长期稳定的接口

- PPO 运行时细节与限制条件
- interceptor 的 `point` 字符串全集
- 单个模型族的细粒度 `processor_kwargs`
- 临时 smoke model / smoke template 能力

## 8. 当前明确受限的能力

- PPO 仍是受限能力，不能视为完整生产功能。
- 当前只有 `qwen3vl` 是正式模型族实现，`smoke_vlm` 仅用于测试。
- 结构化任务评估还未形成独立评估子系统。
- 发布到 Hub 的工具链尚未开始。

## 9. 架构约束清单

### 9.1 允许

- 通过注册表扩展模型、模板、算法、数据源、codec、命令。
- 通过 `ModelMeta -> ShaftModelAdapter` 收敛模型差异。
- 通过 `ShaftDataCenter` 统一多数据源、增强和 mixing。
- 通过 `training/checkpointing.py` 统一 HF 兼容训练状态规则。

### 9.2 禁止

1. 在 `training` 中解析 JSONL 或图像路径。
2. 在 `data` 中写 loss、optimizer、scheduler。
3. 在 `pipeline` 中硬编码模型族模板细节。
4. 在 `template` 中实现任务后处理或数据规约。
5. 在 `export` 中引入自定义模型目录格式。

## 10. 相关文档

- [docs/README.md](README.md)
- [docs/module_reference.md](module_reference.md)
- [docs/config_reference.md](config_reference.md)
- [docs/infer.md](infer.md)
- [docs/export.md](export.md)
- [docs/extension_guide.md](extension_guide.md)
- [docs/development_workflow.md](development_workflow.md)
- [docs/testing.md](testing.md)
- [docs/project_skill.md](project_skill.md)
