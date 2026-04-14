# Shaft 扩展指南（任务：RL、模型族、数据系统）

本指南描述在当前架构上新增能力时的最小改动路径，避免越界耦合。

## 1. 添加新的训练算法（如 DPO/PPO）

### 已有扩展点
- 注册：`src/shaft/algorithms/registry.py`
  - `ALGORITHM_REGISTRY`
  - `register_algorithm(name)` 装饰器
- 协议：`src/shaft/algorithms/base.py`
  - `Algorithm`（`build_trainer(...)`）
- 调度：`src/shaft/pipeline/train.py` 的 `algorithm = ALGORITHM_REGISTRY.get(config.algorithm.name)`
- CLI：`src/shaft/cli/` 下命令路由。

### 标准步骤
1. 在 `src/shaft/algorithms/<algo>.py` 添加实现类并注册：
   - `@register_algorithm("dpo")`
   - `build_trainer(...)` 返回对应 `Trainer` 实例。
2. 在 `src/shaft/cli/` 增加命令（若与现有 `rlhf` 兼容可复用）：
   - 参考 `src/shaft/cli/rlhf.py`
3. 在 `config/schema.py` 增加或完善 `RLHFConfig` 的结构化字段：
   - 目前已预留 `rlhf` 节点，可按算法细分（例如 `rlhf.dpo`, `rlhf.ppo`）。
4. 如需新增 CLI 参数，优先在 `cli/common.py` 的 `add_common_train_args` 中追加白名单参数并映射到 `RuntimeConfig`。
5. 增加测试：
   - `tests/test_pipeline_train.py` 级别做流程冒烟
   - `tests/test_<algo>.py` 覆盖 `build_trainer` 入参与参数映射。

### 关键约束
- DPO/PPO 不能复用 SFT 的损失模块时序；应在各自 trainer 内封装。
- 任何算法必须消费统一的 `RuntimeConfig` / 已构建 `ModelArtifacts` / `SFTDataset`，不直接改 `data`/`model` 读写逻辑。

## 2. 添加模型族（Model Adapter）

### 已有扩展点
- 注册：`src/shaft/model/registry.py`
- 元信息：`src/shaft/model/types.py`
  - `ModelMeta`、`ModelLoader`、`ModelInfo`、`ModelArtifacts`
  - `ModelGroup`（按模型路径自动匹配 template/额外文件/依赖）
- 构建入口：`src/shaft/model/builder.py` 的 `build_model_tokenizer_processor`
- 模板依赖：`template/` 中的 template meta 与实现

### 标准步骤
1. 在 `src/shaft/model` 下创建模型文件，如 `glm4v.py`（含 `@register_model(ModelMeta(...))`）。
2. 定义 `ModelMeta` 字段：
   - `model_type`（必须唯一）
   - `family`, `default_template`
   - `model_groups=default_model_groups(...)`（按路径匹配）
   - `capabilities`, `processor_policy`, `peft_policy`, `requires`, `additional_saved_files`
3. 实现 `ModelLoader` 子类：
   - `build(self, config, model_meta)` 返回 `ModelArtifacts(model, tokenizer, processor, model_meta, model_info, template)`
   - 使用 `resolve_template_meta` 选择模型模板
   - 使用 `apply_finetune_strategy` 注入 full/lora/dora/qlora 策略
4. 在同名模板文件增加模板适配：
   - `template/glm4v.py` 使用 `@register_template(TemplateMeta(...))`
   - 提供 `Template` 子类（可继承 `ShaftChatTemplate` 或自定义）
5. 补充测试：
   - `tests/test_model_registry.py` 覆盖注册
   - `tests/test_template_registry.py` 覆盖模板映射
   - `tests/test_infer_loader.py`（推理构建）可加最小覆盖

### 关键约束
- 模型特化逻辑必须在 `<model>.py` 或 `<model>_adapter` 中，不得散落到通用数据/训练路径中。
- template 名称应贴近模型族（`qwen3vl`、`glm4v`），避免通用名称误导。

## 3. 数据注册中心（多 JSONL、多 dataset、多策略）

### 当前机制
- 数据源注册：`src/shaft/data/registry.py`
- 数据源实现：`src/shaft/data/sources.py`
- 离线/在线增强：`transforms.py` + `mixing.py`
- Pipeline 统一组装：`pipeline/train.py` 的 `build_datasets()`

### 推荐扩展方式（每个 JSONL 一个数据源实例）
1. 在配置中继续使用 `data.datasets` 数组（每个为 `DataSourceConfig`）：
   - `name`、`source_type`、`train_paths` / `val_paths` 支持多个文件。
2. 通过 `weight` 控制采样权重。
3. 通过 `mix_strategy` 控制训练采样策略（`interleave_under`/`interleave_over`/`concat`）。
4. 离线增强（`offline_transforms`）用于预处理一次性应用；
5. 在线增强（`online_transforms`）在 sample 取出后按 `dataset_id` 做二次处理。

### 扩展新数据源
1. 在 `data/sources.py` 增加新 `BaseDataSource` 子类。
2. 用 `@register_data_source("xxx")` 注册。
3. 在 `load_split()` 内只负责返回 `list[SFTRecord]`，不触发训练调度。

## 4. 推理系统扩展（多模型多阶段）

### 现有入口
- `src/shaft/infer/engine.py`: 单引擎推理
- `src/shaft/infer/pipeline.py`: 读取 `InferPipelineConfig` 执行多阶段

### 新场景建议
- 每个阶段绑定不同 `engine` 与 prompt 模板；
- 用 `output_key` 串联阶段上下文；
- 可在 `InferStageConfig` 增加阶段级生成参数（或复用默认生成参数）。

## 5. checkpoints 与 resume/init 规则

- full: 采用 HF export，包含完整权重与附加文件。
- lora/dora/qlora: 必须是 PEFT adapter checkpoint 格式（`adapter_config.json` + `adapter_model.*`）。
- 校验与转换位于：
  - `src/shaft/training/checkpointing.py`
  - `src/shaft/model/builder.py`（`init_from` 适配）

## 6. 交付检查清单（新增功能前置）

1. 注册是否存在（registry 中可查询）？
2. 配置 schema 是否有可落地字段？
3. CLI 是否有可见参数入口且有默认值？
4. 是否有至少 1 个对应的 smoke/pipeline test？
5. 保存/恢复路径是否按模式验证？
6. 是否保证不污染其他层职责边界？

