# Shaft 项目记忆

本文档是面向后续维护者和 agent 的项目级记忆摘要，用来快速恢复当前框架的共识、边界和扩展方式。

## 1. 一句话定位

Shaft 是一个 `HF-first` 的多模态训练与推理框架，当前主目标是把 `Qwen3VL + SFT` 主链做稳，并在此基础上保留面向 RLHF、更多模型族和推理后端的扩展骨架。

## 2. 当前已经定型的共识

- 训练、保存、续训、导出都遵循 HF / PEFT / TRL 标准能力。
- 样本主抽象是语言模型样本，不是任务路由样本。
- `pipeline` 只编排，不承载任务语义。
- `data` 只产出样本和 batch。
- `model` 只负责模型族适配。
- `scripts/*.py` 只做薄包装，真正命令编排在 `src/shaft/cli`。

## 3. 当前稳定主链

### 训练

- `scripts/train.py`
- `src/shaft/cli`
- `src/shaft/pipeline/sft.py`
- `src/shaft/pipeline/rlhf.py`

### 推理

- `scripts/infer.py`
- `src/shaft/infer`

### 导出

- `scripts/export.py`
- `src/shaft/export/hf.py`

## 4. 命名共识

- 框架级抽象统一 `Shaft*`
- 模型专属实现显式带模型族名
- 配置/格式强绑定对象显式反映边界，例如：
  - `TrainConfig`
  - `EvalConfig`
  - `DatasetSourceConfig`
  - `ShaftDatasetMeta`
  - `InferEngineConfig`

## 5. 关键扩展点

- 数据源：`register_data_source()`
- 模型：`register_model()`
- 模板：`register_template()`
- 算法：`register_algorithm()`
- pipeline：`register_pipeline()`
- CLI 命令：`register_command()`
- codec：`register_codec()`
- optimizer / scheduler / loss：注册表接口

## 6. 当前不要误判为稳定生产能力的部分

- PPO
- Reward Model
- 第二真实模型族
- 发布/上传工具链

## 7. 维护时优先看的文档

- `docs/architecture.md`
- `docs/module_reference.md`
- `docs/config_reference.md`
- `docs/extension_guide.md`
- `docs/testing.md`

## 8. 维护时优先检查的问题

1. 有没有把任务语义塞进训练内核。
2. 有没有在错误层落逻辑。
3. 有没有平行实现一套现有能力。
4. 有没有引入非 HF 生态依赖进主干。
5. 文档是否仍和当前命名一致。
