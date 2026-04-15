# Shaft 架构总览与边界

本文档描述 `src/shaft` 当前可运行架构，重点用于指导后续 agent 扩展和重构，避免跨层越界。

## 0. 总原则：当前仅面向 Hugging Face 生态

- Shaft 当前是 **HF-first** 框架，只面向 Hugging Face 生态组织训练、保存、续训与推理接入。
- 当前默认依赖与边界：
  - 模型加载基于 `transformers`
  - 参数高效微调基于 `peft`
  - 强化学习训练基于 `trl`
  - 部署侧仅接入 HF 本地推理与 vLLM 的 OpenAI 兼容接口
- 当前**不**以兼容 ModelScope、自定义 checkpoint 格式、非 HF Trainer 内核、多套模型仓库协议为目标。
- 因此，新增能力时应优先复用 HF 标准对象与目录约定，而不是引入平行生态抽象层。

## 1. 顶层分层

### 1.1 `config`
- 位置：`src/shaft/config`
- 作用：配置 schema、YAML 加载、归一化校验。
- 关键点：
  - `RuntimeConfig` 是唯一入口对象。
  - `algorithm.name` 当前支持：`sft` / `dpo` / `ppo`。
  - `rlhf.dpo` / `rlhf.ppo` 为结构化配置，不再是无类型字典。

### 1.2 `data`
- 位置：`src/shaft/data`
- 作用：数据源注册、离线/在线 transform、样本级 mixing、dataset/collator。
- 数据源：
  - `jsonl_sft` -> `SFTRecord`
  - `jsonl_dpo` -> `DPORecord`
  - `jsonl_ppo` -> `PPORecord`
- 关键点：
  - `ShaftDataCenter` 是数据子系统的正式入口，统一负责：多数据源加载、offline transform、sample-level mixing、dataset-aware online transform 编排。
  - `config` 层支持 `data.registry_path + data.dataset_refs`，可从外部 registry 文件解析命名数据集，再合并到 `data.datasets`。
  - registry 文件中的相对路径按 registry 文件目录解析；主训练 YAML 中内联 `data.datasets` 的相对路径按训练 YAML 所在目录解析。
  - JSONL 解析使用聚合报错，会汇总坏样本行号与错误原因。
  - `ShaftDataCenter` 输出的是“标准 records / dataset pair”，不会感知 loss、optimizer、训练循环。
  - 训练循环、loss、优化器逻辑禁止进入数据层。

### 1.3 `model`
- 位置：`src/shaft/model`
- 作用：模型族适配、processor/tokenizer/template 解析、finetune 策略注入。
- 关键点：
  - 统一入口：`build_model_tokenizer_processor`。
  - 运行时统一适配对象：`ShaftModelAdapter`，负责收口模板选择、processor policy、peft policy、requires、additional_saved_files。
  - `processor policy` / `peft policy` 通过模型层 registry 注册和复用，而不是在各模型文件里直接散落硬编码实例。
  - `init_from_checkpoint` 支持 full 权重和 PEFT adapter 初始化。
  - 模型族特化逻辑必须放在模型专属文件（例如 `qwen3vl.py`）。

### 1.4 `template`
- 位置：`src/shaft/template`
- 作用：模板注册和 prompt 渲染抽象。
- 关键点：
  - 模板层只负责消息拼装与 decode，不负责任务语义解析。

### 1.5 `algorithms`
- 位置：`src/shaft/algorithms`
- 作用：算法注册 + trainer 构建。
- 当前算法：
  - `SFTAlgorithm` -> `ShaftSFTTrainer`
  - `DPOAlgorithm` -> `ShaftDPOTrainer`（TRL DPOTrainer 内核）
  - `PPOAlgorithm` -> `ShaftPPOTrainer`（TRL PPOTrainer 内核）
- 关键点：
  - 算法层负责“如何训练”，不负责“如何读取数据文件”。
  - DPO 当前为基础工业实现，不是 TRL 全特性对齐版本（例如多损失家族/复杂偏好采样策略尚未全部接入）。

### 1.6 `pipeline`
- 位置：`src/shaft/pipeline`
- 作用：按算法编排端到端训练流程。
- 流水线：
  - `shaft_train`：仅 `sft`
  - `shaft_rlhf`：`dpo/ppo`
- 关键点：
  - SFT 与 RLHF 已分流，防止同一 pipeline 里塞 if-else 污染职责。
  - pipeline 只负责装配组件；多数据源读取、增强与 mixing 统一委托给 `ShaftDataCenter`，不得在 pipeline 内重复实现。

### 1.7 `training`
- 位置：`src/shaft/training`
- 作用：Trainer 扩展、loss/optimizer/scheduler 注册、checkpoint 规则。
- 关键点：
  - `ShaftSFTTrainer`
  - `ShaftDPOTrainer`（TRL 包装类）
  - `ShaftPPOTrainer`（TRL 包装类）
  - `LOSS_REGISTRY` / `OPTIMIZER_REGISTRY` / `SCHEDULER_REGISTRY`
  - checkpoint 校验：`ensure_hf_export_layout`、`validate_resume_checkpoint`

### 1.8 `plugins`
- 位置：`src/shaft/plugins`
- 作用：hook/interceptor/proxy 横切机制。
- 关键点：
  - 插件只做横切增强（日志、监控、拦截），不替代主业务流程。

### 1.9 `infer`
- 位置：`src/shaft/infer`
- 作用：推理引擎与多阶段推理 pipeline。
- 关键点：
  - 与训练内核解耦，不依赖训练循环内部实现。
  - `InferEngine` 使用 adapter 抽象（当前已实现 `hf_local` 与 `vllm_openai`）。
  - pipeline 支持 stage 级 codec/retry/fail_fast，并输出 `__trace__` 便于后端排障。

### 1.10 `cli`
- 位置：`src/shaft/cli`
- 作用：训练命令入口和参数覆写。
- 命令：
  - `scripts/train.py sft --config ...`
  - `scripts/train.py rlhf --config ... --algorithm dpo|ppo`

## 2. 关键边界（必须遵守）

1. 数据层不写 loss/优化器/梯度更新逻辑。  
2. 算法层不解析 JSONL 路径和文件细节。  
3. pipeline 层不实现模型族特化细节，也不实现多数据源/mixing/增强编排。  
4. 模型特化能力（target modules/template/processor policy/peft policy）仅在 `model`/`template`。  
5. checkpoint 格式必须遵循 HF/PEFT 生态，不引入自定义保存格式。  

## 3. 训练状态与续训规则

- `init_from_checkpoint`：用于初始化权重（可从 full 或 adapter）。
- `resume_from_checkpoint`：用于恢复 trainer 状态，要求存在 `trainer_state.json`。
- 模式约束：
  - `full` 续训要求 full checkpoint。
  - `lora/dora/qlora` 续训要求 adapter checkpoint。
- 导出约束：
  - `full` 导出必须是 HF full 目录。
  - `lora/dora/qlora` 导出必须是 PEFT adapter 目录。

## 4. 当前数据格式契约

### 4.0 命名数据集注册中心
- 训练 YAML 可选：
  - `data.registry_path`: 指向一个 YAML registry 文件
  - `data.dataset_refs`: 指定要启用的命名数据集列表
- registry 文件支持两种形式：
  - `datasets: {name: {...}}`
  - `datasets: [{name: ..., ...}, ...]`
- 解析顺序：
  - 先按 `data.dataset_refs` 解析 registry 命名数据集
  - 再拼接主配置中的 `data.datasets`
- 若 registry 数据集与 inline 数据集重名，会直接 fail-fast。

### 4.1 SFT (`jsonl_sft`)
- 必填：`image_path|image|images` + `target_text`（或 `messages` 末尾 assistant）。

### 4.2 DPO (`jsonl_dpo`)
- 必填：`image_path|image|images` + `chosen_text|chosen` + `rejected_text|rejected`。

### 4.3 PPO (`jsonl_ppo`)
- 必填：`image_path|image|images` + `messages` 或 `user_prompt|prompt`。
- 说明：PPO 数据只提供 query/prompt，不再在样本内携带离线 `response/reward`。
- 风险保护：
  - `rlhf.ppo.allow_untrained_reward_model=false`（默认）时会拒绝启动，防止误用随机奖励头。
  - 多模态模型默认拒绝走当前 text-only PPO 路径；仅在显式设置 `allow_text_only_multimodal_ppo=true` 时允许 smoke/debug。
  - 当前 PPO 仅允许 `lora/dora/qlora`；`full` 模式会被拒绝。
  - 默认 `value_model_mode=shared_backbone`、`reward_model_mode=adapter_disabled_policy`，用于降低显存占用。

## 5. 测试分层

- 默认回归：`pytest -q`
- 集成：`pytest -q -m integration`
- 手工：`pytest -q -m manual`

新增核心能力时，必须至少补：
1. 单元测试（逻辑边界）  
2. pipeline smoke（端到端最短链路）  
3. 文档同步（架构 + 扩展指南）  

## 6. 能力成熟度说明

- SFT：生产可用。  
- DPO：已切换为 TRL 内核，可训练可回归。  
- PPO：**暂停（非生产）**。当前仅保留 smoke 级能力；未完成项和恢复条件见 [docs/ppo_todo.md](ppo_todo.md)。  
