# AGENTS 开发约束（重构期）

## 1. 项目定位

- 仓库正在重构为“HF-first 的多模态训练框架”。
- 旧实现已迁移到 `old/`，新开发只在 `src/shaft` 主路径进行。

## 2. 模块边界

- `config`：配置定义与严格校验。
- `data`：数据读取、离线/在线增强、样本级 mixing、collator。
- `model`：多模型适配（Qwen3VL/GLM/Gemma 等）。
- `algorithms`：`sft/dpo/ppo` 等训练算法抽象。
- `pipeline`：训练流水线编排（组件装配与阶段调度）。
- `plugins`：注册表、hook、拦截器。

禁止：

- 在训练内核写任务字段级语义解析。
- 在数据层写训练循环逻辑。
- 在算法层耦合具体数据来源路径。

## 3. 训练与保存原则

- 训练与保存遵循 Hugging Face 生态（Trainer/TrainingArguments）。
- 训练主目标统一是 next-token prediction（SFT 阶段）。
- DPO/PPO 可作为算法扩展接入，不改变数据与运行时内核。

## 4. 配置与 CLI 原则

- 入口脚本：`scripts/train.py`（顶层编排，子命令如 `sft`/`rlhf`）；任务命令定义在 `src/shaft/cli`。
- YAML 为主，CLI 只允许无歧义覆写（run-id/seed/epochs/lr/mix-strategy/resume）。

## 5. 测试驱动

- 所有新增内核能力必须配套单测。
- 先写/先改测试，再补实现。
- 提交前至少跑通当前重构路径下的测试集。

## 6. 协作风格

- 先改代码再汇报。
- 进度同步简短直接。
- 未经明确要求，不启动长训练任务。
