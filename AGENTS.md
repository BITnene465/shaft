# AGENTS 开发约束（重构期）

## 1. 项目定位

- 仓库正在重构为“HF-first 的多模态训练框架”。
- 旧实现已迁移到 `old/`，新开发只在 `src/shaft` 主路径进行。

## 2. 模块边界

- `config`：配置定义与严格校验。
- `data`：数据读取、离线/在线增强、样本级 mixing、collator。
- `model`：多模型适配（Qwen3VL/GLM/Gemma 等）。
- `template`：模板元信息与模型族模板实现。
- `algorithms`：`sft/dpo/ppo` 等训练算法抽象。
- `pipeline`：训练流水线编排（组件装配与阶段调度）。
- `plugins`：注册表、hook、拦截器。
- `webui`：面向工程师/科研人员的可视化外壳，只做 YAML 编辑、CLI 调用、日志与状态展示，不承载训练内核逻辑。

禁止：

- 在训练内核写任务字段级语义解析。
- 在数据层写训练循环逻辑。
- 在算法层耦合具体数据来源路径。
- 在 Web UI 中复制一套新的训练语义、数据语义或 checkpoint 语义。

## 3. 命名原则

- 框架级抽象统一使用 `Shaft*` 命名，例如 `ShaftChatTemplate`、`ShaftSFTTrainer`。
- 模型专用实现必须带模型族/模型名，不允许使用会误导为通用能力的泛名。
- 如果一段逻辑当前只服务于 `Qwen3VL`，类名和文件名都应显式包含 `Qwen3VL`，不要命名成通用名字。
- 通用基类可以放在 `base.py`，模型专用实现按模型文件拆分，例如 `template/qwen3vl.py`、`template/glm4v.py`。
- 具体规则：
  - 注册项统一小写，键名语义明确：`register_model("qwen3vl")`、`register_template("qwen3vl")`、`register_algorithm("sft")`、`register_command("sft")`。
  - 运行时类名保持显式：`Qwen3VLLoader`、`Qwen3VLTemplate`、`SFTAlgorithm`、`SFTCommand`、`ShaftSFTPipeline`。
  - 可复用通用实现必须明确是“框架级能力”，例如 `ShaftChatTemplate`、`ShaftSFTTrainer`、`ShaftProgressCallback`。
  - 与配置结构或文件格式强绑定的模块，名称应反映层级边界，如 `TrainConfig`、`EvalConfig`、`SFTDataset`、`DatasetSourceConfig`。
- 禁止在通用类名上承载单模型语义（例如 `Template` 内部仅处理 `qwen3vl` 格式），单模型逻辑必须落在对应模型文件。
- 新增能力必须同步更新 `docs/`，且至少在以下文档之一落地：`docs/architecture.md`、`docs/module_reference.md`、`docs/extension_guide.md`、`docs/project_skill.md`。

## 4. 训练与保存原则

- 训练与保存遵循 Hugging Face 生态（Trainer/TrainingArguments）。
- 训练主目标统一是 next-token prediction（SFT 阶段）。
- DPO/PPO 可作为算法扩展接入，不改变数据与运行时内核。

## 5. 配置与 CLI 原则

- 入口脚本：`scripts/train.py`（顶层编排，子命令如 `sft`/`rlhf`）；任务命令定义在 `src/shaft/cli`。
- 所有 CLI 解析与命令编排必须放在 `src/shaft/cli`；`scripts/*.py` 只能做薄包装入口，不得在脚本文件里直接堆叠业务级 `argparse` 逻辑。
- 新增 CLI 能力时，优先复用现有 `src/shaft/cli` 风格与公共约定，避免在 feature 子模块下再生一套平行 CLI。
- YAML 为主，CLI 只允许无歧义覆写（run-id/seed/epochs/lr/mix-strategy/resume）。
- Web UI 只能依附现有 CLI：先生成 YAML，再调用 `scripts/train.py sft`；不能直接把训练内核做成第二套入口。

## 6. 测试驱动

- 所有新增内核能力必须配套单测。
- 先写/先改测试，再补实现。
- 提交前至少跑通当前重构路径下的测试集。

## 7. 协作风格

- 先改代码再汇报。
- 进度同步简短直接。
- 未经明确要求，不启动长训练任务。
