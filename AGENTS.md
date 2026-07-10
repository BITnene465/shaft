# AGENTS.md

Shaft 仓库级开发规范。面向在本仓库内协作的工程师与编码代理。

---

## 1. 项目概述

### 1.1 项目定位

- 仓库正在重构为一个 **HF-first 的多模态训练与推理框架**。
- 当前主目标是把 **`Qwen3VL + SFT` 主链做稳**，同时保留面向：
  - `DPO / PPO`
  - 更多模型族（如 `GLM / Gemma / Rex-Omni`）
  - 更多推理与导出后端
  的标准扩展骨架。

### 1.2 开发范围

- 旧实现已归档到 `old/`。
- 新开发只在 `src/shaft` 主路径进行。
- `scripts/*.py` 只做薄包装入口，不承载核心业务逻辑。

### 1.3 文档维护原则

- 本文档允许随着**开发进度**和**用户当前需求**持续对齐和更新。
- 允许更新的内容包括：
  - 当前主线目标
  - 推荐流程
  - 测试/验收要求
  - feature 完成后的收口规则
- 但**主体框架边界不变**：
  - `config / data / model / template / algorithms / pipeline / training / infer / codec / metrics / export / plugins / webui`
  这些层级职责不能因为单次需求而漂移。
- 如果用户需求与当前文档不一致，应优先：
  1. 判断是不是文档滞后
  2. 在不破坏主体框架的前提下更新文档
  3. 再推进实现

### 1.4 目标用户

- 工程师
- 科研工作者

该仓库不是面向普通终端用户的产品代码库。优先级是：
- 架构边界清晰
- 训练/推理主链稳定
- 配置和运行时语义明确
- 易于扩展与排障

---

## 2. 模块边界

### 2.1 正式模块

- `config`
  - 配置定义、YAML 加载、catalog 展开、严格校验
- `data`
  - 数据读取、离线/在线增强、sample-level mixing、dataset、collator
- `model`
  - 多模型族适配、HF/PEFT 装配、模块分组元信息
- `template`
  - 模板元信息、chat template、监督 plan、模型族模板实现
- `algorithms`
  - `sft / dpo / ppo` 等训练算法抽象
- `pipeline`
  - 训练流水线编排、组件装配、阶段调度
- `training`
  - trainer、optimizer、scheduler、loss、checkpoint 规则
- `infer`
  - 推理 engine、推理 pipeline、请求/响应 schema
- `codec`
  - 共享输出解析层，供 infer 与在线 eval 共用
- `metrics`
  - 在线 eval metric、聚合器、final score
- `export`
  - HF/PEFT 导出、merge、校验
- `plugins`
  - 注册表、hook、interceptor
- `observability`
  - logging、context、events
- `webui`
  - 面向工程师/科研人员的可视化外壳，只做 YAML 编辑、CLI 调用、日志与状态展示

### 2.2 禁止事项

禁止在错误层承载逻辑：

- 不在训练内核写任务字段级语义解析
- 不在数据层写训练循环逻辑
- 不在算法层耦合具体数据来源路径
- 不在 `pipeline` 中重写数据 mixing / 模型族模板细节
- 不在 `webui` 中复制一套新的训练语义、数据语义或 checkpoint 语义
- 不在 `infer` 中维护一套与共享 `codec` 平行的解析逻辑

---

## 3. 构建与测试命令

### 3.1 环境准备

推荐使用 `uv` 和 Python `3.11`：

```bash
uv venv --python 3.11 --prompt shaft
source .venv/bin/activate
uv pip install -e .
```

按需安装扩展依赖：

```bash
uv pip install -e ".[dev]"
uv pip install -e ".[train]"
uv pip install -e ".[train,gpu]"
uv pip install -e ".[train,rlhf]"
uv pip install -e ".[serve]"
```

代理默认要求：

- **先尝试直接使用 `uv` 管理和调用环境**。
- 运行 Python、pytest、依赖安装时，优先使用：
  - `uv venv`
  - `uv pip`
  - `uv run`
- 如果仓库中已经存在可用的 `.venv/`，可以直接使用其中解释器；但默认优先假设 `uv` 是标准环境入口。
- 除非用户明确要求，不要自行切换到其他环境管理方式。

### 3.2 常用入口

训练：

```bash
python scripts/train.py sft --config configs/train/train_sft_4b.yaml
python scripts/train.py rlhf --config configs/train/train_dpo_4b.yaml --algorithm dpo
python scripts/train.py rlhf --config configs/train/train_ppo_4b.yaml --algorithm ppo
```

推理：

```bash
python scripts/infer.py --config configs/infer/pipeline_smoke.yaml --image /path/to/image.png
```

导出：

```bash
python scripts/export.py inspect --path /path/to/checkpoint
python scripts/export.py validate --path /path/to/export --finetune-mode full --model-type qwen3vl
python scripts/export.py merge-peft --model-type qwen3vl --adapter-path /path/to/adapter --base-model /path/to/base_model --output-dir /path/to/merged_model
```

Web UI：

```bash
python scripts/web.py
```

### 3.3 快速测试命令

全量快速回归：

```bash
pytest -q
```

只跑 CPU smoke：

```bash
pytest -q tests --suite smoke
```

只跑 distributed：

```bash
pytest -q tests --suite distributed
```

只跑 integration：

```bash
pytest -q tests --suite integration
```

只跑 GPU runtime：

```bash
pytest -q tests --suite gpu
```

必要时做编译级检查：

```bash
python3 -m compileall src/shaft tests
```

---

## 4. 代码风格指南

### 4.1 总体风格

- 优先写**边界清楚、可维护、可测试**的代码。
- 优先收敛真源，不接受多处重复维护同一份状态。
- 不接受“先补丁跑通，后面再说”的长期实现。
- 如果实现已经暴露结构问题，应先小重构再继续堆功能。

### 4.2 命名原则

- 框架级抽象统一使用 `Shaft*` 命名，例如：
  - `ShaftChatTemplate`
  - `ShaftSFTTrainer`
  - `ShaftSamplePlan`
- 模型专用实现必须带模型族/模型名，不允许使用误导为通用能力的泛名。
- 如果一段逻辑当前只服务于 `Qwen3VL`，类名和文件名都应显式包含 `Qwen3VL`。
- 通用基类可以放在 `base.py`，模型专用实现按模型文件拆分，例如：
  - `template/qwen3vl.py`
  - `template/glm4v.py`

具体规则：

- 注册项统一小写，键名语义明确：
  - `register_model("qwen3vl")`
  - `register_template("qwen3vl")`
  - `register_algorithm("sft")`
  - `register_command("sft")`
- 运行时类名保持显式：
  - `Qwen3VLLoader`
  - `Qwen3VLTemplate`
  - `SFTAlgorithm`
  - `SFTCommand`
  - `ShaftSFTPipeline`
- 与配置结构或文件格式强绑定的模块，名称应反映层级边界，例如：
  - `TrainConfig`
  - `EvalConfig`
  - `SFTDataset`
  - `DatasetSourceConfig`

### 4.3 代码组织

- 优先复用注册表和适配层，不要平行复制一套流程。
- 相同语义只允许一个真源：
  - 一个中心 plan
  - 一个中心 state
  - 一个中心 adapter signature
- 不要让 `pipeline`、`trainer`、`builder` 各自推导同一份配置结果。

### 4.4 格式与静态质量

- 仓库目标 Python 版本：`3.11`
- `ruff` 配置：
  - `line-length = 100`
- 引入新代码时，保持与现有仓库风格一致：
  - 简洁 dataclass
  - 显式类型注解
  - 不使用无意义缩写
  - 不堆叠过长函数

---

## 5. 配置、CLI 与 Web UI 规范

### 5.1 配置原则

- YAML 为主，CLI 只允许无歧义覆写，例如：
  - `run-id`
  - `seed`
  - `max-steps`
  - `epochs`
  - `lr`
  - `mix-strategy`
  - `resume`
- 新增配置字段必须进入：
  - schema
  - normalize
  - 文档
  - 至少一条消费该字段的测试

### 5.2 CLI 原则

- 顶层入口脚本：
  - `scripts/train.py`
  - `scripts/infer.py`
  - `scripts/export.py`
  - `scripts/web.py`
- 所有 CLI 解析与命令编排必须放在 `src/shaft/cli`
- `scripts/*.py` 只能做薄包装入口，不得直接堆叠业务级 `argparse`

### 5.3 Web UI 原则

- Web UI 只是可视化外壳，不是第二套训练内核
- 真实训练入口仍是 CLI
- Web UI 必须：
  - 生成 YAML
  - 调用现有 CLI
  - 展示日志与状态
- 禁止在 Web UI 中发明新的训练语义、checkpoint 语义或数据语义

---

## 6. 训练、保存与导出原则

- 训练与保存遵循 Hugging Face 生态：
  - `Trainer`
  - `TrainingArguments`
  - `PEFT`
  - `TRL`
- SFT 训练主目标统一是 next-token prediction
- DPO / PPO 是算法扩展，不改变数据与运行时内核
- 导出只接受 HF / PEFT 标准目录格式
- 不引入自定义模型目录格式

对于 adapter 训练：

- 训练态导入和导出必须保证：
  - `target_modules`
  - `modules_to_save`
  - adapter 签名
  的一致性校验
- 若部署后端只接受 full HF model，例如 vLLM，必须先 merge

---

## 7. 测试说明

### 7.1 总体原则

- 所有新增内核能力必须配套单测
- 先写/先改测试，再补实现
- 提交前至少跑通本轮变更对应的 focused 测试
- 涉及主链装配时，再补一轮更接近真实调用链的 smoke / integration

### 7.2 测试层级

- 单元测试
  - 不依赖大模型，不依赖外部服务
- Smoke 测试
  - 跑最短主链，验证组件装配
- Integration 测试
  - 真实模型 / 真实推理后端
- Manual 测试
  - 人工触发、重型、耗时

测试文件的默认执行范围由 `tests/conftest.py` suite manifest 显式管理。新增 `test_*.py` 必须恰好
登记到一个 suite；marker 只描述测试属性，不能代替 suite membership。

### 7.3 变更类型与最低测试责任

- 新增配置字段：
  - `config` 测试
  - 一条消费该字段的 smoke
- 新增数据源 / mixing / collator：
  - `tests/test_data_sources.py`
  - `tests/test_data_center.py`
  - `tests/test_mixing.py`
  - 必要时 `tests/test_collator.py`
- 新增模型族 / 模板：
  - `tests/test_model_registry.py`
  - `tests/test_template_registry.py`
- 新增算法 / pipeline：
  - `tests/test_pipeline_sft.py`
  - `tests/test_pipeline_rlhf.py`
- 新增在线 eval / codec：
  - `tests/test_codec.py`
  - `tests/test_online_eval.py`
- 新增导出能力：
  - `tests/test_export_tools.py`
  - `tests/test_export_cli.py`

更多细节见：
- `docs/testing.md`

---

## 8. 安全注意事项

### 8.1 训练与资源

- 未经明确要求，不启动长训练任务。
- 默认先做：
  - 配置校验
  - 小样本 smoke
  - 首个关键阶段 canary
- 不在不确定资源条件下直接跑重型训练或推理。

### 8.2 数据与路径

- 不在代码中硬编码用户机器专属路径。
- 不覆盖原始数据；数据准备、增强、转换应输出到新目录。
- 对 `train-only` 数据集和 `eval` 数据集语义要明确区分。

### 8.3 导入/导出与兼容性

- 对 checkpoint、adapter、merge 结果必须做显式兼容性校验。
- 对 `modules_to_save`、`target_modules`、finetune mode 不匹配的情况要尽早报错。

### 8.4 外部依赖与服务

- 对需要外部服务或真实模型的能力，优先显式降级或 skip，不要静默失败。
- 对 manual / integration 用例，skip 原因必须清晰。

---

## 9. 功能完成后的全局收口

一个 feature 基本完成后，不要直接提交。必须再做一次项目级别的收口 review，重点看：

- 是否出现重复状态源
- 是否有逻辑落在错误层
- 是否保留了临时桥接代码或双轨实现
- 是否需要先做一次小重构再提交

这套流程已经沉淀为项目 skill：

- `.codex/skills/shaft-project/shaft-feature-review/SKILL.md`

目标不是格式检查，而是回答：

- 当前实现的真源在哪里
- 是否还有冗余状态或冗余语义
- 是否已经补齐测试和文档

---

## 10. 文档联动要求

新增能力必须同步更新 `docs/`，且至少在以下文档之一落地：

- `docs/architecture.md`
- `docs/module_reference.md`
- `docs/extension_guide.md`
- `docs/config_reference.md`
- `docs/online_eval_design.md`

如果新增的是用户会直接调用的能力，还要同步：

- `README.md`
- `docs/README.md`

### 10.1 开发日志强制维护

`docs/development_log.md` 是项目开发日志真源，必须持续维护。

遇到以下任一情况，必须在同一轮变更中更新 `docs/development_log.md`：

- 重复出现或明显可能重复出现的 bug
- 训练、评估、推理指标出现异常并完成根因定位
- 发现 eval / codec / metric / data 标准不一致导致的误判
- 修复了训练/评估语义偏差、数据处理语义偏差或边界层级错误
- 用户明确要求记录经验、事故、开发日志或项目教训

开发日志条目必须至少包含：

- 现象
- 根因
- 影响范围
- 修复方式
- 回归测试
- 后续防线

如果问题涉及评估标准，必须明确区分是模型能力问题，还是 `eval / codec / metric / data` 的误判。不得只把结论留在聊天记录、临时脚本输出或一次性排障说明中。

---

## 11. 协作风格

- 先与用户同步需求，再直接动手改代码，改完代码再汇报
- 进度同步简短直接
- review 结论先给 findings，再给总结
- 如果发现结构问题，不继续堆补丁，先提出并完成必要的小重构
