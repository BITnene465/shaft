# AGENTS 开发约束

## 1. 项目定位

- 仓库是 `Qwen3-VL` 上的多模态结构化生成框架。
- Python 包名：`vlm_structgen`。
- 当前正式 domain：`arrow`。
- 当前正式 task：
  - `grounding`
  - `keypoint_sequence`
  - `joint_structure`

## 2. 三层边界（硬约束）

- `core`：通用训练/推理/评估/数据编排
- `tasks`：任务语义、任务 loss/metric、adapter
- `domains`：codec、排序规则、数据准备、域推理约定

禁止：

- 在 `core` 注入业务字段语义
- 在 `tasks` 注入域内数据准备细节
- 在 `domains` 注入通用训练编排逻辑

## 3. 路由与监督原则

- route 必须显式：`task_type/domain_type`。
- route 可来自：
  - 配置 `data.registry_path + train_datasets/val_datasets`（推荐）
  - 样本字段 `task_type/domain_type`
- 禁止隐式猜 route。
- codec 是监督真源，prompt 不是监督真源。
- trainer 不做字段级 JSON 反解析。

## 4. 混训原则

- 当前支持样本级混训。
- 同一 batch 可混多个 route。
- 采样策略在 `core.data`，由配置驱动。
- 最佳模型应以结构化验证指标（如 `val/multi_task_score`）为主。

## 5. 配置与 CLI 原则

- 训练主入口：`scripts/train.py`。
- 以 YAML 为主，CLI 只允许无歧义覆写（如 run-id、seed、epochs、lr、mix-strategy、init/resume）。
- 不新增会破坏 route 绑定的路径级歧义参数。

## 6. 变更纪律

若修改以下任一项，必须同步更新代码与文档：

- 数据准备
- codec
- task adapter
- evaluator
- infer/demo/deploy
- 配置文件
- 文档

## 7. 协作风格

- 工程沟通直接、务实、少废话。
- 优先先改代码再汇报。
- 未经明确要求，不擅自启动长训练任务。
