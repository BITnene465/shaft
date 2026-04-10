# 训练框架边界规范（v1）

> 状态：Draft（用于指导后续重构）  
> 适用范围：`vlm_structgen` 仓库（训练/评估主线）

## 1. 目标

将仓库收敛为“纯训练框架”：

- 负责：数据读取、混训采样、训练、验证、checkpoint 导出。
- 不负责：在线推理、服务编排、部署启动、请求路由、SLA。

## 2. 非目标

- 不在本仓库内继续扩展在线推理流程。
- 不在 trainer/evaluator/collator 中注入业务字段语义。
- 不把 prompt 当监督真源。

## 3. 分层定义

### 3.1 `core`

职责：

- 配置系统、注册系统、数据管线、训练循环、评估聚合、checkpoint 导出。
- 仅基于 `route` 做任务分派。

禁止：

- 理解业务字段（如 `label/bbox/keypoints`）语义。
- 依赖具体 task/domain 实现模块。

### 3.2 `tasks`

职责：

- 任务 adapter、任务 loss 权重规则、任务指标汇总规则。

禁止：

- 写通用训练编排逻辑。
- 写 domain 数据准备流程。

### 3.3 `domains`

职责：

- codec、域排序约束、域数据准备、域结构化协议解释。

禁止：

- 写 trainer/evaluator 主循环逻辑。

## 4. 中间层与路由

采用注册模式中间层（属于 `core` 能力，不是新业务层）：

- `route` 是唯一主键（例如 `grounding/arrow`）。
- `core` 通过 `route -> adapter` 解析，不在主链路传播 `task_type/domain_type`。
- `task_type/domain_type` 仅允许用于 legacy 兼容与注册解析兜底。

## 5. 依赖方向（硬约束）

允许：

- `tasks -> domains`
- `core -> core公共接口`

禁止：

- `core -> tasks.*具体模块`（如 `tasks.grounding.adapter`）
- `core -> domains.*具体模块`
- `domains -> core.train|core.eval|core.data`

说明：

- 入口脚本可调用 bootstrap 完成注册，但这不改变 core 的无实现依赖约束。

## 6. 训练主链路契约

统一链路：

```text
registry/config
 -> dataset(route)
 -> collator(route)
 -> trainer(next-token loss)
 -> evaluator(route metrics + global score)
 -> checkpoint + protocol artifact
```

关键约束：

- trainer 只做 token-level 优化。
- evaluator 只做通用聚合与 route 指标汇总，不解析业务 JSON 字段。
- `codec` 提供 `target_text + loss_meta`，是监督真源。

## 7. 配置契约

必须支持：

- `route` 级路由配置
- `route_options`（mix/eval/task 参数）
- `route_prompts`（仅输入提示，不是监督真源）
- 数据集 registry（dataset id -> route/path）

约束：

- 新配置禁止新增 `task_type/domain_type` 字段作为主表达。
- legacy 字段仅兼容读取，不允许继续写入模板。

## 8. 协议产物契约（对外）

训练框架必须导出可供外部系统消费的协议文件（见 `protocol_artifact_spec.md`）：

- route 列表
- route 指标主键与归一化规则
- prompt profile 引用
- codec 标识/版本
- tokenizer 关键参数（如 `num_bins`）

外部推理系统只依赖权重 + 协议文件，不依赖本仓库训练实现。

## 9. 验收清单

满足以下条件才允许合并重构：

1. `core` 无对 `tasks/domains` 具体实现 import。
2. 主链路仅使用 `route` 做分派。
3. 单任务与混训配置都能正常训练和评估。
4. route 级指标与全局主指标可用。
5. 协议产物可导出并通过 schema 校验。
6. 文档与测试同步更新。

## 10. 迁移优先级

P0：

- 固化边界检查与路由注册。
- 协议产物导出（checkpoint 附带）。

P1：

- TaskSpec 注册中心化（route 元信息统一声明）。
- 配置 schema 严格校验（含 route_options）。

P2：

- callback 插件化（logging/best-save/early-stop）。

