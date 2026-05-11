# Shaft 文档索引

本文档是 `docs/` 目录的统一入口。

## 1. 架构与边界

- [architecture.md](architecture.md)
  - 正式架构文档
  - 模块边界
  - 训练/推理主链
  - 架构图与时序图

- [webui.md](webui.md)
  - 面向工程师/科研人员的 SFT Web UI 当前实现说明
  - FastAPI/Jinja2 可视化控制台边界
  - 训练真入口与 YAML 真源约定

- [module_reference.md](module_reference.md)
  - 各模块职责
  - 关键类、函数、接口
  - 扩展入口与禁止事项

- [config_reference.md](config_reference.md)
  - `RuntimeConfig` 主要配置块
  - 常用字段和使用原则

- [online_eval_design.md](online_eval_design.md)
  - 单阶段在线 eval 设计说明
  - 多数据集、多任务、共享 codec、final score 设计

- [../projects/eval_bench/README.md](../projects/eval_bench/README.md)
  - 离线 Eval Bench 子项目说明
  - benchmark copy、prediction snapshot、run manifest、pairwise comparison、持久化目录边界

## 2. 开发与扩展

- [development_workflow.md](development_workflow.md)
  - 标准开发流程
  - 测试、文档、提交前检查

- [development_log.md](development_log.md)
  - 已暴露工程问题、指标误判和重复 bug 的开发日志
  - 根因、修复、回归测试和后续防线

- [extension_guide.md](extension_guide.md)
  - 如何新增模型族、模板、数据源、算法、推理后端、导出能力
  - 包含 feature 完成后的全局收口 review 流程入口

## 3. 运行与测试

- [scripts.md](scripts.md)
  - `scripts/` 目录的正式使用说明
  - 顶层入口脚本与 `scripts/tasks/` 的稳定接口
  - 包含 `scripts/eval_bench.py` 入口说明

- [infer.md](infer.md)
  - 推理子系统设计
  - stage / engine 边界

- [export.md](export.md)
  - HF 导出、validate、merge-peft

- [testing.md](testing.md)
  - 测试层级
  - 推荐命令
  - 变更类型与必跑清单

## 4. 待办与限制

- [todo.md](todo.md)
- [ppo_todo.md](ppo_todo.md)
