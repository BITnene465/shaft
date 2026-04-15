# Shaft 文档索引

本文档是 `docs/` 目录的统一入口。

## 1. 架构与边界

- [architecture.md](architecture.md)
  - 正式架构文档
  - 模块边界
  - 训练/推理主链
  - 架构图与时序图

- [module_reference.md](module_reference.md)
  - 各模块职责
  - 关键类、函数、接口
  - 扩展入口与禁止事项

- [config_reference.md](config_reference.md)
  - `RuntimeConfig` 主要配置块
  - 常用字段和使用原则

## 2. 开发与扩展

- [development_workflow.md](development_workflow.md)
  - 标准开发流程
  - 测试、文档、提交前检查

- [extension_guide.md](extension_guide.md)
  - 如何新增模型族、模板、数据源、算法、推理后端、导出能力

- [project_skill.md](project_skill.md)
  - 项目级记忆
  - 稳定共识与当前非目标

## 3. 运行与测试

- [infer.md](infer.md)
  - 推理子系统设计
  - stage / engine / codec 边界

- [export.md](export.md)
  - HF 导出、validate、merge-peft

- [testing.md](testing.md)
  - 测试层级
  - 推荐命令
  - 变更类型与必跑清单

## 4. 待办与限制

- [todo.md](todo.md)
- [ppo_todo.md](ppo_todo.md)
