# 离线任务脚本

`scripts/tasks/` 保存可复现的数据准备、转换和审计工具。这些脚本服务具体项目任务，不属于 Shaft 训练
运行时或公共 CLI。

## 当前任务文档

- [banana_v5_7.md](banana_v5_7.md)：Banana v5.7 的数据、prompt、训练配置、构建顺序与完整性基线。
- [banana_v5_8.md](banana_v5_8.md)：Banana v5.8 的 source/structured/SFT、人工 formulation 与在线随机合同。

## 边界

- 任务脚本必须显式声明输入、输出和清理行为，不覆盖原始数据。
- 任务字段和业务标注规则在离线派生阶段收口，不进入 planner、collator 或训练循环。
- 框架能力与公共接口以 [`docs/`](../../docs/README.md) 为准；任务文档不能作为框架支持声明。
