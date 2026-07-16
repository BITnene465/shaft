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
  - `RuntimeConfig` 顶层职责树与 YAML 加载流程
  - schedule/mixing、transforms、batching、optimizer 的数据执行树与职责矩阵
  - batching 四轴合法组合矩阵、选择决策树、跨层字段关系和常用字段原则

- [online_eval_design.md](online_eval_design.md)
  - 单阶段在线 eval 设计说明
  - 多数据集、多任务、共享 codec、final score 设计

- [training_batch_planning_design.md](training_batch_planning_design.md)
  - sample-level mixing、bounded cost-aware batching、sequence packing 与 DDP/CP 的顶层边界
  - horizon-independent schedule、bounded buffer、Accelerate 分片、committed-state resume 与 DDP
    static-graph bitwise 恢复边界

- [../projects/eval_bench/README.md](../projects/eval_bench/README.md)
  - 离线 Eval Bench 子项目说明
  - benchmark copy、prediction snapshot、run manifest、pairwise comparison、持久化目录边界

- [eval_bench_frontend_reference_study.md](eval_bench_frontend_reference_study.md)
  - VSCode、FiftyOne、CVAT、Codex 客户端源码参考记录
  - Eval Bench 前端模块边界、viewer、快捷键和状态模型设计启发

- [eval_bench_architecture.md](eval_bench_architecture.md)
  - Eval Bench 七层架构、中间层真源和扩展测试责任

- [eval_bench_ui_icon_design.md](eval_bench_ui_icon_design.md)
  - Eval Bench UI 压缩原则
  - image_gen PNG 图标库、资产路径和前端接入边界

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

- [../docker/inference/README.md](../docker/inference/README.md)
  - 业务推理镜像构建与 vLLM 启动
  - `shaft-contract-smoke` 推理契约验收

- [export.md](export.md)
  - HF 导出、validate、merge-peft

- [testing.md](testing.md)
  - unit / component / contract / smoke / integration / manual 分层
  - 默认快速回归与显式 smoke 命令
  - 测试 support 层、真实文件依赖规则和变更必跑清单

## 4. 待办与限制

- [todo.md](todo.md)
- [ppo_todo.md](ppo_todo.md)
