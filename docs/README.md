# Shaft 文档索引

本文是 `docs/` 的唯一导航入口。当前行为以正式参考文档和代码为准；开发日志只用于追溯，不能作为当前
能力矩阵。

## 当前真源

### 架构与接口

- [architecture.md](architecture.md)：模块边界、SFT/RL/Offline-KD/OPD 四训练域、训练/推理/eval 主链。
- [module_reference.md](module_reference.md)：模块职责、关键类与公共扩展点。
- [config_reference.md](config_reference.md)：当前 schema、配置树、合法组合与 fail-closed 规则。
- [data.md](data.md)：SFT JSONL、数据派生边界、PromptSource 静态随机 formulation 与发布检查。

### 专项设计与能力矩阵

- [training_batch_planning_design.md](training_batch_planning_design.md)：mixing、grouping、cardinality、packing、
  varlen、sharded sampler 与 committed resume。
- [online_eval_design.md](online_eval_design.md)：共享 codec、dataset policy、在线 metric 与 score 聚合。
- [infer.md](infer.md)：HF/vLLM 推理 engine 和 stage contract。
- [export.md](export.md)：HF/PEFT inspect、validate、merge 与训练状态边界。

### 开发与运行

- [testing.md](testing.md)：suite 真源、CI、CPU/GPU/manual gate 与测试责任。
- [extension_guide.md](extension_guide.md)：开发收口流程与正式扩展规范。
- [scripts.md](scripts.md)：正式 CLI 与 task 脚本使用说明。
- [Banana v5.10 数据复现](../scripts/tasks/banana_v5_10.md)：版本化准备入口、输入依赖和验收口径。
- [v5.10 Grounding](../scripts/tasks/banana_v5_10_grounding.md)：1M–2M 缩放、JPEG配对增强与测试隔离。
- [v5.10 Line points](../scripts/tasks/banana_v5_10_line_points.md)：真实干净样本+20%JPEG，V10多叉补充。
- [v5.10 真实Shape](../scripts/tasks/banana_v5_10_real_shape.md)：TXT白名单、完整几何准入和独立真实数据源。
- [真实标注质量清洗](../scripts/tasks/clean_real_raw_annotations.md)：全量解码、几何检查与可恢复隔离。

## 唯一总 TODO

- [TODO.md](TODO.md)：仓库唯一的当前待办真源，只记录尚未实现、尚未验收或必须 fail closed 的事项。

不要新增算法级、模型级或 feature 级平行 TODO。完成项立即从总 TODO 删除；稳定能力写入正式参考文档，
事故与迁移过程写入开发日志。

## 历史记录

- [development_log.md](development_log.md)：按时间记录现象、根因、修复、回归与后续防线。历史条目可能描述
  当时已经被后续版本替代的状态。

阅读历史记录时，若与当前参考文档冲突，以当前参考文档、schema 和测试矩阵为准。
