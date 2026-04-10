# 多任务 Qwen3-VL 训练框架规范（草案）

> 版本：v0.3（开发规范）  
> 约定：若代码与本文冲突，以本文为重构指导。

## 1. 目标

框架必须支持在一个共享 `Qwen3-VL` 权重上进行多任务结构化训练。首批任务：

- `grounding/arrow`
- `keypoint_sequence/arrow`

并可平滑扩展到更多结构化视觉任务。

## 2. 非目标

- 不做通用文本大模型训练框架。
- 不把任务语义塞进 trainer。
- 不靠 prompt 文本作为监督真源。
- 不在 trainer 里用正则反解析 JSON。

## 3. 架构硬边界

仅允许三层业务语义：

- `core`
- `tasks`
- `domains`

约束：

- `core`：通用训练/推理/评估/数据编排，不理解业务字段语义。
- `tasks`：任务语义、loss、metric、adapter。
- `domains`：codec、排序、数据准备、域推理约定。

不得新增第四层业务语义层。若新增模块，必须是跨层复用能力，且边界清晰。

## 4. 路由规范

路由定义：

```text
route = <route_id>
```

每条样本必须能显式解析 route（配置绑定或样本字段）。框架禁止通过文件名、目录名或 prompt 猜 route。

中间层要求：

- `core` 仅消费 `route`。
- `route -> adapter` 由注册层负责解析。
- `task_type/domain_type` 只允许出现在注册层或 legacy 兼容路径，不得在 trainer/evaluator/collator 主链路扩散。

## 5. 监督真源规范

- `codec` 是结构化监督真源。
- 训练目标流必须是：

```text
gt_struct -> codec.encode_with_loss_meta() -> target_text + loss_meta
          -> adapter token 权重
          -> collator.loss_weights
          -> trainer 通用 loss
```

- trainer 不得理解 `label/bbox/keypoints` 语义。

## 6. 混训规范

当前正式策略：

- 样本级混训（同一 batch 可混 route）
- 共享一个优化器与一个主干权重
- 路由采样在 `core.data.mixed_loader`

支持策略：

- `concat`
- `interleave_under`
- `interleave_over`

## 7. 评估与选模

- 每个 route 必须有主指标。
- 多任务训练必须提供全局指标（如 `val/multi_task_score`）。
- 最佳模型以配置声明指标为准。
- `eval_loss` 仅作辅助监控，不作为多任务主选模指标。

## 8. 配置模型要求

配置必须能表达：

- 模型与微调模式（full/lora/dora）
- route 及其 `route_options`
- prompt profile（单任务或 route 级）
- 数据路径与 route 绑定
- 混合策略与权重
- 评估主指标与聚合方式

## 9. 重构验收门槛

重构通过需同时满足：

1. 层级边界未破坏。
2. route 解析可追踪且缺失即报错。
3. trainer 未引入业务字段解析。
4. route 指标与全局指标都可用。
5. 单任务配置仍可退化运行。
6. 文档与测试同步更新。

## 10. 风险与应对

- 任务干扰：通过 `mix_weight` 与采样策略调节。
- 指标掩盖：同时看 route 级指标与全局指标。
- 协议漂移：prompt 改动必须联动 codec/eval/infer。
- 小任务欠拟合：提高采样权重或采用 over 策略。

## 11. 关联规范

- 训练框架边界：`docs/dev/training_framework_boundary_spec.md`
- 训练产物协议：`docs/dev/protocol_artifact_spec.md`
