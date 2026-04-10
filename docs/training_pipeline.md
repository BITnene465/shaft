# 训练流程

## 1. 训练入口

```bash
python scripts/train.py --config <train_config.yaml>
```

支持的 CLI 覆写（无歧义）：

- `--run-id`
- `--stage-name`
- `--seed`
- `--epochs`
- `--lr`
- `--mix-strategy`（`concat|interleave_under|interleave_over`）
- `--init-from`
- `--resume-from`

## 2. 端到端流程

```text
加载 YAML
 -> 构建模型/processor/tokenizer
 -> 构建 dataset/collator/dataloader
 -> 构建 optimizer/scheduler
 -> trainer.fit()
 -> 训练/评估/存储 checkpoint
```

## 3. 混训策略与数据混合

混训目标可写为：

```text
min_theta  E_{t~pi(t)} E_{(x,y)~D_t}[L_t(x,y;theta)]
```

其中 `pi(t)` 由 `mix_strategy + mix_weight` 决定。

### 3.1 三种策略

- `concat`
  - 先按 route 拼接样本，再由采样器决定是否打乱。
  - 在当前训练默认 `shuffle=True` 时，最终顺序会被打散。
  - 不做显式欠采样/过采样。
- `interleave_under`
  - 倾向欠采样（大数据路由可能有部分样本本轮不用）。
  - 小数据路由不会被重复太多次。
- `interleave_over`
  - 倾向过采样（小数据路由会重复出现）。
  - 更容易提升小数据路由曝光，但要防止过拟合。

### 3.2 `mix_weight` 作用

- 在 `interleave_under/over` 中：影响路由配额与混合比例。
- 在 `concat` 中：主要用于路由启停（`mix_weight <= 0` 的 route 会被过滤）。

### 3.3 实践建议

1. 先用 `concat` 或 `interleave_under + 等权重` 建基线。
2. 若小任务指标偏低，提高该 route 的 `mix_weight`。
3. 出现重复采样副作用时，回退到 `interleave_under` 并降权。
4. 策略调参时固定学习率等超参，避免变量耦合。

## 4. 路由边界

- 训练核心只做通用 LM 优化。
- `task_type/domain_type` 用于 dataset/collator/evaluator 的路由：
  - prompt 选择
  - codec 编解码
  - 路由级指标聚合
  - 采样混合
- trainer 不直接解析业务字段。

## 5. 优化器分组

当前实现是两组参数：

- `lora_params`：学习率 `lora_learning_rate`，无 weight decay
- `other`：学习率 `learning_rate`，有 weight decay

说明：`embed_learning_rate` 与 `lm_head_learning_rate` 当前不是独立参数组。

## 6. Checkpoint 语义

- `checkpoint.init_from`：加载模型/adapter 权重，优化器状态从头开始。
- `checkpoint.resume_from`：恢复完整训练状态（模型、优化器、调度器、RNG、trainer state）。

## 7. 评估与最佳模型

- 多任务建议使用 `val/multi_task_score` 做 `best_metric`。
- `eval_loss` 保留为辅助监控，不应作为多任务主选模指标。
