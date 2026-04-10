# 训练产物协议规范（v1）

> 状态：Draft（训练框架对外契约）  
> 目标：让外部推理/编排系统仅依赖“权重 + 协议文件”即可正确消费模型。

## 1. 总则

训练框架在导出 checkpoint 时，必须同时导出：

- 模型权重（dense 或 adapter）
- `protocol.json`（结构化协议）

`protocol.json` 是唯一对外契约；外部系统不应依赖训练代码内部实现细节。

## 2. 文件位置

建议路径：

- `checkpoints/<tag>/protocol.json`

版本管理：

- `protocol_version` 必填，采用语义化版本（例如 `1.0.0`）。
- 向后兼容扩展只能新增字段，不可改写既有字段语义。

## 3. 顶层 Schema（建议）

```json
{
  "protocol_version": "1.0.0",
  "model_family": "qwen3_vl",
  "finetune_mode": "full|lora|dora",
  "tokenizer": {
    "num_bins": 1000,
    "add_eos_token": true
  },
  "routes": [
    {
      "route": "grounding/arrow",
      "task_name": "grounding",
      "domain_name": "arrow",
      "adapter": {
        "name": "ArrowGroundingAdapter",
        "codec": "GroundingCodec",
        "codec_version": "vX.Y"
      },
      "prompt": {
        "profile": "arrow.grounding.stage1.v2"
      },
      "evaluation": {
        "primary_metric": "bbox_f1_at_iou50",
        "normalizer": "identity",
        "weight": 1.0,
        "metric_min": null,
        "metric_max": null
      },
      "route_options": {}
    }
  ],
  "global_evaluation": {
    "best_metric": "val/multi_task_score",
    "monitor_mode": "max"
  },
  "compatibility": {
    "legacy_task_domain_fallback": true
  }
}
```

## 4. 字段语义约束

### 4.1 `routes[].route`

- 必须唯一。
- 建议格式：`<task>/<domain>`，但在协议层视为不透明 route id。

### 4.2 `routes[].adapter`

- 描述训练时绑定的 adapter 与 codec 信息。
- 用于外部系统做版本核对与灰度。

### 4.3 `routes[].evaluation`

- `primary_metric`：该 route 的主指标。
- `normalizer`：将指标归一化到 `[0,1]` 的规则。
- `weight`：参与全局分数聚合的权重。

### 4.4 `global_evaluation`

- 明确最佳模型判定逻辑，避免外部系统误读。

## 5. 兼容性策略

前向兼容：

- 外部系统必须忽略未知字段。

后向兼容：

- 若缺少新字段，读取方按默认值回退。

破坏性变更：

- 必须提升 `protocol_version` 主版本号。

## 6. 校验规则

训练导出阶段应进行协议校验：

1. route 唯一且非空。
2. 每个 route 均有 `primary_metric`。
3. `weight >= 0`。
4. `normalizer` 在允许集合中。
5. tokenizer 关键参数存在（至少 `num_bins`）。

## 7. 外部系统使用建议

外部推理/编排系统读取流程：

1. 加载 `protocol.json`。
2. 校验 `protocol_version` 与自身支持矩阵。
3. 按 route 选择对应 prompt 配置与 decode 逻辑。
4. 上报 route 级指标时使用协议中定义的 metric 名称。

## 8. 与训练框架边界关系

本规范只定义产物，不定义在线推理实现。  
在线推理实现（如 vLLM 服务编排、并发控制、重试策略）属于外部系统职责。

