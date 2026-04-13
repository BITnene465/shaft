# Codec 与训练损失

## 1. Codec 职责

`codec` 是 domain 层的监督真源，负责：

1. 坐标量化/反量化（默认 `[0,999]`）
2. 结构化数据与目标文本互转
3. 宽松/严格解码
4. 结构合法性校验

## 2. 量化公式

```python
def _quantize(value, size):
    clipped = max(0.0, min(value, size - 1))
    return round(clipped / (size - 1) * (num_bins - 1))

def _dequantize(value, size):
    return value / (num_bins - 1) * (size - 1)
```

## 3. 当前三种 codec

| Codec | 对应任务 | 输出协议 |
|---|---|---|
| `ArrowCodec` | `joint_structure` | `[{"label","bbox_2d","keypoints_2d"}]` |
| `GroundingCodec` | `grounding` | `[{"label","bbox_2d"}]` |
| `KeypointSequenceCodec` | `keypoint_sequence` | `{"keypoints_2d":[...]}` |

## 4. 训练监督链路

当前训练统一使用标准 SFT loss（LM next-token CE）。

```text
gt_struct
 -> codec.encode(...)
 -> target_text
 -> collator 组装 labels
 -> 模型标准 outputs.loss
```

约束：

- trainer 不解析业务 JSON 字段。
- trainer 不执行字段级 token 加权逻辑。

## 5. 解码鲁棒性

codec 侧已实现：

- Markdown fence 去除
- 平衡括号 JSON 提取
- 截断 JSON 恢复（尽可能恢复完整前缀）
- 宽松/严格双模式解析

## 6. 设计边界

- `codec` 负责监督文本协议。
- `collator` 只负责输入/标签拼接与 padding。
- `trainer` 只消费模型标准 `loss`。
