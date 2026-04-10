# Codec 与加权损失机制

## 1. Codec 职责

`codec` 是 domain 层的监督真源，负责：

1. 坐标量化/反量化（默认 `[0,999]`）
2. 结构化数据与目标文本互转
3. 宽松/严格解码
4. 输出 `loss_meta`（字段字符跨度）
5. 结构合法性校验

## 2. 量化公式

```python
def _quantize(value, size):
    clipped = max(0.0, min(value, size - 1))
    return round(clipped / (size - 1) * (num_bins - 1))

def _dequantize(value, size):
    return value / (num_bins - 1) * (size - 1)
```

## 3. 当前三种 codec

| Codec | 对应任务 | 输出协议 | loss_meta 字段 |
|---|---|---|---|
| `ArrowCodec` | `joint_structure` | `[{"label","bbox_2d","keypoints_2d"}]` | 无（默认损失） |
| `GroundingCodec` | `grounding` | `[{"label","bbox_2d"}]` | `label`、`bbox_2d` |
| `KeypointSequenceCodec` | `keypoint_sequence` | `{"keypoints_2d":[...]}` | `coordinates` |

## 4. `field_char_spans` 与 weighted CE

`encode_with_loss_meta()` 会返回：

- `target_text`
- `loss_meta.field_char_spans`

`field_char_spans` 记录语义字段在目标文本中的字符区间，adapter 基于区间映射 token 权重，再由 trainer 执行通用 weighted CE。

流程：

```text
gt_struct
 -> codec.encode_with_loss_meta()
 -> target_text + loss_meta
 -> adapter.build_target_token_weights()
 -> collator.loss_weights
 -> weighted token CE
```

## 5. 默认权重（当前配置常用值）

- `grounding`
  - `label_token_loss_weight`
  - `bbox_token_loss_weight`
- `keypoint_sequence`
  - `coordinate_token_loss_weight`

约束：

- 权重必须 `>= 1.0`
- `< 1.0` 视为配置错误

## 6. 解码鲁棒性

codec 侧已实现：

- Markdown fence 去除
- 平衡括号 JSON 提取
- 截断 JSON 恢复（尽可能恢复完整前缀）
- 宽松/严格双模式解析

## 7. 设计边界

- trainer 不解析业务 JSON 字段。
- 不在 trainer 里做正则反解析坐标/标签。
- task/domain 语义通过 adapter + codec 传递。
