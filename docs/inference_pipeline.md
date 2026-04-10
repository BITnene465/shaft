# 推理流程

## 1. 单阶段推理

```text
图像 -> model.generate() -> 文本输出 -> codec 解码（宽松/严格）-> 报告
```

关键步骤：

1. 加载 checkpoint 与配置，构建推理 runner。
2. 按推理参数生成（默认贪心）。
3. 两阶段解析：
   - 宽松模式（`strict=False`）：容错解析，支持 fence 去除、截断恢复。
   - 严格模式（`strict=True`）：要求完整合法 JSON。

报告通常包含：

- 生成统计（token 数、停止原因、是否闭合 JSON）
- 宽松解析结果
- 严格解析结果

## 2. 两阶段推理

```text
图像
 -> Stage1 grounding（整图 + tiles）
 -> 聚合去重
 -> 按 bbox 构建 crop
 -> Stage2 keypoint_sequence
 -> 全局坐标回写
 -> 最终结构化输出
```

Stage1：

- 可同时跑整图和多尺度 tile proposal。
- 检测框回写到全图坐标后做 IoU 去重。

Stage2：

1. 依据 Stage1 框和 `padding_ratio` 生成 crop。
2. 对每个 crop 预测 `keypoints_2d`。
3. 将点从 crop 坐标映射回全图。

## 3. 代码入口

- 单阶段：`scripts/arrow/infer.py`
- 两阶段：`scripts/arrow/infer_two_stage.py`

详细参数与示例命令见：

- [docs/tool_scripts.md](/home/tanjingyuan/code/arrow-vlm/docs/tool_scripts.md)
