# 架构总览

## 1. 项目定位

`vlm_structgen` 是基于 `Qwen3-VL` 的多模态结构化生成框架。当前在 `arrow` 域实现三类正式任务：

| 任务 | 阶段 | 输出 |
|---|---|---|
| `joint_structure` | 单阶段 | `label + bbox_2d + keypoints_2d` |
| `grounding` | 两阶段 Stage1 | `label + bbox_2d` |
| `keypoint_sequence` | 两阶段 Stage2 | `keypoints_2d` |

## 2. 三层架构

```text
core      -> 通用训练/推理/评估/数据管线
tasks     -> 任务语义、任务 loss、任务指标、任务 adapter
domains   -> 域语义、codec、排序规则、数据准备、域推理约定
```

核心边界：

- `core` 不理解业务字段语义（如 `label/bbox/keypoints`）。
- `tasks` 不承载域内数据准备细节。
- `domains` 不承载通用训练编排。

## 3. 路由机制

路由统一为：

```text
route = <route_id>
```

示例：

- `grounding/arrow`
- `keypoint_sequence/arrow`
- `joint_structure/arrow`

路由来源：

- 配置绑定（推荐）：`data.registry_path + train_datasets/val_datasets`
- 样本字段（可选）：JSONL 的 `route`
- 兼容兜底：JSONL 的 `task_type + domain_type`（legacy）

框架不会根据文件名或 prompt 猜 route。

中间层注册模式：

- `core.registry` 提供 `route -> adapter binding` 的注册与解析能力。
- `core` 训练/评估/推理链路只消费 `route`，不在链路内传播 `task_type/domain_type`。
- 对未显式注册的 route，当前默认按 `task_type/domain_type` 规则回退解析（用于兼容旧配置）。

## 4. 数据流

```text
原始标注 + 图像
  -> scripts/arrow/prepare*.py
  -> 标准 JSONL
  -> SFTDataset（按 route 取 adapter/codec）
  -> SFTCollator（组装模型输入与 labels）
  -> 模型训练/评估
```

关键点：

- `codec` 负责结构化 GT 与文本监督转换。
- `loss_meta` 由 codec 生成，供 weighted token loss 使用。

## 5. 训练与混训

- 训练目标统一是 LM `next-token prediction`（teacher forcing）。
- 多任务混训是样本级路由（同一 batch 可混 route）。
- 采样策略在 `core.data.mixed_loader`：
  - `concat`
  - `interleave_under`
  - `interleave_over`

说明：采样策略属于通用训练编排，放在 `core.data` 是当前正式设计，不属于 domain 逻辑。

## 6. 推理链路

### 单阶段

```text
图像 -> generate -> codec 解码（宽松/严格）-> 结构化输出
```

### 两阶段

```text
图像
 -> Stage1 grounding（整图 + tiles）
 -> proposal 聚合与去重
 -> 逐目标裁剪
 -> Stage2 keypoint_sequence
 -> 坐标回写与结果合并
```

## 7. 代码目录

```text
src/vlm_structgen/
├── core/
│   ├── config.py
│   ├── registry.py
│   ├── routing.py
│   ├── data/
│   ├── train/
│   ├── eval/
│   └── utils/
├── runtime/
│   └── infer/
├── tasks/
│   ├── grounding/
│   ├── keypoint_sequence/
│   └── joint_structure/
└── domains/arrow/
    ├── codecs/
    ├── data/
    ├── infer/
    ├── ordering.py
    └── task_support.py
```

## 8. 配置主干

`ExperimentRuntimeConfig` 主要字段：

- `experiment`
- `model`
- `tokenizer`
- `task`
- `prompt`
- `data`
- `finetune`
- `lora`
- `train`
- `eval`
- `logging`
- `checkpoint`
