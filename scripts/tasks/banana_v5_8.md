# Banana v5.8 数据准备合同

状态：**准备规范已确定；数据尚未物化，暂不登记 production catalog。**

本文定义 Banana v5.8 应如何准备 source truth、structured、SFT、PromptSource pool 和发布 bundle。它是
具体数据生产任务说明，不是 Shaft 公共接口文档；公共字段与运行时规则见
[`docs/data.md`](../../docs/data.md)。

## 1. v5.8 的核心变化

v5.7 的 reconstruction SFT 行保存一个 materialized `target_text`，运行时只轮换等价 prompt wording。
v5.8 若要让同一 canonical sample 学习多个人工定义的属性子集，应使用 PromptSource 的
**task formulation sampling**。A、B、AB 只是最小说明，正式集合可以是任意数量和任意命名：

```text
一条 source/structured 真值
  -> 一条 canonical SFT row（携带全部可达 formulation 的可重算参数）
  -> 人工声明的 formulations（例如 A / B / AB / AC / BCD / ...）
  -> 对应 prompt + 对应 target 原子产生
```

不要按 formulation 把同一图片预展开为多条 train row。预展开会复制 row identity、放大数据体积，并让
curriculum、resume 和 formulation 比例依赖离线展开方式。也不要只轮换 prompt 而复用同一个全量 target；
这会制造问题与监督不一致。

框架不会自动生成属性幂集，也不会推断“哪些属性必须同时出现”。pool 作者人工枚举合法组合、各自 target 和
权重，运行时只负责从当前正权重集合中在线随机选择。

属性 A/B 在正式 pool 中必须换成业务语义名，例如 `geometry`、`style`、`points` 或 `appearance`。本文的
`attribute_a/attribute_b` 只是结构占位符，不是最终业务 schema。

## 2. 真源与目录

### 2.1 真源

- 人工数据继续以 `data/raw` 的 maintained contract 为真源；compact `size + layout[]` 不在原地改造成
  normalized schema。
- synthetic reconstruction 继续从明确版本的 `gt_standard` 读取属性与几何；selection 只保存
  source/instance identity。
- structured row 保存任务完整语义，每个已声明 formulation 都必须能从它或上游真源确定性重建。
- SFT row 是 PromptSource 输入，不是第二套 target 真源。

若 v5.8 改用新的 synthetic snapshot，必须给 snapshot 独立版本/id，并重新生成 selection；不能沿用旧
selection 后只改 target 字段。

### 2.2 建议目录

```text
data/<task>/
├── selection/train.jsonl       # identity only
├── structured/train.jsonl      # complete task semantics
├── structured/val.jsonl
├── sft/train.jsonl             # canonical PromptSource rows
├── sft/val.jsonl               # policy 见第 5 节
├── images/                     # task-local derived media
├── README.md
└── build_summary.json

configs/prompts/pools/<task>.v5.8.yaml
configs/data/banana_v5_8.yaml                 # 数据物化并验收后再创建
configs/train/banana_*_v5_8.yaml              # catalog 发布后再创建
```

所有 rebuild 先写 staging/新目录，或使用 builder 已支持且目标明确的 `--clean`。不要覆盖 raw 图片，也不要
让正式 `src/shaft` 或训练配置依赖 `subTasks/`。

## 3. Canonical structured row

structured schema 可以按任务保留现有字段，但必须含有稳定 identity、媒体、完整任务真值和 provenance。
示例：

```json
{
  "sample_id": "shape_000001__context_00",
  "image_path": "../images/ab/shape_000001__context_00.png",
  "image_width": 1024,
  "image_height": 768,
  "instances": [
    {
      "label": "shape",
      "bbox": [120, 90, 720, 600],
      "parameters": {
        "attribute_a": {"value": 1},
        "attribute_b": {"value": 2}
      }
    }
  ],
  "extra": {
    "source_json": "...",
    "source_instance_index": 3,
    "split": "train",
    "coordinate_space": "qwen_0_999_context_crop",
    "builder_version": "banana_v5_8"
  }
}
```

实际 `parameters` 应继续遵守 shape/line/image 各自业务 schema；不要为了适配本示例给真源制造泛化的
`attribute_a/attribute_b`。builder 在 structured -> SFT 边界完成明确的字段投影。

## 4. Canonical SFT row

### 4.1 v5.8 多 formulation rendered-target 格式

推荐每个 canonical sample 只写一行：

```json
{"image_path":"../images/ab/shape_000001__context_00.png","sample_id":"shape_000001__context_00","dataset_name":"shape_context_reconstruction","prompt_args":{"proposal_bbox_2d":[115,108,733,621],"attribute_a":{"value":1},"attribute_b":{"value":2}},"extra":{"schema_version":"banana.sft.v5.8","source_sample_id":"shape_000001"}}
```

硬性规则：

- `image_path` 相对于当前 `sft/*.jsonl` 解析；多图改用有序 `images`。
- `sample_id` 在 dataset 内唯一且稳定；不同 formulations 不追加不同 id，因为它们不是多条 source row。
- `dataset_name` 与将来的 catalog key 一致。
- `prompt_args` 是 JSON object，必须与 pool 的 `arguments` exact match，不能缺字段或多字段。
- `system_prompt`、`user_prompt`、`messages` 和 `target_text` 省略。写成空字符串虽可规范化，但新数据应直接
  省略，减少双重真源和歧义。
- `prompt_args` 中的 atomic 属性或每个 `target_<formulation>` 必须由 structured/source truth 确定性计算；
  禁止把人工编辑过的最终答案只放在 SFT 行中。
- `extra` 只放 provenance、schema/build/media 信息；PromptSource 的运行时选择审计由框架写入
  `extra.prompt_source`。

### 4.2 只轮换 wording 的数据

如果某个 v5.8 task 没有多 formulation 监督拆分，只需要旧式等义 prompt rotation，则继续提供 materialized
`target_text`：

```json
{"image_path":"../images/0002.png","sample_id":"0002","prompt_args":{"proposal_bbox_2d":[10,20,300,400]},"target_text":"{\"type\":\"shape\",\"parameters\":{...}}"}
```

对应 pool 使用顶层 `prompts` 简写，或一个显式 `target: materialized` formulation。不要把 rendered-target
和 materialized-target formulation 混在同一个 pool。

## 5. PromptSource pool 与 curriculum

下面的 A/B/AB 只是最小模板。正式 pool 可以人工列出任意合法组合；框架不会根据 arguments 自动生成组合：

```yaml
metadata:
  id: shaft.<task>.formulations.v5.8
  version: v5.8
  task: <task>

arguments:
  proposal_bbox_2d: {type: bbox_2d_0_999}
  attribute_a: {type: json}
  attribute_b: {type: json}

formulations:
  - id: a
    sampling_weight: 1.0
    target_template: '{{ attribute_a | json }}'
    prompts:
      - id: main
        system_prompt: Return compact JSON only.
        user_prompt_template: >-
          Reconstruct attribute A for the target near {{ proposal_bbox_2d | json }}.

  - id: b
    sampling_weight: 1.0
    target_template: '{{ attribute_b | json }}'
    prompts:
      - id: main
        system_prompt: Return compact JSON only.
        user_prompt_template: >-
          Reconstruct attribute B for the target near {{ proposal_bbox_2d | json }}.

  - id: ab
    sampling_weight: 0.0
    target_template: '{"A":{{ attribute_a | json }},"B":{{ attribute_b | json }}}'
    prompts:
      - id: main
        system_prompt: Return compact JSON only.
        user_prompt_template: >-
          Reconstruct attributes A and B for the target near {{ proposal_bbox_2d | json }}.
```

训练 YAML 再定义实验 curriculum，不把 schedule 写死在数据行中：

```yaml
data:
  prompt_sources:
    <task>:
      path: ../prompts/pools/<task>.v5.8.yaml
      apply_to: train
      schedule:
        interpolation: linear
        points:
          - source_draw: 0
            weights: {a: 1.0, b: 1.0, ab: 0.0}
          - source_draw: 50000
            weights: {a: 0.5, b: 0.5, ab: 1.0}
          - source_draw: 150000
            weights: {a: 0.1, b: 0.1, ab: 1.0}
```

这些数字只是结构示例。正式边界要根据该 dataset 在 weighted stream 中预计获得的 dataset-local draw 数和
实验目标确定；它们不是全局 optimizer step。

在线选择是 weighted categorical random，而不是 `A,A,B,B,AB,AB` 或其它固定轮换。相同 logical draw 在
resume/多 worker/多 rank 下会重放同一选择，但短序列可以连续抽到同一个 formulation，也不保证严格满足
比例。若不需要 curriculum，直接用 pool 中每个 formulation 的 `sampling_weight` 即可。

### 5.1 复杂依赖的推荐表达

对于简单组合，SFT row 保存 atomic attributes，`target_template` 手工组装。对于深层嵌套或存在人工业务依赖
的组合，builder 可以从同一 structured truth 为每个**已声明**组合生成一个完整 JSON 参数：

```yaml
arguments:
  target_geometry: {type: json}
  target_style: {type: json}
  target_geometry_style: {type: json}
  target_geometry_text_links: {type: json}

formulations:
  - id: geometry
    sampling_weight: 1.0
    target_template: '{{ target_geometry | json }}'
    prompts: [{id: main, user_prompt: Reconstruct geometry only.}]
  - id: style
    sampling_weight: 1.0
    target_template: '{{ target_style | json }}'
    prompts: [{id: main, user_prompt: Reconstruct style only.}]
  - id: geometry_style
    sampling_weight: 2.0
    target_template: '{{ target_geometry_style | json }}'
    prompts: [{id: main, user_prompt: Reconstruct geometry and style.}]
  - id: geometry_text_links
    sampling_weight: 1.0
    target_template: '{{ target_geometry_text_links | json }}'
    prompts: [{id: main, user_prompt: Reconstruct geometry, text, and links.}]
```

这些 `target_*` 是可重建的 SFT 派生参数，不是 selection/raw 的第二真源。一个 pool 的所有 row 必须支持它的
全部可达 formulations；若不同样本的可用组合不同，按 eligibility class 拆为多个 named datasets/pools，再
在线 mixing。当前不在 row 中加入动态 allowlist，也不让训练运行时猜测依赖。

## 6. Eval policy

二选一并在 catalog/task README 中写清：

1. `apply_to: train`：训练 JSONL 使用 `prompt_args`；val/test 准备完全 materialized 的固定 prompt + target，
   且不携带 `prompt_args`。若要分别报告各个 formulation，每个 formulation 生成独立、稳定 id 的 eval row。
2. `apply_to: all`：train/val 都使用 canonical `prompt_args`，eval 固定在 `source_draw_id=0` 的确定性投影。
   该方式适合只需要一个固定总体 view 的 eval，不适合要求三项独立指标的场景。

任何 validation/test 都不得使用 train-only crop/view augmentation。空 validation 仍然合法，但必须同时设置
`use_for_eval: false` 和 `eval.enabled: false`，不能把空文件误当成待补数据。

## 7. 发布顺序

1. 冻结 v5.8 source snapshot、split manifest 和业务 output schema。
2. 从 source truth 重建 identity-only selection、structured、task-local media 和 canonical SFT。
3. 编译每个 v5.8 pool，并对所有 SFT 行执行 exact `prompt_args` preflight。
4. 对每个人工声明的 formulation 重算 target，使用共享 codec 或 task validator 检查可解析性和 exact schema。
5. 核验 train/val/test 互斥、structured/SFT/media 一一对应、唯一 id/path、图片尺寸和坐标空间。
6. 记录 builder 版本、seed、输入 snapshot、行数、排除项、augmentation 分布和 schema version。
7. 数据真正物化后才创建 `configs/data/banana_v5_8.yaml`，设置新的 `media_snapshot_id`、权重和 eval flags。
8. 创建 v5.8 训练 YAML，确保每个启用 dataset 恰好映射到一个 pool，并用 strict loader 加载全部 recipe。
9. 运行 focused config/data tests、抽样分布检查和最短 CPU/GPU SFT smoke 后再声明 bundle ready。

## 8. 交付前最小检查表

- [ ] 人工允许的 formulation 集合、业务字段名、依赖和各自 exact output schema 已冻结。
- [ ] 每个 canonical sample 的全部可达 formulation 都能从同一 structured/source truth 重算。
- [ ] train row 没有 materialized prompt/messages/target 与 PromptSource 参数并存。
- [ ] pool `arguments` 与每行 `prompt_args` exact match，所有可达 variant/formulation 都通过 preflight。
- [ ] 所有 configured target 都能被对应 codec/validator 解析，键、枚举、坐标系和空值语义一致。
- [ ] split 两两互斥；validation/test 干净且无 train-only view。
- [ ] structured、SFT、media 数量与 identity 一一对应，没有未引用媒体或缺图。
- [ ] catalog、pool、训练 YAML 的 dataset 名、版本、权重、eval flag 和 snapshot id 一致。
- [ ] task README/build summary 记录实际数字；本文中的占位符和示例数字没有被当成发布基线。

在以上检查完成前，v5.8 的状态应保持“preparation”，不能因为 pool 或 builder 文件存在就登记为 production
source。
