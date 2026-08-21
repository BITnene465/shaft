# Banana v5.8 数据准备合同

本文定义 Banana v5.8 的数据准备方式。当前状态是 **preparation**：在业务 formulation 集合、每个 target
schema、builder、物化数据和 smoke 都完成前，不登记 production catalog 或正式训练 recipe。

框架公共合同见 [`docs/data.md`](../../docs/data.md)；v5.7 已发布数据见
[`banana_v5_7.md`](banana_v5_7.md)。

## 1. v5.8 相对 v5.7 的变化

v5.7 的 reconstruction 行已经是标准 materialized SFT：

```json
{"image_path":"../images/example.png","sample_id":"example__context_00","dataset_name":"shape_context_reconstruction","system_prompt":"","user_prompt":"","prompt_args":{"proposal_bbox_2d":[115,108,733,621]},"target_text":"{\"type\":\"shape\",\"parameters\":{...}}","extra":{...}}
```

v5.8 **不改变这份行格式**。当前配置冻结的 task formulation 和 eligibility pool 如下；列表顺序只用于
稳定配置与审计，不代表采样阶段：

| dataset cohort | shared pool / eligible formulations | 监督含义 |
| --- | --- | --- |
| `grounding_layout` | `grounding_layout` / `labels, boxes, objects` | 类别、位置、类别与位置组合 |
| `shape_context_reconstruction` | `shape_context_reconstruction` / `appearance, geometry, reconstruction` | 外观、几何、完整可重建属性 |
| `line_context_reconstruction` | `line_context_reconstruction` / `appearance, points, reconstruction` | 外观、点序列、完整重建 |
| `line_context_points` | `line_context_reconstruction` / `points` | line reconstruction 的真实/审计 points-only cohort |
| `image_context_reconstruction` | `image_context_reconstruction` / `image_type` | 已审核的 13 类 image type |

`line_context_points` 不是独立任务；它是 `line_context_reconstruction` 任务中只能监督 `points`
formulation 的物理数据 cohort。它不虚构 appearance/full，`image_context_reconstruction` 也不虚构
geometry/full。框架不生成幂集、不推断依赖。样本 eligibility 不同时，人工拆成不同 named dataset cohort；
若 task/prompt 语义相同则继续复用同一个 pool，只由各 dataset 的 `formulation_sources` 键选择合法子集。
外层 dataset weight 和内层 formulation weight 共同控制任务总体概率。

每个 formulation 都是一份完整的 v5.7 形态 SFT JSONL，`target_text` 已由 builder 写好。训练在线阶段只做
随机选择，不解析 `parameters`，不通过 `prompt_args` 拼 target。

## 2. Source truth 与派生层级

```text
authoritative raw / gt_standard snapshot
  -> stable split + source identity selection
  -> structured canonical sample
  -> one offline builder invocation
       -> formulation 1: standard SFT JSONL
       -> formulation 2: standard SFT JSONL (when declared)
       -> formulation 3: standard SFT JSONL (when declared)
  -> PromptSource online weighted selection
```

- raw/`gt_standard` 是唯一事实真源。
- structured 保存完整业务语义与坐标变换结果。
- 每个 formulation target 都必须从同一 structured row 确定性重建。
- builder 同时写出当前 dataset cohort 的全部 eligible formulations，避免各脚本产生 identity、crop 或
  provenance 漂移。
- formulation JSONL 是训练派生产物，不是新的业务真源。

如果 v5.8 使用新的 synthetic/raw snapshot，必须冻结新的 snapshot id、split manifest 和 selection；不得只
改目录名后沿用 v5.7 的审计结论。

## 3. 推荐目录

每个 task 使用一套共享 structured/media，以及按 formulation 分开的 SFT 文件：

```text
data/<task>/
├── selection/
│   ├── train.jsonl
│   └── val.jsonl
├── structured/
│   ├── train.jsonl
│   ├── val.jsonl
│   └── images/
├── sft/
│   └── formulations/
│       ├── <formulation-1>/
│       │   ├── train.jsonl
│       │   └── val.jsonl
│       ├── <formulation-2>/
│       │   ├── train.jsonl
│       │   └── val.jsonl
│       └── <formulation-3>/
│           ├── train.jsonl
│           └── val.jsonl
└── reports/
    ├── build_summary.json
    ├── formulation_alignment.json
    └── schema_validation.json
```

多个 formulation 可以引用同一 task-local image；不需要复制图片。

## 4. 每份 JSONL 的完整格式

每行与 v5.7 相同，必须含一个非空 `target_text`：

```json
{
  "image_path": "../../../structured/images/shape_000001__context_00.png",
  "sample_id": "shape_000001__context_00",
  "dataset_name": "shape_context_reconstruction",
  "system_prompt": "",
  "user_prompt": "",
  "prompt_args": {
    "proposal_bbox_2d": [115, 108, 733, 621]
  },
  "target_text": "{\"type\":\"shape\",\"parameters\":{...}}",
  "extra": {
    "schema_version": "banana.sft.v5.8",
    "source_sample_id": "shape_000001",
    "structured_snapshot_id": "<frozen-id>"
  }
}
```

当前框架要求 formulation stores 的 `extra` 也完全一致，因此 formulation id 放在目录、manifest 和配置中，
不要写进每行 `extra`。紧凑形式如下：

```json
{"image_path":"../../../structured/images/shape_000001__context_00.png","sample_id":"shape_000001__context_00","dataset_name":"shape_context_reconstruction","system_prompt":"","user_prompt":"","prompt_args":{"proposal_bbox_2d":[115,108,733,621]},"target_text":"{\"type\":\"shape\",\"parameters\":{...}}","extra":{"schema_version":"banana.sft.v5.8","source_sample_id":"shape_000001","structured_snapshot_id":"<frozen-id>"}}
```

同一 split 的所有 formulation 文件逐行满足：

1. 行数相同、顺序相同；
2. `sample_id` 相同且唯一；
3. 图片路径、dataset name、prompt fields、`prompt_args`、`extra` 完全相同；
4. 只有 `target_text` 不同；
5. target 外层继续遵循 v5.7 的任务结构，例如 reconstruction 仍是
   `{"type":"shape|line|image", "parameters": {...}}`，而不是另造框架格式；
6. 每个 formulation 的 `parameters` exact schema 在业务侧单独冻结和校验。

禁止：

- 一行省略 target、只保存 atomic attributes 到 `prompt_args`；
- 在行内写 `formulation_targets`、`target_a`、`target_ab` 等多答案映射；
- 在 PromptSource pool 中写 `target_template` 或 `target: materialized`；
- 训练运行时从 full target 截字段、合并 JSON 或推断组合依赖。

`prompt_args` 继续只保存 prompt renderer 所需信息，例如 proposal bbox。它不是 target 参数容器。

## 5. PromptSource pool

4 个已冻结的 task pool 位于：

- `configs/prompts/pools/grounding_layout.v5.8.yaml`
- `configs/prompts/pools/shape_context_reconstruction.v5.8.yaml`
- `configs/prompts/pools/line_context_reconstruction.v5.8.yaml`
- `configs/prompts/pools/image_context_reconstruction.v5.8.yaml`

`line_context_reconstruction` 与 `line_context_points` 两个物理 dataset cohort 都指向
`line_context_reconstruction.v5.8.yaml`。`points` formulation 及其 detailed/concise prompts 只在这份 pool
定义一次；points-only cohort 只是把 `formulation_sources` 配置为单个 `points`。

每个 formulation 恰好有两个等义 prompt variant：

- `detailed`：完整任务合同，延续并收紧 v5.7 的输出约束；
- `concise`：专用模型使用的极简任务表达，target schema 与 detailed 完全相同。

所有 variant 都显式配置同一份 unified `system_prompt`。YAML 中 `formulations` 的实际列表顺序是唯一配置
顺序真源，不再额外维护 `formulation_order`，但运行时不会按这个顺序轮换。pool 只列人工允许的组合，不使用
`all_combinations` 或自动依赖规则。

## 6. 训练配置

配置入口：

- catalog：`configs/data/banana_v5_8.yaml`
- PromptSource/source/probability 绑定：
  `configs/train/banana_sft_4b_v5_8_preparation.yaml`

preparation recipe 为每个 dataset eligible formulation id 绑定未来的标准 SFT 路径，不配置 schedule。每次
draw 都直接在该 dataset 的 eligibility 子集内按 pool 固定 `sampling_weight` 做 weighted categorical
sampling：grounding、shape 和 full-capable line 均使用 `1:1:4`；其中 `objects` 或 `reconstruction` 在对应
pool 内约占 66.7%。points-only cohort 的 eligible 子集只有共享 pool 中既有的 `points`，因此确定选择它。

line reconstruction 还包含 points-only cohort：catalog 对 full-capable/points-only 两个 dataset 使用 `6:2`，
与 full-capable pool 内部的 `1:1:4` 合成后，line reconstruction 总体约为 appearance 12.5%、points 37.5%、
reconstruction 50%。修改 catalog weight 或 formulation `sampling_weight` 即可改变这个分布。

dataset 不再同时配置顶层 `train_path`；物理训练来源由对应 PromptSource 管理。外层
`data.schedule.mixing` 仍负责不同 dataset/task 之间的 mixing，PromptSource 内层负责同一 dataset 的
formulation sampling。

## 7. 静态随机概率语义

生产选择是 weighted categorical sampling，不是固定轮换。激活多个 formulation 后，序列可以是
`appearance, appearance, reconstruction, points, ...`，不会承诺 `A,A,B,B,AB,AB`。

- 当前 v5.8 只使用 pool 的静态 `sampling_weight`；框架没有 curriculum schedule 配置。
- 权重是相对比例，不要求归一化；`1:1:4` 等价于约 `16.7%:16.7%:66.7%`。
- 每个 draw 独立随机选择，短序列中允许连续多次抽中同一 formulation。
- 相同 seed、数据 snapshot 和 logical draw 可重放同一选择；改变 pool/source/weight 后不允许冒充 exact
  resume。

## 8. 复杂子集与 eligibility

框架不自动处理业务依赖。v5.8 builder 配置需要人工声明：

- formulation id；
- 从 structured row 构建 target 的确定性函数；
- exact output schema/codec；
- 静态采样权重；
- 所有样本是否都具备该 formulation。

如果某组 line 样本只能支持 `points`，另一组可以支持 `appearance/points/reconstruction`，应拆成两个 named
dataset cohort，但二者复用同一个 `line_context_reconstruction` pool：前者只绑定 `points` source，后者绑定
三个 sources。不要复制 `points` prompt、在一行内动态禁用 formulation，或使用 `null` 监督。

## 9. Eval 策略

推荐正式 eval 为每个 formulation 准备稳定、单独命名的 materialized dataset，以便独立报告指标。若确实要
让同一 eval dataset 走 PromptSource，可使用 `apply_to: all`，并为每个 source 配置逐行对齐的 `val_path`；
此时一次 eval 只得到 deterministic draw 对应的 formulation 样本，不等同于逐 formulation 全覆盖。

## 10. 构建与发布顺序

1. 冻结 source snapshot、split manifest 和 structured schema。
2. 人工冻结合法请求属性集合、每个 formulation 的 target schema 和 prompt wording。
3. builder 从每条 structured row 一次性生成该 dataset eligibility 子集的 formulation SFT 行。
4. 检查每份文件的唯一 identity、媒体存在性、target codec/schema。
5. 检查 formulations 逐行对齐且仅 `target_text` 不同。
6. 编译 pool，并对全部行执行 `prompt_args` exact schema preflight。
7. 用 Shaft strict config loader 加载配置，运行 focused data tests 和最短 SFT smoke。
8. 生成 build/alignment/schema 报告，冻结 `media_snapshot_id`。
9. 只有以上全部通过后，登记 production catalog/recipe 和数据规模基线。

## 11. 发布清单

- [ ] source snapshot 与 split manifest 已冻结并可追溯。
- [ ] 合法 formulation 集合和每个 exact target schema 已人工确认。
- [ ] 每个 formulation 是标准 v5.7 形态 JSONL，逐行都有非空 `target_text`。
- [ ] 同一 dataset eligibility 子集的 JSONL 行数、顺序和 identity 完全一致，只有 target 不同。
- [ ] `prompt_args` 只服务 prompt renderer，不含 target 组合真值。
- [ ] pool 不含 `target_template` / `target`，source 行不含多 target mapping。
- [ ] 所有 target 通过 codec/schema，所有 prompt variants 通过参数 preflight。
- [ ] 静态随机分布、planning/runtime 一致和 resume fingerprint 已 smoke。
- [ ] 数据产物、报告、pool、配置、builder 与文档属于同一 v5.8 版本。

在清单完成前，v5.8 保持 preparation 状态。
