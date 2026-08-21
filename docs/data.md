# Shaft 数据与 PromptSource

本文是 Shaft 数据主链、SFT JSONL 与 PromptSource 的公共合同真源。具体 Banana 数据版本、业务 schema、
构建命令和规模基线属于 `scripts/tasks/`；配置字段的逐项说明见
[config_reference.md](config_reference.md)。

## 1. 数据主链与模块边界

```text
raw / authoritative source truth
  -> task-local structured data
  -> offline SFT builders
  -> standard SFT JSONL
  -> ShaftDataCenter
       -> ordinary source loading / dataset mixing
       -> PromptSource.prepare_records（仅已配置的 dataset）
  -> online transforms
  -> PromptSource formulation + prompt selection
  -> batching / collator / trainer
```

- raw 或外部权威标注是事实真源；structured 和 SFT 都是可重建派生产物。
- 业务字段、属性依赖、合法组合和 `target_text` 构造只存在于离线 builder，不进入训练框架。
- `ShaftDataCenter` 不理解 formulation id、组合依赖或 target 结构；它只调用 PromptSource 的记录准备接口。
- PromptSource 独立拥有 pool/source 合同、对齐校验、静态加权随机选择、fingerprint 和审计。
- trainer、collator 和 template 最终仍只消费普通的 `system_prompt/user_prompt/target_text`。

## 2. 标准 SFT JSONL

每一行始终只有一个已经写好的监督答案。单图使用 `image_path`，多图使用有序 `images`；相对路径以 JSONL
所在目录为基准。

| 字段 | 要求 | 说明 |
| --- | --- | --- |
| `image_path` | 单图必需 | 与 `image`、`images` 三选一 |
| `images` | 多图必需 | 非空有序路径列表 |
| `sample_id` | 生产数据必需 | dataset 内稳定且唯一 |
| `dataset_name` | 推荐 | 运行时以配置中的 source name 为准 |
| `messages` | 可选 | 尾部 assistant message 可提供 materialized target |
| `system_prompt` | 可选 | 普通 materialized 模式使用 |
| `user_prompt` | 可选 | 普通 materialized 模式使用 |
| `target_text` | 必需 | 非空；也可从尾部 assistant message 提取 |
| `prompt_args` | 可选 | 只用于渲染 prompt，必须为 JSON object |
| 其它字段 | 可选 | 进入 `extra`，用于 provenance/build/schema 审计 |

`prompt_args` 不能生成、拼装或选择 target。即使它非空，行内仍必须存在 `target_text` 或尾部 assistant
message。source JSONL 也禁止嵌入 `formulation_targets` 之类的多答案映射。

普通 materialized 行：

```json
{"image_path":"../images/0001.png","sample_id":"0001","system_prompt":"Return JSON only.","user_prompt":"Describe the target.","target_text":"{\"type\":\"shape\"}"}
```

由 PromptSource 轮换措辞、但答案已离线写好的行：

```json
{"image_path":"../images/0001.png","sample_id":"0001","prompt_args":{"proposal_bbox_2d":[10,20,300,400]},"target_text":"{\"type\":\"shape\",\"parameters\":{}}"}
```

## 3. Task formulation 的离线数据合同

同一任务若支持 `A`、`B`、`A+B` 或任意其它人工组合，应把每个 formulation 视为独立的“请求属性集合”，
并分别生成一份标准 SFT JSONL。组合之间不要求包含关系或先后顺序；每份文件仍是一行一个 `target_text`，
格式与 v5.7 完全相同：

```text
sft/formulations/
├── a/train.jsonl
├── b/train.jsonl
└── ab/train.jsonl
```

同一 split 的 formulation 文件必须逐行严格对齐：

- 行数相同；
- `sample_id`、有序图片路径、`dataset_name`、messages、system/user prompt、`prompt_args` 和 `extra` 相同；
- 只有 `target_text` 可以不同；
- 每个 target 都由离线 builder 从 structured/source truth 确定性生成并单独通过业务 codec/schema 校验。

示例：

`a/train.jsonl`：

```json
{"image_path":"../images/0001.png","sample_id":"0001","prompt_args":{"proposal_bbox_2d":[10,20,300,400]},"target_text":"{\"A\":{\"x\":1}}"}
```

`b/train.jsonl`：

```json
{"image_path":"../images/0001.png","sample_id":"0001","prompt_args":{"proposal_bbox_2d":[10,20,300,400]},"target_text":"{\"B\":[2,3]}"}
```

`ab/train.jsonl`：

```json
{"image_path":"../images/0001.png","sample_id":"0001","prompt_args":{"proposal_bbox_2d":[10,20,300,400]},"target_text":"{\"A\":{\"x\":1},\"B\":[2,3]}"}
```

这里没有在线属性解析、target template 或组合逻辑。PromptSource 只在对齐的离线答案中选择一个。

如果不同样本支持的合法 formulation 集合不同，应按 eligibility class 拆成不同 named dataset cohort，再由
`data.schedule.mixing` 混合。只要 task 和 prompt 语义相同，这些 cohort 必须复用同一个 PromptSource pool；
每个 dataset 通过 `formulation_sources` 的键选择自己的合法子集。不要复制 prompt、写空 target、增加
row-level 动态 allowlist，或让框架推断依赖。

## 4. PromptSource 配置

`data.prompt_sources` 以 dataset name 为键。formulation 的物理来源绑定属于 PromptSource 配置，而不是
`DatasetSourceConfig`：

```yaml
data:
  datasets:
    - dataset_name: reconstruction
      use_for_eval: false
    - dataset_name: reconstruction_b_only
      use_for_eval: false

  prompt_sources:
    reconstruction:
      path: ../prompts/reconstruction.formulations.yaml
      apply_to: train       # train | all
      seed: 17              # 省略时继承 experiment.seed
      formulation_sources:
        a:
          train_path: ../data/reconstruction/sft/formulations/a/train.jsonl
        b:
          train_path: ../data/reconstruction/sft/formulations/b/train.jsonl
        ab:
          train_path: ../data/reconstruction/sft/formulations/ab/train.jsonl

    reconstruction_b_only:
      path: ../prompts/reconstruction.formulations.yaml  # 与 reconstruction 复用同一 pool
      apply_to: train
      formulation_sources:
        b:
          train_path: ../data/reconstruction_b_only/sft/formulations/b/train.jsonl
```

约束：

- `formulation_sources` 必须是 pool 中 formulations 的显式非空子集；未知 id 直接失败。
- 该键集合是 dataset 级 eligibility 的唯一真源。运行时只在子集内按 pool 的原始静态权重重新归一化采样；
  单元素子集因此始终选择同一个既有 formulation/prompt。
- formulation 模式不能再为同一 dataset 配置顶层 `train_path/train_paths`，避免两套训练真源。
- `apply_to: train` 时 eval 使用 dataset 顶层已经 materialized 的 `val_path`。
- `apply_to: all` 时每个 formulation source 还必须提供对齐的 `val_path/val_paths`，dataset 顶层不再提供 val。
- 路径相对训练 YAML 解析；每个 source 可以使用单路径或有序多路径。

只需要等义 prompt rotation 时，dataset 继续使用普通顶层 `train_path/val_path`，pool 使用顶层 `prompts`；
这种模式只有一个离线 `target_text`，不需要 `formulation_sources`。

## 5. Formulation pool

Pool 只描述人工允许的 formulation、静态权重和 prompt variants，不描述 target 的生成方法：

```yaml
metadata:
  id: shaft.reconstruction.formulations.v1
  version: v1

arguments:
  proposal_bbox_2d: {type: bbox_2d_0_999}

formulations:
  - id: a
    sampling_weight: 1.0
    prompts:
      - id: direct
        system_prompt: Return compact JSON only.
        user_prompt_template: Reconstruct A near {{ proposal_bbox_2d | json }}.
      - id: inspect
        user_prompt_template: Inspect {{ proposal_bbox_2d | json }} and output A.

  - id: b
    sampling_weight: 1.0
    prompts:
      - id: direct
        user_prompt_template: Reconstruct B near {{ proposal_bbox_2d | json }}.

  - id: ab
    sampling_weight: 4.0
    prompts:
      - id: direct
        user_prompt_template: Reconstruct A and B near {{ proposal_bbox_2d | json }}.
```

- `formulations` 与顶层 `prompts` 只能二选一。
- `target_template`、`target: materialized` 和任何在线 target program 都是非法配置。
- formulation id 和 prompt variant id 在各自作用域唯一；每个 formulation 至少有一个正权重 variant。
- `arguments` 只约束 prompt renderer。支持 `string/enum/integer/float/boolean/json/bbox_2d_0_999`；模板只允许
  `{{ name }}` 与 `{{ name | json }}`。
- 框架不生成属性幂集、不解析组合依赖。复杂集合如 `geometry`、`style`、`geometry_style`、
  `geometry_text_links` 直接由作者逐项列出，并各自绑定一份离线 SFT source。

现有只含顶层 `prompts` 的版本化 pool 会编译成 `default` formulation，因此旧 prompt rotation 是
PromptSource 的单 formulation 子集，不存在第二套运行时。

## 6. 静态权重随机选择

每个 logical draw 先在当前 dataset 的 eligible 子集内按 pool 权重随机选择 formulation，再在其内部随机选择
prompt variant：

```text
logical draw identity
  -> formulation weighted categorical sample
  -> prompt variant weighted categorical sample
  -> selected offline target_text + rendered prompt
```

这不是 round-robin，也不要求序列为 `A,A,B,B,AB,AB`。短序列可以连续抽中同一 formulation。随机值由
`seed + dataset + sample/draw identity` 的 hash 生成，因此同一输入合同可在 planning/runtime、worker、DP
rank 和 exact resume 之间重放。

权重唯一真源是 pool 内各 formulation 的静态 `sampling_weight`。例如 `A / B / A+B` 使用 `1:1:4`
时，长期概率约为 `1/6、1/6、4/6`。PromptSource 不根据 epoch、optimizer step、draw 次数或 wall-clock
修改权重；formulation 只表达本次 prompt 请求和 target 监督的属性集合。

## 7. 校验、fingerprint 与审计

PromptSource 在训练前完成：

1. formulation source id 是 pool id 的显式非空子集；
2. 每份 JSONL 是标准单 target SFT 行；
3. formulation stores 行数和 identity 字段严格对齐；
4. 所有 prompt variants 对 `prompt_args` 通过 exact schema preflight；
5. 每个 formulation 的 `target_text` 非空。

运行时 `extra.prompt_source` 记录 pool/formulation/variant、`draw_id`、静态权重，以及 prompt
program、arguments、user prompt 和选中 target 的 SHA256。execution fingerprint 同时绑定逻辑样本流、所有
formulation source snapshots、media snapshot、pool、dataset eligibility 子集、static weights、seed 和选择算法；
合同改变后旧 checkpoint 只能作为新的初始化点，不能冒充 exact resume。

## 8. 发布检查

每个正式数据 bundle 至少检查：

1. source truth、split manifest 和 sample identity 可追溯，train/val/test 不交叉；
2. 每个 formulation source 都能从同一 structured snapshot 重建；
3. 同一 dataset eligibility 子集中的 formulation JSONL 逐行对齐且只有 `target_text` 不同；
4. 所有 target 通过任务 codec 和 exact output schema；
5. `prompt_args` 只含 prompt renderer 参数，并通过 pool exact schema；
6. pool/source/static weights 配置与训练 recipe 一致；
7. 媒体或 JSONL 快照变化时更新 `media_snapshot_id`；
8. strict loader、focused data/config tests 和最短训练 smoke 通过。

Banana 版本的具体目录和构建顺序见 `scripts/tasks/README.md`。
