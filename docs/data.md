# Shaft 数据与 PromptSource

本文是 Shaft 数据主链、SFT JSONL 和 PromptSource 的公共合同真源。具体 Banana 数据版本、业务标签、
样本规模和构建命令属于 `scripts/tasks/`；配置字段的逐项说明仍以
[config_reference.md](config_reference.md) 为准。

## 1. 数据主链与边界

```text
raw / authoritative source truth
  -> task-local selection（只保存 identity）
  -> structured（可重建的任务语义）+ task-local media
  -> SFT JSONL（规范化训练输入）
  -> ShaftDataCenter / Arrow record store
  -> dataset transforms
  -> PromptSource（可选）
  -> grouping / cardinality / packing / collator
```

- raw 或外部权威标注是事实真源，不能从 SFT 结果反向覆盖。
- selection 只记录 source/instance identity 和采样分层；只要上游可以重读，就不复制 target 真值。
- structured 与 SFT 都是可重建派生产物。字段语义、crop、坐标变换和 target 构造在 task builder 中完成，
  不进入 `ShaftDataCenter`、sampler、collator 或 trainer。
- `ShaftDataCenter` 只负责通用 source 加载、record validation、mixing 和 dataset 装配。
- PromptSource 只改变同一 logical draw 的 prompt/target view，不改变 row identity、数据源权重或 draw 顺序。

正式数据版本必须同时发布可复现 builder、prompt/formulation pool、catalog、训练 recipe 和校验记录。生成的
`data/` 产物可以保持 Git 忽略，但其输入合同和构建逻辑不能只存在于本地脚本或聊天记录。

## 2. SFT JSONL 合同

### 2.1 公共字段

一行代表一个 canonical sample。单图使用 `image_path`，多图使用有序 `images`；路径相对于 JSONL 文件
所在目录解析。

| 字段 | 要求 | 说明 |
| --- | --- | --- |
| `image_path` | 单图必需 | 与 `image`、`images` 三选一 |
| `images` | 多图必需 | 非空有序字符串列表 |
| `sample_id` | 生产数据必需 | 在 dataset 内唯一、稳定，不随 prompt view 改变 |
| `dataset_name` | 推荐 | 应与 catalog 名一致；运行时最终以 catalog/source 名为准 |
| `messages` | materialized messages 模式 | user/content 中的 image 占位符数量必须等于图片数 |
| `system_prompt` | materialized prompt 模式 | 可为空 |
| `user_prompt` | materialized prompt 模式 | 普通单图模板允许为空；PromptSource pool 模式必须为空或省略 |
| `target_text` | 依模式决定 | materialized target 必须非空；rendered target 必须省略或为空 |
| `prompt_args` | PromptSource 模式 | 必须是 JSON object，字段与 pool `arguments` 完全一致 |
| `extra` | 可选 | provenance、坐标系、builder 版本等审计信息 |

图片字段必须三选一；显式 `messages` 与非空 `prompt_args` 不能同时出现。`target_text` 可以来自行内字段，
也可以从最后一个 assistant message 提取。

### 2.2 Materialized 数据

普通 prompt + target：

```json
{"image_path":"../images/0001.png","sample_id":"0001","system_prompt":"Return JSON only.","user_prompt":"Describe the target.","target_text":"{\"type\":\"shape\"}"}
```

标准 messages：

```json
{"images":["../images/a.png","../images/b.png"],"sample_id":"pair-0001","messages":[{"role":"user","content":[{"type":"image"},{"type":"image"},{"type":"text","text":"Compare them."}]},{"role":"assistant","content":[{"type":"text","text":"{\"same\":true}"}]}]}
```

未配置 PromptSource 的 dataset 必须是 materialized 数据，并且不能携带非空 `prompt_args`。这是普通
HF/LLaMA-Factory 风格数据的直接退化路径。

### 2.3 PromptSource canonical 数据

当 prompt 与 target 都由 formulation 生成时，SFT 行只保存共同真值参数：

```json
{"image_path":"../images/0001.png","sample_id":"0001","dataset_name":"reconstruction","prompt_args":{"attribute_a":{"value":1},"attribute_b":{"value":2}}}
```

这类行必须省略 `messages`、非空 `user_prompt` 和 `target_text`。不要为每个 formulation 复制一条 train
row，也不要人工维护彼此独立的 target 真源；builder 应从 structured/source truth 一次性、确定性地产生
完整 `prompt_args`。当组合依赖过于复杂、不适合用受限模板拼装时，derived builder 可以为每个**人工声明**
的 formulation 预计算一个 `target_<formulation>` JSON 参数，但每个参数仍必须能从同一 structured/source
truth 精确重算。

若 pool 只轮换 wording、target 仍已物化，则保留 `target_text`：

```json
{"image_path":"../images/0001.png","sample_id":"0001","prompt_args":{"proposal_bbox_2d":[10,20,300,400]},"target_text":"{\"type\":\"shape\",\"parameters\":{}}"}
```

## 3. PromptSource 术语

PromptSource 把 canonical sample 和 logical draw context 确定性地投影成一次训练消费的完整样本：

```text
canonical sample + logical draw context
  -> task formulation
  -> prompt variant
  -> system_prompt + user_prompt + target_text + audit
```

- **PromptSource**：数据层的投影、选择、fingerprint 和审计边界。
- **task formulation**：决定问什么和监督什么；可以是任意人工声明的属性子集或任务视图，A/B/AB 只是
  最小示例。
- **prompt variant**：同一 formulation 内语义等价的措辞轮换。
- **curriculum sampling**：按 dataset-local logical draw 渐进调整 formulation 权重。

现有 prompt 轮换属于 prompt variant，不是平行运行时。本能力对外使用 “task formulation sampling” 和
“curriculum sampling”；不称为 multi-view learning，避免与多视角表征学习混淆。

## 4. PromptSource 配置

`data.prompt_sources` 以 dataset name 为键。未出现的 dataset 直接消费 materialized 数据。

```yaml
data:
  prompt_sources:
    reconstruction:
      path: ../prompts/reconstruction.formulations.yaml
      apply_to: train       # train | all
      seed: 17              # 省略时继承 experiment.seed
      schedule:
        interpolation: linear  # step | linear
        points:
          - source_draw: 0
            weights: {a: 1.0, b: 1.0, ab: 0.0}
          - source_draw: 50000
            weights: {a: 0.5, b: 0.5, ab: 1.0}
          - source_draw: 150000
            weights: {a: 0.1, b: 0.1, ab: 1.0}
```

- dataset key 必须对应 enabled source；pool 路径相对训练 YAML 解析。
- `apply_to: train` 只投影训练集，eval 必须已经 materialized。
- `apply_to: all` 同时投影 train/val；eval 固定使用 `source_draw_id=0`。若必须分别评估各个 formulation，
  推荐准备固定的 materialized eval rows，而不是依赖一次 formulation 抽样。
- schedule 首点必须为 0，后续 `source_draw` 严格递增。每点完整列出所有 formulation，权重有限、非负，
  且至少一项为正。
- `step` 保持左侧权重；`linear` 对原始权重插值后再归一化。未配置 schedule 时使用 pool 静态权重。

每个 logical draw 都在当时权重大于 0 的 formulations 中做一次 weighted categorical sampling。它不是
round-robin，也不保证短前缀按比例整齐排列；例如权重相同的 A/B/AB 完全可能连续多次抽到 A。随机值由
`seed + dataset + sample/draw identity` 的 hash 产生，因此统计上按权重随机，同时在 planning、worker、
DP rank 和 exact resume 间可复现。

curriculum 不接受 epoch、optimizer step、百分比或 wall-clock。它以当前 dataset 第几次进入逻辑样本流的
`source_draw_id` 为轴，因此修改其他 dataset 的 mixing 权重不会移动本 dataset 内的阶段边界。

## 5. Formulation pool

```yaml
metadata:
  id: shaft.reconstruction.formulations.v1
  version: v1

arguments:
  attribute_a: {type: json}
  attribute_b: {type: json}

formulations:
  - id: a
    sampling_weight: 1.0
    target_template: '{{ attribute_a | json }}'
    prompts:
      - id: direct
        sampling_weight: 1.0
        system_prompt: Return compact JSON only.
        user_prompt: Reconstruct attribute A.
      - id: inspect
        sampling_weight: 1.0
        system_prompt: Return compact JSON only.
        user_prompt: Inspect the image and output attribute A.

  - id: b
    sampling_weight: 1.0
    target_template: '{{ attribute_b | json }}'
    prompts:
      - id: direct
        user_prompt: Reconstruct attribute B.

  - id: ab
    sampling_weight: 0.0
    target_template: '{"A":{{ attribute_a | json }},"B":{{ attribute_b | json }}}'
    prompts:
      - id: direct
        user_prompt: Reconstruct attributes A and B together.
```

约束：

- `metadata.id/version`、formulation id 和 variant id 必须非空且在各自作用域唯一。
- pool 级 `arguments` 是 prompt 与 target program 的共同 schema。类型支持
  `string/enum/integer/float/boolean/json/bbox_2d_0_999`。
- 模板只允许 `{{ name }}` 与 `{{ name | json }}`；不支持 Jinja、属性访问、表达式或任意代码。
- 每个 formulation 必须定义 `target_template`，或显式声明 `target: materialized`。
- 每个 formulation 至少有一个正权重 variant；所有可达 formulation/variant 在 Arrow cache 构建时统一
  preflight，缺参数、多参数和非法 bbox 都直接失败。
- rendered-target formulation 禁止行内非空 `target_text`；materialized-target formulation要求非空
  `target_text`。同一 pool 最好统一 target 模式，避免一行无法同时满足所有可达 formulation。

现有只含顶层 `prompts` 的版本化 pool 是正式简写。它会立即编译成一个 `default` formulation，并使用
materialized target。因此旧 prompt rotation 已经是 PromptSource 的子集，不存在第二套训练运行时。

### 5.1 任意复杂子集由配置人工枚举

PromptSource 不自动生成属性幂集，也不接受 `all_combinations: true` 之类的隐式展开。pool 作者只列出业务上
合法的组合，并为每个组合明确 prompt、target 和静态权重。例如实际合法集合可以是
`geometry`、`style`、`geometry_style`、`geometry_text_links`，而没有其它组合：

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
    prompts:
      - id: main
        user_prompt: Reconstruct geometry only.
  - id: style
    sampling_weight: 1.0
    target_template: '{{ target_style | json }}'
    prompts:
      - id: main
        user_prompt: Reconstruct style only.
  - id: geometry_style
    sampling_weight: 2.0
    target_template: '{{ target_geometry_style | json }}'
    prompts:
      - id: main
        user_prompt: Reconstruct geometry and style.
  - id: geometry_text_links
    sampling_weight: 1.0
    target_template: '{{ target_geometry_text_links | json }}'
    prompts:
      - id: main
        user_prompt: Reconstruct geometry, text, and links.
```

这里四个 `target_*` 都是派生参数，不是新的 source truth。若组合结构简单，也可以只保存 atomic arguments，
由各 formulation 的 `target_template` 显式组装。

当前一个 dataset/pool 的所有 canonical rows 共用同一 argument schema 和可达 formulation 集合。如果某些
row 缺少某类真值、不能支持某些组合，应按 eligibility class 拆成不同 named datasets，并分别绑定人工维护的
pool，再由 `data.schedule.mixing` 混合；不要用 `null` 冒充有效监督，也不要让在线运行时推断依赖关系。

## 6. 确定性、resume 与审计

sample context 中有两个独立位置：

```text
draw_id         全部 dataset 的全局 logical draw
source_draw_id  当前 dataset 自身的 logical draw
```

formulation 和 variant 使用独立的确定性 hash 随机域。新增一句同义 prompt 只改变 formulation 内 wording，
不会改变任意 configured formulation 的分布。这里的“确定性”只表示同一 logical draw 可重放，不表示输出
序列固定轮换。planning item 与 DataLoader runtime item、多 worker、不同 DP rank 和 exact resume 都会在
相同输入合同下得到相同结果。

execution fingerprint 绑定 sample stream、`source_draw_id` 算法、record/media snapshot、pool schema、
prompt/target program、schedule、权重、seed 和 renderer。任一合同变化后不能沿用旧 optimizer schedule；
旧 checkpoint 只能作为 `init_from_checkpoint` 启动新 schedule。

投影后的 `extra.prompt_source` 记录 pool/formulation/variant、两个 draw id、实际权重，以及
arguments/prompt/target 的 SHA256。materialized 模式不伪造该对象。统计 curriculum 时要区分 formulation
sample count 和 supervised token count；AB 样本比例不等于 AB 对 loss 的 token 比例。

## 7. 发布检查

每个正式数据 bundle 至少检查：

1. raw/source truth、split manifest 与 selection identity 可追溯，train/val/test 两两不交叉；
2. structured、SFT 和 task-local media 一一对应，sample id 与媒体路径唯一；
3. 每个媒体路径存在，解码尺寸和坐标空间与 structured 记录一致；
4. SFT `prompt_args` 能从 structured/source truth 精确重算，且通过 pool exact schema validation；
5. 每个可达 formulation 的 target 可以重算、解析，并符合任务 codec/output schema；
6. validation/test 不使用 train-only augmentation；eval 的 materialized/PromptSource policy 明确；
7. catalog 名称、路径、权重、eval flag、prompt pool 和所有训练 recipe 完全一致；
8. 媒体或 JSONL 快照实质变化时更新 `media_snapshot_id`；
9. strict loader、focused data/config tests 和最短训练 smoke 通过。

具体 Banana 数据版本的源、行数、构建命令和审计基线见 `scripts/tasks/README.md`。
