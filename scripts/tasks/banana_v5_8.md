# Banana v5.8 数据准备合同

本文定义 Banana v5.8 的数据准备方式。当前数据 snapshot 已完成物化并通过全量 schema、alignment、媒体解码
和真实训练读取链 smoke；Qwen3.5-4B 正式训练配置已经发布。

框架公共合同见 [`docs/data.md`](../../docs/data.md)；v5.7 已发布数据见
[`banana_v5_7.md`](banana_v5_7.md)。

## 1. v5.8 相对 v5.7 的变化

v5.7 的 reconstruction 行已经是标准 materialized SFT：

```json
{"image_path":"../images/example.png","sample_id":"example__context_00","dataset_name":"shape_context_reconstruction","system_prompt":"","user_prompt":"","prompt_args":{"proposal_bbox_2d":[115,108,733,621]},"target_text":"{\"type\":\"shape\",\"parameters\":{...}}","extra":{...}}
```

v5.8 **不改变这份行格式**。Grounding 仍是一个普通单目标任务；只有确实存在多种监督属性集合的任务才使用
formulation。当前配置如下，列表顺序只用于稳定配置与审计，不代表采样阶段：

| dataset cohort | shared pool / eligible formulations | 监督含义 |
| --- | --- | --- |
| `grounding_layout` | `grounding_layout` / 无 formulation | 始终输出完整 bbox + label objects |
| `background` | `background` / 无 formulation | 判断完整页面是否存在大面积不可编辑视觉背景 |
| `shape_context_reconstruction` | `shape_context_reconstruction` / `appearance, geometry, reconstruction` | 外观、几何、完整可重建属性 |
| `line_context_reconstruction` | `line_context_reconstruction` / `appearance, points, reconstruction` | 外观、点序列、完整重建 |
| `line_context_points` | `line_context_reconstruction` / `points` | line reconstruction 的真实/审计 points-only cohort |
| `image_context_reconstruction` | `image_context_reconstruction` / `image_type` | 已审核的 13 类 image type |

`grounding_layout` 和 `background` 都沿用普通单 `target_text` 数据与顶层 prompt。Grounding 不生成
labels-only 或 boxes-only target；background 只输出审核过的 `{"background":true|false}`。
`line_context_points` 不是独立任务；它是 `line_context_reconstruction` 任务中只能监督 `points`
formulation 的物理数据 cohort。它不虚构 appearance/full，`image_context_reconstruction` 也不虚构
geometry/full。框架不生成幂集、不推断依赖。样本 eligibility 不同时，人工拆成不同 named dataset cohort；
若 task/prompt 语义相同则继续复用同一个 pool，只由各 dataset 的 `formulation_sources` 键选择合法子集。
外层 dataset weight 和内层 formulation weight 共同控制任务总体概率。

Grounding 继续使用一份完整的 v5.7 形态 SFT JSONL。其余显式 formulation 各使用一份相同格式的 JSONL；
所有 `target_text` 均由 builder 预写。训练在线阶段只做随机选择，不解析 `parameters`，不通过
`prompt_args` 拼 target。

## 2. Source truth 与派生层级

```text
authoritative raw / gt_standard snapshot
  -> stable split + source identity selection
  -> structured canonical sample
  -> one offline builder invocation
       -> ordinary task: one standard SFT JSONL
       -> multi-formulation task: one standard SFT JSONL per eligible formulation
  -> PromptSource online weighted selection
```

- raw/`gt_standard` 是唯一事实真源。
- structured 保存完整业务语义与坐标变换结果。
- 每个 ordinary/formulation target 都必须从 structured/source truth 确定性重建。
- builder 同时写出当前 dataset cohort 的全部 eligible formulations，避免各脚本产生 identity、crop 或
  provenance 漂移。
- formulation JSONL 是训练派生产物，不是新的业务真源。

如果 v5.8 使用新的 synthetic/raw snapshot，必须冻结新的 snapshot id、split manifest 和 selection；不得只
改目录名后沿用 v5.7 的审计结论。

### 2.1 已恢复的真实任务

当前 compact raw 已删除 background 字段，也没有 image instance 的 13 类 `image_type` 真值，因此这两个
任务不能从当前 JSON 猜测重建。v5.8 使用显式历史来源恢复：

- `background`：以人工审核后的
  `background_annotations_opus48_reviewed_20260710.jsonl` 为标签真值，并与已验证的 v5.3 task-local
  structured/SFT/media 逐 ID、逐 target 对账；
- `image_context_reconstruction`：原 enriched raw 已不可用，以完整且经过媒体、schema 和来源身份复验的
  v5.3 derived bundle 恢复。它被记录为 `verified_recovered_v5_3_derived_bundle`，不得反向冒充 active raw；
- 两者共享恢复 snapshot `banana-v5.8-reviewed-real-recovery-v1`，不修改当前 `data/raw`；历史审核文件和
  split manifest 只复制到 `data/raw/imports/banana_v5_3_replay_20260722` sidecar；
- raw sidecar 和 task reports 保存历史 selection/structured/SFT、审核标注、split manifests 与实际 prompt
  pool 的 SHA256；snapshot 不只依赖目录名；
- 全部媒体保持原始像素，不做 resize；59,627 个 task-local 媒体均通过完整解码和声明尺寸校验。

恢复结果：

| task | train rows | unique source images | canonical 175 test overlap |
| --- | ---: | ---: | ---: |
| `background` | 38,443 | 38,443 | 0 |
| `image_context_reconstruction` | 21,184 | 4,134 | 0 |

`background` 还排除了三个历史测试 manifest 的 313 个 ID 并证明恢复 ID 精确等于审核标注 ID 减去该并集；
因此 canonical `vlm.test.json` 的 175 张是其严格子集，而不是仅靠当前目录未发现重叠。

### 2.2 V9 合成 reconstruction snapshot

shape/line 使用 `regulated_layout_dataset_v9_20260802`：

- `train.txt` 100,000 项、`val.txt` 500 项，内部唯一且交集为 0；
- 解压后的 `gt_standard/`、`image_source/`、`img/` 各有 100,500 个一一对应文件；
- `checksums.sha256` 的 24 项全部通过，包括两个 JSON zip、20 个图片分卷和 train/val；
- train 全量图像可解码，100,000 个 source document 无 fatal error；有效 shape 3,234,558 个，
  有效 line 409,768 个；120 个包含连续重复点/零长度 segment 的 line instance 直接排除，不修补坐标，
  其它无效实例也只从 reconstruction selection 排除，不修改 V9 真源。

v5.8 selection 位于 `data/reconstruction_v5_8_selection`，策略与结果为：

| selection | rows | 规则 |
| --- | ---: | --- |
| shape | 300,000 | 数量不超过 60,000 的 9 个稀有 shape type 全保留；其余按完整外观/几何 stratum 无放回抽样；rectangle 上限 20%，实际 17.45% |
| line | 297,489 | 118,996 个有效多叉 line 全保留；单段 line 无放回抽样且最终不超过 60% |
| synthetic line points | 15,000 | 从全部 118,996 个有效 V9 多叉 line 选择；完整 line-attribute stratum 数量不超过 256 的 12,011 条先全保留，再从头部补足 |
| real line points | 120,744 | 排除 canonical test 后，active compact raw 中所有非空 points 全保留，不抽样 |

line 的完整 stratum 同时包含 line type/style、segment count、dash、begin/end arrow、fill、border 和
corner style；points synthetic selection 不是只按分支数均衡，也不依赖 full-line selection 的子集。
最终 `line_context_points` 是 120,744 条真实 points + 15,000 条带噪合成多叉 points，共 135,744 条。

三个 V9/points cohort 的最终物化结果：

| dataset cohort | shared crops | materialized formulation rows | 媒体终审 |
| --- | ---: | --- | ---: |
| `shape_context_reconstruction` | 300,000 | `appearance / geometry / reconstruction` 各 300,000 | 300,000 / 300,000 |
| `line_context_reconstruction` | 297,489 | `appearance / points / reconstruction` 各 297,489 | 297,489 / 297,489 |
| `line_context_points` | 135,744 | `points` 135,744 | 135,744 / 135,744 |

733,233 张派生 crop 均完整解码且与声明尺寸一致。Shape/Line/Synthetic-points 的每张 crop 都实际应用
1–3 层 `synthetic_realism_v1`；真实 points 的 120,744 张 crop 均保持 clean。所有 formulation store 均逐行
identity 对齐，exact target schema、0–999 坐标、非退化 line segment 和唯一 `sample_id` 全量通过。

完整 v5.8 配置实际装配为 850,420 个 logical rows。首次 DataCenter smoke 建立并复用了 10 份 Arrow source
cache，在 6,000 个确定性 weighted schedule draw 中覆盖了全部合法 formulation 和 detailed/concise variant；
`line_context_points` 始终只选择共享 pool 的 `points`。六个 cohort 各随机读取一张实际媒体成功，train stream /
execution contract 均完整且 fingerprint 一致。

## 3. 推荐目录

每个 task 使用一套共享 structured/media。普通任务使用 `sft/train.jsonl`；多 formulation 任务按
formulation 分文件：

```text
data/<task>/
├── selection/
│   ├── train.jsonl
│   └── val.jsonl
├── structured/
│   ├── train.jsonl
│   └── val.jsonl
├── images/
│   └── train/
├── sft/
│   ├── train.jsonl                  # ordinary task only
│   ├── val.jsonl                    # ordinary task only
│   └── formulations/                # multi-formulation task only
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

Grounding 使用 `data/grounding_layout/sft/train.jsonl`，background 使用
`data/background/sft/train.jsonl`。多个 formulation 可以引用同一 task-local image；不需要复制图片。

## 4. 每份 JSONL 的完整格式

每行与 v5.7 相同，必须含一个非空 `target_text`：

```json
{
  "image_path": "../../../images/train/ab/shape_000001__context_00.png",
  "sample_id": "shape_000001__context_00",
  "dataset_name": "shape_context_reconstruction",
  "system_prompt": "",
  "user_prompt": "",
  "prompt_args": {
    "proposal_bbox_2d": [115, 108, 733, 621]
  },
  "target_text": "{\"type\":\"shape\",\"parameters\":{...}}",
  "extra": {
    "prompt_pool_id": "shaft.shape_context_reconstruction.formulation_pool.v5.8",
    "source_sample_id": "shape_000001",
    "source_type": "synthetic_gt_standard_context",
    "structured_extra": {...}
  }
}
```

当前框架要求 formulation stores 的 `extra` 也完全一致，因此 formulation id 放在目录、manifest 和配置中，
不要写进每行 `extra`。紧凑形式如下：

```json
{"image_path":"../../../images/train/ab/shape_000001__context_00.png","sample_id":"shape_000001__context_00","dataset_name":"shape_context_reconstruction","system_prompt":"","user_prompt":"","prompt_args":{"proposal_bbox_2d":[115,108,733,621]},"target_text":"{\"type\":\"shape\",\"parameters\":{...}}","extra":{"prompt_pool_id":"shaft.shape_context_reconstruction.formulation_pool.v5.8","source_sample_id":"shape_000001","source_type":"synthetic_gt_standard_context","structured_extra":{...}}}
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

5 个已冻结的 task pool 位于：

- `configs/prompts/pools/grounding_layout.v5.8.yaml`
- `configs/prompts/pools/background.v5.8.yaml`
- `configs/prompts/pools/shape_context_reconstruction.v5.8.yaml`
- `configs/prompts/pools/line_context_reconstruction.v5.8.yaml`
- `configs/prompts/pools/image_context_reconstruction.v5.8.yaml`

`line_context_reconstruction` 与 `line_context_points` 两个物理 dataset cohort 都指向
`line_context_reconstruction.v5.8.yaml`。`points` formulation 及其 detailed/concise prompts 只在这份 pool
定义一次；points-only cohort 只是把 `formulation_sources` 配置为单个 `points`。

除 background 外，每个任务请求都恰好有两个等义 prompt variant：

- `detailed`：完整任务合同，延续并收紧 v5.7 的输出约束；
- `concise`：专用模型使用的极简任务表达，target schema 与 detailed 完全相同。

`background` 按当前业务要求只保留一个 `detailed` variant。它明确区分大面积不可编辑 backing 与普通局部
image object、可编辑 vector panel 以及简单画布，避免把“页面包含图片”误判为“页面具有背景图”。

所有 variant 都显式配置同一份 unified `system_prompt`。Grounding pool 使用顶层 `prompts`；shape/line
等多目标 pool 的 `formulations` 列表是唯一配置顺序真源，不再额外维护 `formulation_order`，但运行时不会按
这个顺序轮换。pool 只列人工允许的组合，不使用 `all_combinations` 或自动依赖规则。

## 6. 训练配置

配置入口：

- catalog：`configs/data/banana_v5_8.yaml`
- PromptSource/source/probability 绑定：
  `configs/train/banana_sft_4b_qwen35_v5_8.yaml`

当前组合媒体 snapshot id 为 `banana-v5.8-v9-20260802-reviewed-real-v1`，同时覆盖 V9 派生 crop、active
compact raw 的真实 line points，以及已复验恢复的 background/image-type 媒体；任一来源或 selection 改变都
必须更新该 id。

Grounding 和 background 的普通 `train_path/val_path` 由 catalog 绑定；grounding pool 在
detailed/concise 之间随机选措辞，background 只选择 detailed，target 分别始终是完整 objects 和审核 boolean。
正式训练 recipe 为其它 dataset eligible formulation id 绑定标准 SFT 路径。shape
和 full-capable line 使用 `1:1:4`；其中 `reconstruction` 在对应 pool 内约占 66.7%。points-only cohort 的
eligible 子集只有共享 pool 中既有的 `points`，因此确定选择它。

line reconstruction 还包含 points-only cohort：catalog 对 full-capable/points-only 两个 dataset 使用 `6:2`，
与 full-capable pool 内部的 `1:1:4` 合成后，line reconstruction 总体约为 appearance 12.5%、points 37.5%、
reconstruction 50%。修改 catalog weight 或 formulation `sampling_weight` 即可改变这个分布。

外层 dataset 权重为 `grounding:background:shape:line:line_points:image = 4:1:5:6:2:1`。
Qwen3.5-4B run 名为 `banana-v5.8-qwen35-4B`，使用 seed 465、12,000 optimizer steps、BF16 full
fine-tuning、DDP、BS1、GA8、
`bounded_cost + fixed`、8,000 token 上限、cosine scheduler、10% warmup、peak LR 2e-5 与
weight decay 0.003。每 2,000 steps 保存完整可恢复 checkpoint，最多保留 10 个；不额外发布 root `best`
final model。

显式 formulation dataset 不再同时配置顶层 `train_path`；物理训练来源由对应 PromptSource 管理。Grounding
和 background 作为普通单目标 dataset 继续由 catalog 提供顶层 source。外层 `data.schedule.mixing` 仍负责
不同 dataset/task 之间的 mixing，PromptSource 内层负责 formulation 或 prompt variant sampling。

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
2. 冻结普通任务 target schema，以及多目标任务的合法 formulation 集合、各 target schema 和 prompt wording。
3. builder 为普通任务生成一份 SFT；为多目标任务一次性生成 eligibility 子集的 formulation SFT。
4. 检查每份文件的唯一 identity、媒体存在性、target codec/schema。
5. 检查 formulations 逐行对齐且仅 `target_text` 不同。
6. 编译 pool，并对全部行执行 `prompt_args` exact schema preflight。
7. 用 Shaft strict config loader 加载配置，运行 focused data tests 和最短 SFT smoke。
8. 生成 build/alignment/schema 报告，冻结 `media_snapshot_id`。
9. 只有以上全部通过后，登记 production catalog/recipe 和数据规模基线。

两个真实任务的可复现恢复命令要求显式提供历史 bundle、审核标注和三个历史测试 manifest：

```bash
uv run python scripts/tasks/recover_v5_8_real_tasks.py \
  --source-bundle-root /path/to/banana_v5_3_replay_20260722 \
  --background-annotations /path/to/background_annotations_opus48_reviewed_20260710.jsonl \
  --test-manifests /path/to/main.test.json /path/to/inpainting.test.json /path/to/vlm.test.json \
  --canonical-test-manifest data/raw/splits/vlm.test.json \
  --output-root data \
  --workers 40 \
  --clean
```

脚本先冻结 canonical manifest 为 175 项及预期 SHA256，并要求历史 manifests 中存在完全相同的副本；再在
同文件系统 staging 中完成 PromptSource、selection/structured 来源 identity、target、媒体解码和尺寸校验，
全部通过后才原子替换两个 task 目录；即使指定 `--clean`，失败也不会先删除现有可用数据。

V9 selection 与三个 reconstruction cohort 的可复现命令：

```bash
uv run python scripts/tasks/prepare_gt_standard_v5_7.py \
  --dataset-root /path/to/regulated_layout_dataset_v9_20260802 \
  --output-root data/reconstruction_v5_8_selection \
  --workers 50 \
  --selection-profile v5.8 \
  --shape-target 300000 \
  --line-target 300000 \
  --line-points-target 15000 \
  --shape-keep-all-threshold 60000 \
  --shape-max-rectangle-fraction 0.20 \
  --line-max-single-segment-fraction 0.60 \
  --line-points-keep-all-stratum-threshold 256 \
  --seed 57

uv run python scripts/tasks/prepare_real_line_context_points.py \
  --output data/reconstruction_v5_8_selection/line_points_real/train.jsonl \
  --workers 50 \
  --clean

uv run python scripts/tasks/build_context_reconstruction_sft.py \
  --synthetic-root /path/to/regulated_layout_dataset_v9_20260802 \
  --raw-root data/raw \
  --output-root data \
  --shape-selection data/reconstruction_v5_8_selection/shape/train.jsonl \
  --line-selection data/reconstruction_v5_8_selection/line/train.jsonl \
  --line-point-real-selection data/reconstruction_v5_8_selection/line_points_real/train.jsonl \
  --line-point-synthetic-selection data/reconstruction_v5_8_selection/line_points/train.jsonl \
  --shape-prompt-pool configs/prompts/pools/shape_context_reconstruction.v5.8.yaml \
  --line-prompt-pool configs/prompts/pools/line_context_reconstruction.v5.8.yaml \
  --line-point-prompt-pool configs/prompts/pools/line_context_reconstruction.v5.8.yaml \
  --tasks shape_context_reconstruction line_context_reconstruction line_context_points \
  --workers 50 \
  --chunksize 16 \
  --seed 42 \
  --png-compress-level 1
```

正式写盘前应给同一 builder 命令追加 `--preflight-only`。它会遍历全部 selection，解析 source truth，执行
确定性 crop/坐标量化、全部 formulation target 投影与噪声 plan 生成，但不写图片、不发布 task 目录；当前
297,489 条 full line 与 135,744 条 points cohort 均已通过该全量 preflight。

builder 只改派生 crop，不改 V9 原图。shape/line 与 synthetic points 的每个 crop 都应用一至三层
`synthetic_realism_v1` 尺寸不变噪声；real points crop 不加合成噪声。所有 formulation 在同一 worker pass
由同一 crop/structured row 生成，因此逐行 identity 一致，只改变 `target_text`。

## 11. 发布清单

- [x] source snapshot 与 split manifest 已冻结并可追溯。
- [x] 所有真实来源与 canonical 175 张 `vlm.test.json` 的 source identity 重叠为 0；V9 train/val 交集为 0。
- [x] 普通任务 schema、合法 formulation 集合和每个 exact target schema 已人工确认。
- [x] 每个 ordinary/formulation source 都是标准 v5.7 形态 JSONL，逐行都有非空 `target_text`。
- [x] 同一 dataset eligibility 子集的 JSONL 行数、顺序和 identity 完全一致，只有 target 不同。
- [x] `prompt_args` 只服务 prompt renderer，不含 target 组合真值。
- [x] pool 不含 `target_template` / `target`，source 行不含多 target mapping。
- [x] 所有 target 通过 codec/schema，所有 prompt variants 通过参数 preflight。
- [x] 静态随机分布、planning/runtime 一致和 resume fingerprint 已 smoke。
- [x] 数据产物、报告、pool、配置、builder 与文档属于同一 v5.8 snapshot。

当前数据 snapshot 已具备训练读取条件；正式训练尚未启动，训练 canary、吞吐基线和模型指标不属于本数据
准备清单的完成条件。
