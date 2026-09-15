# Banana v5.10 数据复现

版本标识：`banana-v5.10`。维护真源是本仓库的脚本、配置和本文档，不是本机的 `subTasks/`。
Git `v5.10` tag 只在整套流程完成、验收并提交后创建；当前 shape 阶段不代表整版已发布。

## 入口与状态

### 已确认的训练数据配置

4B 正式配置已落地：`configs/train/banana_sft_4b_qwen35_v5_10.yaml`。
8卡A800、基模重新训练、AdamW、BS1/GA8、24,000步、LR 3e-5、视觉塔1.5e-5、
warmup 0.1、weight decay 0.003、max length 8000，其他批处理/保存设置继承v5.9。
启动时设置 `WANDB_MODE=offline`。Muon hybrid 已延期至总 TODO，不用于本次训练。
下述片段仍是数据合同真源，focused测试保证正式4B配置与其一致。

0.8B 配置：`configs/train/banana_sft_0_8b_qwen35_v5_10.yaml`，计划在 worker2 八卡 A800
从 `Qwen3.5-0.8B` 基模训练。继承 4B 的七任务数据、配比、24k 步和优化/保存参数；
BS4/GA2（全局 batch 64），length 分组、varlen、无 packing、buffer 512、vision patches
预算 32768、8 workers/prefetch 4，max length 8000。保持 gradient checkpointing 开启，
避免上一版关闭后出现的长尾激活 OOM。启动时设置 `WANDB_MODE=offline`；配置校验不替代
真实八卡长尾显存验证，不自动启动训练。

27B 配置：`configs/train/banana_sft_27b_qwen38_v5_10_full_zero3.yaml`，用于另一集群的
两节点各 8 张 A800 80GB（IB/RDMA）。Qwen3.8-27B 基模、`qwen38vl` 非思维链模板，
七任务数据和配比不变；BS1/GA4、全局 batch 64、16k 步（1,024,000 次采样），主干/视觉塔/aligner LR
分别为 2e-6/1.2e-6/4e-6，warmup 0.13（2080 步）、WD 0.003。bounded_cost/fixed/padded、buffer 512、
max length/token budget 8000、vision patches 16384、6 workers/prefetch 2。
启用 gradient checkpointing，复用 `zero3_bf16_lowmem.json`；未默认开启通信重叠。
每 2000 步保存完整 resume checkpoint，保留最近 3 个，关闭额外 final model。

两节点须使用一致代码、环境、基模和数据内容，路径配置在各节点均可解析；输出 checkpoint
目录须为两节点可见的共享存储，以保存和恢复所有 rank 的 ZeRO 分片。
在各节点仓库根目录运行以下命令；首节点 `NODE_RANK=0`，另一节点改为 `1`，
`MASTER_ADDR` 均填写首节点可达的内网 IP，端口须空闲且可互通。不要使用 `--standalone`。

```bash
export MASTER_ADDR="填写首节点内网IP"
export MASTER_PORT=29500
export NODE_RANK=0

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 WANDB_MODE=offline PYTHONUNBUFFERED=1 \
uv run --no-sync torchrun --nnodes=2 --nproc-per-node=8 \
  --node-rank="$NODE_RANK" --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
  scripts/train.py sft \
  --config configs/train/banana_sft_27b_qwen38_v5_10_full_zero3.yaml
```

命令显式传入节点 rank 和 rendezvous 参数。网卡/HCA 名称由目标集群确定，不能复制本机值；
先确认跨节点 NCCL 实际使用 IB/RDMA，再检查长尾 batch 显存和吞吐。配置通过校验不代表
已完成该集群的通信、显存或恢复验证。

- Catalog：`configs/data/banana_v5_10.yaml`。
- 配套片段：`configs/data/banana_v5_10_runtime.yaml`。不是独立训练入口；框架没有
  YAML include，待模型/优化参数对齐后合入正式训练 YAML，不能只换 catalog 而遗漏 prompt 映射。
- 权重：grounding 8、合成 shape 21.6、真实 shape 2、合成 line 21.6、line points 4、image 1、
  background 1.8，总计60。对应样本占比约13.33%、36.00%、3.33%、36.00%、6.67%、1.67%、3.00%，
  不是 token/loss 占比。
- 4B/0.8B 的24k步/全局batch64对应1,536,000次采样；上述七个来源预计分别为204,800、552,960、51,200、
  552,960、102,400、25,600、46,080次。image/background约为现有视图数量的1.21/1.20倍，
  不是唯一图片覆盖率。warmup ratio仍为0.1（2,400步）；每2k保存，上限10，早期checkpoint会轮替删除。
- weighted + shuffle，seed 465，新 media snapshot；七个来源均 train-only，关闭在线 eval。
- 真实 shape 与合成 shape 共用原池。shape appearance/geometry/reconstruction、
  line appearance/points/reconstruction 均按原池1:1:4随机选择，不按阶段轮换。
  line points只注册points；image/background原样复用，不表示历史数据重新做了测试隔离。
- 详细/简略 variant 权重不变，不复制 prompt，不重新叠加在线像素增强。
- 路径按配置文件所在目录解析；片段的数据路径也适用于 `configs/train/` 下的训练 YAML。
  模型、batch、像素预算、max length和步数另行确认；固定步数不保证遍历所有视图，覆盖率届时计算。

2026-09-09目录收尾：已清理旧派生备份及完成的`.build`工作目录，正式任务的复现锁/选择清单/
内容报告保留；`data/banana_v5_9`历史raw快照和旧reports也已退役。下文历史备份路径记录的是
当时的发布操作，不代表备份现在仍存在。当前目录职责见`data/README.md`。
复现从完整源输入重新prepare；不能再引用已删除的旧work-root或v5.9路径。

- 配置：`configs/data/preparation/banana_v5_10.json`。
- 执行：`scripts/tasks/prepare_banana_v5_10.py`。
- 合同与回归：`tests/test_prepare_banana_v5_10.py`，复用 context reconstruction builder。
- 产物：`data/<task>/{selection,structured,images,sft,reports}`。

| 任务 | 输入真源 | 当前状态 |
| --- | --- | --- |
| shape_context_reconstruction | 修正后的 V10 GT + 原始 img + train/val | 已生成、全量验收并发布 |
| shape_context_reconstruction_real | TXT白名单中的完整真实shape子属性 | 已验收发布：17,006图，51,018条SFT |
| line_context_reconstruction | 相同 V10 快照 | 已生成、全量验收并发布 |
| line_context_points | compact real raw + V10 多叉 line | 已生成、全量验收并发布：240,272行 |
| grounding_layout | 新真实标注快照 + 恢复的3,306份paper增量 | 已生成、全量验收并发布：78,514行 |
| background | 历史人工审核标签及经过验证的历史媒体 | 沿用；必须提供历史输入，不能从 V10 猜测 |
| image_context_reconstruction | 经过验证的历史 image-type bundle | 沿用；当前 raw 无全部历史类型标签 |

**整版要求可复现，不表示当前所有任务已经全部接通。** 尚未确认的采样不沿用隐含默认值。

2026-09-08 新真实标注包先独立替换用户指定 raw 目录，再通过
[真实标注质量清洗](clean_real_raw_annotations.md) 清洗：20,124 份输入保留 20,070 份，
54 份可恢复隔离，删除 126 个重复实例、裁边 12 个轻微越界主框；源点和原图不改，
`subbbox` 不作为质量判断依据。这个真实源快照尚未自动同步至其他 raw 副本或训练 catalog；
接入前仍需任务准入与测试集内容隔离。已完成的 V10 合成 shape/line 无需因此重建。

2026-09-09 经用户确认补回新包未包含的 3,306 份 v5.9 paper 增量，真实源现在有 23,376 份 JSON。
对应图片从 paper 复制到 raw 的 images；全量解码、尺寸匹配以及与冻结 v5.9 图片/标注的字节一致性
检查通过。补回清单和哈希记录于该 raw 的 `restore_v5_9_increment.json`，不覆盖原有 20,070 份标注。
复现需额外提供此清单指定的 paper 图片和旧增量 JSON，不能仅使用本次 20,124 份新标注 ZIP。

罕见 `regular_pentagon / step` 各占 shape 训练抽样 0.5% 是后续训练配置目标，
当前配置中的 `rare_training_probability` 只记录该目标，尚未激活运行时重采样，不重复生成图片。

## 必要输入

Grounding 独立入口和配方见 [banana_v5_10_grounding.md](banana_v5_10_grounding.md)。
Line points 独立入口和配方见 [banana_v5_10_line_points.md](banana_v5_10_line_points.md)。
真实shape独立入口与严格完整性门禁见 [banana_v5_10_real_shape.md](banana_v5_10_real_shape.md)。

1. V10 的 `gt_standard/`、`img/`、`train.txt`、`val.txt`。
   GT revision 为 `df3f66d8e661b60641dc8d10a6f65d7a98dcaab8`，上游 GT ZIP SHA256：
   `cb3c524324a83f19f1f2ecd3d858b5ca279a243c75856986397788abddb44c34`。
   新 GT 坐标已经对应原图像素，禁止再乘二；构建检查 GT 与图像尺寸相同。
2. `real_v1.ids.txt`、`real_v2.ids.txt` 和 canonical `vlm.test.json` 排除清单。
   此处传递文件而非本机测试集目录；按 source ID 排除。先固定 split，再做增强。
3. 其余任务依赖的 compact raw、冻结 split、历史审核结果/恢复包，按上表单独维护输入清单。
   “都有原始数据”必须包含这些标签来源，不仅是图片和 V10 合成数据。

输入根目录与工作/输出目录全部由 CLI 传入，可在其他机器改名。训练 JSONL 引用相对媒体路径，
样本中的 source dataset 使用冻结 snapshot id，不取本机目录名。

## Shape 采样与增强

- seed 为 **465**；SHA256(source ID、instance index、seed) 排序无放回选择，不使用进程局部 RNG。
- rectangle 取 300,000（若不足则全取），按几何/外观粗分层平方根容量分配，限制常见头部占比。
- 左右排列、三区域及以上 card 全保留；斜向/混合稀有布局也保护。
  常见双区域上下排列 card 额外取 25,000，按首区域比例、方向、角点、分隔线、颜色等分层。
- 其余有效类型全部保留，包括 `other`；没有统一总量上限。
- 无效 schema/bbox 只从派生 selection 排除，统计原因，不改源文件。
- 三份 formulation `appearance / geometry / reconstruction` 同行同图，仅 target 不同。
  `other` 三者都是 `{"type":"shape","parameters":{"shape_type":"other"}}`。
- 使用既有 `synthetic_realism_v1`：1–3 个尺寸不变操作；重采样往返、模糊、噪声、JPEG。
  强度由目标在 crop 内的跨度决定，极小目标仅一次轻度操作，完整参数和 noise seed 写入每行 extra。
  参数真源是冻结哈希的 `src/shaft/data/synthetic_realism.py`；不在训练时重复离线噪声。
- 已发布 shape 的 JPEG quality 为 mild 82–95、moderate 62–84、strong 42–68，总范围 42–95；
  只有选到 JPEG 的样本应用此项，不等于所有 shape 图片都经过 JPEG，也不是统一均匀分布。
- Proposal/crop 复用维护中的几何合同；完整 target 与 proposal 都使用 crop 的 0–999 坐标系。
- 不改变原图尺寸、不改原图标注；空 val 表示 train-only，V10 val 不进入本轮训练产物。

本机修正 GT 首轮 selection 已冻结：992,112 个实例，其中 rectangle 300,000、card 51,755、
other 31,676、regular_pentagon 597、step 565；源标注越界排除 5 个实例。
2026-09-08 已完成全量生成与验收，并发布到 `data/shape_context_reconstruction`：
992,112 张共享图片，三种 formulation 各 992,112 行，共 2,976,336 行 SFT。
全部 formulation 的身份、目标投影与媒体引用对齐；全部图片完整解码、尺寸检查和 SHA256 计算通过。
原派生目录保留为同级 `shape_context_reconstruction.previous-lwztwv4k`，原始数据未修改。
本次 content SHA256：`f5b23a5f372db2264f17e5a0cd2067376a6a1f04c833a67750c65d237391f0f7`。
正式验收报告位于 `data/shape_context_reconstruction/reports/reproduction_result.json`。

## 跨机器命令

### Line 采样前审计

```bash
uv run --no-sync python scripts/tasks/audit_banana_v5_10_lines.py \
  --synthetic-root /path/to/v10 \
  --exclude-manifests /path/to/real_v1.ids.txt /path/to/real_v2.ids.txt /path/to/vlm.test.json \
  --output data/banana_v5_10/reports/line_inventory.json --workers 50
```

2026-09-08 修正 GT 审计：train 100,000 张、val 500 张，split 无交集。
Train 原始 line 410,326 条，当前标注有效性检查通过 409,702 条：
多路径 118,768、复杂单路径 207,044、普通单路径 83,890；624 条未通过检查。
Val 2,090 条通过检查，不加入训练。
“复杂”是路径/属性采样启发式，不是实测模型难度；路径数不是图的分叉度。
零宽/零高 bbox 单独记为需要裁图适配的标志，而不是直接排除合法水平/垂直路径。
这些数量不是最终 SFT 数量：尚需冻结分层采样和增强强度，通过 crop/量化门禁。
审计只读原始 GT 和图像尺寸；输出包含源 GT、split、审计脚本与 validator SHA256。

### Line 增强预览

Line 增强校准预览另用下面的入口（不生成训练 SFT，不修改 shape 增强）：

```bash
uv run --no-sync python scripts/tasks/preview_banana_v5_10_lines.py \
  --synthetic-root /path/to/v10 \
  --inventory data/banana_v5_10/reports/line_inventory.json \
  --output data/banana_v5_10/previews/line_calibration_v1 --workers 50
```

校准配置：`configs/data/preparation/banana_v5_10_line_preview.json`。
从审计中的 train stratum examples 确定性挑选 6 类、每类 2 例，不声称这是全数据随机代表样本：
密集多路径、曲线、虚线、小箭头、狭长 path 代理和浅色代理。狭长 bbox 不等于笔画细，浅色不等于
与局部背景低对比度，需要人工看图确认。

每例保留原尺寸 `clean.png`、6 张增强图片、两张对照图和 `plans.json`：

| 强度 | 重采样比例 | 噪声 sigma（0–255） | JPEG quality | 单独模糊 radius |
| --- | --- | --- | --- | --- |
| mild | 0.90 | 2 | 90 | 0.45 |
| moderate | 0.75 | 5 | 75 | 0.85 |
| strong | 0.55 | 9 | 50 | 1.40 |

`compare_stack.png` 固定比较重采样、噪声、JPEG 三层叠加，`compare_blur.png` 单独比较模糊。
四列均为 clean/mild/moderate/strong；相同实例各强度共享 noise seed，避免随机噪声位置混淆对比。
所有实际 crop 尺寸不变；对照排版可能缩小显示，不是训练 resize。预览 clean twin 不进入训练。
`manifest.json` 保存配置、源图/GT 审计、脚本/算子哈希及 Python/NumPy/Pillow/codec 环境。
零跨度 bbox 在预览中只为 locator 加 1px 边界，不改 points；生产 adapter 尚需独立验证。

2026-09-08 目视检查：12 例共 24 张对照图已生成。曲线虚线 `000009__line_0014` 的 strong blur
使离散点和箭头轮廓明显模糊；短线 `000038__line_0003` 在 strong stack 下可辨认性下降；
密集多路径 `024568__line_0022` 的浅色细分支在强模糊下变弱。
结论仅支持 line 单独校准强度，不能外推为已测得准确率损失或已冻结生产概率。
建议弱线/小端点优先轻度、普通清晰线以轻中度为主；强度较高组合仅在可辨认性确认后少量采用。

### Line 生产策略（覆盖上述校准建议）

用户最终确认：不再增加人工对照负担，关闭 blur、重采样、颜色扰动及多操作叠加；
JPEG quality 覆盖 **40–90**，不是 90–97，也不是压缩后文件大小比例。
配置真源为 `configs/data/preparation/banana_v5_10_line.json`：

- 多路径有效实例全保留；单路径完整属性 stratum 数量不超过 256 的优先全保留。
- 复杂单路径目标 200,000、普通简单单路径目标 30,000；稀有保护量超过目标时不截断稀有实例。
- 非极小目标按 80% JPEG、20% Gaussian noise 的显式默认概率单选。
  JPEG 为 40–90 整数均匀分布、`subsampling=0`（4:4:4）；噪声 sigma 为 0.5–1.5/255。
- 原始 bbox 短边小于 16 像素的目标保持 clean；这是保守尺寸代理，不声称测量了实际笔画宽度。
- 这条用户确认的 v5.10 line 规则有意允许 clean，覆盖历史“每个合成 crop 必须加噪”的默认要求。
- Locator 额外留 2px 可见边缘，不改源 GT/points；原始 bbox 单独记录在 `source_bbox_raw`。
- 曲线/折线量化发生连续点重合时拒绝该 view 并报告，不静默删点；未知构建错误直接中止。
- 复用既有像素操作执行器，使用单独 `policy_id` 区分采样策略；shape 策略与已发布图片不变。
- 当前入口只生成 full-capable synthetic line 的三个 formulation，不重新生成真实 points-only cohort。

```bash
uv run --no-sync python scripts/tasks/prepare_banana_v5_10_lines.py \
  --synthetic-root /path/to/v10 \
  --exclude-manifests /path/to/real_v1.ids.txt /path/to/real_v2.ids.txt /path/to/vlm.test.json \
  --work-root data/.build/banana-v5.10-line \
  --output-root data --workers 50 --replace
```

使用新的 work-root；准备、输入校验、生成、全量验收和发布一次完成。源数据和旧正式目录不提前删除。
最终 `reports` 包含 recipe/环境/输入与代码哈希、拒绝 view、生成统计和内容校验结果。
SHA256 和可解码性证明内容一致与媒体有效，不代表所有视觉属性已经自动证明可辨认。
已完成 shape 与校准代码的历史基线提交为 `92cf188`；整版 `v5.10` tag 仍待各任务完成。

生产完成基线：348,768 张共享 crop，appearance/points/reconstruction 各 348,768 行，
总 SFT 1,046,304 行；118,768 条有效多路径全保留，复杂单路径 200,000，普通单路径 30,000。
量化拒绝 view 为 0，静默删除点为 0。JPEG 243,732、noise 61,250、clean 43,786；
逐行检查确认 JPEG 覆盖全部 51 个整数 quality 档位 40–90，全部 subsampling=0，无 blur/resample/叠加。
全量 formulation 目标/身份/媒体引用对齐、图片解码/尺寸/hash 验收通过。
已发布到 `data/line_context_reconstruction`，旧目录为同级 `line_context_reconstruction.previous-tw50dyn2`。
Content SHA256：`c47ca5f3f1880d25bc9b82b8421c0fe4d29998359174b44a2eb5650c44a29c5c`。
这不是整版 v5.10 发布：真实 points-only cohort 和其余任务还需按版本计划处理。

### Shape 生成

在仓库根目录，先使用同一代码版本及 UV 锁定环境。下面的 `/path/to/...` 全部替换为本机路径：

```bash
uv run --no-sync python scripts/tasks/prepare_banana_v5_10.py \
  --synthetic-root /path/to/v10 \
  --exclude-manifests /path/to/real_v1.ids.txt /path/to/real_v2.ids.txt /path/to/vlm.test.json \
  --work-root /path/to/work/banana-v5.10-shape \
  --output-root data --workers 50 --replace
```

`--replace` 仅替换 `shape_context_reconstruction` 派生目录；通过 staging 完成生成和验收后再发布，
原目录移动为同级 `shape_context_reconstruction.previous-*`，失败不会先删除原目录。
`--stage prepare` 可单独扫描与冻结 selection；随后相同命令使用 `--stage build` 消费已有冻结清单。
准备已完成的 work-root 不重复覆盖；重做 selection 使用新的 work-root。

## 复现与验收口径

每个任务发布时携带：

- `reports/reproduction.lock.json`：recipe、源内容摘要、代码/prompt SHA256、环境版本、selection 哈希、
  排除清单、可用/选中数量、质量剔除原因。
- `reports/input_checksums.json`：本任务实际扫描输入的相对路径与文件 SHA256。
- `reports/shape.selection.jsonl`：采样身份与分层，不存一套可被误当真源的完整 target。
- `reports/content_checksums.json`：输出媒体、selection、structured、SFT 的相对路径及 SHA256。
- `reports/reproduction_result.json`：全量目标/行对齐/图片解码验收及 content hash。

复现时需要比较 **input / selection / content hashes**，不是只比较行数或 seed。
改变 workers、工作路径或机器目录名，不应改变样本顺序、目标、增强随机计划。
逐字节图片一致还要求 Python、NumPy、Pillow、JPEG/zlib 编解码版本一致；环境不符时拒绝消费冻结清单。
硬件架构或编译实现差异仍须以最终 content hash 为准，不承诺仅凭版本号就跨所有平台逐字节相同。
报告中的执行路径/worker 数等运维信息不属于训练内容摘要。

首次构建得到锁定清单，后续机器应携带该清单/selection（很小，相比图像）执行 build，
或自行 prepare 后比对 selection/input hash。仅有同名 V10 目录不能证明是同一版本。

focused tests：

```bash
uv run --no-sync pytest -q tests/test_prepare_banana_v5_10.py --suite task
```
