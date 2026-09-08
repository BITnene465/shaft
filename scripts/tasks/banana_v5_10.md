# Banana v5.10 数据复现

版本标识：`banana-v5.10`。维护真源是本仓库的脚本、配置和本文档，不是本机的 `subTasks/`。
Git `v5.10` tag 只在整套流程完成、验收并提交后创建；当前 shape 阶段不代表整版已发布。

## 入口与状态

- 配置：`configs/data/preparation/banana_v5_10.json`。
- 执行：`scripts/tasks/prepare_banana_v5_10.py`。
- 合同与回归：`tests/test_prepare_banana_v5_10.py`，复用 context reconstruction builder。
- 产物：`data/<task>/{selection,structured,images,sft,reports}`。

| 任务 | 输入真源 | 当前状态 |
| --- | --- | --- |
| shape_context_reconstruction | 修正后的 V10 GT + 原始 img + train/val | 已生成、全量验收并发布 |
| line_context_reconstruction | 相同 V10 快照 | 待确定采样数量后接入版本入口 |
| line_context_points | compact real raw + V10 多叉 line | 待确定合成补充量后接入 |
| grounding_layout | v5.9 冻结 raw、增量标注及 split | 继承规则，仍需冻结本版复现输入 |
| background | 历史人工审核标签及经过验证的历史媒体 | 沿用；必须提供历史输入，不能从 V10 猜测 |
| image_context_reconstruction | 经过验证的历史 image-type bundle | 沿用；当前 raw 无全部历史类型标签 |

**整版要求可复现，不表示当前六个任务已经全部接通。** 尚未确认的采样不沿用隐含默认值。
罕见 `regular_pentagon / step` 各占 shape 训练抽样 0.5% 是后续训练配置目标，
当前配置中的 `rare_training_probability` 只记录该目标，尚未激活运行时重采样，不重复生成图片。

## 必要输入

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
